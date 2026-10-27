#!/usr/bin/env python3
"""
Download 1-minute NQ and GC history from Databento and prepare it for calibrate_sessions.py.

SETUP (once)
  1. Create an account at databento.com and copy your API key (new accounts have received free credit).
  2. pip install databento pandas numpy
  3. export DATABENTO_API_KEY="your-key"        # Mac/Linux
     setx DATABENTO_API_KEY "your-key"          # Windows (open a new terminal afterwards)

USAGE
  # Step 1: see the cost first. Nothing is downloaded and nothing is charged.
  python3 download_databento.py --symbols NQ --days 1825          # last ~5 years
  python3 download_databento.py --symbols NQ --all                # everything Databento has

  # Step 2: download once you are happy with the price.
  python3 download_databento.py --symbols NQ --days 1825 --yes

  Or give explicit dates:  --start 2020-01-01 --end 2026-10-01
  Historical data is only available once it is at least a day old, so the window always ends
  at the latest available day (yesterday at the latest), never "right now".

Output: nq_1m.csv and gc_1m.csv (UTC timestamps, back-adjusted prices).

NOTES
  * Databento's continuous contracts are UNADJUSTED, so prices jump at every roll. This script
    back-adjusts them (difference method) so ATR, gaps and range features are not polluted.
    Adjusted prices drift away from the real traded price in older data. That is fine for
    ratios and R-multiples, but do not read old price levels as real quotes.
  * Roll rule: v = volume-based (default), c = calendar, n = open-interest-based.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

DATASET = "GLBX.MDP3"


def back_adjust(df):
    """Difference back-adjustment at each roll (change of instrument_id).
    Earlier prices are shifted by (new contract's first open - old contract's last close)."""
    iid = df["instrument_id"].to_numpy()
    change = np.flatnonzero(iid[1:] != iid[:-1]) + 1
    arr = df[["open", "high", "low", "close"]].to_numpy(float)
    shift = np.zeros(len(df))
    for k in change:
        shift[:k] += arr[k, 0] - arr[k - 1, 3]
    out = df.copy()
    for j, col in enumerate(("open", "high", "low", "close")):
        out[col] = arr[:, j] + shift
    return out, len(change)


def tidy(df):
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df, n_rolls = back_adjust(df)
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out.index.name = "Datetime"
    return out, n_rolls


def fetch_chunk(db, client, a, csym, ws, we, tries=4):
    """One request with retries. Client errors (bad key, bad symbol) are not retried."""
    client_err = getattr(db, "BentoClientError", ())
    for attempt in range(1, tries + 1):
        try:
            data = client.timeseries.get_range(dataset=a.dataset, symbols=[csym], stype_in="continuous",
                                               schema="ohlcv-1m", start=ws, end=we)
            return data.to_df()
        except Exception as exc:
            if (client_err and isinstance(exc, client_err)) or attempt == tries:
                raise
            wait = 3 * 2 ** attempt
            print(f"    attempt {attempt}/{tries} failed ({type(exc).__name__}: {exc}). "
                  f"Retrying in {wait}s ...", flush=True)
            time.sleep(wait)


def windows(start, end, step_days=90):
    """Split [start, end) into chunks so a long download shows progress and is easy to retry."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    while s < e:
        n = min(s + pd.Timedelta(days=step_days), e)
        out.append((s.strftime("%Y-%m-%d"), n.strftime("%Y-%m-%d")))
        s = n
    return out


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def resolve_dates(a, ds_start=None, ds_end=None, today=None):
    """Return (start, end) as YYYY-MM-DD. The end never goes past yesterday (UTC)."""
    today = today if today is not None else pd.Timestamp.now("UTC").tz_localize(None).normalize()
    latest = today - pd.Timedelta(days=1)
    if ds_end is not None:
        latest = min(latest, _naive(ds_end).normalize())
    end = pd.Timestamp(a.end) if a.end else latest
    if a.all:
        if ds_start is None:
            raise SystemExit("--all needs the dataset start date; use --start instead.")
        start = _naive(ds_start).normalize()
    elif a.days:
        start = end - pd.Timedelta(days=a.days)
    elif a.start:
        start = pd.Timestamp(a.start)
    else:
        raise SystemExit("Give --days N, --all, or --start YYYY-MM-DD.")
    if start >= end:
        raise SystemExit(f"Start {start.date()} is not before end {end.date()}.")
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD (exclusive). Default: latest available day")
    ap.add_argument("--days", type=int, help="look back this many days from the latest available day")
    ap.add_argument("--all", action="store_true", help="from the dataset's first available day")
    ap.add_argument("--symbols", nargs="+", default=["NQ", "GC"])
    ap.add_argument("--roll", choices=["v", "c", "n"], default="v")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--yes", action="store_true", help="actually download (otherwise only prices it)")
    ap.add_argument("--chunk-days", type=int, default=90, help="size of each request (default 90 days)")
    ap.add_argument("--refresh", action="store_true", help="ignore the chunk cache and download again")
    a = ap.parse_args()

    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit("Set DATABENTO_API_KEY first (see the setup notes at the top of this file).")
    try:
        import databento as db
    except Exception as exc:       # show the real cause, not a guess
        sys.exit(f"Could not import databento: {type(exc).__name__}: {exc}\n"
                 f"This script is running under: {sys.executable} (Python {sys.version.split()[0]})\n"
                 f"If databento is missing for that Python, install it with:\n"
                 f"  {sys.executable} -m pip install databento")

    client = db.Historical()
    ds_start = ds_end = None
    try:
        rng = client.metadata.get_dataset_range(dataset=a.dataset)
        ds_start, ds_end = rng["start"], rng["end"]
    except Exception as exc:
        print(f"(could not read the dataset range: {exc}; using yesterday as the end date)")
    a.start, a.end = resolve_dates(a, ds_start, ds_end)
    print(f"Date range: {a.start} to {a.end} (end exclusive)")
    plan = []
    total = 0.0
    for sym in a.symbols:
        csym = f"{sym}.{a.roll}.0"
        cost = client.metadata.get_cost(dataset=a.dataset, symbols=[csym], stype_in="continuous",
                                        schema="ohlcv-1m", start=a.start, end=a.end)
        print(f"{csym}: estimated cost ${cost:,.2f}")
        total += float(cost)
        plan.append((sym, csym))
    print(f"Total estimated cost: ${total:,.2f}")
    if not a.yes:
        print("\nNothing downloaded. Re-run with --yes to download.")
        return

    for sym, csym in plan:
        wins = windows(a.start, a.end, a.chunk_days)
        cache_dir = ".databento_cache"
        os.makedirs(cache_dir, exist_ok=True)
        print(f"\nDownloading {csym} in {len(wins)} chunk(s). Finished chunks are saved in "
              f"{cache_dir}/, so a re-run resumes where it stopped.", flush=True)
        t0 = time.time()
        parts = []
        for i, (ws, we) in enumerate(wins, 1):
            cpath = os.path.join(cache_dir, f"{csym}_{ws}_{we}.pkl")
            if os.path.exists(cpath) and not a.refresh:
                part = pd.read_pickle(cpath)
                tag = "cached"
            else:
                part = fetch_chunk(db, client, a, csym, ws, we)
                part.to_pickle(cpath)
                tag = "downloaded"
            parts.append(part)
            print(f"  chunk {i}/{len(wins)} ({ws} to {we}): {len(part):,} bars, {tag}, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
        df = pd.concat([p for p in parts if not p.empty]) if any(not p.empty for p in parts) else pd.DataFrame()
        if df.empty:
            print(f"No data returned for {csym}.")
            continue
        df.columns = [str(c).lower() for c in df.columns]
        print("  adjusting rolls and writing the CSV ...", flush=True)
        out, n_rolls = tidy(df)
        path = f"{sym.lower()}_1m.csv"
        out.to_csv(path)
        print(f"{path}: {len(out):,} one-minute bars, {n_rolls} contract rolls back-adjusted, "
              f"{out.index.min()} to {out.index.max()}")


if __name__ == "__main__":
    main()
