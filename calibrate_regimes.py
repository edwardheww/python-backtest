#!/usr/bin/env python3
"""
Calibrate NQ regime thresholds from 5-minute RTH data.

Usage:
    pip install yfinance pandas numpy
    python calibrate_regimes.py                  # pulls ~59 days of NQ=F 5m from yfinance
    python calibrate_regimes.py --csv my.csv     # your own data: Datetime,Open,High,Low,Close,Volume
    python calibrate_regimes.py --ticker MNQ=F

Notes:
  * yfinance only serves ~60 days of 5-min history. Good for a first look,
    not for a real backtest. Use a proper data vendor for that.
  * CSV timestamps without a timezone are assumed to be America/New_York.
  * Thresholds below are PLACEHOLDERS. The script shows how they split YOUR data.
"""
import argparse
import numpy as np
import pandas as pd

# ---- Placeholder thresholds (edit and re-run) -------------------------------
T = dict(
    or_ratio_a=(1.0, 2.0),    # Regime A: OR size ratio range
    or_ratio_b=(0.5, 1.0),    # Regime B: OR size ratio range
    vol_ratio_min_a=1.0,      # Regime A: 15-min volume ratio floor
    vol_ratio_b=(0.7, 1.2),   # Regime B: 15-min volume ratio band
    vol_ratio_c=0.6,          # Regime C: below this = thin
    adx_a=25.0,               # Regime A: ADX at 9:45
    adx_b=25.0,               # Regime B: ADX below this at 10:30
    crosses_a_max=1,          # Regime A: VWAP crosses by 9:45
    crosses_b_min=2,          # Regime B: VWAP crosses in trailing 60 min at 10:30
    gap_b_max=0.5,            # Regime B: gap / daily ATR ceiling
)


def load(csv, ticker, days):
    if csv:
        df = pd.read_csv(csv, parse_dates=[0], index_col=0)
    else:
        import yfinance as yf
        df = yf.download(ticker, period=f"{days}d", interval="5m",
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize("America/New_York") if idx.tz is None else idx.tz_convert("America/New_York")
    df.index = idx
    return df.between_time("09:30", "15:55")


def wilder_adx(df, n=14):
    # Reset on each day's first bar so the overnight gap does not count as a directional move.
    d = pd.Series(df.index.date, index=df.index)
    new_day = d != d.shift()
    up = df.High.diff().mask(new_day, 0.0)
    dn = (-df.Low.diff()).mask(new_day, 0.0)
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    tr = tr.mask(new_day, df.High - df.Low)
    a = 1.0 / n
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def build_daily(df):
    df = df.copy()
    df["date"] = df.index.date
    df["adx"] = wilder_adx(df)
    tp = (df.High + df.Low + df.Close) / 3
    cv = df.groupby("date").Volume.cumsum()
    cpv = (tp * df.Volume).groupby(df.date).cumsum()
    df["vwap"] = cpv / cv
    side = np.sign(df.Close - df.vwap)
    prev = side.groupby(df.date).shift()
    df["flip"] = ((side != prev) & prev.notna() & (side != 0) & (prev != 0)).astype(int)
    df["cross_60m"] = df.groupby("date")["flip"].transform(lambda s: s.rolling(12, min_periods=1).sum())
    df["cross_cum"] = df.groupby("date")["flip"].cumsum()

    rows = []
    for d, day in df.groupby("date"):
        if len(day) < 70:          # skip half days / partial data (full day = 78 bars)
            continue
        orb = day.iloc[:3]         # 9:30-9:45
        rows.append(dict(
            date=d,
            open=day.Open.iloc[0], close=day.Close.iloc[-1],
            high=day.High.max(), low=day.Low.min(),
            or_high=orb.High.max(), or_low=orb.Low.min(),
            vol15=orb.Volume.sum(),
            adx_945=day.adx.iloc[2], adx_1030=day.adx.iloc[11],
            cross_945=day.cross_cum.iloc[2], cross_1030=day.cross_60m.iloc[11],
            day_cross=day.cross_cum.iloc[-1],
        ))
    out = pd.DataFrame(rows).set_index("date")
    out["or_range"] = out.or_high - out.or_low
    out["prev_close"] = out.close.shift()
    tr = pd.concat([out.high - out.low, (out.high - out.prev_close).abs(),
                    (out.low - out.prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=10).mean().shift()   # known before the open
    out["or_ratio"] = out.or_range / out.or_range.shift().rolling(20, min_periods=10).mean()
    out["vol_ratio"] = out.vol15 / out.vol15.shift().rolling(20, min_periods=10).mean()
    out["gap_ratio"] = (out.open - out.prev_close).abs() / out.atr14
    out["or_to_atr"] = out.or_range / out.atr14
    out["day_range_atr"] = (out.high - out.low) / out.atr14
    out["efficiency"] = (out.close - out.open).abs() / (out.high - out.low)  # 1 = pure trend day
    return out.dropna(subset=["or_ratio", "vol_ratio", "gap_ratio"])


def conditions(d):
    a = pd.DataFrame({
        "OR ratio in A range": d.or_ratio.between(*T["or_ratio_a"]),
        "volume ratio >= min": d.vol_ratio >= T["vol_ratio_min_a"],
        "VWAP crosses <= max (9:45)": d.cross_945 <= T["crosses_a_max"],
        "ADX >= min (9:45)": d.adx_945 >= T["adx_a"],
    })
    b = pd.DataFrame({
        "OR ratio in B range": (d.or_ratio >= T["or_ratio_b"][0]) & (d.or_ratio < T["or_ratio_b"][1]),
        "volume ratio in band": d.vol_ratio.between(*T["vol_ratio_b"]),
        "VWAP crosses >= min (10:30)": d.cross_1030 >= T["crosses_b_min"],
        "ADX < max (10:30)": d.adx_1030 < T["adx_b"],
        "gap ratio < max": d.gap_ratio < T["gap_b_max"],
    })
    return a, b


def assign(d):
    a, b = conditions(d)
    regime = pd.Series("C", index=d.index)
    regime[b.all(axis=1)] = "B"
    regime[a.all(axis=1)] = "A"
    regime[(d.or_ratio > T["or_ratio_a"][1]) | (d.vol_ratio < T["vol_ratio_c"])] = "C"
    return regime


def funnel(c, name):
    n = len(c)
    remaining = c.astype(int).cumprod(axis=1).sum()
    out = pd.DataFrame({
        "pass alone %": (c.mean() * 100).round(0),
        "days left after this and all above": remaining,
    })
    print(f"\nRegime {name} funnel ({n} days). Biggest drop = the condition blocking it:")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--ticker", default="NQ=F")
    ap.add_argument("--days", type=int, default=59)
    ap.add_argument("--out", default="daily_metrics.csv")
    a = ap.parse_args()

    print(f"Running file: {__file__}")
    print(f"Thresholds in use: {T}\n")
    daily = build_daily(load(a.csv, a.ticker, a.days))
    daily["regime"] = assign(daily)
    daily.to_csv(a.out)

    pd.set_option("display.width", 140)
    cols = ["or_ratio", "vol_ratio", "gap_ratio", "or_to_atr", "adx_945", "adx_1030", "cross_945", "cross_1030"]
    print(f"\nDays analysed: {len(daily)}")
    print("\nMetric percentiles (use these to sanity-check thresholds):")
    print(daily[cols].quantile([.1, .25, .5, .75, .9]).round(2).T)
    print("\nRegime counts under current thresholds:")
    print(daily.regime.value_counts().to_string())
    ca, cb = conditions(daily)
    funnel(ca, "A")
    funnel(cb, "B")
    print("\nDo regimes actually separate? (A should show higher efficiency, B lower, "
          "and B more VWAP crosses)")
    print(daily.groupby("regime")[["efficiency", "day_range_atr", "day_cross"]].mean().round(2))
    print(f"\nSaved per-day metrics to {a.out}")
    print("Tip: or_to_atr median tells you what stop cap to use as a fraction of daily ATR.")


if __name__ == "__main__":
    main()
