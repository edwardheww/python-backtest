#!/usr/bin/env python3
"""
Session calibration for the NQ / GC opening-range system (mirrors nq_regime_system.pine).

WHAT IT DOES
  1. Rebuilds the indicator's logic on historical bars: three sessions (Asia, London, NY), each
     with its own opening range, the expansion block rule, regime filters, Strategy A (and
     optionally B), and the simulated trade management.
  2. Searches parameters on a TRAINING period (random search, one calibration per instrument).
  3. Tells you which sessions to filter out. The verdict is decided from TRAINING evidence only,
     then checked on a held-out TEST period. A session is only recommended if both agree.

DATA (one file per instrument)
  1-minute (preferred) or 5-minute OHLCV, CSV or Parquet, INCLUDING overnight bars.
  A timestamp column (or index) plus open, high, low, close, volume.
  Naive timestamps are assumed to be --tz (default America/New_York).
  Use a back-adjusted or at least roll-aware continuous contract. Roll gaps distort the gap
  ratio and ATR; the script warns about suspicious jumps but cannot fix them.

USAGE
  python calibrate_sessions.py --nq nq_1m.parquet --gc gc_1m.parquet calibrate --samples 150 --jobs 4
  python calibrate_sessions.py --nq nq_1m.csv run --preset balanced
  python calibrate_sessions.py --gc gc_1m.csv calibrate --config sessions.json

  sessions.json overrides defaults per instrument and session, for example:
  {"GC": {"NY": {"sess": "0820-1330", "orw": "0820-0835", "entryEnd": 1030, "flat": 1320}},
   "NQ": {"Asia": {"enabled": false}}}

Requires: pip install pandas numpy   (pyarrow for parquet)
Every threshold is a placeholder. Prefer sessions and settings that hold up on the held-out test;
never trust a result that only looks good on the data used to choose it.
"""
import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

NY = "America/New_York"
NAN = float("nan")
SESS_ORDER = ["Asia", "London", "NY"]
ACTIVE = (1, 2, 3)          # session ids analysed; narrowed by --sessions

# Instrument specs. cost = round-trip slippage + fees in POINTS (placeholder, edit it).
INSTR = {
    "NQ": dict(tick=0.25, pv=20.0, cost=1.0, sessions={
        "Asia":   dict(sess="2000-0200", orw="2000-2100", entryEnd=0,    flat=155),
        "London": dict(sess="0200-0930", orw="0200-0300", entryEnd=600,  flat=920),
        "NY":     dict(sess="0930-1600", orw="0930-0945", entryEnd=1100, flat=1200),
    }),
    # Gold: COMEX regular hours open 08:20 ET, so the NY session is anchored there.
    "GC": dict(tick=0.10, pv=100.0, cost=0.3, sessions={
        "Asia":   dict(sess="2000-0200", orw="2000-2100", entryEnd=0,    flat=155),
        "London": dict(sess="0200-0820", orw="0200-0300", entryEnd=600,  flat=810),
        "NY":     dict(sess="0820-1330", orw="0820-0835", entryEnd=1030, flat=1320),
    }),
}

DEFAULTS = dict(
    enableA=True, enableB=False,
    orALo=1.0, orAHi=2.0, orBLo=0.5, orBHi=1.0,
    volMinA=1.0, volBLo=0.7, volBHi=1.2, volC=0.6,
    adxA=25.0, adxB=25.0, crossAMax=1, crossBMin=2, gapBMax=0.5,
    bandSD=2.0, bufTicks=2, stopCapATR=0.25, bandFloorATR=0.1, topPct=0.7,
    aTrendTest="eff", orEffMin=0.5, skipFirst=False, maxBarOR=0.75, maxChaseOR=0.5,
    alignOR=True, gapAgainst=0.0, maxORATR=0.4,
    useDailyRule=True, timeStopBars=6, timeStopMinR=0.5,
    aExitMode="full", aTargetR=2.0, aTimeStopOn=True, trailOn=True,
    bStartMins=60, bEndMins=300,
    blockOn=True, blockSrc="overnight",      # overnight | combined | prev
    expandMode="or", expandOR=2.5, expandTyp=1.5, measureBlocked=False,
    costPts=None,
)
PRESETS = {
    "strict":   dict(orALo=1.0, volMinA=1.0, crossAMax=1, orEffMin=0.5, maxBarOR=0.75, maxChaseOR=0.5, topPct=0.7),
    "balanced": dict(orALo=0.8, volMinA=0.8, crossAMax=2, orEffMin=0.4, maxBarOR=1.0, maxChaseOR=0.75, topPct=0.6),
    "loose":    dict(orALo=0.6, volMinA=0.6, crossAMax=3, orEffMin=0.3, maxBarOR=1.25, maxChaseOR=1.0, topPct=0.5),
}
SPACE_A = dict(
    orALo=[0.6, 0.8, 1.0, 1.2],
    volMinA=[0.0, 0.6, 0.8, 1.0],
    crossAMax=[1, 2, 3, 99],
    orEffMin=[0.0, 0.3, 0.4, 0.5, 0.6],
    maxORATR=[0.0, 0.3, 0.4, 0.5],
    maxBarOR=[0.0, 0.75, 1.0, 1.25],
    maxChaseOR=[0.0, 0.5, 0.75, 1.0],
    topPct=[0.5, 0.6, 0.7],
    alignOR=[False, True],
    bufTicks=[0, 2, 4],
    stopCapATR=[0.15, 0.2, 0.25, 0.3],
    aTargetR=[1.0, 1.5, 2.0, 2.5, 3.0],
    aTimeStopOn=[False, True],
    timeStopBars=[4, 6, 9],
    skipFirst=[False, True],
    blockOn=[False, True],
    blockSrc=["overnight", "combined", "prev"],
    expandMode=["or", "typ"],
    expandOR=[0.0, 2.0, 2.5, 3.0, 4.0],
    expandTyp=[0.0, 1.25, 1.5, 2.0],
    measureBlocked=[False, True],
)
SPACE_B = dict(
    bandSD=[1.5, 2.0, 2.5], crossBMin=[2, 3], adxB=[20.0, 25.0, 30.0],
    orBLo=[0.4, 0.5, 0.6], bandFloorATR=[0.05, 0.1, 0.15],
)
PINE_NAMES = dict(
    orALo="A: OR ratio min", volMinA="A: volume ratio min", crossAMax="A: max VWAP crosses by 9:45",
    orEffMin="A: min OR efficiency", maxORATR="A: max opening range (x daily ATR)",
    maxBarOR="A: max signal bar range (x OR range)", maxChaseOR="A: max close beyond OR edge (x OR range)",
    topPct="A: signal bar close position min", alignOR="A: breakout must match opening drift",
    bufTicks="A: breakout buffer (ticks)", stopCapATR="Stop cap (x daily ATR)",
    aTargetR="A: full-target distance (R)", aTimeStopOn="A: use the time stop",
    timeStopBars="Time stop (5m bars)", skipFirst="A: skip the first candle after the OR",
    blockOn="Enable session block", blockSrc="Block source",
    expandMode="Expansion test (Range vs its OR / Range vs typical)", expandOR="Range vs its OR multiplier",
    expandTyp="Range vs typical multiplier", measureBlocked="Also test sessions that were blocked",
)


def hm(x):
    x = int(x)
    return (x // 100) * 60 + x % 100


def parse_sess(txt):
    a, b = txt.split("-")
    return hm(int(a)), hm(int(b))


def relM(t, st):
    return (t - st) % 1440


# ---------------------------------------------------------------- data
def load_bars(path, tz):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}\nCurrent folder: {os.getcwd()}\n"
                 "The filenames in the examples (nq_1m.csv etc.) are placeholders. "
                 "Pass the path to your own data file.")
    if path.lower().endswith((".parquet", ".pq")):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            sys.exit("Reading .parquet needs pyarrow: run  pip install pyarrow  (or use a CSV file).")
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = next((c for c in ("datetime", "timestamp", "ts_event", "date", "time", "ts") if c in df.columns),
                    df.columns[0])
        s = df[tcol]
        if pd.api.types.is_numeric_dtype(s):
            v = float(s.iloc[0])
            unit = "ns" if v > 1e17 else "ms" if v > 1e11 else "s"
            idx = pd.to_datetime(s, unit=unit, utc=True)
        else:
            try:
                idx = pd.to_datetime(s)
            except (ValueError, TypeError):
                idx = pd.to_datetime(s, utc=True)
        df = df.drop(columns=[tcol])
        df.index = pd.DatetimeIndex(idx)
    idx = df.index
    idx = idx.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward") if idx.tz is None else idx
    df.index = idx.tz_convert(NY)
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            sys.exit(f"{path}: missing column '{col}'")
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    step = int(round(df.index.to_series().diff().dropna().dt.total_seconds().median() / 60.0))
    if step in (1, 5):
        df = fill_gaps(df, step)
    if step == 1:
        df1 = df
        df5 = df.resample("5min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    elif step == 5:
        df1, df5 = None, df
    else:
        sys.exit(f"{path}: unsupported bar size ({step} min). Use 1-minute or 5-minute data.")
    return df5, df1


def fill_gaps(df, step, max_gap_min=30):
    """Vendors like Databento omit bars with no trades. Fill short gaps (<= max_gap_min) with flat
    zero-volume bars so sessions stay contiguous. Long gaps (maintenance break, weekends) stay empty."""
    full = pd.date_range(df.index[0], df.index[-1], freq=f"{step}min")
    d = df.reindex(full)
    miss = d["close"].isna().to_numpy()
    if not miss.any():
        return df
    grp = (~miss).cumsum()
    runlen = pd.Series(miss).groupby(grp).transform("sum").to_numpy()
    fill = miss & (runlen * step <= max_gap_min)
    d["close"] = d["close"].ffill()
    for col in ("open", "high", "low"):
        d[col] = d[col].where(~fill, d["close"])
    d["volume"] = d["volume"].where(~fill, 0.0)
    return d[(~miss) | fill]


def rma_sma(x, n):
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    if len(v) < n:
        return out
    out[n - 1] = np.nanmean(v[:n])
    for i in range(n, len(v)):
        out[i] = (out[i - 1] * (n - 1) + v[i]) / n
    return out


def jump_warning(df5, sym):
    c = df5["close"].to_numpy()
    r = np.abs(np.diff(c))
    med = np.median(r[r > 0]) if (r > 0).any() else 0
    if med > 0:
        big = int((r > 60 * med).sum())
        if big:
            print(f"  WARNING {sym}: {big} bar-to-bar jumps larger than 60x the median move. "
                  "Check for unadjusted contract rolls; they distort ATR and gap features.")


# ---------------------------------------------------------------- sessions and features
def build_instances(df5, df1, sym, cfg, use_1m):
    tick = cfg["tick"]
    idx = df5.index
    n = len(df5)
    mod = (idx.hour * 60 + idx.minute).to_numpy()
    o, h, l, c, v = (df5[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    ts_min = idx.asi8 / 6e10

    defs = []
    sess_of = np.zeros(n, int)
    for k, name in enumerate(SESS_ORDER, 1):
        sc = cfg["sessions"][name]
        if not sc.get("enabled", True):
            defs.append(None)
            continue
        s0, e0 = parse_sess(sc["sess"])
        os_, oe_ = parse_sess(sc["orw"])
        d = dict(name=name, k=k, s=s0, e=e0, orS=relM(os_, s0), orE=relM(oe_, s0),
                 entryRel=relM(hm(sc["entryEnd"]), s0), flatRel=relM(hm(sc["flat"]), s0),
                 length=((e0 - s0) % 1440) or 1440)
        if d["orE"] <= d["orS"]:
            sys.exit(f"{sym} {name}: opening range must end after it starts, inside the session")
        defs.append(d)
        mask = ((mod >= s0) & (mod < e0)) if s0 < e0 else ((mod >= s0) | (mod < e0))
        sess_of = np.where((sess_of == 0) & mask, k, sess_of)

    prev = np.r_[0, sess_of[:-1]]
    gap = np.r_[False, np.diff(ts_min) > 60]
    start_mask = (sess_of != 0) & ((sess_of != prev) | gap)

    # ADX with directional movement reset on the first bar of each session
    pc = np.r_[np.nan, c[:-1]]
    up = np.r_[0.0, np.diff(h)]
    dn = np.r_[0.0, -np.diff(l)]
    up[start_mask] = 0.0
    dn[start_mask] = 0.0
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr = np.where(start_mask | np.isnan(pc), h - l, tr)

    def wil(x):
        return pd.Series(x).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()

    atrw = wil(tr)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, mdi = 100 * wil(plus) / atrw, 100 * wil(minus) / atrw
        den = pdi + mdi
        dx = np.where(den > 0, 100 * np.abs(pdi - mdi) / np.where(den > 0, den, 1), 0.0)
    adx_all = wil(dx)

    # daily ATR over trading days (18:00 ET to 17:00 ET), known before each session starts
    lab = (idx - pd.Timedelta(hours=18)).normalize()
    dd = pd.DataFrame({"h": h, "l": l, "c": c, "lab": lab}).groupby("lab").agg(
        H=("h", "max"), L=("l", "min"), C=("c", "last"), n=("c", "size"))
    dd = dd[dd.n >= 150]
    dpc = dd.C.shift()
    dtr = np.where(dpc.isna(), dd.H - dd.L, np.maximum.reduce([dd.H - dd.L, (dd.H - dpc).abs(), (dd.L - dpc).abs()]))
    d_atr = rma_sma(dtr, 14)
    d_lab = dd.index.tz_localize(None).to_numpy()

    m1_map = {}
    if use_1m and df1 is not None:
        key = df1.index.floor("5min")
        for ts, grp in df1.groupby(key):
            m1_map[ts] = (grp["high"].to_numpy(), grp["low"].to_numpy())

    tp_all = (h + l + c) / 3
    insts = []
    sess_idx = np.flatnonzero(sess_of != 0)
    gid = np.cumsum(start_mask)
    for _, grp in pd.Series(sess_idx).groupby(gid[sess_idx]):
        s, e = int(grp.iloc[0]), int(grp.iloc[-1]) + 1
        d = defs[sess_of[s] - 1]
        nb = e - s
        if nb < 0.7 * d["length"] / 5 or (mod[s] - d["s"]) % 1440 != 0:
            continue
        rel = ((mod[s:e] - d["s"]) % 1440)
        if rel[0] != 0 or np.any(np.diff(rel) < 0):
            continue
        tp, vv = tp_all[s:e], v[s:e]
        cv, cpv, cpv2 = np.cumsum(vv), np.cumsum(tp * vv), np.cumsum(tp * tp * vv)
        with np.errstate(divide="ignore", invalid="ignore"):
            vw = np.where(cv > 0, cpv / cv, tp)
            var = np.where(cv > 0, cpv2 / cv - vw * vw, 0.0)
        sd = np.sqrt(np.maximum(var, 0.0))
        side = np.sign(c[s:e] - vw)
        flip = np.zeros(nb)
        flip[1:] = (side[1:] != 0) & (side[:-1] != 0) & (side[1:] != side[:-1])
        hh3 = pd.Series(h[s:e]).rolling(3, min_periods=1).max().to_numpy()
        ll3 = pd.Series(l[s:e]).rolling(3, min_periods=1).min().to_numpy()
        label = (idx[s] - pd.Timedelta(hours=18)).normalize().tz_localize(None).to_datetime64()
        pos = np.searchsorted(d_lab, label, side="left") - 1
        datr = float(d_atr[pos]) if pos >= 0 else NAN
        insts.append(dict(
            name=d["name"], k=d["k"], date=idx[s].date(), start=idx[s], n=nb, tick=tick,
            tday=(idx[s] - pd.Timedelta(hours=18)).normalize().date(),
            mins=rel.tolist(), o=o[s:e].tolist(), h=h[s:e].tolist(), l=l[s:e].tolist(), c=c[s:e].tolist(),
            v=vv.tolist(), vwap=vw.tolist(), vsd=sd.tolist(), cross=np.cumsum(flip).tolist(),
            adx=adx_all[s:e].tolist(), hh3=hh3.tolist(), ll3=ll3.tolist(), datr=datr,
            orS=d["orS"], oE=d["orE"], entryRel=d["entryRel"], flatRel=d["flatRel"],
            m1=[m1_map.get(t) for t in idx[s:e]] if m1_map else None,
        ))
    return insts


def add_features(insts):
    """Opening-range ratios against each session's own history, plus gap and expansion measures."""
    hist_r = {n: [] for n in SESS_ORDER}
    hist_v = {n: [] for n in SESS_ORDER}
    hist_g = {n: [] for n in SESS_ORDER}
    out = []
    last_close = NAN
    last_close_ny = NAN
    for D in insts:
        mins = D["mins"]
        oS, oE = D["orS"], D["oE"]
        gap_ref = last_close_ny if D["name"] == "NY" else last_close
        ok = (oS in mins) and (oE in mins)
        if ok:
            j0, jc = mins.index(oS), mins.index(oE)
            ok = (jc - j0) == (oE - oS) // 5 and (oE - oS) % 5 == 0
        last_close = D["c"][-1]
        if D["name"] == "NY":
            last_close_ny = last_close
        if not ok:
            continue
        h, l, c, o = D["h"], D["l"], D["c"], D["o"]
        orH, orL = max(h[j0:jc]), min(l[j0:jc])
        rng = orH - orL
        orVol = sum(D["v"][j0:jc])
        orOpen, orClose = o[j0], c[jc - 1]
        nm = D["name"]
        hr, hv, hg = hist_r[nm], hist_v[nm], hist_g[nm]
        if len(hr) >= 5:
            orRatio = rng / (sum(hr[-20:]) / len(hr[-20:]))
            av = sum(hv[-20:]) / len(hv[-20:])
            volRatio = orVol / av if av > 0 else NAN
        else:
            orRatio = volRatio = NAN
        hr.append(rng)
        hv.append(orVol)
        sess_rng = max(h) - min(l)
        typ = sess_rng / (sum(hg[-20:]) / len(hg[-20:])) if len(hg) >= 5 else NAN
        hg.append(sess_rng)
        datr = D["datr"]
        gap_ok = not (math.isnan(datr) or math.isnan(gap_ref) or datr <= 0)
        D2 = dict(D)
        D2.update(
            jc=jc, orH=orH, orL=orL, orRatio=orRatio, volRatio=volRatio,
            orEff=abs(orClose - orOpen) / rng if rng > 0 else 0.0,
            orUp=orClose > orOpen, orDn=orClose < orOpen,
            gapRatio=abs(o[0] - gap_ref) / datr if gap_ok else NAN,
            gapSigned=(o[0] - gap_ref) if not math.isnan(gap_ref) else 0.0,
            crossAtOR=D["cross"][jc - 1], adxAtOR=D["adx"][jc - 1],
            orAtr=rng / datr if (not math.isnan(datr) and datr > 0) else NAN,
            orMult=sess_rng / rng if rng > 0 else NAN, typMult=typ,
            sHi=max(h), sLo=min(l),
        )
        out.append(D2)
    return out


def compute_blocks(days, P):
    """Mirrors the indicator's block rule.
    overnight: Asia/London can block the next session; NY never blocks (so Asia is never blocked by NY).
    combined : Asia+London range vs its typical value blocks the NY session only.
    prev     : the previous session of any kind can block the next one."""
    if not P.get("blockOn", True):
        return [False] * len(days)
    src = P["blockSrc"]
    key = "orMult" if P["expandMode"] == "or" else "typMult"
    thr = P["expandOR"] if P["expandMode"] == "or" else P["expandTyp"]
    out, prev, prev_blocked, last_k = [], None, False, 0
    ov_hi = ov_lo = None
    hist = []
    for D in days:
        k = D["k"]
        b = False
        if src == "combined":
            if k == 3 and last_k != 3 and ov_hi is not None:
                rng = ov_hi - ov_lo
                mult = rng / (sum(hist[-20:]) / len(hist[-20:])) if len(hist) >= 5 else NAN
                hist.append(rng)
                b = P["expandTyp"] > 0 and not math.isnan(mult) and mult >= P["expandTyp"]
            if k in (1, 2):
                if last_k in (0, 3) or ov_hi is None:
                    ov_hi, ov_lo = D["sHi"], D["sLo"]
                else:
                    ov_hi, ov_lo = max(ov_hi, D["sHi"]), min(ov_lo, D["sLo"])
        elif prev is not None and (src == "prev" or prev["k"] != 3):
            pm = prev[key]
            expanded = thr > 0 and not math.isnan(pm) and pm >= thr
            b = expanded and ((not prev_blocked) or P["measureBlocked"])
        out.append(b)
        prev, prev_blocked, last_k = D, b, k
    return out


# ---------------------------------------------------------------- simulation
def first_hit(m1bar, pos, stop_px, t_level):
    if m1bar is None:
        return "stop"
    hs, ls = m1bar
    for a, b in zip(hs, ls):
        sh, th = (b <= stop_px, a >= t_level) if pos == 1 else (a >= stop_px, b <= t_level)
        if sh:
            return "stop"
        if th:
            return "target"
    return "stop"


def sim_session(D, P, out):
    nan = math.isnan
    datr, orRatio, volRatio = D["datr"], D["orRatio"], D["volRatio"]
    if nan(datr) or datr <= 0 or nan(orRatio) or nan(volRatio):
        return
    oRatioOK = P["orALo"] <= orRatio <= P["orAHi"]
    volOK = volRatio >= P["volMinA"]
    crossOK = D["crossAtOR"] <= P["crossAMax"]
    adxOK = D["adxAtOR"] >= P["adxA"]
    effOK = D["orEff"] >= P["orEffMin"]
    tt = P["aTrendTest"]
    trendOK = adxOK if tt == "adx" else (adxOK and effOK) if tt == "both" else effOK
    mo = P["maxORATR"]
    orATROK = mo <= 0 or ((not nan(D["orAtr"])) and D["orAtr"] <= mo)
    aDay = oRatioOK and volOK and crossOK and trendOK and orATROK
    isC = orRatio > P["orAHi"] or volRatio < P["volC"]
    gapRatio = D["gapRatio"]
    bDayOK = ((not isC) and (not aDay) and P["orBLo"] <= orRatio < P["orBHi"]
              and P["volBLo"] <= volRatio <= P["volBHi"] and (not nan(gapRatio)) and gapRatio < P["gapBMax"])
    canA = P["enableA"] and aDay
    canB = P["enableB"] and bDayOK
    if not (canA or canB):
        return

    TICK = D["tick"]
    mins, o, h, l, c = D["mins"], D["o"], D["h"], D["l"], D["c"]
    vwap, vsd, cross, adx = D["vwap"], D["vsd"], D["cross"], D["adx"]
    hh3, ll3, m1 = D["hh3"], D["ll3"], D["m1"]
    n, jc, oE = D["n"], D["jc"], D["oE"]
    aE, flat = D["entryRel"], D["flatRel"]
    orH, orL = D["orH"], D["orL"]
    orRng, orMid = orH - orL, (orH + orL) / 2
    orUp, orDn, gapSigned = D["orUp"], D["orDn"], D["gapSigned"]
    buf, capPts = P["bufTicks"] * TICK, P["stopCapATR"] * datr
    bandSD, bandFloor = P["bandSD"], P["bandFloorATR"]
    topPct, maxBarOR, maxChase = P["topPct"], P["maxBarOR"], P["maxChaseOR"]
    alignOR, gapAgainst, skipFirst = P["alignOR"], P["gapAgainst"], P["skipFirst"]
    crossBMin, adxB, bS, bE = P["crossBMin"], P["adxB"], P["bStartMins"], P["bEndMins"]
    costPts, timeStopBars, timeStopMinR = P["costPts"], P["timeStopBars"], P["timeStopMinR"]
    fullMode, aTargetR, aTimeStopOn, trailOn = P["aExitMode"] == "full", P["aTargetR"], P["aTimeStopOn"], P["trailOn"]
    useRule = P["useDailyRule"]
    sk = D["k"]

    aLongFired = aShortFired = closedAbove = closedBelow = False
    outUp = outDn = inCount = 0
    bOff = aOff = False
    pend = pendKind = 0
    pendStop = 0.0
    pos = posKind = 0
    entryPx = stopPx = riskPts = 0.0
    rAcc, remain, halfTaken, mfe, held, entryMin = 0.0, 1.0, False, 0.0, 0, 0
    tradesToday, dayBlocked = 0, False

    def record(reason, rGross):
        rNet = rGross - costPts / riskPts
        out.append((D["date"], posKind, pos, entryMin, entryPx, riskPts, reason, rNet, mfe / riskPts, held, sk))
        return rNet

    for i in range(min(jc, max(bS // 5, 1)), n):
        m, ci, hi, lo, op = mins[i], c[i], h[i], l[i], o[i]
        vw, sd = vwap[i], vsd[i]
        vU, vL = vw + bandSD * sd, vw - bandSD * sd
        if i < jc:
            if m >= bS:
                outUp = outUp + 1 if ci > vU else 0
                outDn = outDn + 1 if ci < vL else 0
                if outUp >= 3 or outDn >= 3:
                    bOff = True
            continue
        barRng = hi - lo
        posInBar = (ci - lo) / barRng if barRng > 0 else 0.5

        aLong = aShort = bLong = bShort = False
        if canA and (not aOff) and oE <= m < aE:
            if ((not skipFirst) or m >= oE + 5) and (maxBarOR <= 0 or barRng <= maxBarOR * orRng):
                if ((not aLongFired) and (not closedBelow) and ci > orH + buf and ci > vw
                        and posInBar >= topPct and (ci - orMid) <= capPts
                        and (maxChase <= 0 or ci - orH <= maxChase * orRng)
                        and ((not alignOR) or orUp)
                        and (gapAgainst <= 0 or gapSigned >= -gapAgainst * datr)):
                    aLong = True
                if ((not aShortFired) and (not closedAbove) and ci < orL - buf and ci < vw
                        and posInBar <= 1 - topPct and (orMid - ci) <= capPts
                        and (maxChase <= 0 or orL - ci <= maxChase * orRng)
                        and ((not alignOR) or orDn)
                        and (gapAgainst <= 0 or gapSigned <= gapAgainst * datr)):
                    aShort = True
        if canB and (not bOff) and bS <= m <= bE and m < flat and i >= 1:
            cross60 = cross[i] - (cross[i - 12] if i >= 12 else 0)
            if cross60 >= crossBMin and adx[i] < adxB:
                vwP, sdP = vwap[i - 1], vsd[i - 1]
                vUP, vLP = vwP + bandSD * sdP, vwP - bandSD * sdP
                stopS, stopL = hh3[i] + 2 * TICK, ll3[i] - 2 * TICK
                if (c[i - 1] > vUP and ci <= vU and outUp <= 2 and posInBar <= 0.5
                        and (vU - vw) >= bandFloor * datr and (stopS - ci) <= capPts
                        and (ci - vw) >= (stopS - ci)):
                    bShort = True
                elif (c[i - 1] < vLP and ci >= vL and outDn <= 2 and posInBar >= 0.5
                        and (vw - vL) >= bandFloor * datr and (ci - stopL) <= capPts
                        and (vw - ci) >= (ci - stopL)):
                    bLong = True

        if pend != 0 and pos == 0:
            risk0 = op - pendStop if pend == 1 else pendStop - op
            if risk0 > 0:
                pos, posKind, entryPx, stopPx, riskPts = pend, pendKind, op, pendStop, risk0
                rAcc, remain, halfTaken, mfe, held, entryMin = 0.0, 1.0, False, 0.0, 0, m
                tradesToday += 1
            pend = 0

        if pos != 0:
            held += 1
            contrib, reason = None, 0
            if trailOn and halfTaken and posKind == 1 and held > 1:
                stopPx = max(stopPx, l[i - 1]) if pos == 1 else min(stopPx, h[i - 1])
            if m >= flat:
                contrib, reason = remain * pos * (op - entryPx) / riskPts, 4
            else:
                stopFill = min(stopPx, op) if pos == 1 else max(stopPx, op)
                stopHit = (lo <= stopPx) if pos == 1 else (hi >= stopPx)
                tLevel, tMode = None, None
                if posKind == 1:
                    if fullMode:
                        tLevel, tMode = entryPx + pos * aTargetR * riskPts, "full"
                    elif not halfTaken:
                        tLevel, tMode = entryPx + pos * riskPts, "half"
                else:
                    tLevel, tMode = vwap[i - 1], "B"
                tHit = tLevel is not None and ((hi >= tLevel) if pos == 1 else (lo <= tLevel))
                if stopHit and tHit and m1 is not None and first_hit(m1[i], pos, stopPx, tLevel) == "target":
                    stopHit = False
                if stopHit:
                    contrib, reason = remain * pos * (stopFill - entryPx) / riskPts, (2 if halfTaken else 1)
                else:
                    mfe = max(mfe, (hi - entryPx) if pos == 1 else (entryPx - lo))
                    if tHit:
                        if tMode == "half":
                            rAcc += 0.5
                            remain = 0.5
                            halfTaken = True
                            stopPx = entryPx
                        else:
                            fill = max(op, tLevel) if pos == 1 else min(op, tLevel)
                            contrib, reason = remain * pos * (fill - entryPx) / riskPts, 5
                    if contrib is None and held >= timeStopBars:
                        aTS = posKind == 1 and aTimeStopOn and (not halfTaken) and mfe < timeStopMinR * riskPts
                        if aTS or posKind == 2:
                            contrib, reason = remain * pos * (ci - entryPx) / riskPts, 3
            if contrib is not None:
                if record(reason, rAcc + contrib) < 0:
                    dayBlocked = True
                pos = 0
                remain = 1.0

        if pos == 0 and pend == 0 and ((not useRule) or (tradesToday < 2 and not dayBlocked)):
            if aLong:
                pend, pendKind, pendStop = 1, 1, orMid
            elif aShort:
                pend, pendKind, pendStop = -1, 1, orMid
            elif bLong:
                pend, pendKind, pendStop = 1, 2, ll3[i] - 2 * TICK
            elif bShort:
                pend, pendKind, pendStop = -1, 2, hh3[i] + 2 * TICK

        if aLong:
            aLongFired = True
        if aShort:
            aShortFired = True
        if m >= bS:
            outUp = outUp + 1 if ci > vU else 0
            outDn = outDn + 1 if ci < vL else 0
            if outUp >= 3 or outDn >= 3:
                bOff = True
        if ci > orH:
            closedAbove = True
        if ci < orL:
            closedBelow = True
        if (closedAbove or closedBelow) and orL <= ci <= orH:
            inCount += 1
        else:
            inCount = 0
        if inCount >= 2:
            aOff = True

    if pos != 0:
        record(4, rAcc + remain * pos * (c[n - 1] - entryPx) / riskPts)


# ---------------------------------------------------------------- running many parameter sets
_DAYS = None


def _init(days):
    global _DAYS
    _DAYS = days


def _work(P):
    blocks = compute_blocks(_DAYS, P)
    out = []
    for D, b in zip(_DAYS, blocks):
        if not b:
            sim_session(D, P, out)
    return out


def run_many(days, plist, jobs):
    if jobs <= 1 or len(plist) < 4:
        _init(days)
        return [_work(P) for P in plist]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=(days,)) as ex:
        return list(ex.map(_work, plist, chunksize=max(1, len(plist) // (jobs * 4))))


def arrays(trades):
    if not trades:
        return np.array([], dtype="datetime64[D]"), np.array([], float), np.array([], int), np.array([], int)
    return (np.array([np.datetime64(t[0]) for t in trades]), np.array([t[7] for t in trades], float),
            np.array([t[1] for t in trades], int), np.array([t[10] for t in trades], int))


def stats(r):
    n = len(r)
    base = dict(n=n, win=NAN, exp=NAN, t=NAN, pf=NAN, total=0.0, dd=0.0)
    if n == 0:
        return base
    w, lo = r[r > 0], r[r < 0]
    sd = r.std(ddof=1) if n > 1 else NAN
    cum = np.concatenate([[0.0], np.cumsum(r)])
    base.update(win=100.0 * len(w) / n, exp=float(r.mean()),
                t=float(r.mean() / (sd / math.sqrt(n))) if sd and sd > 0 else NAN,
                pf=float(w.sum() / -lo.sum()) if lo.sum() < 0 else float("inf"),
                total=float(r.sum()), dd=float((np.maximum.accumulate(cum) - cum).max()))
    return base


def f2(x, nd=2):
    return "-" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{nd}f}"


def boot_ci(r, reps=2000, seed=7):
    if len(r) < 8:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(reps, len(r)), replace=True).mean(axis=1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def diff_str(P, base):
    d = [f"{k}={v}" for k, v in P.items() if base.get(k) != v]
    return ", ".join(d) if d else "(baseline)"


# ---------------------------------------------------------------- calibration and verdicts
def sample_params(base, space, n, seed):
    rng = random.Random(seed)
    keys = sorted(space)
    plist, seen, tries = [dict(base)], {tuple(base.get(k) for k in keys)}, 0
    while len(plist) < n + 1 and tries < n * 30:
        tries += 1
        p = dict(base)
        for k in keys:
            p[k] = rng.choice(space[k])
        sig = tuple(p[k] for k in keys)
        if sig not in seen:
            seen.add(sig)
            plist.append(p)
    return plist


def session_table(d, r, k, sp):
    rows = {}
    for s in ACTIVE:
        m = k == s
        rows[s] = (stats(r[m & (d < sp)]), stats(r[m & (d >= sp)]))
    return rows


def calibrate_instrument(sym, days, base, space, args, sp):
    cands = sample_params(base, space, args.samples, args.seed)
    results = run_many(days, cands, args.jobs)
    data = []
    for P, res in zip(cands, results):
        d, r, kd, sk = arrays(res)
        data.append((P, d, r, kd, sk))
    # pick the best candidate by TRAIN t-stat (pooled over sessions)
    best_i, best_t = None, -1e9
    for i, (P, d, r, kd, sk) in enumerate(data):
        st = stats(r[d < sp])
        if st["n"] >= args.min_trades and not math.isnan(st["t"]) and st["t"] > best_t:
            best_i, best_t = i, st["t"]
    n_trials = len(data)
    print(f"\n{'=' * 78}\n{sym}: {n_trials} parameter sets tried. Train: before {sp}. Test: from {sp}.")
    print(f"Luck benchmark: the best of {n_trials} no-edge sets typically shows a train t-stat near "
          f"{math.sqrt(2 * math.log(max(n_trials, 2))):.1f}.")
    if best_i is None:
        print(f"No parameter set reached {args.min_trades} train trades. Use more data, loosen filters, "
              "or lower --min-trades.")
        return None, []
    P, d, r, kd, sk = data[best_i]
    bench = math.sqrt(2 * math.log(max(n_trials, 2)))
    pooled_tr, pooled_te = stats(r[d < sp]), stats(r[d >= sp])
    weak_global = pooled_tr["t"] < bench
    print(f"\nChosen set (best train t-stat): {diff_str(P, base)}")
    n_days = max(1, len({D["tday"] for D in days}))
    print(f"  mean trades per trading day (all data): {len(r) / n_days:.2f} over {n_days} trading days")
    print(f"  pooled train: n={pooled_tr['n']} exp={f2(pooled_tr['exp'], 3)}R t={f2(pooled_tr['t'])} "
          f"PF={f2(pooled_tr['pf'])}")
    print(f"  pooled test : n={pooled_te['n']} exp={f2(pooled_te['exp'], 3)}R t={f2(pooled_te['t'])} "
          f"PF={f2(pooled_te['pf'])}")
    if weak_global:
        print(f"  WARNING: the chosen set's train t-stat ({f2(pooled_tr['t'])}) is below the luck benchmark "
              f"({bench:.1f}). Its settings are not distinguishable from the best of {n_trials} random picks.")

    # breadth: how many candidate sets are positive in each session (not cherry-picked)
    breadth = {}
    for s in ACTIVE:
        tr_e, te_e = [], []
        for (Pc, dc, rc, kc, skc) in data:
            m = skc == s
            a = rc[m & (dc < sp)]
            b = rc[m & (dc >= sp)]
            if len(a) >= args.cand_min:
                tr_e.append(a.mean())
                if len(b) >= 5:
                    te_e.append(b.mean())
        breadth[s] = dict(
            n_c=len(tr_e), med_tr=float(np.median(tr_e)) if tr_e else NAN,
            pos_tr=float(np.mean(np.array(tr_e) > 0)) if tr_e else NAN,
            med_te=float(np.median(te_e)) if te_e else NAN,
            pos_te=float(np.mean(np.array(te_e) > 0)) if te_e else NAN)

    tbl = session_table(d, r, sk, sp)
    rows, summary = [], []
    for s in ACTIVE:
        name = SESS_ORDER[s - 1]
        tr, te = tbl[s]
        bd = breadth[s]
        mask_te = (sk == s) & (d >= sp)
        lo, hi = boot_ci(r[mask_te])
        # verdict from TRAIN evidence only
        if tr["n"] < args.min_session_trades:
            train_v, why = "FILTER", f"only {tr['n']} train trades (need {args.min_session_trades})"
        elif tr["exp"] > 0 and bd["med_tr"] > 0 and bd["pos_tr"] >= 0.55:
            train_v, why = "KEEP", "positive on train for the chosen set and for most parameter sets"
        elif tr["exp"] <= 0:
            train_v, why = "FILTER", "chosen set has no edge in this session on train"
        else:
            train_v, why = "FILTER", "edge depends on the chosen parameters (most sets lose here)"
        # out-of-sample confirmation
        if te["n"] < max(10, args.min_session_trades // 3):
            conf = "too few test trades"
        elif te["exp"] > 0 and (math.isnan(bd["med_te"]) or bd["med_te"] > 0):
            conf = "confirmed"
        else:
            conf = "failed on test"
        weak = weak_global or (not math.isnan(lo) and lo < 0)
        if train_v == "KEEP" and conf == "confirmed":
            final = "KEEP (weak evidence)" if weak else "KEEP"
        elif train_v == "KEEP":
            final = "WATCH (paper trade only)"
        else:
            final = "FILTER OUT"
        if train_v != "KEEP":
            conf = "n/a (filtered on train)"
        rows.append({"session": name, "verdict": final,
                     "tr n": tr["n"], "tr exp": f2(tr["exp"], 3), "tr t": f2(tr["t"]),
                     "te n": te["n"], "te exp": f2(te["exp"], 3),
                     "te 90% CI": "-" if math.isnan(lo) else f"[{lo:.2f},{hi:.2f}]",
                     "sets>0 tr": "-" if math.isnan(bd["pos_tr"]) else f"{100 * bd['pos_tr']:.0f}%",
                     "sets>0 te": "-" if math.isnan(bd["pos_te"]) else f"{100 * bd['pos_te']:.0f}%",
                     "test check": conf})
        summary.append(dict(instrument=sym, session=name, verdict=final, train_n=tr["n"], train_exp=tr["exp"],
                            train_t=tr["t"], test_n=te["n"], test_exp=te["exp"], breadth_train=bd["pos_tr"],
                            breadth_test=bd["pos_te"], test_check=conf, reason=why))
    pd.set_option("display.width", 200)
    print("\nSession verdicts (decision uses TRAIN data; TEST only confirms):")
    print(pd.DataFrame(rows).to_string(index=False))
    for x in summary:
        print(f"  {x['session']:7s} -> {x['verdict']}: {x['reason']}; test check: {x['test_check']}")

    kept = [s for s, x in zip(ACTIVE, summary) if x["verdict"].startswith("KEEP")]
    drop = [s for s in ACTIVE if s not in kept]
    kept_te = stats(r[(d >= sp) & np.isin(sk, kept)])
    drop_te = stats(r[(d >= sp) & np.isin(sk, drop)])
    print(f"\nTest period, chosen parameter set:")
    print(f"  all sessions           n={pooled_te['n']:4d}  exp={f2(pooled_te['exp'], 3)}R")
    print(f"  kept sessions only     n={kept_te['n']:4d}  exp={f2(kept_te['exp'], 3) if kept_te['n'] else 'n/a'} R   "
          f"({', '.join(SESS_ORDER[s - 1] for s in kept) or 'none'})")
    print(f"  filtered-out sessions  n={drop_te['n']:4d}  exp={f2(drop_te['exp'], 3) if drop_te['n'] else 'n/a'} R")

    print("\nIndicator settings for this instrument (set 'A filter preset' to Custom):")
    nice = {"overnight": "Overnight sessions only", "combined": "Combined overnight range",
            "prev": "Previous session (any)"}
    for k_, name in PINE_NAMES.items():
        if k_ in P:
            print(f"  {name}: {nice.get(P[k_], P[k_]) if k_ == 'blockSrc' else P[k_]}")
    print("  Sessions to leave ENABLED: " + (", ".join(SESS_ORDER[s - 1] for s in kept) or "none (nothing passed)"))
    print("  Sessions to DISABLE: " + (", ".join(SESS_ORDER[s - 1] for s in drop) or "none"))

    trades = results[best_i]
    return summary, trades


def debug_day(sym, days, P, date_str):
    """Print one session's features so they can be compared with the indicator's table."""
    nan = math.isnan
    target = pd.Timestamp(date_str).date()
    blocks = compute_blocks(days, P)
    hits = [(D, b) for D, b in zip(days, blocks) if D["date"] == target]
    if not hits:
        print(f"\n{sym}: no usable session starting on {target} (check the date and the data range).")
        return
    for D, blocked in hits:
        orRatio, volRatio = D["orRatio"], D["volRatio"]
        oRatioOK = (not nan(orRatio)) and P["orALo"] <= orRatio <= P["orAHi"]
        volOK = (not nan(volRatio)) and volRatio >= P["volMinA"]
        crossOK = D["crossAtOR"] <= P["crossAMax"]
        adxOK = D["adxAtOR"] >= P["adxA"]
        effOK = D["orEff"] >= P["orEffMin"]
        tt = P["aTrendTest"]
        trendOK = adxOK if tt == "adx" else (adxOK and effOK) if tt == "both" else effOK
        mo = P["maxORATR"]
        orATROK = mo <= 0 or ((not nan(D["orAtr"])) and D["orAtr"] <= mo)
        first_fail = ("OR ratio" if not oRatioOK else "Volume" if not volOK else "VWAP crosses" if not crossOK
                      else "Trend test" if not trendOK else "OR too wide" if not orATROK else "none")
        out = []
        if not blocked:
            sim_session(D, P, out)
        print(f"\n{sym} {D['name']} session on {D['date']}   (compare with the indicator's table)")
        print(f"  OR high / low         : {D['orH']:.2f} / {D['orL']:.2f}   (range {D['orH'] - D['orL']:.2f})")
        print(f"  OR ratio              : {f2(orRatio)}")
        print(f"  Volume ratio          : {f2(volRatio)}")
        print(f"  Gap / ATR             : {f2(D['gapRatio'])}")
        print(f"  OR efficiency         : {f2(D['orEff'])}")
        print(f"  OR / daily ATR        : {f2(D['orAtr'])}")
        print(f"  Daily ATR             : {f2(D['datr'])}")
        print(f"  VWAP crosses at OR end: {int(D['crossAtOR'])}      ADX at OR end: {f2(D['adxAtOR'])}")
        print(f"  A blocked by          : {first_fail}")
        print(f"  Session blocked       : {'YES' if blocked else 'no'}")
        if out:
            for t in out:
                when = (D["start"] + pd.Timedelta(minutes=int(t[3]))).strftime("%H:%M")
                side = "long" if t[2] == 1 else "short"
                names = {1: "initial stop", 2: "break-even/trail", 3: "time stop", 4: "hard exit", 5: "target"}
                print(f"  Trade: {'A' if t[1] == 1 else 'B'} {side} filled {when} ET at {t[4]:.2f}, "
                      f"risk {t[5]:.1f} pts, exit {names.get(t[6], t[6])}, {t[7]:+.2f}R")
        else:
            print("  Trades: none")


def P_src(P):
    return {"overnight": "overnight sessions only", "combined": "combined overnight range",
            "prev": "previous session"}.get(P["blockSrc"], P["blockSrc"])


def run_mode(sym, days, base, args, sp):
    res = run_many(days, [base], 1)[0]
    d, r, kd, sk = arrays(res)
    print(f"\n{'=' * 78}\n{sym}: single run, {len(days)} sessions analysed. Train: before {sp}. Test: from {sp}.")
    n_days = max(1, len({D["tday"] for D in days}))
    rows = []
    for s in ACTIVE:
        m = sk == s
        tr, te = stats(r[m & (d < sp)]), stats(r[m & (d >= sp)])
        allv = stats(r[m])
        ok = tr["n"] >= args.min_session_trades and tr["exp"] > 0 and te["n"] >= 10 and te["exp"] > 0
        rows.append({"session": SESS_ORDER[s - 1], "trades": allv["n"], "per day": f2(allv["n"] / n_days),
                     "win%": f2(allv["win"], 0),
                     "exp(R)": f2(allv["exp"], 3), "PF": f2(allv["pf"]), "tr n": tr["n"], "tr exp": f2(tr["exp"], 3),
                     "te n": te["n"], "te exp": f2(te["exp"], 3),
                     "suggestion": "keep" if ok else "filter out / unproven"})
    st = stats(r)
    rows.append({"session": "ALL", "trades": st["n"], "per day": f2(st["n"] / n_days), "win%": f2(st["win"], 0),
                 "exp(R)": f2(st["exp"], 3),
                 "PF": f2(st["pf"]), "tr n": "", "tr exp": "", "te n": "", "te exp": "", "suggestion": ""})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nMean trades per trading day: {st['n'] / n_days:.2f} over {n_days} trading days "
          f"(blocking {'on, source ' + P_src(base) if base.get('blockOn', True) else 'off'})")
    return res


# ---------------------------------------------------------------- main
def parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v.lower() if v.lower() in ("true", "false") else v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="run", choices=["calibrate", "run"])
    ap.add_argument("--nq")
    ap.add_argument("--gc")
    ap.add_argument("--tz", default=NY)
    ap.add_argument("--config", help="JSON file overriding sessions/costs per instrument")
    ap.add_argument("--split", help="YYYY-MM-DD train/test boundary (default: 70%% of the history)")
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--set", action="append")
    ap.add_argument("--include-b", action="store_true")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--min-trades", type=int, default=100, help="min pooled train trades to pick a set")
    ap.add_argument("--min-session-trades", type=int, default=30, help="min train trades for a session verdict")
    ap.add_argument("--cand-min", type=int, default=15, help="min trades for a set to count in breadth stats")
    ap.add_argument("--sessions", help="only analyse these sessions, e.g. NY or London,NY (default: all)")
    ap.add_argument("--debug-date", help="YYYY-MM-DD: print that session's features and trades, then exit")
    ap.add_argument("--no-1m", action="store_true", help="ignore 1-minute data (saves memory; ties hit stop first)")
    ap.add_argument("--out-prefix", default="cal")
    args = ap.parse_args()

    files = {k: v for k, v in (("NQ", args.nq), ("GC", args.gc)) if v}
    if not files:
        sys.exit("Give at least one of --nq / --gc")
    cfg = json.loads(json.dumps(INSTR))
    if args.config:
        over = json.load(open(args.config))
        for sym, v in over.items():
            for key, val in v.items():
                if key in SESS_ORDER:
                    cfg[sym]["sessions"][key].update(val)
                else:
                    cfg[sym][key] = val

    global ACTIVE
    if args.sessions:
        names = {n.lower(): n for n in SESS_ORDER}
        keep = []
        for part in args.sessions.split(","):
            if part.strip().lower() not in names:
                sys.exit(f"Unknown session '{part}'. Choose from: {', '.join(SESS_ORDER)}")
            keep.append(names[part.strip().lower()])
        for sym in cfg:
            for name in SESS_ORDER:
                if name not in keep:
                    cfg[sym]["sessions"][name]["enabled"] = False
        ACTIVE = tuple(i for i, n in enumerate(SESS_ORDER, 1) if n in keep)
        print(f"Analysing sessions: {', '.join(keep)}")

    base = dict(DEFAULTS)
    if args.preset:
        base.update(PRESETS[args.preset])
    base.update(parse_sets(args.set))
    if args.include_b:
        base["enableB"] = True
    space = dict(SPACE_A)
    if base["enableB"]:
        space.update(SPACE_B)

    all_summary = []
    for sym, path in files.items():
        print(f"\nLoading {sym} from {path} ...")
        df5, df1 = load_bars(path, args.tz)
        jump_warning(df5, sym)
        insts = build_instances(df5, df1, sym, cfg[sym], use_1m=not args.no_1m)
        days = add_features(insts)
        if not days:
            print(f"{sym}: no complete sessions found. Check the file has overnight bars and the session windows.")
            continue
        counts = {n: sum(1 for D in days if D["name"] == n) for n in SESS_ORDER}
        print(f"  {len(days)} usable sessions: " + ", ".join(f"{n} {c}" for n, c in counts.items())
              + (" (1-minute tie resolution on)" if (df1 is not None and not args.no_1m) else " (ties assume stop first)"))
        P = dict(base)
        P["costPts"] = base["costPts"] if base["costPts"] is not None else cfg[sym]["cost"]
        dates = sorted({np.datetime64(D["date"]) for D in days})
        sp = np.datetime64(args.split) if args.split else dates[int(len(dates) * 0.7)]
        if args.debug_date:
            debug_day(sym, days, P, args.debug_date)
            continue
        if args.mode == "run":
            res = run_mode(sym, days, P, args, sp)
        else:
            summary, res = calibrate_instrument(sym, days, P, space, args, sp)
            if summary:
                all_summary.extend(summary)
        if res:
            pd.DataFrame(res, columns=["date", "kind", "side", "entry_rel_min", "entry", "risk_pts", "reason", "r",
                                       "mfe_r", "bars_held", "session"]).to_csv(
                f"{args.out_prefix}_{sym}_trades.csv", index=False)

    if all_summary:
        print(f"\n{'=' * 78}\nSESSION FILTER SUMMARY")
        df = pd.DataFrame(all_summary)
        show = df.copy()
        for col in ("train_exp", "test_exp"):
            show[col] = show[col].map(lambda x: f2(x, 3))
        show["train_t"] = show["train_t"].map(f2)
        for col in ("breadth_train", "breadth_test"):
            show[col] = show[col].map(lambda x: "-" if math.isnan(x) else f"{100 * x:.0f}%")
        print(show.drop(columns=["reason"]).to_string(index=False))
        keep = df[df.verdict.str.startswith("KEEP")]
        print("\nTrade only: " + (", ".join(f"{a} {b}" for a, b in zip(keep.instrument, keep.session)) or "nothing passed"))
        print("Everything else: filter out (or paper trade the WATCH rows only).")
        if df.verdict.str.contains("weak").any():
            print("Rows marked 'weak evidence' beat zero in the sample but are not statistically separable "
                  "from luck: paper trade them only.")
        print(f"\nThis summary covers {len(df)} instrument/session cells plus the parameter search, so some "
              "false positives are expected by chance. Treat a KEEP as a candidate for forward testing, "
              "not proof of an edge.")
        df.to_csv(f"{args.out_prefix}_session_filter.csv", index=False)


if __name__ == "__main__":
    main()
