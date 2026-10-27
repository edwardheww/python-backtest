#!/usr/bin/env python3
"""
Download about 59 days of 5-minute NQ and GC bars from Yahoo Finance.
This is for a SMOKE TEST of calibrate_sessions.py only. 59 days is far too little for real verdicts.

  pip install yfinance pandas
  python3 download_test_data.py
  python3 calibrate_sessions.py --nq nq_5m.csv --gc gc_5m.csv run --preset balanced
"""
import pandas as pd
import yfinance as yf

for sym, name in (("NQ=F", "nq_5m.csv"), ("GC=F", "gc_5m.csv")):
    df = yf.download(sym, period="59d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        print(f"No data returned for {sym}. Try again in a few minutes.")
        continue
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Datetime"
    df.to_csv(name)
    print(f"{name}: {len(df)} bars, {df.index.min()} to {df.index.max()}")
