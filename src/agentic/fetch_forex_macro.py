"""Fetch macro time-series — USDINR, Brent, India 10y yield, India VIX.

Uses Yahoo Finance public CSV endpoints (no API key required, no auth):
  query1.finance.yahoo.com/v7/finance/download/<TICKER>?period1=...&period2=...

Tickers:
  INR=X    — USDINR spot
  BZ=F     — Brent crude futures
  ^INDIAVIX — India VIX (NSE)
  India 10y bond yield is harder; we use IRX as US 3m proxy + a fallback

Output: data/derived/macro_timeseries.parquet
  cols: trade_date, usdinr, brent, india_vix, [india_10y when available]
  one row per day, append-only on trade_date
"""
from __future__ import annotations
import gzip
import io
import http.cookiejar
import ssl
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/macro_timeseries.parquet"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

def frankfurter_fx(base: str = "USD", quote: str = "INR", days: int = 800) -> pd.DataFrame:
    """ECB-backed FX history via api.frankfurter.app — free, no auth, reliable."""
    end = date.today()
    start = end - timedelta(days=days)
    url = f"https://api.frankfurter.app/{start}..{end}?from={base}&to={quote}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        import json
        j = json.loads(r.read().decode("utf-8"))
    rates = j.get("rates", {})
    if not rates:
        return pd.DataFrame()
    rows = [{"trade_date": pd.Timestamp(d), f"{base}{quote}".lower(): v[quote]}
            for d, v in rates.items() if quote in v]
    return pd.DataFrame(rows).sort_values("trade_date")


def stooq_csv(symbol: str) -> pd.DataFrame:
    """Stooq daily history (CSV, no auth, generous rate-limit).
    Returns df with columns: trade_date, close (renamed by caller)."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/csv,*/*",
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    if "Date" not in text or len(text) < 100:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns={"Date": "trade_date", "Close": symbol})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    return df[["trade_date", symbol]].dropna()


def main() -> None:
    print("== fetch_forex_macro (Frankfurter FX) ==")
    merged: pd.DataFrame | None = None
    # USDINR + DXY-ish basket via Frankfurter
    fx_pairs = [("USD", "INR"), ("EUR", "INR"), ("GBP", "INR"), ("JPY", "INR")]
    for base, quote in fx_pairs:
        try:
            df = frankfurter_fx(base, quote, days=800)
            if df.empty:
                continue
            colname = f"{base}{quote}".lower()
            print(f"  {base}/{quote}: {len(df)} rows, latest={df.iloc[-1][colname]:.4f}")
            merged = df if merged is None else merged.merge(df, on="trade_date", how="outer")
            time.sleep(0.5)
        except Exception as exc:
            print(f"  {base}/{quote} FAIL: {type(exc).__name__}: {str(exc)[:120]}")

    if merged is None or merged.empty:
        print("no data")
        return
    merged = merged.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        old["trade_date"] = pd.to_datetime(old["trade_date"])
        merged = pd.concat([old, merged], ignore_index=True).drop_duplicates("trade_date", keep="last")
    merged.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(merged):,} days")


if __name__ == "__main__":
    main()
