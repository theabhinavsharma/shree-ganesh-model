"""Fetch India VIX history from NSE (historicalOR/vixhistory), windowed + checkpointed.

Full backfill 2015→today on first run; daily incremental afterwards (last 30 days
re-fetched, dedup on date). Output: data/derived/india_vix.parquet
Columns: date, vix_close (+ open/high/low/prev where served).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
from src.ingest.nse.session import build_session  # noqa: E402
from src.ingest.nse.api import get_json  # noqa: E402

OUT = ROOT / "data/derived/india_vix.parquet"


def fetch_window(s, frm: pd.Timestamp, to: pd.Timestamp) -> pd.DataFrame | None:
    url = ("https://www.nseindia.com/api/historicalOR/vixhistory"
           f"?from={frm.strftime('%d-%m-%Y')}&to={to.strftime('%d-%m-%Y')}")
    j = get_json(s, url)
    d = j.get("data", []) if isinstance(j, dict) else (j or [])
    if not d:
        return None
    df = pd.DataFrame(d)
    datecol = next((c for c in df.columns if "DATE" in c.upper() or c == "EOD_TIMESTAMP"), df.columns[0])
    closecol = next((c for c in df.columns if "CLOS" in c.upper()), None)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[datecol], errors="coerce", dayfirst=True),
        "vix_close": pd.to_numeric(df[closecol], errors="coerce") if closecol else pd.NA,
    })
    return out.dropna(subset=["date"])


def main() -> None:
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["date", "vix_close"])
    if len(old):
        start = pd.to_datetime(old["date"]).max() - pd.Timedelta(days=30)
    else:
        start = pd.Timestamp("2015-01-01")
    s = build_session()
    frames = [old]
    cur = start
    today = pd.Timestamp.today().normalize()
    while cur <= today:
        to = min(cur + pd.Timedelta(days=89), today)
        try:
            df = fetch_window(s, cur, to)
            if df is not None and len(df):
                frames.append(df)
                print(f"  {cur.date()}→{to.date()}: {len(df)} rows", flush=True)
            else:
                print(f"  {cur.date()}→{to.date()}: 0 rows", flush=True)
        except Exception as e:
            print(f"  {cur.date()} ERROR {str(e)[:70]}", flush=True)
            s = build_session()
        cur = to + pd.Timedelta(days=1)
        time.sleep(2)
    allv = pd.concat(frames, ignore_index=True)
    allv["date"] = pd.to_datetime(allv["date"])
    allv = (allv.dropna(subset=["vix_close"]).sort_values("date")
                .drop_duplicates("date", keep="last").reset_index(drop=True))
    allv.to_parquet(OUT, index=False)
    print(f"india_vix.parquet: {len(allv):,} rows · {allv['date'].min().date()} → {allv['date'].max().date()}")


if __name__ == "__main__":
    main()
