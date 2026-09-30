"""NSE bulk deals with buyer / seller names, 2016 -> today, one month per request (2026-09-30).

Plain English: a bulk deal is any trade where one client buys or sells more than 0.5% of a company's shares in a day.
NSE publishes the client's name. This downloads every month's list (NSE's own CSV) slowly, keeps each month as a
raw file, and builds one table. Used by the research queue (item 2) and its A/B test (test_bulk_buyers.py).
Source (official): https://www.nseindia.com/api/historicalOR/bulk-block-short-deals?optionType=bulk_deals&from=DD-MM-YYYY&to=DD-MM-YYYY&csv=true
  (the JSON form of the same address stops at 70 rows; the CSV form returns the whole month).
Files: data/raw/bulk_deals/<YYYYMM>.csv (cached; the current and previous month are re-downloaded),
       data/derived/bulk_deals_history.parquet (+ manifest).
Usage: backfill_bulk_deals.py [--max-minutes N] [--recent]   (--recent: only the last two months, for daily refresh)
Prints COMPLETE when every month from 2016-01 to this month is present.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.nse.session import build_session  # noqa: E402

REF = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"
URL = "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals?optionType=bulk_deals&from={f}&to={t}&csv=true"
RAW = ROOT / "data/raw/bulk_deals"
OUT = ROOT / "data/derived/bulk_deals_history.parquet"
COLS = {"Date": "trade_date", "Symbol": "symbol", "Security Name": "security", "Client Name": "client_name",
        "Buy / Sell": "buy_sell", "Quantity Traded": "qty", "Trade Price / Wght. Avg. Price": "price", "Remarks": "remarks"}


def months(start: date, end: date) -> list[pd.Period]:
    return list(pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--recent", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    today = date.today()
    allm = months(date(2016, 1, 1), today)
    fresh = set(allm[-2:])
    todo = [m for m in allm if m in fresh or not (RAW / f"{m.strftime('%Y%m')}.csv").exists()]
    if a.recent:
        todo = allm[-2:]
    s = build_session(warm=True, referer=REF)
    got = 0
    for m in todo:
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            print("time box reached"); break
        f, t = m.start_time.strftime("%d-%m-%Y"), min(m.end_time.date(), today).strftime("%d-%m-%Y")
        r = s.get(URL.format(f=f, t=t), headers={"Referer": REF}, timeout=90)
        if r.status_code != 200 or "csv" not in r.headers.get("content-type", ""):
            print(f"  {m}: HTTP {r.status_code} {r.headers.get('content-type', '')[:30]} — retry next run"); time.sleep(5); continue
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]
        (RAW / f"{m.strftime('%Y%m')}.csv").write_bytes(r.content)
        print(f"  {m}: {len(df)} deals", flush=True); got += 1
        time.sleep(2.5)
    frames = []
    for p in sorted(RAW.glob("*.csv")):
        df = pd.read_csv(p); df.columns = [c.strip() for c in df.columns]
        frames.append(df.rename(columns=COLS)[list(COLS.values())])
    B = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(COLS.values()))
    B["trade_date"] = pd.to_datetime(B["trade_date"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
    for c in ("symbol", "security", "client_name", "buy_sell", "remarks"):
        B[c] = B[c].astype(str).str.strip()
    B["buy_sell"] = B["buy_sell"].str.upper()
    for c in ("qty", "price"):
        B[c] = pd.to_numeric(B[c].astype(str).str.replace(",", ""), errors="coerce")
    B = B.dropna(subset=["trade_date"]).drop_duplicates().sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    B.to_parquet(OUT, index=False)
    have = {p.stem for p in RAW.glob("*.csv")}
    missing = [m for m in allm if m.strftime("%Y%m") not in have]
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="bulk_deals_history", path=str(OUT.relative_to(ROOT)), rows=len(B),
        first=str(B["trade_date"].min().date()) if len(B) else None, last=str(B["trade_date"].max().date()) if len(B) else None,
        months_missing=[str(m) for m in missing], source=URL.replace("{f}", "DD-MM-YYYY").replace("{t}", "DD-MM-YYYY"),
        columns=dict(trade_date="trade date (IST)", symbol="NSE symbol", client_name="buyer / seller as published by NSE",
                     buy_sell="BUY or SELL", qty="shares", price="rupees per share (trade or weighted average price)", remarks="as published"),
        definition="bulk deal = a client's trades in one stock on one day totalling more than 0.5% of its listed shares",
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"rows {len(B)} · months fetched this run {got} · months missing {len(missing)} · {time.time() - t0:.0f}s")
    if not missing:
        print("COMPLETE")


if __name__ == "__main__":
    main()
