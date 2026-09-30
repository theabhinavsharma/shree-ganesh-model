"""NSE trading holidays (2026-09-30) -> data/derived/nse_holidays.parquet (+ .manifest.json).

Why: the weekly buy list is built on the last session's close and says which day to buy. Without the exchange's
holiday list it assumed "next weekday", so a list built on Thu 2026-10-01 would have said "buy Fri Oct 2", a holiday
(Gandhi Jayanti). Used by nse_calendar.next_session() in notify.py and screen_sri_lakshmi.py.
Source (official): https://www.nseindia.com/api/holiday-master?type=trading, segment CM (equities), fetched through the
repo's NSE session (TLS verification on). NSE publishes the current year only, so every fetch is MERGED into the file:
past years stay, the current year is replaced by NSE's latest list. 0 rows from NSE is treated as a failure.
Columns: date, weekday, description, segment, source_url, fetched_at (NSE gives no publication date for the list).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.nse.api import get_json  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

REF = "https://www.nseindia.com/resources/exchange-communication-holidays"
API = "https://www.nseindia.com/api/holiday-master?type=trading"
OUT = ROOT / "data/derived/nse_holidays.parquet"


def main() -> None:
    j = get_json(build_session(warm=True, referer=REF), API, referer=REF)
    cm = j.get("CM", []) if isinstance(j, dict) else []
    if not cm:
        raise SystemExit("NSE holiday-master returned 0 CM rows — not written (0 rows is a failure, not 'no holidays')")
    now = datetime.now().isoformat(timespec="seconds")
    new = pd.DataFrame([dict(date=pd.to_datetime(r["tradingDate"], format="%d-%b-%Y"), weekday=r.get("weekDay"),
                             description=r.get("description"), segment="CM", source_url=API, fetched_at=now) for r in cm])
    years = set(new["date"].dt.year)
    old = pd.read_parquet(OUT) if OUT.exists() else new.iloc[0:0]
    old = old[~pd.to_datetime(old["date"]).dt.year.isin(years)]
    H = pd.concat([old, new], ignore_index=True).sort_values("date").reset_index(drop=True)
    H.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="nse_holidays.parquet", producer="src/agentic/fetch_nse_holidays.py", source=API, segment="CM (equities)",
        rows=len(H), years=sorted(int(y) for y in pd.to_datetime(H["date"]).dt.year.unique()), updated=now,
        columns=dict(date="holiday date (no normal trading session)", weekday="as NSE gives it", description="as NSE gives it",
                     segment="CM = capital market (equities)", source_url="NSE API the row came from",
                     fetched_at="when we fetched it; NSE gives no publication date"),
        known_gaps="NSE publishes only the current year; earlier years exist only if an earlier fetch stored them. "
                   "Special sessions (Muhurat trading on Diwali) still trade although the day is listed.",
        qc="eval data.holidays_match_panel: the list covers this year, and this year's listed weekday holidays are absent "
           "from the price panel while every other weekday up to the last session is present"), indent=1))
    print(f"NSE holidays: {len(new)} CM rows for {sorted(years)} · file {len(H)} rows")


if __name__ == "__main__":
    main()
