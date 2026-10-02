"""Completeness of our NSE announcements archive vs NSE's own feed (2026-10-02; event-ledger data gate).

Yardstick: for sample days, NSE's whole-day count from api/corporate-announcements (index=equities, from=to=that day)
versus the filings our archive holds for that day. (NSE's results calendar is the wrong yardstick for 2016-18: many
results filings of those years were never posted as announcements; checked on GODREJPROP / INGERRAND / UMESLTD 2017.)
Sample: 8 random trading days per year, 2016 to the latest year, fixed seed. Pacing 2.5 s; hard stop after 15 minutes
or when more than 30% of requests fail after the first 8 (lesson INC-2026-10-02-background-job-unwatched).
Output: logs/news/announcements_completeness.csv (+ manifest); share = ours / NSE, by year.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.nse.api import get_json  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
A = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["an_dt"])
ours = pd.to_datetime(A["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce").dt.normalize().value_counts()
cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["trade_date"])["trade_date"].unique())))
days = pd.Series(cal[cal >= "2016-01-01"]).groupby(lambda i: cal[cal >= "2016-01-01"][i].year).apply(lambda s: s.sample(min(8, len(s)), random_state=11)).tolist()
s = build_session(warm=True, referer=REF)
rows, t0 = [], time.time()
for d in days:
    bad = sum(r["nse"] is None for r in rows)
    if (len(rows) >= 8 and bad / len(rows) > 0.3) or time.time() - t0 > 15 * 60:
        print(f"STOPPED EARLY: {bad}/{len(rows)} failed, {(time.time() - t0) / 60:.0f} min", flush=True); break
    f = d.strftime("%d-%m-%Y")
    try:
        j = get_json(s, f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={f}&to_date={f}", referer=REF)
        n = len(j) if isinstance(j, list) else len(j.get("data", []))
    except Exception:
        n = None
    rows.append(dict(day=str(d.date()), year=d.year, nse=n, ours=int(ours.get(d, 0))))
    time.sleep(2.5)
D = pd.DataFrame(rows)
out = ROOT / "logs/news/announcements_completeness.csv"; out.parent.mkdir(parents=True, exist_ok=True); D.to_csv(out, index=False)
ok = D.dropna(subset=["nse"]); ok = ok[ok["nse"] > 0]
by = ok.groupby("year").apply(lambda x: x["ours"].sum() / x["nse"].sum()).round(3)
out.with_suffix(".csv.manifest.json").write_text(json.dumps(dict(dataset=out.name, producer="src/agentic/check_announcements_completeness.py",
    definitions=__doc__, share_by_year=by.to_dict(), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
print(f"days checked {len(D)} · failed {int(D['nse'].isna().sum())} · minutes {(time.time() - t0) / 60:.1f}")
print("ours / NSE by year:", by.to_dict())
