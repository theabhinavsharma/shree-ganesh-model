"""Promoter pledge / encumbrance events from NSE (SAST Regulation 31(1) & 31(2) disclosures), 2016 onward (2026-10-04).

Why: T2 of the leverage framework (pledges as a veto on V3 picks): a newly pledged promoter stake is a forced-selling risk,
a release removes it. NSE's "Pledged Data" endpoint (api/corporate-pledgedata) returns 0 rows even on NSE's own page
for a full year, so the source is the Regulation 31 page's own call, api/corporate-pledgedata-sast3132 (checked
2026-10-04 in the browser): one row per promoter event, with NSE's broadcast time.
Fetch: one call per calendar month (no page cap: a 6-month call returned exactly the sum of its months, 461 = 461),
checkpointed per month in data/raw/pledge_events/<YYYY-MM>.json; the current and previous month are re-fetched every run.
Polite (2.5 s between calls), TLS on, stops after SGM_BUDGET_MIN minutes (default 20) or when more than half of the
first 5+ calls fail. A month that comes back empty is retried next run and listed, never assumed to be zero.
Output: data/derived/pledge_events.parquet (+ manifest), one row per NSE seqId:
  symbol (as filed), symbol_now (renames from NSE symbolchange.csv), company, promoter, event (Creation / Release /
  Invocation / ...), encumbrance (Pledge / Non disposal undertaking / Others ...), event_from, event_to, shares,
  pct_of_capital (the event's shares as % of total share capital, as NSE states it), pre_shares, pre_pct, post_shares,
  post_pct (that promoter's encumbered holding before / after), reported_on, broadcast_dt (public), attachment, source_url.
Usage: fetch_pledge_events.py [--start 2016-01] [--consolidate-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.nse.api import get_json  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

REF = "https://www.nseindia.com/companies-listing/corporate-filings-regulation-31"
API = "https://www.nseindia.com/api/corporate-pledgedata-sast3132?index=equities&from_date={a}&to_date={b}"
RAW = ROOT / "data/raw/pledge_events"
OUT = ROOT / "data/derived/pledge_events.parquet"


def months(start: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp.now(), freq="MS"))


def fetch(start: str) -> list[str]:
    RAW.mkdir(parents=True, exist_ok=True)
    budget, t0 = float(os.environ.get("SGM_BUDGET_MIN", "20")) * 60, time.time()
    ms = months(start)
    recent = {m.strftime("%Y-%m") for m in ms[-2:]}
    todo = [m for m in ms if not (RAW / f"{m:%Y-%m}.json").exists() or m.strftime("%Y-%m") in recent]
    s, fails, empty = build_session(warm=True, referer=REF), 0, []
    print(f"months to fetch: {len(todo)} of {len(ms)}", flush=True)
    for i, m in enumerate(todo):
        if time.time() - t0 > budget or (i >= 5 and fails / i > 0.5):
            print(f"STOPPED EARLY after {i} of {len(todo)} ({fails} failures); the rest next run", flush=True); break
        a, b = m.strftime("%d-%m-%Y"), (m + pd.offsets.MonthEnd(0)).strftime("%d-%m-%Y")
        try:
            j = get_json(s, API.format(a=a, b=b), referer=REF)
            d = j.get("data", []) if isinstance(j, dict) else (j or [])
        except Exception as x:   # noqa: BLE001
            fails += 1; print(f"  {m:%Y-%m}: {type(x).__name__}", flush=True); s = build_session(warm=True, referer=REF); time.sleep(5); continue
        if not d and m.strftime("%Y-%m") not in recent:
            empty.append(m.strftime("%Y-%m")); print(f"  {m:%Y-%m}: 0 rows — not saved, retried next run", flush=True)
        else:
            (RAW / f"{m:%Y-%m}.json").write_text(json.dumps(dict(url=API.format(a=a, b=b), fetched=datetime.now().isoformat(timespec="seconds"), data=d)))
            if i % 12 == 0:
                print(f"  {m:%Y-%m}: {len(d)} events", flush=True)
        time.sleep(2.5)
    return empty


def consolidate() -> pd.DataFrame:
    rows = []
    for p in sorted(RAW.glob("*.json")):
        j = json.loads(p.read_text())
        for r in j["data"]:
            rows.append(dict(r, source_url=j["url"]))
    D = pd.DataFrame(rows)
    num = lambda c: pd.to_numeric(D[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")  # noqa: E731
    dt = lambda c, f: pd.to_datetime(D[c].astype(str).str.strip(), format=f, errors="coerce")  # noqa: E731
    O = pd.DataFrame(dict(seq_id=D["seqId"].astype(str), symbol=D["symbol"], company=D["companyName"], promoter=D["promoterName"],
                          event=D["eventDetailsType"].astype(str).str.strip(), encumbrance=D["eventDetailsTypeEncumb"].astype(str).str.strip(), entity=D["eventDetailsEntity"],
                          event_from=dt("eventDetailsFromDate", "%d-%b-%Y"), event_to=dt("eventDetailsToDate", "%d-%b-%Y"),
                          shares=num("eventDetailsHolding"), pct_of_capital=num("eventDetailsPerc"),
                          pre_shares=num("preeventHolding"), pre_pct=num("preeventHoldingPerc"),
                          post_shares=num("postEventHolding"), post_pct=num("postEventHoldingPerc"),
                          reported_on=dt("reportingDate", "%d-%b-%Y"), broadcast_dt=dt("broadcastdate", "%d-%b-%Y %H:%M:%S"),
                          attachment=D["attachment"], source_url=D["source_url"]))
    O = O.sort_values("broadcast_dt").drop_duplicates("seq_id", keep="last")
    sys.path.insert(0, str(ROOT / "src/agentic"))
    import nse_symbols
    O["symbol_now"] = O["symbol"].map(nse_symbols.now)                  # renames: NSE symbolchange.csv chains
    O.to_parquet(OUT, index=False)
    by_m = O.groupby(O["broadcast_dt"].dt.to_period("M")).size()
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset=OUT.name, producer="src/agentic/fetch_pledge_events.py", rows=len(O), definitions=__doc__,
        source="NSE SAST Regulation 31(1) & 31(2) disclosures (api/corporate-pledgedata-sast3132), as broadcast",
        units=dict(shares="number of shares", pct_of_capital="percent of total share capital (as NSE states it)",
                   pre_pct="percent", post_pct="percent", broadcast_dt="IST, when NSE made it public"),
        coverage=dict(first=str(O["broadcast_dt"].min()), last=str(O["broadcast_dt"].max()), companies=int(O["symbol_now"].nunique()),
                      events_by_type=O["event"].value_counts().to_dict(), months_with_rows=int((by_m > 0).sum())),
        known_gaps="months that returned 0 rows are not saved and are retried; NSE's api/corporate-pledgedata (Pledged Data page) is empty and not used",
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(f"pledge events: {len(O):,} rows · {O['symbol_now'].nunique()} companies · {O['broadcast_dt'].min()} .. {O['broadcast_dt'].max()} · "
          f"{O['event'].value_counts().head(4).to_dict()}", flush=True)
    return O


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2016-01-01"); ap.add_argument("--consolidate-only", action="store_true")
    a = ap.parse_args()
    if not a.consolidate_only:
        e = fetch(a.start)
        if e:
            print(f"empty months (retried next run): {e}", flush=True)
    consolidate()
