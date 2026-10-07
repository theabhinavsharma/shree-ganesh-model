"""Ran -> fetched -> used, for one data day (2026-10-04; Abhinav: "daily uss databases ko check karo ... qc run kar ki aaj ka
chala > aaj fetch hua > wo use hua").

For each input the strategies read, on day D (default: the date of the latest daily data run):
  ran      the step is in today's run status (logs/daily_data_layer_status.json) as passed
  fetched  rows we hold for D vs NSE's own whole-day count where NSE gives one (corporate-announcements, from=to=D)
  used     the rows of D reached the table built from them: order filings of D -> order_amounts; filings of D -> the
           bad-news ledger; companies whose results XBRL was broadcast on D -> pnl_quarterly
Output: logs/evals/input_chain_<D>.json (read by the eval data.input_chain and by the daily message).
Usage: /usr/bin/python3 src/agentic/trust/input_chain.py [YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
OUT = ROOT / "logs/evals"
MIN_SHARE = 0.95


def _nse_day_count(d: pd.Timestamp) -> int | None:
    try:
        from src.ingest.nse.api import get_json
        from src.ingest.nse.session import build_session
        ref = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
        f = d.strftime("%d-%m-%Y")
        j = get_json(build_session(warm=True, referer=ref), f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={f}&to_date={f}", referer=ref)
        return len(j) if isinstance(j, list) else len(j.get("data", []))
    except Exception:
        return None


def run(day: str | None = None) -> dict:
    st = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())
    # default day = the last IST day that has fully ended (NSE keeps filing until midnight IST; a 6:45 PM IST run would
    # otherwise see a partial day). Same day as before for the old US-Eastern evening run (4:15 AM IST next day).
    D = pd.Timestamp(day).normalize() if day else (pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize() - pd.Timedelta(days=1))
    passed = set(st.get("passed", []))
    rows = []

    def add(name, step, fetched=None, official=None, used=None, of=None, note=""):
        ok_f = official is None or (fetched is not None and official > 0 and fetched / official >= MIN_SHARE) or official == 0
        ok_u = of is None or of == 0 or (used is not None and used / of >= MIN_SHARE)
        rows.append(dict(input=name, step=step, ran=step in passed, fetched=fetched, official=official, used=used, of=of,
                         ok=bool(step in passed and ok_f and ok_u), note=note))

    nse = _nse_day_count(D)
    A = pd.read_parquet(ROOT / "data/events_full_history/normalized/stock_announcements.parquet", columns=["event_date"])
    add("filings (daily feed)", "announcements", fetched=int((pd.to_datetime(A["event_date"]).dt.normalize() == D).sum()), official=nse,
        note="" if nse is not None else "NSE count unavailable")
    H = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "seq_id", "an_dt"])
    H["day"] = pd.to_datetime(H["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce").dt.normalize()
    hd = H[H["day"] == D]
    add("filings archive", "announcements_archive", fetched=len(hd), official=nse)

    import fetch_order_fulltext as fof
    T = fof.targets(); T["day"] = pd.to_datetime(T["sort_date"], errors="coerce").dt.normalize()
    td = T[T["day"] == D]
    O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet", columns=["symbol", "seq_id"])
    have = set(zip(O["symbol"], O["seq_id"].astype(str)))
    add("order amounts", "order_amounts", used=int(sum((s, str(q)) in have for s, q in zip(td["symbol"], td["seq_id"]))), of=len(td),
        note="order filings of the day read from their PDFs")

    L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["filed_at"])
    nl = int((pd.to_datetime(L["filed_at"]).dt.normalize() == D).sum())
    add("bad-news ledger", "event_ledger", used=nl if (nl or not len(hd)) else 0, of=None,
        note=f"{nl} events from {len(hd)} filings" + ("" if nl or not len(hd) else " · NONE from a day with filings"))
    if len(hd) and not nl:
        rows[-1]["ok"] = False

    C = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet", columns=["symbol", "broadcast", "detail_link"])
    import fetch_pnl_history as fph
    C = C[C["detail_link"].str.contains(fph.FIN_FILE, na=False)]
    cd = set(C.loc[pd.to_datetime(C["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce").dt.normalize() == D, "symbol"])
    P = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet", columns=["symbol", "filing_dt"])
    pdone = set(P.loc[pd.to_datetime(P["filing_dt"], errors="coerce").dt.normalize() == D, "symbol"])
    sp = ROOT / "logs/pnl_update/state.json"
    pend = len(json.loads(sp.read_text()).get("pending", [])) if sp.exists() else None
    add("P&L", "pnl_update", fetched=len(cd), used=len(cd & pdone), of=len(cd),
        note=f"companies with results on the day -> in P&L" + (f" · {pend} companies left for next run" if pend else ""))

    Pr = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["trade_date"])
    last = pd.to_datetime(Pr["trade_date"]).max()
    add("prices", "prices", note=f"newest session {last.date()}")
    if "prices" in passed and last.normalize() < D and pd.Timestamp(D).weekday() < 5:
        import nse_calendar
        if nse_calendar.is_session(D):
            rows[-1]["ok"] = False; rows[-1]["note"] += " · the day's prices are missing"

    res = dict(day=str(D.date()), written=datetime.now().isoformat(timespec="seconds"), rows=rows, ok=all(r["ok"] for r in rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"input_chain_{D.date()}.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def line(res: dict) -> str:
    """One message line: ✅ when ran + fetched + used, else the numbers that are off."""
    bits = []
    for r in res["rows"]:
        if r["ok"]:
            bits.append(f"{r['input']} ✅")
            continue
        why = [] if r["ran"] else ["did not run"]
        if r["official"] is not None and r["fetched"] is not None and r["official"] and r["fetched"] / r["official"] < MIN_SHARE:
            why.append(f"fetched {r['fetched']}/{r['official']} of NSE's count")
        if r["of"]:
            why.append(f"used {r['used']}/{r['of']}")
        bits.append(f"{r['input']} ⚠️ " + (", ".join(why) or r["note"]))
    return f"🔗 {pd.Timestamp(res['day']):%b %d} ran→fetched→used: " + " · ".join(bits)


if __name__ == "__main__":
    r = run(sys.argv[1] if len(sys.argv) > 1 else None)
    for x in r["rows"]:
        print(x)
    print(line(r))
