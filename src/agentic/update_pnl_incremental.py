"""Keep quarterly P&L current, every day (2026-10-04; Abhinav: "wire karo before Oct 15 and dhyaan rakhna aage se").

Why: fetch_pnl_history.py crawls company by company and skips a company once it has been fetched, so results filed after
the Aug 29 crawl never arrived: pnl_quarterly.parquet sat 5 weeks stale with no schedule and no guard.
What, each run:
  1. companies to refresh = those with a results-related filing in NSE's daily announcements feed
     (stock_announcements.parquet, is_results_event) since the last run (3-day overlap), plus any left over last run
  2. per company: NSE integrated-filing-results (the endpoint and parser of fetch_event_history_fleet.py); filings not yet
     known are appended to data/derived/results_calendar_integrated.parquet (a copy of the previous file is kept as .prev)
  3. XBRL for filings not yet fetched (fetch_pnl_history.parse_xbrl, same record format and checkpoint integrated2.jsonl)
  4. fetch_pnl_history.stage_normalize -> pnl_quarterly.parquet; fetch_pnl_old_format.py --merge -> pnl_quarterly_enriched
Point in time: a quarter keeps the first filing fetched (as before); filing_dt is NSE's broadcast time.
Budget: stops after SGM_BUDGET_MIN minutes (default 40) or when more than half of the first 20+ requests fail; whatever
is left is listed in logs/pnl_update/state.json and done next run. Normalize and merge always run on what was fetched.
Exit 0 = done or nothing new; 2 = stopped early (partial, still written); 1 = error.
SGM_PNL_PART (catch-up in parallel, 2026-10-04): "lists" = steps 1-2 only; "xbrl:n/k" = step 3 for every k-th filing from
n, into its own checkpoint integrated2_inc_w<n>.jsonl (read by load_done / normalize like the others); "rebuild" = step 4.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
import fetch_pnl_history as fph  # noqa: E402
from fetch_event_history_fleet import parse_integrated  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

STATE = ROOT / "logs/pnl_update/state.json"
CAL = ROOT / "data/derived/results_calendar_integrated.parquet"
FEED = ROOT / "data/events_full_history/normalized/stock_announcements.parquet"
API = "https://www.nseindia.com/api/integrated-filing-results?index=equities&symbol={sym}&period_ended=Quarterly"
BUDGET = float(os.environ.get("SGM_BUDGET_MIN", "40")) * 60
KEY = ["symbol", "period_to", "broadcast", "consolidated", "detail_link"]


PART = os.environ.get("SGM_PNL_PART", "all")


def main() -> int:
    t0 = time.time()
    if PART == "rebuild":
        fph.stage_normalize()
        return subprocess.run([sys.executable, str(ROOT / "src/agentic/fetch_pnl_old_format.py"), "--merge"], cwd=ROOT).returncode
    STATE.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    cal = pd.read_parquet(CAL)
    bd = pd.to_datetime(cal["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    since = pd.Timestamp(st.get("feed_through") or bd.max()) - pd.Timedelta(days=3)
    F = pd.read_parquet(FEED, columns=["symbol", "event_date", "is_results_event"])
    F["event_date"] = pd.to_datetime(F["event_date"], errors="coerce")
    F = F[F["is_results_event"].astype(bool) & (F["event_date"] >= since)]
    syms = sorted(set(F["symbol"].dropna()) | set(st.get("pending", [])))
    feed_through = str(F["event_date"].max().date()) if len(F) else st.get("feed_through")
    print(f"results filings in the feed since {since.date()}: {len(F)} rows, {len(syms)} companies to refresh", flush=True)

    # 2. per-company filing list
    s, new, errs, pending, partial = build_session(), [], 0, [], False
    if PART.startswith("xbrl"):
        syms = []                                                # lists are done by the "lists" part
    for i, sym in enumerate(syms):
        if time.time() - t0 > BUDGET * 0.5 or (i >= 20 and errs / i > 0.5):   # half the budget for lists, half for XBRL
            pending = syms[i:]; partial = True
            print(f"STOPPED EARLY (lists) after {i} of {len(syms)} companies ({errs} errors)", flush=True)
            break
        j = fph.jget(s, API.format(sym=sym))
        if j == "ERR":
            errs += 1; s = build_session(); continue
        df = parse_integrated(sym, j)
        if df is not None and len(df):
            new.append(df)
        time.sleep(fph.SLEEP)
    added = 0
    if new:
        N = pd.concat(new, ignore_index=True)
        k_old = set(map(tuple, cal[KEY].astype(str).values))
        N = N[[tuple(r) not in k_old for r in N[KEY].astype(str).values]].drop_duplicates(KEY)
        added = len(N)
        if added:
            shutil.copy2(CAL, CAL.with_suffix(".parquet.prev"))
            pd.concat([cal, N[cal.columns.intersection(N.columns)]], ignore_index=True).to_parquet(CAL, index=False)
    print(f"filing list: {added} new filings appended to {CAL.name}", flush=True)
    if PART == "lists":
        st["pending"] = pending; st["feed_through"] = feed_through if not pending else st.get("feed_through")
        STATE.write_text(json.dumps(st, indent=1)); return 2 if partial else 0

    # 3. XBRL for filings not yet fetched (same selection, record format and checkpoint as fph.stage_integrated)
    ic = pd.read_parquet(CAL)
    ic["qe"] = pd.to_datetime(ic["period_to"], format="%d-%b-%Y", errors="coerce")
    ic["bd"] = pd.to_datetime(ic["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    ic = ic.dropna(subset=["qe", "detail_link"])
    ic = ic[ic["detail_link"].str.contains(fph.FIN_FILE, na=False)]
    ic["is_con"] = (ic["consolidated"] == "Consolidated").astype(int)
    ic = ic.sort_values("bd").drop_duplicates(["symbol", "qe", "is_con"], keep="last")
    ic["bd"] = fph._xbrl_fallback_fd(ic)
    done = fph.load_done("integrated2")
    todo = [r for _, r in ic.sort_values("bd", ascending=False).iterrows()    # newest filings first: the day's results come
            if f"{r['symbol']}|{r['qe'].date()}|{r['is_con']}" not in done and r["qe"] >= pd.Timestamp("2024-01-01")]   # before old gaps
    shard = "integrated2.jsonl"
    if PART.startswith("xbrl:"):
        n, k = map(int, PART.split(":")[1].split("/")); todo = todo[n::k]; shard = f"integrated2_inc_w{n}.jsonl"
    print(f"XBRL to fetch: {len(todo)} (quarters from 2024 not yet fetched)", flush=True)
    got, xerr = 0, 0
    with open(fph.OUTDIR / shard, "a") as fh:
        for i, r in enumerate(todo):
            if time.time() - t0 > BUDGET or (i >= 20 and xerr / i > 0.5):
                partial = True
                print(f"STOPPED EARLY (XBRL) after {i} of {len(todo)} ({xerr} errors)", flush=True)
                break
            key = f"{r['symbol']}|{r['qe'].date()}|{r['is_con']}"
            try:
                resp = s.get(r["detail_link"], timeout=30)
                d = fph.parse_xbrl(resp.text, r["qe"]) if resp.status_code == 200 else {}
            except Exception:
                xerr += 1; s = build_session(); continue
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                xerr += 1; continue                        # throttled / server error: not written, retried next run
            # 200, or a dead link (404 ...): written (a dead link with no figures) so it is not retried forever
            got += d.get("net_sales") is not None
            fh.write(json.dumps({"_key": key, "_sym": r["symbol"], "_qe": str(r["qe"].date()), "_con": int(r["is_con"]),
                                 "_fd": str(r["bd"]), "d": d}) + "\n"); fh.flush()
            time.sleep(fph.SLEEP)
    print(f"XBRL fetched with sales: {got}", flush=True)

    if PART.startswith("xbrl"):
        print(f"XBRL part {PART}: {got} with sales", flush=True); return 2 if partial else 0
    # 4. rebuild the tables from the checkpoints (always, so whatever arrived is used)
    fph.stage_normalize()
    rc = subprocess.run([sys.executable, str(ROOT / "src/agentic/fetch_pnl_old_format.py"), "--merge"], cwd=ROOT).returncode
    P = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet", columns=["filing_dt"])
    rec = dict(ts=datetime.now().isoformat(timespec="seconds"), companies=len(syms), new_filings=added, xbrl_todo=len(todo),
               xbrl_with_sales=got, pending=len(pending), partial=partial, merge_rc=rc, minutes=round((time.time() - t0) / 60, 1),
               newest_filing=str(pd.to_datetime(P["filing_dt"]).max()))
    STATE.write_text(json.dumps(dict(feed_through=feed_through if not pending else st.get("feed_through"),
                                     pending=pending, last=rec, runs=(st.get("runs", []) + [rec])[-60:]), indent=1))
    print(json.dumps(rec), flush=True)
    return 1 if rc else (2 if partial else 0)


if __name__ == "__main__":
    sys.exit(main())
