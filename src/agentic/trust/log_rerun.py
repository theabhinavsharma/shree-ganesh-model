"""Log a registered test's re-run after a data change (2026-10-04): data gate first, then a RERUN-<tag> line built from the
reproduction output that src/agentic/trust/reproduce.py just wrote (same code, same rules, no re-tuning).

Why: the weekly gate trust.results_reproduce compares each test with its newest RESULT / RERUN line and blocks the
Friday run when it no longer reproduces. After a real data fix the numbers move; the honest record is a RERUN line
with the new numbers and a READY data-gate line, not a looser tolerance.
Gate (industry-score tests): live P&L company-quarters with sales vs NSE's own lists (quarterly results calendar +
integrated filings) for every year the tests read (2018 onward: year-ago quarters for 2019), price panel span.
Usage: log_rerun.py --tag datafill-20261004 --note "..." EXP-ID [EXP-ID ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr  # noqa: E402
import reproduce  # noqa: E402

REFS = {"S1M", "S0", "S1", "BASE", "REF"}


def checks(end_year: int) -> list[dict]:
    P = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet", columns=["symbol", "quarter_end", "net_sales"])
    P["quarter_end"] = pd.to_datetime(P["quarter_end"])
    have = P.dropna(subset=["net_sales"]).drop_duplicates(["symbol", "quarter_end"]).groupby(P["quarter_end"].dt.year).size()
    C = pd.read_parquet(ROOT / "data/derived/results_calendar_full.parquet", columns=["symbol", "toDate", "period"])
    C = C[C["period"].astype(str).str.lower() == "quarterly"]; C["qe"] = pd.to_datetime(C["toDate"], format="%d-%b-%Y", errors="coerce")
    I = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet", columns=["symbol", "period_to"])
    I["qe"] = pd.to_datetime(I["period_to"], format="%d-%b-%Y", errors="coerce")
    off = pd.concat([C[["symbol", "qe"]], I[["symbol", "qe"]]]).dropna().drop_duplicates()
    official = off.groupby(off["qe"].dt.year).size()
    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["trade_date"])["trade_date"]
    return [dr.completeness("live P&L company-quarters vs NSE's results calendar + integrated filings", have, official, range(2018, end_year + 1)),
            dr.span("price panel", px, "2016-01-01", pd.Timestamp.now().normalize() - pd.Timedelta(days=4))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+"); ap.add_argument("--tag", required=True); ap.add_argument("--note", required=True)
    a = ap.parse_args()
    ck = checks(datetime.now().year)
    for exp in a.exps:
        dr.gate(exp, ck, run=a.tag)                        # NOT READY -> exit 3 before any line is written
        csv = ROOT / reproduce.TESTS[exp][1]
        R = pd.read_csv(csv).set_index("arm")
        arms = {k: [round(float(r[c]), 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [r.get("phases_beaten") if "phases_beaten" in R else None]
                for k, r in R.iterrows()}
        verdict = [k for k, r in R.iterrows() if "PASS" in R and str(r["PASS"]) == "True" and k not in REFS] or "no arm passes"
        line = dict(ts=datetime.now().isoformat(timespec="seconds"), id=f"{exp}-RERUN-{a.tag}", note=a.note, verdict=verdict, arms=arms,
                    source=str(csv.relative_to(ROOT)), cols="CAGR, disc, conf, maxDD, phases")
        with (ROOT / "logs/experiments.jsonl").open("a") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
        print(f"{exp}-RERUN-{a.tag}: verdict {verdict}")


if __name__ == "__main__":
    main()
