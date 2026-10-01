"""Quarterly P&L for NSE's OLD-format results filings, 2014-2017 (2026-09-30).

Why: NSE's results JSON (fetch_pnl_history.py) only serves NEW-format filings, which start mid-2016; most 2016-17 results
were filed in the old format, so P&L features had ~0% coverage before 2018 and the model could not use them
(EXP-2026-09-30-v3-pnl). Every old-format filing in NSE's results calendar has an official archived page,
resultDetailedDataLink = https://nsearchives.nseindia.com/archives/financial_results/financial_res_<SYM>_<seq>.html,
holding that filing's own figures as first filed (vintage). Third-party sites (screener, trendlyne) were rejected:
restated numbers, no filing dates, logins, or BSE PDFs.
Rows: calendar format == Old, Quarterly, Non-Cumulative, quarter end 2014-04-01 .. 2018-03-31; latest filing per
(symbol, quarter, consolidated) like fetch_pnl_history; filing_dt = calendar filingDate (same as the NEW rows).
Units: each page states its own unit ("Amount(Rs. in lakhs)", crores, ...); values are converted to Rs LAKH to match
pnl_quarterly's detail_api rows; a page whose unit cannot be read is kept raw and NOT merged.
Stages: fetch (resumable, polite, N workers, NSE session, TLS on) -> data/derived/pnl_history/old_html*.jsonl
        normalize -> data/derived/pnl_quarterly_old.parquet (+ manifest)
        merge     -> data/derived/pnl_quarterly_enriched.parquet = pnl_quarterly.parquet + source == "nse_old_html" rows
                     where no NEW/XBRL row exists for the same (symbol, quarter_end, basis). The LIVE pnl_quarterly.parquet
                     is never touched: if the enriched P&L cleared the model's coverage rule it would silently enter the
                     live model; it goes live only after a registered test passes and Abhinav says go.
Usage: fetch_pnl_old_format.py [--limit N] [--workers 4] [--normalize-only] [--merge]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.nse.api import get_text  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

CAL = ROOT / "data/derived/results_calendar_full.parquet"
OUTDIR = ROOT / "data/derived/pnl_history"
OUT = ROOT / "data/derived/pnl_quarterly_old.parquet"
PNL = ROOT / "data/derived/pnl_quarterly.parquet"
ENRICHED = ROOT / "data/derived/pnl_quarterly_enriched.parquet"
SLEEP = 0.3
TO_LAKH = [(r"lakh|lac", 1.0), (r"crore|cr\b", 100.0), (r"million|mn\b", 10.0), (r"thousand|'000", 0.01)]
LABELS = {  # first matching label wins; value = the number right after it
    "net_sales": [r"net sales\s*/\s*income from operation", r"^\(?a\)?\s*net sales", r"income from operations",
                  r"interest earned"],
    "total_income": [r"^total income$", r"^total income\b(?!.*from operations)"],
    "pbt": [r"profit\s*/\s*\(loss\) from ordinary activities before tax", r"profit.*before tax"],
    "pat_con": [r"consolidated net profit"],
    "pat": [r"net profit\s*/\s*\(loss\) for the period", r"net profit.*for the period", r"net profit\s*/\s*\(loss\)"],
    "eps_basic": [r"basic eps"],
    "eps_diluted": [r"diluted eps"],
    "face_value": [r"face value"],
    "paidup": [r"paid-up equity share capital"],
}


def num(s: str):
    s = s.replace(",", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def parse(page: str) -> dict:
    cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", c))).strip()
             for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", page, re.S | re.I)]
    cells = [c for c in cells if c]
    head = {cells[i]: cells[i + 1] for i in range(len(cells) - 1) if cells[i] in
            ("Symbol", "Consolidated / Non-Consolidated", "Cumulative / Non-Cumulative", "Period", "Period Ended")}
    unit_txt = next((c for c in cells if c.lower().startswith("amount")), "")
    mult = next((m for pat, m in TO_LAKH if re.search(pat, unit_txt, re.I)), None)
    out = dict(head=head, unit=unit_txt, to_lakh=mult)
    pairs = [(cells[i], num(cells[i + 1])) for i in range(len(cells) - 1) if num(cells[i + 1]) is not None or cells[i + 1] == "-"]
    for k, pats in LABELS.items():
        out[k] = next((v for p in pats for lab, v in pairs if re.search(p, lab, re.I)), None)
    # template 2: "Earnings per share (before extraordinary items) ..." heading, then "(a) Basic" / "(b) Diluted" rows
    if out["eps_basic"] is None or out["eps_diluted"] is None:
        ctx = False
        for i in range(len(cells) - 1):
            if re.search(r"earnings per share", cells[i], re.I):
                ctx = ctx or "before" in cells[i].lower() or "after" not in cells[i].lower()
                continue
            if ctx and num(cells[i + 1]) is not None:
                if out["eps_basic"] is None and re.search(r"^\(?a\)?\s*basic", cells[i], re.I):
                    out["eps_basic"] = num(cells[i + 1])
                elif out["eps_diluted"] is None and re.search(r"^\(?b\)?\s*diluted", cells[i], re.I):
                    out["eps_diluted"] = num(cells[i + 1])
    return out


def calendar() -> pd.DataFrame:
    c = pd.read_parquet(CAL)
    c = c[(c["format"] == "Old") & (c["period"].astype(str).str.lower() == "quarterly")
          & (c["cumulative"].astype(str).str.lower().str.startswith("non"))].copy()
    c["qe"] = pd.to_datetime(c["toDate"], format="%d-%b-%Y", errors="coerce")
    c["fd"] = pd.to_datetime(c["filingDate"], format="%d-%b-%Y %H:%M", errors="coerce")
    c = c[(c["qe"] >= "2014-04-01") & (c["qe"] <= "2018-03-31") & c["resultDetailedDataLink"].notna()]
    c["is_con"] = (c["consolidated"] == "Consolidated").astype(int)
    c = c.sort_values("fd").drop_duplicates(["symbol", "qe", "is_con"], keep="last")
    c["key"] = c["symbol"] + "|" + c["qe"].dt.strftime("%Y-%m-%d") + "|" + c["is_con"].astype(str)
    return c


def done_keys() -> set:
    out = set()
    for p in OUTDIR.glob("old_html*.jsonl"):
        for line in open(p):
            if line.strip():
                out.add(json.loads(line)["_key"])
    return out


def fetch(limit: int | None, workers: int) -> None:
    c = calendar()
    todo = c[~c["key"].isin(done_keys())]
    if limit:
        todo = todo.sample(min(limit, len(todo)), random_state=1)   # smoke test: spread across 2014-2017
    print(f"old-format quarterly filings {len(c):,} · to fetch {len(todo):,} · workers {workers}", flush=True)
    chunks = [todo.iloc[i::workers] for i in range(workers)]
    lock, n = threading.Lock(), [0]

    def work(k: int, part: pd.DataFrame) -> None:
        s = build_session(warm=True)
        with open(OUTDIR / f"old_html_w{k}.jsonl", "a") as fh:
            for r in part.itertuples():
                rec = dict(_key=r.key, symbol=r.symbol, qe=str(r.qe.date()), is_con=int(r.is_con), fd=str(r.fd),
                           bank=r.bank, url=r.resultDetailedDataLink)
                try:
                    rec.update(parse(get_text(s, r.resultDetailedDataLink)), status="OK")
                except Exception as x:  # noqa: BLE001 — recorded, retried next run (not marked done below)
                    rec.update(status=f"ERR_{type(x).__name__}")
                if rec["status"] == "OK":
                    fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
                with lock:
                    n[0] += 1
                    if n[0] % 500 == 0:
                        print(f"  {n[0]:,}/{len(todo):,}", flush=True)
                time.sleep(SLEEP)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda a: work(*a), enumerate(chunks)))


def normalize() -> pd.DataFrame:
    recs = [json.loads(l) for p in sorted(OUTDIR.glob("old_html*.jsonl")) for l in open(p) if l.strip()]
    D = pd.DataFrame(recs).drop_duplicates("_key", keep="last")
    m = D["to_lakh"]
    pat = D["pat_con"].where(D["is_con"] == 1).fillna(D["pat"]) if "pat_con" in D else D["pat"]
    O = pd.DataFrame(dict(symbol=D["symbol"], quarter_end=pd.to_datetime(D["qe"]), filing_dt=pd.to_datetime(D["fd"]),
                          basis=D["is_con"].map({1: "con", 0: "sa"}), bank=D["bank"], source="nse_old_html",
                          eps_basic=D["eps_basic"], eps_diluted=D["eps_diluted"], net_sales=D["net_sales"] * m,
                          total_income=D["total_income"] * m, pbt=D["pbt"] * m, pat=pat * m, face_value=D["face_value"],
                          paidup_lakh=D["paidup"] * m, unit=D["unit"], source_url=D["url"]))
    # QC: EPS vs profit / shares (shares = paid-up capital / face value); a large miss flags a unit or label problem
    sh = O["paidup_lakh"] * 1e5 / O["face_value"]
    implied = O["pat"] * 1e5 / sh
    O["eps_from"] = "reported"
    fill = O["eps_basic"].isna() & implied.notna()
    O.loc[fill, "eps_basic"] = implied[fill]; O.loc[fill, "eps_from"] = "implied: pat / (paid-up / face value)"
    O.loc[O["eps_basic"].isna(), "eps_from"] = None
    O["eps_check_ok"] = ((implied - O["eps_basic"]).abs() <= (0.15 * O["eps_basic"].abs()).clip(lower=0.05)) & O["eps_basic"].notna()
    O.to_parquet(OUT, index=False)
    good = O["eps_check_ok"] & m.notna().values
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="pnl_quarterly_old.parquet", producer="src/agentic/fetch_pnl_old_format.py", rows=len(O),
        source="NSE archived results pages (resultDetailedDataLink), old-format filings, as first filed",
        units=dict(net_sales="Rs LAKH (page unit converted)", total_income="Rs LAKH", pbt="Rs LAKH", pat="Rs LAKH",
                   eps_basic="Rs per share", face_value="Rs", paidup_lakh="Rs LAKH"),
        qc=dict(unit_read=float(m.notna().mean()), eps_consistent=float(O["eps_check_ok"].mean()), mergeable=float(good.mean())),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"normalized {len(O):,} rows · unit read {m.notna().mean():.1%} · EPS consistent {O['eps_check_ok'].mean():.1%} · "
          f"quarters {O['quarter_end'].min().date()}..{O['quarter_end'].max().date()}")
    return O


def merge() -> None:
    O = pd.read_parquet(OUT)
    O = O[O["eps_check_ok"] & O["unit"].str.contains("lakh|lac|crore|million|thousand", case=False, na=False)]
    P = pd.read_parquet(PNL)
    P = P[P["source"] != "nse_old_html"]
    have = set(zip(P["symbol"], pd.to_datetime(P["quarter_end"]), P["basis"]))
    add = O[[k not in have for k in zip(O["symbol"], O["quarter_end"], O["basis"])]][list(P.columns)]
    M = pd.concat([P, add], ignore_index=True).sort_values(["symbol", "quarter_end", "basis"])
    M.to_parquet(ENRICHED, index=False)
    ENRICHED.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="pnl_quarterly_enriched.parquet", producer="src/agentic/fetch_pnl_old_format.py --merge",
        definition="pnl_quarterly.parquet + nse_old_html rows (EPS-consistent, unit read) where no NEW/XBRL row exists",
        units="as pnl_quarterly: detail_api and nse_old_html in Rs LAKH, xbrl in Rs; EPS Rs per share",
        rows=len(M), added=len(add), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"enriched P&L: {len(P):,} live rows + {len(add):,} old-format rows = {len(M):,} -> {ENRICHED.name} (live file untouched)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--normalize-only", action="store_true"); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        if not a.normalize_only:
            fetch(a.limit, a.workers)
        normalize()
