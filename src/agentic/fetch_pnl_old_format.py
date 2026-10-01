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


def symbol_at():
    """symbol_at(sym, date): the symbol a company traded under on that date (NSE symbolchange.csv chain, walked back).
    NSE's archive links use today's symbol, but the archived page is filed under the symbol of its day: without this
    every renamed company (TATAMOTORS -> TMPV, SRTRANSFIN -> SHRIRAMFIN, ...) 404s."""
    ch: dict = {}
    for line in open(ROOT / "data/raw/nse_symbol_change/symbolchange.csv", encoding="utf-8", errors="replace"):
        parts = line.strip().rsplit(",", 3)
        if len(parts) == 4:
            d = pd.to_datetime(parts[3], format="%d-%b-%Y", errors="coerce")
            if pd.notna(d):
                ch.setdefault(parts[2].strip(), []).append((parts[1].strip(), d))

    def at(sym: str, when: pd.Timestamp) -> str:
        s, seen = sym, set()
        while s in ch and s not in seen:
            seen.add(s)
            later = [(o, d) for o, d in ch[s] if d > when]
            if not later:
                break
            s = max(later, key=lambda x: x[1])[0]
        return s
    return at


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


def _lines(pattern: str):
    for p in sorted(OUTDIR.glob(pattern)):
        for line in open(p):
            try:
                yield json.loads(line)
            except ValueError:      # a line cut short when a run was stopped mid-write: skipped, refetched next run
                continue


def done_keys() -> set:
    return {r["_key"] for r in _lines("old_html_w*.jsonl")}


def fetch(limit: int | None, workers: int) -> None:
    c = calendar()
    todo = c[~c["key"].isin(done_keys())]
    if limit:
        todo = todo.sample(min(limit, len(todo)), random_state=1)   # smoke test: spread across 2014-2017
    print(f"old-format quarterly filings {len(c):,} · to fetch {len(todo):,} · workers {workers}", flush=True)
    chunks = [todo.iloc[i::workers] for i in range(workers)]
    lock, n, failed = threading.Lock(), [0], []
    at = symbol_at()

    def work(k: int, part: pd.DataFrame) -> None:
        s = build_session(warm=True)
        with open(OUTDIR / f"old_html_w{k}.jsonl", "a") as fh, open(OUTDIR / "old_html_errors.jsonl", "a") as eh:
            for r in part.itertuples():
                rec = dict(_key=r.key, symbol=r.symbol, qe=str(r.qe.date()), is_con=int(r.is_con), fd=str(r.fd),
                           bank=r.bank, url=r.resultDetailedDataLink)
                old = at(r.symbol, r.fd)
                url = r.resultDetailedDataLink.replace(f"financial_res_{r.symbol}_", f"financial_res_{old}_") if old != r.symbol else r.resultDetailedDataLink
                rec.update(url=url, symbol_then=old)
                try:
                    rec.update(parse(get_text(s, url)), status="OK")
                except Exception as x:  # noqa: BLE001 — logged with its message, retried once at the end and on the next run
                    rec.update(status=f"ERR_{type(x).__name__}", error=str(x)[:160])
                if rec["status"] == "OK":
                    fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
                else:
                    eh.write(json.dumps(dict(_key=rec["_key"], status=rec["status"], error=rec["error"], ts=datetime.now().isoformat(timespec="seconds"))) + "\n"); eh.flush()
                    with lock:
                        failed.append(r.Index)
                with lock:
                    n[0] += 1
                    if n[0] % 500 == 0:
                        print(f"  {n[0]:,}/{len(todo):,}", flush=True)
                time.sleep(SLEEP)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda a: work(*a), enumerate(chunks)))
    if failed:                                   # one slower retry pass for the misses (throttling is usually brief)
        print(f"retrying {len(failed):,} misses with 2 workers after 60s", flush=True); time.sleep(60)
        retry = todo.loc[failed]; failed.clear(); n[0] = 0
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda a: work(*a), enumerate([retry.iloc[i::2] for i in range(2)])))
        print(f"still missing after retry: {len(failed):,}", flush=True)


def normalize() -> pd.DataFrame:
    recs = list(_lines("old_html_w*.jsonl"))
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
    """Enriched P&L = live table + old-format rows. Rules (2026-10-01, after the overlap QC):
    - keep only old rows whose unit was read and whose EPS matches profit / shares (eps_check_ok);
    - drop isolated sales spikes (>20x or <0.05x vs BOTH neighbouring quarters: a unit slip on one page);
    - where both sources have the same (symbol, quarter, basis), the NEW row is a restatement filed ~13 months later
      (Ind AS transition; median 395 days, every overlap filed after the old row), so the FIRST-FILED old row is kept and
      the restated row is dropped: what investors saw at the time is the point of the backfill. Known limit: for 2018
      dates, those quarters serve as year-ago bases on the old accounting standard."""
    O = pd.read_parquet(OUT)
    O = O[O["eps_check_ok"] & O["unit"].str.contains("lakh|lac|crore|million|thousand", case=False, na=False)]
    O = O.sort_values(["symbol", "basis", "quarter_end"]).reset_index(drop=True)
    g = O.groupby(["symbol", "basis"])["net_sales"]
    ext = lambda r: (r > 20) | (r < 0.05)  # noqa: E731
    spike = ext(O["net_sales"] / g.shift(1)) & ext(O["net_sales"] / g.shift(-1))
    O = O[~spike]
    P = pd.read_parquet(PNL); P = P[P["source"] != "nse_old_html"].copy()
    P["quarter_end"] = pd.to_datetime(P["quarter_end"]); P["filing_dt"] = pd.to_datetime(P["filing_dt"])
    j = P.reset_index().merge(O[["symbol", "quarter_end", "basis", "filing_dt"]], on=["symbol", "quarter_end", "basis"], suffixes=("", "_old"))
    restated = j.loc[j["filing_dt_old"] < j["filing_dt"], "index"]
    newer_live = set(zip(*[j.loc[j["filing_dt_old"] >= j["filing_dt"], c] for c in ("symbol", "quarter_end", "basis")]))
    P = P.drop(index=restated)
    add = O[[k not in newer_live for k in zip(O["symbol"], O["quarter_end"], O["basis"])]][list(P.columns)]
    M = pd.concat([P, add], ignore_index=True).sort_values(["symbol", "quarter_end", "basis"])
    M.to_parquet(ENRICHED, index=False)
    ENRICHED.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="pnl_quarterly_enriched.parquet", producer="src/agentic/fetch_pnl_old_format.py --merge",
        definition=merge.__doc__, units="as pnl_quarterly: detail_api and nse_old_html in Rs LAKH, xbrl in Rs; EPS Rs per share",
        rows=len(M), added_old=len(add), restated_rows_replaced=int(len(restated)), sales_spikes_dropped=int(spike.sum()),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"enriched P&L: {len(M):,} rows = live {len(P) + len(restated):,} - {len(restated):,} restated (first-filed kept) + "
          f"{len(add):,} old-format · {int(spike.sum())} sales spikes dropped · live file untouched")


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
