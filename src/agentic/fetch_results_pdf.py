"""Quarterly sales and profit from the company's own results PDF where NSE's structured results lists have a hole
(2026-10-04; Abhinav: "dono karo, full power").

Why: NSE's structured lists miss whole quarters for hundreds of companies (no Jun-2022 between Mar-22 and Sep-22; no
Mar-2025 between the old list ending Dec-24 and the integrated list starting Jun-25), so trailing revenue breaks. The
results themselves were filed as PDFs and are in the announcements archive.
Targets: a quarter missing (both bases) between two quarters the company does have, 2018-06-30 onward (holes), from
data/derived/pnl_quarterly.parquet.
Filing: announcements_historical rows of that company 1-90 days after the quarter end whose description or attachment
text mentions financial results / integrated filing (financials) / outcome of board meeting, with an attachment;
earliest first (as first filed). Text: fetch_filing_text._extract; OCR (ocr_order_filings.ocr_blob) when the PDF has
under 500 characters of text.
Parse: the text must name the quarter-end date (any common format). Sections by heading (standalone / consolidated);
in each, the first line labelled revenue from operations / income from operations / net sales / interest earned with a
figure, and the first profit-for-the-period line; the FIRST figure on the line = the current quarter (results tables
list the current quarter first). Unit from the page ("in lakhs", "crore", "million", "thousand"); none printed -> the
unit that puts sales nearest the company's neighbouring quarters, flagged unit_inferred.
Profit is kept only next to same-basis sales and when |profit| <= sales. QC (qc_ok): sales > 0, the date was found, and sales within 1/3x..3x of the median of the company's structured quarters
within +-1 year (same basis if present). Rows go to data/derived/pnl_quarterly_pdf.parquet (Rs LAKH, source
"nse_results_pdf", filing_dt = NSE dissemination time); fetch_pnl_history.stage_normalize adds qc_ok rows where no
structured row exists.
Usage: fetch_results_pdf.py [--workers 8] [--limit N] [--test SYMBOL:YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
import fetch_filing_text as ft  # noqa: E402
from src.ingest.nse.api import _request_headers, _request_with_retries  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
CKPT = ROOT / "data/derived/pnl_history/results_pdf.jsonl"
OUT = ROOT / "data/derived/pnl_quarterly_pdf.parquet"
FACTOR = {"lakh": 1.0, "crore": 100.0, "million": 10.0, "thousand": 0.01, "rupees": 1e-5}   # -> Rs lakh
UNIT_RE = [("crore", r"(?:cr(?:ore)?s?\b)"), ("lakh", r"(?:lakhs?|lacs?)\b"), ("million", r"(?:millions?|mn\b|mio\b)"), ("thousand", r"(?:thousands?|'000)")]
REV = re.compile(r"^\W*(?:\(?[ivxlc0-9a-z]{1,4}\)?[.\s]*)?(?:total\s+)?(?:revenue\s+from\s+operations?|income\s+from\s+operations?|net\s+sales|sales\s*/\s*income\s+from\s+operations|interest\s+earned)\b", re.I)
PAT = re.compile(r"^\W*(?:\(?[ivxlc0-9a-z]{1,4}\)?[.\s]*)?(?:net\s+)?profit\s*/?\s*(?:\(loss\)|\(loss\)\s*)?\s*(?:for\s+the\s+(?:period|quarter|year)|after\s+tax)", re.I)
NUM = re.compile(r"\(?-?\d[\d,]*\.\d+\)?|\(?-?\d{1,3}(?:,\d{2,3})+\)?|\(?-?\d{3,}\)?")
RESULTS = re.compile(r"financial result|integrated filing|outcome of board meeting|results? for the (?:quarter|period)", re.I)


def date_pats(qe: pd.Timestamp) -> re.Pattern:
    d, m, y = qe.day, qe.month, qe.year
    mon, mon3 = qe.strftime("%B"), qe.strftime("%b")
    alts = [f"{d:02d}[./-]{m:02d}[./-]{y}", f"{d}[./-]{m}[./-]{y}", f"{d:02d}[./-]{m:02d}[./-]{str(y)[2:]}\\b",
            f"{mon}\\s*{d}(?:st|nd|rd|th)?,?\\s*{y}", f"{d}(?:st|nd|rd|th)?\\s*(?:day of\\s*)?{mon},?\\s*{y}",
            f"{mon3}[a-z]*[.\\s-]*{d},?\\s*{y}", f"{d}[\\s-]*{mon3}[a-z]*[\\s,-]*{y}", f"{mon3}[a-z]*[\\s'-]*{str(y)[2:]}\\b"]
    return re.compile("|".join(alts), re.I)


def num(tok: str) -> float:
    neg = tok.startswith("(") and tok.endswith(")") or tok.startswith("-")
    v = float(tok.strip("()-").replace(",", ""))
    return -v if neg else v


def first_figure(line: str) -> float | None:
    line = re.sub(r"(\d)\s+,(\d)", r"\1,\2", line)
    line = re.sub(r"\((?:refer\s+)?note[^)]*\)|\([a-z](?:\s*[+&]\s*[a-z])+\)|\([ivx+\s]+\)", " ", line, flags=re.I)
    for t in NUM.findall(line):
        try:
            return num(t)
        except ValueError:
            continue
    return None


def parse(text: str, qe: pd.Timestamp) -> dict:
    text = text.replace("‐", "-").replace("‑", "-")
    unit = None
    for u, p in UNIT_RE:
        if re.search(r"(?:in|rs\.?|₹|inr|amount)[^\n]{0,25}" + p, text, re.I):
            unit = u; break
    out = dict(date_found=bool(date_pats(qe).search(text)), unit=unit)
    basis = "sa"
    got = {}
    for line in text.splitlines():
        low = line.lower()
        if ("consolidated" in low or "standalone" in low) and re.search(r"result|statement|financial", low) and len(low) < 200:
            basis = "con" if "consolidated" in low and "standalone" not in low else ("sa" if "standalone" in low and "consolidated" not in low else basis)
        if REV.search(line) and f"{basis}_sales" not in got:
            v = first_figure(REV.sub("", line, count=1))
            if v is not None:
                got[f"{basis}_sales"] = v
        elif PAT.search(line) and f"{basis}_pat" not in got:
            v = first_figure(PAT.sub("", line, count=1))
            if v is not None:
                got[f"{basis}_pat"] = v
    out.update(got)
    return out


def targets() -> pd.DataFrame:
    P = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet", columns=["symbol", "quarter_end", "net_sales"])
    P["quarter_end"] = pd.to_datetime(P["quarter_end"]); P = P.dropna(subset=["net_sales"])
    rows = []
    for s, g in P.groupby("symbol"):
        have = set(g["quarter_end"])
        allq = pd.date_range(max(min(have), pd.Timestamp("2018-06-30")), max(have), freq="QE")
        rows += [(s, q) for q in allq if q not in have]
    return pd.DataFrame(rows, columns=["symbol", "qe"])


def neighbours(P: pd.DataFrame) -> dict:
    P = P.dropna(subset=["net_sales"]).copy()
    P["lakh"] = P["net_sales"] * np.where(P["source"] == "xbrl", 1e-5, 1.0)
    return {k: g.set_index("quarter_end")["lakh"].sort_index() for k, g in P.groupby(["symbol", "basis"])} | \
           {(k, "any"): g.set_index("quarter_end")["lakh"].sort_index() for k, g in P.groupby("symbol")}


def ref_level(nb: dict, s: str, b: str, qe: pd.Timestamp) -> float | None:
    for key in ((s, b), (s, "any")):
        x = nb.get(key)
        if x is not None:
            w = x[(x.index >= qe - pd.Timedelta(days=380)) & (x.index <= qe + pd.Timedelta(days=380))]
            if len(w):
                return float(w.median())
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int); ap.add_argument("--test")
    a = ap.parse_args()
    A = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "desc", "an_dt", "attchmntFile", "attchmntText"])
    A["t"] = pd.to_datetime(A["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    A = A[A["attchmntFile"].astype(str).str.startswith("http") & (A["desc"].fillna("") + " " + A["attchmntText"].fillna("")).str.contains(RESULTS)]
    As = {s: g.sort_values("t") for s, g in A.groupby("symbol")}
    T = pd.DataFrame([a.test.split(":")], columns=["symbol", "qe"]) if a.test else targets()
    T["qe"] = pd.to_datetime(T["qe"])
    done = set()
    if CKPT.exists() and not a.test:
        done = {json.loads(l)["key"] for l in CKPT.read_text().splitlines() if l.strip()}
    T["key"] = T["symbol"] + "|" + T["qe"].dt.strftime("%Y-%m-%d")
    T = T[~T["key"].isin(done)]
    if a.limit:
        T = T.head(a.limit)
    print(f"missing company-quarters to look up: {len(T):,} (already done {len(done):,})", flush=True)
    lock, n = threading.Lock(), [0]

    def work(part: pd.DataFrame) -> list:
        s, res = build_session(warm=True, referer=REF), []
        for r in part.itertuples():
            g = As.get(r.symbol)
            cand = g[(g["t"] > r.qe + pd.Timedelta(days=1)) & (g["t"] <= r.qe + pd.Timedelta(days=90))] if g is not None else pd.DataFrame()
            rec = dict(key=r.key, symbol=r.symbol, qe=str(r.qe.date()), status="NO_FILING")
            for c in cand.head(3).itertuples():
                try:
                    b = _request_with_retries(s, c.attchmntFile, request_headers=_request_headers(s, referer=REF), referer=REF, timeout=60).content
                    txt, _ = ft._extract(c.attchmntFile, b)
                    if len(txt or "") < 500:
                        import ocr_order_filings as oo
                        txt = oo.ocr_blob(b); ocr = True
                    else:
                        ocr = False
                    p = parse(txt or "", r.qe)
                    if p.get("date_found") and (p.get("sa_sales") is not None or p.get("con_sales") is not None):
                        rec.update(p, status="OK", url=c.attchmntFile, filed=str(c.t), ocr=ocr); break
                    rec.update(status="NO_FIGURES" if p.get("date_found") else "WRONG_DATE", url=c.attchmntFile)
                except Exception as x:   # noqa: BLE001
                    rec.update(status=f"ERR_{type(x).__name__}", error=str(x)[:120])
                time.sleep(0.3)
            res.append(rec)
            with lock:
                n[0] += 1
                if n[0] % 100 == 0:
                    print(f"  {n[0]}/{len(T)}", flush=True)
                if not a.test:
                    with CKPT.open("a") as fh:
                        fh.write(json.dumps(rec, default=str) + "\n")
        return res

    with ThreadPoolExecutor(a.workers) as ex:
        out = [x for part in ex.map(work, [T.iloc[i::a.workers] for i in range(a.workers)]) for x in part]
    if a.test:
        print(json.dumps(out, indent=1, default=str)); return
    build()


def build() -> None:
    R = [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]
    R = pd.DataFrame(R).drop_duplicates("key", keep="last")
    print("lookups by status:", R["status"].value_counts().to_dict(), flush=True)
    P = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet", columns=["symbol", "quarter_end", "basis", "net_sales", "source"])
    P["quarter_end"] = pd.to_datetime(P["quarter_end"])
    nb = neighbours(P[~P["source"].isin(["nse_results_pdf"])])
    rows = []
    for r in R[R["status"] == "OK"].itertuples():
        qe = pd.Timestamp(r.qe)
        for b in ("sa", "con"):
            v = getattr(r, f"{b}_sales", None)
            if v is None or v != v:
                continue
            ref = ref_level(nb, r.symbol, b, qe)
            unit = r.unit if isinstance(r.unit, str) else None
            inferred = unit is None
            if inferred and ref:
                unit = min(FACTOR, key=lambda u: abs(np.log(max(v * FACTOR[u], 1e-9) / ref)))
            if unit is None:
                continue
            f = FACTOR[unit]; sales = v * f
            pat = getattr(r, f"{b}_pat", None)
            if pat is not None and pat == pat and abs(pat) > abs(v):     # profit larger than sales: a stray figure, dropped
                pat = None
            ok = bool(sales > 0 and ref and 1 / 3 <= sales / ref <= 3)
            rows.append(dict(symbol=r.symbol, quarter_end=qe, filing_dt=pd.Timestamp(r.filed), basis=b, bank=None, source="nse_results_pdf",
                             eps_basic=None, eps_diluted=None, net_sales=sales, total_income=None, pbt=None,
                             pat=(pat * f) if pat is not None and pat == pat else None, face_value=None,
                             unit=unit, unit_inferred=inferred, ocr=bool(r.ocr), url=r.url, ref_lakh=ref, qc_ok=ok,
                             qc_note="ok" if ok else ("no neighbour" if not ref else f"sales {sales:,.0f} vs neighbours {ref:,.0f} lakh")))
    D = pd.DataFrame(rows)
    D.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset=OUT.name, producer="src/agentic/fetch_results_pdf.py", rows=len(D), qc_ok=int(D["qc_ok"].sum()) if len(D) else 0,
        source="the company's results PDF from NSE's announcements archive, as first filed", definitions=__doc__,
        units=dict(net_sales="Rs LAKH", pat="Rs LAKH"), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"rows {len(D):,} · qc_ok {int(D['qc_ok'].sum()) if len(D) else 0} · unit inferred {int(D['unit_inferred'].sum()) if len(D) else 0} · "
          f"OCR {int(D['ocr'].sum()) if len(D) else 0}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build-only":
        build()
    else:
        main()
