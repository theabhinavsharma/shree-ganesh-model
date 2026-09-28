"""ORDER BOOK (backlog) from company filings — investor presentations, press releases, results, concall transcripts.

Why (2026-09-28): single order filings are a noisy materiality measure (disclosure is uneven, 40-50% carry no
amount, few independent orders before 2023). The order BOOK a company reports each quarter ("order book stands
at Rs 12,345 crore", "unexecuted order backlog of ...") divided by revenue is the standard measure of revenue
visibility. This extracts every such mention from official NSE filing attachments.

Scope: companies that filed at least one order / L1-tender announcement (event taxonomy in
event_materiality_study.CATS), and their filings that state an order book: investor/analyst presentations,
concall transcripts, and press releases about results (meeting schedules and bare results PDFs are skipped). Source rows and attachment URLs:
data/derived/announcements_historical.parquet (official NSE).
Text: `pdftotext` (poppler) for PDFs; zip members extracted; no OCR (presentations are digital).
Extraction: every amount (Rs/INR/₹ crore/lakh/million/billion, USD at the historical FRED rate) within 140 chars
AFTER a cue, cue kinds:
    backlog  order book | order backlog | unexecuted order(s) | order(s) in hand | outstanding order(s) | pending order(s)
    inflow   order inflow(s) | new orders received | order intake
Amounts inside ±60 chars of revenue/turnover/profit words are skipped. Per filing the backlog figure is the MEDIAN of
its backlog mentions (a deck repeats the same number); inflow likewise.
Output (per shard n of k, resume-safe): data/derived/order_book/mentions_<n>of<k>.jsonl; --consolidate writes
data/derived/order_book_filings.parquet (+manifest): one row per filing with backlog_cr, inflow_cr, n_mentions, snippet.
Usage: fetch_order_book.py --shard 0 --of 2   |   fetch_order_book.py --consolidate
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
from src.ingest.nse.api import _request_headers, _request_with_retries  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402
from event_materiality_study import CATS  # noqa: E402

OUTDIR = ROOT / "data/derived/order_book"
OUT = ROOT / "data/derived/order_book_filings.parquet"
REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
DOC_RE = re.compile(r"investor presentation|analyst presentation|press release|media release|financial result|outcome of board meeting|"
                    r"earnings call|conference call transcript|transcript of|analysts? meet|investor meet", re.I)
CUE_BACKLOG = re.compile(r"order\s*book|order\s*backlog|unexecuted\s+orders?|orders?\s+in\s+hand|outstanding\s+orders?|pending\s+orders?|"
                         r"order\s+position", re.I)
CUE_INFLOW = re.compile(r"order\s+inflows?|new\s+orders?\s+(?:received|won|bagged)|order\s+intake", re.I)
NOISE = re.compile(r"revenue|turnover|profit|ebitda|pat\b|income|sales", re.I)
AMT = re.compile(r"(?:(?:Rs\.?|INR|₹|&#8377;|Rupees)\s*|(USD|US\$)\s*)([\d,]+(?:\.\d+)?)\s*(crores?|crs?\.?|cr\b|lakhs?|lacs?|millions?|mn\b|billions?|bn\b)", re.I)
DELAY = 0.3
PER_QUARTER = 5          # documents per company-quarter (priority deck > transcript > results press release)


def to_cr(num: str, unit: str, usd: bool, fx: float) -> float | None:
    try:
        v = float(num.replace(",", ""))
    except ValueError:
        return None
    u = unit.lower()
    mult = 0.01 if u.startswith(("l",)) else 0.1 if u.startswith("m") else 100.0 if u.startswith("b") else 1.0
    cr = v * mult
    if usd:
        if not fx or np.isnan(fx):
            return None
        cr *= fx
    return cr


def extract(text: str, fx: float) -> list[dict]:
    out = []
    for kind, cue in (("backlog", CUE_BACKLOG), ("inflow", CUE_INFLOW)):
        for m in cue.finditer(text):
            win = text[m.end(): m.end() + 140]
            for a in AMT.finditer(win):
                ctx = win[max(0, a.start() - 60): a.start()]
                if NOISE.search(ctx):
                    continue
                cr = to_cr(a.group(2), a.group(3), bool(a.group(1)), fx)
                if cr and cr > 0:
                    out.append(dict(kind=kind, amount_cr=round(cr, 3), snippet=text[max(0, m.start() - 80): m.end() + 160]))
                break                                            # first amount after the cue only
    return out


def pdf_text(blob: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(blob); f.flush()
        r = subprocess.run(["pdftotext", "-q", "-l", "30", f.name, "-"], capture_output=True, text=True, timeout=120)   # order book sits in the first pages
        return r.stdout


def blob_text(url: str, blob: bytes) -> str:
    u = url.lower()
    if u.endswith(".pdf"):
        return pdf_text(blob)
    if u.endswith(".zip"):
        parts = []
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for n in z.namelist():
                if n.lower().endswith(".pdf"):
                    parts.append(pdf_text(z.read(n)))
                elif n.lower().endswith((".txt", ".htm", ".html", ".xml")):
                    parts.append(z.read(n).decode("utf-8", "replace"))
        return "\n".join(parts)
    return blob.decode("utf-8", "replace") if u.endswith((".txt", ".htm", ".html")) else ""


def worklist() -> pd.DataFrame:
    ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                         columns=["symbol", "seq_id", "desc", "attchmntText", "attchmntFile", "sort_date"])
    t = ah["desc"].fillna("") + " " + ah["attchmntText"].fillna("")
    is_order = pd.Series(False, index=ah.index)
    first = pd.Series(None, index=ah.index, dtype=object)
    for c, p in CATS.items():
        m = first.isna() & t.str.contains(p, case=False, regex=True)
        first[m] = c
    is_order = first.isin(["order", "tender_L1"])
    filers = set(ah.loc[is_order, "symbol"])
    # documents that actually STATE an order book: decks, transcripts, results press releases (not meeting schedules)
    desc = ah["desc"].fillna("")
    deck = desc.str.contains(r"investor presentation|analyst presentation", case=False) | t.str.contains(r"investor presentation", case=False)
    transcript = t.str.contains(r"transcript", case=False)
    results_pr = desc.str.contains(r"press release|media release", case=False) & t.str.contains(r"result|quarter|financial|performance|\bq[1-4]\b", case=False)
    W = ah[ah["symbol"].isin(filers) & (deck | transcript | results_pr) & ah["attchmntFile"].astype(str).str.startswith("http")].copy()
    # 2026-09-28 speed-up: up to PER_QUARTER documents per company-quarter (a deck, its transcript and the results press release restate
    # the same order book). Priority deck > transcript > results press release; latest filing within the quarter wins.
    W["prio"] = np.select([deck[W.index], transcript[W.index]], [0, 1], 2)
    W["cq"] = pd.to_datetime(W["sort_date"], errors="coerce").dt.to_period("Q").astype(str)
    W = W.sort_values(["symbol", "cq", "prio", "sort_date"], ascending=[True, True, True, False])
    W = W.groupby(["symbol", "cq"], sort=False).head(PER_QUARTER)     # user 2026-09-28: 4-5 documents per company-quarter
    W["rank"] = W.groupby(["symbol", "cq"], sort=False).cumcount()     # 0 = the best document of its quarter
    W["seq_id"] = W["seq_id"].astype(str)
    return W.drop_duplicates(["symbol", "seq_id"]).sort_values(["symbol", "sort_date"]).reset_index(drop=True)


WORKLIST = OUTDIR / "worklist.parquet"   # cache so 40 parallel workers do not each load announcements_historical


def crawl(n: int, k: int) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ck = OUTDIR / f"mentions_{n}of{k}.jsonl"
    W = pd.read_parquet(WORKLIST) if WORKLIST.exists() else worklist()
    W = W.sort_values(["rank", "symbol", "cq"], kind="mergesort") if "rank" in W else W   # every quarter's best document first
    W = W[W["symbol"].map(lambda s: zlib.crc32(s.encode()) % k) == n] if k > 1 else W   # stable across processes
    done = set()
    for f in OUTDIR.glob("mentions_*of*.jsonl"):          # every shard's output, so re-sharding never refetches
        recs = (json.loads(l) for l in f.read_text().splitlines() if l.strip())
        done |= {(r["symbol"], r["seq_id"]) for r in recs if not r["status"].startswith(("HTTP_", "ERR_"))}
    W = W[[(s, q) not in done for s, q in zip(W["symbol"], W["seq_id"])]]
    fxs = pd.read_parquet(ROOT / "data/derived/usdinr_history.parquet").set_index("trade_date")["usdinr"].sort_index()
    print(f"shard {n}/{k}: {len(W):,} filings to fetch ({len(done):,} done) from {W['symbol'].nunique()} companies", flush=True)
    s = build_session(warm=True, referer=REF)
    with ck.open("a") as fh:
        for i, r in enumerate(W.itertuples()):
            d = pd.to_datetime(r.sort_date, errors="coerce")
            fx = float(fxs.asof(d)) if pd.notna(d) else float("nan")
            rec = dict(symbol=r.symbol, seq_id=r.seq_id, sort_date=str(r.sort_date), desc=str(r.desc)[:120], url=r.attchmntFile)
            try:
                resp = _request_with_retries(s, r.attchmntFile, request_headers=_request_headers(s, referer=REF), referer=REF, timeout=60)
                txt = re.sub(r"\s+", " ", blob_text(r.attchmntFile, resp.content))
                ms = extract(txt, fx)
                rec.update(status="OK" if ms else ("NO_MENTION" if txt else "NO_TEXT"), text_chars=len(txt), mentions=ms)
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                rec["status"] = f"HTTP_{code}" if code else f"ERR_{type(e).__name__}"
                if not code:
                    for _ in range(10):
                        time.sleep(30)
                        try:
                            s = build_session(warm=True, referer=REF); break
                        except Exception:
                            continue
            fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
            if i % 250 == 0:
                print(f"  [{n}/{k}] {i}/{len(W)} {r.symbol} {rec['status']}", flush=True)
            time.sleep(DELAY)


def consolidate() -> None:
    recs = []
    for f in sorted(OUTDIR.glob("mentions_*of*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    D = pd.DataFrame(recs).drop_duplicates(["symbol", "seq_id"], keep="last")
    rows = []
    for r in D.itertuples():
        ms = r.mentions if isinstance(r.mentions, list) else []
        b = [m["amount_cr"] for m in ms if m["kind"] == "backlog"]
        f_ = [m["amount_cr"] for m in ms if m["kind"] == "inflow"]
        rows.append(dict(symbol=r.symbol, seq_id=r.seq_id, filed=pd.to_datetime(r.sort_date, errors="coerce"), desc=r.desc, status=r.status,
                         backlog_cr=float(np.median(b)) if b else np.nan, inflow_cr=float(np.median(f_)) if f_ else np.nan,
                         n_backlog=len(b), n_inflow=len(f_), snippet=(next((m["snippet"] for m in ms if m["kind"] == "backlog"), None))))
    O = pd.DataFrame(rows).sort_values(["symbol", "filed"])
    O.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="order_book_filings", path=str(OUT.relative_to(ROOT)), rows=len(O), key=["symbol", "seq_id"], producer="src/agentic/fetch_order_book.py",
        source="NSE corporate-announcement attachments (investor presentations, press releases, results, transcripts) of companies that ever filed an order",
        columns=dict(filed="filing timestamp (IST, from announcements_historical.sort_date)", backlog_cr="median order-book/backlog amount stated in the filing, Rs crore (NaN = none stated)",
                     inflow_cr="median order-inflow amount stated, Rs crore", n_backlog="backlog mentions found", n_inflow="inflow mentions found",
                     snippet="text around the first backlog mention (for audit)", status="OK | NO_MENTION | NO_TEXT | HTTP_<code> | ERR_<type>"),
        use="backlog / PIT TTM revenue (pnl_quarterly, manifest units) known at the filing time = revenue visibility in years; USE filed, not a quarter end, for point-in-time joins",
        caveats=["up to PER_QUARTER (5) documents per company-quarter; one quarter can restate the same book several times — take a PIT consensus, not a sum", "regex extraction: a deck may state segment or group backlog; check snippet for outliers", "no OCR (scanned decks yield NO_TEXT)"],
        coverage=dict(filings=len(O), with_backlog=int(O["backlog_cr"].notna().sum()), companies_with_backlog=int(O.loc[O["backlog_cr"].notna(), "symbol"].nunique())),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(f"order_book_filings: {len(O):,} filings · backlog stated in {int(O['backlog_cr'].notna().sum()):,} · companies {O.loc[O['backlog_cr'].notna(), 'symbol'].nunique()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--consolidate", action="store_true"); ap.add_argument("--count", action="store_true")
    ap.add_argument("--build-worklist", action="store_true", help=f"write the worklist cache the crawl workers read")
    a = ap.parse_args()
    if a.build_worklist:
        OUTDIR.mkdir(parents=True, exist_ok=True); W = worklist(); W.to_parquet(WORKLIST, index=False)
        print(f"worklist cached: {len(W):,} filings, {W['symbol'].nunique()} companies, by rank {W['rank'].value_counts().sort_index().to_dict()}")
    elif a.count:
        W = worklist(); print(f"{len(W):,} filings from {W['symbol'].nunique()} companies"); print(W["desc"].str[:40].value_counts().head(8).to_string())
    elif a.consolidate:
        consolidate()
    else:
        crawl(a.shard, a.of)
