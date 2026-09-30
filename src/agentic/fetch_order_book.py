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

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
from src.ingest.nse.api import _request_headers, _request_with_retries  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402
from event_materiality_study import CATS  # noqa: E402

OUTDIR = ROOT / "data/derived/order_book"
OUT = ROOT / "data/derived/order_book_filings.parquet"
REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
DOC_RE = re.compile(r"investor presentation|analyst presentation|press release|media release|financial result|outcome of board meeting|"
                    r"earnings call|conference call transcript|transcript of|analysts? meet|investor meet", re.I)
# 2026-09-28 pre-run review (confirmed): 'order booking(s)' = orders booked (intake), not the book -> inflow cue
CUE_BACKLOG = re.compile(r"order\s*book(?!ings?)|order\s*backlog|unexecuted\s+orders?|orders?\s+in\s+hand|outstanding\s+orders?|pending\s+orders?|"
                         r"order\s+position", re.I)
CUE_INFLOW = re.compile(r"order\s+inflows?|new\s+orders?\s+(?:received|won|bagged)|order\s+intake|order\s*bookings?", re.I)
_CUE_V1 = re.compile(r"order\s*book|order\s*backlog|unexecuted\s+orders?|orders?\s+in\s+hand|outstanding\s+orders?|pending\s+orders?|"
                     r"order\s+position|order\s+inflows?|new\s+orders?\s+(?:received|won|bagged)|order\s+intake", re.I)  # crawl-time cues
NOISE = re.compile(r"revenue|turnover|profit|ebitda|pat\b|income|sales", re.I)
AMT = re.compile(r"(?:(?:Rs\.?|INR|₹|&#8377;|Rupees)\s*|(USD|US\$)\s*)([\d,]+(?:\.\d+)?)\s*((?:lakhs?|lacs?)\s*(?:crores?|crs?\.?|cr\b)|thousand\s*crores?|crores?|crs?\.?|cr\b|lakhs?|lacs?|millions?|mn\b|billions?|bn\b)", re.I)
# 2026-09-28 pre-run review (confirmed): 'Rs 1.2 lakh crore' was read as 1.2 lakh (0.012 cr) -> compound units first
# 2026-09-28 follow-up review (confirmed): after the lakh-crore unit fix, 22 of 38 such figures were market size, budget, pipeline
# or target figures ("Rs 12.2 lakh crore in Union Budget"). A >= 1 lakh crore amount with this context is not a company's book.
# Checked on the text BEFORE the amount (a trailing 'sector'/'opportunities' also follows real books: BHEL, NBCC, MOTHERSON), plus
# 'by FY' / 'in the Budget' straight after it. Of the 16 lakh-crore drops under the first, wider rule, 5 were real books.
MACRO_PRE = re.compile(r"pipeline|opportunit|capex|budget|industry|sector|guid|target|potential|allocation|outlay|NHAI|government|"
                       r"their\s+order|in\s+front\s+of\s+us", re.I)
MACRO_POST = re.compile(r"^\W*(?:\w+\W+){0,2}(?:by\s+FY|in\s+(?:the\s+)?(?:union\s+)?budget)", re.I)
L1_PIPE = re.compile(r"\bL1\b|pipeline", re.I)   # 'order book + L1 of Rs X', 'bid pipeline of Rs X' between cue and amount
DELAY = 0.3
PER_QUARTER = 5          # documents per company-quarter (priority deck > transcript > results press release)


def to_cr(num: str, unit: str, usd: bool, fx: float) -> float | None:
    try:
        v = float(num.replace(",", ""))
    except ValueError:
        return None
    u = unit.lower()
    if re.match(r"(lakhs?|lacs?)\s*cr", u):
        mult = 1e5
    elif u.startswith("thousand"):
        mult = 1e3
    else:
            mult = 0.01 if u.startswith(("l",)) else 0.1 if u.startswith("m") else 100.0 if u.startswith("b") else 1.0
    cr = v * mult
    if usd:
        if not fx or np.isnan(fx):
            return None
        cr *= fx
    return cr


def first_amount(post: str, fx: float) -> float | None:
    """post = the 160 characters after a cue. The first amount in its first 140 characters that is not preceded (within 60)
    by revenue/profit wording; None if that amount follows 'L1'/'pipeline' after the cue, or is >= 1 lakh crore with
    market/budget/pipeline/target wording in the 80 characters before it (or 'by FY'/'in the Budget' right after). Only the first qualifying amount is considered."""
    win = post[:140]
    for a in AMT.finditer(win):
        if NOISE.search(win[max(0, a.start() - 60): a.start()]):
            continue
        if L1_PIPE.search(win[:a.start()]):
            return None
        cr = to_cr(a.group(2), a.group(3), bool(a.group(1)), fx)
        if not cr or cr <= 0:
            return None
        if cr >= 1e5 and (MACRO_PRE.search(post[max(0, a.start() - 80): a.start()]) or MACRO_POST.search(post[a.end(): a.end() + 40])):
            return None
        return cr
    return None


def extract(text: str, fx: float) -> list[dict]:
    out = []
    for kind, cue in (("backlog", CUE_BACKLOG), ("inflow", CUE_INFLOW)):
        for m in cue.finditer(text):
            cr = first_amount(text[m.end(): m.end() + 160], fx)
            if cr:
                out.append(dict(kind=kind, amount_cr=round(cr, 3), snippet=text[max(0, m.start() - 80): m.end() + 160]))
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


def crawl(n: int, k: int, min_rank: int = 0, max_rank: int = 99, by_doc: bool = False) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ck = OUTDIR / f"mentions_{n}of{k}.jsonl"
    W = pd.read_parquet(WORKLIST) if WORKLIST.exists() else worklist()
    W = W.sort_values(["rank", "symbol", "cq"], kind="mergesort") if "rank" in W else W   # every quarter's best document first
    if "rank" in W:                                       # worker groups split by rank so big decks and small PRs download in parallel
        W = W[W["rank"].between(min_rank, max_rank)]
    key = W["seq_id"].astype(str) if by_doc else W["symbol"]      # --by-doc: even tail split when a few companies are left
    W = W[key.map(lambda s: zlib.crc32(s.encode()) % k) == n] if k > 1 else W   # stable across processes
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


def reextract(m: dict, fx: float) -> dict | None:
    """Re-run the CURRENT cue/amount rules on one stored mention without refetching. The snippet is
    text[cue.start - 80 : cue.end + 160] of the whitespace-normalised document, so the 140-char amount window after the cue
    and the 60-char NOISE context both lie inside it. The generating cue sits at offset 80 (or at the first crawl-time cue
    when the document began less than 80 chars before it). Returns the updated mention, or None when the cue no longer
    qualifies or no amount is found."""
    sn = m.get("snippet") or ""
    c = _CUE_V1.match(sn, 80) if len(sn) > 80 else None
    c = c or _CUE_V1.search(sn)
    if c is None:
        return None
    p = c.start()
    cm, kind = CUE_BACKLOG.match(sn, p), "backlog"
    if cm is None:
        cm, kind = CUE_INFLOW.match(sn, p), "inflow"
    if cm is None:
        return None
    cr = first_amount(sn[cm.end(): cm.end() + 160], fx)
    return dict(m, kind=kind, amount_cr=round(cr, 3)) if cr else None


def book_amount(vals: list[float]) -> float:
    """One number per filing. Mentions within 2% of each other are one figure; the figure stated most often wins, ties go
    to the larger (a deck repeats its total book; one-off mentions are segments, inflows or prior years). Replaces the
    2026-09-28 median, which averaged the total with an inflow or segment when a filing had an even number of mentions
    (731 of 4,948 filings; 60% of a 60-filing hand-checked sample was the stated total book)."""
    if len(vals) > 1:                          # a figure > 20x the median of the filing's other mentions is not its book
        vals = [x for i, x in enumerate(vals) if x <= 20 * float(np.median(vals[:i] + vals[i + 1:]))] or vals
    cl: list[list[float]] = []
    for x in sorted(vals):
        if cl and x <= cl[-1][0] * 1.02:
            cl[-1].append(x)
        else:
            cl.append([x])
    top = sorted((c for c in cl if len(c) == max(map(len, cl))), key=lambda c: c[-1], reverse=True)
    while len(top) > 1 and top[0][-1] > 5 * top[1][-1]:   # ties go to the larger only within 5x (a total is ~1.5-5x a segment; headline figures 10x+)
        top = top[1:]
    return float(np.median(top[0]))


def consolidate() -> None:
    recs = []
    for f in sorted(OUTDIR.glob("mentions_*of*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    D = pd.DataFrame(recs)
    D["_ok"] = ~D["status"].astype(str).str.startswith(("HTTP_", "ERR_"))
    D = D.sort_values("_ok", kind="mergesort").drop_duplicates(["symbol", "seq_id"], keep="last")   # a later success beats a stale error
    fxs = pd.read_parquet(ROOT / "data/derived/usdinr_history.parquet").set_index("trade_date")["usdinr"].sort_index()
    rows, changed = [], dict(mentions=0, rescaled_lakh_crore=0, reclassified=0, dropped=0)
    dropped_log = []
    for r in D.itertuples():
        d = pd.to_datetime(r.sort_date, errors="coerce")
        fx = float(fxs.asof(d)) if pd.notna(d) else float("nan")
        ms = []
        for m in (r.mentions if isinstance(r.mentions, list) else []):
            n = reextract(m, fx)
            changed["mentions"] += 1
            if n is None:
                changed["dropped"] += 1
                dropped_log.append(dict(symbol=r.symbol, seq_id=r.seq_id, kind=m["kind"], amount_cr=m["amount_cr"], snippet=m.get("snippet")))
                continue
            changed["reclassified"] += n["kind"] != m["kind"]
            changed["rescaled_lakh_crore"] += n["amount_cr"] >= 1000 * max(m["amount_cr"], 1e-9)
            ms.append(n)
        b = [m["amount_cr"] for m in ms if m["kind"] == "backlog"]
        f_ = [m["amount_cr"] for m in ms if m["kind"] == "inflow"]
        rows.append(dict(symbol=r.symbol, seq_id=r.seq_id, filed=pd.to_datetime(r.sort_date, errors="coerce"), desc=r.desc, status=r.status,
                         backlog_cr=book_amount(b) if b else np.nan, inflow_cr=book_amount(f_) if f_ else np.nan,
                         backlog_median_cr=float(np.median(b)) if b else np.nan, backlog_all_cr=json.dumps(b),
                         n_backlog=len(b), n_inflow=len(f_), snippet=(next((m["snippet"] for m in ms if m["kind"] == "backlog"), None))))
    print(f"mention repair: {changed}", flush=True)
    pd.DataFrame(dropped_log).to_csv(OUTDIR / "dropped_mentions.csv", index=False)   # audit trail for the MACRO / L1 rules
    O = pd.DataFrame(rows).sort_values(["symbol", "filed"])
    O.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="order_book_filings", path=str(OUT.relative_to(ROOT)), rows=len(O), key=["symbol", "seq_id"], producer="src/agentic/fetch_order_book.py",
        source="NSE corporate-announcement attachments (investor presentations, press releases, results, transcripts) of companies that ever filed an order",
        columns=dict(filed="filing timestamp (IST, from announcements_historical.sort_date)", backlog_cr="order-book amount stated in the filing, Rs crore (NaN = none stated): the figure stated most often, mentions within 2% grouped, ties to the larger (book_amount)",
                     inflow_cr="order-inflow amount stated (same rule as backlog_cr), Rs crore", backlog_median_cr="median of all backlog mentions (the pre-2026-09-28-fix rule, kept for comparison)",
                     backlog_all_cr="JSON list of every backlog mention amount in the filing, Rs crore", n_backlog="backlog mentions found", n_inflow="inflow mentions found",
                     snippet="text around the first backlog mention (for audit)", status="OK | NO_MENTION | NO_TEXT | HTTP_<code> | ERR_<type>"),
        use="backlog / PIT TTM revenue (pnl_quarterly, manifest units) known at the filing time = revenue visibility in years; USE filed, not a quarter end, for point-in-time joins",
        repairs=changed, caveats=["2026-09-28 fixes applied at consolidation from stored snippets (no refetch): lakh-crore units, 'order booking' = inflow, per-filing figure rule; 'thousand crore'-only mentions skipped at crawl time are not recoverable without a refetch", "up to PER_QUARTER (5) documents per company-quarter; one quarter can restate the same book several times — take a PIT consensus, not a sum", "regex extraction: a deck may state segment or group backlog; check snippet for outliers", "no OCR (scanned decks yield NO_TEXT)"],
        coverage=dict(filings=len(O), with_backlog=int(O["backlog_cr"].notna().sum()), companies_with_backlog=int(O.loc[O["backlog_cr"].notna(), "symbol"].nunique())),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(f"order_book_filings: {len(O):,} filings · backlog stated in {int(O['backlog_cr'].notna().sum()):,} · companies {O.loc[O['backlog_cr'].notna(), 'symbol'].nunique()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--consolidate", action="store_true"); ap.add_argument("--count", action="store_true")
    ap.add_argument("--by-doc", action="store_true", help="shard by document instead of company")
    ap.add_argument("--min-rank", type=int, default=0); ap.add_argument("--max-rank", type=int, default=99)
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
        crawl(a.shard, a.of, a.min_rank, a.max_rank, a.by_doc)
