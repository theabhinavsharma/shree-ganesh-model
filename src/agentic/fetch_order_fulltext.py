"""Full-text order amounts for every order / L1-tender filing 2016-2026 (EXP-2026-09-27-hot-x-material-order).

78% of order filings state no amount in the NSE headline, so order-size ('materiality') tests ran on
~655 orders only. This downloads each filing's attachment (announcements_historical.attchmntFile,
nsearchives.nseindia.com) through the NSE session/retry path, extracts text (PDF via pdfplumber/pypdf,
ZIP contents, XML/TXT) with fetch_filing_text._extract and stores the full text (checkpoint 'text';
consolidated to order_fulltext_text.parquet) so amount parsing can be revised offline; the PDF itself is
not stored. The loop's own per-record amounts are legacy; consolidate() re-parses the text with
parse_order_amounts (order-attached amount, historical USDINR only).
Output: data/derived/order_fulltext_v2.jsonl (checkpoint, resume-safe) -> order_fulltext.parquet (+manifest)
and order_amounts.parquet (+manifest).
Rate: ~1.3 s/file; ~6,000 files ~ 2-3 h. Status: OK | NO_AMOUNT | NO_ATTACHMENT | NEEDS_OCR (scanned image; tesseract not installed) | HTTP_<code> | ERR_<type>.
`--consolidate-only` rebuilds the parquet outputs from the checkpoint without crawling.

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json). Only the consolidation step changed;
the crawl loop is untouched because the v2 crawl was running when this was written.
  FIXED   :91 `amount_cr = fulltext_amount_cr.fillna(headline_amount_cr)` let the LARGEST full-text figure
          (revenue / group size / order book boilerplate) overwrite a correct headline amount. amount_cr is now
          parse_order_amounts' order-attached amount (headline first, then full text), NaN for filings that the
          current CATS no longer calls order / tender_L1, and order_amounts.parquet (+manifest) is written too.
  FIXED   :52-53,65-67,76 (outputs) the 83.0 USD fallback no longer reaches amount_cr: consolidation re-parses
          the stored text with historical USDINR only (macro_panel.usdinr, else data/derived/usdinr_history.parquet
          = FRED DEXINUS, both as-of the filing date); a USD order with no rate is NaN + usd_unconverted.
          RESIDUAL: the loop still writes headline/fulltext_amount_cr computed with 83.0 into the checkpoint;
          they are renamed legacy_* in the parquet and marked do-not-use.
  FIXED   :99 manifest status list now includes OCR_OK | OCR_NO_AMOUNT | OCR_EMPTY | OCR_SKIPPED_NON_PDF.
  PARTIAL :64-66 timestamp dropped (post-close filings used at that day's close downstream): the outputs now
          carry ts, post_close and d_actionable (d+1 when filed at/after 15:30 IST). The consumer
          test_hot_order_combo.py still keys on d and must switch to d_actionable (not changed here).
  NOTE    the targets() population follows the rewritten CATS, so the next crawl run adds up to 3,201 filings
          (NSE 'Awarding of order(s)/contract(s)', 'orders worth', 'receipt of order', bags/wins ...; fewer
          once rows without an attachment are dropped) and no longer targets 430 honours / arbitration /
          rating filings; already-crawled ones stay in the
          checkpoint and are marked by cat_current. New NEEDS_OCR rows need another ocr_order_filings.py pass
          followed by `fetch_order_fulltext.py --consolidate-only`.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
from src.ingest.nse.api import _request_headers, _request_with_retries  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402
from fetch_filing_text import _extract  # noqa: E402
from event_materiality_study import CATS, AMT, parse_amount_cr  # noqa: E402

CKPT = ROOT / "data/derived/order_fulltext_v2.jsonl"   # v2 (2026-09-27): stores full text; v1 (excerpt only) kept as evidence
OUT = ROOT / "data/derived/order_fulltext.parquet"
REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"


def targets() -> pd.DataFrame:
    ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                         columns=["symbol", "seq_id", "desc", "attchmntText", "attchmntFile", "sort_date"])
    t = ah["desc"].fillna("") + " " + ah["attchmntText"].fillna("")
    first = pd.Series(None, index=ah.index, dtype=object)
    for c, p in CATS.items():                                   # same first-match taxonomy as the event study
        m = first.isna() & t.str.contains(p, case=False, regex=True)
        first[m] = c
    ah = ah[first.isin(["order", "tender_L1"])].copy()
    ah["cat"] = first[ah.index]
    ah["headline_txt"] = t[ah.index]
    ah = ah.dropna(subset=["attchmntFile"]).drop_duplicates(["symbol", "seq_id"])
    return ah


def main() -> None:
    T = targets()
    mac = pd.read_parquet(ROOT / "data/derived/macro_panel.parquet", columns=["trade_date", "usdinr"])
    mac["trade_date"] = pd.to_datetime(mac["trade_date"]); fx = mac.set_index("trade_date")["usdinr"].sort_index()
    done = set()
    if CKPT.exists():
        FINAL = {"OK", "NO_AMOUNT", "NO_ATTACHMENT", "NEEDS_OCR", "OCR_OK", "OCR_NO_AMOUNT", "OCR_EMPTY", "OCR_SKIPPED_NON_PDF"}   # errors retried next run
        done = {(r["symbol"], str(r["seq_id"])) for r in (json.loads(l) for l in CKPT.read_text().splitlines() if l.strip())
                if r.get("status") in FINAL}
    T = T[[(s, str(q)) not in done for s, q in zip(T["symbol"], T["seq_id"])]]
    print(f"{len(T):,} order/L1 filings to fetch ({len(done):,} checkpointed)", flush=True)
    s = build_session(warm=True, referer=REF)
    with CKPT.open("a") as fh:
        for i, r in enumerate(T.itertuples()):
            d = pd.to_datetime(r.sort_date, errors="coerce")
            u = fx.asof(d) if pd.notna(d) else 83.0
            rec = dict(symbol=r.symbol, seq_id=str(r.seq_id), cat=r.cat, d=str(d.date()) if pd.notna(d) else None, url=r.attchmntFile,
                       headline_amount_cr=parse_amount_cr(r.headline_txt, u if pd.notna(u) else 83.0))
            if not str(r.attchmntFile).startswith("http"):
                rec["status"] = "NO_ATTACHMENT"
                fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
                continue
            try:
                resp = _request_with_retries(s, r.attchmntFile, request_headers=_request_headers(s, referer=REF), referer=REF, timeout=60)
                txt, pages = _extract(r.attchmntFile, resp.content)
                txt = re.sub(r"\s+", " ", txt or "").strip()
                amt = parse_amount_cr(txt, u if pd.notna(u) else 83.0) if txt else None
                m = AMT.search(txt) if txt else None
                lo = max(0, m.start() - 1500) if m else 0
                rec.update(pages=pages, text_chars=len(txt), fulltext_amount_cr=amt, excerpt=txt[lo:lo + 4000], text=txt,
                           status="OK" if amt else ("NO_AMOUNT" if txt else "NEEDS_OCR"))
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                rec["status"] = f"HTTP_{code}" if code else f"ERR_{type(e).__name__}"
                if not code and "Connection" in type(e).__name__:
                    for _ in range(12):              # network/DNS blip: wait and rebuild, never die here
                        time.sleep(30)
                        try:
                            s = build_session(warm=True, referer=REF); break
                        except Exception:
                            continue
            fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
            if i % 200 == 0:
                print(f"  {i}/{len(T)} {r.symbol} {rec['status']} {rec.get('fulltext_amount_cr')}", flush=True)
            time.sleep(0.8)
    consolidate()


STATUSES = ("OK | NO_AMOUNT | NO_ATTACHMENT | NEEDS_OCR | OCR_OK | OCR_NO_AMOUNT | OCR_EMPTY | OCR_SKIPPED_NON_PDF | "
            "HTTP_<code> | ERR_<type> (latest record per key; OCR records are appended by ocr_order_filings.py and win)")


def consolidate() -> pd.DataFrame:
    """Checkpoint -> order_fulltext.parquet (+ text parquet) and order_amounts.parquet (2026-09-27 audit fixes).
    amount_cr is the ORDER-ATTACHED amount from parse_order_amounts (headline first, then full text), not the
    largest amount; USD only with a historical USDINR (never a fixed rate); non-orders under the current CATS
    get amount_cr NaN. The crawl loop's own headline_amount_cr / fulltext_amount_cr (largest-amount rule,
    83.0 FX fallback) are kept only as legacy_* columns for comparison."""
    import parse_order_amounts as POA
    D = POA.load_checkpoint(CKPT)
    A = POA.build(D, POA.load_usdinr("derived"))
    POA.write(A, POA.OUT)                                           # one row per (symbol, seq_id)
    D = D.rename(columns={"headline_amount_cr": "legacy_headline_amount_cr", "fulltext_amount_cr": "legacy_fulltext_amount_cr"})
    add = ["cat_current", "ts", "post_close", "d_actionable", "era", "text_source", "order_amount_cr", "parsed_amount_cr", "amount_method",
           "amount_confidence", "currency", "amount_usd_mn", "usdinr", "fx_source", "usd_unconverted", "amount_flags",
           "headline_fulltext_agree"]
    D = D.drop(columns=[c for c in add if c in D.columns]).merge(A[["symbol", "seq_id"] + add], on=["symbol", "seq_id"], how="left")
    D["amount_cr"] = D["order_amount_cr"]                                      # already NaN for non-orders under current CATS
    D.drop(columns=["text"], errors="ignore").to_parquet(OUT, index=False)
    D[["symbol", "seq_id", "text"]].dropna().to_parquet(OUT.with_name("order_fulltext_text.parquet"), index=False)   # full text for offline re-parsing
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="order_fulltext", path=str(OUT.relative_to(ROOT)), rows=len(D), key=["symbol", "seq_id"], producer="src/agentic/fetch_order_fulltext.py",
        source="NSE corporate-announcement attachments (nsearchives.nseindia.com), official",
        columns=dict(cat="order | tender_L1 (event-study taxonomy at crawl time)",
                     cat_current="category under the current event_materiality_study.CATS (2026-09-27 fix)",
                     d="filing date (IST)", ts="filing timestamp (IST, announcements_historical.sort_date)",
                     post_close="filed at/after 15:30 IST", d_actionable="first date whose close may use the filing (d, or d+1 if post_close)",
                     url="attachment URL", amount_cr="order_amount_cr where cat_current is order|tender_L1, else NaN — Rs crore",
                     order_amount_cr="order-attached amount (parse_order_amounts rule), Rs crore; NaN when cat_current is not order|tender_L1",
                     parsed_amount_cr="parsed amount before the cat_current filter, Rs crore",
                     amount_method="how order_amount_cr was found (see order_amounts.parquet.manifest.json)",
                     amount_confidence="high | medium | low", currency="INR | USD | OTHER", amount_usd_mn="USD figure, USD million",
                     usdinr="historical USDINR as-of d (macro_panel, else usdinr_history = FRED DEXINUS), else NaN",
                     fx_source="macro_panel:<usdinr_source> | usdinr_history:fred_dexinus | null",
                     usd_unconverted="USD order with no historical rate -> amount NaN (no fixed rate)",
                     amount_flags="parser flags", headline_fulltext_agree="headline and full-text order amounts agree within 2%",
                     legacy_headline_amount_cr="RETIRED: largest amount in the headline, USD at 83.0 when no rate — do not use",
                     legacy_fulltext_amount_cr="RETIRED: largest amount in the attachment, USD at 83.0 when no rate — do not use",
                     pages="PDF pages", text_chars="extracted characters", excerpt="<=4,000 chars around the first amount",
                     ocr="True when the text came from the OCR pass", status=STATUSES),
        caveat="one amount per filing (not the sum of several listed orders); USD orders with no historical USDINR "
               "in data/derived have amount NaN (usd_unconverted); never a fixed rate",
        status_counts=D["status"].value_counts().to_dict(), updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print("ORDER FULLTEXT COMPLETE", D["status"].value_counts().to_dict(), flush=True)
    return D


if __name__ == "__main__":
    consolidate() if "--consolidate-only" in sys.argv else main()
