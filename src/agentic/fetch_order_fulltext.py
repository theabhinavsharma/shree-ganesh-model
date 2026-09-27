"""Full-text order amounts for every order / L1-tender filing 2016-2026 (EXP-2026-09-27-hot-x-material-order).

78% of order filings state no amount in the NSE headline, so order-size ('materiality') tests ran on
~655 orders only. This downloads each filing's attachment (announcements_historical.attchmntFile,
nsearchives.nseindia.com) through the NSE session/retry path, extracts text (PDF via pdfplumber/pypdf,
ZIP contents, XML/TXT) with fetch_filing_text._extract, and parses the largest Rs/USD amount with
event_materiality_study.parse_amount_cr (USD at that day's USDINR). Full extracted text is stored (checkpoint 'text'; consolidated to order_fulltext_text.parquet) so
amount parsing can be revised offline; the PDF itself is not stored.
Output: data/derived/order_fulltext.jsonl (checkpoint, resume-safe) -> order_fulltext.parquet (+manifest).
Rate: ~1.3 s/file; ~6,000 files ~ 2-3 h. Status: OK | NO_AMOUNT | NO_ATTACHMENT | NEEDS_OCR (scanned image; tesseract not installed) | HTTP_<code> | ERR_<type>.
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
                    time.sleep(5); s = build_session(warm=True, referer=REF)
            fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
            if i % 200 == 0:
                print(f"  {i}/{len(T)} {r.symbol} {rec['status']} {rec.get('fulltext_amount_cr')}", flush=True)
            time.sleep(0.8)
    D = pd.DataFrame([json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]).drop_duplicates(["symbol", "seq_id"], keep="last")
    D["amount_cr"] = D["fulltext_amount_cr"].fillna(D["headline_amount_cr"])
    D.drop(columns=["text"], errors="ignore").to_parquet(OUT, index=False)
    D[["symbol", "seq_id", "text"]].dropna().to_parquet(OUT.with_name("order_fulltext_text.parquet"), index=False)   # full text for offline re-parsing
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="order_fulltext", path=str(OUT.relative_to(ROOT)), rows=len(D), key=["symbol", "seq_id"], producer="src/agentic/fetch_order_fulltext.py",
        source="NSE corporate-announcement attachments (nsearchives.nseindia.com), official",
        columns=dict(cat="order | tender_L1 (event-study taxonomy)", d="filing date", url="attachment URL",
                     headline_amount_cr="largest amount in NSE headline text, Rs crore", fulltext_amount_cr="largest amount in the attachment text, Rs crore",
                     amount_cr="fulltext_amount_cr, else headline_amount_cr", pages="PDF pages", text_chars="extracted characters",
                     excerpt="<=4,000 chars around the first amount", status="OK | NO_AMOUNT | NO_ATTACHMENT | NEEDS_OCR | HTTP_<code> | ERR_<type>"),
        caveat="'largest amount' can pick a cumulative order book or group figure; USD converted at that day's USDINR",
        status_counts=D["status"].value_counts().to_dict(), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print("ORDER FULLTEXT COMPLETE", D["status"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
