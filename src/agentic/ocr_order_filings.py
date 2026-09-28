"""OCR pass for scanned order/L1 filings (companion to fetch_order_fulltext.py).

Takes every record in data/derived/order_fulltext_v2.jsonl whose latest status is NEEDS_OCR (image-only PDF),
re-downloads the attachment through the NSE session, renders the first MAX_PAGES pages at 200 dpi
(pdftoppm) and OCRs them with tesseract (eng), then stores the text and the order-attached amount from
parse_order_amounts.extract_order_amount. Appends an updated record (status OCR_OK = the OCR text contains
at least one currency amount | OCR_NO_AMOUNT | OCR_EMPTY | OCR_SKIPPED_NON_PDF | HTTP_<code> | ERR_<type>,
ocr=True); the latest record per (symbol, seq_id) wins when fetch_order_fulltext.py consolidates (run
`fetch_order_fulltext.py --consolidate-only` afterwards). Records whose latest OCR attempt failed with
HTTP_/ERR_ are retried on the next run. Nothing else is touched.
Requires: tesseract + pdftoppm on PATH (brew install tesseract poppler).

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json):
  FIXED  :57,62,70 USD converted at a fixed 83.0 whenever macro_panel had no USDINR (all dates before
         2024-02-19). Now as-of the filing date (7-day tolerance) from data/derived only: macro_panel.usdinr,
         else usdinr_history.parquet (FRED DEXINUS); no rate -> the USD amount stays NaN. fulltext_amount_cr
         is the ORDER-ATTACHED amount, not the largest one; the consolidation step re-parses the stored text
         anyway, so this field is informational.
  FIXED  :52-55,75-78 a transient HTTP_/ERR_ in the OCR pass was permanent. The todo filter (already patched in
         the committed file) also takes records whose latest status is an OCR-pass HTTP_/ERR_; verified here.
         Non-PDF attachments are now skipped before downloading (same OCR_SKIPPED_NON_PDF status, no request).
  FIXED  OCR_* statuses were missing from the fetch manifest -> fetch_order_fulltext.STATUSES lists them.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
from src.ingest.nse.api import _request_headers, _request_with_retries  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402
from parse_order_amounts import AMT, extract_order_amount, load_usdinr, FX_TOL  # noqa: E402

CKPT = ROOT / "data/derived/order_fulltext_v2.jsonl"
REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
MAX_PAGES = 4


def ocr_pdf(blob: bytes) -> str:
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "f.pdf"; pdf.write_bytes(blob)
        subprocess.run(["pdftoppm", "-r", "200", "-l", str(MAX_PAGES), "-png", str(pdf), str(Path(td) / "p")],
                       check=True, capture_output=True, timeout=120)
        out = []
        for png in sorted(Path(td).glob("p*.png")):
            r = subprocess.run(["tesseract", str(png), "stdout", "-l", "eng"], capture_output=True, text=True, timeout=120)
            out.append(r.stdout)
        return re.sub(r"\s+", " ", " ".join(out)).strip()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="only OCR filings dated on/after this date (e.g. 2018-10-01)")
    args = ap.parse_args()
    for tool in ("tesseract", "pdftoppm"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not on PATH — install first (brew install tesseract poppler)")
    recs = [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]
    latest = {}
    for r in recs:
        latest[(r["symbol"], str(r["seq_id"]))] = r
    todo = [r for r in latest.values() if r.get("status") == "NEEDS_OCR" or (r.get("ocr") and str(r.get("status", "")).startswith(("HTTP_", "ERR_")))]
    if args.since:
        todo = [r for r in todo if (r.get("d") or "") >= args.since]
    fx = load_usdinr().set_index("fx_date")["usdinr"]                   # historical series only, never a fixed rate
    print(f"{len(todo):,} scanned filings to OCR", flush=True)
    s = build_session(warm=True, referer=REF)
    with CKPT.open("a") as fh:
        for i, r in enumerate(todo):
            u = float("nan")
            if r.get("d"):
                d = pd.Timestamp(r["d"]); w = fx.loc[d - FX_TOL:d]
                u = float(w.iloc[-1]) if len(w) else float("nan")
            rec = dict(r, ocr=True)
            try:
                if not str(r.get("url", "")).lower().endswith(".pdf"):
                    rec["status"] = "OCR_SKIPPED_NON_PDF"
                else:
                    resp = _request_with_retries(s, r["url"], request_headers=_request_headers(s, referer=REF), referer=REF, timeout=60)
                    txt = ocr_pdf(resp.content)
                    amt = extract_order_amount(None, txt, u)["order_amount_cr"] if txt else None
                    amt = None if amt is None or amt != amt else float(amt)
                    m = AMT.search(txt) if txt else None
                    lo = max(0, m.start() - 1500) if m else 0
                    rec.update(text_chars=len(txt), fulltext_amount_cr=amt, excerpt=txt[lo:lo + 4000], text=txt,
                               status="OCR_OK" if m else ("OCR_NO_AMOUNT" if txt else "OCR_EMPTY"))
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                rec["status"] = f"HTTP_{code}" if code else f"ERR_{type(e).__name__}"
            fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} {r['symbol']} {rec['status']} {rec.get('fulltext_amount_cr')}", flush=True)
            time.sleep(0.8)
    print("OCR PASS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
