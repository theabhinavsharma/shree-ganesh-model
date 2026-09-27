"""OCR pass for scanned order/L1 filings (companion to fetch_order_fulltext.py).

Takes every record in data/derived/order_fulltext.jsonl whose status is NEEDS_OCR (image-only PDF),
re-downloads the attachment through the NSE session, renders the first MAX_PAGES pages at 200 dpi
(pdftoppm) and OCRs them with tesseract (eng), then parses the largest Rs/USD amount exactly as the
text pass does (event_materiality_study.parse_amount_cr). Appends an updated record (status OCR_OK |
OCR_NO_AMOUNT | OCR_EMPTY | HTTP_<code> | ERR_<type>, ocr=True); the latest record per (symbol, seq_id)
wins when fetch_order_fulltext.py consolidates. Nothing else is touched.
Requires: tesseract + pdftoppm on PATH (brew install tesseract poppler).
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
from event_materiality_study import AMT, parse_amount_cr  # noqa: E402

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
    for tool in ("tesseract", "pdftoppm"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not on PATH — install first (brew install tesseract poppler)")
    recs = [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]
    latest = {}
    for r in recs:
        latest[(r["symbol"], str(r["seq_id"]))] = r
    todo = [r for r in latest.values() if r.get("status") == "NEEDS_OCR" or (r.get("ocr") and str(r.get("status", "")).startswith(("HTTP_", "ERR_")))]
    mac = pd.read_parquet(ROOT / "data/derived/macro_panel.parquet", columns=["trade_date", "usdinr"])
    mac["trade_date"] = pd.to_datetime(mac["trade_date"]); fx = mac.set_index("trade_date")["usdinr"].sort_index()
    print(f"{len(todo):,} scanned filings to OCR", flush=True)
    s = build_session(warm=True, referer=REF)
    with CKPT.open("a") as fh:
        for i, r in enumerate(todo):
            u = fx.asof(pd.Timestamp(r["d"])) if r.get("d") else 83.0
            rec = dict(r, ocr=True)
            try:
                resp = _request_with_retries(s, r["url"], request_headers=_request_headers(s, referer=REF), referer=REF, timeout=60)
                if not r["url"].lower().endswith(".pdf"):
                    rec["status"] = "OCR_SKIPPED_NON_PDF"
                else:
                    txt = ocr_pdf(resp.content)
                    amt = parse_amount_cr(txt, u if pd.notna(u) else 83.0) if txt else None
                    m = AMT.search(txt) if txt else None
                    lo = max(0, m.start() - 1500) if m else 0
                    rec.update(text_chars=len(txt), fulltext_amount_cr=amt, excerpt=txt[lo:lo + 4000], text=txt,
                               status="OCR_OK" if amt else ("OCR_NO_AMOUNT" if txt else "OCR_EMPTY"))
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
