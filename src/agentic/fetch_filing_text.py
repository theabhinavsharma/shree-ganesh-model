"""Full text of specific NSE corporate filings (the attachment, not the headline).

The canonical announcements store (data/events_full_history/normalized/
stock_announcements.parquet) keeps NSE's headline + short attchmntText only; for many
filings (resignations, licence changes, auditor changes, ICDR certificates) that is just
the category title. This fetches the attachment itself through the NSE downloader
(src.ingest.nse session + get_json retry/warm-up path), matched on sequence_id:
  1. corporate-announcements API for the symbol on the filing's date -> attchmntFile URL
  2. download (PDF, or ZIP holding PDFs/XML/TXT) from nsearchives.nseindia.com
  3. extract text: pdfplumber, fallback pypdf; XML/TXT decoded as-is
Raw files: data/events_full_history/attachments/<SYMBOL>/<seq_id>.<ext>
Text store: data/derived/filing_fulltext.parquet (+ .manifest.json), upsert on
(symbol, seq_id). Scanned-image PDFs yield empty text and are recorded as such
(text_chars = 0) — never guessed.

Usage: fetch_filing_text.py MMFL:106745548 UFLEX:106786308 ...
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
from src.ingest.nse.api import _request_headers, _request_with_retries, get_json  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

ANN = ROOT / "data/events_full_history/normalized/stock_announcements.parquet"
RAW = ROOT / "data/events_full_history/attachments"
OUT = ROOT / "data/derived/filing_fulltext.parquet"
REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"


def _pdf_text(blob: bytes) -> tuple[str, int]:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(blob)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages), len(pdf.pages)
    except Exception:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(blob))
        return "\n".join(p.extract_text() or "" for p in r.pages), len(r.pages)


def _extract(name: str, blob: bytes) -> tuple[str, int]:
    low = name.lower()
    if low.endswith(".pdf"):
        return _pdf_text(blob)
    if low.endswith(".zip"):
        parts, pages = [], 0
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for n in z.namelist():
                t, p = _extract(n, z.read(n))
                if t.strip():
                    parts.append(f"=== {n} ===\n{t}")
                pages += p
        return "\n\n".join(parts), pages
    if low.endswith((".xml", ".txt", ".html", ".htm")):
        return blob.decode("utf-8", "replace"), 0
    return "", 0


def fetch(session, sym: str, seq: str, event_date: pd.Timestamp) -> dict:
    d = event_date.strftime("%d-%m-%Y")
    url = (f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={sym}"
           f"&from_date={d}&to_date={d}")
    j = get_json(session, url, referer=REF)
    rows = j if isinstance(j, list) else j.get("data", [])
    hit = next((r for r in rows if str(r.get("seq_id")) == seq), None)
    base = dict(symbol=sym, seq_id=seq, event_date=event_date.normalize(), fetched_at=datetime.now().isoformat(timespec="seconds"))
    if hit is None or not hit.get("attchmntFile"):
        return dict(base, status="NO_ATTACHMENT_IN_API", url=None, headline=None, an_dt=None, pages=0, text="", text_chars=0)
    furl = hit["attchmntFile"]
    resp = _request_with_retries(session, furl, request_headers=_request_headers(session, referer=REF),
                                 referer=REF, timeout=60)
    ext = Path(furl).suffix.lower() or ".bin"
    raw = RAW / sym / f"{seq}{ext}"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(resp.content)
    text, pages = _extract(furl, resp.content)
    text = text.strip()
    return dict(base, status="OK" if text else "EMPTY_TEXT (scanned image?)", url=furl, headline=hit.get("desc"),
                an_dt=hit.get("an_dt"), pages=pages, text=text, text_chars=len(text),
                raw_path=str(raw.relative_to(ROOT)), raw_bytes=len(resp.content))


def main(pairs: list[str]) -> None:
    want = [tuple(p.split(":", 1)) for p in pairs]
    ann = pd.read_parquet(ANN, columns=["symbol", "sequence_id", "event_date"],
                          filters=[("symbol", "in", sorted({s for s, _ in want}))])
    ann["event_date"] = pd.to_datetime(ann["event_date"], errors="coerce")
    s = build_session(warm=True, referer=REF)
    recs = []
    for sym, seq in want:
        m = ann[(ann["symbol"] == sym) & (ann["sequence_id"].astype(str) == seq)]
        if m.empty:
            print(f"{sym}:{seq} not in canonical store — skipped"); continue
        r = fetch(s, sym, seq, m["event_date"].iloc[0])
        recs.append(r)
        print(f"{sym}:{seq} {r['status']} · {r['pages']}p · {r['text_chars']:,} chars · {r.get('url')}")
    if not recs:
        return
    new = pd.DataFrame(recs)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        new = pd.concat([old, new]).drop_duplicates(["symbol", "seq_id"], keep="last")
    new.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="filing_fulltext", path=str(OUT.relative_to(ROOT)), rows=len(new),
        key=["symbol", "seq_id"], producer="src/agentic/fetch_filing_text.py",
        source="NSE corporate-announcements API (attchmntFile) -> nsearchives.nseindia.com; official NSE",
        columns=dict(symbol="NSE symbol", seq_id="NSE announcement sequence_id (joins stock_announcements.sequence_id)",
                     event_date="announcement date", an_dt="NSE broadcast timestamp (IST, as published)",
                     headline="NSE category/description", url="attachment URL", raw_path="downloaded file",
                     pages="PDF pages (0 for XML/TXT)", text="extracted text, unedited",
                     text_chars="len(text); 0 = nothing extractable, NOT 'no content'",
                     status="OK | EMPTY_TEXT (scanned image?) | NO_ATTACHMENT_IN_API", fetched_at="local fetch time"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(new)} rows)")


if __name__ == "__main__":
    main(sys.argv[1:])
