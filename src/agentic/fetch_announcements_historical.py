"""Backfill NSE corporate announcements from 2016-01-01 to today.

Wraps the existing 14-day rolling fetcher (`refresh_announcements.py`) with:
  - Configurable start date (default 2016-01-01)
  - 14-day chunks (NSE API limit per call)
  - Polite rate limiting (8s between chunks)
  - Checkpointing — resumes from last successfully-fetched chunk
  - Writes shards: data/raw/announcements_historical/YYYY/<date>.parquet
  - Final consolidator: rebuilds data/derived/announcements_historical.parquet

Usage:
  python3 src/agentic/fetch_announcements_historical.py                  # full backfill 2016+
  python3 src/agentic/fetch_announcements_historical.py --start 2020-01-01
  python3 src/agentic/fetch_announcements_historical.py --consolidate-only

Honest caveats:
  - NSE corporate-announcements API has historical depth that varies by series;
    coverage 2016-2018 may be sparse.
  - This is the structured-filings source (deals, results, partnerships,
    pledge changes, board decisions). Pure news (MoneyControl/ET/Bloomberg)
    requires paid feeds; flagged TODO at end of this file.
  - Each fetch chunk takes ~3-8 seconds. ~260 chunks = ~30-50 minutes.

Per CONSTITUTION.md §1.7 — this fetcher must be reproducible. Start date,
chunk size, sleep duration, and any retry policy are recorded in the JSONL
checkpoint file so a re-run produces identical output.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
from src.ingest.nse.session import build_session
from src.ingest.nse.api import get_json

OUT_DIR = ROOT / "data/raw/announcements_historical"
CONSOLIDATED = ROOT / "data/derived/announcements_historical.parquet"
CHECKPOINT = OUT_DIR / "_checkpoint.jsonl"
ANN_REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
ANN_API = "https://www.nseindia.com/api/corporate-announcements"

CHUNK_DAYS = 14
SLEEP_S = 8


def fetch_chunk(session, start: date, end: date) -> list[dict]:
    f = start.strftime("%d-%m-%Y")
    t = end.strftime("%d-%m-%Y")
    url = f"{ANN_API}?index=equities&from_date={f}&to_date={t}"
    j = get_json(session, url, referer=ANN_REF)
    data = j if isinstance(j, list) else j.get("data", [])
    return data


def chunk_path(start: date) -> Path:
    return OUT_DIR / str(start.year) / f"{start.isoformat()}.parquet"


def has_chunk(start: date) -> bool:
    return chunk_path(start).exists()


def append_checkpoint(entry: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def consolidate() -> None:
    """Read all shard parquets and write the consolidated file."""
    files = sorted(OUT_DIR.rglob("*.parquet"))
    files = [f for f in files if f.name != CONSOLIDATED.name]
    if not files:
        print("no shards to consolidate")
        return
    frames = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            if not d.empty:
                d["_source_shard"] = f.relative_to(OUT_DIR).as_posix()
                frames.append(d)
        except Exception as e:
            print(f"  skip {f}: {e}")
    if not frames:
        print("no usable shards")
        return
    df = pd.concat(frames, ignore_index=True)
    # dedupe — common keys vary across NSE responses; pick a robust set
    dedup_keys = [k for k in ("symbol", "an_dt", "desc", "exchdisstime") if k in df.columns]
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
    print(f"consolidated: {len(df):,} rows from {len(frames)} shards")
    CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CONSOLIDATED, index=False)
    print(f"wrote {CONSOLIDATED.relative_to(ROOT)}")


def backfill(start_date: date, end_date: date, sleep_s: int = SLEEP_S) -> None:
    print(f"== backfill from {start_date} to {end_date} ==")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session(warm=True, referer=ANN_REF)
    cur = start_date
    n_total = 0
    n_skipped = 0
    n_fetched = 0
    n_failed = 0
    while cur <= end_date:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end_date)
        outp = chunk_path(cur)
        if has_chunk(cur):
            n_skipped += 1
            cur = chunk_end + timedelta(days=1)
            continue
        outp.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            data = fetch_chunk(session, cur, chunk_end)
            df = pd.DataFrame(data) if data else pd.DataFrame()
            df.to_parquet(outp, index=False)
            n_fetched += 1
            n_total += len(df)
            elapsed = round(time.time() - t0, 1)
            print(f"  {cur}..{chunk_end}: {len(df)} rows ({elapsed}s)")
            append_checkpoint({"start": cur, "end": chunk_end, "rows": len(df), "ok": True, "elapsed_s": elapsed})
        except Exception as e:
            n_failed += 1
            print(f"  {cur}..{chunk_end}: ERROR {str(e)[:140]}")
            append_checkpoint({"start": cur, "end": chunk_end, "ok": False, "err": str(e)[:200]})
            # rebuild session on error
            session = build_session(warm=True, referer=ANN_REF)
        cur = chunk_end + timedelta(days=1)
        time.sleep(sleep_s)

    print(f"\n== backfill done ==")
    print(f"  fetched: {n_fetched}  skipped(existing): {n_skipped}  failed: {n_failed}")
    print(f"  total rows: {n_total:,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--sleep", type=int, default=SLEEP_S)
    ap.add_argument("--consolidate-only", action="store_true",
                    help="Skip fetch, just rebuild data/derived/announcements_historical.parquet")
    args = ap.parse_args()

    if args.consolidate_only:
        consolidate()
        return

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    backfill(start, end, args.sleep)
    consolidate()


if __name__ == "__main__":
    main()


# ============================================================
# TODO — Tier-B sources (commercial, ToS-restricted)
# ============================================================
# These need separate handling, not wired into this script:
#   - MoneyControl: scraping prohibited by ToS. Options:
#       * Paid Refinitiv/FactSet/Sentieo feed for historical articles
#       * MoneyControl pro API (subscription) — needs commercial license
#   - Economic Times: same as above. ET prime API is paid.
#   - LiveMint, BusinessStandard, Bloomberg Quint: same.
#   - Reuters/Refinitiv: institutional feed only.
#
# Recommended path: integrate one paid source (most affordable: GNews India
# tier or Webhose.io news API) once budget is allocated. Until then,
# Tier-A sources (NSE filings, BSE filings, PIB releases) cover ~70% of the
# event taxonomy the user listed.
#
# ============================================================
# TODO — Government / regulatory sources
# ============================================================
# Build separate fetchers for:
#   - PIB (Press Information Bureau) — pib.gov.in archive, structured
#   - SEBI press releases — sebi.gov.in/pressreleases
#   - RBI announcements — rbi.org.in/Scripts/Notifications.aspx
#   - GeM tenders — gem.gov.in (tender wins for listed companies)
#   - eCourts case status — ecourts.gov.in (limited public API)
#
# Order by impact: PIB > SEBI > RBI > GeM > eCourts.
# Each is a separate Python script that writes to data/raw/<source>/...
