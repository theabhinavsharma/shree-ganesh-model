"""Fetch Press Information Bureau (PIB) press releases from 2016-present.

PIB (pib.gov.in) is the official government press-release outlet — covers
every ministry's announcements, regulatory approvals, policy changes,
budget items, infrastructure approvals, defence orders, etc. These are
exactly the third-party-news events the user flagged: "even govt
announcements that the company might not be making but govt would (or
dependent parties would) will affect the price and sentiment."

URL pattern (verified May 2026):
  https://pib.gov.in/AllRelease.aspx?ydate=DD/MM/YYYY&LangID=1
  → returns the day's listing; each item links to a release page
  https://pib.gov.in/PressReleasePage.aspx?PRID=<id>

This script:
  1) Walks daily from start_date to end_date
  2) For each day, scrapes the all-releases listing
  3) For each release, fetches the full text
  4) Tags by ministry, date, title, body
  5) Symbol-matches: regex match against NSE listed company names
     (best-effort; many releases are policy-level not company-level)
  6) Writes shards: data/raw/pib_releases/YYYY/<date>.parquet
  7) Consolidates to data/derived/pib_releases.parquet

Output schema:
  pib_id, pub_date, ministry, title, body_text, symbols_matched (list)

Honest constraints:
  - PIB pages are HTML — parser must be robust to layout changes
  - Symbol-matching by company-name regex has false positives; manual
    review needed for high-stakes uses
  - Politeness: 5s sleep between requests
  - Resumes from existing shards (idempotent)

Usage:
  python3 src/agentic/fetch_pib_releases.py                   # full backfill 2016+
  python3 src/agentic/fetch_pib_releases.py --start 2024-01-01
  python3 src/agentic/fetch_pib_releases.py --consolidate-only

NOTE: scope is large (~3650 days × ~50 releases/day = ~180k releases over
10 years). At 5s/request just for fetching the listing pages, that's 5+
hours wall time. The full body fetch adds another 2x. Consider running
this overnight or in chunked batches.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT_DIR = ROOT / "data/raw/pib_releases"
CONSOLIDATED = ROOT / "data/derived/pib_releases.parquet"
CHECKPOINT = OUT_DIR / "_checkpoint.jsonl"

LISTING_URL = "https://pib.gov.in/AllRelease.aspx?ydate={ddmmyyyy}&LangID=1"
RELEASE_URL = "https://pib.gov.in/PressReleasePage.aspx?PRID={prid}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept-Language": "en-US,en;q=0.5",
}

SLEEP_S = 5


def fetch_listing(session, day: date) -> list[dict]:
    """Return list of {prid, title, ministry} for one day."""
    import requests
    from bs4 import BeautifulSoup
    url = LISTING_URL.format(ddmmyyyy=day.strftime("%d/%m/%Y"))
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    # PIB listing structure: items are anchors with PRID query param
    for a in soup.find_all("a", href=re.compile(r"PressReleasePage\.aspx\?PRID=\d+")):
        href = a.get("href", "")
        m = re.search(r"PRID=(\d+)", href)
        if not m:
            continue
        prid = m.group(1)
        title = a.get_text(strip=True)
        # ministry usually appears as text in the parent container — best-effort
        parent = a.find_parent("li") or a.find_parent("tr") or a.find_parent("div")
        ministry = ""
        if parent:
            txt = parent.get_text(" ", strip=True)
            # heuristic: ministry name is often the first segment
            mm = re.search(r"Ministry of [^\|·•]+|Department of [^\|·•]+|Cabinet [^\|·•]+", txt, re.IGNORECASE)
            if mm:
                ministry = mm.group(0).strip()
        rows.append({"prid": prid, "title": title, "ministry": ministry})
    return rows


def fetch_body(session, prid: str) -> str:
    import requests
    from bs4 import BeautifulSoup
    url = RELEASE_URL.format(prid=prid)
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # main content is in a div with id like ContentPlaceHolder1_lblBody or similar
    candidates = soup.select("#ContentPlaceHolder1_lblBody, .release_text, article, main")
    if candidates:
        return candidates[0].get_text("\n", strip=True)[:8000]
    # fallback: body text
    return soup.get_text("\n", strip=True)[:8000]


def shard_path(day: date) -> Path:
    return OUT_DIR / str(day.year) / f"{day.isoformat()}.parquet"


def has_shard(day: date) -> bool:
    return shard_path(day).exists()


def append_checkpoint(entry: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_symbol_universe() -> list[tuple[str, str]]:
    """Load (symbol, company_name) for symbol-matching."""
    candidates = [
        ROOT / "data/derived/stock_master.parquet",
        ROOT / "data/raw/stock_master.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            name_col = next((c for c in df.columns if c.lower() in ("company_name", "name", "company")), None)
            if name_col:
                return [(r["symbol"], str(r[name_col]))
                        for _, r in df.iterrows()
                        if pd.notna(r.get("symbol")) and pd.notna(r.get(name_col))]
    # Fallback: just symbols, no name match
    facts = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
    if facts.exists():
        syms = pd.read_parquet(facts, columns=["symbol"])["symbol"].unique().tolist()
        return [(s, s) for s in syms]
    return []


def match_symbols(text: str, universe: list[tuple[str, str]]) -> list[str]:
    """Best-effort regex match of company names in text."""
    if not text:
        return []
    matches = set()
    text_l = text.lower()
    for sym, name in universe:
        # require name length >= 4 chars to avoid false positives like "TCS" matching "tcs"
        if len(name) < 4:
            continue
        if name.lower() in text_l:
            matches.add(sym)
    return sorted(matches)


def consolidate(symbol_match: bool = False) -> None:
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
                frames.append(d)
        except Exception as e:
            print(f"  skip {f}: {e}")
    if not frames:
        print("no usable shards")
        return
    df = pd.concat(frames, ignore_index=True)
    if "prid" in df.columns:
        df = df.drop_duplicates(subset=["prid"], keep="first")
    print(f"consolidated: {len(df):,} rows from {len(frames)} shards")
    if symbol_match:
        print("running symbol-match against universe …")
        universe = load_symbol_universe()
        df["symbols_matched"] = df["body_text"].fillna("").apply(
            lambda t: match_symbols(t, universe))
        df["n_symbols"] = df["symbols_matched"].apply(len)
    CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CONSOLIDATED, index=False)
    print(f"wrote {CONSOLIDATED.relative_to(ROOT)}")


def backfill(start_date: date, end_date: date, sleep_s: int = SLEEP_S, fetch_bodies: bool = True) -> None:
    import requests
    print(f"== PIB backfill from {start_date} to {end_date} ==")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    cur = start_date
    n_days_done = 0
    n_days_skipped = 0
    n_releases = 0

    while cur <= end_date:
        if has_shard(cur):
            n_days_skipped += 1
            cur = cur + timedelta(days=1)
            continue
        try:
            t0 = time.time()
            listing = fetch_listing(session, cur)
            rows = []
            for item in listing:
                if fetch_bodies:
                    try:
                        body = fetch_body(session, item["prid"])
                    except Exception as e:
                        body = ""
                    time.sleep(sleep_s / 2)
                else:
                    body = ""
                rows.append({
                    "prid": item["prid"],
                    "pub_date": str(cur),
                    "ministry": item.get("ministry", ""),
                    "title": item.get("title", ""),
                    "body_text": body,
                })
            df = pd.DataFrame(rows)
            outp = shard_path(cur)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(outp, index=False)
            n_releases += len(df)
            elapsed = round(time.time() - t0, 1)
            print(f"  {cur}: {len(df)} releases ({elapsed}s)")
            append_checkpoint({"date": str(cur), "n": len(df), "ok": True, "elapsed_s": elapsed})
            n_days_done += 1
        except Exception as e:
            print(f"  {cur}: ERROR {str(e)[:140]}")
            append_checkpoint({"date": str(cur), "ok": False, "err": str(e)[:200]})
        cur = cur + timedelta(days=1)
        time.sleep(sleep_s)

    print(f"\n== PIB backfill done ==")
    print(f"  days done: {n_days_done}  skipped: {n_days_skipped}")
    print(f"  total releases: {n_releases:,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--sleep", type=int, default=SLEEP_S)
    ap.add_argument("--no-bodies", action="store_true",
                    help="Fetch listings only (faster); skip full release text")
    ap.add_argument("--consolidate-only", action="store_true")
    ap.add_argument("--match-symbols", action="store_true",
                    help="At consolidation, regex-match release text against NSE company names")
    args = ap.parse_args()

    if args.consolidate_only:
        consolidate(symbol_match=args.match_symbols)
        return

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    backfill(start, end, args.sleep, fetch_bodies=not args.no_bodies)
    consolidate(symbol_match=args.match_symbols)


if __name__ == "__main__":
    main()
