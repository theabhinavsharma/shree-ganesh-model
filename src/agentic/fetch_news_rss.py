"""
Pull RSS feeds for Indian-market news and tag each item with mentioned NSE symbols.

Sources (free, public RSS):
  - Moneycontrol top news + business
  - ET Markets top stories
  - LiveMint markets
  - Business Standard markets
  - NDTV Profit markets

Output: data/derived/news_feed.parquet (append-only, dedup by (source, link))
"""
from __future__ import annotations
import re, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import requests

OUT = Path("data/derived/news_feed.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

FEEDS = [
    ("moneycontrol_top",   "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("moneycontrol_biz",   "https://www.moneycontrol.com/rss/business.xml"),
    ("moneycontrol_mkt",   "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("moneycontrol_buzz",  "https://www.moneycontrol.com/rss/buzzingstocks.xml"),
    ("moneycontrol_results","https://www.moneycontrol.com/rss/results.xml"),
    ("et_markets",         "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("et_stocks",          "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("et_news",            "https://economictimes.indiatimes.com/news/rssfeeds/1715249553.cms"),
    ("livemint_markets",   "https://www.livemint.com/rss/markets"),
    ("livemint_companies", "https://www.livemint.com/rss/companies"),
    ("bs_markets",         "https://www.business-standard.com/rss/markets-106.rss"),
    ("ndtvprofit_markets", "https://www.ndtvprofit.com/feed"),
]

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def parse_rss(text: str, source: str) -> list[dict]:
    rows = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return rows
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        rows.append({"source": source, "title": title, "link": link, "desc": desc, "pub": pub})
    return rows


def load_symbols() -> set[str]:
    sm = pd.read_parquet("tmp/from_scratch_7d_run/alt2/sector_index_members.parquet")
    return set(sm["symbol"].astype(str).str.upper().unique())


def tag_symbols(text: str, syms: set[str], min_len: int = 4) -> list[str]:
    """Find symbols mentioned in text (uppercase, exact word boundary)."""
    if not text:
        return []
    found = []
    upper_text = text.upper()
    for s in syms:
        if len(s) < min_len:
            continue
        if re.search(rf"\b{re.escape(s)}\b", upper_text):
            found.append(s)
    return found


def main() -> None:
    syms = load_symbols()
    print(f"loaded {len(syms)} symbols for tagging")
    rows = []
    for source, url in FEEDS:
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            items = parse_rss(r.text, source)
            print(f"  {source}: {len(items)} items")
            rows.extend(items)
        except Exception as e:
            print(f"  {source}: ERR {str(e)[:120]}")
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows fetched")
        return

    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    df["pub_ts"] = pd.to_datetime(df["pub"], errors="coerce", utc=True)
    df["item_id"] = df.apply(lambda r: hashlib.md5(f"{r['source']}|{r['link']}".encode()).hexdigest(), axis=1)
    full_text = (df["title"].fillna("") + " " + df["desc"].fillna(""))
    df["symbols"] = full_text.apply(lambda t: tag_symbols(t, syms))
    df["n_symbols"] = df["symbols"].apply(len)

    if OUT.exists():
        old = pd.read_parquet(OUT)
        before = len(old)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates("item_id", keep="first")
        print(f"  appended: {before} → {len(df)} (delta {len(df)-before})")
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(df):,} rows)")
    print(f"\ntop tagged symbols (last 24h):")
    recent = df[df["pub_ts"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)]
    sym_counts = recent.explode("symbols").groupby("symbols").size().sort_values(ascending=False)
    print(sym_counts.head(15).to_string())


if __name__ == "__main__":
    main()
