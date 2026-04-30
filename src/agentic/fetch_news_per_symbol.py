"""Per-symbol news enrichment via Google News RSS.

The headline RSS feeds in fetch_news_rss.py give us top-of-day market headlines
(~300 items/day) → only ~4% of the 2,137-stock universe gets a mention.
This script lifts coverage to 50%+ by querying Google News RSS *per symbol*.

For each symbol in the active universe (top-N by liquidity ∪ current top picks):
  GET https://news.google.com/rss/search?q="{SYMBOL}"+stock+India&hl=en-IN&gl=IN

Each result is appended to news_feed.parquet with source="google_news_<SYM>"
so the existing score_sentiment.py pipeline picks it up automatically.

Stdlib + pandas only. No API key. ~1.5s/symbol pacing.
Output: appends to data/derived/news_feed.parquet
"""
from __future__ import annotations
import argparse
import gzip
import http.cookiejar
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
NEWS_OUT = ROOT / "data/derived/news_feed.parquet"
LIVE_LONG = ROOT / "tmp/from_scratch_7d_run/v3_live_top100.csv"
LIVE_SHORT = ROOT / "tmp/from_scratch_7d_run/short_live_top100.csv"
MH_LIVE = ROOT / "tmp/from_scratch_7d_run/multi_horizon_top.csv"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
DELAY = 1.5
TIMEOUT = 20


def _opener() -> urllib.request.OpenerDirector:
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )


def _get(opener: urllib.request.OpenerDirector, url: str) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/atom+xml,application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    with opener.open(req, timeout=TIMEOUT) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw


def parse_rss(xml_bytes: bytes, source: str) -> list[dict]:
    """RSS 2.0 + Atom feeds. Google News is RSS 2.0."""
    out: list[dict] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out
    # RSS 2.0
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        guid = (item.findtext("guid") or link or title)
        if not title:
            continue
        out.append({
            "source": source,
            "title": title,
            "link": link,
            "desc": desc,
            "pub": pub,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "pub_ts": _parse_pub(pub),
            "item_id": f"{source}:{guid[:200]}",
        })
    return out


def _parse_pub(s: str) -> str:
    """Best-effort parse of RFC 822 / ISO date strings → ISO UTC."""
    if not s:
        return ""
    fmts = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    return ""


def get_active_universe(top_n_liquid: int = 300) -> list[str]:
    """Top-N by ADV ∪ current top picks (long/short/multi-horizon)."""
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "series", "avg_traded_value_20d"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    latest = df["trade_date"].max()
    snap = df[(df["trade_date"] == latest) & (df["series"] == "EQ")]
    snap = snap.sort_values("avg_traded_value_20d", ascending=False).head(top_n_liquid)
    syms = set(snap["symbol"].astype(str).tolist())
    for f in (LIVE_LONG, LIVE_SHORT, MH_LIVE):
        if f.exists():
            try:
                more = pd.read_csv(f)["symbol"].astype(str).tolist()
                syms.update(more[:50])  # top 50 from each
            except Exception:
                pass
    return sorted(syms)


def fetch_for_symbol(opener, sym: str) -> list[dict]:
    q = urllib.parse.quote_plus(f'"{sym}" stock India')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        raw = _get(opener, url)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  [{sym}] ERR {type(exc).__name__}: {str(exc)[:100]}")
        return []
    items = parse_rss(raw, source=f"google_news_{sym}")
    return items


def append_news(items: list[dict]) -> int:
    if not items:
        return 0
    df_new = pd.DataFrame(items)
    NEWS_OUT.parent.mkdir(parents=True, exist_ok=True)
    if NEWS_OUT.exists():
        old = pd.read_parquet(NEWS_OUT)
        # ensure compatible columns
        all_cols = sorted(set(old.columns).union(df_new.columns))
        for c in all_cols:
            if c not in old.columns:
                old[c] = None
            if c not in df_new.columns:
                df_new[c] = None
        merged = pd.concat([old[all_cols], df_new[all_cols]], ignore_index=True)
        merged = merged.drop_duplicates(["item_id"], keep="last")
        merged.to_parquet(NEWS_OUT, index=False)
        return len(merged) - len(old)
    df_new.to_parquet(NEWS_OUT, index=False)
    return len(df_new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n-liquid", type=int, default=300)
    ap.add_argument("--max-symbols", type=int, default=400, help="hard cap on symbols this run")
    args = ap.parse_args()

    syms = get_active_universe(args.top_n_liquid)[:args.max_symbols]
    print(f"== fetch_news_per_symbol  active_universe={len(syms)} ==")
    opener = _opener()
    all_items: list[dict] = []
    started = time.time()
    for i, sym in enumerate(syms, 1):
        items = fetch_for_symbol(opener, sym)
        if items:
            all_items.extend(items)
        if i % 25 == 0:
            elapsed = time.time() - started
            eta_min = (len(syms) - i) * (elapsed / i) / 60
            print(f"  progress {i}/{len(syms)}  items={len(all_items)}  elapsed={elapsed:.0f}s  eta={eta_min:.0f}min", flush=True)
            # checkpoint
            delta = append_news(all_items)
            print(f"    checkpoint +{delta} items appended", flush=True)
            all_items = []
        time.sleep(DELAY)

    delta = append_news(all_items)
    print(f"\n  final +{delta} items appended → {NEWS_OUT}")


if __name__ == "__main__":
    main()
