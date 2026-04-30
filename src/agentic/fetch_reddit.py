"""
Pull recent posts from public Reddit JSON for India-investing subs.
Tag each post with mentioned NSE symbols.

No auth required (uses old.reddit.com .json endpoint with a user-agent).

Output: data/derived/reddit_feed.parquet (append-only, dedup by id)
"""
from __future__ import annotations
import time, re, hashlib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

OUT = Path("data/derived/reddit_feed.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

SUBS = [
    "IndianStockMarket",
    "IndiaInvestments",
    "IndianStreetBets",
    "DalalStreetTalks",
    "StockMarketIndia",
    "investingindia",
    "Nifty50",
]

UA = {"User-Agent": "research-bot/0.1 (contact: anonymous)"}


def load_symbols() -> set[str]:
    sm = pd.read_parquet("tmp/from_scratch_7d_run/alt2/sector_index_members.parquet")
    return set(sm["symbol"].astype(str).str.upper().unique())


def tag_symbols(text: str, syms: set[str], min_len: int = 4) -> list[str]:
    if not text:
        return []
    upper = text.upper()
    return [s for s in syms if len(s) >= min_len and re.search(rf"\b{re.escape(s)}\b", upper)]


def fetch_sub(sub: str, sort: str = "new", limit: int = 100) -> list[dict]:
    url = f"https://www.reddit.com/r/{sub}/{sort}.json?limit={limit}"
    rows = []
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
        j = r.json()
        for child in j.get("data", {}).get("children", []):
            d = child.get("data", {})
            rows.append({
                "sub": sub,
                "id": d.get("id"),
                "title": d.get("title"),
                "selftext": (d.get("selftext") or "")[:5000],
                "score": d.get("score"),
                "ups": d.get("ups"),
                "num_comments": d.get("num_comments"),
                "created_utc": d.get("created_utc"),
                "permalink": d.get("permalink"),
                "author": d.get("author"),
                "url": d.get("url"),
                "flair": d.get("link_flair_text"),
            })
    except Exception as e:
        print(f"  r/{sub} ERR: {str(e)[:120]}")
    return rows


def main() -> None:
    syms = load_symbols()
    rows = []
    for sub in SUBS:
        items = fetch_sub(sub, "new", 100)
        items += fetch_sub(sub, "hot", 50)
        items += fetch_sub(sub, "top", 50)  # daily top
        print(f"  r/{sub}: {len(items)} items")
        rows.extend(items)
        time.sleep(2.5)  # polite

    if not rows:
        print("no rows fetched")
        return
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["created_ts"] = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    text = (df["title"].fillna("") + " " + df["selftext"].fillna(""))
    df["symbols"] = text.apply(lambda t: tag_symbols(t, syms))
    df["n_symbols"] = df["symbols"].apply(len)

    if OUT.exists():
        old = pd.read_parquet(OUT)
        before = len(old)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates("id", keep="last")
        print(f"  appended: {before} → {len(df)} (delta {len(df)-before})")
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(df):,} rows)")

    # leaderboard last 24h
    recent = df[df["created_ts"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)]
    sym_counts = recent.explode("symbols").groupby("symbols").agg(
        mentions=("id", "count"),
        sum_score=("score", "sum"),
        sum_comments=("num_comments", "sum"),
    ).sort_values("mentions", ascending=False)
    print(f"\ntop reddit-mentioned symbols (last 24h):")
    print(sym_counts.head(15).to_string())


if __name__ == "__main__":
    main()
