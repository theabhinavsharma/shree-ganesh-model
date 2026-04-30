"""Aggregate published broker / advisor stock recommendations from free public sources.

Sources & status:
  • Moneycontrol RSS (broker recos)        — best free signal, daily updates
  • Trendlyne broker reports               — has free tier, scrape required
  • ET Markets recommendations             — RSS available
  • HDFC Securities research               — RSS / scrape
  • Axis Direct / Kotak Neo                — paywalled
  • TradingView "Ideas" community          — public, scrapable

Outputs:
  data/derived/broker_recos.parquet — per-recommendation row:
    source, broker, symbol, action (buy/sell/hold),
    target_price, current_price, posted_date, horizon, headline, link

Then `score_broker_track_record.py` (separate) computes per-broker hit rate
over historical recos to identify which brokers actually predict.

The agentic angle:
  • Use broker_consensus as a feature: if 5+ brokers say buy and our model
    score is high → high-conviction stack
  • Use broker_disagreement as a contrarian signal: brokers split = uncertainty
  • Use broker_track_record per-broker: weight calls by historical accuracy
"""
from __future__ import annotations
import gzip
import http.cookiejar
import re
import ssl
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/broker_recos.parquet"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

# Free RSS feeds with broker recommendations
FEEDS = [
    ("moneycontrol_recos",  "https://www.moneycontrol.com/rss/results.xml"),
    ("moneycontrol_brokers","https://www.moneycontrol.com/rss/brokerage.xml"),
    ("moneycontrol_buyback","https://www.moneycontrol.com/rss/buybackoffers.xml"),
    ("moneycontrol_market_outlook","https://www.moneycontrol.com/rss/marketoutlook.xml"),
    ("et_recos",            "https://economictimes.indiatimes.com/markets/stocks/recos/rssfeeds/2146843.cms"),
    ("et_market_views",     "https://economictimes.indiatimes.com/markets/expert-view/rssfeeds/2146846.cms"),
    ("livemint_buy_sell",   "https://www.livemint.com/rss/markets/stockmarketnews"),
    ("bs_brokers",          "https://www.business-standard.com/rss/markets/broker-recommendations-10602.rss"),
]

# patterns to extract action + target price from headline/body
ACTION_PATTERNS = [
    (re.compile(r"\b(buy)\b", re.I),    "BUY"),
    (re.compile(r"\b(sell)\b", re.I),   "SELL"),
    (re.compile(r"\b(hold|maintain)\b", re.I), "HOLD"),
    (re.compile(r"\b(accumulate|add)\b", re.I), "ACCUMULATE"),
    (re.compile(r"\b(reduce|underweight)\b", re.I), "REDUCE"),
    (re.compile(r"\b(overweight|outperform)\b", re.I), "OUTPERFORM"),
    (re.compile(r"\b(underperform)\b", re.I), "UNDERPERFORM"),
]

TP_PATTERN = re.compile(r"target(?:\s*price)?[\s:]*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)", re.I)
SYMBOL_PATTERN = re.compile(r"\b([A-Z]{3,12})\b")  # rough; will need universe filter


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw


def parse_rss(raw: bytes, source: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return out
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        text = title + " " + desc
        # extract action
        action = None
        for pat, lab in ACTION_PATTERNS:
            if pat.search(text):
                action = lab
                break
        # extract target price
        tp = None
        m = TP_PATTERN.search(text)
        if m:
            try:
                tp = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
        # rough broker name extraction (look for known brokers in text)
        broker = None
        for b in ["Motilal Oswal", "ICICI Securities", "ICICI Sec", "HDFC Sec", "Kotak",
                  "Axis", "Nomura", "JPMorgan", "JP Morgan", "Goldman", "CLSA", "Macquarie",
                  "UBS", "Morgan Stanley", "Citi", "Jefferies", "Bernstein", "Emkay",
                  "Anand Rathi", "Edelweiss", "Sharekhan", "Geojit", "Prabhudas", "Ventura"]:
            if b.lower() in text.lower():
                broker = b
                break
        out.append({
            "source": source,
            "broker": broker,
            "headline": title,
            "summary": desc[:500],
            "link": link,
            "posted_date": pub,
            "action": action,
            "target_price": tp,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })
    return out


def main() -> None:
    print(f"== fetch_broker_recos  feeds={len(FEEDS)} ==")
    rows: list[dict] = []
    for src, url in FEEDS:
        try:
            raw = fetch(url)
            items = parse_rss(raw, src)
            n_with_action = sum(1 for r in items if r["action"])
            n_with_target = sum(1 for r in items if r["target_price"])
            print(f"  {src:<28} {len(items):>4} items  ({n_with_action} with action, {n_with_target} with target)")
            rows.extend(items)
        except Exception as exc:
            print(f"  {src} FAIL: {type(exc).__name__}: {str(exc)[:120]}")
        time.sleep(1.0)

    if not rows:
        print("nothing fetched")
        return

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["source", "headline"], keep="last")

    # try to extract NSE symbol from headline by intersecting against universe
    px = pd.read_parquet("data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                          columns=["symbol", "trade_date", "series"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    latest = px["trade_date"].max()
    universe = set(px[(px["trade_date"] == latest) & (px["series"] == "EQ")]["symbol"].str.upper())

    def find_symbol(text: str) -> str | None:
        for m in SYMBOL_PATTERN.findall(text.upper()):
            if m in universe:
                return m
        return None

    df["symbol"] = df.apply(lambda r: find_symbol(str(r["headline"]) + " " + str(r["summary"])), axis=1)
    n_with_sym = df["symbol"].notna().sum()
    print(f"\n  symbols matched against NSE universe: {n_with_sym} / {len(df)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        merged = pd.concat([old, df], ignore_index=True).drop_duplicates(["source", "headline"], keep="last")
    else:
        merged = df
    merged.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(merged):,} rows")

    # quick top-symbol summary
    if n_with_sym > 0:
        recent = df[df["symbol"].notna()].head(10)
        print("\n  recent recos with symbol:")
        for _, r in recent.iterrows():
            print(f"    [{r['source']}] {r['symbol']:<12} {str(r.get('action','—')):<10} "
                  f"target={r.get('target_price','—')}  | {r['headline'][:80]}")


if __name__ == "__main__":
    main()
