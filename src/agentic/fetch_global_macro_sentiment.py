"""Macro-level sentiment — country, market, commodity, central-bank headlines.

Uses Google News RSS for macro/policy queries (NOT per-stock — explicitly aggregate):
  • "RBI monetary policy"
  • "Indian economy GDP"
  • "FII flows India"
  • "Crude oil prices"
  • "Gold prices outlook"
  • "Fed rate decision"
  • "China economy slowdown"
  • "Global recession risk"
  • "Indian rupee"
  • "Geopolitical tension"

For each macro topic, scores latest 30 headlines with finance-lexicon polarity.
Output: data/derived/global_macro_sentiment.parquet
  cols: as_of, topic, headline_count_24h, sentiment_24h, sentiment_7d, sentiment_30d
"""
from __future__ import annotations
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/global_macro_sentiment.parquet"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

TOPICS = {
    "rbi_policy":          '"RBI monetary policy" OR "RBI repo rate" OR "MPC India"',
    "indian_economy":      '"Indian economy" OR "India GDP" OR "India IIP"',
    "fii_flow":            '"FII flow India" OR "FPI flow India" OR "foreign portfolio India"',
    "crude_oil":           '"crude oil price" OR "Brent crude" OR "WTI crude"',
    "gold":                '"gold price" OR "gold outlook"',
    "fed_policy":          '"Federal Reserve rate" OR "Fed FOMC" OR "Powell hawkish"',
    "china_econ":          '"China economy" OR "China GDP" OR "China stimulus"',
    "recession_risk":      '"recession risk" OR "global slowdown" OR "yield curve inversion"',
    "rupee":               '"Indian rupee" OR "USDINR" OR "rupee depreciate"',
    "geopolitics":         '"geopolitical tension" OR "Russia Ukraine" OR "Middle East war"',
    "india_inflation":     '"India CPI inflation" OR "India WPI inflation"',
    "india_credit":        '"India credit growth" OR "RBI credit India"',
    "india_monsoon":       '"India monsoon" OR "IMD monsoon"',
    "earnings_outlook":    '"India earnings outlook" OR "Q4 results India"',
    "global_liquidity":    '"global liquidity" OR "central bank balance sheet"',
}

# Finance-tuned polarity lexicon (small, deterministic)
POS = {
    "surge", "rally", "rebound", "jump", "soar", "climb", "advance", "rise", "gain",
    "boost", "bullish", "positive", "growth", "improve", "strong", "robust", "expand",
    "upgrade", "beat", "exceed", "record", "high", "outperform", "lifts", "raised",
    "ease", "cool", "tame", "boon", "stimulus", "cut", "easing", "dovish",
    "upside", "buy", "rebound", "recovery", "uptick", "demand",
}
NEG = {
    "plunge", "crash", "tumble", "slump", "fall", "drop", "decline", "weak", "bearish",
    "negative", "shrink", "contract", "downgrade", "miss", "disappoint", "low",
    "underperform", "cut", "slash", "concern", "fear", "worry", "risk", "selloff",
    "rout", "tension", "war", "sanction", "inflation", "hawkish", "tighten",
    "recession", "slowdown", "downturn", "pressure", "headwind", "default", "crisis",
}


def google_news_rss(query: str, lang: str = "en-IN") -> list[dict]:
    """Pull headlines for a query from news.google.com/rss."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl=IN&ceid=IN:en"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
    except Exception as e:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        try:
            ts = pd.Timestamp(pub).tz_convert("UTC") if "+0" in pub or "GMT" in pub else pd.Timestamp(pub, tz="UTC")
        except Exception:
            ts = pd.Timestamp(datetime.now(timezone.utc))
        items.append({"title": title, "ts": ts})
    return items


def score(text: str) -> float:
    t = text.lower()
    words = re.findall(r"[a-z]+", t)
    pos_hits = sum(1 for w in words if w in POS)
    neg_hits = sum(1 for w in words if w in NEG)
    total = pos_hits + neg_hits
    if total == 0:
        return 0.0
    return (pos_hits - neg_hits) / max(total, 3)


def main() -> None:
    print("== fetch_global_macro_sentiment ==")
    now = pd.Timestamp(datetime.now(timezone.utc))
    rows = []
    for topic, q in TOPICS.items():
        items = google_news_rss(q)
        if not items:
            print(f"  {topic:<22}  0 headlines")
            continue
        df = pd.DataFrame(items)
        df["score"] = df["title"].apply(score)
        df["age_hr"] = (now - df["ts"]).dt.total_seconds() / 3600
        n_24 = (df["age_hr"] <= 24).sum()
        s_24 = df.loc[df["age_hr"] <= 24, "score"].mean() if n_24 else 0
        s_7d = df.loc[df["age_hr"] <= 24*7, "score"].mean() if (df["age_hr"] <= 24*7).any() else 0
        s_30d = df.loc[df["age_hr"] <= 24*30, "score"].mean() if (df["age_hr"] <= 24*30).any() else 0
        rows.append({
            "as_of": now.tz_convert(None),
            "topic": topic,
            "headline_count_24h": int(n_24),
            "sentiment_24h": float(s_24 if pd.notna(s_24) else 0),
            "sentiment_7d":  float(s_7d if pd.notna(s_7d) else 0),
            "sentiment_30d": float(s_30d if pd.notna(s_30d) else 0),
            "n_total": len(df),
        })
        print(f"  {topic:<22}  {len(df):>3} hl  s24={s_24:+.2f}  s7={s_7d:+.2f}")
        time.sleep(0.5)

    if not rows:
        print("no rows")
        return
    out = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        out = pd.concat([prev, out], ignore_index=True)
        out["as_of"] = pd.to_datetime(out["as_of"])
        out = out.sort_values(["topic", "as_of"]).drop_duplicates(["topic", "as_of"], keep="last")
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out)} rows  ({out['topic'].nunique()} topics)")


if __name__ == "__main__":
    main()
