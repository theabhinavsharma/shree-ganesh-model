"""Finance-tuned sentiment scorer for news / reddit / youtube text.

Two outputs:
  data/derived/news_features.parquet   — per-symbol per-day:
        news_count_5d, news_sentiment_5d, news_count_30d, news_sentiment_30d,
        reddit_mentions_5d, reddit_sentiment_5d,
        youtube_mentions_5d, youtube_sentiment_5d
  data/derived/macro_sentiment.parquet — per-day market-wide sentiment:
        global_macro_sent, domestic_macro_sent, global_count, domestic_count,
        rate_hawkish_score, rate_dovish_score, oil_sentiment, usdinr_sentiment

Lexicon: hand-curated Loughran-McDonald-flavored positive / negative
finance words + India-specific (RBI, SEBI, GST, FII/DII flow, monsoon, etc)
+ global macro (Fed, FOMC, recession, oil, dollar, ECB).

Scoring is bag-of-lemmas: for each text, count positives - negatives,
normalize by sqrt(token count). Negation handled via "not/no/never" within
3-word window flipping the polarity of the next finance term.

No external deps. Pure Python + pandas.
"""
from __future__ import annotations
import re
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
NEWS_RAW = ROOT / "data/derived/news_feed.parquet"
REDDIT_RAW = ROOT / "data/derived/reddit_feed.parquet"
YT_RAW = ROOT / "data/derived/youtube_videos.parquet"

OUT_NEWS = ROOT / "data/derived/news_features.parquet"
OUT_MACRO = ROOT / "data/derived/macro_sentiment.parquet"

# ─────────────────────────────────────────────────────────────────────
# Finance-tuned lexicon (Loughran-McDonald subset + India context)
# ─────────────────────────────────────────────────────────────────────
POS_FINANCE = {
    # earnings / growth
    "beat", "beats", "exceeded", "surpassed", "outperform", "outperformed",
    "record", "strong", "robust", "growth", "expand", "expansion",
    "profitable", "profit", "gain", "gains", "rally", "surge", "surged",
    "rise", "rose", "jump", "jumped", "climb", "climbed", "soar", "soared",
    "boost", "boosted", "upgrade", "upgraded", "bullish",
    "buy", "accumulate", "overweight", "outperform", "raises", "raised",
    # corporate actions
    "acquire", "acquisition", "merger", "buyback", "dividend", "bonus", "split",
    "expansion", "capex", "orderbook", "contract", "won", "deal",
    # results
    "beat estimates", "above estimates", "tops", "topped", "pat surge",
    "margin expansion", "qoq growth", "yoy growth",
    # India-specific positive
    "fii buying", "dii buying", "monsoon normal", "good monsoon",
    "gst growth", "tax buoyancy", "infra push", "pli scheme",
    "make in india", "lower repo", "rate cut",
    "psu rally", "midcap rally", "smallcap rally",
}

NEG_FINANCE = {
    # losses / decline
    "miss", "missed", "missed estimates", "below estimates", "weak",
    "loss", "losses", "decline", "declined", "drop", "dropped", "fall",
    "fell", "plunge", "plunged", "tumble", "tumbled", "slump", "slumped",
    "crash", "crashed", "bearish", "downgrade", "downgraded",
    "underperform", "sell", "underweight", "cut", "cuts",
    # quality concerns
    "fraud", "scam", "irregularity", "probe", "investigation", "raid",
    "lawsuit", "fine", "penalty", "default", "bankruptcy", "insolvency",
    "delisting", "warning", "concern", "concerned",
    # margin / cost
    "margin compression", "cost overrun", "demand weak", "slowdown",
    "recession", "contraction", "headwind", "headwinds",
    # India-specific negative
    "fii selling", "dii selling", "rupee weak", "rupee fall",
    "rate hike", "repo hike", "monsoon deficit", "drought",
    "sebi probe", "ed raid", "income tax raid", "circular issued",
    "regulator concern", "trade ban",
}

# Negation triggers — within 3 tokens, they flip polarity
NEGATIONS = {"not", "no", "never", "without", "n't", "neither", "nor"}

# Global macro — applies at MARKET level, not per-symbol
GLOBAL_MACRO_POS = {
    "fed cut", "fed cuts", "rate cut", "dovish", "soft landing",
    "easing", "qe", "stimulus", "fed pivot", "fomc dovish",
    "us cpi cools", "inflation cools", "inflation eases",
    "oil down", "crude down", "dollar weak", "dxy down",
    "china stimulus", "ecb cut",
}
GLOBAL_MACRO_NEG = {
    "fed hike", "fed hikes", "rate hike", "hawkish", "hard landing",
    "tightening", "qt", "fomc hawkish",
    "us cpi hot", "inflation sticky", "inflation hot",
    "oil up", "crude rally", "dollar strong", "dxy up", "dxy rally",
    "geopolitical", "war", "conflict", "tariff", "trade war",
    "recession risk", "recession fear", "default risk",
}

# Domestic macro — India-specific, applies at MARKET level
DOMESTIC_MACRO_POS = {
    "rbi cut", "rbi pause", "repo cut", "lower repo",
    "fii buying", "dii buying", "fii inflow", "fpi inflow",
    "monsoon normal", "good monsoon", "above normal",
    "gst growth", "tax buoyancy", "fiscal deficit lower",
    "credit growth", "infra push", "pli scheme", "budget positive",
    "make in india", "atma nirbhar", "psu rally",
}
DOMESTIC_MACRO_NEG = {
    "rbi hike", "repo hike", "rate hike",
    "fii selling", "dii selling", "fpi outflow", "fii outflow",
    "rupee weak", "rupee fall", "rupee record low",
    "monsoon deficit", "drought", "el nino",
    "sebi probe", "sebi action", "ed raid", "income tax raid",
    "fiscal deficit higher", "current account deficit",
    "trade deficit", "manufacturing slowdown",
}


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9' \-&]", " ", text)
    return [t for t in text.split() if t]


def _phrase_hits(text: str, phrases: set[str]) -> int:
    """Count how many phrase patterns appear in text (case-insensitive)."""
    if not isinstance(text, str):
        return 0
    t = text.lower()
    return sum(1 for p in phrases if p in t)


def score_text(text: str) -> float:
    """Return a sentiment score in [-1, +1].

    Pos / neg word counts with negation flip. Normalized by sqrt(n_tokens)
    so long articles don't dominate."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    pos, neg = 0, 0
    # phrase scoring (multi-word terms in lexicons need substring match)
    pos += _phrase_hits(text, {p for p in POS_FINANCE if " " in p})
    neg += _phrase_hits(text, {p for p in NEG_FINANCE if " " in p})

    # single-word with negation window
    for i, tok in enumerate(tokens):
        if tok in POS_FINANCE and " " not in tok:
            window = tokens[max(0, i - 3): i]
            if any(w in NEGATIONS for w in window):
                neg += 1
            else:
                pos += 1
        elif tok in NEG_FINANCE and " " not in tok:
            window = tokens[max(0, i - 3): i]
            if any(w in NEGATIONS for w in window):
                pos += 1
            else:
                neg += 1

    net = pos - neg
    if net == 0:
        return 0.0
    norm = max(np.sqrt(len(tokens)), 1.0)
    return float(np.tanh(net / norm * 2))  # tanh keeps it in [-1,+1]


def macro_score_text(text: str, lexicon_pos: set[str], lexicon_neg: set[str]) -> tuple[int, int]:
    if not isinstance(text, str):
        return 0, 0
    return _phrase_hits(text, lexicon_pos), _phrase_hits(text, lexicon_neg)


def get_universe_symbols() -> list[str]:
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "series", "avg_traded_value_20d"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    latest = df["trade_date"].max()
    snap = df[(df["trade_date"] == latest) & (df["series"] == "EQ")]
    snap = snap[snap["avg_traded_value_20d"] >= 1e6]  # ≥ ₹0.1cr/day
    return sorted(snap["symbol"].astype(str).unique().tolist())


def aggregate_news_per_symbol() -> pd.DataFrame:
    """For each liquid symbol, count + sentiment over 5d / 30d windows from
    news_feed, reddit_feed, youtube_videos."""
    syms = get_universe_symbols()
    sym_lower = {s: s.lower() for s in syms}
    today = pd.Timestamp.utcnow()
    cut_5d = today - pd.Timedelta(days=5)
    cut_30d = today - pd.Timedelta(days=30)

    rows = []

    # NEWS (RSS feeds)
    if NEWS_RAW.exists():
        n = pd.read_parquet(NEWS_RAW)
        n["pub_ts"] = pd.to_datetime(n["pub_ts"], errors="coerce", utc=True)
        n["text"] = (n["title"].fillna("") + " " + n["desc"].fillna(""))
        n["sent"] = n["text"].map(score_text)
        n_5d = n[n["pub_ts"] >= cut_5d]
        n_30d = n[n["pub_ts"] >= cut_30d]
    else:
        n_5d = n_30d = pd.DataFrame()

    # REDDIT
    if REDDIT_RAW.exists():
        r = pd.read_parquet(REDDIT_RAW)
        r["created_utc"] = pd.to_numeric(r["created_utc"], errors="coerce")
        r["pub_ts"] = pd.to_datetime(r["created_utc"], unit="s", errors="coerce", utc=True)
        r["text"] = (r["title"].fillna("") + " " + r["selftext"].fillna(""))
        r["sent"] = r["text"].map(score_text)
        r_5d = r[r["pub_ts"] >= cut_5d]
    else:
        r_5d = pd.DataFrame()

    # YOUTUBE
    if YT_RAW.exists():
        y = pd.read_parquet(YT_RAW)
        y["published_ts"] = pd.to_datetime(y["published_ts"], errors="coerce", utc=True)
        y["text"] = y["title"].fillna("")
        y["sent"] = y["text"].map(score_text)
        y_5d = y[y["published_ts"] >= cut_5d]
    else:
        y_5d = pd.DataFrame()

    for sym in syms:
        sym_l = sym_lower[sym]
        pat = re.compile(rf"\b{re.escape(sym_l)}\b", re.I)

        def hit(df):
            if df.empty or "text" not in df.columns:
                return df.iloc[0:0] if not df.empty else df
            return df[df["text"].str.contains(pat, na=False)]

        n5 = hit(n_5d) if len(n_5d) else pd.DataFrame()
        n30 = hit(n_30d) if len(n_30d) else pd.DataFrame()
        r5 = hit(r_5d) if len(r_5d) else pd.DataFrame()
        y5 = hit(y_5d) if len(y_5d) else pd.DataFrame()

        rows.append({
            "symbol": sym,
            "as_of": today.normalize().date(),
            "news_count_5d": int(len(n5)),
            "news_count_30d": int(len(n30)),
            "news_sentiment_5d": float(n5["sent"].mean()) if len(n5) else 0.0,
            "news_sentiment_30d": float(n30["sent"].mean()) if len(n30) else 0.0,
            "reddit_mentions_5d": int(len(r5)),
            "reddit_sentiment_5d": float(r5["sent"].mean()) if len(r5) else 0.0,
            "youtube_mentions_5d": int(len(y5)),
            "youtube_sentiment_5d": float(y5["sent"].mean()) if len(y5) else 0.0,
        })
    out = pd.DataFrame(rows)
    return out


def aggregate_macro() -> pd.DataFrame:
    """Daily market-wide global + domestic macro sentiment."""
    today = pd.Timestamp.utcnow()
    cut_5d = today - pd.Timedelta(days=5)

    g_pos = g_neg = 0
    d_pos = d_neg = 0
    n_total_g = n_total_d = 0
    rate_hawk = rate_dove = 0
    oil_pos = oil_neg = 0
    inr_pos = inr_neg = 0

    sources = []
    if NEWS_RAW.exists():
        n = pd.read_parquet(NEWS_RAW)
        n["pub_ts"] = pd.to_datetime(n["pub_ts"], errors="coerce", utc=True)
        n["text"] = (n["title"].fillna("") + " " + n["desc"].fillna(""))
        sources.append(n[n["pub_ts"] >= cut_5d])
    if REDDIT_RAW.exists():
        r = pd.read_parquet(REDDIT_RAW)
        r["created_utc"] = pd.to_numeric(r["created_utc"], errors="coerce")
        r["pub_ts"] = pd.to_datetime(r["created_utc"], unit="s", errors="coerce", utc=True)
        r["text"] = (r["title"].fillna("") + " " + r["selftext"].fillna(""))
        sources.append(r[r["pub_ts"] >= cut_5d])
    if YT_RAW.exists():
        y = pd.read_parquet(YT_RAW)
        y["published_ts"] = pd.to_datetime(y["published_ts"], errors="coerce", utc=True)
        y["text"] = y["title"].fillna("")
        sources.append(y[y["published_ts"] >= cut_5d])

    for src in sources:
        if not len(src):
            continue
        for txt in src["text"]:
            gp, gn = macro_score_text(txt, GLOBAL_MACRO_POS, GLOBAL_MACRO_NEG)
            dp, dn = macro_score_text(txt, DOMESTIC_MACRO_POS, DOMESTIC_MACRO_NEG)
            g_pos += gp; g_neg += gn
            d_pos += dp; d_neg += dn
            if gp + gn > 0:
                n_total_g += 1
            if dp + dn > 0:
                n_total_d += 1
            # focused sub-themes
            t = txt.lower() if isinstance(txt, str) else ""
            if any(p in t for p in ("rate hike", "fed hike", "rbi hike", "hawkish", "tightening")):
                rate_hawk += 1
            if any(p in t for p in ("rate cut", "fed cut", "rbi cut", "dovish", "easing")):
                rate_dove += 1
            if any(p in t for p in ("oil up", "crude rally", "brent up")):
                oil_neg += 1  # higher oil = bad for India
            if any(p in t for p in ("oil down", "crude down", "brent down")):
                oil_pos += 1
            if any(p in t for p in ("rupee weak", "rupee fall", "inr weak")):
                inr_neg += 1
            if any(p in t for p in ("rupee strong", "rupee gain", "inr strong")):
                inr_pos += 1

    def norm(p, n):
        tot = p + n
        return (p - n) / max(tot, 1)

    out = pd.DataFrame([{
        "as_of": today.normalize().date(),
        "global_macro_sent": round(norm(g_pos, g_neg), 3),
        "domestic_macro_sent": round(norm(d_pos, d_neg), 3),
        "global_count": int(n_total_g),
        "domestic_count": int(n_total_d),
        "rate_hawkish_score": int(rate_hawk),
        "rate_dovish_score": int(rate_dove),
        "oil_sentiment": round(norm(oil_pos, oil_neg), 3),
        "usdinr_sentiment": round(norm(inr_pos, inr_neg), 3),
    }])
    return out


def append_parquet(out_path: Path, df_new: pd.DataFrame, dedupe_keys: list[str]) -> int:
    if df_new.empty:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        old = pd.read_parquet(out_path)
        merged = pd.concat([old, df_new], ignore_index=True)
        merged = merged.drop_duplicates(dedupe_keys, keep="last")
        merged.to_parquet(out_path, index=False)
        return len(merged) - len(old)
    df_new.to_parquet(out_path, index=False)
    return len(df_new)


def main() -> None:
    print("== sentiment scoring (finance lexicon) ==")
    news_df = aggregate_news_per_symbol()
    delta = append_parquet(OUT_NEWS, news_df, ["symbol", "as_of"])
    print(f"  news_features → {len(news_df)} rows for today, +{delta} new in parquet")

    macro_df = aggregate_macro()
    delta = append_parquet(OUT_MACRO, macro_df, ["as_of"])
    print(f"  macro_sentiment → {macro_df.iloc[0].to_dict()}")
    print(f"  +{delta} new macro rows in parquet")

    # quick top-coverage report
    n_with_news = (news_df["news_count_5d"] > 0).sum()
    n_with_reddit = (news_df["reddit_mentions_5d"] > 0).sum()
    print(f"\n  symbols mentioned in news (5d):  {n_with_news} / {len(news_df)}")
    print(f"  symbols mentioned on reddit:     {n_with_reddit} / {len(news_df)}")
    if n_with_news > 0:
        top_pos = news_df.nlargest(5, "news_sentiment_5d")[["symbol", "news_count_5d", "news_sentiment_5d"]]
        top_neg = news_df.nsmallest(5, "news_sentiment_5d")[["symbol", "news_count_5d", "news_sentiment_5d"]]
        print("  top positive:")
        print(top_pos.to_string(index=False))
        print("  top negative:")
        print(top_neg.to_string(index=False))


if __name__ == "__main__":
    main()
