"""Build TWO new sets of polarity-aware event features:

  PATH 1: price-reaction polarity (the market itself tells us if event was good)
    For each announcement, observe day-1 (next trading day) close-to-close return.
    Polarity = +1 if next-day return > +1σ, -1 if < -1σ, 0 otherwise.
    Aggregate per-stock per-date as rolling-window signed-counts.

  PATH 3: refined regex polarity on `attchmntText` (the actual PDF-extracted text)
    The current catalyst_tagger only matches against `desc` field which is a thin
    metadata header (avg 27 chars, mostly "General Updates" / "Analyst Meet").
    The substantive content is in `attchmntText` (avg 151 chars, max 3040, real
    content) — and we never matched against it for polarity keywords.

    New polarity dictionary (signed):
      POSITIVE_KEYWORDS: beat, exceed, ahead of, raised, upgraded, approved,
                         favourable, sanctioned, allotted, awarded, bagged,
                         record-high, all-time high, dividend declared, profit grew
      NEGATIVE_KEYWORDS: miss, below, cut, downgrade, rejected, adverse, default,
                         penalty, sebi notice, decline, loss, qualified opinion,
                         resignation, fraud, insolvency, regulatory action

  Output:
    data/derived/event_polarity_features.parquet
      - per (symbol, trade_date), the 8 new polarity features
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Code/Zoom")
ANN_TAGGED = ROOT / "tmp/from_scratch_7d_run/alt/announcements_tagged.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/event_polarity_features.parquet"

# ──────────────────────────────────────────────────────────────────────────
# PATH 3: refined-regex polarity dictionary
# ──────────────────────────────────────────────────────────────────────────

POSITIVE_PATTERNS = [
    (r"\bbeat(s|en)?\s+(estimates?|street|expectations?|consensus)\b", 1.0),
    (r"\bahead of\s+(estimates?|street|expectations?)\b", 0.8),
    (r"\bexceed(s|ed|ing)?\b", 0.6),
    (r"\b(record|all[\s-]?time)\s+(high|profit|revenue|sales|orders?)\b", 0.8),
    (r"\bhighest ever\b", 0.7),
    (r"\b(raised|raising|hiked|hiking)\s+(guidance|outlook|target)\b", 0.9),
    (r"\b(upgrade|upgraded|up-?graded)\b", 0.7),
    (r"\b(approved|sanctioned|allotted|awarded|bagged|secured|won)\b", 0.5),
    (r"\b(strong|robust|exceptional|outstanding)\b", 0.3),
    (r"\b(dividend declared|interim dividend)\b", 0.4),
    (r"\bprofit (grew|increased|surged|jumped|rose)\b", 0.6),
    (r"\bmargin\s+(expansion|expanded|improved|improvement)\b", 0.5),
    (r"\b(qip\s+priced|qip\s+at\s+a\s+premium)\b", 0.4),  # non-dilutive QIP
    (r"\b(buy[\s-]?back|share repurchase)\s+(approved|announced)\b", 0.7),
]

NEGATIVE_PATTERNS = [
    (r"\bmiss(es|ed)\s+(estimates?|street|expectations?|consensus)\b", -1.0),
    (r"\b(below|short of)\s+(estimates?|street|expectations?)\b", -0.8),
    (r"\b(cut|cutting|lowered|reduce[d]?|trimmed)\s+(guidance|outlook|target|forecast)\b", -0.9),
    (r"\b(downgrade|downgraded|down-?graded)\b", -0.7),
    (r"\b(rejected|disallowed|denied)\b", -0.6),
    (r"\b(adverse|unfavou?rable)\s+(opinion|order|judgement|ruling)\b", -0.7),
    (r"\b(default|defaulted|insolvency|nclt admission|cirp|liquidation)\b", -0.9),
    (r"\b(penalty|fine|show[\s-]?cause|sebi notice|disgorgement|debarred)\b", -0.7),
    (r"\b(resignation|resigned|stepped down).*(md|ceo|cfo|director|chairman)\b", -0.6),
    (r"\b(fraud|misappropriation|forgery|misrepresentation)\b", -1.0),
    (r"\b(loss|decline|drop|fall|weak|deteriorat|slumped|plunged)\b", -0.3),
    (r"\bqualified\s+(audit\s+)?opinion\b", -0.7),
    (r"\b(forensic|whistleblower|cbi raid|ed raid|search warrant)\b", -0.9),
    (r"\b(rights issue\s+at\s+(deep\s+)?discount|preferential.*at.*discount)\b", -0.5),
]


def refined_polarity(text: str) -> float:
    """Score a single announcement text via signed regex matches."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    score = 0.0
    for pat, w in POSITIVE_PATTERNS:
        if re.search(pat, t):
            score += w
    for pat, w in NEGATIVE_PATTERNS:
        if re.search(pat, t):
            score += w
    # cap to [-1.5, +1.5] to keep extreme stacking from dominating
    return max(-1.5, min(1.5, score))


# ──────────────────────────────────────────────────────────────────────────
# PATH 1: price-reaction polarity
# ──────────────────────────────────────────────────────────────────────────

def price_reaction_polarity(ann: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """For each announcement, look up the next-day close-to-close return for that
    symbol. Encode polarity as discrete: +1 if return > +1σ (over 60d), -1 if < -1σ,
    else continuous-scaled return.

    Uses the trading-day-aligned `ann_date` as event day. Looks at `ann_date+1` close
    relative to `ann_date` close. (For events filed post-close, ann_date is already
    advanced to the next trading day, so we measure that day's reaction.)
    """
    px = prices.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    # next-day return for each (symbol, trade_date)
    px["next_close"] = px.groupby("symbol")["close"].shift(-1)
    px["next_day_ret"] = px["next_close"] / px["close"] - 1
    # rolling 60d std for normalization
    px["ret_std_60d"] = px.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(60, min_periods=20).std())

    ann2 = ann.copy()
    ann2["ann_date"] = pd.to_datetime(ann2["ann_date"])
    ann2 = ann2.merge(px[["symbol", "trade_date", "next_day_ret", "ret_std_60d"]],
                        left_on=["symbol", "ann_date"],
                        right_on=["symbol", "trade_date"], how="left")
    # discrete polarity: +1 if next-day return > +1σ; -1 if < -1σ; 0 otherwise
    ann2["px_polarity"] = 0.0
    pos = ann2["next_day_ret"] > ann2["ret_std_60d"]
    neg = ann2["next_day_ret"] < -ann2["ret_std_60d"]
    ann2.loc[pos, "px_polarity"] = 1.0
    ann2.loc[neg, "px_polarity"] = -1.0
    # also continuous (raw next-day return, capped)
    ann2["px_polarity_raw"] = ann2["next_day_ret"].fillna(0).clip(-0.10, 0.10)
    return ann2[["symbol", "ann_date", "px_polarity", "px_polarity_raw"]]


def main() -> None:
    print("== build_event_polarity (path 1 + path 3) ==\n")
    ann = pd.read_parquet(ANN_TAGGED)
    print(f"  loaded {len(ann):,} tagged announcements")

    # PATH 3: refined regex on attchmntText (substantive PDF body text)
    print("\n[PATH 3] computing refined-regex polarity from attchmntText...")
    ann["text_polarity"] = ann["attchmntText"].apply(refined_polarity)
    print(f"  text_polarity dist: mean={ann['text_polarity'].mean():.3f}  "
          f"+ve={(ann['text_polarity']>0).sum()}  -ve={(ann['text_polarity']<0).sum()}  "
          f"zero={(ann['text_polarity']==0).sum()}")

    # PATH 1: price-reaction polarity
    print("\n[PATH 1] computing price-reaction polarity...")
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "return_1d"])
    pr = price_reaction_polarity(ann, px)
    ann = ann.merge(pr, on=["symbol", "ann_date"], how="left")
    print(f"  px_polarity dist: +1: {(ann['px_polarity']==1).sum()}  "
          f"-1: {(ann['px_polarity']==-1).sum()}  "
          f"0: {(ann['px_polarity']==0).sum()}  "
          f"NaN: {ann['px_polarity'].isna().sum()}")

    # ── Aggregate to (symbol, trade_date) panel ──
    print("\nAggregating to per-(symbol, date) panel with rolling windows...")
    ann["ann_date"] = pd.to_datetime(ann["ann_date"])
    daily = ann.groupby(["symbol", "ann_date"]).agg(
        text_pol_sum   =("text_polarity", "sum"),
        text_pol_pos_n =("text_polarity", lambda s: (s > 0).sum()),
        text_pol_neg_n =("text_polarity", lambda s: (s < 0).sum()),
        px_pol_sum     =("px_polarity",   "sum"),
        px_pol_pos_n   =("px_polarity",   lambda s: (s == 1).sum()),
        px_pol_neg_n   =("px_polarity",   lambda s: (s == -1).sum()),
    ).reset_index()
    daily = daily.rename(columns={"ann_date": "trade_date"})

    # Build a complete (symbol, trade_date) index from prices
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    full = px.merge(daily, on=["symbol", "trade_date"], how="left")
    for c in ["text_pol_sum","text_pol_pos_n","text_pol_neg_n",
              "px_pol_sum","px_pol_pos_n","px_pol_neg_n"]:
        full[c] = full[c].fillna(0)
    full = full.sort_values(["symbol", "trade_date"])

    # Rolling sums over 5d and 30d windows
    print("\nRolling 5d and 30d windows per symbol...")
    for w in [5, 30]:
        for col in ["text_pol_sum", "text_pol_pos_n", "text_pol_neg_n",
                    "px_pol_sum",   "px_pol_pos_n",   "px_pol_neg_n"]:
            new = f"evp_{col}_{w}d"
            full[new] = full.groupby("symbol")[col].transform(lambda s: s.rolling(w, min_periods=1).sum())

    # Drop the per-day raw cols (keep only rolling features)
    keep = ["symbol", "trade_date"] + [c for c in full.columns if c.startswith("evp_")]
    out = full[keep]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(f"  rows: {len(out):,}, cols: {len(out.columns)}")
    feature_cols = [c for c in out.columns if c.startswith("evp_")]
    print(f"  features: {feature_cols}")

    # quick coverage on latest date
    latest = out["trade_date"].max()
    snap = out[out["trade_date"] == latest]
    print(f"\nCoverage on {latest:%Y-%m-%d} ({len(snap):,} rows):")
    for c in feature_cols:
        nonzero = (snap[c] != 0).sum()
        print(f"  {c:<35} nonzero={nonzero:>4}  mean={snap[c].mean():.3f}  max={snap[c].max():.0f}")


if __name__ == "__main__":
    main()
