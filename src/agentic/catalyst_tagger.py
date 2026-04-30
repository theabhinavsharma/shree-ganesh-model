"""
NLP-tag NSE corporate announcements into catalyst categories.

Categories (in priority order — first match wins):
  ORDER_WIN     — large order / contract / LOI / supply agreement
  RESULT_BEAT   — quarterly / annual results disclosure
  CAPEX         — capacity expansion, plant commissioning, capex
  FUNDRAISE     — QIP, preferential issue, rights, bond
  M_AND_A       — acquisition, merger, divestment, JV
  BUYBACK       — share buyback / open offer
  DIVIDEND      — interim/final dividend
  BONUS_SPLIT   — bonus issue, stock split
  AGM_BM        — board meeting / AGM (low signal)
  RATING        — credit rating change
  REGULATORY    — penalty, sebi notice, scn, ED, CBI
  GUIDANCE      — outlook update, forward statement
  OTHER

Look-ahead handling: announcements published at or after 15:30 IST roll forward
to the next trading day. The model trained on close[t] features cannot see
post-close news from day t — that is tomorrow's information.

Usage:
  python -m src.agentic.catalyst_tagger \
    --in tmp/from_scratch_7d_run/alt/corp_announcements.parquet \
    --out tmp/from_scratch_7d_run/alt/announcements_tagged.parquet \
    --prices data/derived/stock_daily_facts_adjusted_2015plus.parquet
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

# NSE close = 15:30 IST. Anything stamped at-or-after this is "tomorrow's news"
# from the perspective of a model that fits to today's close.
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30

# (regex_pattern, category, base_signal_score)
PATTERNS = [
    (r"\b(order|contract|loi|letter of intent|work order|tender|bagged|secured|won|awarded|supply (agreement|order))\b", "ORDER_WIN", 0.7),
    (r"\b(financial result|q[1-4]\s*fy|quarterly result|earnings|outcome of board.*result)\b", "RESULT_BEAT", 0.5),
    (r"\b(capex|capacity (expansion|enhancement|addition)|new plant|plant commissioning|greenfield|brownfield|expansion plan)\b", "CAPEX", 0.6),
    (r"\b(qip|qualified institutional|preferential issue|preferential allotment|rights issue|bond issue|ncd|fundrais|placement)\b", "FUNDRAISE", 0.5),
    (r"\b(acquisition|acquired|merger|amalgamation|demerger|divestment|stake (sale|purchase)|joint venture|jv with|subsidiary acqui)\b", "M_AND_A", 0.6),
    (r"\b(buy ?back|open offer)\b", "BUYBACK", 0.7),
    (r"\b(interim dividend|final dividend|special dividend|record date.*dividend)\b", "DIVIDEND", 0.3),
    (r"\b(bonus issue|stock split|sub-?division of (shares|equity)|face value split)\b", "BONUS_SPLIT", 0.6),
    (r"\b(credit rating|rating action|reaffirm|upgrade.*rating|downgrade.*rating|crisil|icra|care ratings|india ratings|fitch)\b", "RATING", 0.4),
    (r"\b(penalty|sebi (notice|order|scn)|show cause|insolvency|nclt|ed (raid|notice)|cbi|nse penalty|bse penalty|fine|disciplinary)\b", "REGULATORY", -0.5),
    (r"\b(outlook|guidance|forward looking|management commentary|analyst meet|investor meet|conference call)\b", "GUIDANCE", 0.2),
    (r"\b(annual general meeting|extraordinary general meeting|board meeting|agm|egm|outcome of board)\b", "AGM_BM", 0.05),
]

POSITIVE_BOOSTS = [
    (r"\b(record (revenue|profit|orders))\b", 0.3),
    (r"\b(highest ever)\b", 0.3),
    (r"\b(beats? (estimates?|street))\b", 0.3),
    (r"\b(margin (expansion|improvement))\b", 0.2),
    (r"\b(robust|strong|exceptional)\b", 0.1),
    (r"\b(rs\.?\s*[0-9,]+\s*crore)\b", 0.1),  # large rupee figure mention
]

NEGATIVE_BOOSTS = [
    (r"\b(loss|decline|drop|fall|miss(ed)?|weak|deteriorat)\b", -0.2),
    (r"\b(resignation|resigned|stepped down).*(md|ceo|cfo|director)\b", -0.3),
    (r"\b(qualified|adverse) opinion\b", -0.4),
]


def tag_one(text: str) -> tuple[str, float]:
    if not isinstance(text, str) or not text.strip():
        return ("OTHER", 0.0)
    t = text.lower()
    cat = "OTHER"
    score = 0.0
    for pat, c, s in PATTERNS:
        if re.search(pat, t):
            cat, score = c, s
            break
    for pat, b in POSITIVE_BOOSTS:
        if re.search(pat, t):
            score += b
    for pat, b in NEGATIVE_BOOSTS:
        if re.search(pat, t):
            score += b
    return (cat, round(score, 3))


def assign_known_date(ts: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map raw announcement timestamp → the first trading day on which the model
    is allowed to use that announcement.

    Rule:
      ts at-or-before 15:30 IST AND date is a trading day  → same day
      otherwise (post-close, or filed on holiday/weekend)   → next trading day

    `trading_days` must be a sorted, normalized DatetimeIndex of NSE session dates.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    is_post_close = (
        (ts.dt.hour > MARKET_CLOSE_HOUR)
        | ((ts.dt.hour == MARKET_CLOSE_HOUR) & (ts.dt.minute >= MARKET_CLOSE_MIN))
    )
    base = ts.dt.normalize()
    # Where post-close, advance one calendar day so searchsorted picks the *next* session.
    base = base.where(~is_post_close, base + pd.Timedelta(days=1))

    td = trading_days.values
    idx = np.searchsorted(td, base.values, side="left")
    idx = np.clip(idx, 0, len(td) - 1)
    out = pd.Series(pd.to_datetime(td[idx]), index=ts.index)
    # Past the last known trading day → leave NaT, downstream will drop.
    out = out.where(idx < len(td), pd.NaT)
    out = out.where(ts.notna(), pd.NaT)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--prices", dest="prices",
                   default="data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                   help="price parquet — used to build the trading-day calendar")
    args = p.parse_args()

    df = pd.read_parquet(args.src)
    text = (df["desc"].fillna("") + " || " + df.get("attchmntText", pd.Series("", index=df.index)).fillna(""))
    tagged = text.apply(tag_one)
    df["catalyst_cat"] = tagged.apply(lambda x: x[0])
    df["catalyst_score"] = tagged.apply(lambda x: x[1])

    # Build NSE trading-day calendar from the price parquet.
    px = pd.read_parquet(args.prices, columns=["trade_date"])
    trading_days = pd.DatetimeIndex(
        pd.to_datetime(px["trade_date"]).dt.normalize().unique()
    ).sort_values()

    # Pick the timestamp source. Prefer ann_ts; fall back to sort_date (date-only).
    if "ann_ts" in df.columns and df["ann_ts"].notna().any():
        ts = pd.to_datetime(df["ann_ts"], errors="coerce")
    else:
        # date-only fallback: treat as 00:00 (pre-open) → same trading day
        ts = pd.to_datetime(df.get("sort_date"), errors="coerce")

    df["ann_date"] = assign_known_date(ts, trading_days)

    n_total = len(df)
    n_shifted = int((df["ann_date"].dt.normalize() != ts.dt.normalize()).sum())
    print(f"tagged {n_total:,} announcements")
    print(f"  shifted to next trading day (post-close / holiday): {n_shifted:,} "
          f"({n_shifted/n_total*100:.1f}%)")

    out = Path(args.dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(df["catalyst_cat"].value_counts().to_string())
    print(f"\nmean catalyst_score by category:")
    print(df.groupby("catalyst_cat")["catalyst_score"].agg(["count", "mean"]).round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
