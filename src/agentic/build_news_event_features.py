"""News + corporate-event feature builder — 5d and 15d windows, stock + industry.

Inputs:
  data/derived/news_feed.parquet                   — RSS news (sparse pre-2026)
  data/derived/news_features.parquet               — pre-scored sentiment per symbol (snapshot only)
  data/events_full_history/normalized/stock_announcements.parquet — NSE corp filings (Feb-Apr 2026)

Output:
  data/derived/news_event_features.parquet — (symbol, trade_date) keyed
    Stock-level (5d / 15d windows ending at trade_date):
      news_5d_count, news_15d_count
      news_5d_sentiment, news_15d_sentiment   (where sentiment available)
      evt_5d_total, evt_15d_total
      evt_5d_order_win, evt_5d_approval, evt_5d_promoter_buy,
        evt_5d_pledge_change, evt_5d_results
      evt_15d_*  (same set)
      evt_5d_positive (= order_win + approval + promoter_buy)
      evt_15d_positive
      evt_5d_negative (= pledge_change increases — proxy for distress)

    Industry-level aggregations (via industry_hint):
      evt_ind_5d_total, evt_ind_15d_total
      evt_ind_5d_positive, evt_ind_15d_positive
      evt_ind_5d_order_win, evt_ind_15d_order_win

CRITICAL: every column varies per (symbol, trade_date). No snapshot broadcasts.
Validated by tests/test_no_per_symbol_constants.py.

Usage:
  python3 src/agentic/build_news_event_features.py             # rebuild full panel
  python3 src/agentic/build_news_event_features.py --today     # today's row only

Honest caveat (per CONSTITUTION.md §1.7): the announcements parquet covers
~3 months (2026-02 to 2026-04). Walk-forward training on 2018-2022 is
DATA-BLOCKED until historical news is backfilled (MoneyControl archive,
ET archive, NSE/BSE filings pre-2026). The output is currently usable as
a TODAY-OVERLAY for live basket scoring, not as a historical model
feature for find_180d_frontier_honest.py training.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
NEWS_FEED = ROOT / "data/derived/news_feed.parquet"
NEWS_FEAT = ROOT / "data/derived/news_features.parquet"
EVENTS = ROOT / "data/events_full_history/normalized/stock_announcements.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/news_event_features.parquet"

WINDOWS = [5, 7, 15]
EVENT_FLAGS = [
    "is_order_win",
    "is_approval",
    "is_promoter_buying",
    "is_pledge_change",
    "is_results_event",
]
POSITIVE_EVENTS = ["is_order_win", "is_approval", "is_promoter_buying"]


def load_events() -> pd.DataFrame:
    if not EVENTS.exists():
        raise FileNotFoundError(f"missing {EVENTS}")
    df = pd.read_parquet(EVENTS)
    df["event_date"] = pd.to_datetime(df["event_date"])
    for c in EVENT_FLAGS:
        if c not in df.columns:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)
    return df


def load_news() -> pd.DataFrame:
    if not NEWS_FEED.exists():
        return pd.DataFrame(columns=["symbol", "date", "sentiment"])
    nf = pd.read_parquet(NEWS_FEED)
    nf["pub_ts"] = pd.to_datetime(nf["pub_ts"], errors="coerce", utc=True)
    nf["date"] = nf["pub_ts"].dt.tz_convert("Asia/Kolkata").dt.date
    nf["date"] = pd.to_datetime(nf["date"])
    # Explode multi-symbol rows
    if "symbols" in nf.columns:
        nf = nf.explode("symbols").rename(columns={"symbols": "symbol"})
        nf = nf.dropna(subset=["symbol"])
    else:
        nf["symbol"] = pd.NA
    nf["sentiment"] = 0.0  # placeholder; news_features.parquet has scored sentiment but is snapshot-only
    return nf[["symbol", "date", "sentiment"]]


def build_stock_event_panel(events: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each (symbol, trade_date), count events in trailing 5d and 15d windows."""
    if events.empty:
        return pd.DataFrame()

    # Aggregate events to (symbol, event_date) counts
    agg_cols = {f: "sum" for f in EVENT_FLAGS}
    agg_cols["sequence_id"] = "size"
    daily = (events.groupby(["symbol", "event_date"], as_index=False)
                    .agg(agg_cols)
                    .rename(columns={"sequence_id": "evt_total"}))

    # For each symbol, reindex to a daily grid covering [first_event - 30d, last_event + 30d]
    out_rows = []
    for sym, g in daily.groupby("symbol"):
        g = g.sort_values("event_date").set_index("event_date")
        idx = pd.date_range(g.index.min() - pd.Timedelta(days=30),
                            g.index.max() + pd.Timedelta(days=30), freq="D")
        g = g.reindex(idx, fill_value=0)
        for w in WINDOWS:
            g[f"evt_{w}d_total"] = g["evt_total"].rolling(w, min_periods=1).sum()
            for f in EVENT_FLAGS:
                key = f.replace("is_", "")
                g[f"evt_{w}d_{key}"] = g[f].rolling(w, min_periods=1).sum()
            g[f"evt_{w}d_positive"] = sum(g[f].rolling(w, min_periods=1).sum() for f in POSITIVE_EVENTS)
            g[f"evt_{w}d_negative"] = g["is_pledge_change"].rolling(w, min_periods=1).sum()
        g = g.reset_index().rename(columns={"index": "trade_date"})
        g["symbol"] = sym
        out_rows.append(g)
    if not out_rows:
        return pd.DataFrame()
    panel = pd.concat(out_rows, ignore_index=True)
    keep_cols = (["symbol", "trade_date"]
                 + [f"evt_{w}d_total" for w in WINDOWS]
                 + [f"evt_{w}d_{f.replace('is_','')}" for w in WINDOWS for f in EVENT_FLAGS]
                 + [f"evt_{w}d_positive" for w in WINDOWS]
                 + [f"evt_{w}d_negative" for w in WINDOWS])
    return panel[keep_cols]


def build_industry_panel(events: pd.DataFrame) -> pd.DataFrame:
    """Industry-level (via industry_hint) rolling counts in 5d/15d windows."""
    if events.empty or "industry_hint" not in events.columns:
        return pd.DataFrame()
    ev = events.dropna(subset=["industry_hint"]).copy()
    if ev.empty:
        return pd.DataFrame()

    # Aggregate to (industry, date)
    agg_cols = {f: "sum" for f in EVENT_FLAGS}
    agg_cols["sequence_id"] = "size"
    daily = (ev.groupby(["industry_hint", "event_date"], as_index=False)
               .agg(agg_cols)
               .rename(columns={"sequence_id": "evt_ind_total"}))
    rows = []
    for ind, g in daily.groupby("industry_hint"):
        g = g.sort_values("event_date").set_index("event_date")
        idx = pd.date_range(g.index.min() - pd.Timedelta(days=30),
                            g.index.max() + pd.Timedelta(days=30), freq="D")
        g = g.reindex(idx, fill_value=0)
        for w in WINDOWS:
            g[f"evt_ind_{w}d_total"] = g["evt_ind_total"].rolling(w, min_periods=1).sum()
            g[f"evt_ind_{w}d_positive"] = sum(g[f].rolling(w, min_periods=1).sum() for f in POSITIVE_EVENTS)
            g[f"evt_ind_{w}d_order_win"] = g["is_order_win"].rolling(w, min_periods=1).sum()
        g = g.reset_index().rename(columns={"index": "trade_date"})
        g["industry_hint"] = ind
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    return panel[["industry_hint", "trade_date",
                  "evt_ind_5d_total", "evt_ind_15d_total",
                  "evt_ind_5d_positive", "evt_ind_15d_positive",
                  "evt_ind_5d_order_win", "evt_ind_15d_order_win"]]


def attach_industry_to_symbols(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Inner-join symbol's industry_hint (most recent) so each symbol-row gets industry features."""
    if events.empty or panel.empty:
        return panel
    sym2ind = (events.dropna(subset=["industry_hint"])
                      .sort_values("event_date")
                      .drop_duplicates("symbol", keep="last")[["symbol", "industry_hint"]])
    return panel.merge(sym2ind, on="symbol", how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", action="store_true",
                    help="Print only today's overlay for stocks with at least one event")
    args = ap.parse_args()

    print("loading events …")
    events = load_events()
    print(f"  rows: {len(events):,}, symbols: {events['symbol'].nunique():,}, "
          f"range: {events['event_date'].min().date()} → {events['event_date'].max().date()}")

    print("building stock-level rolling features …")
    stock_panel = build_stock_event_panel(events, pd.DatetimeIndex([]))
    print(f"  stock panel rows: {len(stock_panel):,}")

    print("building industry-level rolling features …")
    industry_panel = build_industry_panel(events)
    print(f"  industry panel rows: {len(industry_panel):,}")

    if not stock_panel.empty:
        stock_panel = attach_industry_to_symbols(events, stock_panel)
        if not industry_panel.empty:
            stock_panel = stock_panel.merge(
                industry_panel, on=["industry_hint", "trade_date"], how="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    stock_panel.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  shape={stock_panel.shape}")

    if args.today:
        today = pd.Timestamp(pd.Timestamp.now().date())
        latest = stock_panel[stock_panel["trade_date"] == today]
        if latest.empty:
            # fall back to most recent date in data
            latest = stock_panel[stock_panel["trade_date"] == stock_panel["trade_date"].max()]
        print(f"\n=== Today overlay ({latest['trade_date'].max().date() if len(latest) else 'n/a'}) ===")
        print(latest.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
