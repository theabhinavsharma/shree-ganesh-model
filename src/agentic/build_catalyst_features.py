"""
Per-(symbol, trade_date) catalyst feature matrix from tagged announcements + insider trades.

Outputs `data/derived/catalyst_features.parquet` with columns:
  symbol, trade_date,
  ann_5d_count, ann_30d_count,
  ann_order_5d, ann_order_30d,        # binary flags
  ann_result_5d, ann_capex_30d, ann_fundraise_30d, ann_buyback_30d,
  ann_ma_30d, ann_regulatory_30d,
  catalyst_score_5d, catalyst_score_30d,  # cumulative tagged score
  insider_net_60d_inr, insider_buy_60d_inr, insider_stake_delta_60d,
  block_buy_5d_inr (placeholder = 0 today; wired for future feed)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True, help="tagged announcements parquet")
    p.add_argument("--pit", required=True, help="insider trading parquet")
    p.add_argument("--prices", required=True, help="price parquet (for trade_date grid)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ann = pd.read_parquet(args.ann)
    pit = pd.read_parquet(args.pit)
    prices = pd.read_parquet(args.prices, columns=["symbol", "trade_date"])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.normalize()

    ann = ann[ann["ann_date"].notna()].copy()
    ann["ann_date"] = pd.to_datetime(ann["ann_date"]).dt.normalize()
    ann["symbol"] = ann["symbol"].astype(str).str.upper()

    # build daily symbol-date count + categorical flags
    cat_dummies = pd.get_dummies(ann["catalyst_cat"], prefix="cat").astype(float)
    daily = pd.concat([ann[["symbol", "ann_date", "catalyst_score"]], cat_dummies], axis=1)
    daily = daily.groupby(["symbol", "ann_date"]).agg({
        "catalyst_score": "sum",
        **{c: "sum" for c in cat_dummies.columns},
    }).reset_index()
    daily = daily.rename(columns={"ann_date": "trade_date"})
    daily["ann_count"] = daily[[c for c in cat_dummies.columns]].sum(axis=1).clip(upper=1)  # presence
    daily["ann_count_raw"] = ann.groupby(["symbol", "ann_date"]).size().reset_index(name="n")["n"]

    # rolling windows on the price grid (so dates with no announcements still get features)
    sym_dates = prices.drop_duplicates(["symbol", "trade_date"]).sort_values(["symbol", "trade_date"])
    grid = sym_dates.merge(daily, on=["symbol", "trade_date"], how="left").fillna(0.0)

    # roll
    def roll_sum(col, w):
        return grid.groupby("symbol")[col].transform(
            lambda s: s.rolling(w, min_periods=1).sum())

    out = pd.DataFrame({"symbol": grid["symbol"], "trade_date": grid["trade_date"]})
    out["ann_5d_count"] = roll_sum("ann_count_raw", 5)
    out["ann_30d_count"] = roll_sum("ann_count_raw", 30)
    out["catalyst_score_5d"] = roll_sum("catalyst_score", 5)
    out["catalyst_score_30d"] = roll_sum("catalyst_score", 30)

    # category flags 5d / 30d
    flag_map = {
        "ann_order_5d":   ("cat_ORDER_WIN", 5),
        "ann_order_30d":  ("cat_ORDER_WIN", 30),
        "ann_result_5d":  ("cat_RESULT_BEAT", 5),
        "ann_capex_30d":  ("cat_CAPEX", 30),
        "ann_fundraise_30d": ("cat_FUNDRAISE", 30),
        "ann_buyback_30d":   ("cat_BUYBACK", 30),
        "ann_ma_30d":     ("cat_M_AND_A", 30),
        "ann_regulatory_30d": ("cat_REGULATORY", 30),
        "ann_dividend_30d":   ("cat_DIVIDEND", 30),
        "ann_bonussplit_30d": ("cat_BONUS_SPLIT", 30),
        "ann_rating_30d":     ("cat_RATING", 30),
        "ann_guidance_30d":   ("cat_GUIDANCE", 30),
    }
    for new, (src, w) in flag_map.items():
        if src in grid.columns:
            out[new] = (roll_sum(src, w) > 0).astype(int)
        else:
            out[new] = 0

    # insider features
    # intimDt is a date-only field (no intra-day timestamp on the NSE PIT API),
    # so we cannot tell whether a filing was disclosed pre- or post-close.
    # Conservative rule: shift every filing by +1 trading day. The model only
    # "knows" about a Form C the session AFTER the intimation date.
    pit = pit.copy()
    pit["intim_dt"] = pd.to_datetime(pit["intimDt"].astype(str).str.split(" ").str[0],
                                      format="%d-%b-%Y", errors="coerce")
    pit["symbol"] = pit["symbol"].astype(str).str.upper()
    pit = pit[(pit["intim_dt"].notna()) & (pit["delta_pct"].abs() < 5.0)]  # exclude restructurings
    pit["net_buy_inr"] = pit["buyValue"].fillna(0) - pit["sellValue"].fillna(0)

    trading_days = np.sort(prices["trade_date"].unique())
    raw = pit["intim_dt"].values.astype("datetime64[ns]")
    # next session strictly AFTER the intimation date
    target = (pit["intim_dt"] + pd.Timedelta(days=1)).values.astype("datetime64[ns]")
    idx = np.searchsorted(trading_days, target, side="left")
    valid = idx < len(trading_days)
    pit = pit.loc[valid].copy()
    pit["trade_date"] = trading_days[idx[valid]]

    pit_daily = pit.groupby(["symbol", "trade_date"]).agg(
        net_buy=("net_buy_inr", "sum"),
        buy_v=("buyValue", "sum"),
        stake_delta=("delta_pct", "sum"),
    ).reset_index()

    pgrid = sym_dates.merge(pit_daily, on=["symbol", "trade_date"], how="left").fillna(0.0)
    out["insider_net_60d_inr"] = pgrid.groupby("symbol")["net_buy"].transform(
        lambda s: s.rolling(60, min_periods=1).sum()).values
    out["insider_buy_60d_inr"] = pgrid.groupby("symbol")["buy_v"].transform(
        lambda s: s.rolling(60, min_periods=1).sum()).values
    out["insider_stake_delta_60d"] = pgrid.groupby("symbol")["stake_delta"].transform(
        lambda s: s.rolling(60, min_periods=1).sum()).values

    # block feed placeholder (wired to 0 until block-deal endpoint stabilizes)
    out["block_buy_5d_inr"] = 0.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    print(f"rows: {len(out):,}")
    print(f"symbols: {out['symbol'].nunique():,}")
    print(f"date range: {out['trade_date'].min()} → {out['trade_date'].max()}")
    print(f"\nfeature non-zero rates:")
    for c in out.columns:
        if c in ("symbol", "trade_date"):
            continue
        nz = (out[c] != 0).mean() * 100
        print(f"  {c:30s} {nz:6.2f}% non-zero  mean={out[c].mean():.4f}")


if __name__ == "__main__":
    main()
