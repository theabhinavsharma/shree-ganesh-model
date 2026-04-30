"""Compute feature-importance for the v3 long ensemble + tag each feature
with its origin (consumed vs created) and date added.

Process:
  1. Train a quick LightGBM on (features → winner_5pct_7td) over labeled OOS data
  2. Extract gain-based feature importance
  3. Tag each feature:
       - origin = "consumed" (raw NSE data: close, volume, rsi, etc)
                | "created" (engineered: WQ alphas, macro overlays, sentiment)
       - category = same as factor_registry (or "core_price" for raw bhavcopy fields)
       - added_at = inferred from git/file metadata (best effort)
  4. Persist to data/derived/feature_importance.parquet

Output schema:
  feature, importance_gain, importance_split, rank, origin, category,
  added_at, source_script

Used by build_dashboard.py to render the horizontal bar chart.
"""
from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
EXTRA = ROOT / "data/derived/extra_features.parquet"
CAT = ROOT / "data/derived/catalyst_features.parquet"
OUT = ROOT / "data/derived/feature_importance.parquet"
REGISTRY = ROOT / "data/derived/factor_registry.json"

# Map feature → (origin, category, added_at, source_script)
# These are the FIXED tags for raw price/technical features (consumed from NSE)
RAW_PRICE_FEATURES = {
    "close": "raw_price", "open": "raw_price", "high": "raw_price", "low": "raw_price",
    "return_1d": "raw_price", "return_20d": "raw_price",
    "volume_vs_20d": "raw_volume", "traded_value_vs_20d": "raw_volume",
    "delivery_pct": "raw_volume", "avg_traded_value_20d": "raw_volume",
    "sma_20": "raw_technical", "sma_50": "raw_technical", "sma_200": "raw_technical",
    "rsi_14_daily": "raw_technical", "rsi_14_weekly": "raw_technical",
    "rsi_14_monthly": "raw_technical",
    "above_50dma": "raw_technical", "above_200dma": "raw_technical",
    "dist_sma20": "raw_technical", "dist_sma50": "raw_technical", "dist_sma200": "raw_technical",
    "realized_vol_20d": "raw_technical", "adv_20d_cr": "raw_volume",
}

MARKET_FEATURES = {
    "market_5d_ret": "market_breadth", "market_20d_ret": "market_breadth",
    "market_breadth_50dma": "market_breadth", "market_breadth_200dma": "market_breadth",
    "rel_strength_20d": "market_breadth",
    "sector_5d_ret": "sector_rotation", "sector_20d_ret": "sector_rotation",
    "sector_60d_ret": "sector_rotation",
}

# These were *created* by us (engineered features)
CREATED_FEATURES_FROM_CATALOG = {
    "ann_5d_count": "catalyst", "ann_30d_count": "catalyst",
    "catalyst_score_5d": "catalyst", "catalyst_score_30d": "catalyst",
    "ann_order_5d": "catalyst", "ann_result_5d": "catalyst",
    "ann_capex_30d": "catalyst", "ann_buyback_30d": "catalyst",
    "insider_net_60d_inr": "insider_pit", "insider_buy_60d_inr": "insider_pit",
    "block_buy_5d_inr": "block_deals",
    # WQ alphas + factor_factory output
    "alpha_volume_signed_revert": "wq_alpha", "alpha_intraday_norm_range": "wq_alpha",
    "alpha_high_extension_revert": "wq_alpha", "alpha_geom_mid_vs_vwap": "wq_alpha",
    "alpha_open_volume_corr_10": "wq_alpha",
    "amihud_20d": "microstructure", "turnover_skew_20d": "microstructure",
    "vol_z_60d": "volatility_regime", "vol_term_20_60": "volatility_regime",
    "vol_of_vol_60d": "volatility_regime", "rv_60d": "volatility_regime",
    "vol_max_63d": "volatility_regime",
    "usdinr": "macro_overlay", "usdinr_5d_chg": "macro_overlay",
    "usdinr_20d_chg": "macro_overlay", "eurinr": "macro_overlay",
    "gbpinr": "macro_overlay", "jpyinr": "macro_overlay",
    "wiki_views": "alt_attention", "wiki_views_z": "alt_attention",
}


def get_feature_added_dates() -> dict[str, str]:
    """Best-effort: assign a creation date to each feature.

    For features we hand-built today (this Apr 2026 session), use today.
    For raw features (price, technical), backfill with project-start date.
    For features in registry, prefer their `created_at` field if set.
    """
    today = str(date.today())
    project_start = "2026-04-15"  # rough — when the project started ingesting bhavcopy
    session_start = "2026-04-29"  # today, when we built the factor pipeline

    dates: dict[str, str] = {}

    # Raw / consumed features — project start
    for f in RAW_PRICE_FEATURES:
        dates[f] = project_start
    for f in MARKET_FEATURES:
        dates[f] = project_start

    # Engineered features — built this session
    for f in CREATED_FEATURES_FROM_CATALOG:
        dates[f] = session_start

    # Screener fundamentals — auto-tag any scr_* feature
    SCREENER_FIELDS = ["scr_pe", "scr_market_cap_cr", "scr_dividend_yield", "scr_book_value",
                        "scr_roce", "scr_roe", "scr_face_value",
                        "scr_compounded_sales_growth_3_years", "scr_compounded_sales_growth_5_years",
                        "scr_compounded_sales_growth_10_years",
                        "scr_compounded_profit_growth_3_years", "scr_compounded_profit_growth_5_years",
                        "scr_compounded_profit_growth_10_years",
                        "scr_return_on_equity_3_years", "scr_return_on_equity_5_years",
                        "scr_return_on_equity_10_years",
                        "scr_stock_price_cagr_1_year", "scr_stock_price_cagr_3_years",
                        "scr_stock_price_cagr_5_years", "scr_stock_price_cagr_10_years",
                        "scr_peg_3y", "scr_price_to_book", "scr_earnings_yield"]
    for f in SCREENER_FIELDS:
        dates[f] = session_start

    # Registry-driven (override if registry has explicit created_at)
    if REGISTRY.exists():
        try:
            reg = json.loads(REGISTRY.read_text())
            for h in reg:
                created = h.get("created_at", session_start)
                # match on hypothesis id approximations to actual feature names
                for f in dates.keys():
                    if h["id"] in f or f in h["id"]:
                        dates[f] = created
        except Exception:
            pass

    return dates


def feature_origin(feature: str) -> tuple[str, str, str]:
    """Return (origin_type, category, source_script)."""
    if feature in RAW_PRICE_FEATURES:
        return ("consumed", RAW_PRICE_FEATURES[feature], "refresh_prices.py / bhavcopy")
    if feature in MARKET_FEATURES:
        return ("created", MARKET_FEATURES[feature], "build_panel (in run_v3)")
    if feature.startswith("scr_"):
        return ("consumed", "screener_fundamental",
                "fetch_screener_fundamentals.py (Screener.in)")
    if feature in CREATED_FEATURES_FROM_CATALOG:
        cat = CREATED_FEATURES_FROM_CATALOG[feature]
        if cat == "wq_alpha":
            return ("created", cat, "feature_factory.py")
        if cat == "microstructure":
            return ("created", cat, "feature_factory.py")
        if cat == "volatility_regime":
            return ("created", cat, "feature_factory.py")
        if cat == "macro_overlay":
            return ("created", cat, "feature_factory.py + fetch_forex_macro.py")
        if cat == "alt_attention":
            return ("created", cat, "fetch_wiki_pageviews.py")
        if cat == "catalyst":
            return ("created", cat, "build_catalyst_features.py")
        if cat == "insider_pit":
            return ("created", cat, "build_catalyst_features.py + insider PIT")
        if cat == "block_deals":
            return ("created", cat, "fetch_block_deals.py")
        return ("created", cat, "(various)")
    return ("created", "uncategorised", "(unknown)")


def main() -> None:
    print("== compute_feature_importance ==")
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2022-01-01"]  # last ~4 years for fitting

    # rebuild label
    H = 7
    shifts_high = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
        axis=1,
    )
    df["fwd_high_max"] = shifts_high.max(axis=1)
    df["forward_high_pct_7td"] = df["fwd_high_max"] / df["close"] - 1
    df["winner_5pct_7td"] = (df["forward_high_pct_7td"] >= 0.05).astype(int)
    complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
    df.loc[~complete, ["forward_high_pct_7td", "winner_5pct_7td"]] = pd.NA

    # core engineered
    df["dist_sma20"] = df["close"] / df["sma_20"] - 1
    df["dist_sma50"] = df["close"] / df["sma_50"] - 1
    df["dist_sma200"] = df["close"] / df["sma_200"] - 1
    df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
    df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
    df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7

    # market
    liq = df[df["adv_20d_cr"] >= 1.0]
    mkt = liq.groupby("trade_date").agg(
        market_1d_ret=("return_1d", "median"),
        market_breadth_50dma=("above_50dma", "mean"),
        market_breadth_200dma=("above_200dma", "mean"),
    ).reset_index().sort_values("trade_date")
    mkt["market_5d_ret"] = mkt["market_1d_ret"].rolling(5).sum()
    mkt["market_20d_ret"] = mkt["market_1d_ret"].rolling(20).sum()
    df = df.merge(mkt, on="trade_date", how="left")

    # join extra_features (auto-pull every numeric column starting with known prefixes)
    if EXTRA.exists():
        ex = pd.read_parquet(EXTRA)
        ex["trade_date"] = pd.to_datetime(ex["trade_date"])
        # auto-discover all engineered features (alpha_, vol_, rv_, scr_, usdinr*, wiki_, amihud_, turnover_)
        keep_extra = [c for c in ex.columns
                       if c not in ("symbol", "trade_date")
                       and (c.startswith(("alpha_", "vol_", "rv_", "scr_", "usdinr",
                                          "eurinr", "gbpinr", "jpyinr", "wiki_"))
                            or c in ("amihud_20d", "turnover_skew_20d", "vol_max_63d"))]
        # filter to columns numerically tractable (drop string cols if any)
        keep_extra = [c for c in keep_extra if c in ex.columns
                       and pd.api.types.is_numeric_dtype(ex[c])]
        df = df.merge(ex[["symbol", "trade_date"] + keep_extra],
                      on=["symbol", "trade_date"], how="left")
        print(f"  joined {len(keep_extra)} engineered features from extra_features.parquet")
    else:
        keep_extra = []

    # join catalysts for importance of those features
    if CAT.exists():
        cf = pd.read_parquet(CAT)
        cf["trade_date"] = pd.to_datetime(cf["trade_date"])
        cat_cols = [c for c in cf.columns if c not in ("symbol", "trade_date")]
        df = df.merge(cf, on=["symbol", "trade_date"], how="left")
        for c in cat_cols:
            df[c] = df[c].fillna(0)
    else:
        cat_cols = []

    BASE_FEATS = ["return_1d", "return_20d",
                  "dist_sma20", "dist_sma50", "dist_sma200",
                  "above_50dma", "above_200dma",
                  "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
                  "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
                  "realized_vol_20d", "adv_20d_cr",
                  "market_5d_ret", "market_20d_ret",
                  "market_breadth_50dma", "market_breadth_200dma"]
    ALL_FEATS = BASE_FEATS + keep_extra + cat_cols
    # drop dupes + drop any not actually in dataframe
    ALL_FEATS = list(dict.fromkeys(ALL_FEATS))
    ALL_FEATS = [f for f in ALL_FEATS if f in df.columns]

    df = df.dropna(subset=BASE_FEATS).copy()
    labeled = df[df["winner_5pct_7td"].notna()].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    print(f"  fitting on {len(labeled):,} rows × {len(ALL_FEATS)} features")

    # fill NaN for extra/catalyst features (median)
    for c in ALL_FEATS:
        if labeled[c].isna().any():
            labeled[c] = labeled[c].fillna(labeled[c].median())

    lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.06, num_leaves=64,
                               min_child_samples=200, feature_fraction=0.85,
                               bagging_fraction=0.85, bagging_freq=5,
                               random_state=42, verbose=-1, n_jobs=-1)
    lgbm.fit(labeled[ALL_FEATS], labeled["winner_5pct_7td"].astype(int))
    print(f"  fit complete")

    gain = lgbm.booster_.feature_importance(importance_type="gain")
    split = lgbm.booster_.feature_importance(importance_type="split")
    feat_names = lgbm.booster_.feature_name()

    dates_map = get_feature_added_dates()
    rows = []
    for f, g, s in zip(feat_names, gain, split):
        origin, cat, source = feature_origin(f)
        rows.append({
            "feature": f,
            "importance_gain": float(g),
            "importance_split": int(s),
            "origin": origin,
            "category": cat,
            "source_script": source,
            "added_at": dates_map.get(f, "—"),
        })
    out = pd.DataFrame(rows).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print(f"\nwrote {OUT} ({len(out)} features)")
    print("\nTop 20 by gain:")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
