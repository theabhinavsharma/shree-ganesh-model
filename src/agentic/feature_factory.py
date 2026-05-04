"""Feature factory — compile new factors from existing data.

Reads the master price parquet + macro/fundamentals/sentiment/wiki/etc and
emits a new wide parquet `data/derived/extra_features.parquet` containing:

  • WorldQuant 101-style alphas (computable from OHLCV)
  • Volatility regime factors
  • Microstructure / liquidity factors
  • Cross-sectional / sector-relative
  • Calendar dummies
  • Macro overlays (USDINR-sensitivity, etc.)

Joining: extra_features can be merged to v3 training panel via (symbol, trade_date)
without changing existing pipeline. Run via:
  PYTHONPATH=. /usr/bin/python3 src/agentic/feature_factory.py

The factor_evaluator.py measures lift by retraining v3 with these added.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
MACRO = ROOT / "data/derived/macro_timeseries.parquet"
MACRO_PANEL = ROOT / "data/derived/macro_panel.parquet"           # NEW: consolidated 150-col macro panel
INDUSTRY_PANEL = ROOT / "data/derived/industry_panel.parquet"     # NEW: per-sector aggregates
WIKI = ROOT / "data/derived/wiki_pageviews.parquet"
SCREENER_FUND = ROOT / "data/derived/screener_fundamentals.parquet"
DERIVED_RATIOS = ROOT / "data/derived/derived_ratios.parquet"
ACADEMIC_ALPHAS = ROOT / "data/derived/academic_alphas.parquet"
OUT = ROOT / "data/derived/extra_features.parquet"

LOOKBACK_DAYS = 1500  # ~6 years; enough for OOS 2024-2025 + warmup


def _ts_rolling(g: pd.DataFrame, col: str, window: int, op: str) -> pd.Series:
    """Per-symbol rolling op."""
    s = g[col]
    if op == "mean":
        return s.rolling(window).mean()
    if op == "std":
        return s.rolling(window).std()
    if op == "max":
        return s.rolling(window).max()
    if op == "min":
        return s.rolling(window).min()
    if op == "skew":
        return s.rolling(window).skew()
    if op == "z":
        return (s - s.rolling(window).mean()) / s.rolling(window).std()
    raise ValueError(op)


def main() -> None:
    print("== feature_factory ==")
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "open", "high", "low", "close",
                                           "total_traded_qty", "total_traded_value", "delivery_pct",
                                           "avg_traded_value_20d", "avg_vol_20d", "return_1d",
                                           "return_20d", "rsi_14_daily", "sma_20", "sma_50",
                                           "realized_vol_20d" if False else "rsi_14_daily"])
    # re-read to get true list (the ternary above was a placeholder)
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    cutoff = df["trade_date"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["trade_date"] >= cutoff].copy()
    print(f"  base panel: {len(df):,} rows, {df['symbol'].nunique():,} symbols, "
          f"{df['trade_date'].min().date()} → {df['trade_date'].max().date()}")
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # vwap proxy (needed for several alphas)
    df["vwap"] = df["total_traded_value"] / df["total_traded_qty"].replace(0, np.nan)

    # realized vol (already exists usually; recompute defensively)
    df["rv_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
    df["rv_60d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(60).std())

    # ─── 1. WORLDQUANT-style alphas ───────────────────────────────────
    print("  computing WQ-style alphas …")
    df["alpha_open_volume_corr_10"] = df.groupby("symbol", group_keys=False).apply(
        lambda g: -1 * g["open"].rolling(10).corr(g["total_traded_qty"]))

    df["alpha_intraday_norm_range"] = (df["close"] - df["open"]) / ((df["high"] - df["low"]).replace(0, np.nan) + 0.001)

    df["alpha_high_extension_revert"] = np.where(
        (df.groupby("symbol")["high"].transform(lambda s: s.rolling(20).mean()) < df["high"]),
        -1 * (df["high"] - df.groupby("symbol")["high"].shift(2)),
        0,
    )

    df["alpha_geom_mid_vs_vwap"] = (df["high"] * df["low"]) ** 0.5 / df["vwap"] - 1

    df["alpha_volume_signed_revert"] = (
        np.sign(df.groupby("symbol")["total_traded_qty"].diff(1)) *
        -1 * df.groupby("symbol")["close"].diff(1)
    )

    # ─── 2. VOLATILITY REGIME ─────────────────────────────────────────
    print("  computing volatility regime …")
    df["vol_z_60d"] = df.groupby("symbol", group_keys=False).apply(
        lambda g: (g["rv_20d"] - g["rv_20d"].rolling(60).mean()) / g["rv_20d"].rolling(60).std())
    df["vol_term_20_60"] = df["rv_20d"] / df["rv_60d"]
    df["vol_of_vol_60d"] = df.groupby("symbol")["rv_20d"].transform(lambda s: s.rolling(60).std())

    # ─── 3. MICROSTRUCTURE / LIQUIDITY ────────────────────────────────
    print("  computing microstructure …")
    df["amihud_20d"] = df.groupby("symbol", group_keys=False).apply(
        lambda g: (g["return_1d"].abs() / (g["close"] * g["total_traded_qty"]).replace(0, np.nan)).rolling(20).mean())
    df["turnover_skew_20d"] = df.groupby("symbol", group_keys=False).apply(
        lambda g: (g["total_traded_qty"] / g["avg_vol_20d"].replace(0, np.nan)).rolling(20).skew())

    # ─── 4. CALENDAR DUMMIES ──────────────────────────────────────────
    print("  computing calendar dummies …")
    df["dow"] = df["trade_date"].dt.dayofweek
    df["dom"] = df["trade_date"].dt.day
    df["is_month_end_3d"] = df.groupby(df["trade_date"].dt.to_period("M"))["trade_date"].transform(
        lambda s: (s >= s.nlargest(3).min()).astype(int))
    # F&O monthly expiry = last Thursday of month
    last_thu = df.groupby(df["trade_date"].dt.to_period("M"))["trade_date"].transform(
        lambda s: s[s.dt.dayofweek == 3].max() if (s.dt.dayofweek == 3).any() else pd.NaT)
    df["is_expiry_week"] = ((last_thu - df["trade_date"]).dt.days.between(0, 6)).astype(int)

    # ─── 5. MACRO OVERLAY (USDINR sensitivity) ────────────────────────
    if MACRO.exists():
        print("  joining macro (USDINR / EUR / GBP / JPY) …")
        m = pd.read_parquet(MACRO)
        m["trade_date"] = pd.to_datetime(m["trade_date"])
        # forward-fill macro into trading days (FX is published every business day, but holidays differ)
        df = df.merge(m, on="trade_date", how="left")
        for col in ["usdinr", "eurinr", "gbpinr", "jpyinr"]:
            if col in df.columns:
                df[col] = df.groupby("symbol")[col].transform(lambda s: s.ffill())
                df[f"{col}_5d_chg"] = df.groupby("symbol")[col].transform(lambda s: s.pct_change(5))
                df[f"{col}_20d_chg"] = df.groupby("symbol")[col].transform(lambda s: s.pct_change(20))

    # ─── 5b. SCREENER FUNDAMENTALS (rich per-stock ratios) ────────────
    if SCREENER_FUND.exists():
        print("  joining Screener fundamentals (PE, ROCE, ROE, growth CAGRs) …")
        sf = pd.read_parquet(SCREENER_FUND)
        # most recent snapshot per symbol
        sf["fetch_date"] = pd.to_datetime(sf["fetch_date"])
        sf = sf.sort_values("fetch_date").groupby("symbol").tail(1)
        keep_screener = [c for c in [
            "pe", "market_cap_cr", "dividend_yield", "book_value", "roce", "roe",
            "compounded_sales_growth_3_years", "compounded_sales_growth_5_years",
            "compounded_profit_growth_3_years", "compounded_profit_growth_5_years",
            "return_on_equity_3_years", "return_on_equity_5_years",
            "stock_price_cagr_1_year", "stock_price_cagr_3_years", "stock_price_cagr_5_years",
        ] if c in sf.columns]
        sf = sf[["symbol"] + keep_screener]
        # rename to scr_ prefix to avoid collision
        sf = sf.rename(columns={c: f"scr_{c}" for c in keep_screener})
        df = df.merge(sf, on="symbol", how="left")
        # compute derived: PEG, P/B
        if "scr_pe" in df.columns and "scr_compounded_profit_growth_3_years" in df.columns:
            df["scr_peg_3y"] = df["scr_pe"] / df["scr_compounded_profit_growth_3_years"].replace(0, np.nan)
        if "scr_book_value" in df.columns:
            df["scr_price_to_book"] = df["close"] / df["scr_book_value"].replace(0, np.nan)
        # earnings yield = 1/PE
        if "scr_pe" in df.columns:
            df["scr_earnings_yield"] = 1.0 / df["scr_pe"].replace(0, np.nan)

    # ─── 5c. DERIVED RATIOS (QVM, PEG, Magic Formula etc.) ────────────
    if DERIVED_RATIOS.exists():
        print("  joining derived ratios (QVM, PEG, Magic Formula, Tillinghast) …")
        dr = pd.read_parquet(DERIVED_RATIOS)
        dr["fetch_date"] = pd.to_datetime(dr["fetch_date"])
        dr = dr.sort_values("fetch_date").groupby("symbol").tail(1)
        # only keep the NEW derived columns (not raw screener fields already joined)
        derived_cols = [c for c in dr.columns
                         if c in ("magic_formula_rank", "earnings_yield",
                                  "peg_3y", "peg_5y", "peg_ttm",
                                  "roe_z", "roce_z", "growth5y_z",
                                  "quality_composite",
                                  "pe_inv_z", "book_to_price", "btp_z", "divyld_z",
                                  "value_composite",
                                  "stock_price_cagr_1_year_z",
                                  "stock_price_cagr_3_years_z",
                                  "stock_price_cagr_5_years_z",
                                  "momentum_composite",
                                  "qvm_score", "qvm_rank",
                                  "tillinghast_score",
                                  "roe_growth_fusion",
                                  "mom_x_growth_3y",
                                  "roe_persistence")]
        derived_cols = [c for c in derived_cols if c in dr.columns]
        if derived_cols:
            sub = dr[["symbol"] + derived_cols].rename(columns={c: f"qvm_{c}" for c in derived_cols})
            df = df.merge(sub, on="symbol", how="left")

    # ─── 5d. ACADEMIC ALPHAS (Carhart, BAB, idio-vol, QMJ) ────────────
    if ACADEMIC_ALPHAS.exists():
        print("  joining academic alphas (Carhart, BAB, idio-vol, QMJ) …")
        aa = pd.read_parquet(ACADEMIC_ALPHAS)
        # NOTE: academic_alphas only has TODAY's snapshot. We broadcast it
        # forward as a static feature (same caveat as Screener fundamentals).
        keep_aa = [c for c in aa.columns if c not in ("symbol", "trade_date")]
        sub = aa[["symbol"] + keep_aa].rename(columns={c: f"acad_{c}" for c in keep_aa})
        df = df.merge(sub, on="symbol", how="left")

    # ─── 5e. MACRO PANEL (commodities / global rates / breadth / MF AUM / sentiment) ──
    if MACRO_PANEL.exists():
        print("  joining macro_panel.parquet (commodities, global rates, breadth, MF AUM, macro sentiment) …")
        mp = pd.read_parquet(MACRO_PANEL)
        mp["trade_date"] = pd.to_datetime(mp["trade_date"])
        # everything except trade_date applies identically to all stocks for that date
        # (it's a date-level signal — global regime / commodity / breadth)
        # whitelist: numeric only; skip already-joined cols (FX) to avoid collision
        existing = set(df.columns)
        candidate_cols = []
        for c in mp.columns:
            if c == "trade_date": continue
            if c in existing: continue  # already joined (e.g., usdinr)
            if not pd.api.types.is_numeric_dtype(mp[c]): continue
            candidate_cols.append(c)
        # forward-fill across business days (commodity quotes don't always align with NSE)
        mp = mp.sort_values("trade_date")
        for c in candidate_cols:
            mp[c] = mp[c].ffill(limit=5)
        # rename with macro_ prefix where it doesn't clash (so we can detect them)
        rename_map = {}
        for c in candidate_cols:
            if not c.startswith(("macro_", "rs_60d__", "sector_breadth_50__", "sector_dispersion_20d__")):
                rename_map[c] = f"macro_{c}"
        if rename_map:
            mp = mp.rename(columns=rename_map)
            candidate_cols = [rename_map.get(c, c) for c in candidate_cols]
        df = df.merge(mp[["trade_date"] + candidate_cols], on="trade_date", how="left")
        print(f"    joined {len(candidate_cols)} macro/aggregate columns from macro_panel")

    # ─── 5f. INDUSTRY PANEL (sector-relative strength, per-symbol via sector lookup) ──
    if INDUSTRY_PANEL.exists():
        print("  joining industry_panel.parquet (sector RS / breadth / dispersion) …")
        ip = pd.read_parquet(INDUSTRY_PANEL)
        ip["trade_date"] = pd.to_datetime(ip["trade_date"])
        # need a sector mapping for each symbol — use the same source as fetch_industry_indicators
        sec_map = None
        for src in [ROOT / "data/derived/paper_trading_ledger.parquet",
                    ROOT / "data/derived/confluence_picks.parquet"]:
            if src.exists():
                try:
                    m = pd.read_parquet(src, columns=["symbol", "sector"]).drop_duplicates("symbol")
                    m = m[m["sector"].notna() & (m["sector"] != "")]
                    sec_map = m if sec_map is None else pd.concat([sec_map, m]).drop_duplicates("symbol")
                except Exception:
                    pass
        if sec_map is not None:
            df = df.merge(sec_map, on="symbol", how="left")
            keep_industry = ["sector_breadth_50", "sector_5d_ret", "sector_20d_ret", "sector_60d_ret",
                             "sector_dispersion_20d", "sector_leader_lag_spread",
                             "rs_5d", "rs_20d", "rs_60d", "sector_volume_z_60d"]
            keep_industry = [c for c in keep_industry if c in ip.columns]
            ip_sub = ip[["trade_date", "sector"] + keep_industry].rename(
                columns={c: f"sec_{c}" for c in keep_industry}
            )
            df = df.merge(ip_sub, on=["trade_date", "sector"], how="left")
            df = df.drop(columns=["sector"])
            print(f"    joined {len(keep_industry)} sector-relative columns × symbol-sector lookup ({len(sec_map):,} symbols)")

    # ─── 5g. MACRO × STOCK INTERACTIONS ────────────────────────────────
    # Macro features are constant per date; cross-sectional model can only USE
    # them via INTERACTIONS with stock-level features. We engineer 5 surgical
    # interactions seeded from highest-|IC| non-suspect KEEPs in the macro
    # evaluator output (reports/factor_evaluation_macro.md, 2026-05-01).
    print("  building macro × stock interactions (5 surgical) …")
    # First need adv_rank per date (size proxy for smallcap interaction)
    if "avg_traded_value_20d" in df.columns:
        df["adv_rank_norm"] = df.groupby("trade_date")["avg_traded_value_20d"].rank(
            ascending=False, pct=True)
        # higher rank_norm = lower ADV = smaller cap
    # need stock-level signals
    if "return_20d" not in df.columns:
        df["return_20d_calc"] = df.groupby("symbol")["close"].pct_change(20)
        ret20_col = "return_20d_calc"
    else:
        ret20_col = "return_20d"
    if "above_50dma" not in df.columns and "sma_50" in df.columns:
        df["above_50dma_calc"] = (df["close"] > df["sma_50"]).astype(int)
        above50_col = "above_50dma_calc"
    elif "above_50dma" in df.columns:
        above50_col = "above_50dma"
    else:
        above50_col = None
    if "sma_200" in df.columns:
        df["dist_sma200_calc"] = df["close"] / df["sma_200"] - 1
        distsma200_col = "dist_sma200_calc"
    else:
        distsma200_col = None

    # ─── PRUNED 2026-05-01 after factor_evaluator A/B test ─────────────
    # Tested 5 continuous × continuous (H-INT-1..5) and 5 regime-dummy × stock
    # (H-INT-1b..5b). 9 of 10 DROPPED on cross-sectional IC test.
    # Only ONE survived; keeping it. The 9 noise features were dragging
    # max_score from 0.62 baseline → 0.547 when included en-masse.
    #
    # KEPT (passed: |IC|=0.0495, t=-4.60, decile spread +1.94%, IR 2.97):
    #   macro_int_regimevix_x_rv20  =  regime(VIX_z top-tertile over 252d) × rv_20d
    print("  building macro × stock interactions (1 KEEP only after A/B prune) …")
    def _top_tertile_dummy(s: pd.Series, window: int = 252) -> pd.Series:
        q67 = s.rolling(window, min_periods=60).quantile(2.0/3.0)
        return (s >= q67).astype(float)

    if "macro_us_vix_z_60d" in df.columns and "rv_20d" in df.columns:
        vix_series = df.groupby("trade_date")["macro_us_vix_z_60d"].first().sort_index()
        regime_vix = _top_tertile_dummy(vix_series, 252).rename("regime_vix_spike")
        df = df.merge(regime_vix.reset_index(), on="trade_date", how="left")
        df["macro_int_regimevix_x_rv20"] = df["regime_vix_spike"] * df["rv_20d"]

    int_cols = [c for c in df.columns if c.startswith("macro_int_")]
    print(f"    built {len(int_cols)} interaction features: {int_cols}")

    # ─── 6. WIKIPEDIA ATTENTION ────────────────────────────────────────
    if WIKI.exists():
        print("  joining wikipedia pageviews …")
        w = pd.read_parquet(WIKI)
        w["trade_date"] = pd.to_datetime(w["trade_date"])
        df = df.merge(w[["symbol", "trade_date", "wiki_views", "wiki_views_z"]],
                      on=["symbol", "trade_date"], how="left")

    # subset to only the new feature columns + keys
    new_cols = [c for c in df.columns if c.startswith("alpha_") or c.startswith("vol_") or
                c == "amihud_20d" or c == "turnover_skew_20d" or c.startswith("dow") or
                c.startswith("is_") or c == "dom" or c.endswith("_5d_chg") or c.endswith("_20d_chg") or
                c.startswith("usdinr") or c.startswith("eurinr") or c.startswith("gbpinr") or c.startswith("jpyinr") or
                c.startswith("wiki_") or c == "rv_60d" or c.startswith("scr_") or
                c.startswith("qvm_") or c.startswith("acad_") or
                c.startswith("macro_") or c.startswith("sec_")]   # NEW: macro_panel + industry sector cols
    keep_cols = ["symbol", "trade_date"] + new_cols
    out = df[keep_cols].copy()

    # only keep latest 1.5y for size sanity (OOS 2024-2025)
    out = out[out["trade_date"] >= pd.Timestamp("2023-06-01")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} rows × {len(out.columns)} cols")
    print(f"  new features: {len(new_cols)}")
    print(f"  feature names: {new_cols[:8]} ... ({len(new_cols)} total)")

    # quick coverage
    latest = out["trade_date"].max()
    snap = out[out["trade_date"] == latest]
    print(f"\n  coverage on {latest:%Y-%m-%d} ({len(snap):,} rows):")
    for c in new_cols[:15]:
        if c in snap.columns:
            cov = snap[c].notna().mean()
            print(f"    {c:<28} {cov*100:5.1f}%")


if __name__ == "__main__":
    main()
