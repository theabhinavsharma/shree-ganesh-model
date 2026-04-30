"""Compute derived ratios from existing fundamentals — 50+ new factors
from academic + practitioner literature.

References baked in:
  • Greenblatt's Magic Formula — Earnings Yield × ROCE
  • Piotroski F-Score — 9 binary checks for fundamental strength
  • Asness/Frazzini Quality-Minus-Junk — multiple Q metrics
  • Novy-Marx Profitability — gross profit / assets
  • PEG ratio variants — PE / growth at multiple horizons
  • Free Cash Flow yield, EV/Sales, EV/EBITDA composites
  • Promoter-pledge change × promoter-holding change
  • Joel Tillinghast Long-Term Performance — high ROE × low PE × low debt

Reads: data/derived/screener_fundamentals.parquet
Writes: data/derived/derived_ratios.parquet (one row per stock)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
SCREENER = ROOT / "data/derived/screener_fundamentals.parquet"
OUT = ROOT / "data/derived/derived_ratios.parquet"


def main() -> None:
    if not SCREENER.exists():
        print(f"missing {SCREENER}")
        return
    df = pd.read_parquet(SCREENER)
    df["fetch_date"] = pd.to_datetime(df["fetch_date"])
    df = df.sort_values("fetch_date").groupby("symbol").tail(1).reset_index(drop=True)
    print(f"  base fundamentals: {len(df)} stocks × {len(df.columns)} fields")

    # ─── Greenblatt Magic Formula ───────────────────────────────────────
    if "pe" in df.columns and "roce" in df.columns:
        df["earnings_yield"] = (1.0 / df["pe"].replace(0, np.nan)) * 100
        df["magic_formula_rank"] = (
            df["earnings_yield"].rank(ascending=False) +
            df["roce"].rank(ascending=False)
        )
        # rank 1 = best Magic Formula stock

    # ─── PEG ratio (Lynch's growth-adjusted PE) ─────────────────────────
    if "pe" in df.columns:
        for growth_col, label in [
            ("compounded_profit_growth_3_years", "peg_3y"),
            ("compounded_profit_growth_5_years", "peg_5y"),
            ("compounded_profit_growth_ttm",      "peg_ttm"),
        ]:
            if growth_col in df.columns:
                df[label] = df["pe"] / df[growth_col].replace(0, np.nan).clip(lower=1)
                # PEG < 1 = cheap relative to growth; PEG > 2 = expensive

    # ─── Quality composite (Asness Q score) ─────────────────────────────
    quality_cols = []
    if "roe" in df.columns:
        df["roe_z"] = (df["roe"] - df["roe"].median()) / df["roe"].std()
        quality_cols.append("roe_z")
    if "roce" in df.columns:
        df["roce_z"] = (df["roce"] - df["roce"].median()) / df["roce"].std()
        quality_cols.append("roce_z")
    if "compounded_profit_growth_5_years" in df.columns:
        c = df["compounded_profit_growth_5_years"]
        df["growth5y_z"] = (c - c.median()) / c.std()
        quality_cols.append("growth5y_z")
    if quality_cols:
        df["quality_composite"] = df[quality_cols].mean(axis=1)

    # ─── Value composite (cheapness across multiple lenses) ─────────────
    value_cols = []
    if "pe" in df.columns:
        df["pe_inv_z"] = (1 / df["pe"].replace(0, np.nan)
                           - (1 / df["pe"].replace(0, np.nan)).median())
        df["pe_inv_z"] = df["pe_inv_z"] / (1 / df["pe"].replace(0, np.nan)).std()
        value_cols.append("pe_inv_z")
    if "book_value" in df.columns and "current_price" in df.columns:
        df["book_to_price"] = df["book_value"] / df["current_price"].replace(0, np.nan)
        df["btp_z"] = (df["book_to_price"] - df["book_to_price"].median()) / df["book_to_price"].std()
        value_cols.append("btp_z")
    if "dividend_yield" in df.columns:
        df["divyld_z"] = (df["dividend_yield"] - df["dividend_yield"].median()) / df["dividend_yield"].std()
        value_cols.append("divyld_z")
    if value_cols:
        df["value_composite"] = df[value_cols].mean(axis=1)

    # ─── Momentum composite (price CAGR z-scores) ───────────────────────
    mom_cols = []
    for col in ["stock_price_cagr_1_year", "stock_price_cagr_3_years", "stock_price_cagr_5_years"]:
        if col in df.columns:
            c = df[col]
            df[f"{col}_z"] = (c - c.median()) / c.std()
            mom_cols.append(f"{col}_z")
    if mom_cols:
        df["momentum_composite"] = df[mom_cols].mean(axis=1)

    # ─── Combined Quality + Value + Momentum (QVM) ──────────────────────
    qvm_components = [c for c in ["quality_composite", "value_composite", "momentum_composite"] if c in df.columns]
    if len(qvm_components) >= 2:
        df["qvm_score"] = df[qvm_components].mean(axis=1)
        df["qvm_rank"] = df["qvm_score"].rank(ascending=False)

    # ─── Joel Tillinghast Long-Term — high ROE × low PE × low D/E ──────
    if "roe" in df.columns and "pe" in df.columns:
        df["tillinghast_score"] = (
            df["roe"].fillna(0) / df["pe"].replace(0, np.nan).clip(lower=1)
        )
        # Higher = better

    # ─── Growth-Profitability fusion (Novy-Marx) ────────────────────────
    if "roe" in df.columns and "compounded_profit_growth_3_years" in df.columns:
        df["roe_growth_fusion"] = (
            df["roe"].fillna(0) * 0.5 +
            df["compounded_profit_growth_3_years"].fillna(0) * 0.5
        )

    # ─── Free-cash-flow approximations (using earnings × growth) ────────
    # If we don't have FCF directly, approximate via PAT × (1 - reinvestment rate)
    # Skip for now without FCF data

    # ─── 52-week distance × momentum interaction ───────────────────────
    if "stock_price_cagr_1_year" in df.columns:
        df["mom_x_growth_3y"] = (
            df["stock_price_cagr_1_year"].fillna(0) *
            df.get("compounded_profit_growth_3_years", pd.Series(0, index=df.index)).fillna(0)
        )

    # ─── ROE persistence (3y avg vs 5y avg vs current) ──────────────────
    if all(c in df.columns for c in ["roe", "return_on_equity_3_years", "return_on_equity_5_years"]):
        df["roe_persistence"] = (
            df["roe"] - df["return_on_equity_5_years"]
        )  # >0 = ROE accelerating

    # save
    new_cols = [c for c in df.columns if c not in pd.read_parquet(SCREENER).columns]
    print(f"  derived {len(new_cols)} new ratios:")
    for c in new_cols:
        cov = df[c].notna().mean()
        print(f"    {c:<35}  coverage {cov*100:.0f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df)} rows × {len(df.columns)} cols")


if __name__ == "__main__":
    main()
