from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.transform.lagged_join import latest_effective_join


def load_screen_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_daily_screen_universe(
    daily_facts: pd.DataFrame,
    stock_master: pd.DataFrame,
    fundamentals: pd.DataFrame,
    shareholding: pd.DataFrame,
    sector_state_daily: pd.DataFrame,
    config: dict[str, Any],
    *,
    include_missing_inputs: bool = True,
) -> pd.DataFrame:
    universe = daily_facts.merge(stock_master, on="symbol", how="left")
    if not fundamentals.empty:
        universe = latest_effective_join(
            universe,
            fundamentals,
            left_date_col="trade_date",
            right_date_col="effective_from_date",
            by="symbol",
        )
    if not shareholding.empty:
        universe = latest_effective_join(
            universe,
            shareholding,
            left_date_col="trade_date",
            right_date_col="effective_from_date",
            by="symbol",
        )
    if not sector_state_daily.empty and "sector" in universe.columns:
        sector_state_daily = sector_state_daily.rename(columns={"sector_name": "sector"})
        universe = latest_effective_join(
            universe,
            sector_state_daily,
            left_date_col="trade_date",
            right_date_col="trade_date",
            by="sector",
        )
    return apply_screen_filters(universe, config, include_missing_inputs=include_missing_inputs)


def apply_screen_filters(
    universe: pd.DataFrame,
    config: dict[str, Any],
    *,
    include_missing_inputs: bool = True,
) -> pd.DataFrame:
    universe = universe.copy()
    universe = _attach_sector_buying_flags(universe)
    thresholds = config.get("universe", {})
    universe["filter_above_50_dma"] = _apply_required(
        _safe_gt(universe, "close", "sma_50"),
        thresholds.get("require_above_50_dma"),
    )
    universe["filter_above_200_dma"] = _apply_required(
        _safe_gt(universe, "close", "sma_200"),
        thresholds.get("require_above_200_dma"),
    )
    universe["filter_volume_expansion"] = _safe_threshold(universe, "volume_vs_20d", thresholds.get("min_volume_vs_20d"))
    universe["filter_volume_high_3m"] = _safe_required_flag(universe, "volume_high_63d_flag", thresholds.get("require_volume_high_3m"))
    universe["filter_delivery_expansion"] = _safe_required_flag(universe, "delivery_pct_high_63d_flag", thresholds.get("require_delivery_high_3m"))
    universe["filter_delivery_above_5d_avg"] = _safe_required_flag(
        universe,
        "delivery_above_5d_avg_flag",
        thresholds.get("require_delivery_above_5d_avg"),
    )
    universe["filter_rsi_daily"] = _safe_threshold(universe, "rsi_14_daily", thresholds.get("min_rsi_14_daily"))
    universe["filter_rsi_weekly"] = _safe_threshold(universe, "rsi_14_weekly", thresholds.get("min_rsi_14_weekly"))
    universe["filter_rsi_monthly"] = _safe_threshold(universe, "rsi_14_monthly", thresholds.get("min_rsi_14_monthly"))
    universe["filter_rsi"] = _combine_filters(universe, ["filter_rsi_daily", "filter_rsi_weekly", "filter_rsi_monthly"])
    universe["filter_promoter_holding"] = _safe_threshold(universe, "promoter_pct", thresholds.get("min_promoter_pct"))
    universe["filter_market_cap"] = _safe_threshold(universe, "market_cap_cr", thresholds.get("min_market_cap"))
    universe["filter_debt"] = _apply_required(_compute_debt_free_flag(universe), thresholds.get("require_debt_free"))
    universe["filter_sector_institutional_buying"] = _sector_buying_filter(
        universe,
        required=thresholds.get("require_sector_fii_dii_buying_30d"),
    )
    universe["filter_revenue_growth"] = _safe_threshold(universe, "revenue_cagr_5y", thresholds.get("min_revenue_cagr_5y"))
    universe["filter_profit_cagr"] = _safe_threshold(universe, "pat_cagr_5y", thresholds.get("min_pat_cagr_5y"))
    universe["filter_ebitda_positive"] = universe["ebitda_positive_last_5q_flag"] if "ebitda_positive_last_5q_flag" in universe.columns else pd.Series([pd.NA] * len(universe), index=universe.index, dtype="object")
    universe["pe_ttm"] = _compute_pe_ttm(universe)
    universe["filter_pe"] = _safe_lt(universe, "pe_ttm", thresholds.get("max_pe_ttm"))
    universe["strategy_drop_134_pass"] = _combine_filters(
        universe,
        [
            "filter_revenue_growth",
            "filter_profit_cagr",
            "filter_ebitda_positive",
            "filter_volume_expansion",
            "filter_volume_high_3m",
            "filter_delivery_expansion",
            "filter_rsi_daily",
            "filter_rsi_weekly",
            "filter_rsi_monthly",
            "filter_pe",
            "filter_promoter_holding",
            "filter_above_50_dma",
            "filter_above_200_dma",
        ],
    )
    universe["strategy_checklist_pass"] = _combine_filters(
        universe,
        [
            "filter_sector_institutional_buying",
            "filter_market_cap",
            "filter_debt",
            "filter_revenue_growth",
            "filter_profit_cagr",
            "filter_ebitda_positive",
            "filter_volume_expansion",
            "filter_volume_high_3m",
            "filter_delivery_above_5d_avg",
            "filter_rsi_daily",
            "filter_rsi_weekly",
            "filter_rsi_monthly",
            "filter_pe",
            "filter_promoter_holding",
            "filter_above_50_dma",
            "filter_above_200_dma",
        ],
    )
    universe["missing_inputs"] = _missing_inputs(universe) if include_missing_inputs else ""
    return universe


def _safe_gt(df: pd.DataFrame, left: str, right: str) -> pd.Series:
    left_values = pd.to_numeric(df.get(left), errors="coerce")
    right_values = pd.to_numeric(df.get(right), errors="coerce")
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    valid = left_values.notna() & right_values.notna()
    result.loc[valid] = left_values.loc[valid] > right_values.loc[valid]
    return result


def _safe_threshold(df: pd.DataFrame, column: str, threshold: float | None) -> pd.Series:
    if threshold is None or column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    values = pd.to_numeric(df[column], errors="coerce")
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    valid = values.notna()
    result.loc[valid] = values.loc[valid] >= threshold
    return result


def _safe_lt(df: pd.DataFrame, column: str, threshold: float | None) -> pd.Series:
    if threshold is None or column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    values = pd.to_numeric(df[column], errors="coerce")
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    valid = values.notna()
    result.loc[valid] = values.loc[valid] < threshold
    return result


def _safe_required_flag(df: pd.DataFrame, column: str, required: bool | None) -> pd.Series:
    if not required or column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    valid = df[column].notna()
    result.loc[valid] = df.loc[valid, column].astype(bool)
    return result


def _apply_required(series: pd.Series, required: bool | None) -> pd.Series:
    if not required:
        return pd.Series(pd.NA, index=series.index, dtype="boolean")
    return series


def _compute_debt_free_flag(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    any_signal = pd.Series(False, index=df.index, dtype=bool)
    any_positive = pd.Series(False, index=df.index, dtype=bool)
    any_zero = pd.Series(False, index=df.index, dtype=bool)
    numeric_columns = ["debt_equity_ratio", "debt", "net_debt", "face_value_debt", "paid_debt", "debt_redemption"]
    for column in numeric_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        any_signal = any_signal | valid
        any_positive = any_positive | (valid & values.gt(0.01))
        any_zero = any_zero | (valid & values.le(0.01))
    result.loc[any_positive] = False
    result.loc[~any_positive & any_signal & any_zero] = True
    return result


def _compute_pe_ttm(df: pd.DataFrame) -> pd.Series:
    existing = (
        pd.to_numeric(df.get("pe_ttm"), errors="coerce")
        if "pe_ttm" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    close = pd.to_numeric(df.get("close"), errors="coerce")
    eps_ttm = (
        pd.to_numeric(df.get("eps_ttm"), errors="coerce")
        if "eps_ttm" in df.columns
        else pd.Series(np.nan, index=df.index, dtype="float64")
    )
    result = existing.copy()
    valid = result.isna() & close.notna() & eps_ttm.notna() & eps_ttm.gt(0)
    result.loc[valid] = close.loc[valid] / eps_ttm.loc[valid]
    return result


def _combine_filters(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(True, index=df.index, dtype="boolean")
    values = df[columns].copy()
    false_any = values.eq(False).fillna(False).any(axis=1)
    missing_any = values.isna().any(axis=1)
    result = pd.Series(True, index=df.index, dtype="boolean")
    result.loc[false_any] = False
    result.loc[~false_any & missing_any] = pd.NA
    return result


def _attach_sector_buying_flags(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "trade_date" not in enriched.columns:
        return enriched
    exact = pd.Series(pd.NA, index=enriched.index, dtype="boolean")
    if "fii_net_buyer_30d_flag" in enriched.columns and "dii_net_buyer_30d_flag" in enriched.columns:
        fii = enriched["fii_net_buyer_30d_flag"].astype("boolean")
        dii = enriched["dii_net_buyer_30d_flag"].astype("boolean")
        exact = _combine_two_flags(fii, dii)
    elif "sector_fii_dii_buying_exact_flag" in enriched.columns:
        exact = enriched["sector_fii_dii_buying_exact_flag"].astype("boolean")
    enriched["sector_fii_dii_buying_exact_flag"] = exact

    proxy = pd.Series(pd.NA, index=enriched.index, dtype="boolean")
    if {"sector", "fii_fpi_pct_qoq_change", "dii_pct_qoq_change", "trade_date"} <= set(enriched.columns):
        proxy_source = enriched[["trade_date", "sector", "fii_fpi_pct_qoq_change", "dii_pct_qoq_change"]].copy()
        proxy_source["fii_fpi_pct_qoq_change"] = pd.to_numeric(proxy_source["fii_fpi_pct_qoq_change"], errors="coerce")
        proxy_source["dii_pct_qoq_change"] = pd.to_numeric(proxy_source["dii_pct_qoq_change"], errors="coerce")
        proxy_source = proxy_source[proxy_source["sector"].notna()].copy()
        if not proxy_source.empty:
            merge_collision_columns = [
                "sector_fii_qoq_change",
                "sector_dii_qoq_change",
                "sector_fii_dii_buying_proxy_flag",
            ]
            enriched = enriched.drop(columns=[column for column in merge_collision_columns if column in enriched.columns])
            sector_proxy = (
                proxy_source.groupby(["trade_date", "sector"], dropna=False)
                .agg(
                    sector_fii_qoq_change=("fii_fpi_pct_qoq_change", "median"),
                    sector_dii_qoq_change=("dii_pct_qoq_change", "median"),
                )
                .reset_index()
            )
            sector_proxy["sector_fii_dii_buying_proxy_flag"] = (
                sector_proxy["sector_fii_qoq_change"].gt(0) & sector_proxy["sector_dii_qoq_change"].gt(0)
            ).astype("boolean")
            enriched = enriched.merge(
                sector_proxy,
                on=["trade_date", "sector"],
                how="left",
            )
            proxy = enriched["sector_fii_dii_buying_proxy_flag"].astype("boolean")
    enriched["sector_fii_dii_buying_proxy_flag"] = proxy
    return enriched


def _sector_buying_filter(df: pd.DataFrame, *, required: bool | None) -> pd.Series:
    if not required:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    exact = df.get("sector_fii_dii_buying_exact_flag", pd.Series(pd.NA, index=df.index, dtype="boolean")).astype("boolean")
    proxy = df.get("sector_fii_dii_buying_proxy_flag", pd.Series(pd.NA, index=df.index, dtype="boolean")).astype("boolean")
    result = proxy.copy()
    result.loc[exact.notna()] = exact.loc[exact.notna()]
    return result


def _combine_two_flags(left: pd.Series, right: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=left.index, dtype="boolean")
    valid = left.notna() & right.notna()
    result.loc[valid] = left.loc[valid] & right.loc[valid]
    return result


def _missing_inputs(df: pd.DataFrame) -> pd.Series:
    required = [
        "sma_50",
        "sma_200",
        "volume_vs_20d",
        "volume_high_63d_flag",
        "delivery_above_5d_avg_flag",
        "rsi_14_daily",
        "rsi_14_weekly",
        "rsi_14_monthly",
        "promoter_pct",
        "market_cap_cr",
        "revenue_cagr_5y",
        "pat_cagr_5y",
        "ebitda_positive_last_5q_flag",
        "eps_ttm",
    ]
    parts: list[pd.Series] = []
    for column in required:
        if column in df.columns:
            parts.append(df[column].isna().map(lambda missing: f"{column}|" if missing else ""))
        else:
            parts.append(pd.Series(f"{column}|", index=df.index, dtype="object"))
    return pd.concat(parts, axis=1).sum(axis=1).str.rstrip("|")
