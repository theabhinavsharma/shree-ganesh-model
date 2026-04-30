from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from scipy.stats import binomtest
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from src.analysis.week7_15pct_random_forest_allnames import FreshEntryRule
from src.analysis.week7_15pct_random_forest_allnames import _build_historical_trailing_state
from src.analysis.week7_15pct_random_forest_allnames import _build_rationale
from src.analysis.week7_15pct_random_forest_allnames import _build_trailing_state
from src.analysis.week7_15pct_random_forest_allnames import _build_veto_columns
from src.analysis.week7_15pct_random_forest_allnames import _coerce_numeric_series
from src.analysis.week7_15pct_random_forest_allnames import _jsonify
from src.analysis.week7_15pct_random_forest_allnames import _next_trading_day
from src.analysis.week7_15pct_random_forest_allnames import _safe_mean
from src.analysis.week7_15pct_random_forest_allnames import _safe_median
from src.ml.config import ObjectiveSpec
from src.ml.config import load_research_config
from src.ml.expert_pipeline import _build_calibration_table
from src.ml.expert_pipeline import _combine_focus_score
from src.ml.expert_pipeline import _wilson_interval
from src.ml.feature_registry import available_feature_columns
from src.ml.panel import build_current_feature_slice
from src.ml.panel import prepare_feature_panel
from src.ml.preprocess import fit_preprocess
from src.ml.preprocess import transform_frame
from src.ml.walk_forward import build_yearly_walk_forward_folds
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme
from src.utils.io import write_json


@dataclass(frozen=True)
class GBMConfig:
    model_name: str = "xgboost"
    classifier_trees: int = 300
    regressor_trees: int = 250
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 8.0
    min_samples_leaf: int = 200
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    num_leaves: int = 63
    n_jobs: int = 4
    random_state: int = 42


@dataclass(frozen=True)
class MacroVetoRule:
    require_complete_macro: bool = True
    selective_min_breadth_50: float = 0.70
    selective_min_breadth_200: float = 0.30
    selective_min_breadth_volume: float = 0.10
    selective_min_market_median_return_20d: float = 0.03
    selective_min_nifty500_return_20d: float = -0.08
    selective_max_india_vix_return_20d: float = 0.35


def _fit_predict_classifier(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: GBMConfig,
) -> np.ndarray:
    unique = np.unique(train_y)
    if len(unique) < 2:
        return np.full(len(test_x), float(train_y.mean()), dtype=np.float32)
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    scale_pos_weight = float(negatives / positives) if positives > 0 else 1.0
    if config.model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=config.classifier_trees,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            scale_pos_weight=scale_pos_weight,
        )
    elif config.model_name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=config.classifier_trees,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_child_samples=config.min_samples_leaf,
            num_leaves=config.num_leaves,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            objective="binary",
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            is_unbalance=True,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unsupported model_name: {config.model_name}")
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1].astype(np.float32)


def _fit_predict_regressor(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: GBMConfig,
) -> np.ndarray:
    if len(train_y) == 0 or float(np.nanstd(train_y)) < 1e-8:
        fill = float(np.nanmean(train_y)) if len(train_y) else 0.0
        return np.full(len(test_x), fill, dtype=np.float32)
    if config.model_name == "xgboost":
        model = XGBRegressor(
            n_estimators=config.regressor_trees,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_child_weight=config.min_child_weight,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )
    elif config.model_name == "lightgbm":
        model = LGBMRegressor(
            n_estimators=config.regressor_trees,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_child_samples=config.min_samples_leaf,
            num_leaves=config.num_leaves,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            objective="regression",
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unsupported model_name: {config.model_name}")
    model.fit(train_x, train_y)
    return np.clip(model.predict(test_x), -0.25, 0.35).astype(np.float32)


def _apply_calibration_5pct(frame: pd.DataFrame, calibration: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    working = frame.copy()
    if calibration.empty:
        working["calibrated_confidence_5pct_7d"] = pd.NA
        working["calibrated_avg_return_7d"] = pd.NA
        working["calibration_bin"] = pd.NA
        return working
    calibration = calibration.sort_values("score_min").reset_index(drop=True)
    score_mins = calibration["score_min"].to_numpy(dtype=float)
    bins = calibration["calibration_bin"].to_numpy(dtype=int)
    positions = np.searchsorted(
        score_mins,
        pd.to_numeric(working[score_col], errors="coerce").fillna(score_mins[0]).to_numpy(dtype=float),
        side="right",
    ) - 1
    positions = np.clip(positions, 0, len(calibration) - 1)
    working["calibration_bin"] = bins[positions]
    by_bin = calibration.set_index("calibration_bin")
    working["calibrated_confidence_5pct_7d"] = working["calibration_bin"].map(by_bin["hit_rate"])
    working["calibrated_avg_return_7d"] = working["calibration_bin"].map(by_bin["avg_return"])
    return working


def _apply_screened_calibration_5pct(frame: pd.DataFrame, calibration: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    working = frame.copy()
    if calibration.empty:
        working["screened_calibrated_confidence_5pct_7d"] = pd.NA
        working["screened_calibrated_avg_return_7d"] = pd.NA
        working["screened_calibration_bin"] = pd.NA
        return working
    calibration = calibration.sort_values("score_min").reset_index(drop=True)
    score_mins = calibration["score_min"].to_numpy(dtype=float)
    bins = calibration["calibration_bin"].to_numpy(dtype=int)
    positions = np.searchsorted(
        score_mins,
        pd.to_numeric(working[score_col], errors="coerce").fillna(score_mins[0]).to_numpy(dtype=float),
        side="right",
    ) - 1
    positions = np.clip(positions, 0, len(calibration) - 1)
    working["screened_calibration_bin"] = bins[positions]
    by_bin = calibration.set_index("calibration_bin")
    working["screened_calibrated_confidence_5pct_7d"] = working["screened_calibration_bin"].map(by_bin["hit_rate"])
    working["screened_calibrated_avg_return_7d"] = working["screened_calibration_bin"].map(by_bin["avg_return"])
    return working


def _rerank_screened_population(frame: pd.DataFrame, *, rank_col: str) -> pd.DataFrame:
    working = frame.copy()
    if working.empty:
        working[rank_col] = []
        return working
    sort_cols = [
        "screened_calibrated_confidence_5pct_7d",
        "screened_calibrated_avg_return_7d",
        "focus_score",
        "symbol",
    ]
    ascending = [False, False, False, True]
    if "trade_date" in working.columns:
        parts: list[pd.DataFrame] = []
        for _, group in working.groupby("trade_date", sort=False):
            ranked = group.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
            ranked[rank_col] = np.arange(1, len(ranked) + 1)
            parts.append(ranked)
        return pd.concat(parts, ignore_index=True) if parts else working.assign(**{rank_col: []})
    working = working.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    working[rank_col] = np.arange(1, len(working) + 1)
    return working


def _build_macro_gate_columns(frame: pd.DataFrame, *, rule: MacroVetoRule) -> pd.DataFrame:
    working = frame.copy()
    breadth_50 = _coerce_numeric_series(working, "breadth_above_50_dma")
    breadth_200 = _coerce_numeric_series(working, "breadth_above_200_dma")
    breadth_volume = _coerce_numeric_series(working, "breadth_volume_1_5x")
    market_median = _coerce_numeric_series(working, "market_median_return_20d")
    nifty_500 = _coerce_numeric_series(working, "nifty_500_return_20d")
    india_vix = _coerce_numeric_series(working, "india_vix_return_20d")
    macro_risk_on = working.get("macro_risk_on_flag", pd.Series(False, index=working.index))
    macro_risk_on = pd.Series(macro_risk_on, index=working.index).astype("boolean")

    required_ok = (
        breadth_50.notna()
        & breadth_200.notna()
        & breadth_volume.notna()
        & market_median.notna()
        & nifty_500.notna()
        & india_vix.notna()
        & macro_risk_on.notna()
    )
    selective_risk_on = (
        breadth_50.ge(rule.selective_min_breadth_50)
        & breadth_200.ge(rule.selective_min_breadth_200)
        & breadth_volume.ge(rule.selective_min_breadth_volume)
        & market_median.ge(rule.selective_min_market_median_return_20d)
        & nifty_500.ge(rule.selective_min_nifty500_return_20d)
        & india_vix.le(rule.selective_max_india_vix_return_20d)
    )
    direct_risk_on = macro_risk_on.fillna(False).astype(bool)

    macro_state = np.select(
        [direct_risk_on, selective_risk_on],
        ["risk_on", "selective_risk_on"],
        default="risk_off",
    )
    macro_pass = direct_risk_on | selective_risk_on
    if rule.require_complete_macro:
        macro_pass = macro_pass & required_ok
    working["macro_required_fields_ok"] = required_ok
    working["macro_state"] = pd.Series(macro_state, index=working.index, dtype="object")
    working["macro_gate_pass"] = pd.Series(macro_pass, index=working.index, dtype=bool)
    veto_note = np.where(required_ok, "pass", "missing macro regime metrics")
    veto_note = np.where((veto_note == "pass") & ~macro_pass, "macro veto: risk-off tape", veto_note)
    working["macro_veto_note"] = pd.Series(veto_note, index=working.index, dtype="object")
    return working


def _combine_vetoes(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    final_note = np.where(
        ~working["macro_gate_pass"].astype(bool),
        working["macro_veto_note"].astype(str),
        working["veto_note"].astype(str),
    )
    working["screen_pass"] = working["macro_gate_pass"].astype(bool) & working["fresh_entry_pass"].astype(bool)
    working["screen_note"] = pd.Series(final_note, index=working.index, dtype="object")
    return working


def _build_safe_rationale(frame: pd.DataFrame) -> pd.Series:
    working = frame.copy()
    bool_cols = [
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_promoter_or_director_buy_flag",
        "recent_bulk_buy_flag",
    ]
    for col in bool_cols:
        if col in working.columns:
            working[col] = working[col].astype("boolean").fillna(False).astype(bool)
    return _build_rationale(working)


def _evaluate_daily_metrics(predictions: pd.DataFrame, *, top_n: int) -> dict[str, float | int]:
    total_rows = int(len(predictions))
    winner_count = int(predictions["winner_5pct"].sum())
    base_rate = float(winner_count / total_rows) if total_rows else np.nan
    selected_rows = 0
    selected_hits = 0
    mean_returns: list[float] = []
    median_returns: list[float] = []
    p75_returns: list[float] = []
    ge1 = 0
    ge2 = 0
    winner_counts: list[int] = []

    for _, group in predictions.groupby("trade_date", sort=False):
        top = group.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(top_n).copy()
        returns = pd.to_numeric(top["forward_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        hits = int(pd.to_numeric(top["winner_5pct"], errors="coerce").fillna(0).sum())
        selected_rows += int(len(top))
        selected_hits += hits
        mean_returns.append(float(returns.mean()))
        median_returns.append(float(returns.median()))
        p75_returns.append(float(returns.quantile(0.75)))
        winner_counts.append(hits)
        ge1 += int(hits >= 1)
        ge2 += int(hits >= 2)

    precision = float(selected_hits / selected_rows) if selected_rows else np.nan
    recall = float(selected_hits / winner_count) if winner_count else np.nan
    ci_low, ci_high = _wilson_interval(selected_hits, selected_rows)
    p_value = float(binomtest(selected_hits, selected_rows, p=base_rate, alternative="greater").pvalue) if selected_rows else np.nan
    day_count = len(mean_returns)
    return {
        "top_n": top_n,
        "precision_5pct": precision,
        "precision_lift": float(precision / base_rate) if base_rate and not np.isnan(base_rate) else np.nan,
        "recall": recall,
        "p_value": p_value,
        "base_rate_5pct": base_rate,
        "mean_return_mean": _safe_mean(mean_returns),
        "median_stock_return_median": _safe_median(median_returns),
        "p75_stock_return_median": _safe_median(p75_returns),
        "days_with_ge1_winner_rate": float(ge1 / day_count) if day_count else np.nan,
        "days_with_ge2_winners_rate": float(ge2 / day_count) if day_count else np.nan,
        "avg_winners_per_day": _safe_mean(winner_counts),
    }


def _evaluate_weekly_metrics(predictions: pd.DataFrame, *, top_n: int) -> dict[str, float | int]:
    working = predictions.copy()
    iso = pd.to_datetime(working["trade_date"]).dt.isocalendar()
    working["year_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    first_days = working.groupby("year_week", sort=False)["trade_date"].min().reset_index()
    weekly_frames: list[pd.DataFrame] = []
    for _, row in first_days.iterrows():
        group = working.loc[working["trade_date"] == row["trade_date"]].copy()
        if group.empty:
            continue
        top = group.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(top_n).copy()
        top["year_week"] = row["year_week"]
        weekly_frames.append(top)
    if not weekly_frames:
        return {
            "top_n": top_n,
            "weeks": 0,
            "precision_5pct": np.nan,
            "precision_lift": np.nan,
            "avg_winners_per_week": np.nan,
            "weeks_with_ge1_winner_rate": np.nan,
            "weeks_with_ge2_winners_rate": np.nan,
            "weeks_with_ge3_winners_rate": np.nan,
            "mean_return_mean": np.nan,
            "median_stock_return_median": np.nan,
            "p75_stock_return_median": np.nan,
        }
    weekly = pd.concat(weekly_frames, ignore_index=True)
    base_rate = float(predictions["winner_5pct"].mean()) if len(predictions) else np.nan
    precision = float(weekly["winner_5pct"].mean()) if len(weekly) else np.nan

    mean_returns: list[float] = []
    median_returns: list[float] = []
    p75_returns: list[float] = []
    winner_counts: list[int] = []
    ge1 = 0
    ge2 = 0
    ge3 = 0
    weeks = 0
    for _, group in weekly.groupby("year_week", sort=False):
        returns = pd.to_numeric(group["forward_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        hits = int(pd.to_numeric(group["winner_5pct"], errors="coerce").fillna(0).sum())
        weeks += 1
        mean_returns.append(float(returns.mean()))
        median_returns.append(float(returns.median()))
        p75_returns.append(float(returns.quantile(0.75)))
        winner_counts.append(hits)
        ge1 += int(hits >= 1)
        ge2 += int(hits >= 2)
        ge3 += int(hits >= 3)

    return {
        "top_n": top_n,
        "weeks": weeks,
        "precision_5pct": precision,
        "precision_lift": float(precision / base_rate) if base_rate and not np.isnan(base_rate) else np.nan,
        "avg_winners_per_week": _safe_mean(winner_counts),
        "weeks_with_ge1_winner_rate": float(ge1 / weeks) if weeks else np.nan,
        "weeks_with_ge2_winners_rate": float(ge2 / weeks) if weeks else np.nan,
        "weeks_with_ge3_winners_rate": float(ge3 / weeks) if weeks else np.nan,
        "mean_return_mean": _safe_mean(mean_returns),
        "median_stock_return_median": _safe_median(median_returns),
        "p75_stock_return_median": _safe_median(p75_returns),
    }


def _select_historical_macro_fresh_basket(
    predictions: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame,
    trailing_state: pd.DataFrame,
    fresh_rule: FreshEntryRule,
    macro_rule: MacroVetoRule,
    top_candidate_pool: int,
) -> pd.DataFrame:
    keep_cols = [
        "symbol",
        "trade_date",
        "close",
        "return_20d",
        "rsi_14_daily",
        "volume_vs_20d",
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_promoter_or_director_buy_flag",
        "recent_bulk_buy_flag",
        "breadth_above_50_dma",
        "breadth_above_200_dma",
        "breadth_volume_1_5x",
        "market_median_return_20d",
        "nifty_50_return_20d",
        "nifty_500_return_20d",
        "india_vix_return_20d",
        "macro_risk_on_flag",
        "macro_vix_below_20",
    ]
    keep_cols = [col for col in keep_cols if col in feature_frame.columns]
    enriched = predictions.merge(feature_frame[keep_cols], on=["symbol", "trade_date"], how="left")
    enriched = enriched.merge(trailing_state, on=["symbol", "trade_date"], how="left")
    parts: list[pd.DataFrame] = []
    for _, group in enriched.groupby("trade_date", sort=False):
        ranked = group.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(top_candidate_pool).copy()
        ranked = _build_macro_gate_columns(ranked, rule=macro_rule)
        ranked = _build_veto_columns(ranked, rule=fresh_rule)
        ranked = _combine_vetoes(ranked)
        ranked["rationale"] = _build_safe_rationale(ranked)
        survivors = ranked.loc[ranked["screen_pass"]].copy()
        if survivors.empty:
            continue
        survivors["post_veto_rank"] = np.arange(1, len(survivors) + 1)
        parts.append(survivors)
    if not parts:
        return pd.DataFrame(columns=list(predictions.columns) + ["post_veto_rank", "screen_pass", "screen_note"])
    return pd.concat(parts, ignore_index=True)


def _evaluate_selected_daily_metrics(
    selected: pd.DataFrame,
    all_predictions: pd.DataFrame,
    *,
    top_n: int,
) -> dict[str, float | int]:
    total_rows = int(len(all_predictions))
    winner_count = int(all_predictions["winner_5pct"].sum())
    base_rate = float(winner_count / total_rows) if total_rows else np.nan
    total_days = int(all_predictions["trade_date"].nunique())

    day_frames: list[pd.DataFrame] = []
    for _, group in selected.groupby("trade_date", sort=False):
        top = group.sort_values(["post_veto_rank", "symbol"], ascending=[True, True]).head(top_n).copy()
        if top.empty:
            continue
        day_frames.append(top)
    chosen = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame(columns=selected.columns)
    selected_rows = int(len(chosen))
    selected_hits = int(pd.to_numeric(chosen.get("winner_5pct"), errors="coerce").fillna(0).sum()) if len(chosen) else 0

    mean_by_day: dict[pd.Timestamp, float] = {}
    median_by_day: dict[pd.Timestamp, float] = {}
    p75_by_day: dict[pd.Timestamp, float] = {}
    winners_by_day: dict[pd.Timestamp, int] = {}
    for trade_date, group in chosen.groupby("trade_date", sort=False):
        returns = pd.to_numeric(group["forward_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        mean_by_day[trade_date] = float(returns.mean())
        median_by_day[trade_date] = float(returns.median())
        p75_by_day[trade_date] = float(returns.quantile(0.75))
        winners_by_day[trade_date] = int(pd.to_numeric(group["winner_5pct"], errors="coerce").fillna(0).sum())

    all_days = sorted(pd.to_datetime(all_predictions["trade_date"]).dropna().unique())
    mean_returns = [mean_by_day.get(day, 0.0) for day in all_days]
    median_returns = [median_by_day.get(day, 0.0) for day in all_days]
    p75_returns = [p75_by_day.get(day, 0.0) for day in all_days]
    winner_counts = [winners_by_day.get(day, 0) for day in all_days]
    selected_days = sum(int(day in mean_by_day) for day in all_days)
    ge1 = sum(int(count >= 1) for count in winner_counts)
    ge2 = sum(int(count >= 2) for count in winner_counts)

    precision = float(selected_hits / selected_rows) if selected_rows else np.nan
    recall = float(selected_hits / winner_count) if winner_count else np.nan
    ci_low, ci_high = _wilson_interval(selected_hits, selected_rows)
    p_value = float(binomtest(selected_hits, selected_rows, p=base_rate, alternative="greater").pvalue) if selected_rows else np.nan
    return {
        "top_n": top_n,
        "precision_5pct": precision,
        "precision_lift": float(precision / base_rate) if base_rate and not np.isnan(base_rate) else np.nan,
        "recall": recall,
        "p_value": p_value,
        "base_rate_5pct": base_rate,
        "mean_return_mean": _safe_mean(mean_returns),
        "median_stock_return_median": _safe_median(median_returns),
        "p75_stock_return_median": _safe_median(p75_returns),
        "days_selected_rate": float(selected_days / total_days) if total_days else np.nan,
        "days_with_ge1_winner_rate": float(ge1 / total_days) if total_days else np.nan,
        "days_with_ge2_winners_rate": float(ge2 / total_days) if total_days else np.nan,
        "avg_winners_per_day": _safe_mean(winner_counts),
    }


def _evaluate_selected_weekly_metrics(
    selected: pd.DataFrame,
    all_predictions: pd.DataFrame,
    *,
    top_n: int,
) -> dict[str, float | int]:
    total_base = float(all_predictions["winner_5pct"].mean()) if len(all_predictions) else np.nan
    work = selected.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"]).dt.normalize()
    all_work = all_predictions.copy()
    all_work["trade_date"] = pd.to_datetime(all_work["trade_date"]).dt.normalize()
    iso = all_work["trade_date"].dt.isocalendar()
    all_work["year_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    work_iso = work["trade_date"].dt.isocalendar()
    work["year_week"] = work_iso["year"].astype(str) + "-" + work_iso["week"].astype(str).str.zfill(2)

    weeks = sorted(all_work["year_week"].dropna().unique())
    chosen_parts: list[pd.DataFrame] = []
    for week in weeks:
        group = work.loc[work["year_week"] == week].copy()
        if group.empty:
            continue
        first_trade_date = group["trade_date"].min()
        first_group = group.loc[group["trade_date"] == first_trade_date].copy()
        top = first_group.sort_values(["post_veto_rank", "symbol"], ascending=[True, True]).head(top_n).copy()
        if top.empty:
            continue
        top["year_week"] = week
        chosen_parts.append(top)
    chosen = pd.concat(chosen_parts, ignore_index=True) if chosen_parts else pd.DataFrame(columns=work.columns)

    precision = float(pd.to_numeric(chosen.get("winner_5pct"), errors="coerce").fillna(0).mean()) if len(chosen) else np.nan
    mean_by_week: dict[str, float] = {}
    median_by_week: dict[str, float] = {}
    p75_by_week: dict[str, float] = {}
    winners_by_week: dict[str, int] = {}
    for week, group in chosen.groupby("year_week", sort=False):
        returns = pd.to_numeric(group["forward_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        mean_by_week[week] = float(returns.mean())
        median_by_week[week] = float(returns.median())
        p75_by_week[week] = float(returns.quantile(0.75))
        winners_by_week[week] = int(pd.to_numeric(group["winner_5pct"], errors="coerce").fillna(0).sum())

    mean_returns = [mean_by_week.get(week, 0.0) for week in weeks]
    median_returns = [median_by_week.get(week, 0.0) for week in weeks]
    p75_returns = [p75_by_week.get(week, 0.0) for week in weeks]
    winner_counts = [winners_by_week.get(week, 0) for week in weeks]
    selected_weeks = sum(int(week in mean_by_week) for week in weeks)
    ge1 = sum(int(count >= 1) for count in winner_counts)
    ge2 = sum(int(count >= 2) for count in winner_counts)
    ge3 = sum(int(count >= 3) for count in winner_counts)
    return {
        "top_n": top_n,
        "weeks": len(weeks),
        "precision_5pct": precision,
        "precision_lift": float(precision / total_base) if total_base and not np.isnan(total_base) else np.nan,
        "weeks_selected_rate": float(selected_weeks / len(weeks)) if weeks else np.nan,
        "avg_winners_per_week": _safe_mean(winner_counts),
        "weeks_with_ge1_winner_rate": float(ge1 / len(weeks)) if weeks else np.nan,
        "weeks_with_ge2_winners_rate": float(ge2 / len(weeks)) if weeks else np.nan,
        "weeks_with_ge3_winners_rate": float(ge3 / len(weeks)) if weeks else np.nan,
        "mean_return_mean": _safe_mean(mean_returns),
        "median_stock_return_median": _safe_median(median_returns),
        "p75_stock_return_median": _safe_median(p75_returns),
    }


def run_gbm_week7_5pct_allnames_macro_veto(
    *,
    config_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    top_candidate_pool: int,
    force_panel: bool,
    gbm_config: GBMConfig,
    fresh_rule: FreshEntryRule,
    macro_rule: MacroVetoRule,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_research_config(config_path)
    objective = ObjectiveSpec(
        name=f"week_7_5pct_{gbm_config.model_name}_allnames_macro_veto",
        horizon_days=7,
        target_return=0.0,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )

    panel_full, _panel_path = prepare_feature_panel(base_config, objective, force=force_panel)
    feature_columns = available_feature_columns(list(panel_full.columns), base_config.feature_columns)
    backtest_cutoff = pd.Timestamp("2025-12-31")
    panel_backtest = panel_full.loc[pd.to_datetime(panel_full["trade_date"]).le(backtest_cutoff)].copy()

    folds = build_yearly_walk_forward_folds(panel_backtest, min_train_end_year=pd.Timestamp(base_config.train_end_date).year)
    oof_parts: list[pd.DataFrame] = []
    for fold in folds:
        train = panel_backtest.loc[pd.to_datetime(panel_backtest["trade_date"]).le(fold.train_end_date)].copy()
        test = panel_backtest.loc[pd.to_datetime(panel_backtest["trade_date"]).between(fold.test_start_date, fold.test_end_date)].copy()
        if len(train) < base_config.min_train_rows or len(test) < base_config.min_test_rows:
            continue
        stats = fit_preprocess(train, feature_columns)
        train_x = transform_frame(train, stats)
        test_x = transform_frame(test, stats)
        train_return = pd.to_numeric(train["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        test_return = pd.to_numeric(test["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        prob_5 = _fit_predict_classifier(
            train_x=train_x,
            train_y=(train_return >= 0.05).astype(np.int8),
            test_x=test_x,
            config=gbm_config,
        )
        prob_10 = _fit_predict_classifier(
            train_x=train_x,
            train_y=(train_return >= 0.10).astype(np.int8),
            test_x=test_x,
            config=gbm_config,
        )
        pred_return = _fit_predict_regressor(
            train_x=train_x,
            train_y=train_return,
            test_x=test_x,
            config=gbm_config,
        )
        scored = test[["trade_date", "symbol", "forward_return"]].copy()
        scored["winner_5pct"] = test_return >= 0.05
        scored["winner_10pct"] = test_return >= 0.10
        scored["prob_5pct_7d"] = prob_5
        scored["prob_10pct_7d"] = prob_10
        scored["pred_return_7d"] = pred_return
        scored["focus_score"] = _combine_focus_score(prob_5, prob_10, pred_return)
        oof_parts.append(scored)

    if not oof_parts:
        raise RuntimeError("No walk-forward folds produced predictions for the GBM 7D 5% study.")

    predictions = (
        pd.concat(oof_parts, ignore_index=True)
        .sort_values(["trade_date", "focus_score", "symbol"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    calibration = _build_calibration_table(
        predictions,
        score_col="focus_score",
        target_col="winner_5pct",
        return_col="forward_return",
        bins=10,
    )

    raw_daily_backtest = pd.DataFrame([_evaluate_daily_metrics(predictions, top_n=5), _evaluate_daily_metrics(predictions, top_n=10)])
    raw_weekly_backtest = pd.DataFrame([_evaluate_weekly_metrics(predictions, top_n=5), _evaluate_weekly_metrics(predictions, top_n=10)])

    feature_keep_cols = [
        "symbol",
        "trade_date",
        "close",
        "return_20d",
        "rsi_14_daily",
        "volume_vs_20d",
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_promoter_or_director_buy_flag",
        "recent_bulk_buy_flag",
        "breadth_above_50_dma",
        "breadth_above_200_dma",
        "breadth_volume_1_5x",
        "market_median_return_20d",
        "nifty_50_return_20d",
        "nifty_500_return_20d",
        "india_vix_return_20d",
        "macro_risk_on_flag",
        "macro_vix_below_20",
    ]
    feature_keep_cols = [col for col in feature_keep_cols if col in panel_backtest.columns]
    historical_feature_frame = panel_backtest[feature_keep_cols].copy()
    historical_trailing = _build_historical_trailing_state(base_config.paths.daily_facts, end_date=backtest_cutoff)
    selected_historical = _select_historical_macro_fresh_basket(
        predictions,
        feature_frame=historical_feature_frame,
        trailing_state=historical_trailing,
        fresh_rule=fresh_rule,
        macro_rule=macro_rule,
        top_candidate_pool=top_candidate_pool,
    )
    screened_calibration = _build_calibration_table(
        selected_historical,
        score_col="focus_score",
        target_col="winner_5pct",
        return_col="forward_return",
        bins=10,
    )
    selected_historical = _apply_screened_calibration_5pct(
        selected_historical,
        screened_calibration,
        score_col="focus_score",
    )
    selected_historical = _rerank_screened_population(selected_historical, rank_col="post_veto_rank")
    overlay_daily_backtest = pd.DataFrame(
        [
            _evaluate_selected_daily_metrics(selected_historical, predictions, top_n=5),
            _evaluate_selected_daily_metrics(selected_historical, predictions, top_n=10),
        ]
    )
    overlay_weekly_backtest = pd.DataFrame(
        [
            _evaluate_selected_weekly_metrics(selected_historical, predictions, top_n=5),
            _evaluate_selected_weekly_metrics(selected_historical, predictions, top_n=10),
        ]
    )

    final_train = panel_full.copy()
    stats = fit_preprocess(final_train, feature_columns)
    train_x = transform_frame(final_train, stats)
    train_return = pd.to_numeric(final_train["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    prob5_model_y = (train_return >= 0.05).astype(np.int8)
    prob10_model_y = (train_return >= 0.10).astype(np.int8)

    current = build_current_feature_slice(base_config)
    as_of_trade_date = pd.to_datetime(current["trade_date"]).max().normalize()
    current = current.loc[pd.to_datetime(current["trade_date"]).eq(as_of_trade_date)].copy()
    current_x = transform_frame(current, stats)
    current["prob_5pct_7d"] = _fit_predict_classifier(train_x=train_x, train_y=prob5_model_y, test_x=current_x, config=gbm_config)
    current["prob_10pct_7d"] = _fit_predict_classifier(train_x=train_x, train_y=prob10_model_y, test_x=current_x, config=gbm_config)
    current["pred_return_7d"] = _fit_predict_regressor(train_x=train_x, train_y=train_return, test_x=current_x, config=gbm_config)
    current["pred_price_7d"] = pd.to_numeric(current["close"], errors="coerce") * (
        1.0 + pd.to_numeric(current["pred_return_7d"], errors="coerce").fillna(0.0)
    )
    current["focus_score"] = _combine_focus_score(
        current["prob_5pct_7d"].to_numpy(dtype=np.float32),
        current["prob_10pct_7d"].to_numpy(dtype=np.float32),
        current["pred_return_7d"].to_numpy(dtype=np.float32),
    )
    current = _apply_calibration_5pct(current, calibration, score_col="focus_score")
    current = current.sort_values(["focus_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    current["shortlist_rank"] = np.arange(1, len(current) + 1)
    current["confidence_score"] = (current["focus_score"].rank(method="first", pct=True) * 100.0).round(1)
    current["target_5pct_price"] = pd.to_numeric(current["close"], errors="coerce") * 1.05
    current["target_10pct_price"] = pd.to_numeric(current["close"], errors="coerce") * 1.10

    trailing = _build_trailing_state(base_config.paths.daily_facts, as_of_trade_date=as_of_trade_date)
    current = current.merge(trailing, on=["symbol", "trade_date"], how="left")
    current = _build_macro_gate_columns(current, rule=macro_rule)
    current = _build_veto_columns(current, rule=fresh_rule)
    current = _combine_vetoes(current)
    current["rationale"] = _build_safe_rationale(current)

    ranked_pool = current.head(top_candidate_pool).copy()
    survivors = ranked_pool.loc[ranked_pool["screen_pass"]].copy().reset_index(drop=True)
    survivors = _apply_screened_calibration_5pct(
        survivors,
        screened_calibration,
        score_col="focus_score",
    )
    survivors = _rerank_screened_population(survivors, rank_col="post_veto_rank")
    top5 = survivors.head(5).copy()
    top10 = survivors.head(10).copy()
    rejected = ranked_pool.loc[~ranked_pool["screen_pass"]].copy().reset_index(drop=True)

    if len(top5):
        top5["allocation_pct"] = [round(x, 2) for x in np.repeat(100.0 / len(top5), len(top5))]
    else:
        top5["allocation_pct"] = []
    if len(top10):
        weights = pd.to_numeric(top10["focus_score"], errors="coerce").fillna(0.0)
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(np.repeat(1.0 / len(top10), len(top10)))
        top10["allocation_pct"] = (weights * 100.0).round(2)
    else:
        top10["allocation_pct"] = []

    top5_path = output_dir / "current_shortlist_top5.csv"
    top10_path = output_dir / "current_shortlist_top10.csv"
    rejected_path = output_dir / "rejected_candidates.csv"
    raw_path = output_dir / "raw_ranked_pool.csv"
    raw_daily_path = output_dir / "raw_daily_topn_backtest.csv"
    raw_weekly_path = output_dir / "raw_weekly_topn_backtest.csv"
    daily_path = output_dir / "macro_fresh_daily_topn_backtest.csv"
    weekly_path = output_dir / "macro_fresh_weekly_topn_backtest.csv"
    summary_path = output_dir / "summary.json"

    top5.to_csv(top5_path, index=False)
    top10.to_csv(top10_path, index=False)
    rejected.to_csv(rejected_path, index=False)
    ranked_pool.to_csv(raw_path, index=False)
    raw_daily_backtest.to_csv(raw_daily_path, index=False)
    raw_weekly_backtest.to_csv(raw_weekly_path, index=False)
    overlay_daily_backtest.to_csv(daily_path, index=False)
    overlay_weekly_backtest.to_csv(weekly_path, index=False)

    macro_cols = [
        "breadth_above_50_dma",
        "breadth_above_200_dma",
        "breadth_volume_1_5x",
        "market_median_return_20d",
        "nifty_50_return_20d",
        "nifty_500_return_20d",
        "india_vix_return_20d",
        "macro_risk_on_flag",
        "macro_vix_below_20",
        "macro_state",
        "macro_gate_pass",
    ]
    macro_snapshot: dict[str, object] = {}
    if len(current):
        row = current.iloc[0]
        for col in macro_cols:
            macro_snapshot[col] = None if col not in row.index or pd.isna(row[col]) else row[col]

    source_freshness = {
        "daily_facts_max_trade_date": str(pd.to_datetime(pd.read_parquet(base_config.paths.daily_facts, columns=["trade_date"])["trade_date"]).max().date()),
        "macro_max_trade_date": str(pd.to_datetime(pd.read_parquet(base_config.paths.macro_daily, columns=["trade_date"])["trade_date"]).max().date()) if base_config.paths.macro_daily and base_config.paths.macro_daily.exists() else None,
        "announcements_max_event_date": str(pd.to_datetime(pd.read_parquet(base_config.paths.announcements, columns=["event_date"])["event_date"]).max().date()) if base_config.paths.announcements and base_config.paths.announcements.exists() else None,
        "event_daily_max_trade_date": str(pd.to_datetime(pd.read_parquet(base_config.paths.event_daily, columns=["trade_date"])["trade_date"]).max().date()) if base_config.paths.event_daily and base_config.paths.event_daily.exists() else None,
        "fundamentals_max_effective_date": str(pd.to_datetime(pd.read_parquet(base_config.paths.fundamentals, columns=["effective_from_date"])["effective_from_date"]).max().date()) if base_config.paths.fundamentals and base_config.paths.fundamentals.exists() else None,
        "shareholding_max_effective_date": str(pd.to_datetime(pd.read_parquet(base_config.paths.shareholding, columns=["effective_from_date"])["effective_from_date"]).max().date()) if base_config.paths.shareholding and base_config.paths.shareholding.exists() else None,
        "derivatives_oi_status": "missing_official_source_file",
    }

    rejected_counts = rejected["screen_note"].value_counts(dropna=False).to_dict()
    macro_blocked_today = bool(len(ranked_pool) > 0 and not bool(ranked_pool["macro_gate_pass"].iloc[0]))
    overlay_top5 = overlay_daily_backtest.loc[overlay_daily_backtest["top_n"] == 5].iloc[0]
    historical_mean_gate = float(overlay_top5["mean_return_mean"]) >= 0.01 if pd.notna(overlay_top5["mean_return_mean"]) else False
    historical_median_gate = float(overlay_top5["median_stock_return_median"]) >= 0.003 if pd.notna(overlay_top5["median_stock_return_median"]) else False
    live_calibrated_gate = (
        float(pd.to_numeric(top5.get("screened_calibrated_confidence_5pct_7d"), errors="coerce").median()) >= 0.10
        if len(top5)
        else False
    )
    top10_unique_screened_bins = int(pd.Series(top10.get("screened_calibration_bin")).dropna().nunique()) if len(top10) else 0
    top10_screened_spread = (
        float(
            pd.to_numeric(top10.get("screened_calibrated_confidence_5pct_7d"), errors="coerce").max()
            - pd.to_numeric(top10.get("screened_calibrated_confidence_5pct_7d"), errors="coerce").min()
        )
        if len(top10)
        else np.nan
    )
    calibration_separation_gate = bool(top10_unique_screened_bins >= 2)
    summary = {
        "status": "ok",
        "as_of_trade_date": str(as_of_trade_date.date()),
        "trade_for_date": str(_next_trading_day(as_of_trade_date).date()),
        "run_type": f"{gbm_config.model_name}_week7_5pct_allnames_macro_veto",
        "historical_window": {
            "analysis_start_date": analysis_start_date,
            "backtest_end_date": "2025-12-31",
            "final_training_panel_end_date": str(pd.to_datetime(panel_full["trade_date"]).max().date()),
            "min_price": min_price,
        },
        "gbm_config": gbm_config.__dict__,
        "macro_rule": macro_rule.__dict__,
        "fresh_entry_rule": fresh_rule.__dict__,
        "source_freshness": source_freshness,
        "macro_snapshot": macro_snapshot,
        "promotion_gate": {
            "macro_gate_pass_today": not macro_blocked_today,
            "top5_rows_available": int(len(top5)),
            "historical_top5_mean_ge_1pct": historical_mean_gate,
            "historical_top5_median_ge_0p3pct": historical_median_gate,
            "live_median_calibrated_hit_ge_10pct": live_calibrated_gate,
            "top10_screened_calibration_separates": calibration_separation_gate,
            "top10_screened_calibration_unique_bins": top10_unique_screened_bins,
            "top10_screened_calibration_spread": top10_screened_spread,
        },
        "trade_decision": (
            "TRADEABLE"
            if (
                not macro_blocked_today
                and len(top5) > 0
                and historical_mean_gate
                and historical_median_gate
                and live_calibrated_gate
                and calibration_separation_gate
            )
            else "NO_TRADE"
        ),
        "raw_top5_backtest": raw_daily_backtest.loc[raw_daily_backtest["top_n"] == 5].iloc[0].to_dict(),
        "raw_top10_backtest": raw_daily_backtest.loc[raw_daily_backtest["top_n"] == 10].iloc[0].to_dict(),
        "raw_weekly_top5_backtest": raw_weekly_backtest.loc[raw_weekly_backtest["top_n"] == 5].iloc[0].to_dict(),
        "raw_weekly_top10_backtest": raw_weekly_backtest.loc[raw_weekly_backtest["top_n"] == 10].iloc[0].to_dict(),
        "macro_fresh_top5_backtest": overlay_daily_backtest.loc[overlay_daily_backtest["top_n"] == 5].iloc[0].to_dict(),
        "macro_fresh_top10_backtest": overlay_daily_backtest.loc[overlay_daily_backtest["top_n"] == 10].iloc[0].to_dict(),
        "macro_fresh_weekly_top5_backtest": overlay_weekly_backtest.loc[overlay_weekly_backtest["top_n"] == 5].iloc[0].to_dict(),
        "macro_fresh_weekly_top10_backtest": overlay_weekly_backtest.loc[overlay_weekly_backtest["top_n"] == 10].iloc[0].to_dict(),
        "current_candidate_counts": {
            "ranked_rows": int(len(ranked_pool)),
            "macro_pass_rows": int(pd.to_numeric(ranked_pool.get("macro_gate_pass"), errors="coerce").fillna(0).astype(bool).sum()) if len(ranked_pool) else 0,
            "screen_pass_rows": int(len(survivors)),
            "screen_fail_rows": int(len(rejected)),
            "top5_rows": int(len(top5)),
            "top10_rows": int(len(top10)),
        },
        "historical_candidate_counts": {
            "ranked_prediction_rows": int(len(predictions)),
            "screen_pass_rows": int(len(selected_historical)),
            "screen_fail_rows": int(len(predictions) - len(selected_historical)),
        },
        "rejected_counts": rejected_counts,
        "retrospective_improvements": [
            "Removed universe preselection. Every stock is scored inside all_names before any veto is applied.",
            "Added a macro-first veto that blocks risk-off tapes even before anti-bloat checks are considered.",
            "Kept the anti-bloat layer strict on trailing 7, 15, 20, and 30 trading-day returns plus daily RSI.",
            "Kept the data consistency veto to catch corporate-action or adjustment anomalies.",
            "Kept missing official derivatives OI explicit instead of fabricating or silently substituting it.",
        ],
        "notes": [
            f"All stocks were scored first with the {gbm_config.model_name} route, then macro-first and fresh-entry vetoes were applied to the top ranked candidate pool.",
            "The calibrated 5 percent hit rate is the honest confidence proxy; raw GBM probabilities are ranking signals only.",
        ],
    }
    summary = _jsonify(summary)
    write_json(summary, summary_path)

    for path, df in [
        (top5_path, top5),
        (top10_path, top10),
        (rejected_path, rejected),
        (raw_path, ranked_pool),
        (raw_daily_path, raw_daily_backtest),
        (raw_weekly_path, raw_weekly_backtest),
        (daily_path, overlay_daily_backtest),
        (weekly_path, overlay_weekly_backtest),
    ]:
        write_dataframe_manifest(
            path,
            df,
            generated_by="src/analysis/week7_5pct_gbm_allnames_macro_veto.py",
            as_of_date=str(as_of_trade_date.date()),
            extra_notes=[f"All-stocks {gbm_config.model_name} 7-day 5 percent study with macro-first and anti-bloat vetoes."],
        )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_5pct_gbm_allnames_macro_veto.py",
        as_of_date=str(as_of_trade_date.date()),
        extra_notes=["Official NSE market-data path only. Derivatives OI remained unavailable and was not fabricated."],
    )
    write_report_directory_readme(
        output_dir,
        title=f"Week 7 Five Percent {gbm_config.model_name.title()} All Names Macro Veto",
        intro_lines=[
            f"This folder contains the no-universe 7-day 5 percent {gbm_config.model_name} study with a macro-first veto and anti-bloat layer.",
            "All stocks were ranked first, then the macro regime veto and fresh-entry veto removed risk-off or already-bloated candidates.",
            "The macro_fresh daily and weekly backtests are the correct benchmark for the live shortlist; raw pre-veto backtests are included for comparison.",
        ],
        files=[summary_path, top5_path, top10_path, rejected_path, raw_path, raw_daily_path, raw_weekly_path, daily_path, weekly_path],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the all-stocks 7-day 5 percent GBM study with macro-first veto and anti-bloat screening.")
    parser.add_argument("--config", default="configs/ml_research.yaml")
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2026-04-22")
    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--output-dir", default="reports/week7_5pct_xgboost_allnames_macro_veto")
    parser.add_argument("--model-name", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--top-candidate-pool", type=int, default=100)
    parser.add_argument("--classifier-trees", type=int, default=300)
    parser.add_argument("--regressor-trees", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()

    summary = run_gbm_week7_5pct_allnames_macro_veto(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        min_price=args.min_price,
        top_candidate_pool=args.top_candidate_pool,
        force_panel=args.force_panel,
        gbm_config=GBMConfig(
            model_name=args.model_name,
            classifier_trees=args.classifier_trees,
            regressor_trees=args.regressor_trees,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            min_samples_leaf=args.min_samples_leaf,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            num_leaves=args.num_leaves,
            n_jobs=args.n_jobs,
        ),
        fresh_rule=FreshEntryRule(
            max_return_7td=0.15,
            max_return_15td=0.30,
            max_return_20d=0.25,
            max_return_30td=0.40,
            max_rsi_14_daily=74.0,
        ),
        macro_rule=MacroVetoRule(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
