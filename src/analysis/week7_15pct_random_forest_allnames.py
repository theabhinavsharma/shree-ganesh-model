from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from src.ml.config import ObjectiveSpec
from src.ml.config import load_research_config
from src.ml.expert_pipeline import _build_calibration_table
from src.ml.expert_pipeline import _squash_return
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
class RFConfig:
    classifier_trees: int = 24
    regressor_trees: int = 12
    max_depth: int = 7
    min_samples_leaf: int = 400
    n_jobs: int = 2
    random_state: int = 42


@dataclass(frozen=True)
class FreshEntryRule:
    data_ok_required: bool = True
    max_return_7td: float = 0.20
    max_return_15td: float = 0.35
    max_return_20d: float = 0.30
    max_return_30td: float = 0.50
    max_rsi_14_daily: float = 78.0
    min_return_15td: float = 0.0
    min_return_30td: float = -0.05
    min_rsi_14_daily: float = 50.0
    require_close_above_sma50: bool = True


def _combine_focus_score(prob_15: np.ndarray, prob_20: np.ndarray, pred_return: np.ndarray) -> np.ndarray:
    return (
        0.55 * prob_15
        + 0.25 * prob_20
        + 0.20 * _squash_return(pred_return)
    ).astype(np.float32)


def _fit_predict_classifier(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: RFConfig,
) -> np.ndarray:
    unique = np.unique(train_y)
    if len(unique) < 2:
        return np.full(len(test_x), float(train_y.mean()), dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=config.classifier_trees,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight="balanced_subsample",
        max_features="sqrt",
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1].astype(np.float32)


def _fit_predict_regressor(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: RFConfig,
) -> np.ndarray:
    if len(train_y) == 0 or float(np.nanstd(train_y)) < 1e-8:
        fill = float(np.nanmean(train_y)) if len(train_y) else 0.0
        return np.full(len(test_x), fill, dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=config.regressor_trees,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features="sqrt",
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )
    model.fit(train_x, train_y)
    return np.clip(model.predict(test_x), -0.25, 0.60).astype(np.float32)


def _apply_calibration_15pct(frame: pd.DataFrame, calibration: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    working = frame.copy()
    if calibration.empty:
        working["calibrated_confidence_15pct_7d"] = pd.NA
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
    working["calibrated_confidence_15pct_7d"] = working["calibration_bin"].map(by_bin["hit_rate"])
    working["calibrated_avg_return_7d"] = working["calibration_bin"].map(by_bin["avg_return"])
    return working


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else np.nan


def _evaluate_daily_metrics(predictions: pd.DataFrame, *, top_n: int) -> dict[str, float | int]:
    total_rows = int(len(predictions))
    winner_count = int(predictions["winner_15pct"].sum())
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
        hits = int(pd.to_numeric(top["winner_15pct"], errors="coerce").fillna(0).sum())
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
        "precision_15pct": precision,
        "precision_lift": float(precision / base_rate) if base_rate and not np.isnan(base_rate) else np.nan,
        "recall": recall,
        "p_value": p_value,
        "base_rate_15pct": base_rate,
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
            "precision_15pct": np.nan,
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
    base_rate = float(predictions["winner_15pct"].mean()) if len(predictions) else np.nan
    precision = float(weekly["winner_15pct"].mean()) if len(weekly) else np.nan

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
        hits = int(pd.to_numeric(group["winner_15pct"], errors="coerce").fillna(0).sum())
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
        "precision_15pct": precision,
        "precision_lift": float(precision / base_rate) if base_rate and not np.isnan(base_rate) else np.nan,
        "avg_winners_per_week": _safe_mean(winner_counts),
        "weeks_with_ge1_winner_rate": float(ge1 / weeks) if weeks else np.nan,
        "weeks_with_ge2_winners_rate": float(ge2 / weeks) if weeks else np.nan,
        "weeks_with_ge3_winners_rate": float(ge3 / weeks) if weeks else np.nan,
        "mean_return_mean": _safe_mean(mean_returns),
        "median_stock_return_median": _safe_median(median_returns),
        "p75_stock_return_median": _safe_median(p75_returns),
    }


def _build_trailing_state(daily_facts_path: Path, *, as_of_trade_date: pd.Timestamp) -> pd.DataFrame:
    daily = pd.read_parquet(daily_facts_path, columns=["symbol", "trade_date", "close"])
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
    daily = daily.loc[daily["trade_date"].le(as_of_trade_date)].copy()
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["symbol", "trade_date", "close"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    grouped = daily.groupby("symbol", sort=False)["close"]
    for periods in (7, 15, 30):
        lag = grouped.shift(periods)
        daily[f"return_{periods}td"] = (daily["close"] / lag) - 1.0
    daily["sma_50"] = grouped.transform(lambda s: s.rolling(50, min_periods=20).mean())
    daily["sma_200"] = grouped.transform(lambda s: s.rolling(200, min_periods=60).mean())
    latest = daily.loc[daily["trade_date"] == as_of_trade_date, ["symbol", "trade_date", "sma_50", "sma_200", "return_7td", "return_15td", "return_30td"]].copy()
    return latest


def _build_historical_trailing_state(daily_facts_path: Path, *, end_date: pd.Timestamp) -> pd.DataFrame:
    daily = pd.read_parquet(daily_facts_path, columns=["symbol", "trade_date", "close"])
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
    daily = daily.loc[daily["trade_date"].le(end_date)].copy()
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["symbol", "trade_date", "close"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = daily.groupby("symbol", sort=False)["close"]
    for periods in (7, 15, 30):
        lag = grouped.shift(periods)
        daily[f"return_{periods}td"] = (daily["close"] / lag) - 1.0
    daily["sma_50"] = grouped.transform(lambda s: s.rolling(50, min_periods=20).mean())
    daily["sma_200"] = grouped.transform(lambda s: s.rolling(200, min_periods=60).mean())
    return daily[["symbol", "trade_date", "sma_50", "sma_200", "return_7td", "return_15td", "return_30td"]].copy()


def _build_veto_columns(frame: pd.DataFrame, *, rule: FreshEntryRule) -> pd.DataFrame:
    working = frame.copy()
    close = _coerce_numeric_series(working, "close")
    sma_50 = _coerce_numeric_series(working, "sma_50")
    sma_200 = _coerce_numeric_series(working, "sma_200")
    ratio_50 = close / sma_50.replace(0.0, np.nan)
    ratio_200 = close / sma_200.replace(0.0, np.nan)
    ratio_50_ok = ratio_50.between(0.2, 5.0, inclusive="both") | sma_50.isna()
    ratio_200_ok = ratio_200.between(0.2, 5.0, inclusive="both") | sma_200.isna()
    working["data_ok"] = close.gt(0) & ratio_50_ok & ratio_200_ok

    return_7td = pd.to_numeric(working.get("return_7td"), errors="coerce")
    return_15td = pd.to_numeric(working.get("return_15td"), errors="coerce")
    return_20d = pd.to_numeric(working.get("return_20d"), errors="coerce")
    return_30td = pd.to_numeric(working.get("return_30td"), errors="coerce")
    rsi_daily = pd.to_numeric(working.get("rsi_14_daily"), errors="coerce")
    required_fresh_metrics_ok = (
        return_7td.notna()
        & return_15td.notna()
        & return_20d.notna()
        & return_30td.notna()
        & rsi_daily.notna()
    )
    if rule.require_close_above_sma50:
        required_fresh_metrics_ok = required_fresh_metrics_ok & sma_50.notna()

    working["data_ok"] = working["data_ok"] & required_fresh_metrics_ok
    veto_note = pd.Series("pass", index=working.index, dtype="object")
    if rule.data_ok_required:
        veto_note = np.where(required_fresh_metrics_ok, veto_note, "missing fresh-entry metrics")
        veto_note = np.where((veto_note == "pass") & ~working["data_ok"], "data consistency veto", veto_note)
    if rule.require_close_above_sma50:
        veto_note = np.where((veto_note == "pass") & close.lt(sma_50), "below 50 DMA", veto_note)
    veto_note = np.where((veto_note == "pass") & return_15td.lt(rule.min_return_15td), f"15d return below {int(rule.min_return_15td * 100)}%", veto_note)
    veto_note = np.where((veto_note == "pass") & return_30td.lt(rule.min_return_30td), f"30d return below {int(rule.min_return_30td * 100)}%", veto_note)
    veto_note = np.where((veto_note == "pass") & rsi_daily.lt(rule.min_rsi_14_daily), f"daily RSI below {rule.min_rsi_14_daily:.0f}", veto_note)
    veto_note = np.where((veto_note == "pass") & return_7td.gt(rule.max_return_7td), f"7d return above {int(rule.max_return_7td * 100)}%", veto_note)
    veto_note = np.where((veto_note == "pass") & return_15td.gt(rule.max_return_15td), f"15d return above {int(rule.max_return_15td * 100)}%", veto_note)
    veto_note = np.where((veto_note == "pass") & return_20d.gt(rule.max_return_20d), f"20d return above {int(rule.max_return_20d * 100)}%", veto_note)
    veto_note = np.where((veto_note == "pass") & return_30td.gt(rule.max_return_30td), f"30d return above {int(rule.max_return_30td * 100)}%", veto_note)
    veto_note = np.where((veto_note == "pass") & rsi_daily.gt(rule.max_rsi_14_daily), f"daily RSI above {rule.max_rsi_14_daily:.0f}", veto_note)
    working["veto_note"] = pd.Series(veto_note, index=working.index)
    working["fresh_entry_pass"] = working["veto_note"].eq("pass")
    return working


def _coerce_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    candidates = [column, f"{column}_y", f"{column}_x"]
    for candidate in candidates:
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _build_rationale(frame: pd.DataFrame) -> pd.Series:
    notes: list[str] = []
    for _, row in frame.iterrows():
        pieces: list[str] = []
        close = pd.to_numeric(row.get("close"), errors="coerce")
        sma_50 = pd.to_numeric(row.get("sma_50"), errors="coerce")
        volume_vs_20d = pd.to_numeric(row.get("volume_vs_20d"), errors="coerce")
        if pd.notna(volume_vs_20d) and volume_vs_20d >= 1.5:
            pieces.append(f"volume {volume_vs_20d:.2f}x 20d")
        rsi_daily = pd.to_numeric(row.get("rsi_14_daily"), errors="coerce")
        if pd.notna(rsi_daily):
            pieces.append(f"daily RSI {rsi_daily:.1f}")
        if pd.notna(close) and pd.notna(sma_50) and close >= sma_50:
            pieces.append("above 50 DMA")
        if bool(row.get("recent_results_flag", False)):
            pieces.append("fresh results")
        if bool(row.get("recent_order_win_flag", False)):
            pieces.append("recent order win")
        if bool(row.get("recent_approval_flag", False)):
            pieces.append("recent approval")
        if bool(row.get("recent_promoter_or_director_buy_flag", False)):
            pieces.append("promoter/director buy")
        if bool(row.get("recent_bulk_buy_flag", False)):
            pieces.append("bulk buy flow")
        notes.append(", ".join(pieces[:5]) if pieces else "high RF rank")
    return pd.Series(notes, index=frame.index, dtype="object")


def _next_trading_day(as_of_trade_date: pd.Timestamp) -> pd.Timestamp:
    candidate = as_of_trade_date + pd.Timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += pd.Timedelta(days=1)
    return candidate


def _jsonify(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonify(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def _select_historical_fresh_entry_basket(
    predictions: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame,
    trailing_state: pd.DataFrame,
    fresh_rule: FreshEntryRule,
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
    ]
    keep_cols = [col for col in keep_cols if col in feature_frame.columns]
    enriched = predictions.merge(feature_frame[keep_cols], on=["symbol", "trade_date"], how="left")
    enriched = enriched.merge(trailing_state, on=["symbol", "trade_date"], how="left")
    parts: list[pd.DataFrame] = []
    for _, group in enriched.groupby("trade_date", sort=False):
        ranked = group.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(top_candidate_pool).copy()
        ranked = _build_veto_columns(ranked, rule=fresh_rule)
        ranked["rationale"] = _build_rationale(ranked)
        survivors = ranked.loc[ranked["fresh_entry_pass"]].copy()
        if survivors.empty:
            continue
        survivors["post_veto_rank"] = np.arange(1, len(survivors) + 1)
        parts.append(survivors)
    if not parts:
        return pd.DataFrame(columns=list(predictions.columns) + ["post_veto_rank", "fresh_entry_pass", "veto_note"])
    return pd.concat(parts, ignore_index=True)


def _evaluate_selected_daily_metrics(
    selected: pd.DataFrame,
    all_predictions: pd.DataFrame,
    *,
    top_n: int,
) -> dict[str, float | int]:
    total_rows = int(len(all_predictions))
    winner_count = int(all_predictions["winner_15pct"].sum())
    base_rate = float(winner_count / total_rows) if total_rows else np.nan
    total_days = int(all_predictions["trade_date"].nunique())

    selected_rows = 0
    selected_hits = 0
    day_frames: list[pd.DataFrame] = []
    for _, group in selected.groupby("trade_date", sort=False):
        top = group.sort_values(["post_veto_rank", "symbol"], ascending=[True, True]).head(top_n).copy()
        if top.empty:
            continue
        day_frames.append(top)
    chosen = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame(columns=selected.columns)
    selected_rows = int(len(chosen))
    selected_hits = int(pd.to_numeric(chosen.get("winner_15pct"), errors="coerce").fillna(0).sum()) if len(chosen) else 0

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
        winners_by_day[trade_date] = int(pd.to_numeric(group["winner_15pct"], errors="coerce").fillna(0).sum())

    all_days = sorted(pd.to_datetime(all_predictions["trade_date"]).dropna().unique())
    mean_returns = [mean_by_day.get(day, 0.0) for day in all_days]
    median_returns = [median_by_day.get(day, 0.0) for day in all_days]
    p75_returns = [p75_by_day.get(day, 0.0) for day in all_days]
    winner_counts = [winners_by_day.get(day, 0) for day in all_days]
    ge1 = sum(int(count >= 1) for count in winner_counts)
    ge2 = sum(int(count >= 2) for count in winner_counts)

    precision = float(selected_hits / selected_rows) if selected_rows else np.nan
    recall = float(selected_hits / winner_count) if winner_count else np.nan
    ci_low, ci_high = _wilson_interval(selected_hits, selected_rows)
    p_value = float(binomtest(selected_hits, selected_rows, p=base_rate, alternative="greater").pvalue) if selected_rows else np.nan
    return {
        "top_n": top_n,
        "precision_15pct": precision,
        "precision_lift": float(precision / base_rate) if base_rate and not np.isnan(base_rate) else np.nan,
        "recall": recall,
        "p_value": p_value,
        "base_rate_15pct": base_rate,
        "mean_return_mean": _safe_mean(mean_returns),
        "median_stock_return_median": _safe_median(median_returns),
        "p75_stock_return_median": _safe_median(p75_returns),
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
    total_base = float(all_predictions["winner_15pct"].mean()) if len(all_predictions) else np.nan
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

    precision = float(pd.to_numeric(chosen.get("winner_15pct"), errors="coerce").fillna(0).mean()) if len(chosen) else np.nan
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
        winners_by_week[week] = int(pd.to_numeric(group["winner_15pct"], errors="coerce").fillna(0).sum())

    mean_returns = [mean_by_week.get(week, 0.0) for week in weeks]
    median_returns = [median_by_week.get(week, 0.0) for week in weeks]
    p75_returns = [p75_by_week.get(week, 0.0) for week in weeks]
    winner_counts = [winners_by_week.get(week, 0) for week in weeks]
    ge1 = sum(int(count >= 1) for count in winner_counts)
    ge2 = sum(int(count >= 2) for count in winner_counts)
    ge3 = sum(int(count >= 3) for count in winner_counts)
    return {
        "top_n": top_n,
        "weeks": len(weeks),
        "precision_15pct": precision,
        "precision_lift": float(precision / total_base) if total_base and not np.isnan(total_base) else np.nan,
        "avg_winners_per_week": _safe_mean(winner_counts),
        "weeks_with_ge1_winner_rate": float(ge1 / len(weeks)) if weeks else np.nan,
        "weeks_with_ge2_winners_rate": float(ge2 / len(weeks)) if weeks else np.nan,
        "weeks_with_ge3_winners_rate": float(ge3 / len(weeks)) if weeks else np.nan,
        "mean_return_mean": _safe_mean(mean_returns),
        "median_stock_return_median": _safe_median(median_returns),
        "p75_stock_return_median": _safe_median(p75_returns),
    }


def run_rf_week7_15pct_allnames(
    *,
    config_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    top_candidate_pool: int,
    force_panel: bool,
    rf_config: RFConfig,
    fresh_rule: FreshEntryRule,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_research_config(config_path)
    objective = ObjectiveSpec(
        name="week_7_15pct_rf_allnames",
        horizon_days=7,
        target_return=0.15,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )

    panel_full, panel_path = prepare_feature_panel(base_config, objective, force=force_panel)
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

        prob_15 = _fit_predict_classifier(
            train_x=train_x,
            train_y=(train_return >= 0.15).astype(np.int8),
            test_x=test_x,
            config=rf_config,
        )
        prob_20 = _fit_predict_classifier(
            train_x=train_x,
            train_y=(train_return >= 0.20).astype(np.int8),
            test_x=test_x,
            config=rf_config,
        )
        pred_return = _fit_predict_regressor(
            train_x=train_x,
            train_y=train_return,
            test_x=test_x,
            config=rf_config,
        )
        scored = test[["trade_date", "symbol", "forward_return"]].copy()
        scored["winner_15pct"] = test_return >= 0.15
        scored["winner_20pct"] = test_return >= 0.20
        scored["prob_15pct_7d"] = prob_15
        scored["prob_20pct_7d"] = prob_20
        scored["pred_return_7d"] = pred_return
        scored["focus_score"] = _combine_focus_score(prob_15, prob_20, pred_return)
        oof_parts.append(scored)

    if not oof_parts:
        raise RuntimeError("No walk-forward folds produced predictions for the RF 7D 15% study.")

    predictions = pd.concat(oof_parts, ignore_index=True).sort_values(["trade_date", "focus_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
    calibration = _build_calibration_table(
        predictions,
        score_col="focus_score",
        target_col="winner_15pct",
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
    ]
    feature_keep_cols = [col for col in feature_keep_cols if col in panel_backtest.columns]
    historical_feature_frame = panel_backtest[feature_keep_cols].copy()
    historical_trailing = _build_historical_trailing_state(
        base_config.paths.daily_facts,
        end_date=backtest_cutoff,
    )
    selected_historical = _select_historical_fresh_entry_basket(
        predictions,
        feature_frame=historical_feature_frame,
        trailing_state=historical_trailing,
        fresh_rule=fresh_rule,
        top_candidate_pool=top_candidate_pool,
    )
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
    prob15_model_y = (train_return >= 0.15).astype(np.int8)
    prob20_model_y = (train_return >= 0.20).astype(np.int8)

    current = build_current_feature_slice(base_config)
    as_of_trade_date = pd.to_datetime(current["trade_date"]).max().normalize()
    current = current.loc[pd.to_datetime(current["trade_date"]).eq(as_of_trade_date)].copy()
    current_x = transform_frame(current, stats)
    current["prob_15pct_7d"] = _fit_predict_classifier(
        train_x=train_x,
        train_y=prob15_model_y,
        test_x=current_x,
        config=rf_config,
    )
    current["prob_20pct_7d"] = _fit_predict_classifier(
        train_x=train_x,
        train_y=prob20_model_y,
        test_x=current_x,
        config=rf_config,
    )
    current["pred_return_7d"] = _fit_predict_regressor(
        train_x=train_x,
        train_y=train_return,
        test_x=current_x,
        config=rf_config,
    )
    current["pred_price_7d"] = pd.to_numeric(current["close"], errors="coerce") * (1.0 + pd.to_numeric(current["pred_return_7d"], errors="coerce").fillna(0.0))
    current["focus_score"] = _combine_focus_score(
        current["prob_15pct_7d"].to_numpy(dtype=np.float32),
        current["prob_20pct_7d"].to_numpy(dtype=np.float32),
        current["pred_return_7d"].to_numpy(dtype=np.float32),
    )
    current = _apply_calibration_15pct(current, calibration, score_col="focus_score")
    current = current.sort_values(["focus_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    current["shortlist_rank"] = np.arange(1, len(current) + 1)
    current["confidence_score"] = (current["focus_score"].rank(method="first", pct=True) * 100.0).round(1)
    current["target_15pct_price"] = pd.to_numeric(current["close"], errors="coerce") * 1.15
    current["target_10pct_price"] = pd.to_numeric(current["close"], errors="coerce") * 1.10
    current["stop_loss_price"] = pd.to_numeric(current["close"], errors="coerce") * 0.95

    trailing = _build_trailing_state(base_config.paths.daily_facts, as_of_trade_date=as_of_trade_date)
    current = current.merge(trailing, on=["symbol", "trade_date"], how="left")
    current = _build_veto_columns(current, rule=fresh_rule)
    current["rationale"] = _build_rationale(current)

    ranked_pool = current.head(top_candidate_pool).copy()
    survivors = ranked_pool.loc[ranked_pool["fresh_entry_pass"]].copy().reset_index(drop=True)
    survivors["selected_universe"] = "all_names"
    top5 = survivors.head(5).copy()
    top10 = survivors.head(10).copy()
    rejected = ranked_pool.loc[~ranked_pool["fresh_entry_pass"]].copy().reset_index(drop=True)

    top5["allocation_pct"] = [round(x, 2) for x in np.repeat(100.0 / max(len(top5), 1), len(top5))] if len(top5) else []
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
    daily_path = output_dir / "fresh_entry_daily_topn_backtest.csv"
    weekly_path = output_dir / "fresh_entry_weekly_topn_backtest.csv"
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
        "macro_risk_on_flag",
        "macro_vix_below_20",
    ]
    macro_snapshot = {}
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

    rejected_counts = rejected["veto_note"].value_counts(dropna=False).to_dict()
    summary = {
        "status": "ok",
        "as_of_trade_date": str(as_of_trade_date.date()),
        "trade_for_date": str(_next_trading_day(as_of_trade_date).date()),
        "run_type": "random_forest_week7_15pct_allnames_fresh_entry",
        "historical_window": {
            "analysis_start_date": analysis_start_date,
            "backtest_end_date": "2025-12-31",
            "final_training_panel_end_date": str(pd.to_datetime(panel_full["trade_date"]).max().date()),
            "min_price": min_price,
        },
        "rf_config": rf_config.__dict__,
        "fresh_entry_rule": fresh_rule.__dict__,
        "source_freshness": source_freshness,
        "macro_snapshot": macro_snapshot,
        "raw_top5_backtest": raw_daily_backtest.loc[raw_daily_backtest["top_n"] == 5].iloc[0].to_dict(),
        "raw_top10_backtest": raw_daily_backtest.loc[raw_daily_backtest["top_n"] == 10].iloc[0].to_dict(),
        "raw_weekly_top5_backtest": raw_weekly_backtest.loc[raw_weekly_backtest["top_n"] == 5].iloc[0].to_dict(),
        "raw_weekly_top10_backtest": raw_weekly_backtest.loc[raw_weekly_backtest["top_n"] == 10].iloc[0].to_dict(),
        "fresh_entry_top5_backtest": overlay_daily_backtest.loc[overlay_daily_backtest["top_n"] == 5].iloc[0].to_dict(),
        "fresh_entry_top10_backtest": overlay_daily_backtest.loc[overlay_daily_backtest["top_n"] == 10].iloc[0].to_dict(),
        "fresh_entry_weekly_top5_backtest": overlay_weekly_backtest.loc[overlay_weekly_backtest["top_n"] == 5].iloc[0].to_dict(),
        "fresh_entry_weekly_top10_backtest": overlay_weekly_backtest.loc[overlay_weekly_backtest["top_n"] == 10].iloc[0].to_dict(),
        "current_candidate_counts": {
            "ranked_rows": int(len(ranked_pool)),
            "fresh_entry_pass_rows": int(len(survivors)),
            "fresh_entry_fail_rows": int(len(rejected)),
            "top5_rows": int(len(top5)),
            "top10_rows": int(len(top10)),
        },
        "historical_candidate_counts": {
            "ranked_prediction_rows": int(len(predictions)),
            "fresh_entry_pass_rows": int(len(selected_historical)),
            "fresh_entry_fail_rows": int(len(predictions) - len(selected_historical)),
        },
        "rejected_counts": rejected_counts,
        "retrospective_improvements": [
            "Blocked stale side-input reuse by requiring refreshed market, macro, announcements, and event_daily layers through the same as-of date.",
            "Removed universe preselection. Every stock is scored inside all_names before any fresh-entry veto is applied.",
            "Added explicit anti-bloat vetoes on trailing 7, 15, 20, and 30 trading-day returns plus daily RSI.",
            "Added data consistency veto using close-to-SMA ratios to catch corporate-action or adjustment anomalies.",
            "Kept missing derivatives OI explicit instead of fabricating or silently substituting it.",
        ],
        "notes": [
            "All stocks were scored first with the Random Forest route, then fresh-entry vetoes were applied to the top ranked candidate pool.",
            "The calibrated 15 percent hit rate is the honest confidence proxy; raw RF probabilities are still ranking signals.",
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
            generated_by="src/analysis/week7_15pct_random_forest_allnames.py",
            as_of_date=str(as_of_trade_date.date()),
            extra_notes=["All-stocks Random Forest 7-day 15 percent study with fresh-entry vetoes."],
        )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_15pct_random_forest_allnames.py",
        as_of_date=str(as_of_trade_date.date()),
        extra_notes=["Official NSE market-data path only. Derivatives OI remained unavailable and was not fabricated."],
    )
    write_report_directory_readme(
        output_dir,
        title="Week 7 Fifteen Percent Random Forest All Names",
        intro_lines=[
            "This folder contains the no-universe 7-day 15 percent Random Forest study with the refreshed official source stack.",
            "All stocks were ranked first, then the fresh-entry vetoes removed already-bloated or inconsistent candidates.",
            "Open `summary.json` first, then `current_shortlist_top10.csv`, `rejected_candidates.csv`, and the fresh-entry daily/weekly backtest tables.",
            "The raw pre-veto backtests are also included for comparison, but the fresh-entry tables are the correct benchmark for the live shortlist.",
        ],
        files=[summary_path, top5_path, top10_path, rejected_path, raw_path, raw_daily_path, raw_weekly_path, daily_path, weekly_path],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the all-stocks 7-day 15 percent Random Forest study and produce a fresh-entry-only shortlist.")
    parser.add_argument("--config", default="configs/ml_research.yaml")
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2026-04-21")
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--output-dir", default="reports/week7_15pct_rf_allnames_20260421_for_20260422")
    parser.add_argument("--top-candidate-pool", type=int, default=50)
    parser.add_argument("--classifier-trees", type=int, default=24)
    parser.add_argument("--regressor-trees", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--min-samples-leaf", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()

    summary = run_rf_week7_15pct_allnames(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        min_price=args.min_price,
        top_candidate_pool=args.top_candidate_pool,
        force_panel=args.force_panel,
        rf_config=RFConfig(
            classifier_trees=args.classifier_trees,
            regressor_trees=args.regressor_trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=args.n_jobs,
        ),
        fresh_rule=FreshEntryRule(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
