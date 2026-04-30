from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.analysis.week7_15pct_gbm_allnames import GBMConfig
from src.analysis.week7_15pct_gbm_allnames import _fit_predict_classifier as _fit_predict_gbm_classifier
from src.analysis.week7_15pct_gbm_allnames import _fit_predict_regressor as _fit_predict_gbm_regressor
from src.analysis.week7_15pct_random_forest_allnames import FreshEntryRule
from src.analysis.week7_15pct_random_forest_allnames import RFConfig
from src.analysis.week7_15pct_random_forest_allnames import _apply_calibration_15pct
from src.analysis.week7_15pct_random_forest_allnames import _build_historical_trailing_state
from src.analysis.week7_15pct_random_forest_allnames import _build_rationale
from src.analysis.week7_15pct_random_forest_allnames import _build_trailing_state
from src.analysis.week7_15pct_random_forest_allnames import _build_veto_columns
from src.analysis.week7_15pct_random_forest_allnames import _combine_focus_score
from src.analysis.week7_15pct_random_forest_allnames import _coerce_numeric_series
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_selected_daily_metrics
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_selected_weekly_metrics
from src.analysis.week7_15pct_random_forest_allnames import _fit_predict_classifier as _fit_predict_rf_classifier
from src.analysis.week7_15pct_random_forest_allnames import _fit_predict_regressor as _fit_predict_rf_regressor
from src.analysis.week7_15pct_random_forest_allnames import _jsonify
from src.analysis.week7_15pct_random_forest_allnames import _next_trading_day
from src.ml.config import ObjectiveSpec
from src.ml.config import load_research_config
from src.ml.expert_pipeline import _build_calibration_table
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
class ClusterOverlayConfig:
    top_candidate_pool: int = 100
    candidate_history_min_rows: int = 500
    candidate_history_min_days: int = 30
    cluster_options: tuple[int, ...] = (5, 7, 9, 11)
    overlay_weight_options: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45)
    shrinkage_rows: int = 120
    refit_every_days: int = 5
    random_state: int = 42


CLUSTER_NUMERIC_FEATURES = [
    "return_7td",
    "return_15td",
    "return_30td",
    "return_20d",
    "rsi_14_daily",
    "rsi_14_weekly",
    "rsi_14_monthly",
    "volume_vs_20d",
    "traded_value_vs_20d",
    "delivery_pct_vs_20d",
    "close_to_sma50",
    "close_to_sma200",
]

CLUSTER_FLAG_FEATURES = [
    "recent_results_flag",
    "recent_order_win_flag",
    "recent_approval_flag",
    "recent_promoter_or_director_buy_flag",
    "recent_bulk_buy_flag",
]


def _make_relaxed_rule() -> FreshEntryRule:
    return FreshEntryRule(
        data_ok_required=True,
        max_return_7td=0.20,
        max_return_15td=0.35,
        max_return_20d=0.30,
        max_return_30td=0.50,
        max_rsi_14_daily=78.0,
        min_return_15td=-1.0,
        min_return_30td=-1.0,
        min_rsi_14_daily=0.0,
        require_close_above_sma50=False,
    )


def _select_fit_predictors(model_name: str):
    if model_name == "random_forest":
        return _fit_predict_rf_classifier, _fit_predict_rf_regressor
    if model_name in {"xgboost", "lightgbm"}:
        return _fit_predict_gbm_classifier, _fit_predict_gbm_regressor
    raise ValueError(f"Unsupported model_name: {model_name}")


def _model_config_dict(model_name: str) -> dict[str, object]:
    if model_name == "random_forest":
        return asdict(RFConfig())
    if model_name == "xgboost":
        return asdict(GBMConfig(model_name="xgboost"))
    if model_name == "lightgbm":
        return asdict(GBMConfig(model_name="lightgbm"))
    raise ValueError(f"Unsupported model_name: {model_name}")


def _build_feature_keep_cols(panel_backtest: pd.DataFrame) -> list[str]:
    keep_cols = [
        "symbol",
        "trade_date",
        "close",
        "return_20d",
        "rsi_14_daily",
        "rsi_14_weekly",
        "rsi_14_monthly",
        "volume_vs_20d",
        "traded_value_vs_20d",
        "delivery_pct_vs_20d",
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_promoter_or_director_buy_flag",
        "recent_bulk_buy_flag",
        "company_name",
        "sector",
        "industry",
        "basic_industry",
        "sma_50",
        "sma_200",
    ]
    return [column for column in keep_cols if column in panel_backtest.columns]


def _merge_enrichment(
    predictions: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame,
    trailing_state: pd.DataFrame,
    fresh_rule: FreshEntryRule,
) -> pd.DataFrame:
    enriched = predictions.merge(feature_frame, on=["symbol", "trade_date"], how="left")
    enriched = enriched.merge(trailing_state, on=["symbol", "trade_date"], how="left")
    close = pd.to_numeric(enriched.get("close"), errors="coerce")
    sma_50 = pd.to_numeric(enriched.get("sma_50"), errors="coerce").replace(0.0, np.nan)
    sma_200 = pd.to_numeric(enriched.get("sma_200"), errors="coerce").replace(0.0, np.nan)
    enriched["close_to_sma50"] = (close / sma_50).replace([np.inf, -np.inf], np.nan)
    enriched["close_to_sma200"] = (close / sma_200).replace([np.inf, -np.inf], np.nan)
    enriched = _build_veto_columns(enriched, rule=fresh_rule)
    enriched["rationale"] = _build_rationale(enriched)
    return enriched


def _build_candidate_pool(enriched: pd.DataFrame, *, top_candidate_pool: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for trade_date, group in enriched.groupby("trade_date", sort=True):
        ranked = group.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(top_candidate_pool).copy()
        ranked = ranked.loc[ranked["fresh_entry_pass"]].copy()
        if ranked.empty:
            continue
        ranked["base_rank"] = np.arange(1, len(ranked) + 1)
        if len(ranked) == 1:
            ranked["base_rank_pct"] = 1.0
        else:
            ranked["base_rank_pct"] = 1.0 - ((ranked["base_rank"] - 1) / (len(ranked) - 1))
        ranked["trade_date"] = pd.to_datetime(trade_date).normalize()
        parts.append(ranked)
    if not parts:
        return pd.DataFrame(columns=list(enriched.columns) + ["base_rank", "base_rank_pct"])
    return pd.concat(parts, ignore_index=True)


def _prepare_cluster_feature_matrix(
    frame: pd.DataFrame,
    *,
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    work = pd.DataFrame(index=frame.index)
    for column in CLUSTER_NUMERIC_FEATURES:
        series = pd.to_numeric(frame.get(column), errors="coerce")
        if column.startswith("return_"):
            series = series.clip(lower=-0.95, upper=1.50)
        elif column.startswith("rsi_"):
            series = series.clip(lower=0.0, upper=100.0)
        elif column.startswith("close_to_sma"):
            series = series.clip(lower=0.0, upper=5.0)
        else:
            series = series.clip(lower=0.0, upper=15.0)
        work[column] = series
    for column in CLUSTER_FLAG_FEATURES:
        if column in frame.columns:
            source = pd.Series(frame[column], index=frame.index)
            work[column] = np.where(source.notna() & source.astype(bool), 1.0, 0.0)
        else:
            work[column] = 0.0
    if medians is None:
        medians = work.median(numeric_only=True)
    work = work.fillna(medians)
    work = work.replace([np.inf, -np.inf], 0.0)
    return work, medians


def _build_cluster_quality(history: pd.DataFrame, *, shrinkage_rows: int) -> pd.DataFrame:
    cluster_stats = history.groupby("cluster_id", sort=True).agg(
        cluster_count=("symbol", "size"),
        precision_15pct=("winner_15pct", "mean"),
        mean_return=("forward_return", "mean"),
        median_return=("forward_return", "median"),
        p75_return=("forward_return", lambda values: float(pd.Series(values).quantile(0.75))),
    )
    global_precision = float(pd.to_numeric(history["winner_15pct"], errors="coerce").fillna(0).mean())
    global_mean = float(pd.to_numeric(history["forward_return"], errors="coerce").mean())
    global_median = float(pd.to_numeric(history["forward_return"], errors="coerce").median())

    shrink = cluster_stats["cluster_count"] / (cluster_stats["cluster_count"] + shrinkage_rows)
    cluster_stats["precision_shrunk"] = shrink * cluster_stats["precision_15pct"] + (1.0 - shrink) * global_precision
    cluster_stats["mean_shrunk"] = shrink * cluster_stats["mean_return"] + (1.0 - shrink) * global_mean
    cluster_stats["median_shrunk"] = shrink * cluster_stats["median_return"] + (1.0 - shrink) * global_median
    cluster_stats["count_rank"] = cluster_stats["cluster_count"].rank(method="average", pct=True)
    cluster_stats["precision_rank"] = cluster_stats["precision_shrunk"].rank(method="average", pct=True)
    cluster_stats["mean_rank"] = cluster_stats["mean_shrunk"].rank(method="average", pct=True)
    cluster_stats["median_rank"] = cluster_stats["median_shrunk"].rank(method="average", pct=True)
    cluster_stats["cluster_quality"] = (
        0.40 * cluster_stats["median_rank"]
        + 0.30 * cluster_stats["mean_rank"]
        + 0.20 * cluster_stats["precision_rank"]
        + 0.10 * cluster_stats["count_rank"]
    )
    return cluster_stats.reset_index()


def _rerank_with_cluster_overlay(
    candidate_pool: pd.DataFrame,
    *,
    overlay_config: ClusterOverlayConfig,
    n_clusters: int,
    cluster_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_pool.empty:
        return candidate_pool.copy(), pd.DataFrame()

    candidate_pool = candidate_pool.copy()
    candidate_pool["trade_date"] = pd.to_datetime(candidate_pool["trade_date"]).dt.normalize()
    unique_dates = sorted(candidate_pool["trade_date"].dropna().unique())
    reranked_parts: list[pd.DataFrame] = []
    cluster_records: list[pd.DataFrame] = []
    cached_scaler: StandardScaler | None = None
    cached_model: KMeans | None = None
    cached_profile: pd.DataFrame | None = None
    cached_medians: pd.Series | None = None

    for date_idx, trade_date in enumerate(unique_dates):
        current_day = candidate_pool.loc[candidate_pool["trade_date"] == trade_date].copy()
        history = candidate_pool.loc[candidate_pool["trade_date"] < trade_date].copy()
        if len(history) < overlay_config.candidate_history_min_rows or history["trade_date"].nunique() < overlay_config.candidate_history_min_days:
            current_day["cluster_id"] = pd.NA
            current_day["cluster_quality"] = 0.5
            current_day["rerank_score"] = current_day["base_rank_pct"]
            current_day["cluster_reason"] = "warmup"
        else:
            need_refit = (
                cached_scaler is None
                or cached_model is None
                or cached_profile is None
                or cached_medians is None
                or (date_idx % max(overlay_config.refit_every_days, 1) == 0)
            )
            if need_refit:
                history_x, cached_medians = _prepare_cluster_feature_matrix(history)
                cluster_count = max(2, min(n_clusters, len(history_x)))
                cached_scaler = StandardScaler()
                history_scaled = cached_scaler.fit_transform(history_x.to_numpy(dtype=np.float32))
                cached_model = KMeans(n_clusters=cluster_count, random_state=overlay_config.random_state, n_init=10)
                history = history.copy()
                history["cluster_id"] = cached_model.fit_predict(history_scaled)
                cached_profile = _build_cluster_quality(history, shrinkage_rows=overlay_config.shrinkage_rows)
                cluster_records.append(cached_profile.assign(trade_date=pd.Timestamp(trade_date)))
            current_x, _ = _prepare_cluster_feature_matrix(current_day, medians=cached_medians)
            current_scaled = cached_scaler.transform(current_x.to_numpy(dtype=np.float32))
            current_day["cluster_id"] = cached_model.predict(current_scaled)
            quality_map = cached_profile.set_index("cluster_id")["cluster_quality"]
            current_day["cluster_quality"] = current_day["cluster_id"].map(quality_map).fillna(0.5)
            current_day["rerank_score"] = (
                (1.0 - cluster_weight) * pd.to_numeric(current_day["base_rank_pct"], errors="coerce").fillna(0.0)
                + cluster_weight * pd.to_numeric(current_day["cluster_quality"], errors="coerce").fillna(0.5)
            )
            current_day["cluster_reason"] = f"cluster-rerank-{overlay_config.refit_every_days}d"
        current_day = current_day.sort_values(
            ["rerank_score", "focus_score", "symbol"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        current_day["post_veto_rank"] = np.arange(1, len(current_day) + 1)
        reranked_parts.append(current_day)

    reranked = pd.concat(reranked_parts, ignore_index=True) if reranked_parts else candidate_pool.iloc[0:0].copy()
    cluster_profile = pd.concat(cluster_records, ignore_index=True) if cluster_records else pd.DataFrame(
        columns=[
            "cluster_id",
            "cluster_count",
            "precision_15pct",
            "mean_return",
            "median_return",
            "p75_return",
            "precision_shrunk",
            "mean_shrunk",
            "median_shrunk",
            "count_rank",
            "precision_rank",
            "mean_rank",
            "median_rank",
            "cluster_quality",
            "trade_date",
        ]
    )
    return reranked, cluster_profile


def _evaluate_combo_score(
    *,
    daily_top5: dict[str, float | int],
    weekly_top5: dict[str, float | int],
) -> tuple[float, bool]:
    daily_mean = float(daily_top5.get("mean_return_mean", np.nan))
    daily_median = float(daily_top5.get("median_stock_return_median", np.nan))
    weekly_mean = float(weekly_top5.get("mean_return_mean", np.nan))
    weekly_median = float(weekly_top5.get("median_stock_return_median", np.nan))
    meets_target = (
        daily_mean >= 0.01
        and daily_median >= 0.003
        and weekly_mean >= 0.01
        and weekly_median >= 0.003
    )
    if meets_target:
        score = (
            1000.0
            + 100.0 * float(daily_top5.get("precision_15pct", 0.0))
            + 40.0 * daily_mean
            + 80.0 * daily_median
            + 20.0 * weekly_mean
            + 40.0 * weekly_median
        )
    else:
        score = (
            100.0 * float(daily_top5.get("precision_15pct", 0.0))
            + 20.0 * daily_mean
            + 20.0 * max(daily_median, -0.05)
            + 10.0 * weekly_mean
            + 10.0 * max(weekly_median, -0.05)
        )
    return score, meets_target


def _build_live_quality_snapshot(frame: pd.DataFrame) -> dict[str, object]:
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    sma_50 = pd.to_numeric(frame.get("sma_50"), errors="coerce")
    return_15td = pd.to_numeric(frame.get("return_15td"), errors="coerce")
    return_30td = pd.to_numeric(frame.get("return_30td"), errors="coerce")
    rsi_daily = pd.to_numeric(frame.get("rsi_14_daily"), errors="coerce")
    return {
        "rows": int(len(frame)),
        "below_50_dma": int((close < sma_50).fillna(False).sum()),
        "negative_15td": int(return_15td.lt(0).fillna(False).sum()),
        "negative_30td": int(return_30td.lt(0).fillna(False).sum()),
        "rsi_below_50": int(rsi_daily.lt(50).fillna(False).sum()),
        "mean_7td": float(pd.to_numeric(frame.get("return_7td"), errors="coerce").mean()) if len(frame) else np.nan,
        "mean_15td": float(return_15td.mean()) if len(frame) else np.nan,
        "mean_30td": float(return_30td.mean()) if len(frame) else np.nan,
    }


def _summarize_live_top(frame: pd.DataFrame, *, model_name: str, variant: str) -> pd.DataFrame:
    columns = [
        "symbol",
        "company_name",
        "close",
        "shortlist_rank",
        "post_veto_rank",
        "base_rank",
        "base_rank_pct",
        "focus_score",
        "cluster_quality",
        "rerank_score",
        "prob_15pct_7d",
        "prob_20pct_7d",
        "pred_return_7d",
        "calibrated_confidence_15pct_7d",
        "return_7td",
        "return_15td",
        "return_30td",
        "rsi_14_daily",
        "volume_vs_20d",
        "close_to_sma50",
        "close_to_sma200",
        "veto_note",
        "rationale",
    ]
    keep = [column for column in columns if column in frame.columns]
    summary = frame[keep].copy()
    summary.insert(0, "variant", variant)
    summary.insert(0, "model_name", model_name)
    return summary


def _run_single_model(
    *,
    model_name: str,
    base_config_path: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    force_panel: bool,
    overlay_config: ClusterOverlayConfig,
    output_dir: Path,
) -> dict[str, object]:
    base_config = load_research_config(base_config_path)
    objective = ObjectiveSpec(
        name=f"week_7_15pct_{model_name}_allnames_cluster_overlay",
        horizon_days=7,
        target_return=0.15,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    panel_full, _panel_path = prepare_feature_panel(base_config, objective, force=force_panel)
    feature_columns = available_feature_columns(list(panel_full.columns), base_config.feature_columns)
    backtest_cutoff = pd.Timestamp("2025-12-31")
    panel_backtest = panel_full.loc[pd.to_datetime(panel_full["trade_date"]).le(backtest_cutoff)].copy()

    classifier_predictor, regressor_predictor = _select_fit_predictors(model_name)
    folds = build_yearly_walk_forward_folds(
        panel_backtest,
        min_train_end_year=pd.Timestamp(base_config.train_end_date).year,
    )
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
        model_config = RFConfig() if model_name == "random_forest" else GBMConfig(model_name=model_name)
        prob_15 = classifier_predictor(
            train_x=train_x,
            train_y=(train_return >= 0.15).astype(np.int8),
            test_x=test_x,
            config=model_config,
        )
        prob_20 = classifier_predictor(
            train_x=train_x,
            train_y=(train_return >= 0.20).astype(np.int8),
            test_x=test_x,
            config=model_config,
        )
        pred_return = regressor_predictor(
            train_x=train_x,
            train_y=train_return,
            test_x=test_x,
            config=model_config,
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
        raise RuntimeError(f"No walk-forward folds produced predictions for {model_name}.")

    predictions = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["trade_date", "focus_score", "symbol"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    calibration = _build_calibration_table(
        predictions,
        score_col="focus_score",
        target_col="winner_15pct",
        return_col="forward_return",
        bins=10,
    )

    feature_keep_cols = _build_feature_keep_cols(panel_backtest)
    historical_feature_frame = panel_backtest[feature_keep_cols].copy()
    historical_trailing = _build_historical_trailing_state(base_config.paths.daily_facts, end_date=backtest_cutoff)
    relaxed_rule = _make_relaxed_rule()
    historical_enriched = _merge_enrichment(
        predictions,
        feature_frame=historical_feature_frame,
        trailing_state=historical_trailing,
        fresh_rule=relaxed_rule,
    )
    candidate_pool = _build_candidate_pool(historical_enriched, top_candidate_pool=overlay_config.top_candidate_pool)

    baseline_selected = candidate_pool.copy()
    baseline_selected["post_veto_rank"] = baseline_selected["base_rank"]
    baseline_daily_top5 = _evaluate_selected_daily_metrics(baseline_selected, predictions, top_n=5)
    baseline_daily_top10 = _evaluate_selected_daily_metrics(baseline_selected, predictions, top_n=10)
    baseline_weekly_top5 = _evaluate_selected_weekly_metrics(baseline_selected, predictions, top_n=5)
    baseline_weekly_top10 = _evaluate_selected_weekly_metrics(baseline_selected, predictions, top_n=10)

    combo_rows: list[dict[str, object]] = []
    best_combo: dict[str, object] | None = None
    best_selected: pd.DataFrame | None = None
    best_cluster_profile: pd.DataFrame | None = None
    for n_clusters in overlay_config.cluster_options:
        for cluster_weight in overlay_config.overlay_weight_options:
            selected, cluster_profile = _rerank_with_cluster_overlay(
                candidate_pool,
                overlay_config=overlay_config,
                n_clusters=n_clusters,
                cluster_weight=cluster_weight,
            )
            daily_top5 = _evaluate_selected_daily_metrics(selected, predictions, top_n=5)
            daily_top10 = _evaluate_selected_daily_metrics(selected, predictions, top_n=10)
            weekly_top5 = _evaluate_selected_weekly_metrics(selected, predictions, top_n=5)
            weekly_top10 = _evaluate_selected_weekly_metrics(selected, predictions, top_n=10)
            score, meets_target = _evaluate_combo_score(daily_top5=daily_top5, weekly_top5=weekly_top5)
            row = {
                "model_name": model_name,
                "n_clusters": n_clusters,
                "cluster_weight": cluster_weight,
                "meets_target": meets_target,
                "combo_score": score,
                "daily_top5_precision": daily_top5["precision_15pct"],
                "daily_top5_mean_return": daily_top5["mean_return_mean"],
                "daily_top5_median_return": daily_top5["median_stock_return_median"],
                "weekly_top5_precision": weekly_top5["precision_15pct"],
                "weekly_top5_mean_return": weekly_top5["mean_return_mean"],
                "weekly_top5_median_return": weekly_top5["median_stock_return_median"],
                "daily_top10_precision": daily_top10["precision_15pct"],
                "weekly_top10_precision": weekly_top10["precision_15pct"],
            }
            combo_rows.append(row)
            if best_combo is None or score > float(best_combo["combo_score"]):
                best_combo = row
                best_selected = selected
                best_cluster_profile = cluster_profile

    if best_combo is None or best_selected is None or best_cluster_profile is None:
        raise RuntimeError(f"Unable to select a cluster overlay combo for {model_name}.")

    final_train = panel_full.copy()
    stats = fit_preprocess(final_train, feature_columns)
    train_x = transform_frame(final_train, stats)
    train_return = pd.to_numeric(final_train["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    prob15_y = (train_return >= 0.15).astype(np.int8)
    prob20_y = (train_return >= 0.20).astype(np.int8)
    model_config = RFConfig() if model_name == "random_forest" else GBMConfig(model_name=model_name)

    current = build_current_feature_slice(base_config)
    as_of_trade_date = pd.to_datetime(current["trade_date"]).max().normalize()
    current = current.loc[pd.to_datetime(current["trade_date"]).eq(as_of_trade_date)].copy()
    current_x = transform_frame(current, stats)
    current["prob_15pct_7d"] = classifier_predictor(train_x=train_x, train_y=prob15_y, test_x=current_x, config=model_config)
    current["prob_20pct_7d"] = classifier_predictor(train_x=train_x, train_y=prob20_y, test_x=current_x, config=model_config)
    current["pred_return_7d"] = regressor_predictor(train_x=train_x, train_y=train_return, test_x=current_x, config=model_config)
    current["pred_price_7d"] = pd.to_numeric(current["close"], errors="coerce") * (
        1.0 + pd.to_numeric(current["pred_return_7d"], errors="coerce").fillna(0.0)
    )
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
    close = _coerce_numeric_series(current, "close")
    sma_50 = _coerce_numeric_series(current, "sma_50").replace(0.0, np.nan)
    sma_200 = _coerce_numeric_series(current, "sma_200").replace(0.0, np.nan)
    current["close_to_sma50"] = (close / sma_50).replace([np.inf, -np.inf], np.nan)
    current["close_to_sma200"] = (close / sma_200).replace([np.inf, -np.inf], np.nan)
    current = _build_veto_columns(current, rule=relaxed_rule)
    current["rationale"] = _build_rationale(current)

    live_ranked_pool = current.head(overlay_config.top_candidate_pool).copy()
    live_survivors = live_ranked_pool.loc[live_ranked_pool["fresh_entry_pass"]].copy().reset_index(drop=True)
    live_survivors["selected_universe"] = "all_names"
    live_survivors["base_rank"] = np.arange(1, len(live_survivors) + 1)
    if len(live_survivors) == 1:
        live_survivors["base_rank_pct"] = 1.0
    elif len(live_survivors):
        live_survivors["base_rank_pct"] = 1.0 - ((live_survivors["base_rank"] - 1) / (len(live_survivors) - 1))
    else:
        live_survivors["base_rank_pct"] = []

    historical_survivors = candidate_pool.copy()
    if len(historical_survivors) >= overlay_config.candidate_history_min_rows and not live_survivors.empty:
        history_x, history_medians = _prepare_cluster_feature_matrix(historical_survivors)
        live_x, _ = _prepare_cluster_feature_matrix(live_survivors, medians=history_medians)
        live_cluster_count = max(2, min(int(best_combo["n_clusters"]), len(history_x)))
        scaler = StandardScaler()
        history_scaled = scaler.fit_transform(history_x.to_numpy(dtype=np.float32))
        live_scaled = scaler.transform(live_x.to_numpy(dtype=np.float32))
        cluster_model = KMeans(n_clusters=live_cluster_count, random_state=overlay_config.random_state, n_init=20)
        historical_survivors = historical_survivors.copy()
        historical_survivors["cluster_id"] = cluster_model.fit_predict(history_scaled)
        live_survivors["cluster_id"] = cluster_model.predict(live_scaled)
        live_cluster_profile = _build_cluster_quality(historical_survivors, shrinkage_rows=overlay_config.shrinkage_rows)
        quality_map = live_cluster_profile.set_index("cluster_id")["cluster_quality"]
        live_survivors["cluster_quality"] = live_survivors["cluster_id"].map(quality_map).fillna(0.5)
        live_survivors["rerank_score"] = (
            (1.0 - float(best_combo["cluster_weight"])) * pd.to_numeric(live_survivors["base_rank_pct"], errors="coerce").fillna(0.0)
            + float(best_combo["cluster_weight"]) * pd.to_numeric(live_survivors["cluster_quality"], errors="coerce").fillna(0.5)
        )
    else:
        live_cluster_profile = pd.DataFrame()
        live_survivors["cluster_id"] = pd.NA
        live_survivors["cluster_quality"] = 0.5
        live_survivors["rerank_score"] = pd.to_numeric(live_survivors["base_rank_pct"], errors="coerce").fillna(0.0)
    live_survivors = live_survivors.sort_values(
        ["rerank_score", "focus_score", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    live_survivors["post_veto_rank"] = np.arange(1, len(live_survivors) + 1)

    live_top10_cluster = live_survivors.head(10).copy()
    live_top10_base = live_survivors.sort_values(["base_rank", "symbol"], ascending=[True, True]).head(10).copy()

    live_quality_base = _build_live_quality_snapshot(live_top10_base)
    live_quality_cluster = _build_live_quality_snapshot(live_top10_cluster)

    per_model_dir = output_dir / model_name
    per_model_dir.mkdir(parents=True, exist_ok=True)
    combos_df = pd.DataFrame(combo_rows).sort_values(["meets_target", "combo_score", "daily_top5_precision"], ascending=[False, False, False]).reset_index(drop=True)
    best_cluster_profile.to_csv(per_model_dir / "historical_cluster_profile.csv", index=False)
    combos_df.to_csv(per_model_dir / "combo_grid.csv", index=False)
    _summarize_live_top(live_top10_base, model_name=model_name, variant="base").to_csv(per_model_dir / "live_top10_base.csv", index=False)
    _summarize_live_top(live_top10_cluster, model_name=model_name, variant="cluster").to_csv(per_model_dir / "live_top10_cluster.csv", index=False)

    summary = {
        "model_name": model_name,
        "trade_for_date": str(_next_trading_day(as_of_trade_date).date()),
        "as_of_trade_date": str(as_of_trade_date.date()),
        "model_config": _model_config_dict(model_name),
        "relaxed_rule": asdict(relaxed_rule),
        "overlay_config": asdict(overlay_config),
        "best_combo": best_combo,
        "baseline_top5": baseline_daily_top5,
        "baseline_weekly_top5": baseline_weekly_top5,
        "cluster_top5": _evaluate_selected_daily_metrics(best_selected, predictions, top_n=5),
        "cluster_top10": _evaluate_selected_daily_metrics(best_selected, predictions, top_n=10),
        "cluster_weekly_top5": _evaluate_selected_weekly_metrics(best_selected, predictions, top_n=5),
        "cluster_weekly_top10": _evaluate_selected_weekly_metrics(best_selected, predictions, top_n=10),
        "live_quality_base": live_quality_base,
        "live_quality_cluster": live_quality_cluster,
        "current_candidate_counts": {
            "ranked_rows": int(len(live_ranked_pool)),
            "fresh_entry_pass_rows": int(len(live_survivors)),
            "top10_cluster_rows": int(len(live_top10_cluster)),
        },
    }
    summary = _jsonify(summary)
    write_json(summary, per_model_dir / "summary.json")

    for path in [
        per_model_dir / "historical_cluster_profile.csv",
        per_model_dir / "combo_grid.csv",
        per_model_dir / "live_top10_base.csv",
        per_model_dir / "live_top10_cluster.csv",
    ]:
        frame = pd.read_csv(path)
        write_dataframe_manifest(
            path,
            frame,
            generated_by="src/analysis/week7_15pct_cluster_rerank_compare.py",
            as_of_date=str(as_of_trade_date.date()),
            extra_notes=[f"{model_name} all-stocks cluster-overlay review for 7-day 15 percent target."],
        )
    write_json_manifest(
        per_model_dir / "summary.json",
        summary,
        generated_by="src/analysis/week7_15pct_cluster_rerank_compare.py",
        as_of_date=str(as_of_trade_date.date()),
        extra_notes=["Cluster overlay uses only prior candidate history; no universe preselection was used."],
    )
    write_report_directory_readme(
        per_model_dir,
        title=f"{model_name.title()} Cluster Overlay Review",
        intro_lines=[
            "This folder compares the base all-stocks fail-closed route with a softer cluster-aware rerank.",
            "The rerank uses only prior candidate history and does not use any universe preselection.",
            "Read `summary.json` first, then `combo_grid.csv`, `live_top10_cluster.csv`, and `historical_cluster_profile.csv`.",
        ],
        files=[
            per_model_dir / "summary.json",
            per_model_dir / "combo_grid.csv",
            per_model_dir / "live_top10_base.csv",
            per_model_dir / "live_top10_cluster.csv",
            per_model_dir / "historical_cluster_profile.csv",
        ],
    )
    return {
        "summary": summary,
        "live_top10_base": _summarize_live_top(live_top10_base, model_name=model_name, variant="base"),
        "live_top10_cluster": _summarize_live_top(live_top10_cluster, model_name=model_name, variant="cluster"),
        "combo_grid": combos_df,
    }


def run_cluster_overlay_compare(
    *,
    config_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    force_panel: bool,
    overlay_config: ClusterOverlayConfig,
    model_names: list[str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_results: list[dict[str, object]] = []
    live_frames: list[pd.DataFrame] = []
    combo_frames: list[pd.DataFrame] = []
    for model_name in model_names:
        result = _run_single_model(
            model_name=model_name,
            base_config_path=config_path,
            analysis_start_date=analysis_start_date,
            evaluation_end_date=evaluation_end_date,
            min_price=min_price,
            force_panel=force_panel,
            overlay_config=overlay_config,
            output_dir=output_dir,
        )
        summary = result["summary"]
        model_results.append(
            {
                "model_name": model_name,
                "meets_target": bool(summary["best_combo"]["meets_target"]),
                "best_n_clusters": int(summary["best_combo"]["n_clusters"]),
                "best_cluster_weight": float(summary["best_combo"]["cluster_weight"]),
                "baseline_daily_top5_precision": float(summary["baseline_top5"]["precision_15pct"]),
                "baseline_daily_top5_mean_return": float(summary["baseline_top5"]["mean_return_mean"]),
                "baseline_daily_top5_median_return": float(summary["baseline_top5"]["median_stock_return_median"]),
                "cluster_daily_top5_precision": float(summary["cluster_top5"]["precision_15pct"]),
                "cluster_daily_top5_mean_return": float(summary["cluster_top5"]["mean_return_mean"]),
                "cluster_daily_top5_median_return": float(summary["cluster_top5"]["median_stock_return_median"]),
                "cluster_weekly_top5_precision": float(summary["cluster_weekly_top5"]["precision_15pct"]),
                "cluster_weekly_top5_mean_return": float(summary["cluster_weekly_top5"]["mean_return_mean"]),
                "cluster_weekly_top5_median_return": float(summary["cluster_weekly_top5"]["median_stock_return_median"]),
                "live_cluster_below_50_dma": int(summary["live_quality_cluster"]["below_50_dma"]),
                "live_cluster_negative_15td": int(summary["live_quality_cluster"]["negative_15td"]),
                "live_cluster_negative_30td": int(summary["live_quality_cluster"]["negative_30td"]),
                "live_cluster_rsi_below_50": int(summary["live_quality_cluster"]["rsi_below_50"]),
            }
        )
        live_frames.append(result["live_top10_base"])
        live_frames.append(result["live_top10_cluster"])
        combo_frames.append(result["combo_grid"].assign(model_name=model_name))

    comparison = pd.DataFrame(model_results).sort_values(
        ["meets_target", "cluster_daily_top5_precision", "cluster_daily_top5_median_return", "cluster_daily_top5_mean_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    live_side_by_side = pd.concat(live_frames, ignore_index=True)
    combo_grid = pd.concat(combo_frames, ignore_index=True)

    best_model_row = comparison.iloc[0].to_dict()
    summary = _jsonify(
        {
            "status": "ok",
            "run_type": "week7_15pct_allnames_cluster_overlay_compare",
            "analysis_start_date": analysis_start_date,
            "evaluation_end_date": evaluation_end_date,
            "min_price": min_price,
            "overlay_config": asdict(overlay_config),
            "best_model": best_model_row,
            "target_definition": {
                "daily_top5_mean_return_floor": 0.01,
                "daily_top5_median_return_floor": 0.003,
                "weekly_top5_mean_return_floor": 0.01,
                "weekly_top5_median_return_floor": 0.003,
            },
            "notes": [
                "All stocks were scored directly; no universe preselection was used.",
                "Cluster reranking uses only prior candidate history from the walk-forward prediction stream.",
                "The live shortlist quality diagnostics are descriptive checks, not outcome labels.",
            ],
        }
    )

    comparison_path = output_dir / "model_comparison.csv"
    live_path = output_dir / "live_top10_side_by_side.csv"
    combo_path = output_dir / "combo_grid_all_models.csv"
    summary_path = output_dir / "summary.json"
    comparison.to_csv(comparison_path, index=False)
    live_side_by_side.to_csv(live_path, index=False)
    combo_grid.to_csv(combo_path, index=False)
    write_json(summary, summary_path)

    for path, frame in [
        (comparison_path, comparison),
        (live_path, live_side_by_side),
        (combo_path, combo_grid),
    ]:
        write_dataframe_manifest(
            path,
            frame,
            generated_by="src/analysis/week7_15pct_cluster_rerank_compare.py",
            as_of_date=str(best_model_row.get("trade_for_date", "")) if best_model_row.get("trade_for_date") else None,
            extra_notes=["All-stocks cluster-overlay model comparison for the 7-day 15 percent target."],
        )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_15pct_cluster_rerank_compare.py",
        extra_notes=["The comparison promotes the highest-precision model among those meeting the mean/median thresholds."],
    )
    write_report_directory_readme(
        output_dir,
        title="Week 7 Fifteen Percent Cluster Overlay Comparison",
        intro_lines=[
            "This folder compares XGBoost, LightGBM, and Random Forest on the all-stocks 7-day 15 percent target with a softer cluster-aware rerank.",
            "The objective is to preserve the strong mean and median basket returns while reducing obviously weak live shortlist states.",
            "Open `summary.json` first, then `model_comparison.csv`, `live_top10_side_by_side.csv`, and each model subfolder.",
        ],
        files=[summary_path, comparison_path, live_path, combo_path],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all-stocks 7-day 15 percent models with a time-respecting cluster-aware rerank overlay.")
    parser.add_argument("--config", default="configs/ml_research.yaml")
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2026-04-21")
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--output-dir", default="reports/week7_15pct_cluster_overlay_compare")
    parser.add_argument("--top-candidate-pool", type=int, default=100)
    parser.add_argument("--candidate-history-min-rows", type=int, default=500)
    parser.add_argument("--candidate-history-min-days", type=int, default=30)
    parser.add_argument("--cluster-options", default="5,7,9,11")
    parser.add_argument("--overlay-weight-options", default="0.15,0.25,0.35,0.45")
    parser.add_argument("--shrinkage-rows", type=int, default=120)
    parser.add_argument("--refit-every-days", type=int, default=5)
    parser.add_argument("--model-names", default="xgboost,lightgbm,random_forest")
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()

    overlay_config = ClusterOverlayConfig(
        top_candidate_pool=args.top_candidate_pool,
        candidate_history_min_rows=args.candidate_history_min_rows,
        candidate_history_min_days=args.candidate_history_min_days,
        cluster_options=tuple(int(part.strip()) for part in args.cluster_options.split(",") if part.strip()),
        overlay_weight_options=tuple(float(part.strip()) for part in args.overlay_weight_options.split(",") if part.strip()),
        shrinkage_rows=args.shrinkage_rows,
        refit_every_days=args.refit_every_days,
    )

    summary = run_cluster_overlay_compare(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        min_price=args.min_price,
        force_panel=args.force_panel,
        overlay_config=overlay_config,
        model_names=[part.strip() for part in args.model_names.split(",") if part.strip()],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
