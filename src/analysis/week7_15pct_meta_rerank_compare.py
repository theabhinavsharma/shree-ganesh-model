from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.week7_15pct_cluster_rerank_compare import _build_candidate_pool
from src.analysis.week7_15pct_cluster_rerank_compare import _build_feature_keep_cols
from src.analysis.week7_15pct_cluster_rerank_compare import _make_relaxed_rule
from src.analysis.week7_15pct_cluster_rerank_compare import _merge_enrichment
from src.analysis.week7_15pct_cluster_rerank_compare import _model_config_dict
from src.analysis.week7_15pct_cluster_rerank_compare import _select_fit_predictors
from src.analysis.week7_15pct_gbm_allnames import GBMConfig
from src.analysis.week7_15pct_random_forest_allnames import _apply_calibration_15pct
from src.analysis.week7_15pct_random_forest_allnames import _build_rationale
from src.analysis.week7_15pct_random_forest_allnames import _build_trailing_state
from src.analysis.week7_15pct_random_forest_allnames import _combine_focus_score
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_selected_daily_metrics
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_selected_weekly_metrics
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
class MetaRankConfig:
    top_candidate_pool: int = 100
    focus_weight_options: tuple[float, ...] = (0.65, 0.75, 0.85, 0.95)
    trend_weight_options: tuple[float, ...] = (0.05, 0.10, 0.20)
    event_weight_options: tuple[float, ...] = (0.00, 0.05, 0.10)
    dying_weight_options: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    stretch_weight_options: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10)


def _clip_score(series: pd.Series, low: float, high: float) -> pd.Series:
    clipped = pd.to_numeric(series, errors="coerce").clip(lower=low, upper=high)
    return ((clipped - low) / (high - low)).clip(lower=0.0, upper=1.0)


def _build_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["trend_score"] = (
        0.15 * _clip_score(work.get("return_7td"), -0.05, 0.15)
        + 0.25 * _clip_score(work.get("return_15td"), -0.05, 0.30)
        + 0.20 * _clip_score(work.get("return_30td"), -0.05, 0.35)
        + 0.20 * _clip_score(work.get("rsi_14_daily"), 45.0, 75.0)
        + 0.10 * _clip_score(work.get("close_to_sma50"), 0.95, 1.15)
        + 0.10 * _clip_score(work.get("close_to_sma200"), 0.85, 1.15)
    )
    event_cols = [
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_promoter_or_director_buy_flag",
        "recent_bulk_buy_flag",
    ]
    event_matrix = []
    for column in event_cols:
        if column in work.columns:
            raw = pd.Series(work[column], index=work.index)
            event_matrix.append(pd.Series(np.where(raw.eq(True), 1.0, 0.0), index=work.index, dtype=np.float32))
        else:
            event_matrix.append(pd.Series(0.0, index=work.index))
    work["event_score"] = pd.concat(event_matrix, axis=1).mean(axis=1)

    work["dying_penalty"] = (
        0.30 * pd.to_numeric(work.get("return_15td"), errors="coerce").lt(0).fillna(False).astype(float)
        + 0.25 * pd.to_numeric(work.get("return_30td"), errors="coerce").lt(0).fillna(False).astype(float)
        + 0.20 * pd.to_numeric(work.get("rsi_14_daily"), errors="coerce").lt(50).fillna(False).astype(float)
        + 0.15 * pd.to_numeric(work.get("close_to_sma50"), errors="coerce").lt(1.0).fillna(False).astype(float)
        + 0.10 * pd.to_numeric(work.get("close_to_sma200"), errors="coerce").lt(0.9).fillna(False).astype(float)
    )
    work["stretch_penalty"] = (
        0.40 * _clip_score(work.get("return_7td"), 0.10, 0.20)
        + 0.35 * _clip_score(work.get("return_15td"), 0.20, 0.35)
        + 0.25 * _clip_score(work.get("rsi_14_daily"), 68.0, 78.0)
    )
    return work


def _rerank_with_meta_score(
    candidate_pool: pd.DataFrame,
    *,
    weights: dict[str, float],
) -> pd.DataFrame:
    if candidate_pool.empty:
        return candidate_pool.copy()
    work = _build_meta_features(candidate_pool)
    work["meta_score"] = (
        weights["focus_weight"] * pd.to_numeric(work.get("base_rank_pct"), errors="coerce").fillna(0.0)
        + weights["trend_weight"] * pd.to_numeric(work.get("trend_score"), errors="coerce").fillna(0.0)
        + weights["event_weight"] * pd.to_numeric(work.get("event_score"), errors="coerce").fillna(0.0)
        - weights["dying_weight"] * pd.to_numeric(work.get("dying_penalty"), errors="coerce").fillna(0.0)
        - weights["stretch_weight"] * pd.to_numeric(work.get("stretch_penalty"), errors="coerce").fillna(0.0)
    )
    parts: list[pd.DataFrame] = []
    for trade_date, group in work.groupby("trade_date", sort=True):
        ranked = group.sort_values(["meta_score", "focus_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        ranked["post_veto_rank"] = np.arange(1, len(ranked) + 1)
        parts.append(ranked)
    return pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0].copy()


def _evaluate_selected_state_quality(selected: pd.DataFrame, *, top_n: int) -> dict[str, float]:
    rows = []
    for _, group in selected.groupby("trade_date", sort=False):
        top = group.sort_values(["post_veto_rank", "symbol"], ascending=[True, True]).head(top_n).copy()
        if top.empty:
            continue
        rows.append(top)
    chosen = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=selected.columns)
    if chosen.empty:
        return {
            "negative_15td_rate": np.nan,
            "negative_30td_rate": np.nan,
            "rsi_below_50_rate": np.nan,
            "below_50_dma_rate": np.nan,
            "mean_15td": np.nan,
            "mean_30td": np.nan,
        }
    close_to_sma50 = pd.to_numeric(chosen.get("close_to_sma50"), errors="coerce")
    return {
        "negative_15td_rate": float(pd.to_numeric(chosen.get("return_15td"), errors="coerce").lt(0).fillna(False).mean()),
        "negative_30td_rate": float(pd.to_numeric(chosen.get("return_30td"), errors="coerce").lt(0).fillna(False).mean()),
        "rsi_below_50_rate": float(pd.to_numeric(chosen.get("rsi_14_daily"), errors="coerce").lt(50).fillna(False).mean()),
        "below_50_dma_rate": float(close_to_sma50.lt(1.0).fillna(False).mean()),
        "mean_15td": float(pd.to_numeric(chosen.get("return_15td"), errors="coerce").mean()),
        "mean_30td": float(pd.to_numeric(chosen.get("return_30td"), errors="coerce").mean()),
    }


def _evaluate_meta_combo_score(
    *,
    daily_top5: dict[str, float | int],
    weekly_top5: dict[str, float | int],
    state_quality: dict[str, float],
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
    dying_penalty = (
        4.0 * float(state_quality.get("negative_15td_rate", 0.0))
        + 4.0 * float(state_quality.get("negative_30td_rate", 0.0))
        + 3.0 * float(state_quality.get("rsi_below_50_rate", 0.0))
        + 2.0 * float(state_quality.get("below_50_dma_rate", 0.0))
    )
    score = (
        (1000.0 if meets_target else 0.0)
        + 120.0 * float(daily_top5.get("precision_15pct", 0.0))
        + 30.0 * daily_mean
        + 70.0 * daily_median
        + 15.0 * weekly_mean
        + 40.0 * weekly_median
        - dying_penalty
    )
    return score, meets_target


def _build_live_quality_snapshot(frame: pd.DataFrame) -> dict[str, object]:
    close_to_sma50 = pd.to_numeric(frame.get("close_to_sma50"), errors="coerce")
    return {
        "rows": int(len(frame)),
        "below_50_dma": int(close_to_sma50.lt(1.0).fillna(False).sum()),
        "negative_15td": int(pd.to_numeric(frame.get("return_15td"), errors="coerce").lt(0).fillna(False).sum()),
        "negative_30td": int(pd.to_numeric(frame.get("return_30td"), errors="coerce").lt(0).fillna(False).sum()),
        "rsi_below_50": int(pd.to_numeric(frame.get("rsi_14_daily"), errors="coerce").lt(50).fillna(False).sum()),
        "mean_7td": float(pd.to_numeric(frame.get("return_7td"), errors="coerce").mean()) if len(frame) else np.nan,
        "mean_15td": float(pd.to_numeric(frame.get("return_15td"), errors="coerce").mean()) if len(frame) else np.nan,
        "mean_30td": float(pd.to_numeric(frame.get("return_30td"), errors="coerce").mean()) if len(frame) else np.nan,
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
        "meta_score",
        "trend_score",
        "event_score",
        "dying_penalty",
        "stretch_penalty",
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
    config_path: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    force_panel: bool,
    meta_config: MetaRankConfig,
    output_dir: Path,
) -> dict[str, object]:
    base_config = load_research_config(config_path)
    objective = ObjectiveSpec(
        name=f"week_7_15pct_{model_name}_allnames_meta_rerank",
        horizon_days=7,
        target_return=0.15,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    panel_full, _ = prepare_feature_panel(base_config, objective, force=force_panel)
    feature_columns = available_feature_columns(list(panel_full.columns), base_config.feature_columns)
    backtest_cutoff = pd.Timestamp("2025-12-31")
    panel_backtest = panel_full.loc[pd.to_datetime(panel_full["trade_date"]).le(backtest_cutoff)].copy()
    classifier_predictor, regressor_predictor = _select_fit_predictors(model_name)
    model_config = GBMConfig(model_name=model_name)

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
        prob_15 = classifier_predictor(train_x=train_x, train_y=(train_return >= 0.15).astype(np.int8), test_x=test_x, config=model_config)
        prob_20 = classifier_predictor(train_x=train_x, train_y=(train_return >= 0.20).astype(np.int8), test_x=test_x, config=model_config)
        pred_return = regressor_predictor(train_x=train_x, train_y=train_return, test_x=test_x, config=model_config)
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

    predictions = pd.concat(oof_parts, ignore_index=True).sort_values(["trade_date", "focus_score", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
    calibration = _build_calibration_table(
        predictions,
        score_col="focus_score",
        target_col="winner_15pct",
        return_col="forward_return",
        bins=10,
    )

    feature_keep_cols = _build_feature_keep_cols(panel_backtest)
    historical_feature_frame = panel_backtest[feature_keep_cols].copy()
    from src.analysis.week7_15pct_random_forest_allnames import _build_historical_trailing_state
    historical_trailing = _build_historical_trailing_state(base_config.paths.daily_facts, end_date=backtest_cutoff)
    relaxed_rule = _make_relaxed_rule()
    historical_enriched = _merge_enrichment(
        predictions,
        feature_frame=historical_feature_frame,
        trailing_state=historical_trailing,
        fresh_rule=relaxed_rule,
    )
    candidate_pool = _build_candidate_pool(historical_enriched, top_candidate_pool=meta_config.top_candidate_pool)

    baseline_selected = _build_meta_features(candidate_pool.copy())
    baseline_selected["post_veto_rank"] = baseline_selected["base_rank"]
    baseline_top5 = _evaluate_selected_daily_metrics(baseline_selected, predictions, top_n=5)
    baseline_weekly_top5 = _evaluate_selected_weekly_metrics(baseline_selected, predictions, top_n=5)
    baseline_quality = _evaluate_selected_state_quality(baseline_selected, top_n=5)

    combo_rows: list[dict[str, object]] = []
    best_combo: dict[str, object] | None = None
    best_selected: pd.DataFrame | None = None
    for focus_weight in meta_config.focus_weight_options:
        for trend_weight in meta_config.trend_weight_options:
            for event_weight in meta_config.event_weight_options:
                for dying_weight in meta_config.dying_weight_options:
                    for stretch_weight in meta_config.stretch_weight_options:
                        weights = {
                            "focus_weight": focus_weight,
                            "trend_weight": trend_weight,
                            "event_weight": event_weight,
                            "dying_weight": dying_weight,
                            "stretch_weight": stretch_weight,
                        }
                        selected = _rerank_with_meta_score(candidate_pool, weights=weights)
                        daily_top5 = _evaluate_selected_daily_metrics(selected, predictions, top_n=5)
                        weekly_top5 = _evaluate_selected_weekly_metrics(selected, predictions, top_n=5)
                        state_quality = _evaluate_selected_state_quality(selected, top_n=5)
                        score, meets_target = _evaluate_meta_combo_score(
                            daily_top5=daily_top5,
                            weekly_top5=weekly_top5,
                            state_quality=state_quality,
                        )
                        row = {
                            **weights,
                            "meets_target": meets_target,
                            "combo_score": score,
                            "daily_top5_precision": daily_top5["precision_15pct"],
                            "daily_top5_mean_return": daily_top5["mean_return_mean"],
                            "daily_top5_median_return": daily_top5["median_stock_return_median"],
                            "weekly_top5_precision": weekly_top5["precision_15pct"],
                            "weekly_top5_mean_return": weekly_top5["mean_return_mean"],
                            "weekly_top5_median_return": weekly_top5["median_stock_return_median"],
                            **state_quality,
                        }
                        combo_rows.append(row)
                        if best_combo is None or score > float(best_combo["combo_score"]):
                            best_combo = row
                            best_selected = selected
    if best_combo is None or best_selected is None:
        raise RuntimeError(f"Unable to select a meta-rank combo for {model_name}.")

    final_train = panel_full.copy()
    stats = fit_preprocess(final_train, feature_columns)
    train_x = transform_frame(final_train, stats)
    train_return = pd.to_numeric(final_train["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    prob15_y = (train_return >= 0.15).astype(np.int8)
    prob20_y = (train_return >= 0.20).astype(np.int8)

    current = build_current_feature_slice(base_config)
    as_of_trade_date = pd.to_datetime(current["trade_date"]).max().normalize()
    current = current.loc[pd.to_datetime(current["trade_date"]).eq(as_of_trade_date)].copy()
    current_x = transform_frame(current, stats)
    current["prob_15pct_7d"] = classifier_predictor(train_x=train_x, train_y=prob15_y, test_x=current_x, config=model_config)
    current["prob_20pct_7d"] = classifier_predictor(train_x=train_x, train_y=prob20_y, test_x=current_x, config=model_config)
    current["pred_return_7d"] = regressor_predictor(train_x=train_x, train_y=train_return, test_x=current_x, config=model_config)
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
    close = pd.to_numeric(current.get("close"), errors="coerce")
    sma_50 = pd.to_numeric(current.filter(regex=r"^sma_50(_x|_y)?$").iloc[:, 0] if len(current.filter(regex=r"^sma_50(_x|_y)?$").columns) else pd.Series(np.nan, index=current.index), errors="coerce").replace(0.0, np.nan)
    sma_200 = pd.to_numeric(current.filter(regex=r"^sma_200(_x|_y)?$").iloc[:, 0] if len(current.filter(regex=r"^sma_200(_x|_y)?$").columns) else pd.Series(np.nan, index=current.index), errors="coerce").replace(0.0, np.nan)
    current["close_to_sma50"] = (close / sma_50).replace([np.inf, -np.inf], np.nan)
    current["close_to_sma200"] = (close / sma_200).replace([np.inf, -np.inf], np.nan)
    from src.analysis.week7_15pct_random_forest_allnames import _build_veto_columns
    current = _build_veto_columns(current, rule=relaxed_rule)
    current["rationale"] = _build_rationale(current)
    ranked_pool = current.head(meta_config.top_candidate_pool).copy()
    survivors = ranked_pool.loc[ranked_pool["fresh_entry_pass"]].copy().reset_index(drop=True)
    survivors["selected_universe"] = "all_names"
    survivors["base_rank"] = np.arange(1, len(survivors) + 1)
    if len(survivors) == 1:
        survivors["base_rank_pct"] = 1.0
    elif len(survivors):
        survivors["base_rank_pct"] = 1.0 - ((survivors["base_rank"] - 1) / (len(survivors) - 1))
    else:
        survivors["base_rank_pct"] = []
    live_base = _build_meta_features(survivors.copy())
    live_base["post_veto_rank"] = live_base["base_rank"]
    live_meta = _rerank_with_meta_score(survivors, weights={
        "focus_weight": float(best_combo["focus_weight"]),
        "trend_weight": float(best_combo["trend_weight"]),
        "event_weight": float(best_combo["event_weight"]),
        "dying_weight": float(best_combo["dying_weight"]),
        "stretch_weight": float(best_combo["stretch_weight"]),
    })
    live_top10_base = live_base.sort_values(["base_rank", "symbol"], ascending=[True, True]).head(10).copy()
    live_top10_meta = live_meta.head(10).copy()

    per_model_dir = output_dir / model_name
    per_model_dir.mkdir(parents=True, exist_ok=True)
    combo_grid = pd.DataFrame(combo_rows).sort_values(["meets_target", "combo_score", "daily_top5_precision"], ascending=[False, False, False]).reset_index(drop=True)
    combo_path = per_model_dir / "combo_grid.csv"
    base_path = per_model_dir / "live_top10_base.csv"
    meta_path = per_model_dir / "live_top10_meta.csv"
    summary_path = per_model_dir / "summary.json"
    combo_grid.to_csv(combo_path, index=False)
    _summarize_live_top(live_top10_base, model_name=model_name, variant="base").to_csv(base_path, index=False)
    _summarize_live_top(live_top10_meta, model_name=model_name, variant="meta").to_csv(meta_path, index=False)

    summary = {
        "model_name": model_name,
        "trade_for_date": str(_next_trading_day(as_of_trade_date).date()),
        "as_of_trade_date": str(as_of_trade_date.date()),
        "model_config": _model_config_dict(model_name),
        "meta_rank_config": asdict(meta_config),
        "best_combo": best_combo,
        "baseline_top5": baseline_top5,
        "baseline_weekly_top5": baseline_weekly_top5,
        "baseline_state_quality": baseline_quality,
        "meta_top5": _evaluate_selected_daily_metrics(best_selected, predictions, top_n=5),
        "meta_weekly_top5": _evaluate_selected_weekly_metrics(best_selected, predictions, top_n=5),
        "meta_state_quality": _evaluate_selected_state_quality(best_selected, top_n=5),
        "live_quality_base": _build_live_quality_snapshot(live_top10_base),
        "live_quality_meta": _build_live_quality_snapshot(live_top10_meta),
        "current_candidate_counts": {
            "ranked_rows": int(len(ranked_pool)),
            "fresh_entry_pass_rows": int(len(survivors)),
            "top10_meta_rows": int(len(live_top10_meta)),
        },
    }
    summary = _jsonify(summary)
    write_json(summary, summary_path)
    for path in [combo_path, base_path, meta_path]:
        df = pd.read_csv(path)
        write_dataframe_manifest(
            path,
            df,
            generated_by="src/analysis/week7_15pct_meta_rerank_compare.py",
            as_of_date=str(as_of_trade_date.date()),
            extra_notes=[f"{model_name} all-stocks meta-rank review for the 7-day 15 percent target."],
        )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_15pct_meta_rerank_compare.py",
        as_of_date=str(as_of_trade_date.date()),
        extra_notes=["Meta-ranker directly optimizes walk-forward Top-5 basket outcomes and penalizes dying-trend states."],
    )
    write_report_directory_readme(
        per_model_dir,
        title=f"{model_name.title()} Meta-Rank Review",
        intro_lines=[
            "This folder compares the base all-stocks fail-closed route with a second-stage meta-ranker.",
            "The meta-ranker optimizes Top-5 mean and median returns while penalizing historically weak trend states.",
            "Read `summary.json` first, then `combo_grid.csv`, `live_top10_meta.csv`, and `live_top10_base.csv`.",
        ],
        files=[summary_path, combo_path, base_path, meta_path],
    )
    return {
        "summary": summary,
        "live_top10_base": _summarize_live_top(live_top10_base, model_name=model_name, variant="base"),
        "live_top10_meta": _summarize_live_top(live_top10_meta, model_name=model_name, variant="meta"),
    }


def run_meta_rerank_compare(
    *,
    config_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    force_panel: bool,
    meta_config: MetaRankConfig,
    model_names: list[str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_results: list[dict[str, object]] = []
    live_frames: list[pd.DataFrame] = []
    for model_name in model_names:
        result = _run_single_model(
            model_name=model_name,
            config_path=config_path,
            analysis_start_date=analysis_start_date,
            evaluation_end_date=evaluation_end_date,
            min_price=min_price,
            force_panel=force_panel,
            meta_config=meta_config,
            output_dir=output_dir,
        )
        summary = result["summary"]
        model_results.append({
            "model_name": model_name,
            "meets_target": bool(summary["best_combo"]["meets_target"]),
            "meta_daily_top5_precision": float(summary["meta_top5"]["precision_15pct"]),
            "meta_daily_top5_mean_return": float(summary["meta_top5"]["mean_return_mean"]),
            "meta_daily_top5_median_return": float(summary["meta_top5"]["median_stock_return_median"]),
            "meta_weekly_top5_precision": float(summary["meta_weekly_top5"]["precision_15pct"]),
            "meta_weekly_top5_mean_return": float(summary["meta_weekly_top5"]["mean_return_mean"]),
            "meta_weekly_top5_median_return": float(summary["meta_weekly_top5"]["median_stock_return_median"]),
            "baseline_daily_top5_precision": float(summary["baseline_top5"]["precision_15pct"]),
            "baseline_daily_top5_mean_return": float(summary["baseline_top5"]["mean_return_mean"]),
            "baseline_daily_top5_median_return": float(summary["baseline_top5"]["median_stock_return_median"]),
            "live_meta_negative_15td": int(summary["live_quality_meta"]["negative_15td"]),
            "live_meta_negative_30td": int(summary["live_quality_meta"]["negative_30td"]),
            "live_meta_rsi_below_50": int(summary["live_quality_meta"]["rsi_below_50"]),
            "live_meta_below_50_dma": int(summary["live_quality_meta"]["below_50_dma"]),
        })
        live_frames.append(result["live_top10_base"])
        live_frames.append(result["live_top10_meta"])
    comparison = pd.DataFrame(model_results).sort_values(
        ["meets_target", "meta_daily_top5_precision", "meta_daily_top5_median_return", "meta_daily_top5_mean_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    live_side_by_side = pd.concat(live_frames, ignore_index=True)
    summary = _jsonify({
        "status": "ok",
        "as_of_trade_date": "2026-04-21",
        "trade_for_date": "2026-04-22",
        "run_type": "week7_15pct_meta_rerank_compare",
        "meta_rank_config": asdict(meta_config),
        "best_model": comparison.iloc[0].to_dict(),
        "target_definition": {
            "daily_top5_mean_return_floor": 0.01,
            "daily_top5_median_return_floor": 0.003,
            "weekly_top5_mean_return_floor": 0.01,
            "weekly_top5_median_return_floor": 0.003,
        },
        "notes": [
            "All stocks were scored directly; no universe preselection was used.",
            "The second-stage meta-ranker selects from the candidate pool using walk-forward basket outcomes, not single-name fit.",
            "The score explicitly rewards trend quality and event support while penalizing historically dying-trend states and stretched entries.",
        ],
    })
    comparison_path = output_dir / "model_comparison.csv"
    live_path = output_dir / "live_top10_side_by_side.csv"
    summary_path = output_dir / "summary.json"
    comparison.to_csv(comparison_path, index=False)
    live_side_by_side.to_csv(live_path, index=False)
    write_json(summary, summary_path)
    for path in [comparison_path, live_path]:
        df = pd.read_csv(path)
        write_dataframe_manifest(
            path,
            df,
            generated_by="src/analysis/week7_15pct_meta_rerank_compare.py",
            as_of_date="2026-04-21",
            extra_notes=["Meta-ranker comparison across GBM models for the 7-day 15 percent target."],
        )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_15pct_meta_rerank_compare.py",
        as_of_date="2026-04-21",
        extra_notes=["Built from completed per-model meta-rank runs."],
    )
    write_report_directory_readme(
        output_dir,
        title="Week 7 Fifteen Percent Meta-Rank Compare",
        intro_lines=[
            "This folder compares the completed meta-ranker runs for the all-stocks 7-day 15 percent target.",
            "The meta-ranker directly optimizes Top-5 mean and median return while penalizing historically weak trend states.",
            "Open `summary.json` first, then `model_comparison.csv`, then inspect each model subfolder.",
        ],
        files=[summary_path, comparison_path, live_path],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all-stocks 7-day 15 percent GBM models with a direct candidate-pool meta-ranker.")
    parser.add_argument("--config", default="configs/ml_research.yaml")
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2026-04-21")
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--output-dir", default="reports/week7_15pct_meta_rerank_compare")
    parser.add_argument("--top-candidate-pool", type=int, default=100)
    parser.add_argument("--model-names", default="lightgbm,xgboost")
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()

    meta_config = MetaRankConfig(top_candidate_pool=args.top_candidate_pool)
    summary = run_meta_rerank_compare(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        min_price=args.min_price,
        force_panel=args.force_panel,
        meta_config=meta_config,
        model_names=[part.strip() for part in args.model_names.split(",") if part.strip()],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
