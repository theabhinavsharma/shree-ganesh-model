from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.week7_15pct_gbm_allnames import GBMConfig
from src.analysis.week7_15pct_gbm_allnames import _fit_predict_classifier
from src.analysis.week7_15pct_gbm_allnames import _fit_predict_regressor
from src.analysis.week7_15pct_random_forest_allnames import FreshEntryRule
from src.analysis.week7_15pct_random_forest_allnames import _build_historical_trailing_state
from src.analysis.week7_15pct_random_forest_allnames import _build_rationale
from src.analysis.week7_15pct_random_forest_allnames import _build_veto_columns
from src.analysis.week7_15pct_random_forest_allnames import _combine_focus_score
from src.analysis.week7_15pct_random_forest_allnames import _coerce_numeric_series
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_selected_daily_metrics
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_selected_weekly_metrics
from src.analysis.week7_15pct_random_forest_allnames import _jsonify
from src.ml.config import ObjectiveSpec
from src.ml.config import load_research_config
from src.ml.feature_registry import available_feature_columns
from src.ml.panel import prepare_feature_panel
from src.ml.preprocess import fit_preprocess
from src.ml.preprocess import transform_frame
from src.ml.walk_forward import build_yearly_walk_forward_folds
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme
from src.utils.io import write_json


def _apply_extra_filters(
    ranked_pool: pd.DataFrame,
    *,
    min_close: float,
    min_volume_vs_20d: float,
    min_return_7td: float,
    min_return_15td: float,
    min_return_30td: float,
    min_rsi_14_daily: float,
    max_rsi_14_daily: float,
    require_close_above_sma200: bool,
) -> pd.DataFrame:
    working = ranked_pool.copy()
    close = _coerce_numeric_series(working, "close")
    volume_vs_20d = _coerce_numeric_series(working, "volume_vs_20d")
    return_7td = _coerce_numeric_series(working, "return_7td")
    return_15td = _coerce_numeric_series(working, "return_15td")
    return_30td = _coerce_numeric_series(working, "return_30td")
    rsi_14_daily = _coerce_numeric_series(working, "rsi_14_daily")
    sma_200 = _coerce_numeric_series(working, "sma_200")

    extra_pass = (
        close.ge(min_close)
        & volume_vs_20d.ge(min_volume_vs_20d)
        & return_7td.ge(min_return_7td)
        & return_15td.ge(min_return_15td)
        & return_30td.ge(min_return_30td)
        & rsi_14_daily.ge(min_rsi_14_daily)
        & rsi_14_daily.le(max_rsi_14_daily)
    )
    if require_close_above_sma200:
        extra_pass = extra_pass & close.ge(sma_200)

    selected = working.loc[extra_pass].copy().reset_index(drop=True)
    selected["post_veto_rank"] = np.arange(1, len(selected) + 1)
    return selected


def _collect_candidate_pool(
    *,
    config_path: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    top_candidate_pool: int,
    gbm_config: GBMConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_config = load_research_config(config_path)
    objective = ObjectiveSpec(
        name="week_7_15pct_xgboost_allnames_median_study",
        horizon_days=7,
        target_return=0.15,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    panel_full, _panel_path = prepare_feature_panel(base_config, objective, force=False)
    feature_columns = available_feature_columns(list(panel_full.columns), base_config.feature_columns)
    backtest_cutoff = pd.Timestamp("2025-12-31")
    panel_backtest = panel_full.loc[pd.to_datetime(panel_full["trade_date"]).le(backtest_cutoff)].copy()

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

        prob_15 = _fit_predict_classifier(
            train_x=train_x,
            train_y=(train_return >= 0.15).astype(np.int8),
            test_x=test_x,
            config=gbm_config,
        )
        prob_20 = _fit_predict_classifier(
            train_x=train_x,
            train_y=(train_return >= 0.20).astype(np.int8),
            test_x=test_x,
            config=gbm_config,
        )
        pred_return = _fit_predict_regressor(
            train_x=train_x,
            train_y=train_return,
            test_x=test_x,
            config=gbm_config,
        )
        scored = test[["trade_date", "symbol", "forward_return", "close"]].copy()
        scored["winner_15pct"] = test_return >= 0.15
        scored["winner_20pct"] = test_return >= 0.20
        scored["prob_15pct_7d"] = prob_15
        scored["prob_20pct_7d"] = prob_20
        scored["pred_return_7d"] = pred_return
        scored["focus_score"] = _combine_focus_score(prob_15, prob_20, pred_return)
        oof_parts.append(scored)

    if not oof_parts:
        raise RuntimeError("No walk-forward folds produced predictions for the XGBoost median study.")

    predictions = (
        pd.concat(oof_parts, ignore_index=True)
        .sort_values(["trade_date", "focus_score", "symbol"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

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
        "sma_50",
        "sma_200",
    ]
    feature_keep_cols = [col for col in feature_keep_cols if col in panel_backtest.columns]
    feature_frame = panel_backtest[feature_keep_cols].copy()
    trailing_state = _build_historical_trailing_state(base_config.paths.daily_facts, end_date=backtest_cutoff)
    enriched = predictions.merge(feature_frame, on=["symbol", "trade_date"], how="left")
    enriched = enriched.merge(trailing_state, on=["symbol", "trade_date"], how="left")

    ranked_parts: list[pd.DataFrame] = []
    base_rule = FreshEntryRule()
    for _, group in enriched.groupby("trade_date", sort=False):
        ranked = group.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(top_candidate_pool).copy()
        ranked = _build_veto_columns(ranked, rule=base_rule)
        ranked["rationale"] = _build_rationale(ranked)
        ranked_parts.append(ranked)
    ranked_pool = pd.concat(ranked_parts, ignore_index=True)
    return predictions, ranked_pool


def run_study(
    *,
    config_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    top_candidate_pool: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, ranked_pool = _collect_candidate_pool(
        config_path=config_path,
        analysis_start_date=analysis_start_date,
        evaluation_end_date=evaluation_end_date,
        min_price=min_price,
        top_candidate_pool=top_candidate_pool,
        gbm_config=GBMConfig(
            model_name="xgboost",
            classifier_trees=220,
            regressor_trees=160,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=10.0,
            min_samples_leaf=200,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=4,
        ),
    )

    sweep_rows: list[dict[str, object]] = []
    grids = itertools.product(
        [0.0, 20.0],
        [0.0, 1.0, 1.5],
        [0.0, 0.03],
        [0.0, 0.05, 0.10],
        [-0.05, 0.0, 0.05],
        [50.0, 55.0, 60.0],
        [72.0, 75.0, 78.0],
        [False, True],
    )
    all_days = int(predictions["trade_date"].nunique())
    for (
        min_close,
        min_volume_vs_20d,
        min_return_7td,
        min_return_15td,
        min_return_30td,
        min_rsi_14_daily,
        max_rsi_14_daily,
        require_close_above_sma200,
    ) in grids:
        if min_rsi_14_daily >= max_rsi_14_daily:
            continue
        selected_frames: list[pd.DataFrame] = []
        days_with_five = 0
        avg_survivors: list[int] = []
        for _, group in ranked_pool.groupby("trade_date", sort=False):
            selected = _apply_extra_filters(
                group,
                min_close=min_close,
                min_volume_vs_20d=min_volume_vs_20d,
                min_return_7td=min_return_7td,
                min_return_15td=min_return_15td,
                min_return_30td=min_return_30td,
                min_rsi_14_daily=min_rsi_14_daily,
                max_rsi_14_daily=max_rsi_14_daily,
                require_close_above_sma200=require_close_above_sma200,
            )
            avg_survivors.append(len(selected))
            if len(selected) >= 5:
                days_with_five += 1
            if not selected.empty:
                selected_frames.append(selected)
        selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(columns=ranked_pool.columns)
        daily5 = _evaluate_selected_daily_metrics(selected_all, predictions, top_n=5)
        weekly5 = _evaluate_selected_weekly_metrics(selected_all, predictions, top_n=5)
        sweep_rows.append(
            {
                "min_close": min_close,
                "min_volume_vs_20d": min_volume_vs_20d,
                "min_return_7td": min_return_7td,
                "min_return_15td": min_return_15td,
                "min_return_30td": min_return_30td,
                "min_rsi_14_daily": min_rsi_14_daily,
                "max_rsi_14_daily": max_rsi_14_daily,
                "require_close_above_sma200": require_close_above_sma200,
                "days_with_full_top5_rate": days_with_five / all_days if all_days else np.nan,
                "avg_survivors_per_day": float(np.mean(avg_survivors)) if avg_survivors else np.nan,
                "daily_top5_precision": daily5["precision_15pct"],
                "daily_top5_mean_return": daily5["mean_return_mean"],
                "daily_top5_median_return": daily5["median_stock_return_median"],
                "daily_top5_p75_return": daily5["p75_stock_return_median"],
                "weekly_top5_precision": weekly5["precision_15pct"],
                "weekly_top5_mean_return": weekly5["mean_return_mean"],
                "weekly_top5_median_return": weekly5["median_stock_return_median"],
            }
        )

    sweep = pd.DataFrame(sweep_rows)
    sweep = sweep.sort_values(
        ["daily_top5_median_return", "daily_top5_precision", "days_with_full_top5_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    feasible = sweep.loc[
        sweep["daily_top5_median_return"].ge(0.003)
        & sweep["days_with_full_top5_rate"].ge(0.25)
    ].copy()
    best = feasible.head(20) if not feasible.empty else sweep.head(20)

    sweep_path = output_dir / "threshold_sweep.csv"
    feasible_path = output_dir / "feasible_configs.csv"
    best_path = output_dir / "top20_configs.csv"
    summary_path = output_dir / "summary.json"

    sweep.to_csv(sweep_path, index=False)
    feasible.to_csv(feasible_path, index=False)
    best.to_csv(best_path, index=False)

    summary = {
        "status": "ok",
        "study_target": "Push daily top-5 median return toward +0.3% on the all-stocks 7D 15% XGBoost route.",
        "historical_window": {
            "analysis_start_date": analysis_start_date,
            "backtest_end_date": "2025-12-31",
            "top_candidate_pool": top_candidate_pool,
        },
        "baseline_trend_floor": FreshEntryRule().__dict__,
        "grid_size": int(len(sweep)),
        "feasible_count": int(len(feasible)),
        "best_config": best.iloc[0].to_dict() if len(best) else None,
        "note": (
            "Feasible configs require daily top-5 median return >= 0.3% and at least 25% of historical days with five or more survivors. "
            "If feasible_count is zero, the target was not reached with this grid."
        ),
    }
    summary = _jsonify(summary)
    write_json(summary, summary_path)

    for path, df, note in [
        (sweep_path, sweep, "Full threshold sweep for the XGBoost top-5 median study."),
        (feasible_path, feasible, "Configs that met the median target and minimum survivor coverage."),
        (best_path, best, "Top 20 configs ranked by daily top-5 median return."),
    ]:
        write_dataframe_manifest(
            path,
            df,
            generated_by="src/analysis/week7_15pct_xgboost_median_study.py",
            as_of_date="2025-12-31",
            extra_notes=[note],
        )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_15pct_xgboost_median_study.py",
        as_of_date="2025-12-31",
        extra_notes=["Historical threshold study only. No live market data was fabricated or substituted."],
    )
    write_report_directory_readme(
        output_dir,
        title="Week 7 Fifteen Percent XGBoost Median Study",
        intro_lines=[
            "This folder contains a threshold sweep to improve the daily top-5 median return for the all-stocks 7-day 15 percent XGBoost route.",
            "The study reuses the same walk-forward XGBoost predictions and then sweeps stricter post-rank quality filters on the top-100 daily candidate pool.",
            "Feasible configs must hit the target median and still produce at least five names on a meaningful fraction of days.",
        ],
        files=[summary_path, sweep_path, feasible_path, best_path],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a threshold sweep to push the all-stocks 7D 15% XGBoost top-5 median return higher.")
    parser.add_argument("--config", default="configs/ml_research.yaml")
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2026-04-21")
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--top-candidate-pool", type=int, default=100)
    parser.add_argument("--output-dir", default="reports/week7_15pct_xgboost_median_study_20260422")
    args = parser.parse_args()

    summary = run_study(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        min_price=args.min_price,
        top_candidate_pool=args.top_candidate_pool,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
