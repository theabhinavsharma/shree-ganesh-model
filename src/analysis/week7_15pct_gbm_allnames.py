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
from src.analysis.week7_15pct_random_forest_allnames import _apply_calibration_15pct
from src.analysis.week7_15pct_random_forest_allnames import _build_historical_trailing_state
from src.analysis.week7_15pct_random_forest_allnames import _build_rationale
from src.analysis.week7_15pct_random_forest_allnames import _build_trailing_state
from src.analysis.week7_15pct_random_forest_allnames import _build_veto_columns
from src.analysis.week7_15pct_random_forest_allnames import _combine_focus_score
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_daily_metrics
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_selected_daily_metrics
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_selected_weekly_metrics
from src.analysis.week7_15pct_random_forest_allnames import _evaluate_weekly_metrics
from src.analysis.week7_15pct_random_forest_allnames import _jsonify
from src.analysis.week7_15pct_random_forest_allnames import _next_trading_day
from src.analysis.week7_15pct_random_forest_allnames import _safe_mean
from src.analysis.week7_15pct_random_forest_allnames import _safe_median
from src.analysis.week7_15pct_random_forest_allnames import _select_historical_fresh_entry_basket
from src.ml.config import ObjectiveSpec
from src.ml.config import load_research_config
from src.ml.expert_pipeline import _build_calibration_table
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
    return np.clip(model.predict(test_x), -0.25, 0.60).astype(np.float32)


def run_gbm_week7_15pct_allnames(
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
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_research_config(config_path)
    objective = ObjectiveSpec(
        name=f"week_7_15pct_{gbm_config.model_name}_allnames",
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
        scored = test[["trade_date", "symbol", "forward_return"]].copy()
        scored["winner_15pct"] = test_return >= 0.15
        scored["winner_20pct"] = test_return >= 0.20
        scored["prob_15pct_7d"] = prob_15
        scored["prob_20pct_7d"] = prob_20
        scored["pred_return_7d"] = pred_return
        scored["focus_score"] = _combine_focus_score(prob_15, prob_20, pred_return)
        oof_parts.append(scored)

    if not oof_parts:
        raise RuntimeError("No walk-forward folds produced predictions for the GBM 7D 15% study.")

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
    historical_trailing = _build_historical_trailing_state(base_config.paths.daily_facts, end_date=backtest_cutoff)
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
    current["prob_15pct_7d"] = _fit_predict_classifier(train_x=train_x, train_y=prob15_model_y, test_x=current_x, config=gbm_config)
    current["prob_20pct_7d"] = _fit_predict_classifier(train_x=train_x, train_y=prob20_model_y, test_x=current_x, config=gbm_config)
    current["pred_return_7d"] = _fit_predict_regressor(train_x=train_x, train_y=train_return, test_x=current_x, config=gbm_config)
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

    model_tag = gbm_config.model_name
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

    rejected_counts = rejected["veto_note"].value_counts(dropna=False).to_dict()
    summary = {
        "status": "ok",
        "as_of_trade_date": str(as_of_trade_date.date()),
        "trade_for_date": str(_next_trading_day(as_of_trade_date).date()),
        "run_type": f"{model_tag}_week7_15pct_allnames_fresh_entry",
        "historical_window": {
            "analysis_start_date": analysis_start_date,
            "backtest_end_date": "2025-12-31",
            "final_training_panel_end_date": str(pd.to_datetime(panel_full["trade_date"]).max().date()),
            "min_price": min_price,
        },
        "gbm_config": gbm_config.__dict__,
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
            f"All stocks were scored first with the {model_tag} route, then fresh-entry vetoes were applied to the top ranked candidate pool.",
            "The calibrated 15 percent hit rate is the honest confidence proxy; raw GBM probabilities are still ranking signals.",
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
            generated_by="src/analysis/week7_15pct_gbm_allnames.py",
            as_of_date=str(as_of_trade_date.date()),
            extra_notes=[f"All-stocks {model_tag} 7-day 15 percent study with fresh-entry vetoes."],
        )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_15pct_gbm_allnames.py",
        as_of_date=str(as_of_trade_date.date()),
        extra_notes=["Official NSE market-data path only. Derivatives OI remained unavailable and was not fabricated."],
    )
    write_report_directory_readme(
        output_dir,
        title=f"Week 7 Fifteen Percent {model_tag.title()} All Names",
        intro_lines=[
            f"This folder contains the no-universe 7-day 15 percent {model_tag} study with the refreshed official source stack.",
            "All stocks were ranked first, then the fresh-entry vetoes removed already-bloated or inconsistent candidates.",
            "The fresh-entry daily and weekly backtests are the correct benchmark for the live shortlist; raw pre-veto backtests are included for comparison.",
        ],
        files=[summary_path, top5_path, top10_path, rejected_path, raw_path, raw_daily_path, raw_weekly_path, daily_path, weekly_path],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the all-stocks 7-day 15 percent Gradient Boosting study and produce a fresh-entry-only shortlist.")
    parser.add_argument("--config", default="configs/ml_research.yaml")
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2026-04-21")
    parser.add_argument("--min-price", type=float, default=0.0)
    parser.add_argument("--output-dir", default="reports/week7_15pct_xgboost_allnames")
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

    summary = run_gbm_week7_15pct_allnames(
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
        fresh_rule=FreshEntryRule(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
