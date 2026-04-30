from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from src.analysis.day1_5pct_model import TRADABLE_UNIVERSES
from src.analysis.day1_5pct_model import summarize_universes
from src.ml.config import ObjectiveSpec
from src.ml.config import load_research_config
from src.ml.expert_pipeline import ExpertConfig
from src.ml.expert_pipeline import ExpertHorizonSpec
from src.ml.expert_pipeline import _combine_focus_score
from src.ml.expert_pipeline import _wilson_interval
from src.ml.expert_pipeline import load_or_evaluate_focus_horizon
from src.ml.feature_registry import available_feature_columns
from src.ml.panel import prepare_feature_panel
from src.ml.preprocess import fit_preprocess
from src.ml.preprocess import transform_frame
from src.ml.universes import build_universe_masks
from src.ml.walk_forward import build_yearly_walk_forward_folds
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme
from src.utils.io import write_json
from src.utils.io import write_parquet


@dataclass(frozen=True)
class ChallengerConfig:
    model_name: str
    classifier_trees: int
    regressor_trees: int
    max_depth: int
    min_samples_leaf: int
    n_jobs: int
    random_state: int = 42


@dataclass(frozen=True)
class ClassifierModel:
    constant_probability: float | None
    model: RandomForestClassifier | None


@dataclass(frozen=True)
class RegressorModel:
    constant_value: float | None
    model: RandomForestRegressor | None


def run_day1_model_challenger(
    *,
    config_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    shortlist_size: int,
    calibration_bins: int,
    challenger: ChallengerConfig,
    universe_names: tuple[str, ...],
    force_panel: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_research_config(config_path)
    scoped_config = replace(base_config, universes=list(universe_names))
    objective = ObjectiveSpec(
        name="day_1_5pct_eval",
        horizon_days=1,
        target_return=0.0,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    panel, panel_path = prepare_feature_panel(scoped_config, objective, force=force_panel)
    feature_columns = available_feature_columns(list(panel.columns), scoped_config.feature_columns)

    champion_predictions, champion_summary_df = _load_champion_oof(
        panel=panel,
        panel_path=panel_path,
        feature_columns=feature_columns,
        config_path=config_path,
        base_config=scoped_config,
        analysis_start_date=analysis_start_date,
        evaluation_end_date=evaluation_end_date,
        min_price=min_price,
        shortlist_size=shortlist_size,
        calibration_bins=calibration_bins,
        force=force_panel,
        universe_names=universe_names,
    )
    champion_metrics = summarize_universes(
        champion_predictions,
        champion_summary_df,
        top_quantile=scoped_config.top_quantile,
        top_n=shortlist_size,
    )

    challenger_predictions = _evaluate_random_forest_day1(
        panel=panel,
        feature_columns=feature_columns,
        base_config=scoped_config,
        challenger=challenger,
        universe_names=universe_names,
    )
    challenger_summary = _summarize_raw_predictions(
        challenger_predictions,
        top_quantile=scoped_config.top_quantile,
        top_n=shortlist_size,
    )
    challenger_metrics = summarize_universes(
        challenger_predictions,
        challenger_summary,
        top_quantile=scoped_config.top_quantile,
        top_n=shortlist_size,
    )

    comparison = _merge_comparison(champion_metrics, challenger_metrics, challenger_name=challenger.model_name)
    summary = _build_summary(
        panel=panel,
        config_path=config_path,
        panel_path=panel_path,
        analysis_start_date=analysis_start_date,
        evaluation_end_date=evaluation_end_date,
        min_price=min_price,
        challenger=challenger,
        comparison=comparison,
        universe_names=universe_names,
    )

    champion_metrics_path = output_dir / "champion_metrics.csv"
    challenger_metrics_path = output_dir / "challenger_metrics.csv"
    comparison_path = output_dir / "comparison_vs_champion.csv"
    challenger_predictions_path = output_dir / "challenger_oof_predictions.parquet"
    summary_path = output_dir / "summary.json"

    champion_metrics.to_csv(champion_metrics_path, index=False)
    challenger_metrics.to_csv(challenger_metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    write_parquet(challenger_predictions, challenger_predictions_path)
    write_json(summary, summary_path)

    as_of_date = evaluation_end_date
    write_dataframe_manifest(
        champion_metrics_path,
        champion_metrics,
        generated_by="src/analysis/day1_model_challenger.py",
        as_of_date=as_of_date,
        extra_notes=["Champion is the current production linear plus HistGradientBoosting ensemble."],
    )
    write_dataframe_manifest(
        challenger_metrics_path,
        challenger_metrics,
        generated_by="src/analysis/day1_model_challenger.py",
        as_of_date=as_of_date,
        extra_notes=[f"Challenger is {challenger.model_name} on the same day-1 walk-forward folds and tradable universes."],
    )
    write_dataframe_manifest(
        comparison_path,
        comparison,
        generated_by="src/analysis/day1_model_challenger.py",
        as_of_date=as_of_date,
        extra_notes=["Positive delta columns mean the challenger beat the champion on that metric."],
    )
    write_dataframe_manifest(
        challenger_predictions_path,
        challenger_predictions.head(min(len(challenger_predictions), 25000)),
        generated_by="src/analysis/day1_model_challenger.py",
        as_of_date=as_of_date,
        extra_notes=["Manifest is profiled on a compact sample. The parquet contains the full challenger OOF table."],
    )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/day1_model_challenger.py",
        as_of_date=as_of_date,
    )
    write_report_directory_readme(
        output_dir,
        title="Day 1 Model Challenger Benchmark",
        intro_lines=[
            "This folder compares the current production day-1 5 percent model against one challenger on the same walk-forward folds.",
            "Open `summary.json` first, then `comparison_vs_champion.csv`, then the universe metric files.",
            "All metrics are out-of-sample. No random shuffle or sampling is used in the benchmark itself.",
        ],
        files=[summary_path, comparison_path, champion_metrics_path, challenger_metrics_path, challenger_predictions_path],
    )
    return summary


def _load_champion_oof(
    *,
    panel: pd.DataFrame,
    panel_path: Path,
    feature_columns: list[str],
    config_path: Path,
    base_config,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    shortlist_size: int,
    calibration_bins: int,
    force: bool,
    universe_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_horizon = ExpertHorizonSpec(
        name="day_1",
        horizon_days=1,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    expert_config = ExpertConfig(
        base_config_path=config_path,
        base_config=base_config,
        horizons=[focus_horizon],
        focus_horizon=focus_horizon.name,
        shortlist_size=shortlist_size,
        calibration_bins=calibration_bins,
        run_output_dir=Path("reports/tmp_day1_champion_cache"),
    )
    predictions, raw_summaries = load_or_evaluate_focus_horizon(
        panel,
        feature_columns=feature_columns,
        config=expert_config,
        horizon_spec=focus_horizon,
        panel_path=panel_path,
        force=force,
    )
    summary_df = pd.DataFrame(raw_summaries).sort_values(
        ["sort_primary", "sort_secondary", "sort_tertiary", "universe_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return predictions.loc[predictions["universe_name"].isin(universe_names)].copy(), summary_df


def _evaluate_random_forest_day1(
    *,
    panel: pd.DataFrame,
    feature_columns: list[str],
    base_config,
    challenger: ChallengerConfig,
    universe_names: tuple[str, ...],
) -> pd.DataFrame:
    universe_masks = build_universe_masks(panel)
    min_train_end_year = pd.Timestamp(base_config.train_end_date).year
    folds = build_yearly_walk_forward_folds(panel, min_train_end_year=min_train_end_year)
    all_predictions: list[pd.DataFrame] = []

    for universe_name in universe_names:
        if universe_name not in universe_masks:
            continue
        scoped = panel.loc[universe_masks[universe_name].fillna(False).astype(bool)].copy()
        if scoped.empty:
            continue
        fold_predictions: list[pd.DataFrame] = []
        for fold in folds:
            train_mask = scoped["trade_date"].le(fold.train_end_date)
            test_mask = scoped["trade_date"].between(fold.test_start_date, fold.test_end_date)
            train = scoped.loc[train_mask].copy()
            test = scoped.loc[test_mask].copy()
            if len(train) < base_config.min_train_rows or len(test) < base_config.min_test_rows:
                continue

            stats = fit_preprocess(train, feature_columns)
            train_x = transform_frame(train, stats)
            test_x = transform_frame(test, stats)

            train_return = pd.to_numeric(train["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            test_return = pd.to_numeric(test["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

            five_model = _fit_random_forest_classifier(
                train_x,
                (train_return >= 0.05).astype(np.int8),
                challenger=challenger,
            )
            ten_model = _fit_random_forest_classifier(
                train_x,
                (train_return >= 0.10).astype(np.int8),
                challenger=challenger,
            )
            reg_model = _fit_random_forest_regressor(
                train_x,
                train_return,
                challenger=challenger,
            )

            prob_5 = _predict_random_forest_classifier(five_model, test_x)
            prob_10 = _predict_random_forest_classifier(ten_model, test_x)
            pred_return = _predict_random_forest_regressor(reg_model, test_x)

            fold_frame = test[["trade_date", "symbol", "close", "forward_return"]].copy()
            fold_frame["fold_name"] = fold.fold_name
            fold_frame["universe_name"] = universe_name
            fold_frame["horizon_name"] = "day_1"
            fold_frame["winner_5pct"] = test_return >= 0.05
            fold_frame["prob_5pct"] = prob_5
            fold_frame["prob_10pct"] = prob_10
            fold_frame["pred_return"] = pred_return
            fold_frame["focus_score"] = _combine_focus_score(prob_5, prob_10, pred_return)
            fold_predictions.append(fold_frame)

        if fold_predictions:
            all_predictions.append(pd.concat(fold_predictions, ignore_index=True))

    if not all_predictions:
        return pd.DataFrame()
    return pd.concat(all_predictions, ignore_index=True)


def _fit_random_forest_classifier(
    x: np.ndarray,
    y: np.ndarray,
    *,
    challenger: ChallengerConfig,
) -> ClassifierModel:
    if len(np.unique(y)) < 2:
        return ClassifierModel(constant_probability=float(y.mean()), model=None)
    model = RandomForestClassifier(
        n_estimators=challenger.classifier_trees,
        max_depth=challenger.max_depth,
        min_samples_leaf=challenger.min_samples_leaf,
        class_weight="balanced_subsample",
        max_features="sqrt",
        n_jobs=challenger.n_jobs,
        random_state=challenger.random_state,
    )
    model.fit(x, y)
    return ClassifierModel(constant_probability=None, model=model)


def _predict_random_forest_classifier(model: ClassifierModel, x: np.ndarray) -> np.ndarray:
    if model.constant_probability is not None:
        return np.full(len(x), model.constant_probability, dtype=np.float32)
    return model.model.predict_proba(x)[:, 1].astype(np.float32)


def _fit_random_forest_regressor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    challenger: ChallengerConfig,
) -> RegressorModel:
    if len(y) == 0 or float(np.nanstd(y)) < 1e-8:
        return RegressorModel(constant_value=float(np.nanmean(y)) if len(y) else 0.0, model=None)
    model = RandomForestRegressor(
        n_estimators=challenger.regressor_trees,
        max_depth=challenger.max_depth,
        min_samples_leaf=challenger.min_samples_leaf,
        max_features="sqrt",
        n_jobs=challenger.n_jobs,
        random_state=challenger.random_state,
    )
    model.fit(x, y)
    return RegressorModel(constant_value=None, model=model)


def _predict_random_forest_regressor(model: RegressorModel, x: np.ndarray) -> np.ndarray:
    if model.constant_value is not None:
        return np.full(len(x), model.constant_value, dtype=np.float32)
    return np.clip(model.model.predict(x), -0.15, 0.25).astype(np.float32)


def _summarize_raw_predictions(
    predictions: pd.DataFrame,
    *,
    top_quantile: float,
    top_n: int,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "horizon_name",
                "universe_name",
                "row_count",
                "base_rate_5pct",
                "top_quantile_precision_5pct",
                "top_quantile_return",
                "top_n_precision_5pct",
                "top_n_return",
                "selected_row_count",
                "selected_hit_count",
                "selected_precision_pooled",
                "selected_precision_ci_low",
                "selected_precision_ci_high",
                "selected_precision_p_value",
                "sort_primary",
                "sort_secondary",
                "sort_tertiary",
            ]
        )
    rows: list[dict[str, object]] = []
    for universe_name, frame in predictions.groupby("universe_name", sort=False):
        working = frame.sort_values(["trade_date", "focus_score", "symbol"], ascending=[True, False, True]).copy()
        top_quantile_rows = _select_daily_top_quantile(working, score_col="focus_score", top_quantile=top_quantile)
        hit_count = int(pd.to_numeric(top_quantile_rows["winner_5pct"], errors="coerce").fillna(0).sum())
        total_selected = int(len(top_quantile_rows))
        base_rate = float(pd.to_numeric(working["winner_5pct"], errors="coerce").fillna(0).mean())
        top_quantile_precision = float(hit_count / total_selected) if total_selected else None
        top_quantile_return = float(pd.to_numeric(top_quantile_rows["forward_return"], errors="coerce").mean()) if total_selected else None
        top_n_rows = _select_daily_top_n(working, top_n=top_n)
        top_n_hits = int(pd.to_numeric(top_n_rows["winner_5pct"], errors="coerce").fillna(0).sum())
        top_n_precision = float(top_n_hits / len(top_n_rows)) if len(top_n_rows) else None
        top_n_return = float(pd.to_numeric(top_n_rows["forward_return"], errors="coerce").mean()) if len(top_n_rows) else None
        ci_low, ci_high = _wilson_interval(hit_count, total_selected)
        rows.append(
            {
                "horizon_name": "day_1",
                "universe_name": universe_name,
                "row_count": int(len(working)),
                "base_rate_5pct": base_rate,
                "top_quantile_precision_5pct": top_quantile_precision,
                "top_quantile_return": top_quantile_return,
                "top_n_precision_5pct": top_n_precision,
                "top_n_return": top_n_return,
                "selected_row_count": total_selected,
                "selected_hit_count": hit_count,
                "selected_precision_pooled": top_quantile_precision,
                "selected_precision_ci_low": ci_low,
                "selected_precision_ci_high": ci_high,
                "selected_precision_p_value": None,
                "sort_primary": float(top_quantile_precision or -1.0),
                "sort_secondary": float(top_quantile_return or -1.0),
                "sort_tertiary": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _select_daily_top_quantile(frame: pd.DataFrame, *, score_col: str, top_quantile: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("trade_date", sort=False):
        ordered = group.sort_values([score_col, "symbol"], ascending=[False, True])
        top_n = max(1, int(np.ceil(len(ordered) * top_quantile)))
        parts.append(ordered.head(top_n).copy())
    if not parts:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(parts, ignore_index=True)


def _select_daily_top_n(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("trade_date", sort=False):
        ordered = group.sort_values(["focus_score", "symbol"], ascending=[False, True])
        parts.append(ordered.head(top_n).copy())
    if not parts:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(parts, ignore_index=True)


def _merge_comparison(
    champion_metrics: pd.DataFrame,
    challenger_metrics: pd.DataFrame,
    *,
    challenger_name: str,
) -> pd.DataFrame:
    keep_columns = [
        "universe_name",
        "row_count",
        "base_rate_5pct",
        "top_bucket_hit_rate",
        "top10_precision_5pct",
        "top10_recall",
        "top10_mean_return_mean",
        "top10_median_stock_return_median",
        "top10_p75_stock_return_median",
        "selection_rank_score",
    ]
    champion = champion_metrics[keep_columns].copy()
    challenger = challenger_metrics[keep_columns].copy()
    merged = champion.merge(challenger, on=["universe_name"], suffixes=("_champion", f"_{challenger_name}"))
    merged["delta_top10_precision_5pct"] = merged[f"top10_precision_5pct_{challenger_name}"] - merged["top10_precision_5pct_champion"]
    merged["delta_top10_recall"] = merged[f"top10_recall_{challenger_name}"] - merged["top10_recall_champion"]
    merged["delta_top10_mean_return_mean"] = merged[f"top10_mean_return_mean_{challenger_name}"] - merged["top10_mean_return_mean_champion"]
    merged["delta_top10_median_stock_return_median"] = merged[f"top10_median_stock_return_median_{challenger_name}"] - merged["top10_median_stock_return_median_champion"]
    merged["delta_top10_p75_stock_return_median"] = merged[f"top10_p75_stock_return_median_{challenger_name}"] - merged["top10_p75_stock_return_median_champion"]
    merged["delta_selection_rank_score"] = merged[f"selection_rank_score_{challenger_name}"] - merged["selection_rank_score_champion"]
    merged["winner_by_selection_rank_score"] = np.where(
        merged[f"selection_rank_score_{challenger_name}"] > merged["selection_rank_score_champion"],
        challenger_name,
        np.where(
            merged[f"selection_rank_score_{challenger_name}"] < merged["selection_rank_score_champion"],
            "champion",
            "tie",
        ),
    )
    return merged.sort_values("delta_selection_rank_score", ascending=False).reset_index(drop=True)


def _build_summary(
    *,
    panel: pd.DataFrame,
    config_path: Path,
    panel_path: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    challenger: ChallengerConfig,
    comparison: pd.DataFrame,
    universe_names: tuple[str, ...],
) -> dict[str, object]:
    best_row = comparison.sort_values(
        ["delta_selection_rank_score", f"selection_rank_score_{challenger.model_name}"],
        ascending=[False, False],
    ).iloc[0].to_dict()
    return {
        "status": "ok",
        "objective": {
            "name": "day_1_5pct_eval",
            "horizon_days": 1,
            "target_return": 0.05,
            "analysis_start_date": analysis_start_date,
            "evaluation_end_date": evaluation_end_date,
            "min_price": min_price,
        },
        "config_path": str(config_path),
        "panel_path": str(panel_path),
        "panel_rows": int(len(panel)),
        "challenger": {
            "name": challenger.model_name,
            "classifier_trees": challenger.classifier_trees,
            "regressor_trees": challenger.regressor_trees,
            "max_depth": challenger.max_depth,
            "min_samples_leaf": challenger.min_samples_leaf,
            "n_jobs": challenger.n_jobs,
            "random_state": challenger.random_state,
        },
        "universe_names": list(universe_names),
        "best_delta_universe": best_row.get("universe_name"),
        "best_delta_summary": {key: _jsonify(value) for key, value in best_row.items()},
        "notes": [
            "Champion is the current production day-1 stack: linear classifier plus HistGradientBoosting for classification, and Ridge plus HistGradientBoosting for return regression.",
            f"Challenger is {challenger.model_name} trained on the same tradable universes and yearly walk-forward folds.",
            "Positive delta columns in comparison_vs_champion.csv mean the challenger improved on that metric.",
        ],
    }


def _jsonify(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest one day model challenger versus the current champion.")
    parser.add_argument("--config", type=Path, default=Path("configs/ml_research.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2025-12-31")
    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--shortlist-size", type=int, default=10)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--classifier-trees", type=int, default=120)
    parser.add_argument("--regressor-trees", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--universes",
        default="mid_small,liquid_5cr_plus,liquid_20cr_plus",
        help="Comma-separated universes to benchmark. Defaults to the main live tradable universes.",
    )
    parser.add_argument("--force-panel", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    challenger = ChallengerConfig(
        model_name="random_forest",
        classifier_trees=args.classifier_trees,
        regressor_trees=args.regressor_trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
    )
    universe_names = tuple(item.strip() for item in str(args.universes).split(",") if item.strip())
    run_day1_model_challenger(
        config_path=args.config,
        output_dir=args.output_dir,
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        min_price=args.min_price,
        shortlist_size=args.shortlist_size,
        calibration_bins=args.calibration_bins,
        challenger=challenger,
        universe_names=universe_names,
        force_panel=args.force_panel,
    )


if __name__ == "__main__":
    main()
