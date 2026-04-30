from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from src.ml.config import ObjectiveSpec
from src.ml.config import load_research_config
from src.ml.expert_pipeline import _combine_focus_score
from src.ml.expert_pipeline import _wilson_interval
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


@dataclass(frozen=True)
class RFQuickConfig:
    universe_name: str
    classifier_trees: int
    regressor_trees: int
    max_depth: int
    min_samples_leaf: int
    n_jobs: int
    random_state: int = 42


def run_quick_compare(
    *,
    config_path: Path,
    champion_metrics_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    challenger: RFQuickConfig,
    top_n: int,
    force_panel: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_research_config(config_path)
    objective = ObjectiveSpec(
        name="week_7_5pct_eval",
        horizon_days=7,
        target_return=0.0,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    panel, panel_path = prepare_feature_panel(base_config, objective, force=force_panel)
    feature_columns = available_feature_columns(list(panel.columns), base_config.feature_columns)
    universe_masks = build_universe_masks(panel)
    scoped = panel.loc[universe_masks[challenger.universe_name].fillna(False).astype(bool)].copy()
    if scoped.empty:
        raise ValueError(f"Universe {challenger.universe_name} is empty in the prepared panel.")

    challenger_metrics = _evaluate_rf_metrics(
        frame=scoped,
        feature_columns=feature_columns,
        train_end_date=base_config.train_end_date,
        min_train_rows=base_config.min_train_rows,
        min_test_rows=base_config.min_test_rows,
        top_n=top_n,
        challenger=challenger,
    )
    champion_metrics = _load_champion_metrics(champion_metrics_path, challenger.universe_name, top_n)
    comparison = _compare_rows(champion_metrics, challenger_metrics)

    challenger_df = pd.DataFrame([challenger_metrics])
    comparison_df = pd.DataFrame([comparison])
    challenger_path = output_dir / "challenger_metrics.csv"
    comparison_path = output_dir / "comparison_vs_champion.csv"
    summary_path = output_dir / "summary.json"

    challenger_df.to_csv(challenger_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    summary = {
        "status": "ok",
        "objective": {
            "name": "week_7_5pct_eval",
            "horizon_days": 7,
            "target_return": 0.05,
            "analysis_start_date": analysis_start_date,
            "evaluation_end_date": evaluation_end_date,
            "min_price": min_price,
            "top_n": top_n,
        },
        "panel_path": str(panel_path),
        "panel_rows_total": int(len(panel)),
        "panel_rows_universe": int(len(scoped)),
        "universe_name": challenger.universe_name,
        "champion_metrics_path": str(champion_metrics_path),
        "challenger": {
            "name": "random_forest",
            "classifier_trees": challenger.classifier_trees,
            "regressor_trees": challenger.regressor_trees,
            "max_depth": challenger.max_depth,
            "min_samples_leaf": challenger.min_samples_leaf,
            "n_jobs": challenger.n_jobs,
            "random_state": challenger.random_state,
        },
        "comparison": comparison,
    }
    write_json(summary, summary_path)

    write_dataframe_manifest(
        challenger_path,
        challenger_df,
        generated_by="src/analysis/week7_random_forest_quick_compare.py",
        as_of_date=evaluation_end_date,
        extra_notes=["Challenger is Random Forest on the same 7-day target and yearly walk-forward folds."],
    )
    write_dataframe_manifest(
        comparison_path,
        comparison_df,
        generated_by="src/analysis/week7_random_forest_quick_compare.py",
        as_of_date=evaluation_end_date,
        extra_notes=["Positive delta columns mean Random Forest improved on the champion metric."],
    )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_random_forest_quick_compare.py",
        as_of_date=evaluation_end_date,
    )
    write_report_directory_readme(
        output_dir,
        title="Week 7 Random Forest Quick Compare",
        intro_lines=[
            "This folder compares a Random Forest challenger against the existing 7-day champion for one universe.",
            "The comparison uses the same full-history walk-forward setup and computes basket metrics directly to stay reliable on this machine.",
            "Open `summary.json` first, then `comparison_vs_champion.csv`.",
        ],
        files=[summary_path, comparison_path, challenger_path],
    )
    return summary


def _evaluate_rf_metrics(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_end_date: str,
    min_train_rows: int,
    min_test_rows: int,
    top_n: int,
    challenger: RFQuickConfig,
) -> dict[str, float | str]:
    folds = build_yearly_walk_forward_folds(frame, min_train_end_year=pd.Timestamp(train_end_date).year)
    total_rows = 0
    winner_count = 0
    selected_rows = 0
    selected_hits = 0
    daily_mean_returns: list[float] = []
    daily_median_returns: list[float] = []
    daily_p75_returns: list[float] = []

    for fold in folds:
        train_mask = frame["trade_date"].le(fold.train_end_date)
        test_mask = frame["trade_date"].between(fold.test_start_date, fold.test_end_date)
        train = frame.loc[train_mask].copy()
        test = frame.loc[test_mask].copy()
        if len(train) < min_train_rows or len(test) < min_test_rows:
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
            challenger=challenger,
        )
        prob_10 = _fit_predict_classifier(
            train_x=train_x,
            train_y=(train_return >= 0.10).astype(np.int8),
            test_x=test_x,
            challenger=challenger,
        )
        pred_return = _fit_predict_regressor(
            train_x=train_x,
            train_y=train_return,
            test_x=test_x,
            challenger=challenger,
        )

        scored = test[["trade_date", "symbol", "forward_return"]].copy()
        scored["winner_5pct"] = test_return >= 0.05
        scored["focus_score"] = _combine_focus_score(prob_5, prob_10, pred_return)
        total_rows += int(len(scored))
        winner_count += int(scored["winner_5pct"].sum())

        for _, group in scored.groupby("trade_date", sort=False):
            top = group.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(top_n).copy()
            returns = pd.to_numeric(top["forward_return"], errors="coerce").dropna()
            if returns.empty:
                continue
            selected_rows += int(len(top))
            selected_hits += int(pd.to_numeric(top["winner_5pct"], errors="coerce").fillna(0).sum())
            daily_mean_returns.append(float(returns.mean()))
            daily_median_returns.append(float(returns.median()))
            daily_p75_returns.append(float(returns.quantile(0.75)))

    base_rate = float(winner_count / total_rows) if total_rows else np.nan
    top10_precision = float(selected_hits / selected_rows) if selected_rows else np.nan
    top10_recall = float(selected_hits / winner_count) if winner_count else np.nan
    top10_ci_low, top10_ci_high = _wilson_interval(selected_hits, selected_rows)
    top10_p_value = float(binomtest(selected_hits, selected_rows, p=base_rate, alternative="greater").pvalue) if selected_rows else np.nan
    return {
        "universe_name": challenger.universe_name,
        "row_count": total_rows,
        "base_rate_5pct": base_rate,
        "winner_count": winner_count,
        "top10_precision_5pct": top10_precision,
        "top10_recall": top10_recall,
        "top10_ci_low": top10_ci_low,
        "top10_ci_high": top10_ci_high,
        "top10_p_value": top10_p_value,
        "top10_mean_return_mean": _safe_mean(daily_mean_returns),
        "top10_median_stock_return_median": _safe_median(daily_median_returns),
        "top10_p75_stock_return_median": _safe_median(daily_p75_returns),
        "precision_lift_top10": float(top10_precision / base_rate) if base_rate and not np.isnan(base_rate) else np.nan,
        "model_name": "random_forest",
    }


def _fit_predict_classifier(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    challenger: RFQuickConfig,
) -> np.ndarray:
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(train_y.mean()), dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=challenger.classifier_trees,
        max_depth=challenger.max_depth,
        min_samples_leaf=challenger.min_samples_leaf,
        class_weight="balanced_subsample",
        max_features="sqrt",
        n_jobs=challenger.n_jobs,
        random_state=challenger.random_state,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1].astype(np.float32)


def _fit_predict_regressor(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    challenger: RFQuickConfig,
) -> np.ndarray:
    if len(train_y) == 0 or float(np.nanstd(train_y)) < 1e-8:
        constant = float(np.nanmean(train_y)) if len(train_y) else 0.0
        return np.full(len(test_x), constant, dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=challenger.regressor_trees,
        max_depth=challenger.max_depth,
        min_samples_leaf=challenger.min_samples_leaf,
        max_features="sqrt",
        n_jobs=challenger.n_jobs,
        random_state=challenger.random_state,
    )
    model.fit(train_x, train_y)
    return np.clip(model.predict(test_x), -0.20, 0.25).astype(np.float32)


def _load_champion_metrics(path: Path, universe_name: str, top_n: int) -> dict[str, float | str]:
    metrics = pd.read_csv(path)
    row = metrics.loc[(metrics["universe_name"] == universe_name) & (metrics["top_n"] == top_n)]
    if row.empty:
        raise ValueError(f"Universe {universe_name} top_n {top_n} not found in champion metrics file {path}.")
    return row.iloc[0].to_dict()


def _compare_rows(champion: dict[str, object], challenger: dict[str, object]) -> dict[str, object]:
    return {
        "universe_name": challenger["universe_name"],
        "top_n": champion["top_n"],
        "top10_precision_champion": champion["precision_5pct"],
        "top10_precision_random_forest": challenger["top10_precision_5pct"],
        "delta_top10_precision": float(challenger["top10_precision_5pct"]) - float(champion["precision_5pct"]),
        "precision_lift_champion": champion["precision_lift"],
        "precision_lift_random_forest": challenger["precision_lift_top10"],
        "delta_precision_lift": float(challenger["precision_lift_top10"]) - float(champion["precision_lift"]),
        "top10_recall_champion": champion["recall"],
        "top10_recall_random_forest": challenger["top10_recall"],
        "delta_top10_recall": float(challenger["top10_recall"]) - float(champion["recall"]),
        "top10_mean_return_champion": champion["mean_return_mean"],
        "top10_mean_return_random_forest": challenger["top10_mean_return_mean"],
        "delta_top10_mean_return": float(challenger["top10_mean_return_mean"]) - float(champion["mean_return_mean"]),
        "top10_median_return_champion": champion["median_stock_return_median"],
        "top10_median_return_random_forest": challenger["top10_median_stock_return_median"],
        "delta_top10_median_return": float(challenger["top10_median_stock_return_median"]) - float(champion["median_stock_return_median"]),
        "top10_p75_return_champion": champion["p75_stock_return_median"],
        "top10_p75_return_random_forest": challenger["top10_p75_stock_return_median"],
        "delta_top10_p75_return": float(challenger["top10_p75_stock_return_median"]) - float(champion["p75_stock_return_median"]),
        "winner_metric": "random_forest" if float(challenger["top10_precision_5pct"]) > float(champion["precision_5pct"]) else "champion",
    }


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else np.nan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick Random Forest versus champion compare for week-7 5 percent target.")
    parser.add_argument("--config", type=Path, default=Path("configs/ml_research.yaml"))
    parser.add_argument("--champion-metrics", type=Path, default=Path("reports/week7_topn_backtest_20260414/daily_topn_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2025-12-31")
    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--universe", default="mid_small")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--classifier-trees", type=int, default=20)
    parser.add_argument("--regressor-trees", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--force-panel", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    challenger = RFQuickConfig(
        universe_name=args.universe,
        classifier_trees=args.classifier_trees,
        regressor_trees=args.regressor_trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
    )
    run_quick_compare(
        config_path=args.config,
        champion_metrics_path=args.champion_metrics,
        output_dir=args.output_dir,
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        min_price=args.min_price,
        challenger=challenger,
        top_n=args.top_n,
        force_panel=args.force_panel,
    )


if __name__ == "__main__":
    main()
