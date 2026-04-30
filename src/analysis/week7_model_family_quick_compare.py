from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR

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
class QuickConfig:
    model_name: str
    universe_name: str
    n_jobs: int
    use_regressor: bool
    svm_c: float = 0.5
    svm_epsilon: float = 0.01
    svm_max_iter: int = 3000
    knn_neighbors: int = 64


def run_quick_compare(
    *,
    config_path: Path,
    champion_metrics_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    min_price: float,
    challenger: QuickConfig,
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

    challenger_metrics = _evaluate_metrics(
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
            "name": challenger.model_name,
            "n_jobs": challenger.n_jobs,
            "use_regressor": challenger.use_regressor,
            "svm_c": challenger.svm_c,
            "svm_epsilon": challenger.svm_epsilon,
            "svm_max_iter": challenger.svm_max_iter,
            "knn_neighbors": challenger.knn_neighbors,
        },
        "comparison": comparison,
        "notes": [
            "The SVM challenger uses linear SVM models because kernel SVM is not computationally practical on this full-size tabular panel.",
            "The k-NN challenger uses standardized features with distance weighting.",
        ],
    }
    write_json(summary, summary_path)

    write_dataframe_manifest(
        challenger_path,
        challenger_df,
        generated_by="src/analysis/week7_model_family_quick_compare.py",
        as_of_date=evaluation_end_date,
        extra_notes=[f"Challenger is {challenger.model_name} on the same 7-day target and yearly walk-forward folds."],
    )
    write_dataframe_manifest(
        comparison_path,
        comparison_df,
        generated_by="src/analysis/week7_model_family_quick_compare.py",
        as_of_date=evaluation_end_date,
        extra_notes=[f"Positive delta columns mean {challenger.model_name} improved on the champion metric."],
    )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_model_family_quick_compare.py",
        as_of_date=evaluation_end_date,
    )
    write_report_directory_readme(
        output_dir,
        title="Week 7 Model Family Quick Compare",
        intro_lines=[
            "This folder compares one challenger model family against the existing 7-day champion for one universe.",
            "The comparison uses the same full-history walk-forward setup and computes basket metrics directly to stay reliable on this machine.",
            "Open `summary.json` first, then `comparison_vs_champion.csv`.",
        ],
        files=[summary_path, comparison_path, challenger_path],
    )
    return summary


def _evaluate_metrics(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_end_date: str,
    min_train_rows: int,
    min_test_rows: int,
    top_n: int,
    challenger: QuickConfig,
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

        if challenger.model_name == "svm":
            prob_5 = _svm_score_classifier(train_x, (train_return >= 0.05).astype(np.int8), test_x, challenger)
            prob_10 = _svm_score_classifier(train_x, (train_return >= 0.10).astype(np.int8), test_x, challenger)
            if challenger.use_regressor:
                pred_return = _svm_predict_regressor(train_x, train_return, test_x, challenger)
            else:
                pred_return = np.zeros(len(test_x), dtype=np.float32)
        elif challenger.model_name == "knn":
            prob_5 = _knn_score_classifier(train_x, (train_return >= 0.05).astype(np.int8), test_x, challenger)
            prob_10 = _knn_score_classifier(train_x, (train_return >= 0.10).astype(np.int8), test_x, challenger)
            if challenger.use_regressor:
                pred_return = _knn_predict_regressor(train_x, train_return, test_x, challenger)
            else:
                pred_return = np.zeros(len(test_x), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported model family: {challenger.model_name}")

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
    precision = float(selected_hits / selected_rows) if selected_rows else np.nan
    recall = float(selected_hits / winner_count) if winner_count else np.nan
    ci_low, ci_high = _wilson_interval(selected_hits, selected_rows)
    p_value = float(binomtest(selected_hits, selected_rows, p=base_rate, alternative="greater").pvalue) if selected_rows else np.nan
    return {
        "universe_name": challenger.universe_name,
        "row_count": total_rows,
        "base_rate_5pct": base_rate,
        "winner_count": winner_count,
        "top10_precision_5pct": precision,
        "top10_recall": recall,
        "top10_ci_low": ci_low,
        "top10_ci_high": ci_high,
        "top10_p_value": p_value,
        "top10_mean_return_mean": _safe_mean(daily_mean_returns),
        "top10_median_stock_return_median": _safe_median(daily_median_returns),
        "top10_p75_stock_return_median": _safe_median(daily_p75_returns),
        "precision_lift_top10": float(precision / base_rate) if base_rate and not np.isnan(base_rate) else np.nan,
        "model_name": challenger.model_name,
    }


def _svm_score_classifier(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, challenger: QuickConfig) -> np.ndarray:
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(train_y.mean()), dtype=np.float32)
    model = LinearSVC(C=challenger.svm_c, class_weight="balanced", max_iter=challenger.svm_max_iter, dual=False, random_state=42)
    model.fit(train_x, train_y)
    train_scores = model.decision_function(train_x)
    score_scale = float(np.nanstd(train_scores)) if len(train_scores) else 1.0
    if not np.isfinite(score_scale) or score_scale <= 1e-8:
        score_scale = 1.0
    test_scores = model.decision_function(test_x) / score_scale
    test_scores = np.clip(test_scores, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-test_scores))).astype(np.float32)


def _svm_predict_regressor(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, challenger: QuickConfig) -> np.ndarray:
    if len(train_y) == 0 or float(np.nanstd(train_y)) < 1e-8:
        constant = float(np.nanmean(train_y)) if len(train_y) else 0.0
        return np.full(len(test_x), constant, dtype=np.float32)
    model = LinearSVR(C=challenger.svm_c, epsilon=challenger.svm_epsilon, max_iter=challenger.svm_max_iter, random_state=42)
    model.fit(train_x, train_y)
    return np.clip(model.predict(test_x), -0.20, 0.25).astype(np.float32)


def _knn_score_classifier(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, challenger: QuickConfig) -> np.ndarray:
    if len(np.unique(train_y)) < 2:
        return np.full(len(test_x), float(train_y.mean()), dtype=np.float32)
    model = KNeighborsClassifier(
        n_neighbors=challenger.knn_neighbors,
        weights="distance",
        algorithm="auto",
        n_jobs=challenger.n_jobs,
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1].astype(np.float32)


def _knn_predict_regressor(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, challenger: QuickConfig) -> np.ndarray:
    if len(train_y) == 0 or float(np.nanstd(train_y)) < 1e-8:
        constant = float(np.nanmean(train_y)) if len(train_y) else 0.0
        return np.full(len(test_x), constant, dtype=np.float32)
    model = KNeighborsRegressor(
        n_neighbors=challenger.knn_neighbors,
        weights="distance",
        algorithm="auto",
        n_jobs=challenger.n_jobs,
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
        "top10_precision_challenger": challenger["top10_precision_5pct"],
        "delta_top10_precision": float(challenger["top10_precision_5pct"]) - float(champion["precision_5pct"]),
        "precision_lift_champion": champion["precision_lift"],
        "precision_lift_challenger": challenger["precision_lift_top10"],
        "delta_precision_lift": float(challenger["precision_lift_top10"]) - float(champion["precision_lift"]),
        "top10_recall_champion": champion["recall"],
        "top10_recall_challenger": challenger["top10_recall"],
        "delta_top10_recall": float(challenger["top10_recall"]) - float(champion["recall"]),
        "top10_mean_return_champion": champion["mean_return_mean"],
        "top10_mean_return_challenger": challenger["top10_mean_return_mean"],
        "delta_top10_mean_return": float(challenger["top10_mean_return_mean"]) - float(champion["mean_return_mean"]),
        "top10_median_return_champion": champion["median_stock_return_median"],
        "top10_median_return_challenger": challenger["top10_median_stock_return_median"],
        "delta_top10_median_return": float(challenger["top10_median_stock_return_median"]) - float(champion["median_stock_return_median"]),
        "top10_p75_return_champion": champion["p75_stock_return_median"],
        "top10_p75_return_challenger": challenger["top10_p75_stock_return_median"],
        "delta_top10_p75_return": float(challenger["top10_p75_stock_return_median"]) - float(champion["p75_stock_return_median"]),
        "winner_metric": challenger["model_name"] if float(challenger["top10_precision_5pct"]) > float(champion["precision_5pct"]) else "champion",
    }


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else np.nan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick SVM or k-NN versus champion compare for week-7 5 percent target.")
    parser.add_argument("--config", type=Path, default=Path("configs/ml_research.yaml"))
    parser.add_argument("--champion-metrics", type=Path, default=Path("reports/week7_topn_backtest_20260414/daily_topn_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2025-12-31")
    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--model", choices=["svm", "knn"], required=True)
    parser.add_argument("--universe", default="mid_small")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--no-regressor", action="store_true", help="Skip the challenger regressor leg and rank on classification outputs only.")
    parser.add_argument("--svm-c", type=float, default=0.5)
    parser.add_argument("--svm-epsilon", type=float, default=0.01)
    parser.add_argument("--svm-max-iter", type=int, default=3000)
    parser.add_argument("--knn-neighbors", type=int, default=64)
    parser.add_argument("--force-panel", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    challenger = QuickConfig(
        model_name=args.model,
        universe_name=args.universe,
        n_jobs=args.n_jobs,
        use_regressor=not args.no_regressor,
        svm_c=args.svm_c,
        svm_epsilon=args.svm_epsilon,
        svm_max_iter=args.svm_max_iter,
        knn_neighbors=args.knn_neighbors,
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
