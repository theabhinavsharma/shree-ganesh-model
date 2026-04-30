from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.config import ObjectiveSpec
from src.ml.config import ResearchConfig
from src.ml.feature_registry import available_feature_columns
from src.ml.logistic import LogisticRegressionGD
from src.ml.metrics import brier_score
from src.ml.metrics import daily_top_n_metrics
from src.ml.metrics import daily_top_quantile_metrics
from src.ml.metrics import log_loss
from src.ml.metrics import roc_auc
from src.ml.panel import build_current_feature_slice
from src.ml.panel import prepare_feature_panel
from src.ml.preprocess import fit_preprocess
from src.ml.preprocess import transform_frame
from src.ml.universes import build_universe_masks
from src.ml.walk_forward import WalkForwardFold
from src.ml.walk_forward import build_yearly_walk_forward_folds
from src.utils.io import write_json


@dataclass(frozen=True)
class FoldResult:
    universe_name: str
    objective_name: str
    fold_name: str
    train_rows: int
    test_rows: int
    positive_rate_train: float | None
    positive_rate_test: float | None
    auc: float | None
    brier: float | None
    log_loss: float | None
    mean_top_quantile_precision: float | None
    mean_top_quantile_return: float | None
    mean_top_n_precision: float | None
    mean_top_n_return: float | None


def run_research_pipeline(config: ResearchConfig, *, force_panel: bool = False) -> dict[str, object]:
    run_root = config.paths.run_output_dir / pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=True)
    all_fold_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    selected_models: list[dict[str, object]] = []

    for objective in config.objectives:
        panel, panel_path = prepare_feature_panel(config, objective, force=force_panel)
        feature_columns = available_feature_columns(list(panel.columns), config.feature_columns)
        universe_masks = build_universe_masks(panel)
        folds = build_yearly_walk_forward_folds(panel)
        objective_dir = run_root / objective.name
        objective_dir.mkdir(parents=True, exist_ok=True)

        best_summary: dict[str, object] | None = None
        best_state: dict[str, object] | None = None

        for universe_name in config.universes:
            if universe_name not in universe_masks:
                continue
            scoped = panel.loc[universe_masks[universe_name].fillna(False).astype(bool)].copy()
            if scoped.empty:
                continue
            universe_fold_rows: list[dict[str, object]] = []
            for fold in folds:
                result = _evaluate_fold(
                    scoped,
                    feature_columns=feature_columns,
                    objective=objective,
                    universe_name=universe_name,
                    fold=fold,
                    config=config,
                )
                if result is None:
                    continue
                row = asdict(result)
                universe_fold_rows.append(row)
                all_fold_rows.append(row)
            if not universe_fold_rows:
                continue
            fold_df = pd.DataFrame(universe_fold_rows)
            summary_row = {
                "objective_name": objective.name,
                "universe_name": universe_name,
                "panel_path": str(panel_path),
                "feature_count": len(feature_columns),
                "fold_count": int(len(fold_df)),
                "mean_auc": _mean_or_none(fold_df["auc"]),
                "mean_brier": _mean_or_none(fold_df["brier"]),
                "mean_log_loss": _mean_or_none(fold_df["log_loss"]),
                "mean_top_quantile_precision": _mean_or_none(fold_df["mean_top_quantile_precision"]),
                "mean_top_quantile_return": _mean_or_none(fold_df["mean_top_quantile_return"]),
                "mean_top_n_precision": _mean_or_none(fold_df["mean_top_n_precision"]),
                "mean_top_n_return": _mean_or_none(fold_df["mean_top_n_return"]),
                "mean_test_rows": _mean_or_none(fold_df["test_rows"]),
            }
            all_summary_rows.append(summary_row)
            if best_summary is None or _summary_sort_key(summary_row) > _summary_sort_key(best_summary):
                best_summary = summary_row
                best_state = _fit_final_model(scoped, feature_columns=feature_columns, config=config)
        if best_summary is None or best_state is None:
            continue
        current_slice = build_current_feature_slice(config)
        current_scores = _score_current_rows(
            current_slice,
            feature_columns=feature_columns,
            universe_name=str(best_summary["universe_name"]),
            model_state=best_state,
        )
        current_scores.to_csv(objective_dir / "current_scores.csv", index=False)
        write_json(
            {
                "objective": objective.__dict__,
                "selected_universe": best_summary["universe_name"],
                "summary": best_summary,
                "model": best_state["model"].to_dict(),
                "preprocess": {
                    "feature_columns": best_state["stats"].feature_columns,
                    "numeric_columns": best_state["stats"].numeric_columns,
                    "boolean_columns": best_state["stats"].boolean_columns,
                    "medians": best_state["stats"].medians,
                    "means": best_state["stats"].means,
                    "stds": best_state["stats"].stds,
                },
            },
            objective_dir / "selected_model.json",
        )
        selected_models.append(
            {
                "objective_name": objective.name,
                "selected_universe": best_summary["universe_name"],
                "current_scores_path": str(objective_dir / "current_scores.csv"),
                **best_summary,
            }
        )

    fold_df = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(all_summary_rows)
    selected_df = pd.DataFrame(selected_models)
    fold_df.to_csv(run_root / "fold_metrics.csv", index=False)
    summary_df.to_csv(run_root / "universe_summary.csv", index=False)
    selected_df.to_csv(run_root / "selected_models.csv", index=False)
    write_json(
        {
            "run_root": str(run_root),
            "selected_models": selected_models,
            "objective_count": len(config.objectives),
            "universe_count": len(config.universes),
        },
        run_root / "run_manifest.json",
    )
    return {"run_root": str(run_root), "selected_models": selected_models}


def _evaluate_fold(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    objective: ObjectiveSpec,
    universe_name: str,
    fold: WalkForwardFold,
    config: ResearchConfig,
) -> FoldResult | None:
    train_mask = frame["trade_date"].le(fold.train_end_date)
    test_mask = frame["trade_date"].between(fold.test_start_date, fold.test_end_date)
    train = frame.loc[train_mask].copy()
    test = frame.loc[test_mask].copy()
    if len(train) < config.min_train_rows or len(test) < config.min_test_rows:
        return None
    train_y = pd.to_numeric(train["winner_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    test_y = pd.to_numeric(test["winner_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    stats = fit_preprocess(train, feature_columns)
    train_x = transform_frame(train, stats)
    test_x = transform_frame(test, stats)
    model = LogisticRegressionGD(
        learning_rate=config.model.learning_rate,
        epochs=config.model.epochs,
        l2=config.model.l2,
        batch_size=config.model.batch_size,
        seed=config.model.seed,
        positive_class_weight=config.model.positive_class_weight,
    ).fit(train_x, train_y)
    test_prob = model.predict_proba(test_x)[:, 1]
    scored = test[["trade_date", "symbol", "forward_return", "winner_flag"]].copy()
    scored["score"] = test_prob
    top_q = daily_top_quantile_metrics(
        scored,
        score_col="score",
        winner_col="winner_flag",
        return_col="forward_return",
        top_quantile=config.top_quantile,
    )
    top_n = daily_top_n_metrics(
        scored,
        score_col="score",
        winner_col="winner_flag",
        return_col="forward_return",
        top_n=config.top_n_daily,
    )
    return FoldResult(
        universe_name=universe_name,
        objective_name=objective.name,
        fold_name=fold.fold_name,
        train_rows=int(len(train)),
        test_rows=int(len(test)),
        positive_rate_train=float(train_y.mean()) if len(train_y) else None,
        positive_rate_test=float(test_y.mean()) if len(test_y) else None,
        auc=roc_auc(test_y, test_prob),
        brier=brier_score(test_y, test_prob),
        log_loss=log_loss(test_y, test_prob),
        mean_top_quantile_precision=top_q["mean_top_quantile_precision"],
        mean_top_quantile_return=top_q["mean_top_quantile_return"],
        mean_top_n_precision=top_n["mean_top_n_precision"],
        mean_top_n_return=top_n["mean_top_n_return"],
    )


def _fit_final_model(frame: pd.DataFrame, *, feature_columns: list[str], config: ResearchConfig) -> dict[str, object]:
    stats = fit_preprocess(frame, feature_columns)
    x = transform_frame(frame, stats)
    y = pd.to_numeric(frame["winner_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    model = LogisticRegressionGD(
        learning_rate=config.model.learning_rate,
        epochs=config.model.epochs,
        l2=config.model.l2,
        batch_size=config.model.batch_size,
        seed=config.model.seed,
        positive_class_weight=config.model.positive_class_weight,
    ).fit(x, y)
    return {"model": model, "stats": stats}


def _score_current_rows(panel: pd.DataFrame, *, feature_columns: list[str], universe_name: str, model_state: dict[str, object]) -> pd.DataFrame:
    universe_masks = build_universe_masks(panel)
    scoped = panel.loc[universe_masks[universe_name].fillna(False).astype(bool)].copy()
    latest_date = scoped["trade_date"].max()
    current = scoped.loc[scoped["trade_date"] == latest_date].copy()
    stats = model_state["stats"]
    model = model_state["model"]
    x = transform_frame(current, stats)
    current["score"] = model.predict_proba(x)[:, 1]
    return current.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)


def _summary_sort_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row.get("mean_top_quantile_precision") or -1.0),
        float(row.get("mean_top_n_precision") or -1.0),
        float(row.get("mean_top_quantile_return") or -1.0),
        float(row.get("mean_auc") or -1.0),
    )


def _mean_or_none(series: pd.Series) -> float | None:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())
