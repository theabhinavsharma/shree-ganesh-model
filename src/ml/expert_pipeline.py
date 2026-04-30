from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge

from src.ml.config import ObjectiveSpec
from src.ml.config import ResearchConfig
from src.ml.config import load_research_config
from src.ml.feature_registry import available_feature_columns
from src.ml.metrics import daily_top_n_metrics
from src.ml.metrics import daily_top_quantile_metrics
from src.ml.panel import build_current_feature_slice
from src.ml.panel import prepare_feature_panel
from src.ml.preprocess import PreprocessStats
from src.ml.preprocess import fit_preprocess
from src.ml.preprocess import transform_frame
from src.ml.universes import build_universe_masks
from src.ml.walk_forward import build_yearly_walk_forward_folds
from src.utils.io import write_json
from src.utils.io import write_parquet


@dataclass(frozen=True)
class ExpertHorizonSpec:
    name: str
    horizon_days: int
    analysis_start_date: str
    analysis_end_date: str
    min_price: float = 20.0


@dataclass(frozen=True)
class ExpertConfig:
    base_config_path: Path
    base_config: ResearchConfig
    horizons: list[ExpertHorizonSpec]
    focus_horizon: str
    shortlist_size: int
    calibration_bins: int
    run_output_dir: Path


@dataclass(frozen=True)
class ClassifierBundle:
    constant_probability: float | None
    linear_model: object | None
    tree_model: object | None


@dataclass(frozen=True)
class RegressorBundle:
    constant_value: float | None
    linear_model: object | None
    tree_model: object | None


TREE_ROW_LIMIT = 850_000
FOCUS_OOF_CONTEXT_COLUMNS = [
    "avg_traded_value_20d_cr",
    "volume_vs_20d",
    "rsi_14_daily",
    "rsi_14_weekly",
    "macro_risk_on_flag",
    "macro_vix_below_20",
    "breadth_above_50_dma",
    "breadth_above_200_dma",
    "breadth_volume_1_5x",
    "market_median_return_20d",
    "nifty_50_return_20d",
    "nifty_500_return_20d",
]


def load_expert_config(path: Path) -> ExpertConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    base_config_path = Path(raw["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = (path.parent.parent / base_config_path).resolve()
    base_config = load_research_config(base_config_path)
    horizons = [ExpertHorizonSpec(**item) for item in raw.get("horizons", [])]
    settings = raw.get("settings", {})
    run_output_dir = Path(settings.get("run_output_dir", "data/ml/expert_runs"))
    if not run_output_dir.is_absolute():
        run_output_dir = (path.parent.parent / run_output_dir).resolve()
    return ExpertConfig(
        base_config_path=base_config_path,
        base_config=base_config,
        horizons=horizons,
        focus_horizon=str(settings.get("focus_horizon", "week_7")),
        shortlist_size=int(settings.get("shortlist_size", 10)),
        calibration_bins=int(settings.get("calibration_bins", 10)),
        run_output_dir=run_output_dir,
    )


def run_expert_pipeline(config: ExpertConfig, *, force_panel: bool = False) -> dict[str, object]:
    run_root = config.run_output_dir / pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=True)

    focus_spec = _require_horizon(config, config.focus_horizon)
    focus_objective = _to_objective(focus_spec)
    focus_panel, focus_panel_path = prepare_feature_panel(config.base_config, focus_objective, force=force_panel)
    focus_feature_columns = available_feature_columns(list(focus_panel.columns), config.base_config.feature_columns)
    focus_predictions, focus_summaries = load_or_evaluate_focus_horizon(
        focus_panel,
        feature_columns=focus_feature_columns,
        config=config,
        horizon_spec=focus_spec,
        panel_path=focus_panel_path,
        force=force_panel,
    )
    focus_summary_df = pd.DataFrame(focus_summaries).sort_values(
        ["sort_primary", "sort_secondary", "sort_tertiary"],
        ascending=[False, False, False],
    )
    focus_summary_df.to_csv(run_root / "week_7_universe_summary.csv", index=False)
    write_parquet(focus_predictions, run_root / "week_7_oof_predictions.parquet")

    if focus_summary_df.empty:
        write_json(
            {
                "status": "blocked",
                "reason": "No valid 7-day walk-forward universe results.",
                "panel_path": str(focus_panel_path),
            },
            run_root / "run_manifest.json",
        )
        return {"run_root": str(run_root), "status": "blocked"}

    best_universe = str(focus_summary_df.iloc[0]["universe_name"])
    best_focus_predictions = focus_predictions.loc[focus_predictions["universe_name"] == best_universe].copy()
    calibration = _build_calibration_table(
        best_focus_predictions,
        score_col="focus_score",
        target_col="winner_5pct",
        return_col="forward_return",
        bins=config.calibration_bins,
    )
    calibration.to_csv(run_root / "week_7_calibration.csv", index=False)

    final_focus = _fit_focus_models(
        focus_panel.loc[build_universe_masks(focus_panel)[best_universe].fillna(False).astype(bool)].copy(),
        feature_columns=focus_feature_columns,
    )
    current_slice = build_current_feature_slice(config.base_config)
    current_focus = _score_focus_current(
        current_slice,
        feature_columns=focus_feature_columns,
        universe_name=best_universe,
        bundle=final_focus,
        calibration=calibration,
    )

    horizon_current_frames: list[pd.DataFrame] = [current_focus]
    horizon_run_details: list[dict[str, object]] = []
    for horizon_spec in config.horizons:
        objective = _to_objective(horizon_spec)
        panel, panel_path = prepare_feature_panel(config.base_config, objective, force=force_panel)
        feature_columns = available_feature_columns(list(panel.columns), config.base_config.feature_columns)
        scoped = panel.loc[build_universe_masks(panel)[best_universe].fillna(False).astype(bool)].copy()
        if scoped.empty:
            continue
        aux_bundle = _fit_aux_models(scoped, feature_columns=feature_columns)
        current_aux = _score_aux_current(
            current_slice,
            feature_columns=feature_columns,
            universe_name=best_universe,
            bundle=aux_bundle,
            horizon_name=horizon_spec.name,
        )
        horizon_current_frames.append(current_aux)
        horizon_run_details.append(
            {
                "horizon_name": horizon_spec.name,
                "horizon_days": horizon_spec.horizon_days,
                "panel_path": str(panel_path),
                "training_rows": int(len(scoped)),
            }
        )

    merged_current = _merge_current_frames(horizon_current_frames)
    merged_current = _finalize_shortlist(merged_current, shortlist_size=config.shortlist_size)
    merged_current.to_csv(run_root / "current_all_scores.csv", index=False)
    current_shortlist = merged_current.head(config.shortlist_size).copy()
    current_shortlist.to_csv(run_root / "current_shortlist_top10.csv", index=False)

    write_json(
        {
            "status": "ok",
            "run_root": str(run_root),
            "focus_horizon": focus_spec.__dict__,
            "selected_universe": best_universe,
            "focus_panel_path": str(focus_panel_path),
            "focus_feature_count": len(focus_feature_columns),
            "calibration_bins": config.calibration_bins,
            "shortlist_size": config.shortlist_size,
            "auxiliary_horizons": horizon_run_details,
            "selected_summary": _jsonify(focus_summary_df.iloc[0].to_dict()),
        },
        run_root / "run_manifest.json",
    )
    return {
        "run_root": str(run_root),
        "selected_universe": best_universe,
        "shortlist_path": str(run_root / "current_shortlist_top10.csv"),
        "current_all_scores": merged_current,
        "current_shortlist": current_shortlist,
        "focus_oof_predictions": focus_predictions,
        "focus_summary": focus_summary_df,
    }


def load_or_evaluate_focus_horizon(
    panel: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: ExpertConfig,
    horizon_spec: ExpertHorizonSpec,
    panel_path: Path,
    force: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    cache_paths = _focus_eval_cache_paths(panel_path, config=config, horizon_spec=horizon_spec)
    if not force and cache_paths["predictions"].exists() and cache_paths["summaries"].exists():
        predictions = pd.read_parquet(cache_paths["predictions"])
        summaries = pd.read_json(cache_paths["summaries"], orient="records")
        return predictions, summaries.to_dict(orient="records")

    predictions, summaries = _evaluate_focus_horizon(
        panel,
        feature_columns=feature_columns,
        config=config,
        horizon_spec=horizon_spec,
    )
    cache_paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    write_parquet(predictions, cache_paths["predictions"])
    pd.DataFrame(summaries).to_json(cache_paths["summaries"], orient="records", indent=2)
    return predictions, summaries


def _evaluate_focus_horizon(
    panel: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: ExpertConfig,
    horizon_spec: ExpertHorizonSpec,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    predictions: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    universe_masks = build_universe_masks(panel)
    min_train_end_year = pd.Timestamp(config.base_config.train_end_date).year
    folds = build_yearly_walk_forward_folds(panel, min_train_end_year=min_train_end_year)

    for universe_name in config.base_config.universes:
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
            if len(train) < config.base_config.min_train_rows or len(test) < config.base_config.min_test_rows:
                continue
            stats = fit_preprocess(train, feature_columns)
            train_x = transform_frame(train, stats)
            test_x = transform_frame(test, stats)

            train_return = pd.to_numeric(train["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            test_return = pd.to_numeric(test["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            train_up = (train_return > 0.0).astype(np.float32)
            train_5 = (train_return >= 0.05).astype(np.float32)
            train_10 = (train_return >= 0.10).astype(np.float32)

            up_bundle = _fit_classifier_bundle(train_x, train_up)
            five_bundle = _fit_classifier_bundle(train_x, train_5)
            ten_bundle = _fit_classifier_bundle(train_x, train_10)
            reg_bundle = _fit_regressor_bundle(train_x, train_return)

            test_prob_up = _predict_classifier_bundle(up_bundle, test_x)
            test_prob_5 = _predict_classifier_bundle(five_bundle, test_x)
            test_prob_10 = _predict_classifier_bundle(ten_bundle, test_x)
            test_pred_return = _predict_regressor_bundle(reg_bundle, test_x)

            base_columns = ["trade_date", "symbol", "close", "forward_return"]
            context_columns = [column for column in FOCUS_OOF_CONTEXT_COLUMNS if column in test.columns]
            fold_frame = test[base_columns + context_columns].copy()
            fold_frame["fold_name"] = fold.fold_name
            fold_frame["universe_name"] = universe_name
            fold_frame["horizon_name"] = horizon_spec.name
            fold_frame["winner_up"] = test_return > 0.0
            fold_frame["winner_5pct"] = test_return >= 0.05
            fold_frame["winner_10pct"] = test_return >= 0.10
            fold_frame["prob_up"] = test_prob_up
            fold_frame["prob_5pct"] = test_prob_5
            fold_frame["prob_10pct"] = test_prob_10
            fold_frame["pred_return"] = test_pred_return
            fold_frame["focus_score"] = _combine_focus_score(test_prob_5, test_prob_10, test_pred_return)
            fold_predictions.append(fold_frame)

        if not fold_predictions:
            continue

        universe_predictions = pd.concat(fold_predictions, ignore_index=True)
        predictions.append(universe_predictions)
        top_quantile = daily_top_quantile_metrics(
            universe_predictions,
            score_col="focus_score",
            winner_col="winner_5pct",
            return_col="forward_return",
            top_quantile=config.base_config.top_quantile,
        )
        top_n = daily_top_n_metrics(
            universe_predictions,
            score_col="focus_score",
            winner_col="winner_5pct",
            return_col="forward_return",
            top_n=config.base_config.top_n_daily,
        )
        top_selected = _select_daily_top_quantile(
            universe_predictions,
            score_col="focus_score",
            top_quantile=config.base_config.top_quantile,
        )
        total_selected = int(len(top_selected))
        hit_count = int(pd.to_numeric(top_selected["winner_5pct"], errors="coerce").fillna(0).sum())
        base_rate = float(pd.to_numeric(universe_predictions["winner_5pct"], errors="coerce").fillna(0).mean())
        top_precision = float(hit_count / total_selected) if total_selected else None
        ci_low, ci_high = _wilson_interval(hit_count, total_selected)
        p_value = float(binomtest(hit_count, total_selected, p=base_rate, alternative="greater").pvalue) if total_selected else None
        summaries.append(
            {
                "horizon_name": horizon_spec.name,
                "universe_name": universe_name,
                "row_count": int(len(universe_predictions)),
                "base_rate_5pct": base_rate,
                "top_quantile_precision_5pct": top_quantile["mean_top_quantile_precision"],
                "top_quantile_return": top_quantile["mean_top_quantile_return"],
                "top_n_precision_5pct": top_n["mean_top_n_precision"],
                "top_n_return": top_n["mean_top_n_return"],
                "selected_row_count": total_selected,
                "selected_hit_count": hit_count,
                "selected_precision_pooled": top_precision,
                "selected_precision_ci_low": ci_low,
                "selected_precision_ci_high": ci_high,
                "selected_precision_p_value": p_value,
                "sort_primary": float(top_quantile["mean_top_quantile_precision"] or -1.0),
                "sort_secondary": float(top_quantile["mean_top_quantile_return"] or -1.0),
                "sort_tertiary": float(-(p_value or 1.0)),
            }
        )

    if not predictions:
        return pd.DataFrame(), []
    return pd.concat(predictions, ignore_index=True), summaries


def _fit_focus_models(frame: pd.DataFrame, *, feature_columns: list[str]) -> dict[str, object]:
    stats = fit_preprocess(frame, feature_columns)
    x = transform_frame(frame, stats)
    forward_return = pd.to_numeric(frame["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return {
        "stats": stats,
        "up": _fit_classifier_bundle(x, (forward_return > 0.0).astype(np.float32)),
        "five": _fit_classifier_bundle(x, (forward_return >= 0.05).astype(np.float32)),
        "ten": _fit_classifier_bundle(x, (forward_return >= 0.10).astype(np.float32)),
        "reg": _fit_regressor_bundle(x, forward_return),
    }


def _fit_aux_models(frame: pd.DataFrame, *, feature_columns: list[str]) -> dict[str, object]:
    stats = fit_preprocess(frame, feature_columns)
    x = transform_frame(frame, stats)
    forward_return = pd.to_numeric(frame["forward_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return {
        "stats": stats,
        "up": _fit_classifier_bundle(x, (forward_return > 0.0).astype(np.float32)),
        "reg": _fit_regressor_bundle(x, forward_return),
    }


def _score_focus_current(
    current: pd.DataFrame,
    *,
    feature_columns: list[str],
    universe_name: str,
    bundle: dict[str, object],
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    universe_masks = build_universe_masks(current)
    scoped = current.loc[universe_masks[universe_name].fillna(False).astype(bool)].copy()
    stats: PreprocessStats = bundle["stats"]
    x = transform_frame(scoped, stats)
    prob_up = _predict_classifier_bundle(bundle["up"], x)
    prob_5 = _predict_classifier_bundle(bundle["five"], x)
    prob_10 = _predict_classifier_bundle(bundle["ten"], x)
    pred_return = _predict_regressor_bundle(bundle["reg"], x)
    scoped["prob_up_7d"] = prob_up
    scoped["prob_5pct_7d"] = prob_5
    scoped["prob_10pct_7d"] = prob_10
    scoped["pred_return_7d"] = pred_return
    scoped["pred_price_7d"] = pd.to_numeric(scoped["close"], errors="coerce") * (1.0 + scoped["pred_return_7d"])
    scoped["focus_score"] = _combine_focus_score(prob_5, prob_10, pred_return)
    scoped = _apply_calibration(scoped, calibration, score_col="focus_score")
    keep_columns = [
        "trade_date",
        "symbol",
        "company_name",
        "sector",
        "industry",
        "basic_industry",
        "close",
        "market_cap_cr",
        "avg_traded_value_20d_cr",
        "volume_vs_20d",
        "traded_value_vs_20d",
        "return_20d",
        "rsi_14_daily",
        "rsi_14_weekly",
        "rsi_14_monthly",
        "above_50_dma_flag",
        "above_200_dma_flag",
        "volume_3m_high_flag",
        "delivery_pct_3m_high_flag",
        "breadth_above_50_dma",
        "breadth_above_200_dma",
        "breadth_volume_1_5x",
        "market_median_return_20d",
        "nifty_50_return_20d",
        "nifty_500_return_20d",
        "macro_risk_on_flag",
        "macro_vix_below_20",
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_promoter_buy_flag",
        "recent_pledge_change_flag",
        "revenue_yoy",
        "pat_yoy",
        "eps_yoy",
        "revenue_yoy_acceleration",
        "pat_yoy_acceleration",
        "insider_net_value_30d",
        "promoter_director_net_value_90d",
        "recent_insider_buy_flag",
        "recent_promoter_or_director_buy_flag",
        "bulk_net_value_30d",
        "block_net_value_30d",
        "recent_bulk_buy_flag",
        "recent_block_buy_flag",
        "oi_share_of_mwpl",
        "oi_change_1d",
        "oi_change_pct_1d",
        "oi_share_of_mwpl_change_1d",
        "promoter_pct",
        "promoter_pct_qoq_change",
        "pe_ttm",
        "revenue_cagr_5y",
        "pat_cagr_5y",
        "debt_to_equity",
        "ebitda_positive_last_5q_flag",
        "prob_up_7d",
        "prob_5pct_7d",
        "prob_10pct_7d",
        "pred_return_7d",
        "pred_price_7d",
        "focus_score",
        "calibrated_confidence_5pct_7d",
        "calibrated_avg_return_7d",
        "calibration_bin",
    ]
    keep_columns = [column for column in keep_columns if column in scoped.columns]
    return scoped[keep_columns].copy()


def _score_aux_current(
    current: pd.DataFrame,
    *,
    feature_columns: list[str],
    universe_name: str,
    bundle: dict[str, object],
    horizon_name: str,
) -> pd.DataFrame:
    universe_masks = build_universe_masks(current)
    scoped = current.loc[universe_masks[universe_name].fillna(False).astype(bool)].copy()
    stats: PreprocessStats = bundle["stats"]
    x = transform_frame(scoped, stats)
    prob_up = _predict_classifier_bundle(bundle["up"], x)
    pred_return = _predict_regressor_bundle(bundle["reg"], x)
    suffix = horizon_name.replace("-", "_")
    scoped[f"prob_up_{suffix}"] = prob_up
    scoped[f"pred_return_{suffix}"] = pred_return
    scoped[f"pred_price_{suffix}"] = pd.to_numeric(scoped["close"], errors="coerce") * (1.0 + pred_return)
    keep_columns = [
        "trade_date",
        "symbol",
        f"prob_up_{suffix}",
        f"pred_return_{suffix}",
        f"pred_price_{suffix}",
    ]
    return scoped[keep_columns].copy()


def _merge_current_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["trade_date", "symbol"], how="left")
    return merged


def _finalize_shortlist(frame: pd.DataFrame, *, shortlist_size: int) -> pd.DataFrame:
    working = frame.copy()
    prob_up_day_1 = _safe_numeric_column(working, "prob_up_day_1")
    prob_up_day_15 = _safe_numeric_column(working, "prob_up_day_15")
    working["target_zone_7d_flag"] = (
        pd.to_numeric(working.get("pred_return_7d"), errors="coerce").between(0.05, 0.10, inclusive="both")
    )
    working["ranking_score"] = (
        pd.to_numeric(working.get("calibrated_confidence_5pct_7d"), errors="coerce").fillna(0.0) * 0.55
        + pd.to_numeric(working.get("prob_10pct_7d"), errors="coerce").fillna(0.0) * 0.20
        + pd.to_numeric(working.get("pred_return_7d"), errors="coerce").fillna(0.0).clip(-0.20, 0.20) * 0.75
        + prob_up_day_1 * 0.05
        + prob_up_day_15 * 0.05
    )
    working = working.sort_values(
        ["target_zone_7d_flag", "ranking_score", "calibrated_confidence_5pct_7d", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    working["shortlist_rank"] = np.arange(1, len(working) + 1)
    return working.head(max(shortlist_size, 10)).copy()


def _safe_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _fit_classifier_bundle(x: np.ndarray, y: np.ndarray) -> ClassifierBundle:
    if len(np.unique(y)) < 2:
        return ClassifierBundle(constant_probability=float(y.mean()), linear_model=None, tree_model=None)
    linear = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0005,
        class_weight="balanced",
        max_iter=40,
        tol=1e-3,
        random_state=42,
    )
    linear.fit(x, y)
    tree = None
    if len(x) <= TREE_ROW_LIMIT:
        tree = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=80,
            min_samples_leaf=300,
            l2_regularization=1.0,
            max_leaf_nodes=31,
            early_stopping=False,
            random_state=42,
        )
        tree.fit(x, y)
    return ClassifierBundle(constant_probability=None, linear_model=linear, tree_model=tree)


def _predict_classifier_bundle(bundle: ClassifierBundle, x: np.ndarray) -> np.ndarray:
    if bundle.constant_probability is not None:
        return np.full(len(x), bundle.constant_probability, dtype=np.float32)
    linear_prob = bundle.linear_model.predict_proba(x)[:, 1]
    if bundle.tree_model is None:
        return linear_prob.astype(np.float32)
    tree_prob = bundle.tree_model.predict_proba(x)[:, 1]
    return ((linear_prob + tree_prob) / 2.0).astype(np.float32)


def _fit_regressor_bundle(x: np.ndarray, y: np.ndarray) -> RegressorBundle:
    if len(y) == 0 or float(np.nanstd(y)) < 1e-8:
        return RegressorBundle(constant_value=float(np.nanmean(y)) if len(y) else 0.0, linear_model=None, tree_model=None)
    linear = Ridge(alpha=8.0)
    linear.fit(x, y)
    tree = None
    if len(x) <= TREE_ROW_LIMIT:
        tree = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=3,
            max_iter=80,
            min_samples_leaf=300,
            l2_regularization=1.0,
            max_leaf_nodes=31,
            early_stopping=False,
            random_state=42,
        )
        tree.fit(x, y)
    return RegressorBundle(constant_value=None, linear_model=linear, tree_model=tree)


def _predict_regressor_bundle(bundle: RegressorBundle, x: np.ndarray) -> np.ndarray:
    if bundle.constant_value is not None:
        return np.full(len(x), bundle.constant_value, dtype=np.float32)
    linear_pred = bundle.linear_model.predict(x)
    if bundle.tree_model is None:
        return np.clip(linear_pred, -0.15, 0.25).astype(np.float32)
    tree_pred = bundle.tree_model.predict(x)
    return np.clip((linear_pred + tree_pred) / 2.0, -0.15, 0.25).astype(np.float32)


def _combine_focus_score(prob_5: np.ndarray, prob_10: np.ndarray, pred_return: np.ndarray) -> np.ndarray:
    return (
        0.55 * prob_5
        + 0.25 * prob_10
        + 0.20 * _squash_return(pred_return)
    ).astype(np.float32)


def _squash_return(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(values / 0.05)))


def _select_daily_top_quantile(frame: pd.DataFrame, *, score_col: str, top_quantile: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("trade_date", sort=False):
        ordered = group.sort_values(score_col, ascending=False)
        top_n = max(1, int(np.ceil(len(ordered) * top_quantile)))
        parts.append(ordered.head(top_n).copy())
    if not parts:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(parts, ignore_index=True)


def _build_calibration_table(
    frame: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
    return_col: str,
    bins: int,
) -> pd.DataFrame:
    working = frame[[score_col, target_col, return_col]].copy()
    working = working.dropna(subset=[score_col]).reset_index(drop=True)
    if working.empty:
        return pd.DataFrame(columns=["calibration_bin", "score_min", "score_max", "count", "hit_rate", "avg_return"])
    ranks = working[score_col].rank(method="first")
    bin_codes = pd.qcut(ranks, q=min(bins, len(working)), labels=False, duplicates="drop")
    working["calibration_bin"] = pd.to_numeric(bin_codes, errors="coerce").fillna(0).astype(int)
    grouped = (
        working.groupby("calibration_bin", dropna=False)
        .agg(
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
            count=(score_col, "size"),
            hit_rate=(target_col, "mean"),
            avg_return=(return_col, "mean"),
        )
        .reset_index()
        .sort_values("score_min")
        .reset_index(drop=True)
    )
    return grouped


def _apply_calibration(frame: pd.DataFrame, calibration: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    if calibration.empty:
        frame["calibrated_confidence_5pct_7d"] = pd.NA
        frame["calibrated_avg_return_7d"] = pd.NA
        frame["calibration_bin"] = pd.NA
        return frame
    calibration = calibration.sort_values("score_min").reset_index(drop=True)
    score_mins = calibration["score_min"].to_numpy(dtype=float)
    bins = calibration["calibration_bin"].to_numpy(dtype=int)
    positions = np.searchsorted(score_mins, pd.to_numeric(frame[score_col], errors="coerce").fillna(score_mins[0]).to_numpy(dtype=float), side="right") - 1
    positions = np.clip(positions, 0, len(calibration) - 1)
    frame["calibration_bin"] = bins[positions]
    by_bin = calibration.set_index("calibration_bin")
    frame["calibrated_confidence_5pct_7d"] = frame["calibration_bin"].map(by_bin["hit_rate"])
    frame["calibrated_avg_return_7d"] = frame["calibration_bin"].map(by_bin["avg_return"])
    return frame


def _wilson_interval(successes: int, total: int, *, alpha: float = 0.05) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = successes / total
    denominator = 1.0 + (z**2 / total)
    center = (phat + (z**2 / (2.0 * total))) / denominator
    margin = (z * np.sqrt((phat * (1.0 - phat) / total) + (z**2 / (4.0 * total**2)))) / denominator
    return float(center - margin), float(center + margin)


def _to_objective(spec: ExpertHorizonSpec) -> ObjectiveSpec:
    return ObjectiveSpec(
        name=spec.name,
        horizon_days=spec.horizon_days,
        target_return=0.0,
        analysis_start_date=spec.analysis_start_date,
        analysis_end_date=spec.analysis_end_date,
        min_price=spec.min_price,
    )


def _focus_eval_cache_paths(panel_path: Path, *, config: ExpertConfig, horizon_spec: ExpertHorizonSpec) -> dict[str, Path]:
    payload = {
        "focus_horizon": horizon_spec.name,
        "horizon_days": horizon_spec.horizon_days,
        "analysis_start_date": horizon_spec.analysis_start_date,
        "analysis_end_date": horizon_spec.analysis_end_date,
        "min_price": horizon_spec.min_price,
        "universes": config.base_config.universes,
        "train_end_date": config.base_config.train_end_date,
        "min_train_rows": config.base_config.min_train_rows,
        "min_test_rows": config.base_config.min_test_rows,
        "top_quantile": config.base_config.top_quantile,
        "top_n_daily": config.base_config.top_n_daily,
        "feature_columns": list(config.base_config.feature_columns),
        "model": {
            "learning_rate": config.base_config.model.learning_rate,
            "epochs": config.base_config.model.epochs,
            "l2": config.base_config.model.l2,
            "batch_size": config.base_config.model.batch_size,
            "seed": config.base_config.model.seed,
            "positive_class_weight": config.base_config.model.positive_class_weight,
        },
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_dir = panel_path.parent / "focus_eval_cache" / digest
    return {
        "predictions": cache_dir / "focus_oof_predictions.parquet",
        "summaries": cache_dir / "focus_universe_summaries.json",
    }


def _require_horizon(config: ExpertConfig, name: str) -> ExpertHorizonSpec:
    for horizon in config.horizons:
        if horizon.name == name:
            return horizon
    raise KeyError(f"Unknown focus horizon: {name}")


def _jsonify(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonify(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
