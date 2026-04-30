from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.ml.config import ObjectiveSpec
from src.ml.config import ResearchConfig
from src.ml.config import load_research_config
from src.ml.expert_pipeline import ExpertConfig
from src.ml.expert_pipeline import ExpertHorizonSpec
from src.ml.expert_pipeline import _build_calibration_table
from src.ml.expert_pipeline import _combine_focus_score
from src.ml.expert_pipeline import _fit_focus_models
from src.ml.expert_pipeline import _predict_classifier_bundle
from src.ml.expert_pipeline import _predict_regressor_bundle
from src.ml.expert_pipeline import _select_daily_top_quantile
from src.ml.expert_pipeline import _wilson_interval
from src.ml.expert_pipeline import load_or_evaluate_focus_horizon
from src.ml.feature_registry import available_feature_columns
from src.ml.panel import build_current_feature_slice
from src.ml.panel import prepare_feature_panel
from src.ml.preprocess import transform_frame
from src.ml.universes import build_universe_masks
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme
from src.utils.io import write_json
from src.utils.io import write_parquet


TRADABLE_UNIVERSES: tuple[str, ...] = (
    "mid_small",
    "liquid_5cr_plus",
    "liquid_20cr_plus",
    "mcap_1000cr_plus",
)


def run_day1_5pct_model(
    *,
    config_path: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    live_end_date: str,
    min_price: float,
    shortlist_size: int,
    calibration_bins: int,
    output_dir: Path,
    force_panel: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_research_config(config_path)
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
        run_output_dir=output_dir,
    )
    eval_objective = ObjectiveSpec(
        name="day_1_5pct_eval",
        horizon_days=1,
        target_return=0.0,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    eval_panel, eval_panel_path = prepare_feature_panel(base_config, eval_objective, force=force_panel)
    feature_columns = available_feature_columns(list(eval_panel.columns), base_config.feature_columns)
    predictions, raw_summaries = load_or_evaluate_focus_horizon(
        eval_panel,
        feature_columns=feature_columns,
        config=expert_config,
        horizon_spec=focus_horizon,
        panel_path=eval_panel_path,
        force=force_panel,
    )
    summary_df = pd.DataFrame(raw_summaries).sort_values(
        ["sort_primary", "sort_secondary", "sort_tertiary", "universe_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    universe_metrics = summarize_universes(
        predictions,
        summary_df,
        top_quantile=base_config.top_quantile,
        top_n=shortlist_size,
    )
    best_universe_any = select_best_universe(universe_metrics, tradable_only=False)
    best_universe_tradable = select_best_universe(universe_metrics, tradable_only=True)

    live_objective = replace(eval_objective, name="day_1_5pct_live", analysis_end_date=live_end_date)
    live_panel, live_panel_path = prepare_feature_panel(base_config, live_objective, force=force_panel)
    live_feature_columns = available_feature_columns(list(live_panel.columns), base_config.feature_columns)

    shortlist_any = score_current_shortlist(
        config=base_config,
        panel=live_panel,
        feature_columns=live_feature_columns,
        predictions=predictions,
        universe_name=best_universe_any,
        calibration_bins=calibration_bins,
        shortlist_size=shortlist_size,
    )
    shortlist_tradable = score_current_shortlist(
        config=base_config,
        panel=live_panel,
        feature_columns=live_feature_columns,
        predictions=predictions,
        universe_name=best_universe_tradable,
        calibration_bins=calibration_bins,
        shortlist_size=shortlist_size,
    )
    if best_universe_any == best_universe_tradable:
        shortlist_tradable = shortlist_any.copy()

    summary_path = output_dir / "summary.json"
    raw_summary_path = output_dir / "universe_summary_raw.csv"
    metrics_path = output_dir / "universe_metrics.csv"
    predictions_path = output_dir / "oof_predictions.parquet"
    shortlist_any_path = output_dir / "current_shortlist_best_any.csv"
    shortlist_tradable_path = output_dir / "current_shortlist_best_tradable.csv"

    summary_df.to_csv(raw_summary_path, index=False)
    universe_metrics.to_csv(metrics_path, index=False)
    write_parquet(predictions, predictions_path)
    shortlist_any.to_csv(shortlist_any_path, index=False)
    shortlist_tradable.to_csv(shortlist_tradable_path, index=False)

    as_of_trade_date = str(pd.to_datetime(shortlist_any["trade_date"]).max().date()) if not shortlist_any.empty else live_end_date
    summary = {
        "status": "ok",
        "objective": {
            "horizon_days": 1,
            "target_return": 0.05,
            "analysis_start_date": analysis_start_date,
            "evaluation_end_date": evaluation_end_date,
            "live_training_end_date": live_end_date,
            "min_price": min_price,
            "shortlist_size": shortlist_size,
        },
        "as_of_trade_date": as_of_trade_date,
        "evaluation_panel_path": str(eval_panel_path),
        "live_panel_path": str(live_panel_path),
        "feature_count": len(live_feature_columns),
        "best_universe_any": best_universe_any,
        "best_universe_tradable": best_universe_tradable,
        "best_universe_any_metrics": _extract_summary_metrics(universe_metrics, best_universe_any),
        "best_universe_tradable_metrics": _extract_summary_metrics(universe_metrics, best_universe_tradable),
        "notes": [
            "Raw model scores are ranking scores, not literal confidence percentages.",
            "calibrated_confidence_5pct_1d is the more honest single-number estimate because it comes from out-of-sample score-bucket behavior.",
            "This study uses full 2015 onward history, time-ordered yearly walk-forward folds, no random shuffle, and no sampling.",
        ],
    }
    write_json(summary, summary_path)

    write_dataframe_manifest(
        raw_summary_path,
        summary_df,
        generated_by="src/analysis/day1_5pct_model.py",
        as_of_date=as_of_trade_date,
    )
    write_dataframe_manifest(
        metrics_path,
        universe_metrics,
        generated_by="src/analysis/day1_5pct_model.py",
        as_of_date=as_of_trade_date,
        extra_notes=[
            "top10 metrics are the most relevant for the current live shortlist because the model ranks and acts on a top basket",
            "top bucket metrics come from calibration on out-of-sample predictions only",
        ],
    )
    write_dataframe_manifest(
        predictions_path,
        predictions.head(min(len(predictions), 25000)),
        generated_by="src/analysis/day1_5pct_model.py",
        as_of_date=as_of_trade_date,
        extra_notes=["Manifest profile is sampled to keep the sidecar compact; the parquet itself contains the full OOF table."],
    )
    write_dataframe_manifest(
        shortlist_any_path,
        shortlist_any,
        generated_by="src/analysis/day1_5pct_model.py",
        as_of_date=as_of_trade_date,
        extra_notes=["This is the highest-ranking universe across all universes, including less tradable pockets."],
    )
    write_dataframe_manifest(
        shortlist_tradable_path,
        shortlist_tradable,
        generated_by="src/analysis/day1_5pct_model.py",
        as_of_date=as_of_trade_date,
        extra_notes=["This shortlist is restricted to more tradable universes only."],
    )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/day1_5pct_model.py",
        as_of_date=as_of_trade_date,
    )
    write_report_directory_readme(
        output_dir,
        title="Day 1 Five Percent Model",
        intro_lines=[
            "This folder contains the dedicated walk-forward model study for 5 percent or more in the next 1 trading day.",
            "Open `summary.json` first, then `universe_metrics.csv`, then the current shortlist files.",
            "The best-any shortlist can drift into less tradable pockets. The best-tradable shortlist is the better live trading candidate.",
        ],
        files=[summary_path, raw_summary_path, metrics_path, shortlist_any_path, shortlist_tradable_path],
    )
    return summary


def summarize_universes(
    predictions: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    top_quantile: float,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for universe_name, universe_predictions in predictions.groupby("universe_name", sort=False):
        working = universe_predictions.sort_values(["trade_date", "focus_score", "symbol"], ascending=[True, False, True]).copy()
        base_rate = float(pd.to_numeric(working["winner_5pct"], errors="coerce").fillna(0).mean())
        winner_count = int(pd.to_numeric(working["winner_5pct"], errors="coerce").fillna(0).sum())
        calibration = _build_calibration_table(
            working,
            score_col="focus_score",
            target_col="winner_5pct",
            return_col="forward_return",
            bins=10,
        )
        top_bucket_hit_rate = float(calibration.sort_values("score_min").iloc[-1]["hit_rate"]) if not calibration.empty else np.nan
        top_bucket_avg_return = float(calibration.sort_values("score_min").iloc[-1]["avg_return"]) if not calibration.empty else np.nan

        top_quantile_rows = _select_daily_top_quantile(
            working,
            score_col="focus_score",
            top_quantile=top_quantile,
        )
        top_n_rows = _select_daily_top_n(working, top_n=top_n)

        top_quantile_hits = int(pd.to_numeric(top_quantile_rows["winner_5pct"], errors="coerce").fillna(0).sum())
        top_n_hits = int(pd.to_numeric(top_n_rows["winner_5pct"], errors="coerce").fillna(0).sum())
        top_quantile_precision = float(top_quantile_hits / len(top_quantile_rows)) if len(top_quantile_rows) else np.nan
        top_n_precision = float(top_n_hits / len(top_n_rows)) if len(top_n_rows) else np.nan
        top_quantile_recall = float(top_quantile_hits / winner_count) if winner_count else np.nan
        top_n_recall = float(top_n_hits / winner_count) if winner_count else np.nan

        top_quantile_ci_low, top_quantile_ci_high = _wilson_interval(top_quantile_hits, int(len(top_quantile_rows)))
        top_n_ci_low, top_n_ci_high = _wilson_interval(top_n_hits, int(len(top_n_rows)))
        top_quantile_p_value = float(binomtest(top_quantile_hits, int(len(top_quantile_rows)), p=base_rate, alternative="greater").pvalue) if len(top_quantile_rows) else np.nan
        top_n_p_value = float(binomtest(top_n_hits, int(len(top_n_rows)), p=base_rate, alternative="greater").pvalue) if len(top_n_rows) else np.nan

        top_quantile_daily = _summarize_daily_selection(top_quantile_rows)
        top_n_daily = _summarize_daily_selection(top_n_rows)
        base_returns = pd.to_numeric(working["forward_return"], errors="coerce")
        summary_row = summary_df.loc[summary_df["universe_name"] == universe_name].iloc[0].to_dict()
        rows.append(
            {
                "universe_name": universe_name,
                "row_count": int(len(working)),
                "unique_symbols": int(working["symbol"].nunique()),
                "base_rate_5pct": base_rate,
                "winner_count": winner_count,
                "top_bucket_hit_rate": top_bucket_hit_rate,
                "top_bucket_avg_return": top_bucket_avg_return,
                "top_quantile_precision_5pct": top_quantile_precision,
                "top_quantile_recall": top_quantile_recall,
                "top_quantile_ci_low": top_quantile_ci_low,
                "top_quantile_ci_high": top_quantile_ci_high,
                "top_quantile_p_value": top_quantile_p_value,
                "top_quantile_avg_return_mean": top_quantile_daily["mean_return_mean"],
                "top_quantile_median_stock_return_median": top_quantile_daily["median_stock_return_median"],
                "top_quantile_p75_stock_return_median": top_quantile_daily["p75_stock_return_median"],
                "top10_precision_5pct": top_n_precision,
                "top10_recall": top_n_recall,
                "top10_ci_low": top_n_ci_low,
                "top10_ci_high": top_n_ci_high,
                "top10_p_value": top_n_p_value,
                "top10_mean_return_mean": top_n_daily["mean_return_mean"],
                "top10_mean_return_median": top_n_daily["mean_return_median"],
                "top10_mean_return_p75": top_n_daily["mean_return_p75"],
                "top10_median_stock_return_mean": top_n_daily["median_stock_return_mean"],
                "top10_median_stock_return_median": top_n_daily["median_stock_return_median"],
                "top10_median_stock_return_p75": top_n_daily["median_stock_return_p75"],
                "top10_p75_stock_return_mean": top_n_daily["p75_stock_return_mean"],
                "top10_p75_stock_return_median": top_n_daily["p75_stock_return_median"],
                "top10_p75_stock_return_p75": top_n_daily["p75_stock_return_p75"],
                "base_avg_return": float(base_returns.mean()),
                "base_median_return": float(base_returns.median()),
                "base_p75_return": float(base_returns.quantile(0.75)),
                "raw_summary_top_quantile_precision": float(summary_row["top_quantile_precision_5pct"]),
                "raw_summary_top_n_precision": float(summary_row["top_n_precision_5pct"]),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["precision_lift_top10"] = metrics["top10_precision_5pct"] / metrics["base_rate_5pct"]
    metrics["precision_lift_top_quantile"] = metrics["top_quantile_precision_5pct"] / metrics["base_rate_5pct"]
    metrics["selection_rank_score"] = (
        metrics["top10_precision_5pct"].fillna(-1.0) * 0.50
        + metrics["top_bucket_hit_rate"].fillna(-1.0) * 0.25
        + metrics["top10_mean_return_mean"].fillna(-1.0) * 2.0
        + metrics["top10_median_stock_return_median"].fillna(-1.0) * 2.0
    )
    return metrics.sort_values(
        ["selection_rank_score", "top10_precision_5pct", "top_bucket_hit_rate", "universe_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def select_best_universe(universe_metrics: pd.DataFrame, *, tradable_only: bool) -> str:
    working = universe_metrics.copy()
    if tradable_only:
        working = working.loc[working["universe_name"].isin(TRADABLE_UNIVERSES)].copy()
    if working.empty:
        raise ValueError("No universes available for selection.")
    ordered = working.sort_values(
        [
            "selection_rank_score",
            "top10_precision_5pct",
            "top_bucket_hit_rate",
            "top10_mean_return_mean",
            "top10_median_stock_return_median",
            "universe_name",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    return str(ordered.iloc[0]["universe_name"])


def score_current_shortlist(
    *,
    config: ResearchConfig,
    panel: pd.DataFrame,
    feature_columns: list[str],
    predictions: pd.DataFrame,
    universe_name: str,
    calibration_bins: int,
    shortlist_size: int,
) -> pd.DataFrame:
    masks = build_universe_masks(panel)
    scoped_panel = panel.loc[masks[universe_name].fillna(False).astype(bool)].copy()
    if scoped_panel.empty:
        return pd.DataFrame()
    bundle = _fit_focus_models(scoped_panel, feature_columns=feature_columns)
    current_slice = build_current_feature_slice(config)
    calibration = _build_calibration_table(
        predictions.loc[predictions["universe_name"] == universe_name].copy(),
        score_col="focus_score",
        target_col="winner_5pct",
        return_col="forward_return",
        bins=calibration_bins,
    )
    scored = _score_current_focus(
        current_slice,
        feature_columns=feature_columns,
        universe_name=universe_name,
        bundle=bundle,
        calibration=calibration,
        output_suffix="1d",
    )
    scored["target_zone_1d_flag"] = pd.to_numeric(scored.get("pred_return_1d"), errors="coerce").between(0.05, 0.10, inclusive="both")
    scored["ranking_score"] = (
        pd.to_numeric(scored.get("calibrated_confidence_5pct_1d"), errors="coerce").fillna(0.0) * 0.60
        + pd.to_numeric(scored.get("prob_10pct_1d"), errors="coerce").fillna(0.0) * 0.30
        + _squash_return(pd.to_numeric(scored.get("pred_return_1d"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)) * 0.10
    )
    scored = _drop_etf_like_rows(scored)
    scored = scored.sort_values(
        ["target_zone_1d_flag", "ranking_score", "calibrated_confidence_5pct_1d", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    if len(scored):
        scored["confidence_score"] = (scored["ranking_score"] / float(scored["ranking_score"].max()) * 100.0).round(1)
    else:
        scored["confidence_score"] = pd.Series(dtype="float64")
    scored["target_5pct_price"] = (pd.to_numeric(scored["close"], errors="coerce") * 1.05).round(2)
    scored["rationale"] = scored.apply(build_rationale, axis=1)
    scored["selected_universe"] = universe_name
    scored["shortlist_rank"] = np.arange(1, len(scored) + 1)
    keep_columns = [
        "shortlist_rank",
        "selected_universe",
        "trade_date",
        "symbol",
        "company_name",
        "sector",
        "industry",
        "basic_industry",
        "close",
        "target_5pct_price",
        "confidence_score",
        "calibrated_confidence_5pct_1d",
        "calibrated_avg_return_1d",
        "prob_up_1d",
        "prob_5pct_1d",
        "prob_10pct_1d",
        "pred_return_1d",
        "pred_price_1d",
        "return_1d",
        "return_20d",
        "volume_vs_20d",
        "traded_value_vs_20d",
        "rsi_14_daily",
        "rsi_14_weekly",
        "rsi_14_monthly",
        "filter_above_50_dma",
        "filter_above_200_dma",
        "volume_high_63d_flag",
        "delivery_pct_high_63d_flag",
        "market_cap_cr",
        "avg_traded_value_20d_cr",
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_promoter_buy_flag",
        "recent_promoter_or_director_buy_flag",
        "recent_bulk_buy_flag",
        "promoter_pct",
        "pe_ttm",
        "revenue_yoy",
        "pat_yoy",
        "eps_yoy",
        "bulk_net_value_30d",
        "promoter_director_net_value_90d",
        "rationale",
    ]
    keep_columns = [column for column in keep_columns if column in scored.columns]
    return scored[keep_columns].head(shortlist_size).copy()


def _score_current_focus(
    current: pd.DataFrame,
    *,
    feature_columns: list[str],
    universe_name: str,
    bundle: dict[str, object],
    calibration: pd.DataFrame,
    output_suffix: str,
) -> pd.DataFrame:
    universe_masks = build_universe_masks(current)
    scoped = current.loc[universe_masks[universe_name].fillna(False).astype(bool)].copy()
    x = transform_frame(scoped, bundle["stats"])
    prob_up = _predict_classifier_bundle(bundle["up"], x)
    prob_5 = _predict_classifier_bundle(bundle["five"], x)
    prob_10 = _predict_classifier_bundle(bundle["ten"], x)
    pred_return = _predict_regressor_bundle(bundle["reg"], x)
    scoped[f"prob_up_{output_suffix}"] = prob_up
    scoped[f"prob_5pct_{output_suffix}"] = prob_5
    scoped[f"prob_10pct_{output_suffix}"] = prob_10
    scoped[f"pred_return_{output_suffix}"] = pred_return
    scoped[f"pred_price_{output_suffix}"] = pd.to_numeric(scoped["close"], errors="coerce") * (1.0 + pred_return)
    scoped["focus_score"] = _combine_focus_score(prob_5, prob_10, pred_return)
    scoped = apply_calibration_generic(
        scoped,
        calibration,
        score_col="focus_score",
        confidence_col=f"calibrated_confidence_5pct_{output_suffix}",
        avg_return_col=f"calibrated_avg_return_{output_suffix}",
    )
    return scoped


def apply_calibration_generic(
    frame: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    score_col: str,
    confidence_col: str,
    avg_return_col: str,
) -> pd.DataFrame:
    working = frame.copy()
    if calibration.empty:
        working[confidence_col] = pd.NA
        working[avg_return_col] = pd.NA
        working["calibration_bin"] = pd.NA
        return working
    ordered = calibration.sort_values("score_min").reset_index(drop=True)
    score_mins = ordered["score_min"].to_numpy(dtype=float)
    positions = np.searchsorted(
        score_mins,
        pd.to_numeric(working[score_col], errors="coerce").fillna(score_mins[0]).to_numpy(dtype=float),
        side="right",
    ) - 1
    positions = np.clip(positions, 0, len(ordered) - 1)
    working["calibration_bin"] = ordered.iloc[positions]["calibration_bin"].to_numpy(dtype=int)
    by_bin = ordered.set_index("calibration_bin")
    working[confidence_col] = working["calibration_bin"].map(by_bin["hit_rate"])
    working[avg_return_col] = working["calibration_bin"].map(by_bin["avg_return"])
    return working


def _select_daily_top_n(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("trade_date", sort=False):
        ordered = group.sort_values(["focus_score", "symbol"], ascending=[False, True])
        parts.append(ordered.head(top_n).copy())
    if not parts:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(parts, ignore_index=True)


def _summarize_daily_selection(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "mean_return_mean": np.nan,
            "mean_return_median": np.nan,
            "mean_return_p75": np.nan,
            "median_stock_return_mean": np.nan,
            "median_stock_return_median": np.nan,
            "median_stock_return_p75": np.nan,
            "p75_stock_return_mean": np.nan,
            "p75_stock_return_median": np.nan,
            "p75_stock_return_p75": np.nan,
        }
    daily = (
        frame.groupby("trade_date", as_index=False)
        .agg(
            mean_return=("forward_return", "mean"),
            median_stock_return=("forward_return", "median"),
            p75_stock_return=("forward_return", lambda values: pd.Series(values).quantile(0.75)),
        )
        .sort_values("trade_date")
    )
    return {
        "mean_return_mean": float(daily["mean_return"].mean()),
        "mean_return_median": float(daily["mean_return"].median()),
        "mean_return_p75": float(daily["mean_return"].quantile(0.75)),
        "median_stock_return_mean": float(daily["median_stock_return"].mean()),
        "median_stock_return_median": float(daily["median_stock_return"].median()),
        "median_stock_return_p75": float(daily["median_stock_return"].quantile(0.75)),
        "p75_stock_return_mean": float(daily["p75_stock_return"].mean()),
        "p75_stock_return_median": float(daily["p75_stock_return"].median()),
        "p75_stock_return_p75": float(daily["p75_stock_return"].quantile(0.75)),
    }


def _drop_etf_like_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    symbol = working.get("symbol", pd.Series("", index=working.index, dtype="string")).astype("string")
    etf_like = symbol.str.contains(r"(ETF|BEES)", case=False, na=False)
    return working.loc[~etf_like].copy()


def build_rationale(row: pd.Series) -> str:
    parts: list[str] = []
    volume_vs_20d = _maybe_float(row.get("volume_vs_20d"))
    if volume_vs_20d is not None and volume_vs_20d >= 1.5:
        parts.append(f"volume {volume_vs_20d:.2f}x 20d")
    rsi_daily = _maybe_float(row.get("rsi_14_daily"))
    if rsi_daily is not None and rsi_daily >= 60:
        parts.append(f"daily RSI {rsi_daily:.1f}")
    if _truthy(row.get("recent_results_flag")):
        parts.append("fresh results")
    if _truthy(row.get("recent_order_win_flag")):
        parts.append("recent order win")
    if _truthy(row.get("recent_approval_flag")):
        parts.append("recent approval")
    if _truthy(row.get("recent_promoter_buy_flag")) or _truthy(row.get("recent_promoter_or_director_buy_flag")):
        parts.append("promoter/director buy")
    if _truthy(row.get("recent_bulk_buy_flag")):
        parts.append("bulk buy flow")
    calibrated = _maybe_float(row.get("calibrated_confidence_5pct_1d"))
    if calibrated is not None:
        parts.append(f"calibrated 1d 5% hit rate {calibrated:.1%}")
    return ", ".join(parts[:5]) if parts else "model-ranked 1-day 5% candidate"


def _extract_summary_metrics(frame: pd.DataFrame, universe_name: str) -> dict[str, object]:
    row = frame.loc[frame["universe_name"] == universe_name]
    if row.empty:
        return {}
    payload = row.iloc[0].to_dict()
    return {str(key): _jsonify(value) for key, value in payload.items()}


def _jsonify(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonify(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _maybe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _squash_return(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(values / 0.02)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the dedicated 1-day 5 percent walk-forward NSE model and current shortlist.")
    parser.add_argument("--config", type=Path, default=Path("configs/ml_research.yaml"))
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2025-12-31")
    parser.add_argument("--live-end-date", default="2026-04-10")
    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--shortlist-size", type=int, default=10)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/day1_5pct_model"))
    parser.add_argument("--force-panel", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_day1_5pct_model(
        config_path=args.config,
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        live_end_date=args.live_end_date,
        min_price=args.min_price,
        shortlist_size=args.shortlist_size,
        calibration_bins=args.calibration_bins,
        output_dir=args.output_dir,
        force_panel=args.force_panel,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
