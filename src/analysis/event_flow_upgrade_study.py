from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.derivatives.nse_oi import NseDerivativesOiFetchConfig
from src.ingest.derivatives.nse_oi import load_derivatives_oi_from_nse
from src.ingest.events.nse import NseAnnouncementFetchConfig
from src.ingest.events.nse import load_announcements_from_nse
from src.ingest.events.nse_bulk_block import NseBulkBlockFetchConfig
from src.ingest.events.nse_bulk_block import load_bulk_block_deals_from_nse
from src.ingest.events.nse_insider import NseInsiderFetchConfig
from src.ingest.events.nse_insider import load_insider_trades_from_nse
from src.ingest.fundamentals.nse import _add_growth_fields
from src.ingest.fundamentals.nse import select_preferred_statement_scope
from src.ml.config import ObjectiveSpec
from src.ml.config import ResearchConfig
from src.ml.config import load_research_config
from src.ml.expert_pipeline import ExpertConfig
from src.ml.expert_pipeline import ExpertHorizonSpec
from src.ml.expert_pipeline import _apply_calibration
from src.ml.expert_pipeline import _build_calibration_table
from src.ml.expert_pipeline import _evaluate_focus_horizon
from src.ml.expert_pipeline import _finalize_shortlist
from src.ml.expert_pipeline import _fit_focus_models
from src.ml.expert_pipeline import _select_daily_top_quantile
from src.ml.expert_pipeline import _score_focus_current
from src.ml.feature_registry import available_feature_columns
from src.ml.panel import build_current_feature_slice
from src.ml.panel import prepare_feature_panel
from src.ml.universes import build_universe_masks
from src.transform.event_daily import build_event_feature_daily
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme
from src.utils.io import write_json
from src.utils.io import write_parquet


NEW_EVENT_FLOW_FEATURES: tuple[str, ...] = (
    "revenue_yoy",
    "pat_yoy",
    "eps_yoy",
    "ebitda_yoy",
    "revenue_yoy_acceleration",
    "pat_yoy_acceleration",
    "eps_yoy_acceleration",
    "ebitda_yoy_acceleration",
    "revenue_yoy_positive_flag",
    "pat_yoy_positive_flag",
    "eps_yoy_positive_flag",
    "revenue_yoy_acceleration_positive_flag",
    "pat_yoy_acceleration_positive_flag",
    "insider_buy_value_30d",
    "insider_sell_value_30d",
    "insider_net_value_30d",
    "insider_buy_count_30d",
    "promoter_director_buy_value_90d",
    "promoter_director_net_value_90d",
    "promoter_director_buy_count_90d",
    "days_since_insider_buy",
    "days_since_promoter_or_director_buy",
    "recent_insider_buy_flag",
    "recent_promoter_or_director_buy_flag",
    "bulk_buy_value_30d",
    "bulk_sell_value_30d",
    "bulk_net_value_30d",
    "bulk_buy_count_30d",
    "days_since_bulk_buy",
    "recent_bulk_buy_flag",
    "block_buy_value_30d",
    "block_sell_value_30d",
    "block_net_value_30d",
    "block_buy_count_30d",
    "days_since_block_buy",
    "recent_block_buy_flag",
    "oi_share_of_mwpl",
    "oi_change_1d",
    "oi_change_pct_1d",
    "futeq_oi_change_1d",
    "oi_share_of_mwpl_change_1d",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the official NSE event-flow layer and compare it against the baseline 7-day model.")
    parser.add_argument("--config", default="configs/ml_research.yaml")
    parser.add_argument("--objective", default="week_5pct")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2026-03-25")
    parser.add_argument("--oi-start-date", default="2023-01-01")
    parser.add_argument("--output-dir", default="tmp/event_flow_upgrade_7d")
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    base_config = load_research_config(config_path)
    objective = _require_objective(base_config, args.objective)

    start_date = pd.Timestamp(args.start_date).date()
    end_date = pd.Timestamp(args.end_date).date()
    oi_start_date = pd.Timestamp(args.oi_start_date).date()

    sources = _refresh_sources(base_config, start_date=start_date, end_date=end_date, oi_start_date=oi_start_date)
    event_daily = _build_event_flow_daily(base_config, output_dir=output_dir, **sources)

    baseline_features = [column for column in base_config.feature_columns if column not in NEW_EVENT_FLOW_FEATURES]
    enriched_features = list(base_config.feature_columns)
    baseline_config = replace(base_config, feature_columns=baseline_features)
    enriched_config = replace(base_config, feature_columns=enriched_features)

    baseline_result = _run_model_comparison(
        baseline_config,
        objective=objective,
        output_dir=output_dir / "baseline",
        force_panel=args.force_panel,
    )
    enriched_result = _run_model_comparison(
        enriched_config,
        objective=objective,
        output_dir=output_dir / "event_flow",
        force_panel=args.force_panel,
    )

    comparison = _build_comparison_frame(baseline_result["universe_metrics"], enriched_result["universe_metrics"])
    comparison_path = output_dir / "event_flow_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    write_dataframe_manifest(
        comparison_path,
        comparison,
        generated_by="src/analysis/event_flow_upgrade_study.py",
        as_of_date=str(end_date),
        extra_notes=[
            "baseline excludes the new event-flow and results-surprise features",
            "event_flow includes official insider, bulk/block, open-interest, and filing-derived surprise features",
        ],
    )

    current_shortlist = enriched_result["current_shortlist"]
    current_shortlist_path = output_dir / "event_flow_current_shortlist.csv"
    current_shortlist.to_csv(current_shortlist_path, index=False)
    write_dataframe_manifest(
        current_shortlist_path,
        current_shortlist,
        generated_by="src/analysis/event_flow_upgrade_study.py",
        as_of_date=str(end_date),
    )

    summary = {
        "status": "ok",
        "objective": objective.__dict__,
        "source_summary": _source_summary_dict(sources, event_daily),
        "baseline_best_universe": baseline_result["best_universe"],
        "event_flow_best_universe": enriched_result["best_universe"],
        "targets": {
            "recall_above_14_9pct": bool((comparison["event_flow_top_decile_recall"] > 0.149).any()),
            "top_bucket_above_28_5pct": bool((comparison["event_flow_top_bucket_hit_rate"] > 0.285).any()),
            "top10_median_stock_return_positive": bool((comparison["event_flow_top10_median_stock_return_median"] > 0.0).any()),
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary, summary_path)
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/event_flow_upgrade_study.py",
        as_of_date=str(end_date),
    )
    write_report_directory_readme(
        output_dir,
        title="Event-Flow Upgrade Study",
        intro_lines=[
            "This folder compares the prior 7-day breakout model against the upgraded event-flow version.",
            "All new source inputs are official NSE archives or APIs, and event/flow features are shifted to the next trade date before use.",
            "Open `summary.json` first, then `event_flow_model_comparison.csv`, then `event_flow_current_shortlist.csv`.",
        ],
        files=[summary_path, comparison_path, current_shortlist_path],
    )

    print(json.dumps(summary, indent=2))


def _refresh_sources(
    base_config: ResearchConfig,
    *,
    start_date: date,
    end_date: date,
    oi_start_date: date,
) -> dict[str, pd.DataFrame]:
    events_root = Path("data/events_full_history")
    insider_root = events_root / "insider"
    bulk_block_root = events_root / "bulk_block"
    oi_root = Path("data/derivatives_full_history")

    announcements = load_announcements_from_nse(
        NseAnnouncementFetchConfig(
            output_dir=events_root,
            start_date=start_date,
            end_date=end_date,
            delay_seconds=0.0,
            window_days=31,
        )
    )
    insider = load_insider_trades_from_nse(
        NseInsiderFetchConfig(
            output_dir=insider_root,
            start_date=start_date,
            end_date=end_date,
            delay_seconds=0.0,
            window_days=31,
        )
    )
    bulk_block = load_bulk_block_deals_from_nse(
        NseBulkBlockFetchConfig(
            output_dir=bulk_block_root,
            start_date=max(start_date, oi_start_date),
            end_date=end_date,
            delay_seconds=0.0,
        )
    )

    daily_facts = pd.read_parquet(base_config.paths.daily_facts, columns=["trade_date"])
    trade_dates = {
        ts.date()
        for ts in pd.to_datetime(daily_facts["trade_date"]).dt.normalize().drop_duplicates().tolist()
        if oi_start_date <= ts.date() <= end_date
    }
    derivatives_oi = load_derivatives_oi_from_nse(
        NseDerivativesOiFetchConfig(
            output_dir=oi_root,
            start_date=oi_start_date,
            end_date=end_date,
            trade_dates=trade_dates,
            delay_seconds=0.0,
        )
    )
    fundamentals = pd.read_parquet(base_config.paths.fundamentals)
    fundamentals = _add_growth_fields(fundamentals.sort_values(["symbol", "fiscal_period_end", "effective_from_date"]).reset_index(drop=True))
    fundamentals = select_preferred_statement_scope(fundamentals)
    fundamentals = fundamentals.sort_values(["symbol", "fiscal_period_end", "effective_from_date"]).reset_index(drop=True)
    write_parquet(fundamentals, base_config.paths.fundamentals)
    source_outputs = {
        events_root / "normalized" / "stock_announcements.parquet": announcements,
        insider_root / "normalized" / "stock_insider_trades.parquet": insider,
        bulk_block_root / "normalized" / "stock_bulk_block_deals.parquet": bulk_block,
        oi_root / "normalized" / "stock_derivatives_oi.parquet": derivatives_oi,
        base_config.paths.fundamentals: fundamentals,
    }
    for path, frame in source_outputs.items():
        if frame.empty:
            continue
        date_col = "trade_date" if "trade_date" in frame.columns else "event_date" if "event_date" in frame.columns else "effective_from_date"
        write_dataframe_manifest(
            path,
            frame,
            generated_by="src/analysis/event_flow_upgrade_study.py",
            as_of_date=str(pd.to_datetime(frame[date_col]).max().date()),
        )

    return {
        "announcements": announcements,
        "insider_trades": insider,
        "bulk_block_deals": bulk_block,
        "derivatives_oi": derivatives_oi,
        "fundamentals": fundamentals,
    }


def _build_event_flow_daily(
    base_config: ResearchConfig,
    *,
    output_dir: Path,
    announcements: pd.DataFrame,
    insider_trades: pd.DataFrame,
    bulk_block_deals: pd.DataFrame,
    derivatives_oi: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    daily_facts = pd.read_parquet(base_config.paths.daily_facts, columns=["symbol", "trade_date"])
    trade_calendar = daily_facts[["symbol", "trade_date"]].drop_duplicates().sort_values(["symbol", "trade_date"])
    event_daily = build_event_feature_daily(
        trade_calendar,
        announcements,
        insider_trades=insider_trades,
        bulk_block_deals=bulk_block_deals,
        derivatives_oi=derivatives_oi,
    )
    write_parquet(event_daily, base_config.paths.event_daily)
    write_dataframe_manifest(
        base_config.paths.event_daily,
        event_daily,
        generated_by="src/analysis/event_flow_upgrade_study.py",
        as_of_date=str(pd.to_datetime(event_daily["trade_date"]).max().date()),
        extra_notes=[
            "announcements, insider, bulk/block, and OI features are shifted to the next trade date before use",
            "results surprise proxies live in the quarterly fundamentals table and join separately by effective_from_date",
        ],
    )
    write_report_directory_readme(
        base_config.paths.event_daily.parent,
        title="Normalized Event-Flow Datasets",
        intro_lines=[
            "This folder holds the lag-safe daily event-flow feature table and its source-normalized inputs.",
            "The daily file is keyed by symbol and trade_date and is meant to be merged into the ML panel or live screen universe.",
        ],
        files=[
            base_config.paths.event_daily,
            Path("data/events_full_history/normalized/stock_announcements.parquet"),
            Path("data/events_full_history/insider/normalized/stock_insider_trades.parquet"),
            Path("data/events_full_history/bulk_block/normalized/stock_bulk_block_deals.parquet"),
            Path("data/derivatives_full_history/normalized/stock_derivatives_oi.parquet"),
            base_config.paths.fundamentals,
        ],
    )
    return event_daily


def _run_model_comparison(
    config: ResearchConfig,
    *,
    objective: ObjectiveSpec,
    output_dir: Path,
    force_panel: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon = ExpertHorizonSpec(
        name=objective.name,
        horizon_days=objective.horizon_days,
        analysis_start_date=objective.analysis_start_date,
        analysis_end_date=objective.analysis_end_date,
        min_price=objective.min_price,
    )
    expert_config = ExpertConfig(
        base_config_path=Path("configs/ml_research.yaml"),
        base_config=config,
        horizons=[horizon],
        focus_horizon=objective.name,
        shortlist_size=10,
        calibration_bins=10,
        run_output_dir=output_dir,
    )
    panel, panel_path = prepare_feature_panel(config, objective, force=force_panel)
    feature_columns = available_feature_columns(list(panel.columns), config.feature_columns)
    predictions, summaries = _evaluate_focus_horizon(
        panel,
        feature_columns=feature_columns,
        config=expert_config,
        horizon_spec=horizon,
    )
    summary_df = pd.DataFrame(summaries).sort_values(
        ["sort_primary", "sort_secondary", "sort_tertiary"],
        ascending=[False, False, False],
    )
    summary_path = output_dir / "universe_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    write_dataframe_manifest(
        summary_path,
        summary_df,
        generated_by="src/analysis/event_flow_upgrade_study.py",
        as_of_date=str(pd.to_datetime(panel["trade_date"]).max().date()),
    )

    oof_path = output_dir / "oof_predictions.parquet"
    write_parquet(predictions, oof_path)
    write_dataframe_manifest(
        oof_path,
        predictions.head(min(len(predictions), 25000)),
        generated_by="src/analysis/event_flow_upgrade_study.py",
        as_of_date=str(pd.to_datetime(panel["trade_date"]).max().date()),
        extra_notes=["Manifest profile is sampled to keep the sidecar compact; the parquet itself contains the full OOF table."],
    )

    universe_metrics = _summarize_universes(predictions, summary_df, top_n=config.top_n_daily)
    universe_metrics_path = output_dir / "universe_metrics.csv"
    universe_metrics.to_csv(universe_metrics_path, index=False)
    write_dataframe_manifest(
        universe_metrics_path,
        universe_metrics,
        generated_by="src/analysis/event_flow_upgrade_study.py",
        as_of_date=str(pd.to_datetime(panel["trade_date"]).max().date()),
    )

    best_universe = str(universe_metrics.sort_values(
        ["top_bucket_hit_rate", "top_decile_recall", "top10_median_stock_return_median"],
        ascending=[False, False, False],
    ).iloc[0]["universe_name"])
    current_shortlist = _score_current_shortlist(
        config,
        panel=panel,
        feature_columns=feature_columns,
        predictions=predictions,
        best_universe=best_universe,
    )
    return {
        "panel_path": str(panel_path),
        "summary_df": summary_df,
        "universe_metrics": universe_metrics,
        "best_universe": best_universe,
        "current_shortlist": current_shortlist,
    }


def _summarize_universes(predictions: pd.DataFrame, summary_df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for universe_name, universe_predictions in predictions.groupby("universe_name", sort=False):
        universe_predictions = universe_predictions.sort_values(["trade_date", "focus_score"], ascending=[True, False]).copy()
        calibration = _build_calibration_table(
            universe_predictions,
            score_col="focus_score",
            target_col="winner_5pct",
            return_col="forward_return",
            bins=10,
        )
        top_bucket_hit_rate = float(calibration.sort_values("score_min").iloc[-1]["hit_rate"]) if not calibration.empty else np.nan
        total_winners = int(pd.to_numeric(universe_predictions["winner_5pct"], errors="coerce").fillna(0).sum())
        top_decile = _select_daily_top_quantile(
            universe_predictions,
            score_col="focus_score",
            top_quantile=0.10,
        )
        top_decile_hits = int(pd.to_numeric(top_decile["winner_5pct"], errors="coerce").fillna(0).sum())
        top_decile_recall = float(top_decile_hits / total_winners) if total_winners else np.nan
        top_n_rows = _select_daily_top_n(universe_predictions, top_n=top_n)
        daily_top_n = (
            top_n_rows.groupby("trade_date", as_index=False)
            .agg(
                topn_precision=("winner_5pct", "mean"),
                topn_mean_return=("forward_return", "mean"),
                topn_median_stock_return=("forward_return", "median"),
            )
            .sort_values("trade_date")
        )
        base_avg_return = float(pd.to_numeric(universe_predictions["forward_return"], errors="coerce").mean())
        base_median_return = float(pd.to_numeric(universe_predictions["forward_return"], errors="coerce").median())
        base_p75_return = float(pd.to_numeric(universe_predictions["forward_return"], errors="coerce").quantile(0.75))
        summary_row = summary_df.loc[summary_df["universe_name"] == universe_name].iloc[0].to_dict()
        rows.append(
            {
                "universe_name": universe_name,
                "row_count": int(len(universe_predictions)),
                "base_rate_5pct": float(summary_row["base_rate_5pct"]),
                "top_quantile_precision_5pct": float(summary_row["top_quantile_precision_5pct"]),
                "selected_precision_pooled": float(summary_row["selected_precision_pooled"]),
                "top_bucket_hit_rate": top_bucket_hit_rate,
                "top_decile_recall": top_decile_recall,
                "base_avg_return": base_avg_return,
                "base_median_return": base_median_return,
                "base_p75_return": base_p75_return,
                "top10_mean_return_mean": float(daily_top_n["topn_mean_return"].mean()),
                "top10_mean_return_median": float(daily_top_n["topn_mean_return"].median()),
                "top10_mean_return_p75": float(daily_top_n["topn_mean_return"].quantile(0.75)),
                "top10_median_stock_return_mean": float(daily_top_n["topn_median_stock_return"].mean()),
                "top10_median_stock_return_median": float(daily_top_n["topn_median_stock_return"].median()),
                "top10_median_stock_return_p75": float(daily_top_n["topn_median_stock_return"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows).sort_values(["top_bucket_hit_rate", "top_decile_recall"], ascending=[False, False]).reset_index(drop=True)


def _score_current_shortlist(
    config: ResearchConfig,
    *,
    panel: pd.DataFrame,
    feature_columns: list[str],
    predictions: pd.DataFrame,
    best_universe: str,
) -> pd.DataFrame:
    calibration = _build_calibration_table(
        predictions.loc[predictions["universe_name"] == best_universe].copy(),
        score_col="focus_score",
        target_col="winner_5pct",
        return_col="forward_return",
        bins=10,
    )
    masks = build_universe_masks(panel)
    scoped_panel = panel.loc[masks[best_universe].fillna(False).astype(bool)].copy()
    bundle = _fit_focus_models(scoped_panel, feature_columns=feature_columns)
    current_slice = build_current_feature_slice(config)
    current = _score_focus_current(
        current_slice,
        feature_columns=feature_columns,
        universe_name=best_universe,
        bundle=bundle,
        calibration=calibration,
    )
    current = _apply_calibration(current, calibration, score_col="focus_score")
    return _finalize_shortlist(current, shortlist_size=10)


def _build_comparison_frame(baseline: pd.DataFrame, event_flow: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline.add_prefix("baseline_")
    event_flow = event_flow.add_prefix("event_flow_")
    merged = baseline.merge(
        event_flow,
        left_on="baseline_universe_name",
        right_on="event_flow_universe_name",
        how="outer",
    )
    merged["universe_name"] = merged["baseline_universe_name"].fillna(merged["event_flow_universe_name"])
    keep = ["universe_name"]
    for prefix in ("baseline", "event_flow"):
        keep.extend(
            [
                f"{prefix}_base_rate_5pct",
                f"{prefix}_selected_precision_pooled",
                f"{prefix}_top_bucket_hit_rate",
                f"{prefix}_top_decile_recall",
                f"{prefix}_top10_mean_return_mean",
                f"{prefix}_top10_mean_return_p75",
                f"{prefix}_top10_median_stock_return_median",
            ]
        )
    merged = merged[keep].copy()
    merged["recall_delta"] = merged["event_flow_top_decile_recall"] - merged["baseline_top_decile_recall"]
    merged["top_bucket_delta"] = merged["event_flow_top_bucket_hit_rate"] - merged["baseline_top_bucket_hit_rate"]
    merged["top10_median_delta"] = (
        merged["event_flow_top10_median_stock_return_median"] - merged["baseline_top10_median_stock_return_median"]
    )
    merged["meets_recall_target"] = merged["event_flow_top_decile_recall"] > 0.149
    merged["meets_top_bucket_target"] = merged["event_flow_top_bucket_hit_rate"] > 0.285
    merged["meets_top10_median_target"] = merged["event_flow_top10_median_stock_return_median"] > 0
    return merged.sort_values(["event_flow_top_bucket_hit_rate", "event_flow_top_decile_recall"], ascending=[False, False]).reset_index(drop=True)


def _source_summary_dict(sources: dict[str, pd.DataFrame], event_daily: pd.DataFrame) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, frame in sources.items():
        if frame.empty:
            payload[name] = {"rows": 0}
            continue
        date_column = "trade_date" if "trade_date" in frame.columns else "event_date" if "event_date" in frame.columns else "effective_from_date"
        payload[name] = {
            "rows": int(len(frame)),
            "date_min": str(pd.to_datetime(frame[date_column]).min()),
            "date_max": str(pd.to_datetime(frame[date_column]).max()),
            "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else None,
        }
    payload["event_feature_daily"] = {
        "rows": int(len(event_daily)),
        "date_min": str(pd.to_datetime(event_daily["trade_date"]).min()),
        "date_max": str(pd.to_datetime(event_daily["trade_date"]).max()),
        "symbols": int(event_daily["symbol"].nunique()),
    }
    return payload


def _select_daily_top_n(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("trade_date", sort=False):
        parts.append(group.sort_values("focus_score", ascending=False).head(top_n).copy())
    if not parts:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(parts, ignore_index=True)


def _require_objective(config: ResearchConfig, name: str) -> ObjectiveSpec:
    for objective in config.objectives:
        if objective.name == name:
            return objective
    raise KeyError(f"Unknown objective: {name}")


if __name__ == "__main__":
    main()
