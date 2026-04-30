from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.analysis.weekly_run_gate_search import GateSearchConfig
from src.analysis.weekly_run_gate_search import search_weekly_run_gates
from src.ml.expert_pipeline import _build_calibration_table
from src.ml.expert_pipeline import _finalize_shortlist
from src.ml.expert_pipeline import _fit_aux_models
from src.ml.expert_pipeline import _fit_focus_models
from src.ml.expert_pipeline import _merge_current_frames
from src.ml.expert_pipeline import _require_horizon
from src.ml.expert_pipeline import _score_aux_current
from src.ml.expert_pipeline import _score_focus_current
from src.ml.expert_pipeline import _to_objective
from src.ml.expert_pipeline import ExpertConfig
from src.ml.expert_pipeline import load_or_evaluate_focus_horizon
from src.ml.expert_pipeline import load_expert_config
from src.ml.feature_registry import available_feature_columns
from src.ml.panel import build_current_feature_slice
from src.ml.panel import prepare_feature_panel
from src.ml.universes import build_universe_masks
from src.report.stateful_weekly_winners import generate_stateful_weekly_decision_sheet
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme
from src.utils.io import write_json
from src.utils.io import write_parquet


@dataclass(frozen=True)
class ProductionWeeklyWinnersConfig:
    expert_config_path: Path
    run_output_dir: Path
    portfolio_state_dir: Path
    selection_universes: tuple[str, ...]
    top_n: int
    objective_min_winners: int
    min_search_weeks: int
    min_test_weeks: int
    min_search_success_rate: float
    min_test_success_rate: float
    min_all_success_rate: float
    cash_buffer_pct: float
    cadence_day_local: str
    cadence_time_local: str
    timezone_name: str
    search_years: tuple[int, ...]
    test_years: tuple[int, ...]


@dataclass(frozen=True)
class RunCheck:
    name: str
    passed: bool
    message: str
    details: dict[str, Any]


def load_weekly_winners_config(path: Path) -> ProductionWeeklyWinnersConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    settings = raw.get("settings", {})
    expert_config_path = Path(raw["expert_config"])
    if not expert_config_path.is_absolute():
        expert_config_path = (path.parent.parent / expert_config_path).resolve()
    run_output_dir = Path(settings.get("run_output_dir", "reports/ml_weekly_winners"))
    if not run_output_dir.is_absolute():
        run_output_dir = (path.parent.parent / run_output_dir).resolve()
    portfolio_state_dir = Path(settings.get("portfolio_state_dir", "data/portfolio_state"))
    if not portfolio_state_dir.is_absolute():
        portfolio_state_dir = (path.parent.parent / portfolio_state_dir).resolve()
    return ProductionWeeklyWinnersConfig(
        expert_config_path=expert_config_path,
        run_output_dir=run_output_dir,
        portfolio_state_dir=portfolio_state_dir,
        selection_universes=tuple(str(value) for value in settings.get("selection_universes", [])),
        top_n=int(settings.get("top_n", 12)),
        objective_min_winners=int(settings.get("objective_min_winners", 2)),
        min_search_weeks=int(settings.get("min_search_weeks", 20)),
        min_test_weeks=int(settings.get("min_test_weeks", 10)),
        min_search_success_rate=float(settings.get("min_search_success_rate", 0.55)),
        min_test_success_rate=float(settings.get("min_test_success_rate", 0.60)),
        min_all_success_rate=float(settings.get("min_all_success_rate", 0.55)),
        cash_buffer_pct=float(settings.get("cash_buffer_pct", 10.0)),
        cadence_day_local=str(settings.get("cadence_day_local", "MONDAY")),
        cadence_time_local=str(settings.get("cadence_time_local", "20:30")),
        timezone_name=str(settings.get("timezone_name", "Asia/Kolkata")),
        search_years=tuple(int(value) for value in settings.get("search_years", [2023, 2024])),
        test_years=tuple(int(value) for value in settings.get("test_years", [2025])),
    )


def run_production_weekly_winners(
    config: ProductionWeeklyWinnersConfig,
    *,
    force_panel: bool = False,
) -> dict[str, Any]:
    expert_config = load_expert_config(config.expert_config_path)
    selection_universes = _resolve_selection_universes(config, expert_config)
    run_root = config.run_output_dir / pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_root.mkdir(parents=True, exist_ok=True)

    focus_spec = _require_horizon(expert_config, expert_config.focus_horizon)
    focus_objective = _to_objective(focus_spec)
    focus_panel, focus_panel_path = prepare_feature_panel(expert_config.base_config, focus_objective, force=force_panel)
    focus_feature_columns = available_feature_columns(list(focus_panel.columns), expert_config.base_config.feature_columns)
    focus_predictions, focus_summaries = load_or_evaluate_focus_horizon(
        focus_panel,
        feature_columns=focus_feature_columns,
        config=expert_config,
        horizon_spec=focus_spec,
        panel_path=focus_panel_path,
        force=force_panel,
    )
    focus_summary = pd.DataFrame(focus_summaries).sort_values(
        ["sort_primary", "sort_secondary", "sort_tertiary"],
        ascending=[False, False, False],
    )
    focus_summary_path = run_root / "weekly_winners_universe_summary.csv"
    focus_summary.to_csv(focus_summary_path, index=False)
    write_dataframe_manifest(
        focus_summary_path,
        focus_summary,
        generated_by="src.report.production_weekly_winners",
    )

    if focus_predictions.empty or focus_summary.empty:
        return _write_blocked_run(
            run_root,
            reason="No valid walk-forward predictions were produced for the 7-day objective.",
            config=config,
            extra={"focus_panel_path": str(focus_panel_path)},
        )

    oof_dir = run_root / "oof_by_universe"
    oof_dir.mkdir(parents=True, exist_ok=True)
    for universe_name, group in focus_predictions.groupby("universe_name", sort=False):
        path = oof_dir / f"{universe_name}_oof.parquet"
        write_parquet(group, path)
        write_dataframe_manifest(
            path,
            group,
            generated_by="src.report.production_weekly_winners",
        )

    gate_candidates = search_weekly_run_gates(
        GateSearchConfig(
            input_dir=oof_dir,
            output_csv=run_root / "weekly_winners_gate_candidates.csv",
            universes=selection_universes,
            top_n_values=(config.top_n,),
            min_winner_values=(config.objective_min_winners,),
            search_years=config.search_years,
            test_years=config.test_years,
            min_search_weeks=config.min_search_weeks,
            min_test_weeks=config.min_test_weeks,
        )
    )
    gate_candidates_path = run_root / "weekly_winners_gate_candidates.csv"

    current_slice = build_current_feature_slice(expert_config.base_config)
    if current_slice.empty:
        return _write_blocked_run(
            run_root,
            reason="Current feature slice is empty, so there is no live trade-date to score.",
            config=config,
            extra={"focus_panel_path": str(focus_panel_path)},
        )
    current_regime = _extract_current_regime(current_slice)

    current_by_universe = _build_current_universe_scores(
        expert_config=expert_config,
        focus_panel=focus_panel,
        focus_feature_columns=focus_feature_columns,
        focus_predictions=focus_predictions,
        current_slice=current_slice,
        selection_universes=selection_universes,
        force_panel=force_panel,
        top_n=config.top_n,
    )
    if not current_by_universe:
        return _write_blocked_run(
            run_root,
            reason="Could not build current scores for any configured universe.",
            config=config,
            extra={"focus_panel_path": str(focus_panel_path)},
        )

    gate_candidates = gate_candidates.copy()
    gate_candidates["active_today"] = gate_candidates["gate"].map(lambda gate: _gate_is_active(str(gate), current_regime))
    gate_candidates["passes_min_thresholds"] = (
        gate_candidates["search_success_rate"].ge(config.min_search_success_rate)
        & gate_candidates["test_success_rate"].ge(config.min_test_success_rate)
        & gate_candidates["all_success_rate"].ge(config.min_all_success_rate)
    )
    gate_candidates.to_csv(gate_candidates_path, index=False)
    write_dataframe_manifest(
        gate_candidates_path,
        gate_candidates,
        generated_by="src.report.production_weekly_winners",
    )

    eligible = gate_candidates.loc[
        gate_candidates["active_today"].fillna(False) & gate_candidates["passes_min_thresholds"].fillna(False)
    ].copy()
    if not eligible.empty:
        universe_priority = {name: idx for idx, name in enumerate(selection_universes)}
        eligible["universe_priority"] = eligible["universe_name"].map(universe_priority).fillna(len(universe_priority)).astype(int)
    if eligible.empty:
        return _write_blocked_run(
            run_root,
            reason="No historically validated weekly gate is active today with the required minimum success thresholds.",
            config=config,
            extra={
                "current_regime": current_regime,
                "focus_panel_path": str(focus_panel_path),
                "gate_candidates_path": str(gate_candidates_path),
            },
            checks=[
                RunCheck(
                    name="active_validated_gate",
                    passed=False,
                    message="No gate is both active today and above the minimum success thresholds.",
                    details={"candidate_count": int(len(gate_candidates)), "eligible_count": 0},
                )
            ],
        )

    selected_gate = eligible.sort_values(
        ["stability_score", "test_success_rate", "test_avg_winners", "search_success_rate", "universe_priority", "gate"],
        ascending=[False, False, False, False, True, True],
    ).iloc[0]
    universe_name = str(selected_gate["universe_name"])
    shortlist = current_by_universe[universe_name].head(int(selected_gate["top_n"])).copy()
    shortlist["selected_gate"] = str(selected_gate["gate"])
    shortlist["selected_universe"] = universe_name
    shortlist["selected_gate_test_success_rate"] = float(selected_gate["test_success_rate"])
    shortlist["selected_gate_test_avg_winners"] = float(selected_gate["test_avg_winners"])

    checks = validate_weekly_winner_shortlist(shortlist, expected_count=int(selected_gate["top_n"]))
    if any(not check.passed for check in checks):
        return _write_blocked_run(
            run_root,
            reason="Current shortlist failed production sanity checks.",
            config=config,
            extra={
                "current_regime": current_regime,
                "selected_gate": _jsonify(selected_gate.to_dict()),
                "focus_panel_path": str(focus_panel_path),
            },
            checks=checks,
        )

    shortlist_path = run_root / "weekly_winners_shortlist.csv"
    shortlist.to_csv(shortlist_path, index=False)
    write_dataframe_manifest(
        shortlist_path,
        shortlist,
        generated_by="src.report.production_weekly_winners",
        as_of_date=str(shortlist["trade_date"].iloc[0]) if not shortlist.empty else None,
        extra_notes=[
            "This shortlist is fail-closed: it exists only because an active historical regime gate passed the configured thresholds.",
            "The objective is at least 2 names up 5% or more over the next 7 calendar days, using a top-12 basket.",
        ],
    )
    decision_artifacts = generate_stateful_weekly_decision_sheet(
        shortlist=shortlist,
        live_market_frame=current_slice,
        output_dir=run_root / "decision_sheet",
        state_dir=config.portfolio_state_dir,
        as_of_trade_date=str(shortlist["trade_date"].iloc[0]),
        objective_name="weekly_7d_5pct",
        cadence_day_local=config.cadence_day_local,
        cadence_time_local=config.cadence_time_local,
        timezone_name=config.timezone_name,
        cash_buffer_pct=config.cash_buffer_pct,
    )

    summary = {
        "status": "ok",
        "run_root": str(run_root),
        "as_of_trade_date": str(shortlist["trade_date"].iloc[0]) if not shortlist.empty else None,
        "objective": {
            "horizon_days": 7,
            "target_return": 0.05,
            "top_n": int(selected_gate["top_n"]),
            "objective_min_winners": int(selected_gate["min_winners"]),
        },
        "current_regime": current_regime,
        "selection_universes": list(selection_universes),
        "selected_gate": _jsonify(selected_gate.to_dict()),
        "focus_panel_path": str(focus_panel_path),
        "checks": [asdict(check) for check in checks],
        "top_symbols": shortlist["symbol"].tolist(),
        "portfolio_workflow": {
            "cadence_day_local": config.cadence_day_local,
            "cadence_time_local": config.cadence_time_local,
            "timezone_name": config.timezone_name,
            "cash_buffer_pct": config.cash_buffer_pct,
            "portfolio_state_dir": str(config.portfolio_state_dir),
            "decision_sheet_path": str(decision_artifacts.csv_path),
        },
    }
    summary_path = run_root / "weekly_winners_summary.json"
    summary = _jsonify(summary)
    write_json(summary, summary_path)
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src.report.production_weekly_winners",
        as_of_date=summary["as_of_trade_date"],
    )

    write_report_directory_readme(
        run_root,
        title="Weekly Winners Run",
        intro_lines=[
            "This folder contains the fail-closed weekly 7-day winners run.",
            "A shortlist is produced only when an active regime gate has cleared the configured historical thresholds.",
            "Open `weekly_winners_summary.json` first, then the shortlist CSV and the stateful decision sheet.",
        ],
        files=[summary_path, shortlist_path, decision_artifacts.csv_path, gate_candidates_path, focus_summary_path],
    )

    return summary


def validate_weekly_winner_shortlist(frame: pd.DataFrame, *, expected_count: int) -> list[RunCheck]:
    required_columns = [
        "symbol",
        "trade_date",
        "close",
        "prob_5pct_7d",
        "focus_score",
        "calibrated_confidence_5pct_7d",
        "pred_return_7d",
    ]
    checks = [
        RunCheck(
            name="shortlist_row_count",
            passed=len(frame) >= expected_count,
            message="shortlist has the expected number of rows" if len(frame) >= expected_count else "shortlist is undersized",
            details={"row_count": int(len(frame)), "expected_count": int(expected_count)},
        ),
        RunCheck(
            name="shortlist_duplicate_symbols",
            passed=not frame["symbol"].duplicated().any() if "symbol" in frame.columns else False,
            message="shortlist has unique symbols" if ("symbol" in frame.columns and not frame["symbol"].duplicated().any()) else "duplicate symbols found",
            details={"duplicate_count": int(frame["symbol"].duplicated().sum()) if "symbol" in frame.columns else None},
        ),
        RunCheck(
            name="shortlist_required_columns",
            passed=all(column in frame.columns for column in required_columns),
            message="all required shortlist columns are present" if all(column in frame.columns for column in required_columns) else "required shortlist columns are missing",
            details={"required_columns": required_columns},
        ),
    ]
    if all(column in frame.columns for column in required_columns):
        null_issues = {column: int(frame[column].isna().sum()) for column in required_columns if frame[column].isna().any()}
        checks.append(
            RunCheck(
                name="shortlist_no_required_nulls",
                passed=not null_issues,
                message="required shortlist columns have no nulls" if not null_issues else "required shortlist columns contain nulls",
                details={"null_issues": null_issues},
            )
        )
        if {"target_zone_7d_flag", "ranking_score", "calibrated_confidence_5pct_7d", "symbol"}.issubset(frame.columns):
            expected_order = (
                frame.reset_index(drop=True)
                .sort_values(
                    ["target_zone_7d_flag", "ranking_score", "calibrated_confidence_5pct_7d", "symbol"],
                    ascending=[False, False, False, True],
                    kind="mergesort",
                )
                .reset_index(drop=True)
            )
            monotonic = frame.reset_index(drop=True)["symbol"].tolist() == expected_order["symbol"].tolist()
            sort_column = "target_zone_7d_flag,ranking_score,calibrated_confidence_5pct_7d,symbol"
        else:
            sort_column = "ranking_score" if "ranking_score" in frame.columns else "focus_score"
            monotonic = frame[sort_column].fillna(-1e9).is_monotonic_decreasing
        checks.append(
            RunCheck(
                name="shortlist_sorted_by_focus_score",
                passed=bool(monotonic),
                message=(
                    f"shortlist is sorted by descending {sort_column}"
                    if monotonic
                    else f"shortlist sorting drifted from {sort_column} order"
                ),
                details={"sort_column": sort_column},
            )
        )
    return checks


def _build_current_universe_scores(
    *,
    expert_config: ExpertConfig,
    focus_panel: pd.DataFrame,
    focus_feature_columns: list[str],
    focus_predictions: pd.DataFrame,
    current_slice: pd.DataFrame,
    selection_universes: tuple[str, ...],
    force_panel: bool,
    top_n: int,
) -> dict[str, pd.DataFrame]:
    universe_masks_focus = build_universe_masks(focus_panel)
    focus_spec = _require_horizon(expert_config, expert_config.focus_horizon)
    aux_specs = [spec for spec in expert_config.horizons if spec.name != focus_spec.name]
    aux_panels: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for spec in aux_specs:
        objective = _to_objective(spec)
        panel, _ = prepare_feature_panel(expert_config.base_config, objective, force=force_panel)
        aux_panels[spec.name] = (
            panel,
            available_feature_columns(list(panel.columns), expert_config.base_config.feature_columns),
        )

    current_by_universe: dict[str, pd.DataFrame] = {}
    for universe_name in selection_universes:
        if universe_name not in universe_masks_focus:
            continue
        scoped_focus = focus_panel.loc[universe_masks_focus[universe_name].fillna(False).astype(bool)].copy()
        if scoped_focus.empty:
            continue
        universe_predictions = focus_predictions.loc[focus_predictions["universe_name"] == universe_name].copy()
        if universe_predictions.empty:
            continue
        calibration = _build_calibration_table(
            universe_predictions,
            score_col="focus_score",
            target_col="winner_5pct",
            return_col="forward_return",
            bins=expert_config.calibration_bins,
        )
        focus_bundle = _fit_focus_models(scoped_focus, feature_columns=focus_feature_columns)
        current_frames = [
            _score_focus_current(
                current_slice,
                feature_columns=focus_feature_columns,
                universe_name=universe_name,
                bundle=focus_bundle,
                calibration=calibration,
            )
        ]
        for spec in aux_specs:
            panel, feature_columns = aux_panels[spec.name]
            scoped_aux = panel.loc[build_universe_masks(panel)[universe_name].fillna(False).astype(bool)].copy()
            if scoped_aux.empty:
                continue
            aux_bundle = _fit_aux_models(scoped_aux, feature_columns=feature_columns)
            current_frames.append(
                _score_aux_current(
                    current_slice,
                    feature_columns=feature_columns,
                    universe_name=universe_name,
                    bundle=aux_bundle,
                    horizon_name=spec.name,
                )
            )
        merged = _merge_current_frames(current_frames)
        merged = _finalize_shortlist(merged, shortlist_size=top_n)
        merged["universe_name"] = universe_name
        current_by_universe[universe_name] = merged
    return current_by_universe


def _resolve_selection_universes(
    config: ProductionWeeklyWinnersConfig,
    expert_config: ExpertConfig,
) -> tuple[str, ...]:
    if config.selection_universes:
        ordered = []
        seen: set[str] = set()
        for name in config.selection_universes:
            normalized = str(name)
            if normalized in seen:
                continue
            if normalized not in expert_config.base_config.universes:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        if ordered:
            return tuple(ordered)
    return tuple(expert_config.base_config.universes)


def _extract_current_regime(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    row = frame.iloc[0]
    keys = [
        "trade_date",
        "breadth_above_50_dma",
        "breadth_above_200_dma",
        "breadth_volume_1_5x",
        "market_median_return_20d",
        "nifty_50_return_20d",
        "nifty_500_return_20d",
        "macro_risk_on_flag",
        "macro_vix_below_20",
    ]
    return {key: _jsonify(row.get(key)) for key in keys if key in frame.columns}


def _gate_is_active(gate_label: str, current_regime: dict[str, Any]) -> bool:
    if gate_label == "no_gate":
        return True
    clauses = [item.strip() for item in gate_label.split("&")]
    for clause in clauses:
        if not clause:
            continue
        if clause.endswith("=True"):
            column = clause[:-5]
            if bool(current_regime.get(column)) is not True:
                return False
            continue
        if ">=" not in clause:
            return False
        column, raw_value = clause.split(">=", 1)
        current_value = current_regime.get(column.strip())
        if current_value is None:
            return False
        try:
            if float(current_value) < float(raw_value.strip()):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _write_blocked_run(
    run_root: Path,
    *,
    reason: str,
    config: ProductionWeeklyWinnersConfig,
    extra: dict[str, Any] | None = None,
    checks: list[RunCheck] | None = None,
) -> dict[str, Any]:
    summary = {
        "status": "blocked",
        "reason": reason,
        "config": _jsonify(asdict(config)),
        "checks": [asdict(check) for check in (checks or [])],
        **(extra or {}),
    }
    summary = _jsonify(summary)
    summary_path = run_root / "weekly_winners_summary.json"
    write_json(summary, summary_path)
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src.report.production_weekly_winners",
    )
    write_report_directory_readme(
        run_root,
        title="Weekly Winners Run",
        intro_lines=[
            "This folder contains a blocked weekly 7-day winners run.",
            "The pipeline is fail-closed: no live shortlist is emitted when a required gate or sanity check fails.",
            "Open `weekly_winners_summary.json` first to see the exact block reason.",
        ],
        files=[summary_path],
    )
    return summary


def _jsonify(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Series, pd.Index)):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fail-closed weekly 7-day winners production model.")
    parser.add_argument("--config", type=Path, default=Path("configs/ml_weekly_production.yaml"))
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()
    config = load_weekly_winners_config(args.config)
    payload = run_production_weekly_winners(config, force_panel=args.force_panel)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
