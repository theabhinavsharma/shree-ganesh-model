from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_parent
from src.utils.schema import load_contract


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_key: str
    friendly_name: str
    description: str
    grain: str
    contract_path: Path | None = None
    primary_key: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    column_help: dict[str, str] | None = None


def write_dataframe_manifest(
    path: Path,
    df: pd.DataFrame,
    *,
    generated_by: str,
    as_of_date: str | None = None,
    extra_notes: list[str] | None = None,
) -> Path:
    spec = resolve_artifact_spec(path)
    contract = _load_contract_map(spec.contract_path)
    manifest = {
        "artifact_key": spec.artifact_key,
        "friendly_name": spec.friendly_name,
        "description": spec.description,
        "grain": spec.grain,
        "path": str(path),
        "file_name": path.name,
        "file_format": path.suffix.lstrip("."),
        "generated_by": generated_by,
        "as_of_date": as_of_date,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "primary_key": list(spec.primary_key or tuple(contract.get("primary_key") or [])),
        "contract_path": str(spec.contract_path) if spec.contract_path else None,
        "notes": [*spec.notes, *(extra_notes or [])],
        "columns": [_profile_column(df, column, contract=contract.get("columns", {}), extra_help=(spec.column_help or {})) for column in df.columns],
    }
    manifest_path = sidecar_manifest_path(path)
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest_path


def write_json_manifest(
    path: Path,
    payload: dict[str, Any],
    *,
    generated_by: str,
    as_of_date: str | None = None,
    extra_notes: list[str] | None = None,
) -> Path:
    spec = resolve_artifact_spec(path)
    manifest = {
        "artifact_key": spec.artifact_key,
        "friendly_name": spec.friendly_name,
        "description": spec.description,
        "grain": spec.grain,
        "path": str(path),
        "file_name": path.name,
        "file_format": path.suffix.lstrip("."),
        "generated_by": generated_by,
        "as_of_date": as_of_date,
        "primary_key": list(spec.primary_key),
        "notes": [*spec.notes, *(extra_notes or [])],
        "top_level_keys": sorted(payload.keys()),
    }
    manifest_path = sidecar_manifest_path(path)
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest_path


def write_report_directory_readme(
    output_dir: Path,
    *,
    title: str,
    intro_lines: list[str],
    files: list[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.extend(intro_lines)
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for path in sorted(files, key=lambda item: item.name):
        if not path.exists():
            continue
        spec = resolve_artifact_spec(path)
        lines.append(f"- `{path.name}`: {spec.description}")
    lines.append("")
    lines.append("## How to read this folder")
    lines.append("")
    lines.append("- Open `summary.json` first when it exists.")
    lines.append("- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.")
    lines.append("- `individual_counts` means a rule tested alone across the universe.")
    lines.append("- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.")
    lines.append("- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.")
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def sidecar_manifest_path(path: Path) -> Path:
    return Path(f"{path}.manifest.json")


def resolve_artifact_spec(path: Path) -> ArtifactSpec:
    name = path.name
    if name == "current_universe_enriched.parquet":
        return ArtifactSpec(
            artifact_key="current_universe_enriched",
            friendly_name="Current Enriched Screening Universe",
            description="One row per current stock candidate with price, quality, ownership, and screening fields merged into a single live universe table.",
            grain="One row per symbol for the as-of trade date.",
            contract_path=Path("data_contracts/daily_screen_universe.yaml"),
            primary_key=("symbol",),
            notes=("This file is a live screening snapshot, not a historical panel.",),
        )
    if name == "summary.json":
        return ArtifactSpec(
            artifact_key="screen_run_summary",
            friendly_name="Screen Run Summary",
            description="Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.",
            grain="One JSON document per screen run folder.",
            notes=("Use this as the first file to understand a screen output folder.",),
        )
    weekly_match = re.fullmatch(r"weekly_portfolio_\d{8}\.(csv|html)", name)
    if weekly_match:
        suffix = weekly_match.group(1)
        return ArtifactSpec(
            artifact_key=f"weekly_portfolio_{suffix}",
            friendly_name="Weekly Portfolio Report",
            description="Human-readable weekly portfolio output derived from the scored universe.",
            grain="One row per recommended stock in the weekly portfolio." if suffix == "csv" else "One HTML report per weekly run.",
            column_help={
                "current_price": "Latest price used for the report.",
                "buy_price_low": "Lower end of the suggested buy range.",
                "buy_price_high": "Upper end of the suggested buy range.",
                "sell_target": "Suggested upside target for the chosen horizon.",
                "stop_loss": "Suggested downside exit level.",
                "confidence_score": "Relative model confidence on a 0-100 style scale.",
                "allocation_pct": "Suggested percentage of total portfolio capital.",
            },
        )
    if name == "weekly_winners_shortlist.csv":
        return ArtifactSpec(
            artifact_key="weekly_winners_shortlist",
            friendly_name="Weekly Winners Shortlist",
            description="Final fail-closed shortlist for the 7-day 5 percent winners model after the active regime gate and production validations pass.",
            grain="One row per shortlisted stock for the current weekly run.",
            primary_key=("trade_date", "symbol"),
            column_help={
                "prob_5pct_7d": "Raw model probability-like score for clearing 5 percent in 7 days. Use mainly for ranking, not as a literal probability.",
                "calibrated_confidence_5pct_7d": "Out-of-sample bucket hit rate from historical calibration for the current score bucket.",
                "pred_return_7d": "Model-implied 7-day return estimate. This is secondary to the calibrated bucket hit rate.",
                "selected_gate_test_success_rate": "Historical weekly-run success rate in the holdout years for the selected regime gate.",
            },
        )
    weekly_decision_match = re.fullmatch(r"weekly_position_decision_sheet_\d{8}\.csv", name)
    if weekly_decision_match:
        return ArtifactSpec(
            artifact_key="weekly_position_decision_sheet",
            friendly_name="Weekly Position Decision Sheet",
            description="Stateful weekly investing sheet that compares this week's shortlist against confirmed open positions and outputs Buy New, Buy More, Hold, Sell Partly, or Sell Wholly.",
            grain="One row per shortlisted stock or currently held stock requiring a decision.",
            column_help={
                "recommended_action": "What to do this week after comparing the latest shortlist with the confirmed open position book.",
                "action_rationale": "Plain-English explanation of why the action is recommended.",
                "current_allocation_pct": "Current confirmed allocation as a percent of total portfolio capital.",
                "recommended_allocation_pct": "New target allocation as a percent of total portfolio capital.",
                "allocation_delta_pct": "Recommended allocation minus current confirmed allocation, in percentage points.",
                "unrealized_return_pct": "Current mark-to-market return since entry, measured from the recorded entry price.",
            },
        )
    if name == "current_positions.csv":
        return ArtifactSpec(
            artifact_key="portfolio_current_positions",
            friendly_name="Current Open Positions",
            description="Durable snapshot of currently held positions after confirmed trade execution updates.",
            grain="One row per currently open stock position.",
            primary_key=("symbol",),
            column_help={
                "entry_price": "Recorded entry price from the first buy or weighted-average add price.",
                "current_allocation_pct": "Current confirmed allocation as a percent of total portfolio capital.",
                "last_sell_target": "Most recent tactical sell target carried from the confirmed decision sheet.",
                "last_stop_loss": "Most recent tactical stop-loss carried from the confirmed decision sheet.",
            },
        )
    if name == "executed_trade_ledger.csv":
        return ArtifactSpec(
            artifact_key="portfolio_execution_ledger",
            friendly_name="Executed Trade Ledger",
            description="Append-only confirmation ledger of trades that the user says were actually placed.",
            grain="One row per confirmed action.",
            primary_key=("execution_id",),
            column_help={
                "previous_allocation_pct": "Allocation before the confirmed action.",
                "target_allocation_pct": "Allocation asked for by the decision sheet.",
                "resulting_allocation_pct": "Allocation after the confirmed action is assumed executed.",
            },
        )
    if name == "workflow_settings.json":
        return ArtifactSpec(
            artifact_key="portfolio_workflow_settings",
            friendly_name="Portfolio Workflow Settings",
            description="Cadence and objective settings for the durable weekly investing workflow.",
            grain="One JSON document for the portfolio state folder.",
        )
    if name == "weekly_winners_gate_candidates.csv":
        return ArtifactSpec(
            artifact_key="weekly_winners_gate_candidates",
            friendly_name="Weekly Winners Gate Candidates",
            description="Historical gate-search results showing which market-regime gates improved weekly basket success, including whether each gate is active today.",
            grain="One row per universe, gate, top-N basket size, and minimum-winner combination.",
        )
    if name == "weekly_winners_universe_summary.csv":
        return ArtifactSpec(
            artifact_key="weekly_winners_universe_summary",
            friendly_name="Weekly Winners Universe Summary",
            description="Walk-forward summary metrics by universe for the 7-day winners model.",
            grain="One row per tested universe.",
        )
    if name == "weekly_winners_summary.json":
        return ArtifactSpec(
            artifact_key="weekly_winners_summary",
            friendly_name="Weekly Winners Summary",
            description="Top-level status for the weekly winners production run, including the selected gate or the exact block reason.",
            grain="One JSON document per weekly winners run.",
        )
    if name == "stock_insider_trades.parquet":
        return ArtifactSpec(
            artifact_key="stock_insider_trades",
            friendly_name="Normalized Insider Trades",
            description="Official NSE promoter and insider transaction filings normalized to one row per filing event.",
            grain="One row per symbol per insider filing event.",
            primary_key=("symbol", "filing_id"),
        )
    if name == "stock_bulk_block_deals.parquet":
        return ArtifactSpec(
            artifact_key="stock_bulk_block_deals",
            friendly_name="Normalized Bulk And Block Deals",
            description="Official NSE bulk and block deals normalized to one row per reported client-side deal line.",
            grain="One row per symbol, trade date, client, and side.",
        )
    if name == "stock_derivatives_oi.parquet":
        return ArtifactSpec(
            artifact_key="stock_derivatives_oi",
            friendly_name="Normalized Derivatives Open Interest",
            description="Official NSE/NCL stock futures open-interest history normalized to one row per symbol per trade date.",
            grain="One row per symbol per trade date.",
            primary_key=("symbol", "trade_date"),
        )
    if name == "event_feature_daily.parquet":
        return ArtifactSpec(
            artifact_key="event_feature_daily",
            friendly_name="Daily Event-Flow Feature Table",
            description="Lag-safe daily event, insider, deal-flow, and open-interest features keyed by symbol and trade date.",
            grain="One row per symbol per trade date.",
            primary_key=("symbol", "trade_date"),
        )
    if name == "event_flow_model_comparison.csv":
        return ArtifactSpec(
            artifact_key="event_flow_model_comparison",
            friendly_name="Event-Flow Model Comparison",
            description="Side-by-side comparison of baseline versus upgraded 7-day model metrics across universes.",
            grain="One row per universe.",
        )
    if name == "event_flow_current_shortlist.csv":
        return ArtifactSpec(
            artifact_key="event_flow_current_shortlist",
            friendly_name="Current Event-Flow Shortlist",
            description="Current top-ranked 7-day candidates from the upgraded event-flow model.",
            grain="One row per shortlisted stock.",
        )
    mcap_match = re.fullmatch(r"mcap_(\d+)_(individual_counts|sequential_counts|final_shortlist|cutoff_before_.+|cutoff_after_.+)\.csv", name)
    if mcap_match:
        threshold = mcap_match.group(1)
        kind = mcap_match.group(2)
        if kind == "individual_counts":
            return ArtifactSpec(
                artifact_key="screen_individual_counts",
                friendly_name=f"Rule Counts For {threshold} Cr Screen",
                description=f"Per-rule pass counts for the {threshold} Cr market-cap variant when each rule is evaluated independently.",
                grain="One row per rule.",
                column_help={
                    "individual_pass_count": "How many stocks pass this one rule by itself.",
                    "missing_count": "How many stocks do not have enough information to evaluate this rule.",
                },
            )
        if kind == "sequential_counts":
            return ArtifactSpec(
                artifact_key="screen_sequential_counts",
                friendly_name=f"Sequential Rule Counts For {threshold} Cr Screen",
                description=f"Checklist-style survivor counts for the {threshold} Cr market-cap variant as rules are applied in order.",
                grain="One row per rule step.",
                column_help={
                    "survivors_before": "Stocks still alive before this rule is applied.",
                    "survivors_after": "Stocks still alive after this rule is applied.",
                    "rule_false_in_prior_survivors": "Previously alive stocks that failed this rule.",
                    "rule_missing_in_prior_survivors": "Previously alive stocks that could not be evaluated for this rule.",
                },
            )
        if kind == "final_shortlist":
            return ArtifactSpec(
                artifact_key="screen_final_shortlist",
                friendly_name=f"Final Shortlist For {threshold} Cr Screen",
                description=f"Stocks that pass every rule in the {threshold} Cr market-cap variant.",
                grain="One row per shortlisted stock.",
                contract_path=Path("data_contracts/daily_screen_universe.yaml"),
                primary_key=("symbol",),
            )
        if kind.startswith("cutoff_before_"):
            return ArtifactSpec(
                artifact_key="screen_cutoff_before",
                friendly_name=f"Cutoff Before File For {threshold} Cr Screen",
                description=f"Stocks still alive just before the first rule where the {threshold} Cr screen drops below 30 survivors.",
                grain="One row per stock still alive before the cutoff rule.",
                contract_path=Path("data_contracts/daily_screen_universe.yaml"),
                primary_key=("symbol",),
            )
        if kind.startswith("cutoff_after_"):
            return ArtifactSpec(
                artifact_key="screen_cutoff_after",
                friendly_name=f"Cutoff After File For {threshold} Cr Screen",
                description=f"Stocks still alive just after the first rule where the {threshold} Cr screen drops below 30 survivors.",
                grain="One row per stock still alive after the cutoff rule.",
                contract_path=Path("data_contracts/daily_screen_universe.yaml"),
                primary_key=("symbol",),
            )
    return ArtifactSpec(
        artifact_key="generic_artifact",
        friendly_name=path.stem.replace("_", " ").title(),
        description="Generated artifact.",
        grain="See matching code path and sidecar metadata.",
    )


def _load_contract_map(contract_path: Path | None) -> dict[str, Any]:
    if contract_path is None or not contract_path.exists():
        return {}
    contract = load_contract(contract_path)
    columns = {
        str(column.get("name")): {
            "definition": column.get("definition"),
            "source": column.get("source"),
            "nullable": column.get("nullable"),
            "derived_formula": column.get("derived_formula"),
            "lag_rule": column.get("lag_rule"),
        }
        for column in contract.get("columns", [])
    }
    return {"primary_key": contract.get("primary_key", []), "columns": columns}


def _profile_column(
    df: pd.DataFrame,
    column: str,
    *,
    contract: dict[str, dict[str, Any]],
    extra_help: dict[str, str],
) -> dict[str, Any]:
    series = df[column]
    non_null = int(series.notna().sum())
    null_count = int(series.isna().sum())
    sample_values = []
    for value in series.dropna().astype("string").unique()[:3]:
        sample_values.append(str(value))
    metadata = contract.get(column, {})
    definition = metadata.get("definition") or extra_help.get(column)
    return {
        "name": column,
        "dtype": str(series.dtype),
        "non_null_count": non_null,
        "null_count": null_count,
        "definition": definition,
        "source": metadata.get("source"),
        "nullable_contract": metadata.get("nullable"),
        "lag_rule": metadata.get("lag_rule"),
        "derived_formula": metadata.get("derived_formula"),
        "sample_values": sample_values,
    }
