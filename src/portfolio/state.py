from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.io import write_json


DEFAULT_PORTFOLIO_STATE_DIR = Path("data/portfolio_state")
CURRENT_POSITIONS_FILE = "current_positions.csv"
EXECUTION_LEDGER_FILE = "executed_trade_ledger.csv"
WORKFLOW_SETTINGS_FILE = "workflow_settings.json"

CURRENT_POSITIONS_COLUMNS = [
    "symbol",
    "stock_name",
    "objective_name",
    "cadence_day_local",
    "cadence_time_local",
    "timezone_name",
    "entry_trade_date",
    "entry_price",
    "current_allocation_pct",
    "last_rebalance_date",
    "last_confirmed_action",
    "last_stop_loss",
    "last_sell_target",
    "last_confidence_score",
    "last_calibrated_confidence_5pct_7d",
    "last_ranking_score",
    "last_focus_score",
    "latest_reference_price",
    "latest_decision_sheet_path",
    "notes",
]

EXECUTION_LEDGER_COLUMNS = [
    "execution_id",
    "execution_date",
    "symbol",
    "stock_name",
    "objective_name",
    "action",
    "executed_price",
    "previous_allocation_pct",
    "target_allocation_pct",
    "resulting_allocation_pct",
    "decision_sheet_path",
    "confirmed_by",
    "note",
]


def _empty_current_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=CURRENT_POSITIONS_COLUMNS)


def _empty_execution_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=EXECUTION_LEDGER_COLUMNS)


def _state_readme_lines() -> list[str]:
    return [
        "# Portfolio State",
        "",
        "This folder is the durable memory for executed weekly investing decisions.",
        "",
        "## Files",
        "",
        f"- `{CURRENT_POSITIONS_FILE}`: current open positions only. One row per stock still held.",
        f"- `{EXECUTION_LEDGER_FILE}`: append-only trade confirmation ledger. One row per confirmed action.",
        f"- `{WORKFLOW_SETTINGS_FILE}`: operating cadence and workflow defaults for future runs.",
        "",
        "## How to use this folder",
        "",
        "- Open `workflow_settings.json` first to confirm the cadence and objective.",
        "- Open `current_positions.csv` to see what is currently held.",
        "- Open `executed_trade_ledger.csv` to audit all confirmed buy, add, partial sell, and full exit actions.",
        "- For each CSV or JSON, open the matching `.manifest.json` sidecar for column meanings and sample values.",
        "",
        "## Important rules",
        "",
        "- This state is updated only after the user confirms trades were placed.",
        "- Recommendations and positions are separate. A recommendation does not become a position until confirmation is recorded.",
        "- Allocation percentages are percentages of total portfolio capital, not percent change returns.",
    ]


def _write_state_readme(state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "README.md"
    path.write_text("\n".join(_state_readme_lines()) + "\n", encoding="utf-8")
    return path


def prepare_portfolio_state(
    state_dir: Path = DEFAULT_PORTFOLIO_STATE_DIR,
    *,
    objective_name: str,
    cadence_day_local: str,
    cadence_time_local: str,
    timezone_name: str,
    note: str = "",
) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    positions_path = state_dir / CURRENT_POSITIONS_FILE
    ledger_path = state_dir / EXECUTION_LEDGER_FILE
    workflow_path = state_dir / WORKFLOW_SETTINGS_FILE
    readme_path = _write_state_readme(state_dir)

    if not positions_path.exists():
        _empty_current_positions().to_csv(positions_path, index=False)
    if not ledger_path.exists():
        _empty_execution_ledger().to_csv(ledger_path, index=False)

    workflow_payload = {
        "objective_name": objective_name,
        "cadence_day_local": cadence_day_local,
        "cadence_time_local": cadence_time_local,
        "timezone_name": timezone_name,
        "note": note,
    }
    write_json(workflow_payload, workflow_path)

    write_dataframe_manifest(
        positions_path,
        load_current_positions(state_dir),
        generated_by="src.portfolio.state",
    )
    write_dataframe_manifest(
        ledger_path,
        load_execution_ledger(state_dir),
        generated_by="src.portfolio.state",
    )
    write_json_manifest(
        workflow_path,
        workflow_payload,
        generated_by="src.portfolio.state",
    )

    return {
        "state_dir": str(state_dir),
        "positions_path": str(positions_path),
        "ledger_path": str(ledger_path),
        "workflow_path": str(workflow_path),
        "readme_path": str(readme_path),
    }


def load_current_positions(state_dir: Path = DEFAULT_PORTFOLIO_STATE_DIR) -> pd.DataFrame:
    path = state_dir / CURRENT_POSITIONS_FILE
    if not path.exists():
        return _empty_current_positions()
    frame = pd.read_csv(path)
    for column in CURRENT_POSITIONS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[CURRENT_POSITIONS_COLUMNS].copy()


def load_execution_ledger(state_dir: Path = DEFAULT_PORTFOLIO_STATE_DIR) -> pd.DataFrame:
    path = state_dir / EXECUTION_LEDGER_FILE
    if not path.exists():
        return _empty_execution_ledger()
    frame = pd.read_csv(path)
    for column in EXECUTION_LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[EXECUTION_LEDGER_COLUMNS].copy()


def _save_current_positions(frame: pd.DataFrame, state_dir: Path) -> Path:
    path = state_dir / CURRENT_POSITIONS_FILE
    ordered = frame.copy()
    if ordered.empty:
        ordered = _empty_current_positions()
    else:
        ordered["current_allocation_pct"] = pd.to_numeric(ordered["current_allocation_pct"], errors="coerce")
        ordered = ordered.sort_values(["current_allocation_pct", "symbol"], ascending=[False, True])
        ordered = ordered[CURRENT_POSITIONS_COLUMNS].reset_index(drop=True)
    ordered.to_csv(path, index=False)
    write_dataframe_manifest(
        path,
        ordered,
        generated_by="src.portfolio.state",
    )
    return path


def _save_execution_ledger(frame: pd.DataFrame, state_dir: Path) -> Path:
    path = state_dir / EXECUTION_LEDGER_FILE
    ordered = frame.copy()
    if ordered.empty:
        ordered = _empty_execution_ledger()
    else:
        ordered = ordered[EXECUTION_LEDGER_COLUMNS].reset_index(drop=True)
    ordered.to_csv(path, index=False)
    write_dataframe_manifest(
        path,
        ordered,
        generated_by="src.portfolio.state",
    )
    return path


def confirm_decision_sheet_execution(
    *,
    decision_sheet_path: Path,
    state_dir: Path = DEFAULT_PORTFOLIO_STATE_DIR,
    execution_date: str | None = None,
    confirmed_by: str = "user",
    note: str = "",
) -> dict[str, Any]:
    if not decision_sheet_path.exists():
        raise FileNotFoundError(f"Decision sheet not found: {decision_sheet_path}")

    decision = pd.read_csv(decision_sheet_path)
    if decision.empty:
        raise ValueError("Decision sheet is empty; there is nothing to confirm.")
    required = {"symbol", "stock_name", "recommended_action", "recommended_allocation_pct", "current_price"}
    missing = required.difference(decision.columns)
    if missing:
        raise ValueError(f"Decision sheet is missing required columns: {sorted(missing)}")

    objective_name = str(decision.get("objective_name", pd.Series(["weekly_7d_5pct"])).iloc[0] or "weekly_7d_5pct")
    cadence_day_local = str(decision.get("cadence_day_local", pd.Series(["MONDAY"])).iloc[0] or "MONDAY")
    cadence_time_local = str(decision.get("cadence_time_local", pd.Series(["20:30"])).iloc[0] or "20:30")
    timezone_name = str(decision.get("timezone_name", pd.Series(["Asia/Kolkata"])).iloc[0] or "Asia/Kolkata")
    prepare_portfolio_state(
        state_dir,
        objective_name=objective_name,
        cadence_day_local=cadence_day_local,
        cadence_time_local=cadence_time_local,
        timezone_name=timezone_name,
    )

    execution_ts = pd.Timestamp(execution_date).normalize() if execution_date else pd.Timestamp.today().normalize()
    positions = load_current_positions(state_dir)
    ledger = load_execution_ledger(state_dir)
    positions_map = {str(row["symbol"]): row for _, row in positions.iterrows()}

    actionable = decision.loc[decision["recommended_action"].astype(str).ne("Hold")].copy()
    new_ledger_rows: list[dict[str, Any]] = []

    for _, row in actionable.iterrows():
        symbol = str(row["symbol"])
        stock_name = str(row.get("stock_name", symbol))
        action = str(row["recommended_action"])
        current_price = float(pd.to_numeric(row.get("current_price"), errors="coerce"))
        previous = positions_map.get(symbol)
        previous_alloc = float(pd.to_numeric(previous["current_allocation_pct"], errors="coerce")) if previous is not None else 0.0
        target_alloc = float(pd.to_numeric(row.get("recommended_allocation_pct"), errors="coerce"))
        resulting_alloc = target_alloc if action != "Sell Wholly" else 0.0

        if previous is None and action in {"Buy New", "Buy More"}:
            entry_price = current_price
            entry_trade_date = execution_ts.date().isoformat()
        elif previous is not None and action == "Buy More":
            old_entry = float(pd.to_numeric(previous.get("entry_price"), errors="coerce"))
            add_alloc = max(target_alloc - previous_alloc, 0.0)
            entry_price = ((old_entry * previous_alloc) + (current_price * add_alloc)) / max(target_alloc, 1e-9)
            entry_trade_date = str(previous.get("entry_trade_date", execution_ts.date().isoformat()))
        elif previous is not None:
            entry_price = float(pd.to_numeric(previous.get("entry_price"), errors="coerce"))
            entry_trade_date = str(previous.get("entry_trade_date", execution_ts.date().isoformat()))
        else:
            entry_price = current_price
            entry_trade_date = execution_ts.date().isoformat()

        new_ledger_rows.append(
            {
                "execution_id": uuid4().hex,
                "execution_date": execution_ts.date().isoformat(),
                "symbol": symbol,
                "stock_name": stock_name,
                "objective_name": objective_name,
                "action": action,
                "executed_price": round(current_price, 4),
                "previous_allocation_pct": round(previous_alloc, 4),
                "target_allocation_pct": round(target_alloc, 4),
                "resulting_allocation_pct": round(resulting_alloc, 4),
                "decision_sheet_path": str(decision_sheet_path),
                "confirmed_by": confirmed_by,
                "note": note,
            }
        )

        if action == "Sell Wholly":
            positions_map.pop(symbol, None)
            continue

        positions_map[symbol] = {
            "symbol": symbol,
            "stock_name": stock_name,
            "objective_name": objective_name,
            "cadence_day_local": cadence_day_local,
            "cadence_time_local": cadence_time_local,
            "timezone_name": timezone_name,
            "entry_trade_date": entry_trade_date,
            "entry_price": round(entry_price, 4),
            "current_allocation_pct": round(resulting_alloc, 4),
            "last_rebalance_date": execution_ts.date().isoformat(),
            "last_confirmed_action": action,
            "last_stop_loss": pd.to_numeric(row.get("stop_loss"), errors="coerce"),
            "last_sell_target": pd.to_numeric(row.get("sell_target"), errors="coerce"),
            "last_confidence_score": pd.to_numeric(row.get("confidence_score"), errors="coerce"),
            "last_calibrated_confidence_5pct_7d": pd.to_numeric(row.get("calibrated_confidence_5pct_7d"), errors="coerce"),
            "last_ranking_score": pd.to_numeric(row.get("ranking_score"), errors="coerce"),
            "last_focus_score": pd.to_numeric(row.get("focus_score"), errors="coerce"),
            "latest_reference_price": round(current_price, 4),
            "latest_decision_sheet_path": str(decision_sheet_path),
            "notes": note,
        }

    updated_positions = pd.DataFrame(list(positions_map.values()), columns=CURRENT_POSITIONS_COLUMNS)
    updated_positions = updated_positions.loc[
        pd.to_numeric(updated_positions["current_allocation_pct"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    new_ledger = pd.DataFrame(new_ledger_rows, columns=EXECUTION_LEDGER_COLUMNS)
    if ledger.empty:
        updated_ledger = new_ledger.copy()
    elif new_ledger.empty:
        updated_ledger = ledger.copy()
    else:
        updated_ledger = pd.concat([ledger, new_ledger], ignore_index=True)

    positions_path = _save_current_positions(updated_positions, state_dir)
    ledger_path = _save_execution_ledger(updated_ledger, state_dir)

    confirmation_payload = {
        "decision_sheet_path": str(decision_sheet_path),
        "execution_date": execution_ts.date().isoformat(),
        "confirmed_by": confirmed_by,
        "action_count": int(len(new_ledger_rows)),
        "open_position_count": int(len(updated_positions)),
        "positions_path": str(positions_path),
        "ledger_path": str(ledger_path),
    }
    confirmation_path = state_dir / "latest_confirmation.json"
    write_json(confirmation_payload, confirmation_path)
    write_json_manifest(
        confirmation_path,
        confirmation_payload,
        generated_by="src.portfolio.state",
        as_of_date=execution_ts.date().isoformat(),
    )
    return confirmation_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage durable portfolio state for weekly investing runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize or refresh the portfolio state folder.")
    init_parser.add_argument("--state-dir", type=Path, default=DEFAULT_PORTFOLIO_STATE_DIR)
    init_parser.add_argument("--objective-name", default="weekly_7d_5pct")
    init_parser.add_argument("--cadence-day-local", default="MONDAY")
    init_parser.add_argument("--cadence-time-local", default="20:30")
    init_parser.add_argument("--timezone-name", default="Asia/Kolkata")
    init_parser.add_argument("--note", default="")

    confirm_parser = subparsers.add_parser("confirm", help="Confirm that the actions in a decision sheet were executed.")
    confirm_parser.add_argument("--decision-sheet-path", type=Path, required=True)
    confirm_parser.add_argument("--state-dir", type=Path, default=DEFAULT_PORTFOLIO_STATE_DIR)
    confirm_parser.add_argument("--execution-date", default="")
    confirm_parser.add_argument("--confirmed-by", default="user")
    confirm_parser.add_argument("--note", default="")

    args = parser.parse_args()
    if args.command == "init":
        payload = prepare_portfolio_state(
            args.state_dir,
            objective_name=args.objective_name,
            cadence_day_local=args.cadence_day_local,
            cadence_time_local=args.cadence_time_local,
            timezone_name=args.timezone_name,
            note=args.note,
        )
    else:
        payload = confirm_decision_sheet_execution(
            decision_sheet_path=args.decision_sheet_path,
            state_dir=args.state_dir,
            execution_date=args.execution_date or None,
            confirmed_by=args.confirmed_by,
            note=args.note,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
