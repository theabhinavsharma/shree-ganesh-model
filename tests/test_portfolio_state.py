from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.portfolio.state import confirm_decision_sheet_execution
from src.portfolio.state import load_current_positions
from src.portfolio.state import load_execution_ledger
from src.portfolio.state import prepare_portfolio_state


def test_confirm_decision_sheet_updates_positions_and_ledger(tmp_path: Path) -> None:
    state_dir = tmp_path / "portfolio_state"
    prepare_portfolio_state(
        state_dir,
        objective_name="weekly_7d_5pct",
        cadence_day_local="MONDAY",
        cadence_time_local="20:30",
        timezone_name="Asia/Kolkata",
    )
    decision = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "stock_name": ["Alpha", "Beta"],
            "recommended_action": ["Buy New", "Buy New"],
            "recommended_allocation_pct": [12.0, 8.0],
            "current_price": [100.0, 200.0],
            "stop_loss": [94.0, 188.0],
            "sell_target": [107.0, 212.0],
            "confidence_score": [82.0, 79.0],
            "calibrated_confidence_5pct_7d": [0.28, 0.25],
            "ranking_score": [0.6, 0.5],
            "focus_score": [0.55, 0.49],
            "objective_name": ["weekly_7d_5pct", "weekly_7d_5pct"],
            "cadence_day_local": ["MONDAY", "MONDAY"],
            "cadence_time_local": ["20:30", "20:30"],
            "timezone_name": ["Asia/Kolkata", "Asia/Kolkata"],
        }
    )
    decision_path = tmp_path / "decision.csv"
    decision.to_csv(decision_path, index=False)

    payload = confirm_decision_sheet_execution(
        decision_sheet_path=decision_path,
        state_dir=state_dir,
        execution_date="2026-03-30",
    )

    positions = load_current_positions(state_dir)
    ledger = load_execution_ledger(state_dir)
    assert payload["action_count"] == 2
    assert set(positions["symbol"]) == {"AAA", "BBB"}
    assert len(ledger) == 2


def test_confirm_decision_sheet_sells_wholly(tmp_path: Path) -> None:
    state_dir = tmp_path / "portfolio_state"
    prepare_portfolio_state(
        state_dir,
        objective_name="weekly_7d_5pct",
        cadence_day_local="MONDAY",
        cadence_time_local="20:30",
        timezone_name="Asia/Kolkata",
    )
    positions = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "stock_name": ["Alpha"],
            "objective_name": ["weekly_7d_5pct"],
            "cadence_day_local": ["MONDAY"],
            "cadence_time_local": ["20:30"],
            "timezone_name": ["Asia/Kolkata"],
            "entry_trade_date": ["2026-03-23"],
            "entry_price": [100.0],
            "current_allocation_pct": [12.0],
            "last_rebalance_date": ["2026-03-23"],
            "last_confirmed_action": ["Buy New"],
            "last_stop_loss": [94.0],
            "last_sell_target": [107.0],
            "last_confidence_score": [80.0],
            "last_calibrated_confidence_5pct_7d": [0.27],
            "last_ranking_score": [0.58],
            "last_focus_score": [0.54],
            "latest_reference_price": [100.0],
            "latest_decision_sheet_path": ["seed.csv"],
            "notes": [""],
        }
    )
    positions.to_csv(state_dir / "current_positions.csv", index=False)

    decision = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "stock_name": ["Alpha"],
            "recommended_action": ["Sell Wholly"],
            "recommended_allocation_pct": [0.0],
            "current_price": [95.0],
            "objective_name": ["weekly_7d_5pct"],
            "cadence_day_local": ["MONDAY"],
            "cadence_time_local": ["20:30"],
            "timezone_name": ["Asia/Kolkata"],
        }
    )
    decision_path = tmp_path / "decision_sell.csv"
    decision.to_csv(decision_path, index=False)

    confirm_decision_sheet_execution(
        decision_sheet_path=decision_path,
        state_dir=state_dir,
        execution_date="2026-03-30",
    )
    updated_positions = load_current_positions(state_dir)
    assert updated_positions.empty
