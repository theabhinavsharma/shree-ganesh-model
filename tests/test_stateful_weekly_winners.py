from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.report.stateful_weekly_winners import generate_stateful_weekly_decision_sheet


def test_generate_stateful_weekly_decision_sheet_marks_new_buys(tmp_path: Path) -> None:
    shortlist = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-03-30", "2026-03-30"]),
            "symbol": ["AAA", "BBB"],
            "company_name": ["Alpha", "Beta"],
            "sector": ["Tech", "Chemicals"],
            "industry": ["Software", "Specialty Chemicals"],
            "basic_industry": ["Software", "Chemicals"],
            "close": [100.0, 200.0],
            "avg_traded_value_20d_cr": [25.0, 6.0],
            "volume_vs_20d": [2.0, 1.7],
            "pred_return_7d": [0.06, 0.05],
            "pred_price_7d": [106.0, 210.0],
            "focus_score": [0.6, 0.5],
            "ranking_score": [0.62, 0.51],
            "calibrated_confidence_5pct_7d": [0.28, 0.24],
            "prob_up_7d": [0.62, 0.55],
            "prob_5pct_7d": [0.44, 0.41],
            "prob_10pct_7d": [0.18, 0.14],
            "shortlist_rank": [1, 2],
        }
    )
    live_market_frame = shortlist.copy()

    artifacts = generate_stateful_weekly_decision_sheet(
        shortlist=shortlist,
        live_market_frame=live_market_frame,
        output_dir=tmp_path / "run",
        state_dir=tmp_path / "state",
        as_of_trade_date="2026-03-30",
    )
    report = artifacts.report_frame
    assert set(report["recommended_action"]) == {"Buy New"}
    assert len(report) == 2


def test_generate_stateful_weekly_decision_sheet_marks_sell_wholly_for_dropped_position(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    positions = pd.DataFrame(
        {
            "symbol": ["OLD"],
            "stock_name": ["Old Name"],
            "objective_name": ["weekly_7d_5pct"],
            "cadence_day_local": ["MONDAY"],
            "cadence_time_local": ["20:30"],
            "timezone_name": ["Asia/Kolkata"],
            "entry_trade_date": ["2026-03-23"],
            "entry_price": [100.0],
            "current_allocation_pct": [10.0],
            "last_rebalance_date": ["2026-03-23"],
            "last_confirmed_action": ["Buy New"],
            "last_stop_loss": [94.0],
            "last_sell_target": [107.0],
            "last_confidence_score": [80.0],
            "last_calibrated_confidence_5pct_7d": [0.25],
            "last_ranking_score": [0.5],
            "last_focus_score": [0.48],
            "latest_reference_price": [100.0],
            "latest_decision_sheet_path": ["seed.csv"],
            "notes": [""],
        }
    )
    positions.to_csv(state_dir / "current_positions.csv", index=False)

    shortlist = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-03-30"]),
            "symbol": ["AAA"],
            "company_name": ["Alpha"],
            "sector": ["Tech"],
            "industry": ["Software"],
            "basic_industry": ["Software"],
            "close": [100.0],
            "avg_traded_value_20d_cr": [25.0],
            "volume_vs_20d": [2.0],
            "pred_return_7d": [0.06],
            "pred_price_7d": [106.0],
            "focus_score": [0.6],
            "ranking_score": [0.62],
            "calibrated_confidence_5pct_7d": [0.28],
            "prob_up_7d": [0.62],
            "prob_5pct_7d": [0.44],
            "prob_10pct_7d": [0.18],
            "shortlist_rank": [1],
        }
    )
    live_market_frame = pd.concat(
        [
            shortlist.copy(),
            pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(["2026-03-30"]),
                    "symbol": ["OLD"],
                    "company_name": ["Old Name"],
                    "sector": ["Legacy"],
                    "industry": ["Legacy"],
                    "basic_industry": ["Legacy"],
                    "close": [98.0],
                }
            ),
        ],
        ignore_index=True,
    )

    artifacts = generate_stateful_weekly_decision_sheet(
        shortlist=shortlist,
        live_market_frame=live_market_frame,
        output_dir=tmp_path / "run",
        state_dir=state_dir,
        as_of_trade_date="2026-03-30",
    )
    report = artifacts.report_frame
    old_row = report.loc[report["symbol"] == "OLD"].iloc[0]
    assert old_row["recommended_action"] == "Sell Wholly"
