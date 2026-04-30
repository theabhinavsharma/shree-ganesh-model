from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.weekly_run_gate_search import _weekly_success
from src.analysis.weekly_run_gate_search import _gate_columns_available
from src.report.production_weekly_winners import _gate_is_active
from src.report.production_weekly_winners import _resolve_selection_universes
from src.report.production_weekly_winners import load_weekly_winners_config
from src.report.production_weekly_winners import validate_weekly_winner_shortlist


def test_gate_is_active_handles_numeric_and_boolean_clauses() -> None:
    regime = {
        "breadth_volume_1_5x": 0.21,
        "market_median_return_20d": 0.03,
        "macro_risk_on_flag": True,
    }
    assert _gate_is_active("breadth_volume_1_5x>=0.15", regime)
    assert _gate_is_active("breadth_volume_1_5x>=0.15 & market_median_return_20d>=0.0", regime)
    assert _gate_is_active("macro_risk_on_flag=True", regime)
    assert not _gate_is_active("breadth_volume_1_5x>=0.25", regime)
    assert not _gate_is_active("macro_risk_on_flag=True & market_median_return_20d>=0.05", regime)


def test_validate_weekly_winner_shortlist_catches_duplicates_and_sorting() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "trade_date": pd.to_datetime(["2026-03-25", "2026-03-25"]),
            "close": [100.0, 90.0],
            "prob_5pct_7d": [0.7, 0.6],
            "focus_score": [0.5, 0.8],
            "calibrated_confidence_5pct_7d": [0.22, 0.19],
            "pred_return_7d": [0.03, 0.04],
        }
    )
    checks = validate_weekly_winner_shortlist(frame, expected_count=2)
    failed = {check.name for check in checks if not check.passed}
    assert "shortlist_duplicate_symbols" in failed
    assert "shortlist_sorted_by_focus_score" in failed


def test_load_weekly_winners_config_reads_defaults() -> None:
    cfg = load_weekly_winners_config(Path("configs/ml_weekly_production.yaml"))
    assert cfg.top_n == 12
    assert cfg.objective_min_winners == 2
    assert cfg.min_test_success_rate >= 0.60
    assert cfg.selection_universes[:2] == ("liquid_20cr_plus", "liquid_5cr_plus")


def test_weekly_success_falls_back_to_generic_pred_return_column() -> None:
    frame = pd.DataFrame(
        {
            "run_week": ["2025-10", "2025-10", "2025-11"],
            "focus_score": [0.9, 0.8, 0.7],
            "prob_10pct": [0.4, 0.3, 0.2],
            "pred_return": [0.02, 0.01, 0.03],
            "winner_5pct": [1, 0, 1],
        }
    )
    weekly = _weekly_success(frame, top_n=1, min_winners=1)
    assert weekly["winner_count"].tolist() == [1, 1]


def test_gate_columns_available_detects_missing_gate_fields() -> None:
    frame = pd.DataFrame({"breadth_volume_1_5x": [0.2]})
    assert _gate_columns_available(frame, (("breadth_volume_1_5x", "num", 0.15),))
    assert not _gate_columns_available(frame, (("macro_risk_on_flag", "bool", True),))


def test_resolve_selection_universes_preserves_order_and_filters_unknowns() -> None:
    cfg = load_weekly_winners_config(Path("configs/ml_weekly_production.yaml"))

    class DummyBaseConfig:
        universes = ("liquid_20cr_plus", "liquid_5cr_plus", "mid_small", "mcap_1000cr_plus", "all_names")

    class DummyExpertConfig:
        base_config = DummyBaseConfig()

    custom = cfg.__class__(
        expert_config_path=cfg.expert_config_path,
        run_output_dir=cfg.run_output_dir,
        portfolio_state_dir=cfg.portfolio_state_dir,
        selection_universes=("liquid_20cr_plus", "unknown", "mid_small", "liquid_20cr_plus"),
        top_n=cfg.top_n,
        objective_min_winners=cfg.objective_min_winners,
        min_search_weeks=cfg.min_search_weeks,
        min_test_weeks=cfg.min_test_weeks,
        min_search_success_rate=cfg.min_search_success_rate,
        min_test_success_rate=cfg.min_test_success_rate,
        min_all_success_rate=cfg.min_all_success_rate,
        cash_buffer_pct=cfg.cash_buffer_pct,
        cadence_day_local=cfg.cadence_day_local,
        cadence_time_local=cfg.cadence_time_local,
        timezone_name=cfg.timezone_name,
        search_years=cfg.search_years,
        test_years=cfg.test_years,
    )
    assert _resolve_selection_universes(custom, DummyExpertConfig()) == ("liquid_20cr_plus", "mid_small")
