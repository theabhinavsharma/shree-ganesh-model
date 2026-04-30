from __future__ import annotations

import pandas as pd

from src.ml.expert_pipeline import _apply_calibration
from src.ml.expert_pipeline import _build_calibration_table
from src.ml.expert_pipeline import _finalize_shortlist
from src.ml.expert_pipeline import FOCUS_OOF_CONTEXT_COLUMNS
from src.ml.expert_pipeline import _wilson_interval


def test_wilson_interval_bounds_are_ordered() -> None:
    low, high = _wilson_interval(12, 40)
    assert low is not None
    assert high is not None
    assert 0.0 <= low <= high <= 1.0


def test_calibration_table_and_mapping_work() -> None:
    frame = pd.DataFrame(
        {
            "focus_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "winner_5pct": [0, 0, 1, 0, 1, 1],
            "forward_return": [-0.02, 0.00, 0.07, -0.01, 0.08, 0.12],
        }
    )
    calibration = _build_calibration_table(
        frame,
        score_col="focus_score",
        target_col="winner_5pct",
        return_col="forward_return",
        bins=3,
    )
    scored = _apply_calibration(
        pd.DataFrame({"focus_score": [0.15, 0.35, 0.55]}),
        calibration,
        score_col="focus_score",
    )
    assert scored["calibrated_confidence_5pct_7d"].notna().all()
    assert scored["calibrated_avg_return_7d"].notna().all()


def test_finalize_shortlist_prefers_target_zone_and_score() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-03-25"] * 3),
            "symbol": ["AAA", "BBB", "CCC"],
            "close": [100.0, 120.0, 150.0],
            "calibrated_confidence_5pct_7d": [0.30, 0.40, 0.20],
            "prob_10pct_7d": [0.10, 0.15, 0.05],
            "pred_return_7d": [0.06, 0.12, 0.08],
            "prob_up_day_1": [0.55, 0.60, 0.52],
            "prob_up_day_15": [0.58, 0.63, 0.56],
        }
    )
    ranked = _finalize_shortlist(frame, shortlist_size=3)
    assert ranked.iloc[0]["symbol"] == "AAA"
    assert ranked["shortlist_rank"].tolist() == [1, 2, 3]


def test_focus_oof_context_columns_include_gate_features() -> None:
    required = {
        "macro_risk_on_flag",
        "macro_vix_below_20",
        "breadth_above_50_dma",
        "breadth_above_200_dma",
        "breadth_volume_1_5x",
        "market_median_return_20d",
        "nifty_50_return_20d",
        "nifty_500_return_20d",
    }
    assert required.issubset(set(FOCUS_OOF_CONTEXT_COLUMNS))
