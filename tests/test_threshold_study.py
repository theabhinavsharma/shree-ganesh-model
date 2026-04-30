from __future__ import annotations

from datetime import date

import pandas as pd

from src.analysis.threshold_study import _select_numeric_threshold
from src.analysis.threshold_study import build_forward_1y_labels


def test_build_forward_1y_labels_uses_next_available_trade_date() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC", "ABC"],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-02"]),
            "close": [100.0, 110.0, 160.0],
        }
    )
    labeled = build_forward_1y_labels(
        df,
        analysis_start_date=date(2024, 1, 1),
        analysis_end_date=date(2024, 1, 1),
        horizon_days=365,
        min_price=None,
    )
    row = labeled.iloc[0]
    assert row["forward_trade_date"] == pd.Timestamp("2025-01-02")
    assert round(float(row["forward_1y_return"]), 4) == 0.6
    assert bool(row["winner_1y_50_flag"]) is True


def test_select_numeric_threshold_returns_candidate_with_min_coverage() -> None:
    valid = pd.DataFrame(
        {
            "volume_vs_20d": [1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0],
            "winner_1y_50_flag": [False, False, False, False, True, True, True, True, True, True],
        }
    )
    selected = _select_numeric_threshold(valid, column="volume_vs_20d", direction="ge", baseline=0.6)
    assert selected["threshold"] is not None
    assert float(selected["coverage"]) >= 0.05
    assert float(selected["precision"]) >= 0.6
