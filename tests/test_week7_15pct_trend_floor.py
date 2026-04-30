from __future__ import annotations

import pandas as pd

from src.analysis.week7_15pct_random_forest_allnames import FreshEntryRule
from src.analysis.week7_15pct_random_forest_allnames import _build_veto_columns


def test_trend_floor_rejects_missing_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "MISS",
                "close": 100.0,
                "sma_50": 95.0,
                "sma_200": 90.0,
                "return_7td": 0.05,
                "return_15td": None,
                "return_20d": 0.10,
                "return_30td": 0.04,
                "rsi_14_daily": 58.0,
            }
        ]
    )
    out = _build_veto_columns(frame, rule=FreshEntryRule())
    assert bool(out.loc[0, "fresh_entry_pass"]) is False
    assert out.loc[0, "veto_note"] == "missing fresh-entry metrics"


def test_trend_floor_rejects_weak_structure() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "WEAK",
                "close": 90.0,
                "sma_50": 100.0,
                "sma_200": 95.0,
                "return_7td": 0.03,
                "return_15td": -0.12,
                "return_20d": 0.01,
                "return_30td": -0.08,
                "rsi_14_daily": 44.0,
            }
        ]
    )
    out = _build_veto_columns(frame, rule=FreshEntryRule())
    assert bool(out.loc[0, "fresh_entry_pass"]) is False
    assert out.loc[0, "veto_note"] == "below 50 DMA"


def test_trend_floor_accepts_healthy_candidate() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "GOOD",
                "close": 120.0,
                "sma_50": 110.0,
                "sma_200": 100.0,
                "return_7td": 0.08,
                "return_15td": 0.10,
                "return_20d": 0.15,
                "return_30td": 0.04,
                "rsi_14_daily": 61.0,
            }
        ]
    )
    out = _build_veto_columns(frame, rule=FreshEntryRule())
    assert bool(out.loc[0, "fresh_entry_pass"]) is True
    assert out.loc[0, "veto_note"] == "pass"
