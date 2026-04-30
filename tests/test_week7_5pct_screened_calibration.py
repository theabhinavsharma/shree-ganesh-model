import pandas as pd

from src.analysis.week7_5pct_gbm_allnames_macro_veto import _apply_screened_calibration_5pct
from src.analysis.week7_5pct_gbm_allnames_macro_veto import _rerank_screened_population


def test_apply_screened_calibration_maps_distinct_bins() -> None:
    calibration = pd.DataFrame(
        [
            {"calibration_bin": 0, "score_min": 0.10, "score_max": 0.20, "count": 10, "hit_rate": 0.08, "avg_return": 0.002},
            {"calibration_bin": 1, "score_min": 0.20, "score_max": 0.30, "count": 10, "hit_rate": 0.12, "avg_return": 0.004},
            {"calibration_bin": 2, "score_min": 0.30, "score_max": 0.40, "count": 10, "hit_rate": 0.20, "avg_return": 0.010},
        ]
    )
    frame = pd.DataFrame(
        [
            {"symbol": "A", "focus_score": 0.12},
            {"symbol": "B", "focus_score": 0.24},
            {"symbol": "C", "focus_score": 0.35},
        ]
    )

    out = _apply_screened_calibration_5pct(frame, calibration, score_col="focus_score")

    assert out["screened_calibration_bin"].tolist() == [0, 1, 2]
    assert out["screened_calibrated_confidence_5pct_7d"].tolist() == [0.08, 0.12, 0.20]


def test_rerank_screened_population_prefers_screened_calibration_then_score() -> None:
    frame = pd.DataFrame(
        [
            {
                "trade_date": "2026-04-24",
                "symbol": "B",
                "focus_score": 0.40,
                "screened_calibrated_confidence_5pct_7d": 0.10,
                "screened_calibrated_avg_return_7d": 0.004,
            },
            {
                "trade_date": "2026-04-24",
                "symbol": "A",
                "focus_score": 0.30,
                "screened_calibrated_confidence_5pct_7d": 0.15,
                "screened_calibrated_avg_return_7d": 0.003,
            },
            {
                "trade_date": "2026-04-24",
                "symbol": "C",
                "focus_score": 0.35,
                "screened_calibrated_confidence_5pct_7d": 0.15,
                "screened_calibrated_avg_return_7d": 0.005,
            },
        ]
    )

    out = _rerank_screened_population(frame, rank_col="post_veto_rank")
    ordered = out.sort_values("post_veto_rank")["symbol"].tolist()

    assert ordered == ["C", "A", "B"]
