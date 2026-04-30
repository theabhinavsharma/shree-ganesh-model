from __future__ import annotations

import pandas as pd

from src.analysis.day1_5pct_model import select_best_universe
from src.analysis.day1_5pct_model import summarize_universes


def test_select_best_universe_honors_tradable_filter() -> None:
    universe_metrics = pd.DataFrame(
        {
            "universe_name": ["cheap_micro", "liquid_20cr_plus", "mid_small"],
            "selection_rank_score": [0.90, 0.80, 0.70],
            "top10_precision_5pct": [0.25, 0.20, 0.18],
            "top_bucket_hit_rate": [0.30, 0.24, 0.23],
            "top10_mean_return_mean": [0.02, 0.01, 0.005],
            "top10_median_stock_return_median": [0.01, 0.004, 0.002],
        }
    )
    assert select_best_universe(universe_metrics, tradable_only=False) == "cheap_micro"
    assert select_best_universe(universe_metrics, tradable_only=True) == "liquid_20cr_plus"


def test_summarize_universes_reports_top10_and_base_returns() -> None:
    predictions = pd.DataFrame(
        {
            "universe_name": ["all_names"] * 4,
            "trade_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "focus_score": [0.90, 0.10, 0.80, 0.20],
            "winner_5pct": [1, 0, 0, 1],
            "forward_return": [0.06, -0.02, 0.01, 0.07],
        }
    )
    summary_df = pd.DataFrame(
        {
            "universe_name": ["all_names"],
            "top_quantile_precision_5pct": [0.50],
            "top_n_precision_5pct": [0.50],
        }
    )
    metrics = summarize_universes(predictions, summary_df, top_quantile=0.50, top_n=1)
    row = metrics.iloc[0]
    assert row["universe_name"] == "all_names"
    assert abs(row["base_rate_5pct"] - 0.50) < 1e-9
    assert abs(row["top10_precision_5pct"] - 0.50) < 1e-9
    assert abs(row["top10_mean_return_mean"] - 0.035) < 1e-9
    assert abs(row["base_avg_return"] - 0.03) < 1e-9
