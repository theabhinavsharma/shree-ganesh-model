import pandas as pd

from src.transform.lagged_join import latest_effective_join
from src.utils.validation import assert_no_future_leakage


def test_latest_effective_join_is_backward_only() -> None:
    left = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "trade_date": pd.to_datetime(["2024-01-10", "2024-01-20"]),
        }
    )
    right = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "effective_from_date": pd.to_datetime(["2024-01-15", "2024-01-18"]),
            "value": [1, 2],
        }
    )
    result = latest_effective_join(
        left,
        right,
        left_date_col="trade_date",
        right_date_col="effective_from_date",
        by="symbol",
    )
    assert pd.isna(result.iloc[0]["value"])
    assert result.iloc[1]["value"] == 2
    assert_no_future_leakage(result, "trade_date", "effective_from_date")
