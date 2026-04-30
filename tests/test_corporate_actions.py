import pandas as pd

from src.ingest.corporate_actions.nse import _parse_bonus_factor
from src.ingest.corporate_actions.nse import _parse_split_factor
from src.transform.corporate_actions import apply_split_bonus_adjustments


def test_parse_bonus_and_split_factors() -> None:
    assert _parse_bonus_factor("Bonus 1:1 / Face Value Split From Rs 10/- Per Share To Rs 2/- Per Share") == 2.0
    assert _parse_split_factor("Bonus 1:1 / Face Value Split From Rs 10/- Per Share To Rs 2/- Per Share") == 5.0
    assert _parse_bonus_factor("Final Dividend") is None


def test_apply_split_bonus_adjustments_uses_future_ex_dates_only() -> None:
    daily = pd.DataFrame(
        {
            "symbol": ["ABC"] * 4,
            "trade_date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-07-01", "2021-01-01"]),
            "open": [100.0, 120.0, 60.0, 80.0],
            "high": [100.0, 120.0, 60.0, 80.0],
            "low": [100.0, 120.0, 60.0, 80.0],
            "last_price": [100.0, 120.0, 60.0, 80.0],
            "close": [100.0, 120.0, 60.0, 80.0],
            "avg_price": [100.0, 120.0, 60.0, 80.0],
            "prev_close": [99.0, 119.0, 59.0, 79.0],
            "total_traded_qty": [10.0, 20.0, 30.0, 40.0],
            "deliverable_qty": [5.0, 10.0, 15.0, 20.0],
        }
    )
    actions = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC"],
            "ex_date": pd.to_datetime(["2020-07-01", "2021-01-01"]),
            "adjustment_factor": [2.0, 5.0],
        }
    )
    result = apply_split_bonus_adjustments(daily, actions)
    first = result.iloc[0]
    on_first_ex_date = result.iloc[2]
    on_second_ex_date = result.iloc[3]

    assert first["price_adjustment_factor_to_present"] == 0.1
    assert first["share_adjustment_factor_to_present"] == 10.0
    assert first["close"] == 10.0
    assert first["total_traded_qty"] == 100.0
    assert on_first_ex_date["price_adjustment_factor_to_present"] == 0.2
    assert on_first_ex_date["close"] == 12.0
    assert on_second_ex_date["price_adjustment_factor_to_present"] == 1.0
    assert on_second_ex_date["close"] == 80.0
