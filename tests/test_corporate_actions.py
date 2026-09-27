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


def test_parse_split_factor_abbreviated_pre2018_format():
    """2026-09-19: NSE's old 'Fv Splt Frm Rs 10 To Re 1' wording has no 'split' token."""
    assert _parse_split_factor("Fv Splt Frm Rs 10 To Re 1") == 10.0
    assert _parse_split_factor("Fv Splt Frm Rs 10 To Rs 2") == 5.0
    assert _parse_split_factor("Fv Splt Frm Rs 5 To Re 1") == 5.0
    assert _parse_split_factor("Face Value Split From Rs 10 To Re 1") == 10.0


def test_price_only_factor_moves_price_not_quantity():
    import pandas as pd
    from src.transform.corporate_actions import apply_split_bonus_adjustments
    df = pd.DataFrame({"symbol": ["X"] * 3, "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                       "close": [100.0, 100.0, 30.0], "total_traded_qty": [10.0, 10.0, 10.0]})
    po = pd.DataFrame({"symbol": ["X"], "ex_date": [pd.Timestamp("2020-01-03")], "price_factor": [0.3]})
    out = apply_split_bonus_adjustments(df, pd.DataFrame(), price_only=po)
    assert list(out["close"].round(6)) == [30.0, 30.0, 30.0]
    assert list(out["total_traded_qty"]) == [10.0, 10.0, 10.0]
    assert list(out["price_adjustment_factor_to_present"]) == [0.3, 0.3, 1.0]


def test_empty_price_only_is_split_bonus_only():
    import pandas as pd
    from src.transform.corporate_actions import apply_split_bonus_adjustments
    df = pd.DataFrame({"symbol": ["X"] * 2, "trade_date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                       "close": [100.0, 50.0], "total_traded_qty": [10.0, 20.0]})
    ca = pd.DataFrame({"symbol": ["X"], "ex_date": [pd.Timestamp("2020-01-03")], "adjustment_factor": [2.0]})
    out = apply_split_bonus_adjustments(df, ca, price_only=pd.DataFrame(columns=["symbol", "ex_date", "price_factor"]))
    assert list(out["close"]) == [50.0, 50.0] and list(out["total_traded_qty"]) == [20.0, 20.0]


def test_expected_price_factor_combines_split_and_price_only():
    import numpy as np, pandas as pd
    from src.transform.corporate_actions import expected_price_factor
    td = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]).to_numpy(dtype="datetime64[ns]")
    sa = pd.DataFrame({"ex_date": [pd.Timestamp("2020-01-15")], "adjustment_factor": [2.0]})
    po = pd.DataFrame({"ex_date": [pd.Timestamp("2020-02-15")], "price_factor": [0.5]})
    assert np.allclose(expected_price_factor(td, sa, po), [0.25, 0.5, 1.0])
