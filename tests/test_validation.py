import pandas as pd
import pytest

from src.utils.validation import assert_unique_key


def test_duplicate_key_detection() -> None:
    df = pd.DataFrame({"trade_date": ["2024-01-01", "2024-01-01"], "symbol": ["A", "A"]})
    with pytest.raises(AssertionError):
        assert_unique_key(df, ["trade_date", "symbol"])
