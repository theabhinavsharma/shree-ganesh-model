from __future__ import annotations

import numpy as np
import pandas as pd


def build_universe_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    price = _numeric_series(frame, "close")
    liq = _numeric_series(frame, "avg_traded_value_20d_cr")
    market_cap = _numeric_series(frame, "market_cap_cr")
    return {
        "all_names": pd.Series(True, index=frame.index, dtype=bool),
        "cheap_micro": price.lt(50) & liq.lt(1),
        "mid_small": price.ge(50) & price.lt(200) & liq.ge(1) & liq.lt(5),
        "liquid_5cr_plus": liq.ge(5),
        "liquid_20cr_plus": liq.ge(20),
        "mcap_1000cr_plus": market_cap.ge(1000),
    }


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")
