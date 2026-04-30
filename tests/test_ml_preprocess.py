from __future__ import annotations

import pandas as pd

from src.ml.preprocess import fit_preprocess
from src.ml.preprocess import transform_frame


def test_preprocess_handles_boolean_and_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "x_num": [1.0, None, 3.0],
            "flag": [True, None, False],
        }
    )
    stats = fit_preprocess(frame, ["x_num", "flag"])
    matrix = transform_frame(frame, stats)
    assert matrix.shape == (3, 2)

