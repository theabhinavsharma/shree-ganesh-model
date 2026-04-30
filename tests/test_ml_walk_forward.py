from __future__ import annotations

import pandas as pd

from src.ml.walk_forward import build_yearly_walk_forward_folds


def test_walk_forward_folds_do_not_leak() -> None:
    frame = pd.DataFrame({"trade_date": pd.to_datetime(["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"])})
    folds = build_yearly_walk_forward_folds(frame, min_train_end_year=2018)
    assert folds
    for fold in folds:
        assert fold.train_end_date < fold.test_start_date

