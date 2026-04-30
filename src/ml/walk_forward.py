from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_name: str
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def build_yearly_walk_forward_folds(frame: pd.DataFrame, *, min_train_end_year: int = 2019) -> list[WalkForwardFold]:
    trade_dates = pd.to_datetime(frame["trade_date"]).dt.normalize()
    years = sorted(int(value) for value in trade_dates.dt.year.dropna().unique())
    folds: list[WalkForwardFold] = []
    for year in years:
        if year <= min_train_end_year:
            continue
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year, month=12, day=31)
        if not trade_dates.between(test_start, test_end).any():
            continue
        train_end = test_start - pd.Timedelta(days=1)
        folds.append(
            WalkForwardFold(
                fold_name=f"{year}",
                train_end_date=train_end,
                test_start_date=test_start,
                test_end_date=test_end,
            )
        )
    return folds

