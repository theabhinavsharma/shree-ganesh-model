from __future__ import annotations

import pandas as pd


def assert_unique_key(df: pd.DataFrame, key_columns: list[str]) -> None:
    if df.duplicated(key_columns).any():
        duplicate_rows = df[df.duplicated(key_columns, keep=False)]
        raise AssertionError(f"Duplicate primary keys found for {key_columns}: {duplicate_rows.to_dict(orient='records')}")


def assert_no_future_leakage(df: pd.DataFrame, left_date_col: str, effective_date_col: str) -> None:
    invalid = df[df[effective_date_col].notna() & (df[effective_date_col] > df[left_date_col])]
    if not invalid.empty:
        raise AssertionError("Future leakage detected in lagged join.")
