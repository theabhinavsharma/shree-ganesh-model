from __future__ import annotations

import pandas as pd


def latest_effective_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_date_col: str,
    right_date_col: str,
    by: str,
) -> pd.DataFrame:
    if right.empty:
        return left.copy()
    left_clean = left[left[left_date_col].notna() & left[by].notna()].copy()
    left_missing = left[~(left[left_date_col].notna() & left[by].notna())].copy()
    right_sorted = right[right[right_date_col].notna() & right[by].notna()].copy()
    if right_sorted.empty:
        return left.copy()
    left_sorted = left_clean.sort_values([left_date_col, by]).copy()
    right_sorted = right_sorted.sort_values([right_date_col, by]).copy()
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_date_col,
        right_on=right_date_col,
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    if left_missing.empty:
        return merged
    for column in merged.columns:
        if column not in left_missing.columns:
            left_missing[column] = pd.NA
    combined = pd.concat([merged, left_missing[merged.columns]], ignore_index=True)
    return combined.sort_values([left_date_col, by], na_position="last").reset_index(drop=True)
