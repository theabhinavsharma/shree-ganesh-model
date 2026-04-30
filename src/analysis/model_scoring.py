from __future__ import annotations

import math

import numpy as np
import pandas as pd


def apply_feature_score(
    frame: pd.DataFrame,
    feature_results: pd.DataFrame,
    *,
    min_test_lift: float = 1.0,
) -> pd.DataFrame:
    scored = frame.copy()
    score = np.zeros(len(scored), dtype=float)
    pass_count = np.zeros(len(scored), dtype=int)

    selected = feature_results.copy()
    selected = selected[selected["test_lift"].notna() & selected["test_lift"].ge(min_test_lift)].copy()

    for _, feature in selected.iterrows():
        column = str(feature["column"])
        if column not in scored.columns:
            continue
        lift = float(feature["test_lift"])
        passed = _evaluate_rule(scored[column], feature_type=str(feature["feature_type"]), direction=str(feature["direction"]), threshold=feature["selected_threshold"])
        score += passed.astype(float).to_numpy() * math.log(lift)
        pass_count += passed.astype(int).to_numpy()

    scored["model_score"] = score
    scored["model_pass_count"] = pass_count
    return scored


def _evaluate_rule(series: pd.Series, *, feature_type: str, direction: str, threshold: object) -> pd.Series:
    if feature_type == "boolean":
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if pd.isna(threshold):
        return pd.Series(False, index=series.index)
    if direction == "le":
        return numeric.le(float(threshold)).fillna(False)
    return numeric.ge(float(threshold)).fillna(False)
