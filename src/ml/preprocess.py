from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml.feature_registry import BOOLEAN_FEATURE_COLUMNS


@dataclass(frozen=True)
class PreprocessStats:
    feature_columns: list[str]
    numeric_columns: list[str]
    boolean_columns: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]


def fit_preprocess(frame: pd.DataFrame, feature_columns: list[str]) -> PreprocessStats:
    numeric_columns = [column for column in feature_columns if column not in BOOLEAN_FEATURE_COLUMNS]
    boolean_columns = [column for column in feature_columns if column in BOOLEAN_FEATURE_COLUMNS]
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in numeric_columns:
        series = pd.to_numeric(frame.get(column), errors="coerce")
        median = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(median)
        std = float(filled.std(ddof=0))
        medians[column] = median
        means[column] = float(filled.mean())
        stds[column] = std if std > 1e-8 else 1.0
    return PreprocessStats(
        feature_columns=list(feature_columns),
        numeric_columns=numeric_columns,
        boolean_columns=boolean_columns,
        medians=medians,
        means=means,
        stds=stds,
    )


def transform_frame(frame: pd.DataFrame, stats: PreprocessStats) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for column in stats.numeric_columns:
        series = pd.to_numeric(frame.get(column), errors="coerce").fillna(stats.medians[column])
        values = ((series.to_numpy(dtype=np.float32) - stats.means[column]) / stats.stds[column]).reshape(-1, 1)
        matrices.append(values)
    for column in stats.boolean_columns:
        series = frame.get(column)
        if series is None:
            values = np.zeros((len(frame), 1), dtype=np.float32)
        else:
            clean = pd.Series(series, index=frame.index).astype("boolean").fillna(False)
            values = clean.astype(np.float32).to_numpy().reshape(-1, 1)
        matrices.append(values)
    if not matrices:
        return np.zeros((len(frame), 0), dtype=np.float32)
    return np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
