from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    clipped = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(y_prob).rank(method="average").to_numpy(dtype=float)
    pos_rank_sum = float(ranks[y_true.astype(bool)].sum())
    auc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def daily_top_quantile_metrics(frame: pd.DataFrame, *, score_col: str, winner_col: str, return_col: str, top_quantile: float) -> dict[str, float | None]:
    precisions: list[float] = []
    avg_returns: list[float] = []
    med_returns: list[float] = []
    for _, group in frame.groupby("trade_date", sort=False):
        ordered = group.sort_values(score_col, ascending=False)
        top_n = max(1, int(np.ceil(len(ordered) * top_quantile)))
        top = ordered.head(top_n)
        winners = pd.to_numeric(top[winner_col], errors="coerce").fillna(0.0)
        returns = pd.to_numeric(top[return_col], errors="coerce").dropna()
        if not winners.empty:
            precisions.append(float(winners.mean()))
        if not returns.empty:
            avg_returns.append(float(returns.mean()))
            med_returns.append(float(returns.median()))
    return {
        "mean_top_quantile_precision": _mean_or_none(precisions),
        "mean_top_quantile_return": _mean_or_none(avg_returns),
        "median_top_quantile_return": _mean_or_none(med_returns),
    }


def daily_top_n_metrics(frame: pd.DataFrame, *, score_col: str, winner_col: str, return_col: str, top_n: int) -> dict[str, float | None]:
    precisions: list[float] = []
    avg_returns: list[float] = []
    for _, group in frame.groupby("trade_date", sort=False):
        ordered = group.sort_values(score_col, ascending=False).head(top_n)
        winners = pd.to_numeric(ordered[winner_col], errors="coerce").fillna(0.0)
        returns = pd.to_numeric(ordered[return_col], errors="coerce").dropna()
        if not winners.empty:
            precisions.append(float(winners.mean()))
        if not returns.empty:
            avg_returns.append(float(returns.mean()))
    return {
        "mean_top_n_precision": _mean_or_none(precisions),
        "mean_top_n_return": _mean_or_none(avg_returns),
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))

