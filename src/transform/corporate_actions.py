from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "last_price", "close", "avg_price", "prev_close")
QTY_COLUMNS = ("total_traded_qty", "deliverable_qty")


def apply_split_bonus_adjustments(
    daily_facts: pd.DataFrame,
    corporate_actions: pd.DataFrame,
) -> pd.DataFrame:
    adjusted = daily_facts.sort_values(["symbol", "trade_date"]).copy()
    adjusted["trade_date"] = pd.to_datetime(adjusted["trade_date"]).dt.normalize()
    if corporate_actions.empty:
        return _attach_identity_adjustment_columns(adjusted)

    actions = corporate_actions.copy()
    actions["symbol"] = actions["symbol"].astype(str).str.strip().str.upper()
    actions["ex_date"] = pd.to_datetime(actions["ex_date"], errors="coerce").dt.normalize()
    actions["adjustment_factor"] = pd.to_numeric(actions["adjustment_factor"], errors="coerce")
    actions = actions[
        actions["symbol"].notna()
        & actions["ex_date"].notna()
        & actions["adjustment_factor"].notna()
        & actions["adjustment_factor"].gt(0)
    ].copy()
    if actions.empty:
        return _attach_identity_adjustment_columns(adjusted)

    actions = (
        actions.groupby(["symbol", "ex_date"], as_index=False)
        .agg(adjustment_factor=("adjustment_factor", "prod"), action_count=("adjustment_factor", "size"))
        .sort_values(["symbol", "ex_date"])
        .reset_index(drop=True)
    )

    pieces: list[pd.DataFrame] = []
    for symbol, symbol_df in adjusted.groupby("symbol", sort=False):
        piece = symbol_df.copy()
        symbol_actions = actions.loc[actions["symbol"] == symbol]
        if symbol_actions.empty:
            pieces.append(_attach_identity_adjustment_columns(piece))
            continue

        trade_dates = piece["trade_date"].to_numpy(dtype="datetime64[ns]")
        ex_dates = symbol_actions["ex_date"].to_numpy(dtype="datetime64[ns]")
        factors = symbol_actions["adjustment_factor"].astype(float).to_numpy()

        suffix_product = np.cumprod(factors[::-1])[::-1]
        idx = np.searchsorted(ex_dates, trade_dates, side="right")
        share_factor = np.ones(len(piece), dtype=float)
        future_count = np.zeros(len(piece), dtype=int)
        has_future = idx < len(ex_dates)
        share_factor[has_future] = suffix_product[idx[has_future]]
        future_count[has_future] = len(ex_dates) - idx[has_future]
        price_factor = 1.0 / share_factor

        piece["share_adjustment_factor_to_present"] = share_factor
        piece["price_adjustment_factor_to_present"] = price_factor
        piece["future_split_bonus_action_count"] = future_count

        for column in PRICE_COLUMNS:
            if column not in piece.columns:
                continue
            raw_column = f"raw_{column}"
            if raw_column not in piece.columns:
                piece[raw_column] = piece[column]
            piece[column] = pd.to_numeric(piece[column], errors="coerce") * price_factor
        for column in QTY_COLUMNS:
            if column not in piece.columns:
                continue
            raw_column = f"raw_{column}"
            if raw_column not in piece.columns:
                piece[raw_column] = piece[column]
            piece[column] = pd.to_numeric(piece[column], errors="coerce") * share_factor
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True).sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _attach_identity_adjustment_columns(df: pd.DataFrame) -> pd.DataFrame:
    piece = df.copy()
    piece["share_adjustment_factor_to_present"] = 1.0
    piece["price_adjustment_factor_to_present"] = 1.0
    piece["future_split_bonus_action_count"] = 0
    return piece
