from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "last_price", "close", "avg_price", "prev_close")
QTY_COLUMNS = ("total_traded_qty", "deliverable_qty")

# Price-only corporate actions (demerger, scheme of arrangement, rights at TERP, special dividend,
# capital reduction / consolidation) — they move the price without the share-count identity that
# splits/bonuses obey, so they live in their own factor table (built by
# src/agentic/build_price_only_ca_factors.py). price_factor multiplies every price BEFORE ex_date
# (e.g. 0.28 for a demerger that transfers 72% of value); quantities are untouched.
PRICE_ONLY_PATH = (Path(__file__).resolve().parents[2]
                   / "data/corporate_actions_full_history/normalized/price_only_ca_factors.parquet")


def load_price_only_factors(path: Path | None = None) -> pd.DataFrame:
    """Applied price-only factors (symbol, ex_date, price_factor); empty frame if the table is absent."""
    p = Path(path) if path else PRICE_ONLY_PATH
    cols = ["symbol", "ex_date", "price_factor"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    f = pd.read_parquet(p)
    if "applied" in f.columns:
        f = f[f["applied"].astype(bool)]
    f = f[cols].copy()
    f["symbol"] = f["symbol"].astype(str).str.strip().str.upper()
    f["ex_date"] = pd.to_datetime(f["ex_date"], errors="coerce").dt.normalize()
    f["price_factor"] = pd.to_numeric(f["price_factor"], errors="coerce")
    return f[f["ex_date"].notna() & f["price_factor"].gt(0)]


def _suffix_factor(trade_dates: np.ndarray, ex_dates: np.ndarray, factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Product of factors whose ex_date is strictly after each trade date (1 when none), and their count."""
    out = np.ones(len(trade_dates), dtype=float)
    cnt = np.zeros(len(trade_dates), dtype=int)
    if len(ex_dates):
        suffix = np.cumprod(factors[::-1])[::-1]
        idx = np.searchsorted(ex_dates, trade_dates, side="right")
        has = idx < len(ex_dates)
        out[has] = suffix[idx[has]]
        cnt[has] = len(ex_dates) - idx[has]
    return out, cnt


def expected_price_factor(trade_dates: np.ndarray, share_actions: pd.DataFrame, price_only: pd.DataFrame) -> np.ndarray:
    """price_adjustment_factor_to_present the adjuster produces for one symbol:
    (1 / split-bonus share suffix) x (price-only suffix). Used by refresh_prices' LATE-CA SELF-HEAL."""
    sa = share_actions.sort_values("ex_date")
    share, _ = _suffix_factor(trade_dates, sa["ex_date"].to_numpy(dtype="datetime64[ns]"),
                              sa["adjustment_factor"].astype(float).to_numpy())
    po = price_only.sort_values("ex_date")
    pof, _ = _suffix_factor(trade_dates, po["ex_date"].to_numpy(dtype="datetime64[ns]"),
                            po["price_factor"].astype(float).to_numpy())
    return pof / share


def apply_split_bonus_adjustments(
    daily_facts: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    price_only: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Split/bonus adjustment (price / share, quantity x share) plus price-only factors.

    price_only=None loads the production table (load_price_only_factors); pass an empty frame
    to reproduce split/bonus-only behaviour exactly.
    """
    adjusted = daily_facts.sort_values(["symbol", "trade_date"]).copy()
    adjusted["trade_date"] = pd.to_datetime(adjusted["trade_date"]).dt.normalize()
    po = load_price_only_factors() if price_only is None else price_only.copy()
    if not po.empty:
        po["symbol"] = po["symbol"].astype(str).str.strip().str.upper()
        po["ex_date"] = pd.to_datetime(po["ex_date"], errors="coerce").dt.normalize()
        po = (po.dropna(subset=["ex_date"]).groupby(["symbol", "ex_date"], as_index=False)["price_factor"].prod()
                .sort_values(["symbol", "ex_date"]))

    actions = corporate_actions.copy() if not corporate_actions.empty else pd.DataFrame(columns=["symbol", "ex_date", "adjustment_factor"])
    if not actions.empty:
        actions["symbol"] = actions["symbol"].astype(str).str.strip().str.upper()
        actions["ex_date"] = pd.to_datetime(actions["ex_date"], errors="coerce").dt.normalize()
        actions["adjustment_factor"] = pd.to_numeric(actions["adjustment_factor"], errors="coerce")
        actions = actions[
            actions["symbol"].notna()
            & actions["ex_date"].notna()
            & actions["adjustment_factor"].notna()
            & actions["adjustment_factor"].gt(0)
        ].copy()
    if actions.empty and po.empty:
        return _attach_identity_adjustment_columns(adjusted)
    if not actions.empty:
        actions = (
            actions.groupby(["symbol", "ex_date"], as_index=False)
            .agg(adjustment_factor=("adjustment_factor", "prod"), action_count=("adjustment_factor", "size"))
            .sort_values(["symbol", "ex_date"])
            .reset_index(drop=True)
        )
    act_syms = set(actions["symbol"]) if not actions.empty else set()
    po_syms = set(po["symbol"]) if not po.empty else set()

    pieces: list[pd.DataFrame] = []
    for symbol, symbol_df in adjusted.groupby("symbol", sort=False):
        piece = symbol_df.copy()
        if symbol not in act_syms and symbol not in po_syms:
            pieces.append(_attach_identity_adjustment_columns(piece))
            continue

        trade_dates = piece["trade_date"].to_numpy(dtype="datetime64[ns]")
        sa = actions.loc[actions["symbol"] == symbol] if symbol in act_syms else actions.iloc[0:0]
        share_factor, future_count = _suffix_factor(
            trade_dates, sa["ex_date"].to_numpy(dtype="datetime64[ns]"), sa["adjustment_factor"].astype(float).to_numpy())
        sp = po.loc[po["symbol"] == symbol] if symbol in po_syms else po.iloc[0:0]
        po_factor, _ = _suffix_factor(
            trade_dates, sp["ex_date"].to_numpy(dtype="datetime64[ns]"), sp["price_factor"].astype(float).to_numpy())
        price_factor = po_factor / share_factor

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
