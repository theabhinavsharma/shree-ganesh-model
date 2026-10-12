"""Corporate cash flows in backtests and labels (2026-10-12, INC-2026-10-12-payout-reinvested-in-stub; user: treat big
payouts and demergers as cash, and count dividends as gains).

1. Dividends count as gains. The adjusted price series keeps ordinary dividends out (a price-return series: dividends under
   10% of the price are not adjusted). Every such dividend is added on its ex-session as dividend / previous raw close, i.e.
   reinvested that day, as a total-return index does. Dividends already in the adjusted series (applied special dividends)
   are not added twice.
2. Value-removal events: an APPLIED price-only factor below VR_MAX (a payout, demerger, scheme, rights issue or capital
   reduction moving more than half of the value out of the company). The adjusted series alone would reinvest that value in
   the leftover stub (MAJESCO 2020: +837% instead of about +40%). A position held through one keeps only the factor share in
   the stock; the rest becomes cash at the ex-session and stays cash until the position's planned exit. Positions bought after
   the event trade the stub normally.
Source: data/corporate_actions_full_history/normalized/price_only_ca_factors.parquet (NSE corporate actions, parsed by
build_price_only_ca.py). Old symbols are mapped to the panel columns they trade under (nse_symbols.now_map).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PO = ROOT / "data/corporate_actions_full_history/normalized/price_only_ca_factors.parquet"
VR_MAX = 0.5
DIV_KINDS = ("large_dividend", "special_dividend")


_PO_CACHE: dict = {}


def _po() -> pd.DataFrame:
    if "po" in _PO_CACHE:
        return _PO_CACHE["po"]
    p = pd.read_parquet(PO, columns=["symbol", "ex_date", "kind", "price_factor", "applied", "ex_session", "dividend", "raw_prev_close"])
    p["ex_date"] = pd.to_datetime(p["ex_date"]); p["ex_session"] = pd.to_datetime(p["ex_session"], errors="coerce")
    p["day"] = p["ex_session"].fillna(p["ex_date"])
    _PO_CACHE["po"] = p
    return p


def _row(cal: pd.DatetimeIndex, d) -> int:
    i = int(cal.searchsorted(pd.Timestamp(d)))
    return i if i < len(cal) else -1


def dividend_yields(cal: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """cal x columns frame of dividend / previous raw close on ex-sessions (zeros elsewhere), for dividends not already in
    the adjusted price series."""
    p = _po()
    d = p[p["kind"].isin(DIV_KINDS) & ~p["applied"] & p["dividend"].notna() & (p["raw_prev_close"] > 0)].copy()
    d["y"] = d["dividend"] / d["raw_prev_close"]
    d = d[(d["y"] > 0) & (d["y"] < 0.5)]
    out = pd.DataFrame(0.0, index=cal, columns=list(columns))
    cols = set(out.columns)
    for s, day, y in zip(d["symbol"], d["day"], d["y"]):
        if s in cols:
            r = _row(cal, day)
            if r >= 0:
                out.iat[r, out.columns.get_loc(s)] += y
    return out


def vr_events(cal: pd.DatetimeIndex, columns) -> dict:
    """{column index: [(session row of the ex-session, price factor), ...]} for applied value-removal events."""
    p = _po()
    v = p[p["applied"] & (p["price_factor"] < VR_MAX)]
    idx = {s: i for i, s in enumerate(columns)}
    out: dict = {}
    for s, day, f in zip(v["symbol"], v["day"], v["price_factor"]):
        if s in idx:
            r = _row(cal, day)
            if r >= 0:
                out.setdefault(idx[s], []).append((r, float(f)))
    return {k: sorted(x) for k, x in out.items()}


def apply_vr(growth: np.ndarray, i0: int, cols, vr: dict) -> np.ndarray:
    """growth[r, q] = value of position q at session i0+1+r relative to its entry. For a value-removal event at session k
    inside the hold, from the session after k the position is f x (stub path since k) + (1 - f) x cash. Edits in place."""
    if not vr:
        return growth
    n = growth.shape[0]
    for q, c in enumerate(np.asarray(cols)):
        for k, f in vr.get(int(c), ()):
            r = k - i0 - 1
            if 0 <= r < n - 1:
                base = growth[r, q]
                if base != 0 and np.isfinite(base):
                    growth[r + 1:, q] = base * (f * growth[r + 1:, q] / base + (1.0 - f))
                break                                             # first event in the hold only (later ones act on a small stub)
    return growth


def label_path(high_rel: np.ndarray, close_rel: np.ndarray, i0: int, col: int, vr: dict) -> np.ndarray:
    """For labels: highs relative to the entry close, with a value-removal event in the window turned into cash as above."""
    h = np.asarray(high_rel, dtype=float).copy()
    for k, f in vr.get(int(col), ()):
        r = k - i0 - 1
        if 0 <= r < len(h) - 1:
            base = close_rel[r]
            if base and np.isfinite(base):
                h[r + 1:] = base * (f * h[r + 1:] / base + (1.0 - f))
            break
    return h


def trade_total_return(symbol: str, dates, closes, entry: float) -> float:
    """One holding's return including dividends received (cash, from the day after the entry session) and with a big payout /
    demerger turned into cash as in apply_vr. dates / closes: the holding's sessions from the entry session on (adjusted
    closes); entry: the entry price on the same basis. Used by the paper scorer for live batches (2026-10-12)."""
    rel = np.asarray(closes, dtype=float) / entry
    if not len(rel) or not np.isfinite(entry) or entry <= 0:
        return float("nan")
    d = pd.to_datetime(pd.Series(list(dates))).dt.normalize().to_numpy()
    p = _po(); ps = p[p["symbol"] == symbol]
    for day, f in zip(ps.loc[ps["applied"] & (ps["price_factor"] < VR_MAX), "day"], ps.loc[ps["applied"] & (ps["price_factor"] < VR_MAX), "price_factor"]):
        r = int(np.searchsorted(d, np.datetime64(pd.Timestamp(day).normalize())))
        if 0 <= r < len(rel) - 1 and rel[r]:
            rel[r + 1:] = rel[r] * (float(f) * rel[r + 1:] / rel[r] + (1.0 - float(f)))
            break
    divs = ps[ps["kind"].isin(DIV_KINDS) & ~ps["applied"] & ps["dividend"].notna() & (ps["raw_prev_close"] > 0)]
    cash = 0.0
    for day, y in zip(divs["day"], divs["dividend"] / divs["raw_prev_close"]):
        r = int(np.searchsorted(d, np.datetime64(pd.Timestamp(day).normalize())))
        if 1 <= r < len(rel) and r < len(d) and d[r] == np.datetime64(pd.Timestamp(day).normalize()) and 0 < y < 0.5:
            cash += y * rel[r - 1]
    return float(rel[-1] + cash - 1.0)
