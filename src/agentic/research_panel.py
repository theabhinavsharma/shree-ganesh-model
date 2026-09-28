"""Shared, audited primitives for research backtests (2026-09-27, after the 73-finding audit).

Every research script had re-implemented panel loading, forward windows and portfolio returns, and the
audit (logs/audits/audit_20260927_research_code.md) found the same defects in each copy:
  * windows counted in a symbol's own ROWS, so gaps (suspensions, formerly-missing BE/BZ sessions)
    stretched a '95-session' window over 100-300+ sessions;
  * returns across a gap zeroed by pct_change(fill_method=None).fillna(0), deleting the move;
  * the back-adjusted close used as a price LEVEL (close > 50, log price, PE), which encodes future
    splits/bonuses/demergers;
  * renames treated as delistings;
  * entry at the signal-day close although the live contract enters at the next open.
Use these helpers instead of hand-rolled equivalents.

Conventions
  - "session" = a date on the GLOBAL trading calendar (every date present in the panel).
  - Adjusted prices (open/high/low/close) are for RETURNS; raw_price() gives the then-traded price for LEVELS.
  - Wide frames are dates x symbols on the global calendar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PANEL = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
MASTER = ROOT / "data/derived/security_master.parquet"
sys.path.insert(0, str(ROOT / "src/agentic"))


def equity_filter(symbols: pd.Series) -> pd.Series:
    """True for operating companies (ISIN master fund units, NSE ETF list, exclusion file and regex removed)."""
    from generate_hybrid_basket import non_equity
    fund = set()
    if MASTER.exists():
        m = pd.read_parquet(MASTER, columns=["symbol", "is_fund_unit"])
        fund = set(m.loc[m["is_fund_unit"], "symbol"])
    return ~symbols.isin(fund) & ~non_equity(symbols)


def load_panel(columns: list[str], filters=None, equities_only: bool = True) -> pd.DataFrame:
    """Panel rows with trade_date as datetime, sorted by (symbol, trade_date). Always pass explicit columns."""
    cols = list(dict.fromkeys(["symbol", "trade_date"] + list(columns)))
    px = pd.read_parquet(PANEL, columns=cols, filters=filters)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    if equities_only:
        px = px[equity_filter(px["symbol"])]
    return px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def session_calendar(px: pd.DataFrame | None = None) -> pd.DatetimeIndex:
    """Global trading calendar: every date on which the panel has any row."""
    if px is None:
        px = pd.read_parquet(PANEL, columns=["trade_date"])
    return pd.DatetimeIndex(sorted(pd.to_datetime(px["trade_date"]).unique()))


def raw_price(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Then-traded price: adjusted price / price_adjustment_factor_to_present. Use for LEVEL filters
    (price floors, log price, PE). Requires the factor column in df."""
    f = pd.to_numeric(df["price_adjustment_factor_to_present"], errors="coerce").replace(0, np.nan).fillna(1.0)
    return df[col] / f


def wide(px: pd.DataFrame, value: str, cal: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """dates x symbols frame on the global calendar (NaN where a symbol has no row that session)."""
    w = px.pivot(index="trade_date", columns="symbol", values=value).sort_index()
    return w.reindex(cal) if cal is not None else w


def gap_aware_returns(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Daily close-to-close returns on the session grid. A symbol that misses sessions keeps its last
    close (no return while it cannot trade) and the full move across the gap lands on the first session
    it trades again. After its final row the value is frozen (0 return) — the holder exits at the last
    close. Before its first row returns are NaN (not holdable)."""
    first = close_wide.apply(lambda s: s.first_valid_index())
    ff = close_wide.ffill()
    r = ff.pct_change(fill_method=None)
    for sym, d0 in first.items():
        if d0 is not None:
            r.loc[:d0, sym] = np.nan
    return r


def forward_window(px_wide_close: pd.DataFrame, px_wide_high: pd.DataFrame, px_wide_low: pd.DataFrame,
                   h: int) -> dict[str, pd.DataFrame]:
    """Session-calendar forward outcomes from each date's CLOSE over the next h sessions:
      exit  = last available close on or before session t+h (frozen at the final close if the symbol
              stopped trading; NaN if the window runs past the panel end for a still-trading symbol),
      hi/lo = max high / min low over sessions t+1..t+h using the rows that exist,
      span  = number of sessions the window covers (always h here; exported for audits)."""
    n = len(px_wide_close)
    ff = px_wide_close.ffill()
    exit_ = ff.shift(-h)
    last_valid = px_wide_close.apply(lambda s: s.last_valid_index())
    panel_end = px_wide_close.index[-1]
    still_trading = last_valid >= panel_end - pd.Timedelta(days=10)
    # windows that run past the panel end are unknown for still-trading names
    exit_.iloc[max(0, n - h):, still_trading.values] = np.nan
    # a delisted name exits at its final close
    for sym in last_valid.index[~still_trading.values]:
        lv = last_valid[sym]
        if lv is not None:
            exit_.loc[:lv, sym] = exit_.loc[:lv, sym].fillna(px_wide_close.at[lv, sym])
    hi = px_wide_high[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1)
    lo = px_wide_low[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
    return {"exit": exit_, "hi": hi, "lo": lo}


def rename_map() -> dict[str, str]:
    """old symbol -> new symbol (NSE symbolchange via security_master.renamed_to; first successor)."""
    if not MASTER.exists():
        return {}
    m = pd.read_parquet(MASTER, columns=["symbol", "renamed_to"]).dropna()
    return {r.symbol: str(r.renamed_to).split(",")[0] for r in m.itertuples() if r.renamed_to}


def stitch_renames(ret_wide: pd.DataFrame, close_wide: pd.DataFrame) -> pd.DataFrame:
    """After an old symbol's last row, continue its return series with its successor's returns when the
    successor starts trading within 10 days — a rename is not a delisting."""
    out = ret_wide.copy()
    rm = rename_map()

    def depth(sym: str, seen: frozenset = frozenset()) -> int:
        nxt = rm.get(sym)
        return 0 if nxt is None or nxt in seen else 1 + depth(nxt, seen | {sym})

    # 2026-09-27: follow rename CHAINS (ALSTOMT&D -> GET&D -> GVT&D): stitch the latest links first so an
    # earlier symbol inherits a successor series that is itself already continued.
    for old in sorted(rm, key=depth):
        new = rm[old]
        if old not in out.columns or new not in out.columns:
            continue
        lo_ = close_wide[old].last_valid_index(); fn = close_wide[new].first_valid_index()
        if lo_ is None or fn is None or (fn - lo_).days > 10 or fn < lo_:
            continue
        after = out.index > lo_
        out.loc[after, old] = out.loc[after, new].fillna(0)
    return out


def next_open_entry(open_wide: pd.DataFrame, high_wide: pd.DataFrame, low_wide: pd.DataFrame,
                    close_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Live-contract entry for a signal at the close of t: buy at the OPEN of session t+1.
    'locked' flags t+1 sessions that are upper-circuit locked (open == high == low and the open is
    >= 4.9% above the prior close) — not buyable; callers should skip or treat as cash."""
    o1 = open_wide.shift(-1); h1 = high_wide.shift(-1); l1 = low_wide.shift(-1)
    prev = close_wide.ffill()
    locked = (o1 >= prev * 1.049) & np.isclose(o1, h1) & np.isclose(h1, l1)
    return {"entry_px": o1, "locked": locked.fillna(False)}


def cost_rt(adv_cr: pd.Series | float) -> pd.Series | float:
    """Round-trip cost as a fraction, scaled by liquidity: 0.5% at ADV >= 5cr, 1.0% at 1-5cr, 2.0% below 1cr."""
    a = pd.Series(adv_cr) if not isinstance(adv_cr, pd.Series) else adv_cr
    c = np.where(a >= 5, 0.005, np.where(a >= 1, 0.010, 0.020))
    return pd.Series(c, index=a.index) if isinstance(adv_cr, pd.Series) else float(c[0])


def nav_metrics(nav: pd.Series, era_split: str = "2023-01-01") -> dict:
    """CAGR, max drawdown (true daily NAV), Sharpe, per-era CAGR, calendar-year returns (prior year-end to
    year-end, so no session is dropped)."""
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    out = dict(cagr=((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan,
               maxdd=(nav / nav.cummax() - 1).min() * 100,
               sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan)
    cut = pd.Timestamp(era_split)
    for name, seg in (("disc", nav[nav.index < cut]), ("conf", nav[nav.index >= cut - pd.Timedelta(days=7)])):
        seg = seg.dropna()
        y = (seg.index[-1] - seg.index[0]).days / 365.25 if len(seg) > 1 else 0
        out[f"cagr_{name}"] = ((seg.iloc[-1] / seg.iloc[0]) ** (1 / y) - 1) * 100 if y > 0 else np.nan
    ye = nav.groupby(nav.index.year).last()
    prev = ye.shift(1); prev.iloc[0] = nav.iloc[0]
    out["years"] = {int(k): round((ye[k] / prev[k] - 1) * 100, 1) for k in ye.index}
    return out
