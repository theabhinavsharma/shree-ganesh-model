"""Freshness gate — refuses to let the pipeline proceed if any critical input is stale.

Two levels of checks:
  1. FILE-level:   max(date_col) or file mtime vs contract
  2. COLUMN-level: last non-null date per critical column vs contract  ← catches the "fresh file, stale column" bug

Why COLUMN-level exists:
  On 2026-07-01 macro_panel.parquet had trade_date=2026-06-30 (looked fresh) but internal columns
  dxy/us_10y/brent had their last non-null value on 2026-05-07 (55 days stale — forward-filled).
  The announcements file was also stale: max event_date 2026-04-27 (65 calendar days old).
  File-level checks let this through. Column-level catches it.

Usage:
  python3 src/agentic/verify_freshness.py
    exit 0 → all fresh
    exit 1 → at least one stale; prints table of every stale input

  from verify_freshness import verify_or_die, snapshot
  verify_or_die()  # raises SystemExit(1) with the table printed
  snapshot()       # returns list[dict] for downstream reporting (dashboard)
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")


@dataclass(frozen=True)
class Contract:
    path: str                              # relative to ROOT
    date_col: str | None                   # None → use mtime
    max_stale_bd: int                      # file-level tolerance
    tag: str                               # short label
    col_checks: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    # col_checks: [(column_name, max_stale_bd_for_this_column), ...] — checked when date_col is set
    # Empty → no column-level checks (file-level only)


# Contracts — the pipeline's data-freshness contract
CONTRACTS: list[Contract] = [
    # === PRICES ===
    Contract(
        path="data/derived/stock_daily_facts_adjusted_2015plus.parquet",
        date_col="trade_date", max_stale_bd=3, tag="PRICES",
        col_checks=(
            ("close",         3),  # actual price column
            ("rsi_14_daily",  3),
            ("return_20d",    3),
            ("volume_vs_20d", 3),
        ),
    ),

    # === ENGINES (mtime-based since these are prediction outputs regenerated each run) ===
    Contract("data/derived/compare_short_horizons.parquet",        None, 3, "CS_ENGINE"),
    Contract("data/derived/high_conviction_predictions.parquet",   None, 3, "HC_ENGINE"),
    Contract("data/derived/multibagger_today_predictions.parquet", None, 3, "MB_ENGINE"),
    Contract("data/derived/180d_today_predictions.parquet",        None, 3, "F180_ENGINE"),
    Contract("tmp/from_scratch_7d_run/multi_horizon_top.csv",      None, 3, "MH_ENGINE"),
    Contract("data/derived/missed_winner_classifier.parquet",      None, 3, "ML_CLASSIFIER"),

    # === MACRO PANEL — with per-column checks (this is where the 2026-07-01 bug lived) ===
    Contract(
        path="data/derived/macro_panel.parquet",
        date_col="trade_date", max_stale_bd=3, tag="MACRO_PANEL",
        col_checks=(
            ("usdinr",  3),   # daily FX — must be current
            ("brent",   5),   # commodities — daily FRED (was stale 55d)
            ("wti",     5),
            ("us_10y",  5),   # US rates — daily FRED (was stale 55d)
            ("dxy",     5),   # dollar index — daily FRED (was stale 55d)
            ("us_vix",  5),   # vol — daily FRED (was stale 55d)
            ("spx",     5),   # US equities — daily FRED (was stale 55d)
        ),
    ),

    # === INDUSTRY / SECTOR ===
    Contract(
        path="data/derived/industry_panel.parquet",
        date_col="trade_date", max_stale_bd=3, tag="INDUSTRY",
        col_checks=(
            ("sector_5d_ret",  3),
            ("sector_20d_ret", 3),
            ("rs_20d",         3),
        ),
    ),

    # === NEWS EVENTS ===
    Contract(
        path="data/derived/news_event_features.parquet",
        date_col="trade_date", max_stale_bd=5, tag="NEWS_EVENTS",
    ),

    # === ANNOUNCEMENTS ===
    Contract(
        path="data/events_full_history/normalized/stock_announcements.parquet",
        date_col="event_date", max_stale_bd=10, tag="ANNOUNCEMENTS",
    ),
]


def _business_days_between(a: date, b: date) -> int:
    """Count of business days from a to b inclusive (Mon-Fri)."""
    if a >= b: return 0
    days = pd.bdate_range(start=a + timedelta(days=1), end=b)
    return len(days)


def _today() -> date:
    """Isolate today() so tests can override."""
    return date.today()


def _last_non_null_date(fp: Path, date_col: str, value_col: str) -> date | None:
    """Return the latest date on which value_col has a non-null value.

    This is the key check: even if the file has a fresh trade_date row, the value_col
    within that row may be NaN (or forward-filled from months ago). We want the
    latest date on which the column actually received fresh data.
    """
    try:
        df = pd.read_parquet(fp, columns=[date_col, value_col])
    except Exception:
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    s = df.dropna(subset=[date_col, value_col])
    if not len(s): return None
    return s[date_col].max().date()


def check_one(c: Contract) -> dict:
    """Check a single contract. Returns a snapshot dict:
      {
        tag, path, kind ('file'|'file+col'), max_stale_bd,
        file_max_date, file_stale_bd, file_is_stale,
        col_results: [{col, last_date, stale_bd, limit, is_stale}, ...],
        is_stale (overall), msg
      }
    """
    fp = ROOT / c.path
    today = _today()
    snap = {
        "tag": c.tag, "path": c.path, "kind": "file",
        "max_stale_bd": c.max_stale_bd,
        "file_max_date": None, "file_stale_bd": 9999, "file_is_stale": True,
        "col_results": [], "is_stale": True, "msg": "",
    }

    if not fp.exists():
        snap["msg"] = "MISSING FILE"
        return snap

    # FILE-level check
    if c.date_col is None:
        mtime = datetime.fromtimestamp(fp.stat().st_mtime).date()
        snap["file_max_date"] = mtime
        snap["file_stale_bd"] = _business_days_between(mtime, today)
        snap["msg"] = f"mtime={mtime}"
    else:
        try:
            df = pd.read_parquet(fp, columns=[c.date_col])
        except Exception as e:
            snap["msg"] = f"READ ERROR: {e}"
            return snap
        df[c.date_col] = pd.to_datetime(df[c.date_col], errors="coerce")
        s = df.dropna(subset=[c.date_col])
        if not len(s):
            snap["msg"] = "no valid dates"
            return snap
        max_d = s[c.date_col].max().date()
        snap["file_max_date"] = max_d
        snap["file_stale_bd"] = _business_days_between(max_d, today)
        snap["msg"] = f"max={max_d}"

    snap["file_is_stale"] = snap["file_stale_bd"] > c.max_stale_bd

    # COLUMN-level checks
    if c.col_checks and c.date_col is not None:
        snap["kind"] = "file+col"
        for col_name, col_limit in c.col_checks:
            last_d = _last_non_null_date(fp, c.date_col, col_name)
            if last_d is None:
                snap["col_results"].append({
                    "col": col_name, "last_date": None,
                    "stale_bd": 9999, "limit": col_limit, "is_stale": True,
                    "err": "column missing or all-null",
                })
                continue
            stale_bd = _business_days_between(last_d, today)
            snap["col_results"].append({
                "col": col_name, "last_date": last_d,
                "stale_bd": stale_bd, "limit": col_limit,
                "is_stale": stale_bd > col_limit,
            })

    # Overall is_stale = file stale OR any column stale
    any_col_stale = any(cr["is_stale"] for cr in snap["col_results"])
    snap["is_stale"] = snap["file_is_stale"] or any_col_stale
    return snap


def snapshot() -> list[dict]:
    """Return the full snapshot as a list of dicts — for the dashboard."""
    return [check_one(c) for c in CONTRACTS]


def _fmt_row_file(s: dict) -> str:
    status = "❌ FAIL" if s["file_is_stale"] else "✅ OK"
    stale_str = "N/A" if s["file_stale_bd"] >= 9999 else str(s["file_stale_bd"])
    p = s["path"] if len(s["path"]) < 62 else "..." + s["path"][-59:]
    return f"  {status:>7s}  {s['tag']:<15s}  {stale_str:>4s} / {s['max_stale_bd']:<4d}bd   {p:<62s}  {s['msg']}"


def _fmt_row_col(s: dict, cr: dict) -> str:
    status = "❌ FAIL" if cr["is_stale"] else "✅ OK"
    stale_str = "N/A" if cr["stale_bd"] >= 9999 else str(cr["stale_bd"])
    last = cr["last_date"] or "—"
    tail = cr.get("err", f"last non-null={last}")
    return f"    {status:>7s}  ↳ col={cr['col']:<15s}  {stale_str:>4s} / {cr['limit']:<4d}bd   {tail}"


def print_report(rows: list[dict]) -> None:
    today = _today()
    print("\n" + "═" * 110)
    print(f" FRESHNESS GATE  ·  {today.isoformat()}  ·  {len(rows)} inputs checked")
    print("═" * 110)
    print(f"  {'STATUS':>7s}  {'TAG':<15s}  {'STALE/LIM':>11s}   {'PATH':<62s}  DETAIL")
    print("  " + "─" * 108)
    n_stale_files, n_stale_cols = 0, 0
    for s in rows:
        print(_fmt_row_file(s))
        if s["file_is_stale"]: n_stale_files += 1
        for cr in s["col_results"]:
            print(_fmt_row_col(s, cr))
            if cr["is_stale"]: n_stale_cols += 1
    print("═" * 110)
    if not any(s["is_stale"] for s in rows):
        print(f"  ✅ ALL FRESH — pipeline may proceed.")
    else:
        print(f"  ❌ STALE — {n_stale_files} file-level failure(s), {n_stale_cols} column-level failure(s).")
        print(f"  Fix scripts:")
        print(f"    PRICES         → src/agentic/refresh_prices.py")
        print(f"    MACRO_PANEL    → src/agentic/fetch_forex_macro.py + fetch_commodity_prices.py + fetch_global_macro.py, then build_macro_panel.py")
        print(f"    INDUSTRY       → src/agentic/fetch_industry_indicators.py")
        print(f"    NEWS_EVENTS    → src/agentic/build_news_event_features.py")
        print(f"    ANNOUNCEMENTS  → src/agentic/refresh_announcements.py")
        print(f"    <ENGINE>       → src/agentic/<engine_script>.py")
    print()


def verify_or_die() -> None:
    """For downstream scripts to call: fails hard if anything is stale."""
    rows = snapshot()
    print_report(rows)
    if any(r["is_stale"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    verify_or_die()
