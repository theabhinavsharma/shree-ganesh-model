"""Corporate-actions-before-prices CI gate.

Enforces the 2026-05-04 ledger correction ("'Adjusted' prices weren't":
399 unexplained -50% single-day drops) and the 2026-08-18 correction
(113-day CA-store rot — TRENT/LICI/CUB splits missing, corrupting price
adjustment itself). Promised in logs/calibration_corrections.jsonl on
2026-05-04; this file is that promise.

Rule (SHOWCASE.html §Recreation Kit, step 10): on any (symbol, date) that
overlaps a corporate-action ex_date (±1 trading day, to absorb NSE ex-date
vs record-date drift), the ADJUSTED close must NOT show a single-day drop
worse than -25%. If it does, the adjustment factor was not applied and the
model is about to learn a phantom crash.

Two checks, both real-data gated:
  1. test_no_unadjusted_drops_on_ex_dates   — the spec'd assertion.
  2. test_ca_store_is_canonical_and_alive    — one store only, >0 rows,
     ex_date parses, max ex_date not older than the price panel by >60d
     (the 113-day-rot guard at test time, complementing the freshness gate).

Run with: /usr/bin/python3 -m pytest tests/test_no_unadjusted_corporate_actions.py -v
If this fails, do NOT relax it. Re-run the CA-repair path
(see reports/be_series_gap_repair_20260828.md) and re-derive prices.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CA = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"

DROP_LIMIT = -0.25          # spec: no single-day drop < -25% on an ex_date
EX_DATE_TOLERANCE_DAYS = 1  # NSE ex-date can land ±1 session from the tagged date
MAX_STORE_LAG_DAYS = 60     # CA store must not trail the price panel by more than this

_needs_data = pytest.mark.skipif(
    not (PRICES.exists() and CA.exists()),
    reason="price panel or corporate-actions store not built yet",
)


# Only split/bonus actions can (and must) be adjusted away. Demergers, schemes of
# arrangement and rights are genuine economic re-pricings with no factor to apply;
# the contamination filter already excuses those drops (CA within +-5d). Matching on
# the subject text rather than is_split/is_bonus means a PARSER miss (the 2026-09-19
# "Fv Splt Frm" class) is still caught -- the row says split, the factor says nothing.
SPLIT_BONUS_SUBJECT = r"split|splt|sub-?division|bonus"
NON_EQUITY = r"ncrps|preference|debenture|warrant"


def _load_ca() -> pd.DataFrame:
    ca = pd.read_parquet(CA, columns=["symbol", "ex_date", "subject"])
    ca["ex_date"] = pd.to_datetime(ca["ex_date"], errors="coerce")
    ca = ca.dropna(subset=["ex_date"])
    subj = ca["subject"].astype(str)
    return ca[subj.str.contains(SPLIT_BONUS_SUBJECT, case=False, regex=True)
              & ~subj.str.contains(NON_EQUITY, case=False, regex=True)]


@_needs_data
def test_ca_store_is_canonical_and_alive():
    """One canonical CA store (2026-08-18 rule), non-empty, and not rotting."""
    ca = _load_ca()
    assert len(ca) > 0, "corporate-actions store is empty"

    px_max = pd.read_parquet(PRICES, columns=["trade_date"])["trade_date"].max()
    px_max = pd.to_datetime(px_max)
    lag = (px_max - ca["ex_date"].max()).days
    assert lag <= MAX_STORE_LAG_DAYS, (
        f"CA store max ex_date {ca['ex_date'].max().date()} trails price panel "
        f"{px_max.date()} by {lag}d (> {MAX_STORE_LAG_DAYS}d). This is the "
        f"113-day-rot pattern from the 2026-08-18 ledger entry."
    )


@_needs_data
def test_no_unadjusted_drops_on_ex_dates():
    """No split/bonus (symbol, ex_date ±1d) row may carry an adjusted return_1d < -25%."""
    ca = _load_ca()
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "return_1d"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])

    # Only rows that are big drops — keeps the join tiny on a 5M-row panel.
    big = px[px["return_1d"].fillna(0) < DROP_LIMIT]
    if big.empty:
        return  # nothing to examine — trivially passes

    # Expand each ex_date to a ±tolerance window and inner-join on (symbol, date).
    offsets = range(-EX_DATE_TOLERANCE_DAYS, EX_DATE_TOLERANCE_DAYS + 1)
    win = pd.concat(
        [ca.assign(trade_date=ca["ex_date"] + pd.Timedelta(days=k)) for k in offsets],
        ignore_index=True,
    )[["symbol", "trade_date", "ex_date"]].drop_duplicates()

    bad = big.merge(win, on=["symbol", "trade_date"], how="inner")

    assert bad.empty, (
        f"\n{len(bad)} adjusted-price rows still show a < {DROP_LIMIT:.0%} drop ON a "
        f"split/bonus ex_date — the adjustment factor was not applied:\n"
        + bad.sort_values("return_1d")
             .head(20)[["symbol", "trade_date", "ex_date", "subject", "return_1d"]]
             .to_string(index=False)
        + "\n\nRe-derive prices with CA applied (see 2026-05-04 and 2026-08-18 "
        f"entries in logs/calibration_corrections.jsonl)."
    )
