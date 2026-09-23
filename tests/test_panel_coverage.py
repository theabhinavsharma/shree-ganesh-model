"""Coverage CI gate (2026-09-24): every symbol that traded in the latest raw bhavcopy
(EQ/BE/BZ) must have a row in the adjusted panel for that date, and BE/BZ rows must be
present whenever raw has them. Guards the series filter in build_daily_facts.py — the
08-28 fix was silently lost on 09-10 and 12 sessions went out ~9% short.
Run: /usr/bin/python3 -m pytest tests/test_panel_coverage.py -v
"""
from __future__ import annotations
import glob
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
RAW = ROOT / "data/raw/nse_full_history_official"

from src.transform.build_daily_facts import EQUITY_SERIES, T2T_SERIES  # noqa: E402


def test_series_constants_include_t2t():
    assert {"EQ", "BE", "BZ"} <= set(EQUITY_SERIES)
    assert {"BE", "BZ"} <= set(T2T_SERIES)


@pytest.mark.skipif(not PRICES.exists(), reason="panel not built")
def test_latest_session_covers_every_traded_symbol():
    dmax = pd.to_datetime(pd.read_parquet(PRICES, columns=["trade_date"])["trade_date"]).max()
    files = glob.glob(str(RAW / f"trade_date={dmax.date()}" / "sec_bhavdata_full_*.csv"))
    if not files:
        pytest.skip("no raw partition for the panel's latest date")
    raw = pd.read_csv(files[0], low_memory=False); raw.columns = [c.strip() for c in raw.columns]
    raw["SYMBOL"] = raw["SYMBOL"].str.strip(); raw["SERIES"] = raw["SERIES"].astype(str).str.strip()
    raw = raw[raw["SERIES"].isin(EQUITY_SERIES)]
    panel = pd.read_parquet(PRICES, columns=["symbol", "series"], filters=[("trade_date", "==", dmax)])
    missing = sorted(set(raw["SYMBOL"]) - set(panel["symbol"]))
    cov = 1 - len(missing) / len(raw)
    assert cov >= 0.995, f"{dmax.date()}: {len(missing)} traded symbols missing from panel ({cov:.2%}): {missing[:20]}"
    for s in T2T_SERIES:
        if (raw["SERIES"] == s).any():
            assert (panel["series"] == s).any(), f"raw has {s}-series rows on {dmax.date()}, panel has none — series filter regressed"
