from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.forward_return_study import _read_stock_master
from src.ml.panel import _attach_latest_quote_snapshot_enrichment


def test_read_stock_master_preserves_optional_identity_columns(tmp_path: Path) -> None:
    stock_master_path = tmp_path / "stock_master.parquet"
    pd.DataFrame(
        {
            "symbol": ["ABC"],
            "sector": ["Industrials"],
            "industry": ["Machinery"],
            "basic_industry": ["Capital Goods"],
            "instrument_type": ["Equity"],
            "company_name": ["ABC Limited"],
            "issued_size": [123456789],
        }
    ).to_parquet(stock_master_path, index=False)

    loaded = _read_stock_master(stock_master_path)

    assert "company_name" in loaded.columns
    assert "issued_size" in loaded.columns
    assert loaded.loc[0, "company_name"] == "ABC Limited"
    assert loaded.loc[0, "issued_size"] == 123456789


def test_attach_latest_quote_snapshot_enrichment_hydrates_market_cap(monkeypatch, tmp_path: Path) -> None:
    quote_path = tmp_path / "quote_snapshot.parquet"
    pd.DataFrame(
        {
            "symbol": ["ABC"],
            "company_name": ["ABC Limited"],
            "instrument_type": ["Equity"],
            "issued_size": [100_000_000],
            "quote_pe_ttm": [12.5],
            "quote_last_price": [101.0],
        }
    ).to_parquet(quote_path, index=False)

    monkeypatch.setattr("src.ml.panel._find_latest_quote_snapshot_path", lambda: quote_path)

    panel = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "trade_date": pd.to_datetime(["2026-04-07"]),
            "close": [100.0],
            "instrument_type": [pd.NA],
        }
    )

    enriched = _attach_latest_quote_snapshot_enrichment(panel)

    assert enriched.loc[0, "company_name"] == "ABC Limited"
    assert enriched.loc[0, "instrument_type"] == "Equity"
    assert enriched.loc[0, "issued_size"] == 100_000_000
    assert enriched.loc[0, "market_cap_cr"] == 1000.0
    assert enriched.loc[0, "pe_ttm"] == 12.5
