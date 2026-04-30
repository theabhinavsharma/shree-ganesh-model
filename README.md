# Indian Equities Research Pipeline

This repository contains a production-style, leak-safe research pipeline skeleton for NSE-listed equities. The implementation prioritizes auditability and truthful historical joins over apparent completeness. The approved v1 market-data path starts from the provided NSE UDiFF bhavcopy archive downloader using `https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_ddmmyyyy.csv`, and now falls back to the older official NSE bhavcopy plus delivery archives for pre-2020 dates so daily market data can be pulled back to 2015.

## Structure

```text
src/
  ingest/
    nse/
    fundamentals/
    shareholding/
    sector_flow/
  master/
  transform/
  features/
  screen/
  utils/
tests/
docs/
data_contracts/
configs/
```

## Pipeline flow

1. `src/ingest/nse/*` fetches and stores raw NSE bhavcopy files by date. For 2020 onward it uses the provided UDiFF archive URL pattern; for earlier dates it uses the older official NSE bhavcopy zip plus the official delivery archive.
2. `src/transform/build_daily_facts.py` normalizes bhavcopy rows and computes derived daily features.
3. `src/master/stock_master.py` builds stock identity data from official NSE quote/meta APIs and can merge an optional user-provided NSE industry mapping file.
4. `src/ingest/shareholding/*` fetches historical shareholding pattern and promoter pledge data from official NSE corporate filing APIs on a symbol-by-symbol basis.
5. `src/ingest/fundamentals/*` fetches historical quarterly financial-result records from official NSE corporate filing APIs on a symbol-by-symbol basis.
6. `src/ingest/sector_flow/*` fetches official NSDL fortnightly sector-flow reports and normalizes them into a research table.
7. Quarterly and fortnightly datasets are joined only through `effective_from_date` lag-safe joins.
8. `src/screen/build_universe.py` builds a daily screening universe and marks missing inputs instead of fabricating pass/fail outputs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Current status

- Implemented: project structure, contracts, refactored NSE UDiFF fetcher, bhavcopy normalization, delivery-aware daily features, official NSE stock-master enrichment, official NSE historical shareholding loader, official NSE historical quarterly-results loader, official NSDL sector-flow parser, lagged joins, screening builder, validations, tests, and documentation.
- Partially implemented: sector-flow publication-date handling remains unresolved because the official NSDL report pages expose period-end data but not a clean machine-readable publication timestamp; the normalized sector-flow table is built but should not be forward-filled historically until publication timing is resolved or approved.
- Still missing approved source inputs: official NSE industry classification mapping by symbol, point-in-time market-cap and valuation inputs, and balance-sheet/cash-flow fields not cleanly exposed in the current mapped NSE quarterly-results payload.
- Daily market data can now be ingested back to 2015 from official NSE archives, but long-history return studies still need a separate corporate-action adjustment layer if they are meant to be split/bonus adjusted.
- A production shortlist can now be automated with rules `1`, `3`, and `4` removed. The code expresses the remaining rules with explicit formulas, including 5Y growth checks, trailing-63-day volume and delivery highs, daily/weekly/monthly RSI thresholds, lag-valid PE from trailing EPS, promoter holding, and moving-average filters.

## Reading Data Safely

- Do not guess meaning from a file name or suffix alone.
- Open the matching data contract in `data_contracts/` for canonical column definitions.
- For generated report folders, open `summary.json` first, then the folder `README.md`, then the matching `.manifest.json` sidecar for any parquet or csv file you want to inspect.
- Sidecar manifests are meant for humans and future agents. They record row grain, primary keys, null counts, sample values, and plain-English column definitions where available.
- See [docs/data_storage.md](docs/data_storage.md) for the storage conventions and the safe way to inspect outputs.
