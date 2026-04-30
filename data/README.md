# Data Folder Guide

Use this folder guide before reading any dataset directly.

## Main subfolders

- `raw/`: source files as downloaded
- `normalized/`: parsed source-specific tables
- `derived/`: research-ready joined or feature-rich tables
- `*_full_history/`: long-history caches for a specific domain such as fundamentals, shareholding, events, macro, or corporate actions

## Safe reading order

1. Read [docs/data_storage.md](../docs/data_storage.md)
2. Read the matching contract in `data_contracts/`
3. Read the dataset sidecar manifest if one exists
4. Read the actual parquet or csv file

## Important warning

Do not assume all `*_pct` columns use the same unit across the repo. Always verify the contract or manifest first.
