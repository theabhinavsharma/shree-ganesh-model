# Repository Data Inventory

This folder contains a machine-readable catalog of the repository's data layout as of `2026-04-28`.

## Files

- `summary.json`: small status file with scope, counts, and key warnings.
- `summary.json.manifest.json`: sidecar manifest for `summary.json`.
- `repo_data_inventory.json`: machine-readable inventory of folders, datasets, trust levels, lag rules, producer modules, and consumer modules.
- `repo_data_inventory.json.manifest.json`: sidecar manifest for `repo_data_inventory.json`.

## How to use this folder

1. Open `summary.json` first.
2. Open `repo_data_inventory.json` for the full machine-readable catalog.
3. For any canonical table mentioned there, open the matching contract in `data_contracts/`.
4. For any output artifact, open its own `.manifest.json` sidecar before using the data.

## Important rules captured here

- Official NSE bhavcopy ingestion is the authoritative v1 daily market-data path.
- `tmp/`, `reports/`, and `data/ml/panels/` are not canonical source data.
- Do not infer units from suffixes such as `_pct`; use the contract or manifest.
- If a required source is unavailable, the correct machine action is `blocked` or `unknown`, not silent approximation.
