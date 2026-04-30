# Data Storage Conventions

This file explains how data is stored in this repo so a later engineer, AI agent, or non-specialist reader does not have to guess.

## Folder meanings

- `data/raw/`
  Raw source files exactly as fetched from the upstream source wherever practical.
- `data/*/normalized/`
  Source-specific tables after parsing and light standardization, but before cross-source joins.
- `data/derived/`
  Research-ready tables produced from normalized inputs and explicit formulas.
- `tmp/`
  Run-specific outputs, experiments, and investigation folders. These are useful, but they should never be treated as canonical source data.
- `reports/`
  Production-style report outputs meant for recurring review.

## Naming rules

- Partitioned raw market data uses `trade_date=YYYY-MM-DD`.
- Generated report files should have explicit names that describe their purpose.
- Every important parquet or csv output should have a sidecar manifest named:
  `original_file_name.ext.manifest.json`
- Every report folder should contain a `README.md` that explains the files inside it.

## How to inspect a dataset safely

1. Open the folder `README.md` if it exists.
2. Open the matching `.manifest.json` sidecar for the file you care about.
3. If the dataset maps to a canonical table, open the matching contract in `data_contracts/`.
4. Only then inspect the data itself.

This order matters because the same suffix can mean different units across tables. Example:

- `delivery_pct` in daily market data is stored as a `0-1 ratio`
- `promoter_pct` in shareholding data is stored as percentage points such as `54.3`

Do not infer units from the suffix alone.

## What a sidecar manifest tells you

- plain-English file purpose
- row grain
- primary key
- row count and column count
- null counts
- sample values
- column definitions from the data contract where available

## Fail-closed principle

If a required field is missing or not provable, downstream screens should prefer `fail` or `blocked` over silent assumptions. This is especially important for:

- debt-free checks
- valuation filters
- 5-year growth rules
- short-history technical rules

## Human-readable output rules

- Prefer `summary.json` as the first file in a run folder.
- Provide `individual_counts` and `sequential_counts` separately so a reader can distinguish:
  - how many names pass a rule on its own
  - how many names survive the full checklist order
- Keep business-facing file names explicit rather than abbreviated when creating new outputs.

## Canonical truth hierarchy

1. raw source file
2. normalized table
3. derived table
4. screen/report output

If two downstream files disagree, move upward in that order until the discrepancy is understood.
