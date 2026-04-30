# Day 1 Random Forest Intraday Sheet

This folder contains the Random Forest next-day 5 percent shortlist and a practical top-5 intraday execution table for 16 April 2026.
Open summary.json first, then rf_intraday_action_table_top5_20260416.csv, then rf_intraday_reserve_table_20260416.csv.
Price facts are official through 15 April 2026. Event and macro side inputs may lag same-day 15 April filings.

## Files

- `official_daily_facts_current_20260415.parquet`: Generated artifact.
- `rf_current_shortlist_liquid_5cr_plus_20260415_for_20260416.csv`: Generated artifact.
- `rf_intraday_action_table_top5_20260416.csv`: Generated artifact.
- `rf_intraday_reserve_table_20260416.csv`: Generated artifact.
- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
