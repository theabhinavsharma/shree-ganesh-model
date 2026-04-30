# Day 1 Champion All-Names Monday Sheet

This folder contains the refreshed champion 1-day 5 percent model run for trading on 2026-04-20 using official data through 2026-04-17.
No universe shortlist was used in the final live ranking. Historical backtest and live scoring were run in all_names mode.
Open summary.json first, then intraday_action_table_top5_20260420.csv, then top10.csv for reserves.

## Files

- `all_names_backtest_metrics.json`: Generated artifact.
- `intraday_action_table_top5_20260420.csv`: Generated artifact.
- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.
- `top10.csv`: Generated artifact.
- `top5.csv`: Generated artifact.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
