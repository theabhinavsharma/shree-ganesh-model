# Manual Monday Action Table

This folder contains the Monday trade action table built from the fresh April 10 shortlist and the confirmed current holdings.
LIQUIDETF was excluded from the buy basket as a non-equity leakage in the stock shortlist.

## Files

- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.
- `weekly_action_table_20260412.csv`: Generated artifact.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
