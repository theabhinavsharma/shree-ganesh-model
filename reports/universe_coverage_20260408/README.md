# Universe Coverage Review For 2026-04-08

This folder explains how much of the stock universe is analyzable in the ML pipeline and where missing data still matters.
The goal is to separate deliberate research limits from accidental universe loss.

## Files

- `coverage_by_universe.csv`: Generated artifact.
- `missing_by_column.csv`: Generated artifact.
- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
