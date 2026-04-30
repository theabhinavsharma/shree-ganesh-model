# Day 1 Random Forest Quick Compare

This folder compares a Random Forest challenger against the existing day-1 champion for one universe.
The comparison uses the same full-history walk-forward setup and avoids a giant OOF concat so it can finish reliably on this machine.
Open `summary.json` first, then `comparison_vs_champion.csv`.

## Files

- `challenger_metrics.csv`: Generated artifact.
- `comparison_vs_champion.csv`: Generated artifact.
- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
