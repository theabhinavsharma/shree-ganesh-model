# Repeatability Check

This folder compares the completed independent reruns for the weekly 7-day winners model.
Open summary.json first, then run_comparison.csv, then shortlist_overlap.csv.

## Files

- `run_comparison.csv`: Generated artifact.
- `shortlist_overlap.csv`: Generated artifact.
- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
