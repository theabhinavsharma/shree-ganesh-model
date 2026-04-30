# Day1 5pct All Names 2026-04-23

This folder contains the all-stocks day-1 5 percent shortlist for trading on 2026-04-23.
Historical evaluation comes from the cached 2015-2025 day-1 model; the current slice is refreshed to the official 2026-04-22 overlay.

## Files

- `current_shortlist_allnames_20260422_for_20260423.csv`: Generated artifact.
- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
