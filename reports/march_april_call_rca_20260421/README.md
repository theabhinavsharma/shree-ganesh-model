# March April Call RCA

This folder audits matured March-April call sheets against official daily OHLC outcomes.
Fill feasibility is separated from realized performance so gap misses are not misread as pure ranking failures.

## Files

- `audited_calls.csv`: Generated artifact.
- `call_set_summary.csv`: Generated artifact.
- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
