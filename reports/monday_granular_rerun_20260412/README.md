# Monday Granular Rerun

This folder contains the module-by-module rerun diagnostics for the Monday basket from fresh April 10 official data.
Read summary.json first, then universe summary, active gates, shortlist overlay, and ITDC-specific row.

## Files

- `module_01_universe_summary.csv`: Generated artifact.
- `module_02_active_gates.csv`: Generated artifact.
- `module_03_shortlist_overlay.csv`: Generated artifact.
- `module_04_extension_rca.csv`: Generated artifact.
- `module_05_itdc_row.csv`: Generated artifact.
- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
