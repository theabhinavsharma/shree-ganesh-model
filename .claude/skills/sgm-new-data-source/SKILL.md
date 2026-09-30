---
name: sgm-new-data-source
description: Add a new data feed to the SGM data layer the safe way. Use when the user wants new data fetched (a new NSE endpoint, a government dataset, a macro series).
---

# SGM new data source

1. Official source first (NSE for market data, the ministry / regulator for government data). Never swap in an unofficial source silently.
2. Fetcher in `src/agentic/fetch_<name>.py`: resumable, polite, TLS verification on (this Mac is behind Netskope — use curl or the keychain bundle, never disable verification).
3. Every row carries its source URL and the date it became public. Nothing estimated; gaps stay blank and are listed.
4. Write `<file>.manifest.json`: columns, units, source, coverage, known gaps.
5. Add a QC eval to `evals/registry.yaml` (+ check in `eval_checks.py`): coverage, freshness, reconciliation against the source's own totals where possible.
6. Schedule it (daily layer, or the monthly / yearly list) and make sure the missed-run alarm covers it.
7. Spot-check 3 values against the original documents and report them.
