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

## Lessons from incidents
(appended by src/agentic/trust/incident.py; never edit or delete these lines)
- 2026-09-30 [INC-2026-09-30-insider-feed-silent-zero] A feed that returns 0 rows is broken until the source proves 0 is real (compare the same window a year earlier), and every feed gets a freshness eval.
- 2026-09-30 [INC-2026-09-30-renamed-symbol-cliffs] Key corporate data by company identity (ISIN issuer + NSE symbolchange.csv), never by ticker alone, and check the whole history for one-day moves of 45%+ without an NSE record, not only the new rows.
- 2026-09-30 [INC-2026-09-30-archive-dii-wrong] When restoring or combining two sources, compare them on the overlap first and use only the stretch where they agree.
- 2026-09-30 [INC-2026-09-30-xbrl-promoter-member] In filings with several tables (shareholding XBRL), check which table a category sits in before summing it; test the sum on a known company.
- 2026-09-30 [INC-2026-09-30-keyword-substrings] Keyword matching on filings: match at word boundaries, remove the company's own name first, and eyeball the match counts before using them.
- 2026-09-30 [INC-2026-09-30-api-page-cap] A suspiciously round count per request (exactly 70, 100, 500) is a page cap, not the data: find the full export before backfilling.
- 2026-09-30 [INC-2026-09-30-feeds-without-guards] Add every new feed to configs/feed_guards.json in the same change (file, date column, allowed weekdays); data.every_feed_guarded blocks the daily run for a feed without one. Rules that matter go into an eval, not only into a skill.
- 2026-09-30 [INC-2026-09-30-model-scores-stale] Every input a live screen reads (model scores, market caps, industry scores) needs a scheduled rebuild and a check that stops the run when it is stale; a warning on a live input is not a guard.
- 2026-10-02 [INC-2026-10-02-weekly-freshness-deadlock] A gate may only require inputs that are produced before it runs; outputs refreshed later in the same run belong to that later step's own check. Count file age in NSE sessions (nse_calendar), never plain weekdays.
- 2026-10-02 [INC-2026-10-02-ab-on-incomplete-data] No verdict on data below the bar (GIGO): every registered test calls src/agentic/trust/data_ready.gate() before computing outcomes, with completeness vs the official count, per-year coverage of every input an arm uses, and span checks; NOT READY means fix the data, not lower the bar.
- 2026-10-02 [INC-2026-10-02-background-job-unwatched] Every background fetch gets a time budget and a stop rule (e.g. >50% failures after 5 tries) in code, and I check it at its expected finish time; never leave a job running unwatched past its ETA.
- 2026-10-04 [INC-2026-10-04-derived-inputs-unscheduled] A table a strategy or test reads is an input even if a script built it once: it gets a scheduled builder that can update incrementally, a feed guard, and a ran->fetched->used check in the same change; a by-hand build is not done.
