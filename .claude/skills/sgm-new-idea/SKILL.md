---
name: sgm-new-idea
description: Test a new trading idea for the Sri Lakshmi / leader model the honest way (pre-register, then test). Use whenever the user proposes a new signal, filter, exit rule, data source for picking, or asks "does X work?".
---

# SGM new idea — register first, test second

No rule changes without a registered test. This is what stops the backtest from lying.

1. Write the idea in one sentence and what "works" means for the user's money (both periods, worst fall, hit rate).
2. BEFORE computing any outcome, append a REGISTERED line to `logs/experiments.jsonl`: id `EXP-<date>-<slug>`, hypothesis, arms, engine, eras (2019–2022 / 2023+), pass rule (default: beats the reference in BOTH eras, max drawdown not worse by more than 2 points, and in >= 4 of 5 weekly entry schedules), known limits.
3. Build the test on the shared primitives (`research_panel.py`, `sim_screen_rank_exit.run_exit`). Point-in-time inputs only.
4. Wiring check first: the new code must reproduce the reference picks exactly before any comparison counts.
5. Run it. Append the RESULT line (pass or fail). No retuning after seeing results; a new variant is a new registration.
6. Report in plain English with the numbers from the run output (receipts), including what failed and why it might be luck.

## Lessons from incidents
(appended by src/agentic/trust/incident.py; never edit or delete these lines)
- 2026-09-30 [INC-2026-09-30-verdicts-on-broken-data] After any data fix, re-run the registered tests (src/agentic/trust/reproduce.py) and re-check every adopted rule before the next live batch; say which verdicts changed.
- 2026-09-30 [INC-2026-09-30-keyword-substrings] Keyword matching on filings: match at word boundaries, remove the company's own name first, and eyeball the match counts before using them.
- 2026-09-30 [INC-2026-09-30-reproduce-pinned-to-prefix] After any data or score rebuild, re-run the registered tests and log <exp>-RERUN-<tag> lines (same rules, no re-tuning), then run reproduce.py until it says REPRODUCED before the next Friday.
- 2026-10-02 [INC-2026-10-02-ab-on-incomplete-data] No verdict on data below the bar (GIGO): every registered test calls src/agentic/trust/data_ready.gate() before computing outcomes, with completeness vs the official count, per-year coverage of every input an arm uses, and span checks; NOT READY means fix the data, not lower the bar.
- 2026-10-04 [INC-2026-10-04-derived-inputs-unscheduled] A table a strategy or test reads is an input even if a script built it once: it gets a scheduled builder that can update incrementally, a feed guard, and a ran->fetched->used check in the same change; a by-hand build is not done.
- 2026-10-12 [INC-2026-10-12-payout-reinvested-in-stub] A price-only factor below 0.5 (payout or demerger moving most of the value out) makes the adjusted series reinvest that value in the stub; any backtest position or label held across one must treat it as cash or hold the demerged shares — check raw vs adjusted returns of the best and worst picks (src/agentic/audit_ca_picks.py).
- 2026-10-12 [INC-2026-10-12-insider-gap-mar-apr-2026] When a source moves to a new system, check the handover weeks month by month (no empty month between the old history's last row and the new feed's first); a freshness check only sees the newest date.
- 2026-10-16 [INC-2026-10-16-reproduce-blocks-on-market-drift] A check that compares results across weeks must separate code drift from data drift: re-run on this week's data first, block on verdict flips of the live rule, and reproduce against the fresh numbers.
