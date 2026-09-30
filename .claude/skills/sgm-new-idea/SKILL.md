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
