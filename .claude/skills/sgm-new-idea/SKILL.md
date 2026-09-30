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
