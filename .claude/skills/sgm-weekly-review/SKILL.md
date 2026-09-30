---
name: sgm-weekly-review
description: Saturday review of the new Sri Lakshmi / model / production batches before any money moves. Use on Saturdays, or when the user asks "what are we buying this week?".
---

# SGM weekly review

1. Run `/usr/bin/python3 src/agentic/run_evals.py --cadence weekly`. If BLOCKED, stop and use sgm-triage.
2. For each track's newest screen (`logs/sri_lakshmi/`, `logs/model_screen/`, `logs/leader_sleeve/`): list the 9 picks, reserves, industry, and every flag (takeover, red-flag filing, jump > 20% without a filing, pending corporate action).
3. Compare the open paper batches with the backtest range (return, hit rate, worst dip). Say plainly if anything is outside it.
4. Prepare `evals/human_review/<track>_<screen_id>.md` from `TEMPLATE.md` with the facts filled in, boxes left for the user to tick. Real money stays blocked until the user signs.
5. Build the order sheet (`src/agentic/notify.py weekly`): 9 names, after-market limit (+5% on the last close), rupee amount each (1/9 of the batch), sell date.
6. Summarise in plain English, times in US Eastern.
