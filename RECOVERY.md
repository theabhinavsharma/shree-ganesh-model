# RECOVERY — Restore Shree Ganesh Model from cold

> If you're reading this on a fresh laptop, with no memory of the prior setup,
> in a new Claude account: **you can be back operating in under 10 minutes.**
> Everything below is the canonical, tested sequence.

---

## The three commands

```bash
git clone https://github.com/theabhinavsharma/shree-ganesh-model.git
cd shree-ganesh-model
# Open HANDOFF.md → paste §0 prompt into a fresh Claude session
# Run src/agentic/daily_pipeline.sh to regenerate parquets (1-2 hours)
```

That's it. The first command pulls the entire mission package. The second
puts you in the working directory. The third and fourth are manual steps
the operator (you) does.

---

## What you get back instantly (in the clone)

- **HANDOFF.md** — the single source of truth, current state, open issues
- **ARCHITECTURE.md** — system layout, where every layer lives
- **prompts/*.md** — 7 Anthropic-grade structured agent prompts
- **src/agentic/** — daily pipeline, fetchers, models, discipline gates,
  devil's-advocate audit (10-vector integrity battery), agent runner
- **logs/calibration_corrections.jsonl** — ledger of mistakes the prior
  Claude made; do NOT repeat
- **logs/hypothesis_loop_log.jsonl** — research history
- **data/derived/factor_registry.json** — the hypothesis catalog
- **reports/*.md** — dynamic-gated backtest, 180d honest frontier,
  devil's-advocate audit, risk envelope, achievable frontier
- **reports/dashboard.html** — single-page visualizer
- **configs/**, **data_contracts/**, **docs/**, **tests/**

## What you do NOT get back instantly (regenerable)

- All `.parquet` market-data caches under `data/`, `tmp/`, and the
  `reports/*/` replay caches
- Trained model artifacts (`*.pkl`, `*.joblib`)
- Daily pipeline log files

These are deliberately excluded from git (too large, regenerable in
1-2 hours by running `src/agentic/daily_pipeline.sh`).

---

## The first prompt to paste into a fresh Claude session

After cloning, open a new Claude conversation in the project root and paste
this verbatim (also lives in HANDOFF.md §0):

```
I'm continuing a prior Claude session. Project: Shree Ganesh Model.
Repo: https://github.com/theabhinavsharma/shree-ganesh-model
Local path on this machine: <wherever you cloned it>

Before answering anything, read these files in order:

  1. HANDOFF.md                                  ← mission + current state
  2. ARCHITECTURE.md                             ← system layout
  3. prompts/00_SYSTEM_PRINCIPLES.md             ← non-negotiables
  4. prompts/META_REPROMPT.md                    ← how to operate each turn
  5. prompts/05_DEVILS_ADVOCATE.md               ← THE most important guardrail
  6. logs/calibration_corrections.jsonl          ← mistakes the prior Claude made; do NOT repeat
  7. reports/devils_advocate_audit.md            ← unresolved overfit issues
  8. reports/dynamic_gated_backtest.md           ← honest 9-year prospective evidence
  9. reports/180d_honest_frontier.md             ← what's actually achievable

Once you've read these, summarize back to me in <300 words:
  - The verified ann ROI ceiling (with caveat)
  - The open critical issues
  - What today's pipeline output says

Then ask me what I want to work on. Do not generate trade ideas before
running the relevant claim through src/agentic/devils_advocate.py.
```

---

## Verification: how do you know recovery worked?

After the clone, you should see all of these (run from project root):

```bash
ls HANDOFF.md ARCHITECTURE.md README.md RECOVERY.md            # 4 files
ls src/agentic/ | wc -l                                        # ~50+ Python files
ls prompts/*.md                                                # 7 prompt files
ls logs/calibration_corrections.jsonl                          # 7 entries minimum
ls data/derived/factor_registry.json                           # the catalog
ls reports/*.md | wc -l                                        # ~30+ markdown reports
```

If any are missing, the clone failed — re-run `git pull`.

---

## Re-running the pipeline (regenerate parquets)

Once you've read HANDOFF.md and the prompts, restore the data:

```bash
# requires Python 3.10+, pandas, lightgbm, xgboost, scikit-learn
pip install -e .[dev]   # or however you set up the env

# regenerate everything
bash src/agentic/daily_pipeline.sh
```

ETA: 1-2 hours on a 2024-era Mac. Outputs land in `data/derived/`,
`reports/`, and `logs/`. After this you have a fully-functional system.

---

## If GitHub is also gone (worst case)

The mission docs alone (HANDOFF.md + ARCHITECTURE.md + prompts/*.md +
src/agentic/daily_pipeline.sh) are enough to recreate the system in
~1 day from scratch. Everything else is regeneratable from public NSE
data via the daily pipeline.

Print this file. Email it to yourself. Save a copy to a USB stick.
The point of `RECOVERY.md` is that it survives even when nothing else does.
