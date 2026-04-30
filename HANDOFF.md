# HANDOFF — Read this first if you're a new Claude / new account

**Project name**: Shree Ganesh Model
**Repo**: https://github.com/theabhinavsharma/shree-ganesh-model (private)
**Local working directory**: `/Users/abhinavs./Documents/Zoom/` (kept historically — many Python files hardcode this path; not worth renaming)

> **Mission**: build a calibrated, anti-overfit NSE equity trading system that
> targets minimum 30% annualised, accepts up to 200%+ in best years, with
> downside floor of -30% annualised. Operating principle: discipline >
> opportunism. Devil's advocate fires before claims, not after.

This file is the **single source of truth** for picking up where the prior
session left off. Read it in 5 minutes; you'll be fully up to speed.

If you've lost your laptop and are reading this on a new machine, jump to
[`RECOVERY.md`](RECOVERY.md) — three commands and you're back operating.

**The binding operating rules live in [`CONSTITUTION.md`](CONSTITUTION.md).**
Read it before generating any claim. It is not aspirational; it is the
gate every commit, trade, and prediction passes through.

---

## 0. The first prompt to use (copy verbatim into a fresh Claude session)

```
I'm continuing a prior Claude session in /Users/abhinavs./Documents/Zoom/.
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

## 1. Core context

### The user
- Indian retail trader, capital ~₹25-50L
- Goal: double money this year (aspirational); 30% min ann (binding)
- Risk floor: -30% drawdown ann (hard cap)
- Constraints: no F&O / no options / stocks-only (per their stated preference)
- Reads English; explanations should be plain, not jargon

### The mission
- Prospectively-validated NSE equity strategy
- Daily pipeline runs at 18:00 IST and produces a trade plan
- Devil's advocate fires before any new claim ships

### What's actually true (verified, NOT marketing claims)
- **180-day horizon, +15% target, ~85% prospective hit rate** — the only honestly validated combo (uses BASE_FEATS only; survives 2026-05-01 leakage audit)
- **Dynamic gate at 0.95 calibrated**: 9-year median ~+8% ann (cash-dominated; uses BASE_FEATS only; clean)
- **2018-19 negatives gone with discipline gate** (cash on no-fire days)
- **30% min target NOT yet met unlevered**; needs market-neutral overlay or 1.5-2× MTF leverage
- **9-year backtest with strict prospective protocol**: median ROI ~+8%, range +7% to +19%

### CONTAMINATED — do NOT cite outputs from these (as of 2026-05-01)
- `find_high_conviction.py` daily picks (auto-loads 47 leaked features via `scr_*`, `qvm_*`, `acad_*` prefixes — see `reports/leakage_audit_20260501.md`)
- Any "calibrated daily score 0.85+" claim from that pipeline
- Any expected-return number derived from extras-enabled scoring

Remediation tracked under CRITICAL #1 in `reports/devils_advocate_audit.md`.

### What's a LIE that the prior Claude almost shipped (don't repeat)
- ~~"26 names today at 90% calibrated to double in 180d"~~ — overfit, in-sample calibration
- ~~"+262% expected ann ROI"~~ — based on wrong hit rate
- ~~"Joint stacking 3+ signals → 80%+ confidence"~~ — falsified empirically (signals correlate)
- ~~"5 KEEP factors found via IC test"~~ — failed portfolio A/B, demoted to DROP_AB_FAIL

---

## 2. Where everything lives

| Layer | Files | Purpose |
|---|---|---|
| **Mission docs** | `HANDOFF.md` (this) · `ARCHITECTURE.md` · `AGENTIC_README.md` · `README.md` | Read once, hands a stranger the system |
| **Agent prompts** | `prompts/*.md` (7 files) | Anthropic-grade structured prompts |
| **Pipeline orchestrator** | `src/agentic/daily_pipeline.sh` | 25-step daily run at 18:00 IST |
| **Data fetchers** | `src/agentic/fetch_*.py` (~16 scripts) | Pull NSE bhavcopy, news, FII/DII, fundamentals, etc. |
| **Models** | `src/agentic/run_v3_with_catalysts.py` etc. | Train + score |
| **Discipline gates** | `src/agentic/data_completeness.py` · `filter_cascade.py` · `monitor_for_conviction.py` | Refuse to trade when conditions not met |
| **Devil's advocate** | `src/agentic/devils_advocate.py` | Automated 10-vector integrity check |
| **Agent runner** | `src/agentic/run_agent.py` | Programmatic invocation of any prompt file |
| **Outputs** | `reports/*.md` · `data/derived/*.parquet` | Daily artifacts |
| **Calibration ledger** | `logs/calibration_corrections.jsonl` (created by this handoff) | Mistakes corrected; don't repeat |
| **Pipeline logs** | `logs/daily_pipeline_*.log` · `logs/hypothesis_loop_log.jsonl` | Run history |
| **Visualizer** | `reports/dashboard.html` | Public single-page dashboard |
| **Workflow diagram** | `reports/WORKFLOW.md` | Auto-generated from `daily_pipeline.sh` |

---

## 3. Today's known state (snapshot at handoff time)

- Latest snapshot date: **2026-04-30**
- Top high-conviction score (5%/7d, 10%/15d, 20%/30d max): **0.685**
- Names ≥ 0.95 calibrated bar today: **0**
- Dynamic-gated backtest 9-year median ann ROI: **7.0%** (range 6.9% to 18.6%)
- Devil's advocate audit: **2 CRITICAL, 3 HIGH** issues open


## 4. Open issues (in order of severity)

| # | Issue | Severity | Action needed |
|---|---|---|---|
| 1 | Screener / qvm / academic features broadcast TODAY's snapshot to all historical dates → look-ahead | CRITICAL | Fetch quarterly historical snapshots; rebuild |
| 2 | Regime gate parameters not documented as derived on prior period | CRITICAL | Train gate on 2018-22, test 2023-25; report year-by-year |
| 3 | Isotonic calibrator fit on test data → 5-10pp inflation | HIGH | Already fixed in `find_180d_frontier_honest.py`; backport pattern to all models |
| 4 | 70 frontier combos at α=0.05 → Bonferroni α should be 0.0007 | HIGH | Re-evaluate at strict significance |
| 5 | Cherry-pick "best of N" reporting | HIGH | Always report median achievable, not max |
| 6 | Survivorship: 33% of 2016 universe absent in 2025 | MEDIUM | Confirm backtest uses universe-as-of-each-date |
| 7 | LGB/XGB hyperparameters likely hand-tuned | MEDIUM | Lock parameters via search on 2018-22 only |
| 8 | Tactical 7d/30d horizons NOT honestly prospectively validated | MEDIUM | Run honest protocol like `find_180d_frontier_honest.py` did |

---

## 5. Operating rules (binding for any future Claude)

### Never do these
1. ❌ **Surface a long pick when filter_cascade returns 0** — encoded in `generate_pro_brief.py`
2. ❌ **Quote in-sample calibration as if prospective** — 90% in-sample → ~12% prospective is the gap pattern
3. ❌ **Force trades on no-fire days** — 2018, 2019 lost money historically because of this
4. ❌ **Stack signals naïvely → claim 80%+** — empirically falsified (signals correlate)
5. ❌ **Promise > 50% ann unlevered** — backtest says 30-50% is the realistic ceiling

### Always do these
1. ✅ **Run claims through `devils_advocate.py` BEFORE shipping** — not after
2. ✅ **Cite parquet path + year for every number** — "9-year median +8% from `dynamic_gated_backtest.parquet`"
3. ✅ **Lead with contradictions** — when newer evidence overrides prior claim, quote the prior claim verbatim
4. ✅ **Walk-forward strictly** — train ≤ year_X-1, calibrate on year_X, test on year_X+1
5. ✅ **Report median, not just mean** — fat-tail returns mislead

### When user pushes for more
- They will push for "double money this year" → realistic answer is "30-50% ann unlevered, 60-100% with 1.5-2× MTF"
- They will push for "trade today" → if cascade=0, the answer is "park in cash, ~7% ann"
- They will push for "should I just YOLO" → no. Sizing per `risk_envelope.py`, SLs binding
- They will push for "Claude can beat Renaissance" → no. Public Claude trading experiments delivered modest outperformance, not 40%/yr for 30 years

---

## 6. The daily flow (after handoff)

```
1. Open reports/dashboard.html in a browser   ← single visual page
2. Open reports/trade_plan_<today>.md          ← consolidated action
3. If gate green AND conviction alert fires    ← place trades per the plan
4. Else                                        ← park in LIQUIDPLUS
5. Devil's advocate audit must show 0 CRITICAL ← if not, freeze new claims
```

---

## 7. Recovery scenarios

### Scenario A: Same machine, new Claude account
1. Open `~/Documents/Zoom/HANDOFF.md`
2. Paste the §0 first prompt into a fresh Claude conversation
3. Verify Claude reads the 9 files listed
4. You're back

### Scenario B: New machine, lost local files
**Already done.** Repo lives at https://github.com/theabhinavsharma/shree-ganesh-model
On any new machine, restore the operating state with three commands:

```bash
git clone https://github.com/theabhinavsharma/shree-ganesh-model.git
cd shree-ganesh-model
# Open HANDOFF.md → paste §0 prompt into a fresh Claude session
# Run src/agentic/daily_pipeline.sh to regenerate parquets (1-2 hours)
```

See [`RECOVERY.md`](RECOVERY.md) for the canonical version of this sequence.

The parquet files are deliberately NOT in git (they're too large + regenerable).
On restore:
- Either run the daily pipeline once to regenerate them (ETA 1-2 hours)
- Or upload to S3 / Google Drive / Backblaze B2 (cheap object storage) for a hot restore

### Scenario C: Total loss (machine + cloud)
The mission docs (`HANDOFF.md` + `ARCHITECTURE.md` + `prompts/`) plus
`daily_pipeline.sh` are enough to recreate the system in ~1 day.
Everything else is regeneratable from the daily pipeline.

---

## 8. The single most important rule

> **If a future Claude (you, reader) catches a result that contradicts something
> in this handoff: lead with the contradiction. Don't smooth it over.**
>
> The prior Claude almost shipped multiple overfit claims because it didn't.
> The user pushed back and forced the corrections. Your job is to make those
> corrections automatic.

That's the bar.

---

_Generated: this file is regenerated by `src/agentic/build_handoff.py` after
every major pipeline change. If you're reading this and the file looks stale,
re-run that script to get a fresh snapshot._
