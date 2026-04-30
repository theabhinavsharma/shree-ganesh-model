# NSE Agentic Forecasting System

> **An honest 7-day forward prediction pipeline for the Indian equity universe (NSE).**
> 5 layers, 38 hypotheses in registry, 25-step daily pipeline, hard discipline gate
> that refuses to trade on low-conviction days. No paid APIs, all free public sources.

## What it does (in 1 paragraph)

Every weekday at 18:00 IST, the system pulls fresh data from 14 sources (NSE bhavcopy,
catalysts, fundamentals, news RSS + per-symbol Google News, Reddit, YouTube, sentiment,
Wikipedia pageviews, FX, FII/DII flows). It compiles 30+ features (8 WorldQuant-101
alphas, volatility regime, microstructure, calendar, macro overlays). It retrains 4
ensembles (long, short, multi-horizon triangulation, sector-weak overlay). It runs an
8-stage discipline cascade. And it produces a per-stock dossier with explicit
Bull/Base/Bear probabilities grounded in real OOS band statistics. **If conviction
is below the floor, it refuses to surface trades.**

## Architecture (5 layers)

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full mermaid diagrams. Summary:

```
[CONTROL]   LaunchAgent daily 18:00  +  cron weekly Sun 19:00
                ↓
[DATA]      14 fetchers (prices, catalysts, news, fundamentals, FX, …)
                ↓
[FEATURE]   sector returns + macro + sentiment + WQ-101 alphas (30 features)
                ↓
[MODEL]     v3 long  +  short-side  +  multi-horizon  +  sector-weak overlay
                ↓
[DISCIPLINE] completeness audit → filter cascade (8 gates) → no-trade-if-zero
                ↓
[OUTPUT]    daily_pro_brief.md  +  status.md  +  actionable_today.csv  +  inspect_symbol
```

## Routines (just 2)

```bash
# 1. Daily pipeline (LaunchAgent — install once)
cp src/agentic/com.zoom.daily-pipeline.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.zoom.daily-pipeline.plist

# 2. Weekly hypothesis cycle (cron)
crontab -e
0 19 * * 0 cd /Users/abhinavs./Documents/Zoom && python src/agentic/agent_loop.py
```

## On-demand tools

```bash
# Inspect any stock — full dossier (price, fund, sentiment, model scores, signals)
python src/agentic/inspect_symbol.py OFSS WEBELSOLAR RELIANCE

# Refresh status dashboard
python src/agentic/build_status_dashboard.py && open reports/status.md

# Re-render workflow diagram
python src/agentic/build_workflow_diagram.py
```

## The agent loop (weekly hypothesis cycle)

`agent_loop.py` reads [`factor_registry.json`](data/derived/factor_registry.json)
which catalogs **38 alpha hypotheses across 9 categories**. Each cycle:

1. Picks N PROPOSED hypotheses
2. For each: checks data availability, queues fetcher if missing, compiles via
   `feature_factory.py`, evaluates via `factor_evaluator.py` (IC + IR test)
3. Updates verdict: `IC_PASSED` / `DROP` / `BLOCKED`
4. `IC_PASSED` factors graduate to portfolio A/B test (`backtest_10yr_with_factors.py`)
5. Survivors of A/B graduate to `KEEP` (used in production)
6. Failures become `DROP_AB_FAIL` (the lesson: high IC ≠ portfolio lift)

See state diagram in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 9-year walk-forward backtest (honest)

Top-5 daily basket, equal weight, 7-day hold, ADV ≥ ₹1cr filter:

| Year | Mean 7d | Days ≥ +5% | Days < 0 |
|---:|---:|---:|---:|
| 2017 | +1.71% | 32% | 43% |
| **2018** | **-0.22%** | 21% | **55%** |
| **2019** | **-0.66%** | 26% | **58%** |
| 2020 | +3.86% | 45% | 36% |
| 2021 | +5.77% | 49% | 30% |
| 2022 | +1.58% | 29% | 43% |
| 2023 | +6.83% | 42% | 30% |
| 2024 | +0.13% | 29% | 48% |
| 2025 | +0.20% | 22% | 52% |
| **Combined** | **+2.18%** | **33%** | **44%** |

**Realistic ann ROI: 30-50% unlevered.** Full report: [`reports/backtest_10yr_summary.md`](reports/backtest_10yr_summary.md).

## Discipline rules (binding)

1. `filter_cascade` returns 0 names → **NO TRADE** (park in cash)
2. RISK_OFF macro → patience floor jumps from 0.65 to 0.75
3. RSI > 90 or < 20 → automatic exclusion
4. ADV < ₹5cr/day → halve position size
5. Sector cap: 25% of capital
6. Single name cap: 8% of capital
7. -10% portfolio drawdown → fully cash, wait for next 0.95+ trigger

These rules are codified in code, not "guidelines" — `generate_pro_brief.py`
literally refuses to surface long picks when cascade returns 0.

## Honest limitations

- **No real-time data** — bhavcopy lands ~5pm IST; brief runs at 18:00
- **Options chain blocked** from this IP — 6 derivative-based hypotheses unusable
  (need a different host to populate)
- **2x in 90d goal** — requires 3-4× MTF leverage on top of ~30-50% base; expected
  P(success) is 25-40%, not "always works"
- **2018, 2019 were losing years** in walk-forward — the strategy isn't a magic
  printer; bear regimes hurt
- **Production v3 catalyst+sentiment lift is unstable** — +14.6pp in 2024 vs
  -0.3pp in 2025; full-feature backtest queued for validation
- **Bhavcopy gap on holidays** — pipeline auto-handles via `assign_known_date()`
  with NSE 15:30 cutoff

## Agent prompts (Anthropic-grade)

The pipeline isn't just code — every agent has a **structured system prompt**
in [`prompts/`](prompts/) written to Anthropic engineering standards (XML
tags, output contracts, quality gates, anti-patterns, reference reading).

| Agent | Prompt file | Persona |
|---|---|---|
| **System Principles** (umbrella) | [`prompts/00_SYSTEM_PRINCIPLES.md`](prompts/00_SYSTEM_PRINCIPLES.md) | Calibrated researcher; no sycophancy; honest disagreement protocol |
| **Hypothesis Generator** | [`prompts/01_HYPOTHESIS_GENERATOR.md`](prompts/01_HYPOTHESIS_GENERATOR.md) | Senior quant researcher (Renaissance / Two Sigma pedigree) |
| **Data Fetcher** | [`prompts/02_DATA_FETCHER.md`](prompts/02_DATA_FETCHER.md) | Senior data engineer (idempotent + QC + provenance) |
| **Backtest Validator** | [`prompts/03_BACKTEST_VALIDATOR.md`](prompts/03_BACKTEST_VALIDATOR.md) | Statistician + risk officer (Lopez de Prado discipline) |
| **Trade Reviewer / PM** | [`prompts/04_TRADE_REVIEWER.md`](prompts/04_TRADE_REVIEWER.md) | PM + CRO; cascade-binding; no sycophancy |
| **Meta-Reprompt** (self) | [`prompts/META_REPROMPT.md`](prompts/META_REPROMPT.md) | Claude's own self-prompt for every user turn |

These exist so:
1. **Programmatic agents** can be invoked with `system=PRINCIPLES + ROLE`
2. **Manual LLM ops** can prefix `[Apply prompts/04_TRADE_REVIEWER.md]` to scope the call
3. **Future Claude Code skills** can wrap them as `.claude/skills/<role>/SKILL.md`

References inside each prompt:
- Anthropic, "Building effective agents" (Schluntz & Zhang, 2024)
- Sharma et al., "Towards Understanding Sycophancy" (2023)
- Bai et al., "Constitutional AI" (2022)
- Lopez de Prado, "Advances in Financial Machine Learning" (2018) — backtest overfitting
- WorldQuant 101 Alphas (Kakushadze, 2015)
- Kreiger, "Productive AI" public talk (2024)

## Stack

- Python 3.9 · pandas · numpy · scikit-learn · LightGBM · XGBoost · scipy
- Pure stdlib + pandas where possible (urllib, http.cookiejar — no `requests` to
  avoid auth headers polluting NSE cookies)
- macOS LaunchAgent for persistence; cron for weekly cycle
- All data via free public endpoints (NSE bhavcopy, Frankfurter ECB FX, Wikimedia)

## Files

- 24 Python scripts in `src/agentic/`
- 1 bash orchestrator (`daily_pipeline.sh`, 25 sequential steps)
- 1 LaunchAgent plist (`com.zoom.daily-pipeline.plist`)
- 1 hypothesis registry (`factor_registry.json`, 38 entries)
- Auto-generated docs: `reports/WORKFLOW.md`, `reports/status.md`,
  `reports/daily_pro_brief_*.md`, `reports/filter_cascade_*.md`,
  `reports/backtest_10yr_summary.md`, `reports/factor_evaluation.md`,
  `reports/data_completeness_*.md`

## License & disclaimer

Personal research project. **Not investment advice. Past performance ≠ future returns.**
The system has documented losing years (2018, 2019) in walk-forward; the discipline
gate exists because forced trades on no-conviction days have been costly.
