# Models by market regime (exploration, 2026-10-16)

Question (user): among all the models and rules we have used, does a different one do better in a certain kind of market?

Answer: no. This covers 348 weekly lists from Apr 2019 to Apr 2026, each held 6 months, after costs. In 8 market states (trend up/down, breadth wide/narrow, India VIX high/low, FPI buying/selling), no model or rule beat V3.2 by a margin outside the bootstrap range.
- The closest challengers were within about 1 point, with 95% ranges spanning zero:
  - V3.1 in down-trends and narrow breadth (+0.4 / +0.8).
  - 7 hot + 2 mid industries in up-trends, high VIX and FPI buying (+0.4 to +0.7).
- Every significant difference went against the challenger: V3, no hot-industry gate, and single models (logit, LightGBM) in several states.
- The ensemble beat every single model in every state; XGBoost was closest.
- Each state holds only about 4-9 independent 6-month periods. Switching models by regime would be tuning to noise. This matches the two failed market-timing tests (EXP-2026-10-10-v3-regime-switch, EXP-2026-10-11-v31-breadth-pause).

Files: weekly_lists.csv (regime tags + each arm's 6-month return per list), by_regime.csv (per state and arm), manifests, run.log.
Exploration only: any rule built on this needs a registered walk-forward test.
