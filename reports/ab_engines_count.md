# engines_count sizing A/B — EXP-2026-09-24-engines-count-sizing

_generated 2026-09-23 21:09 · engine sets: data/derived/engine_replay/top30_*.parquet (walk-forward, yearly refit) · trades: data/derived/backtest_10yr_15d5pct.parquet 2020-01-06..2026-08-31 · C2 exits, next-open entry, 0.30% RT_

**Verdict: FAIL** — bar: Tier-1 beats Tier-2 on C2 net/trade by >= 0.5pp in both eras, n >= 200 per arm per era.

Gap (tier1 − tier2): disc 0.13pp (n ok: False), conf 0.18pp (n ok: True).

## Primary — backtest trade table (top-8 per week)

| era | engines_count | n | C2 net/trade % | median % | touch % | SL % |
|---|---|---|---|---|---|---|
| disc | 0 | 835 | 0.23 | 3.45 | 65.0 | 27.1 |
| disc | 1 | 250 | 0.93 | 3.45 | 67.6 | 28.4 |
| disc | 2 | 97 | 0.1 | 3.45 | 61.9 | 30.9 |
| disc | 3+ | 58 | 1.21 | 3.45 | 74.1 | 19.0 |
| disc | **tier1** (>=2) | 155 | 0.52 | 3.45 | 66.5 | 26.5 |
| disc | **tier2** (<=1) | 1085 | 0.39 | 3.45 | 65.6 | 27.4 |
| conf | 0 | 1056 | 0.66 | 3.45 | 65.2 | 24.4 |
| conf | 1 | 270 | -0.87 | 3.45 | 57.0 | 34.8 |
| conf | 2 | 117 | 0.3 | 3.45 | 65.8 | 30.8 |
| conf | 3+ | 91 | 0.84 | 3.45 | 68.1 | 26.4 |
| conf | **tier1** (>=2) | 208 | 0.53 | 3.45 | 66.8 | 28.8 |
| conf | **tier2** (<=1) | 1326 | 0.35 | 3.45 | 63.6 | 26.5 |

## Secondary (descriptive, no decision) — every QC-clean name per window, 2020-01-06..2026-08-31

| era | engines_count | n | C2 net/trade % | median % | touch % | SL % |
|---|---|---|---|---|---|---|
| disc | 0 | 43539 | 0.27 | 3.45 | 53.8 | 34.0 |
| disc | 1 | 1877 | 0.94 | 3.45 | 58.9 | 31.2 |
| disc | 2 | 288 | 0.61 | 3.45 | 61.8 | 31.9 |
| disc | 3+ | 73 | 1.16 | 3.45 | 74.0 | 20.5 |
| disc | **tier1** (>=2) | 361 | 0.72 | 3.45 | 64.3 | 29.6 |
| disc | **tier2** (<=1) | 45416 | 0.3 | 3.45 | 54.1 | 33.9 |
| conf | 0 | 96445 | -0.17 | 1.16 | 47.3 | 37.3 |
| conf | 1 | 1771 | 0.3 | 3.45 | 57.5 | 30.8 |
| conf | 2 | 340 | 0.42 | 3.45 | 64.4 | 27.1 |
| conf | 3+ | 123 | 1.13 | 3.45 | 71.5 | 22.8 |
| conf | **tier1** (>=2) | 463 | 0.61 | 3.45 | 66.3 | 25.9 |
| conf | **tier2** (<=1) | 98216 | -0.16 | 1.26 | 47.4 | 37.1 |

## Caveats

- Engines refit once per scoring year (live refits every run); labels embargoed at the scoring day.
- cs/hc extras exist only from 2023-06; median-filled before, exactly as live does.
- mh uses today's sector-index membership (not point-in-time) and catalyst_features ends 2026-06-01, both as live.
- disc era here is 2020-2022 only (engines_count needs 2018-19 history for mh).
