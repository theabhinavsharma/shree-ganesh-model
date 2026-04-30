# System Status — 2026-04-29 12:01 

_Auto-generated. Re-run: `python src/agentic/build_status_dashboard.py`_

## Last pipeline run
- File: `logs/daily_pipeline_20260428_1526.log`  (20h ago)
- Started: **8 steps**, OK: **8**, FAIL: **0**

## Background jobs (recent activity)
- `b4l30pkfb` — output file modified within last 30 min
- `bwhodh6k2` — output file modified within last 30 min
- `ba3xljxkq` — output file modified within last 30 min
- `bga4vtdwz` — output file modified within last 30 min
- `by6arx122` — output file modified within last 30 min

## Data sources

| Source | Exists | Rows | Latest | File age |
|---|:---:|---:|---|---|
| Prices | ✓ | 4,783,786 | 2026-04-28 | 15h ago |
| Catalysts | ✓ | 4,781,307 | 2026-04-27 | 17h ago |
| Fundamentals | ✓ | 1,937 | 2026-04-29 | 40s ago |
| News (raw) | ✓ | 395 | 2026-04-28 | 19h ago |
| Reddit | ✓ | 562 | 2026-04-28 | 19h ago |
| YouTube | ✓ | 45 | 2026-04-28 | 20h ago |
| News (per-sym) | ✓ | 2,137 | 2026-04-28 | 10h ago |
| Macro sent. | ✓ | 1 | 2026-04-28 | 10h ago |
| FX (USDINR) | ✓ | 558 | 2026-04-28 | 7m ago |
| FII/DII | ❌ | — | — | — |
| Wiki views | ✓ | 2,383 | 2026-04-28 | 7m ago |
| Block deals | ✓ | 15 | 2026-04-28 | 15h ago |
| Options | ❌ | — | — | — |
| Paper ledger | ✓ | 20 | 2026-04-28 | 15h ago |
| Completeness | ✓ | 93 | 2026-04-29 | 10h ago |

## Parameter completeness (today)
_Audit: 2026-04-29_

| Group | Avg coverage | # params |
|---|---:|---:|
| BLOCK_BULK | 0.4% | 13 ⚠️ |
| CATALYST | 100.0% | 12 ✓ |
| FUNDAMENTAL | 11.9% | 11 ⚠️ |
| INSIDER_PIT | 100.0% | 3 ✓ |
| MACRO_SENT | 100.0% | 6 ✓ |
| MARKET_MACRO | 100.0% | 6 ✓ |
| MODEL_OUTPUTS | 3.2% | 8 ⚠️ |
| NEWS_SOCIAL | 100.0% | 8 ✓ |
| OPTIONS_FNO | 0.0% | 6 ⚠️ |
| PRICE_TECHNICAL | 98.2% | 16 ✓ |
| SECTOR | 100.0% | 4 ✓ |

## Macro state (today)
- Global: -0.83  •  Domestic: -1.00  •  Overall: **🔴 RISK_OFF**
- USDINR sent: -1.00  •  Oil: +0.00
- Hawkish/dovish rates: 5 / 6

## Actionable picks (filter cascade output)
- ⚠️ **0 names cleared all gates today** — park in cash.

## Factor / hypothesis registry
- Total hypotheses: **38**
  - PROPOSED: 31
  - DROP: 1
  - KEEP: 3
  - EVALUATED: 3
- KEEP factors: vol_of_vol, amihud, turnover_skew

## Quick links

- **Workflow diagram**: [`reports/WORKFLOW.md`](WORKFLOW.md)
- **Today's brief**: latest `reports/daily_pro_brief_*.md`
- **Filter cascade**: latest `reports/filter_cascade_*.md`
- **Completeness audit**: latest `reports/data_completeness_*.md`
- **Factor evaluation**: [`reports/factor_evaluation.md`](factor_evaluation.md)
