# 2x-model qual indicators at 15D/+5% — EXP-2026-09-24-2x-qual-at-15d

_generated 2026-09-23 21:18 · weekly grid 2018-06-04..2026-09-07, investable ADV>=5cr & close>50, 294,242 stock-weeks · touch = 15d high >= +5% over signal close · C2 net from next open, 0.30% RT_

**Verdict:** IND_HOT FAIL, HOT_LEADER FAIL, EXTENDED FAIL, LOSS FAIL, CHEAP_IND FAIL, PROM_UP FAIL, ABOVE_200 FAIL

Bar (test B): tilt beats BASE top-8 by >= 4.0pp weekly touch in BOTH eras, C2 net/week not lower, pooled paired p < 0.0071.

## B — z-band pool overlay (decides)

| flag | era | weeks | BASE touch % | TILT touch % | gap pp | BASE C2/wk % | TILT C2/wk % | flagged in top-8 | p pooled |
|---|---|---|---|---|---|---|---|---|---|
| EXTENDED | disc | 234 | 58.8 | 62.3 | +3.6 | 0.01 | 0.11 | 6.7 | 7.7e-07 |
| EXTENDED | conf | 190 | 57.2 | 62.9 | +5.7 | -0.31 | 0.01 | 8.0 |  |
| IND_HOT | disc | 234 | 58.8 | 61.0 | +2.2 | 0.01 | 0.25 | 5.0 | 0.026 |
| IND_HOT | conf | 190 | 57.2 | 59.0 | +1.8 | -0.31 | 0.15 | 6.3 |  |
| CHEAP_IND | disc | 234 | 58.8 | 60.5 | +1.8 | 0.01 | 0.26 | 6.1 | 0.061 |
| CHEAP_IND | conf | 190 | 57.2 | 59.1 | +1.9 | -0.31 | 0.1 | 8.0 |  |
| LOSS | disc | 234 | 58.8 | 60.2 | +1.4 | 0.01 | -0.08 | 4.0 | 0.015 |
| LOSS | conf | 190 | 57.2 | 59.8 | +2.6 | -0.31 | -0.44 | 7.3 |  |
| PROM_UP | disc | 234 | 58.8 | 59.5 | +0.7 | 0.01 | 0.03 | 2.0 | 0.084 |
| PROM_UP | conf | 190 | 57.2 | 59.4 | +2.2 | -0.31 | 0.27 | 7.7 |  |
| ABOVE_200 | disc | 234 | 58.8 | 59.4 | +0.6 | 0.01 | 0.14 | 8.0 | 0.21 |
| ABOVE_200 | conf | 190 | 57.2 | 58.3 | +1.1 | -0.31 | -0.18 | 8.0 |  |
| HOT_LEADER | disc | 234 | 58.8 | 59.2 | +0.5 | 0.01 | 0.03 | 1.4 | 0.074 |
| HOT_LEADER | conf | 190 | 57.2 | 58.6 | +1.4 | -0.31 | -0.18 | 1.7 |  |

## A — universe-wide (descriptive): flag vs rest, ranked by the weaker era's lift

| flag | era | n flag | share % | touch flag % | touch rest % | lift pp | C2 flag % | C2 rest % | recall % |
|---|---|---|---|---|---|---|---|---|---|
| EXTENDED | disc | 32,961 | 31.3 | 64.7 | 56.6 | +8.1 | -0.09 | -0.15 | 34.2 |
| EXTENDED | conf | 44,159 | 26.5 | 65.7 | 53.3 | +12.4 | 0.09 | -0.19 | 30.8 |
| HOT_LEADER | disc | 1,866 | 2.6 | 64.5 | 58.4 | +6.1 | -0.16 | -0.09 | 2.9 |
| HOT_LEADER | conf | 2,058 | 2.4 | 66.8 | 55.7 | +11.0 | 0.13 | -0.07 | 2.9 |
| LOSS | disc | 5,506 | 7.7 | 65.7 | 59.8 | +5.9 | 0.12 | -0.0 | 8.4 |
| LOSS | conf | 9,099 | 6.1 | 62.4 | 57.0 | +5.4 | -0.2 | -0.12 | 6.7 |
| IND_HOT | disc | 5,952 | 8.3 | 63.0 | 58.2 | +4.8 | 0.08 | -0.11 | 8.9 |
| IND_HOT | conf | 7,555 | 8.8 | 59.3 | 55.7 | +3.7 | 0.06 | -0.08 | 9.4 |
| CHEAP_IND | disc | 14,546 | 47.1 | 60.9 | 56.8 | +4.1 | 0.01 | -0.03 | 48.8 |
| CHEAP_IND | conf | 27,094 | 47.5 | 57.5 | 54.3 | +3.2 | -0.04 | -0.06 | 49.0 |
| ABOVE_200 | disc | 67,265 | 63.1 | 60.1 | 57.4 | +2.8 | -0.08 | -0.24 | 64.1 |
| ABOVE_200 | conf | 107,652 | 63.4 | 57.8 | 54.5 | +3.2 | -0.05 | -0.25 | 64.8 |
| PROM_UP | disc | 2,987 | 10.7 | 59.3 | 55.6 | +3.6 | -0.24 | -0.52 | 11.3 |
| PROM_UP | conf | 12,599 | 7.6 | 59.4 | 57.5 | +1.9 | 0.04 | -0.13 | 7.8 |

## Caveats

- Industry = modal smIndustry over all announcements (not point-in-time), as in the 2x model.
- TTM needs 4 filed quarters (pnl_quarterly); PROM_UP needs two SHP quarters, known at quarter_end + 45d.
- A raw touch counts a +5% high even if C2 would have stopped out first; C2 net is the tradeable number.
- SUE, landmine filings and FinBERT tone were not retested: each failed its own registered 15d A/B (4b66c60).
