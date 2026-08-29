# April-1 engine replay — evaluation vs Apr-1→Aug-27

Universe 1113 investable · 80 doubled from Apr-1 close · base rate 7.2%.
'mean end ret' = equal-weight buy Apr-1 close → Aug-27 close (the sleeve view).

| flag list | n | 2x hits | lift | mean end ret | mean peak ret | mean maxDD | ≥+25% |
|---|---|---|---|---|---|---|---|
| mb: clears any 100% bar | 0 | — | — | — | — | — | — |
| mb: top 20 by best 100% score | 20 | 0 (0%) | 0.0x | +12.8% | +57.4% | -15.3% | 85% |
| mb: top 20 by score_50pct_180d | 20 | 3 (15%) | 2.1x | +18.4% | +59.8% | -9.1% | 80% |
| mb: top 20 by score_75pct_180d | 20 | 3 (15%) | 2.1x | +17.2% | +59.4% | -10.1% | 75% |
| 180d: top 20 by score_5pct | 20 | 0 (0%) | 0.0x | +22.3% | +40.9% | -6.8% | 80% |
| 180d: top 20 by score_10pct | 20 | 2 (10%) | 1.4x | +27.6% | +59.7% | -12.2% | 70% |
| 180d: top 20 by score_15pct | 20 | 3 (15%) | 2.1x | +23.3% | +66.0% | -11.6% | 90% |
| 180d: top 20 by score_20pct | 20 | 2 (10%) | 1.4x | +20.2% | +64.2% | -12.4% | 95% |
| hc: top 10 by score_5pct_7d_cal | 10 | 1 (10%) | 1.4x | +18.0% | +61.5% | -7.7% | 100% |
| hc: top 20 by score_5pct_7d_cal | 20 | 2 (10%) | 1.4x | +12.1% | +62.7% | -12.2% | 100% |
| hc: top 10 by score_10pct_15d_cal | 10 | 1 (10%) | 1.4x | +33.5% | +70.8% | -6.8% | 100% |
| hc: top 20 by score_10pct_15d_cal | 20 | 2 (10%) | 1.4x | +21.6% | +61.6% | -10.9% | 90% |
| hc: top 10 by score_20pct_30d_cal | 10 | 2 (20%) | 2.8x | +24.6% | +70.3% | -8.9% | 100% |
| hc: top 20 by score_20pct_30d_cal | 20 | 3 (15%) | 2.1x | +22.9% | +68.0% | -9.0% | 100% |


## CORRECTED — flag lists restricted to the investable universe (ADV≥5cr, >₹50 at Apr-1)

Benchmark: universe equal-weight Apr-1→Aug-27 mean end **+25.1%**, mean peak +44.6%, maxDD -6.8%, 2x rate 7.2%.
(The uncorrected table above let sub-₹50 penny names into hc/180d lists — engines only floor ADV≥1cr.)

| list (investable) | n | 2x | prec | end | peak | maxDD | ≥+25% |
|---|---|---|---|---|---|---|---|
| mb top20 50pct_180d | 20 | 2 | 10% | +14.7% | +45.8% | -8.1% | 75% |
| mb top20 best_100pct | 20 | 0 | 0% | +13.2% | +43.9% | -8.0% | 85% |
| 180d top20 score_15pct | 20 | 2 | 10% | **+32.9%** | +57.4% | -6.3% | 85% |
| 180d top20 score_30pct | 20 | 3 | 15% | +21.2% | +54.6% | -9.3% | 75% |
| 180d top20 score_50pct | 20 | 3 | 15% | +22.8% | +58.4% | -8.6% | 85% |
| hc top10 score_5pct_7d | 10 | 1 | 10% | **+30.5%** | +67.9% | -7.6% | 100% |
| hc top20 score_5pct_7d | 20 | 3 | 15% | **+31.0%** | +68.7% | -8.9% | 90% |
| hc top10 score_20pct_30d | 10 | 1 | 10% | **+30.9%** | +64.7% | -7.6% | 90% |

Verdicts: (1) mb's 100% bar cleared ZERO names — the "90% calibrated multibagger" engine is
inert as designed, and its rank order carries no 2x signal (0/20). (2) No list transforms 2x
odds — best precision 15% ≈ 2.1x lift, matching the C6 miner cell. (3) Modest sleeve edge:
hc/180d top lists beat the +25.1% tape by +5-8pp over 5 months at similar drawdown — real but
single-window. (4) Peaks (+55-69%) vs ends (+21-33%): the exit design, not selection, is where
most of the money leaks — a trailing-stop hold sleeve captures what fixed holds hand back.
