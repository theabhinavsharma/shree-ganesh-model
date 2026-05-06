# News-feature A/B backtest — 2026-05-06

**Question**: do news/event-window features (5d / 7d / 15d, stock + industry) add prospective lift over BASE_FEATS alone?

**Constitution gate (§1.4)**: KEEP only if Δ(top-decile precision) >= 1pp AND ΔAUC >= 0.005 vs M0_base.
Bonferroni: 6 alternative hypotheses → α = 0.05/6 = 0.0083

## Setup

- Train: 2026-02-15 → 2026-03-20 (39,841 rows)
- Test:  2026-03-21 → 2026-04-20 (34,916 rows)
- Target: 5-day forward high ≥ +3% (binary)
- Test base rate: 77.7%

## Results

| Variant | #feats | AUC | Top-Decile precision | Lift vs base | Top-5 basket max-ret |
|---|---:|---:|---:|---:|---:|
| M0_base | 15 | 0.5854 | 0.9203 | +14.6pp | +8.9% |
| M1_base+news5d | 23 | 0.5876 | 0.9331 | +15.9pp | +13.6% |
| M2_base+news7d | 23 | 0.5854 | 0.9203 | +14.6pp | +8.9% |
| M3_base+news15d | 23 | 0.5857 | 0.9258 | +15.2pp | +8.8% |
| M4_base+all_news | 39 | 0.5902 | 0.924 | +15.0pp | +10.8% |
| M5_base+industry | 21 | 0.577 | 0.9151 | +14.1pp | +9.2% |
| M6_base+all+industry | 45 | 0.577 | 0.9179 | +14.4pp | +9.7% |

## Verdict

- **M1_base+news5d**: ΔAUC=+0.0022, Δlift=+1.29pp → DROP_AB_FAIL
- **M2_base+news7d**: ΔAUC=+0.0000, Δlift=+0.00pp → DROP_AB_FAIL
- **M3_base+news15d**: ΔAUC=+0.0003, Δlift=+0.55pp → DROP_AB_FAIL
- **M4_base+all_news**: ΔAUC=+0.0048, Δlift=+0.37pp → DROP_AB_FAIL
- **M5_base+industry**: ΔAUC=-0.0084, Δlift=-0.51pp → DROP_AB_FAIL
- **M6_base+all+industry**: ΔAUC=-0.0084, Δlift=-0.24pp → DROP_AB_FAIL

**Survivors**: NONE — news features did not add lift over BASE_FEATS at this sample size.

## Honest caveats

- News data covers only 2026-02-01 to 2026-04-27 (~3 months) → train+test windows are tight.
- Sample-size noise dominates at this scale; AUC differences <0.01 are within noise floor.
- A real verdict requires backfilling news from 2018+ (MoneyControl/ET archives).
- Top-5 basket returns vary heavily across 5-day windows; treat as illustrative, not decisive.