# Model diversity — daily ensemble

## Per-model OOS performance (2024-2025)

| Year | Model | Fit time | AUC | Top-5 hit | Basket mean 7d | Basket median 7d | Days >=+5% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2024 | LightGBM | 9.3s | 0.620 | 73.9% | +0.33% | +0.60% | 77/261 |
| 2024 | XGBoost | 4.7s | 0.618 | 74.6% | +0.74% | +1.16% | 86/261 |
| 2024 | RandomForest | 81.3s | 0.634 | 74.2% | -0.75% | -1.38% | 66/261 |
| 2024 | ExtraTrees | 7.7s | 0.602 | 66.7% | -2.94% | -2.81% | 44/261 |
| 2024 | LogisticL2 | 2.5s | 0.618 | 59.6% | -0.22% | -0.14% | 49/261 |
| 2024 | ENSEMBLE_5_AVG | 105.4s | 0.629 | 74.2% | -0.77% | -0.45% | 64/261 |
| 2025 | LightGBM | 10.7s | 0.661 | 67.1% | +0.65% | +0.18% | 57/250 |
| 2025 | XGBoost | 5.0s | 0.666 | 66.9% | +0.48% | -0.06% | 57/250 |
| 2025 | RandomForest | 109.1s | 0.674 | 68.2% | -0.09% | -0.83% | 59/250 |
| 2025 | ExtraTrees | 10.1s | 0.656 | 63.8% | -0.47% | -0.89% | 54/250 |
| 2025 | LogisticL2 | 2.4s | 0.634 | 51.3% | -1.46% | -2.05% | 32/250 |
| 2025 | ENSEMBLE_5_AVG | 137.3s | 0.671 | 67.5% | -0.36% | -0.67% | 53/250 |

## Score correlation matrix (5 models)

| | LightGBM | XGBoost | RandomForest | ExtraTrees | LogisticL2 |
|---|---:|---:|---:|---:|---:|
| LightGBM | 1.000 | 0.965 | 0.916 | 0.780 | 0.624 |
| XGBoost | 0.965 | 1.000 | 0.926 | 0.787 | 0.628 |
| RandomForest | 0.916 | 0.926 | 1.000 | 0.834 | 0.699 |
| ExtraTrees | 0.780 | 0.787 | 0.834 | 1.000 | 0.587 |
| LogisticL2 | 0.624 | 0.628 | 0.699 | 0.587 | 1.000 |

**Reading:** correlations near 1.0 = redundant models. Lower = genuine diversity.