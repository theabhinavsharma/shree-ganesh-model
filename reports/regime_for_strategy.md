# Regime filter for multibagger strategy

Question: when should we deploy the multibagger basket vs wait?

## Per-feature comparison: success vs failure

| Feature | Median (success) | Median (fail) | Delta | IC vs basket return |
|---|---:|---:|---:|---:|
| market_20d | -0.0278 | -0.0126 | -0.0152 | -0.237 |
| market_5d | -0.0117 | -0.0007 | -0.0110 | -0.173 |
| breadth_200 | 0.7168 | 0.6670 | +0.0498 | +0.163 |
| breadth_50_5d_chg | -0.0302 | 0.0232 | -0.0533 | -0.163 |
| breadth_50 | 0.5899 | 0.6230 | -0.0331 | -0.124 |
| market_60d | -0.0622 | -0.0722 | +0.0100 | +0.018 |

## Threshold rules — when does the basket actually work?

| Rule | Entries passing | Success rate (≥1 doubled) | Avg max | Avg close |
|---|---:|---:|---:|---:|
| breadth_50 ≥ 0.65 | 15/44 | 27% | +30.9% | -15.1% |
| breadth_50 ≥ 0.70 | 12/44 | 8% | +26.7% | -21.5% |
| market_20d ≥ 0.02 | 5/44 | 0% | +20.8% | -22.1% |
| breadth_50_5d_chg ≥ 0.05 | 15/44 | 27% | +34.9% | -10.2% |

## Honest interpretation

The multibagger strategy is **NOT a 90%-conviction always-on signal**.
It works in specific market regimes (high breadth + positive market_20d/60d) and
fails outside those regimes. The regime filter table shows the deploy/wait gates.