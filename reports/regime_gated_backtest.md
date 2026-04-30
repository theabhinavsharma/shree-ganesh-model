# Regime-gated multibagger strategy backtest

## Comparison: gated vs all-in

| Strategy | n deploys | Coverage | Success rate | Avg max % | Avg close % |
|---|---:|---:|---:|---:|---:|
| ALL-IN (no gate) | 44/44 | 100% | 41% | +38.7% | -4.8% |
| GATED (gate_v1) | 14/44 | 32% | 64% | +49.8% | +4.1% |
| GATED (gate_v2) | 19/44 | 43% | 47% | +37.7% | +4.2% |
| GATED (gate_v3) | 20/44 | 45% | 55% | +43.6% | +5.4% |

## Honest interpretation

If the GATED rows show meaningfully higher success rate and avg returns, the regime filter improves the strategy.
If they don't, the model has no clean regime gate and the all-in baseline is the honest expectation.
