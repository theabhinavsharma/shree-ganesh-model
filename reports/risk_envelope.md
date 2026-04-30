# Risk envelope — bounds & basket fit

## User-defined envelope

| Bound | Value | Meaning |
|---|---:|---|
| Min target | **30% ann** | Don't bother if expected return is below this |
| Max target | **200%+ ann** | Aim for double-money or better in best case |
| Max drawdown floor | **-30% ann** | Worst-case annualised loss we accept |

## Today's multibagger basket

- **5 names selected** after liquidity-aware sizing + 95% cap
- **100% deployed** · 0% cash buffer
- **Avg calibrated score: 0.976** (≥ 0.86 = 90% to double)

## Per-name allocations

| Symbol | Score | ADV cr | Size | Stop-loss | Target |
|---|---:|---:|---:|---:|---:|
| **KOTAKBANK** | 0.997 | 691.1 | 20% | -15% per name | +100% per name |
| **ANGELONE** | 1.000 | 435.2 | 20% | -15% per name | +100% per name |
| **ROLEXRINGS** | 0.973 | 196.8 | 20% | -15% per name | +100% per name |
| **TATAINVEST** | 1.000 | 168.1 | 20% | -15% per name | +100% per name |
| **AGIIL** | 0.910 | 141.3 | 20% | -15% per name | +100% per name |

## Outcome scenarios

| Scenario | Per-turn return | Annualised (~2 turns/yr) |
|---|---:|---:|
| **Bear** (all names SL) | -15.0% | -28% |
| **Worst plausible** (4/5 SL, rest hit) | +8.0% | +17% |
| **Expected** (90% hit rate per name) | -0.7% | -1% |
| **Bull** (all hit target) | +100.0% | +308% |

## Constraint check

| Constraint | Pass? | Reading |
|---|:---:|---|
| Expected ann ≥ 30% min target | ❌ | expected -1% |
| Bull ann ≥ 200% (2x ceiling reachable) | ✅ | bull +308% |
| Worst-plausible ann ≥ -30% (downside floor) | ✅ | worst-plausible +17% |
| Bear (all SL) ann ≥ -30% (absolute floor) | ✅ | bear -28% |

### Recommended deployment cap

To keep bear-case annualised ≥ -30%, deploy **at most 100%** (rest in cash / LIQUIDPLUS).
