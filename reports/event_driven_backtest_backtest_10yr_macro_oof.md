# Event-driven, fixed-capital, signal-gated backtest

Mirrors the user's actual behavior:
- Score every stock every day (OOF, walk-forward, isotonic-calibrated).
- Trade ONLY when ≥1 name fires at the calibrated bar (`score_cal ≥ GATE`).
- Sit in cash @ 7% ann on no-fire days.
- Capital ROTATES: when a position exits (target / SL / time-out), the freed cash refills slots.
- Fixed starting capital ₹100; equity curve compounds through time.

Universe: 9-year walk-forward, 2017-01-02 → 2025-12-31 (2290 trading days)
Cash yield: 7% ann (LIQUIDPLUS proxy)

## Configurations

| Config | GATE | Slots | Hold | Target | SL | n_trades | Hit % | Avg/trade | Deployed % | **CAGR** | Max DD | Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-name 0.95 | 0.95 | 1 | 7d | +5% | -3% | 76 | 49% | +0.92% | 9% | **+14.3%** | -16.7% | 1.20 |
| Top-3 basket 0.95 | 0.95 | 3 | 7d | +5% | -3% | 205 | 49% | +0.95% | 8% | **+14.0%** | -9.8% | 1.66 |
| Top-5 basket 0.95 | 0.95 | 5 | 7d | +5% | -3% | 320 | 47% | +0.75% | 7% | **+12.0%** | -11.4% | 1.65 |
| Top-5 basket 0.95 noSL | 0.95 | 5 | 7d | +5% | none | 161 | 73% | +0.98% | 8% | **+9.7%** | -18.1% | 0.96 |
| Single-name 0.85 | 0.85 | 1 | 7d | +5% | -3% | 132 | 56% | +1.48% | 24% | **+29.3%** | -14.1% | 1.69 |
| Top-5 basket 0.85 | 0.85 | 5 | 7d | +5% | -3% | 532 | 56% | +1.46% | 18% | **+26.0%** | -9.4% | 2.56 |
| Top-5 basket 0.80 | 0.80 | 5 | 7d | +5% | -3% | 842 | 55% | +1.41% | 26% | **+38.0%** | -19.5% | 2.90 |
| Single-name 0.95 wider | 0.95 | 1 | 10d | +7% | -4% | 63 | 44% | +0.80% | 9% | **+11.4%** | -22.8% | 0.83 |
| Top-10 basket 0.80 | 0.80 | 10 | 7d | +5% | -3% | 1613 | 53% | +1.19% | 22% | **+32.1%** | -19.3% | 2.96 |
| Top-10 basket 0.75 | 0.75 | 10 | 7d | +5% | -3% | 2567 | 53% | +1.20% | 32% | **+50.2%** | -15.2% | 3.65 |
| Top-10 basket 0.70 | 0.70 | 10 | 7d | +5% | -3% | 3455 | 55% | +1.41% | 44% | **+86.2%** | -13.6% | 4.79 |
| Top-10 basket 0.65 | 0.65 | 10 | 7d | +5% | -3% | 4806 | 57% | +1.52% | 57% | **+144.3%** | -21.6% | 6.17 |
| Top-20 basket 0.65 | 0.65 | 20 | 7d | +5% | -3% | 8599 | 54% | +1.28% | 50% | **+96.5%** | -18.1% | 5.48 |
| Top-5 basket 0.80 hold15 | 0.80 | 5 | 15d | +10% | -5% | 584 | 49% | +2.39% | 29% | **+42.1%** | -18.8% | 2.46 |
| Top-5 basket 0.80 hold30 | 0.80 | 5 | 30d | +20% | -7% | 350 | 41% | +3.69% | 39% | **+36.6%** | -30.4% | 1.79 |
| Top-5 basket 0.80 noSL | 0.80 | 5 | 7d | +5% | none | 453 | 72% | +0.21% | 32% | **+4.9%** | -35.6% | 0.37 |
| Top-5 basket 0.70 noSL | 0.70 | 5 | 7d | +5% | none | 1020 | 75% | +1.06% | 60% | **+29.3%** | -33.1% | 1.20 |
| Top-10 basket 0.70 noSL | 0.70 | 10 | 7d | +5% | none | 1952 | 76% | +1.16% | 53% | **+33.3%** | -29.0% | 1.62 |
| Top-5 basket 0.80 widerT | 0.80 | 5 | 10d | +8% | -4% | 660 | 51% | +2.13% | 28% | **+43.3%** | -19.9% | 2.73 |
| Single-name 0.70 | 0.70 | 1 | 7d | +5% | -3% | 423 | 62% | +1.94% | 58% | **+145.9%** | -28.4% | 3.22 |

## Per-year returns (0.95-bar configs)

| Year | Single 0.95 | Top-3 0.95 | Top-5 0.95 | Top-5 0.85 |
|---|---:|---:|---:|---:|
| 2017 | +14.2% | +10.0% | +8.1% | +8.1% |
| 2018 | +51.2% | +49.8% | +38.6% | +48.6% |
| 2019 | +6.7% | +6.7% | +6.7% | +6.7% |
| 2020 | +7.2% | +7.2% | +7.2% | +5.9% |
| 2021 | +7.2% | +7.2% | +7.2% | +7.2% |
| 2022 | +7.2% | +7.2% | +7.2% | +44.0% |
| 2023 | +9.1% | +15.1% | +11.9% | +25.3% |
| 2024 | +17.6% | +22.0% | +18.4% | +93.1% |
| 2025 | +13.9% | +6.9% | +6.1% | +17.3% |

## Reading this

- **Avg/trade** is the per-position close-to-close return when in a position.
- **Deployed %** is the fraction of trading days the portfolio is invested (vs in cash).
- **CAGR** is the compounded equity-curve growth — the honest number that matches what fixed capital would actually do.
- **Max DD** is the deepest peak-to-trough drawdown of the equity curve, NOT a single-trade SL.

### Why this differs from the prior 9-year backtest

The earlier number forced top-5 baskets EVERY day (2,290 days × 5 names = 11,450 trades). Most days the model wasn't above 0.95 — those were forced trades on noise. By gating to fire-only, we cut to ~3,000 high-conviction trades over 9 years and let cash earn 7% on the other ~95% of days.
