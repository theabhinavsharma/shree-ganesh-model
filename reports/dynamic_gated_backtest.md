# Dynamic-Gated Backtest — does sitting in cash on no-fire days fix 2018-19?

Method: every day, model scores all stocks. If any name has calibrated score ≥ 0.95, take top-5 of them (basket). Otherwise → sit in cash @ 7%/yr.

Walk-forward: train on years ≤ yr-2, calibrate on yr-1, test on yr (strictly prospective).

## Per-year results

| Year | Trading days | Fire days (0.95+) | Fire % of days | Basket return when fires | Hit rate | **Blended ann ROI** |
|---|---:|---:|---:|---:|---:|---:|
| 2018 | 246 | 112 | 46% | +0.18% | 53% | **+7%** |
| 2019 | 244 | 0 | 0% | +0.00% | 0% | **+7%** |
| 2020 | 261 | 27 | 10% | +2.94% | 63% | **+19%** |
| 2021 | 261 | 0 | 0% | +0.00% | 0% | **+7%** |
| 2022 | 259 | 0 | 0% | +0.00% | 0% | **+7%** |
| 2023 | 260 | 0 | 0% | +0.00% | 0% | **+7%** |
| 2024 | 261 | 5 | 2% | +9.78% | 100% | **+14%** |
| 2025 | 250 | 0 | 0% | +0.00% | 0% | **+7%** |

**Strategy: GATE = 0.95 calibrated; trade only on fire days; cash on others.**
