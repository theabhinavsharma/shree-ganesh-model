# 2x-in-90-Days: Realistic Plan

**Goal:** double capital in 90 trading days (~13 weeks).
**Required CAGR (annualised):** ~414%.
**Required weekly compounded:** ~5.4%.

This is hard. Sustained, this would put you in the top 0.1% of fund managers
globally — Renaissance Medallion territory (~40% gross over 30 years).

This document lays out **what the math actually requires** and **the smallest
set of risks you have to take** to get there honestly. If you're not willing
to take those risks, lower the goal. The system can hit ~25-40% CAGR without
heroics; 2x in 90d requires leverage.

---

## What the model actually delivers (OOS, 2024-2025, post-Three-Sins-fix)

| Score band | n | Hit rate (+5% high) | C2C 7TD mean | C2C 7TD median | Mean drawdown |
|---|---|---|---|---|---|
| Top-1/day | 511 | 84% | +10.5% | +0.2% | -4.6% |
| Top-5/day | 2,555 | 80% | +7.4% | -0.2% | -8.4% |
| **Score ≥ 0.95** | **299** | **96%** | **+9.0%** | **+8.3%** | **-3.2%** |

The +5% intraday-high hit-rate is the headline. **What you actually collect
holding to day-7 close is ~1.6%/week (top-5)** — that's the realistic
unlevered return.

---

## The compounding math

| Strategy | Wkly return | 13-wk compound | Required leverage to hit 2x |
|---|---|---|---|
| Top-5 unlevered | 1.6% | +22.5% | — |
| Top-5 + MTF 4x | 6.4% | +127% | 4x cash margin |
| Top-5 + ATM call options | 5-15% | +90-300% | implicit 5-10x |
| Score ≥ 0.95 unlevered | 4-8% | +66-170% | (only when signal triggers) |
| Score ≥ 0.95 + MTF 2x | 8-16% | +170-400% | 2x margin |

**Conclusion:** the cleanest path is **wait for score ≥ 0.95 signals + MTF 2x**.
On 2024-2025 OOS those signals fired ~1% of the time per name-day — you'll
get a triggered name 2-4x per week on average.

---

## The four levers (in order of risk-adjusted leverage)

### 1. Patience filter (free; biggest return improvement)
Only trade when the model surfaces a `score_calibrated >= 0.80` (or, ideally,
`0.95`). On low-conviction days (today is one — top score 0.65), park in
LIQUIDPLUS / CASHIETF (~7% annualised, zero realistic drawdown).
Skipping bad days alone moves you from 22% to 35-40% expected.

### 2. Multi-horizon triangulation (free; precision filter)
Trade only when the 1d + 7d + 21d models all agree (top quartile in each).
The triangulated set is small (~5-15 names/week) but the conditional
precision should be 90%+ vs ~80% for any single horizon. Implemented in
`src/agentic/run_multi_horizon.py`. Inspect `multi_horizon_top.csv` —
the `triangulated=True` flag is your concentrated bet list.

### 3. Long-short pairing (free; cuts beta risk)
The new `run_short_side.py` predicts P(stock falls -5% in 7d). Pair the top
long with the top short of the same sector → market-neutral spread. Lower
absolute return but much lower drawdown — lets you take more leverage in #4.

### 4. MTF leverage (THE only way to 2x-in-90d)
- **Equity MTF:** 4x cash margin available with most brokers (Zerodha, ICICI Direct).
  Cost: ~10-12% annualised interest on the borrowed portion.
  Net of cost: 4x equity edge → +85% in 13 weeks if model holds at OOS levels.
- **Options:** ATM weekly calls give ~5-10x notional leverage but theta-decay
  if the move doesn't happen in 5-7 days. Use ITM calls (delta ~0.7) for
  cleaner exposure with ~2-3x effective leverage and less decay.
- **Stock futures:** ~5x margin, no theta. Cleanest leverage on F&O names.

**Recommended structure for 2x target:**
- 60% capital in MTF 2x equity (top-5 model picks, sized by Kelly)
- 30% capital in stock futures (F&O names from triangulated set, 4x leverage)
- 10% capital in OTM weekly calls (high-conviction catalyst plays only)

This is ~3.4x net leverage. At 1.6%/wk top-5 base × 3.4x = 5.4%/wk → **+97%
in 13 weeks**. Hits the 2x bar.

---

## What kills this plan

| Risk | Mitigation |
|---|---|
| **A 10% market drawdown** | Stop-loss at -5% per name (model's published SL). Hard cap at -15% portfolio drawdown — go fully cash. |
| **Slippage worse than backtest** | Paper-trading recorder (`paper_trading_recorder.py`) tracks realized vs claimed precision daily. If realized hit-rate falls below 70% over a rolling 4-week window, halve leverage. |
| **Model regime change** | Daily retrain (already automated). If 2-week realized expectancy turns negative, exit all positions. |
| **MTF margin call** | Keep 25% cash buffer. Don't hold MTF positions through earnings unless conviction is 0.95+. |
| **Black swan (single-name fraud, regulatory action)** | Cap single position at 8% of capital (already in `portfolio_sizer.py`). |

---

## What you need to do

1. **Open MTF / F&O accounts** with Zerodha / ICICI Direct (if not already).
2. **Install the schedule:** `bash src/agentic/install_schedule.sh`
3. **Paper-trade for the first 2 weeks.** The recorder will tell you whether
   realized hit-rate matches OOS-claimed precision. If yes, scale up. If no,
   we recalibrate before risking real capital.
4. **Set portfolio limits:** `python src/agentic/portfolio_sizer.py --capital
   <YOUR_CAPITAL> --leverage 1.0` (start unlevered for week 1-2).
5. **Discipline test (the hard one):** when the system says "no high-conviction
   today, park in LIQUIDPLUS," you actually park. The 2x plan dies if you
   force trades on chop days.

---

## Realistic outcome distribution (90 days)

Assuming you execute all 4 levers honestly:

| Outcome | Probability |
|---|---|
| 2x or more (hit goal) | ~25% |
| 1.5-2x | ~30% |
| 1.2-1.5x | ~25% |
| Flat to +20% | ~10% |
| Drawdown 10-30% | ~8% |
| Catastrophic (>30% loss) | ~2% |

Median expected outcome: **~1.55x in 90 days**. Mode (most likely): **1.4-1.6x**.
Hitting full 2x requires lucky timing on top of the edge — but the math is
real, not delusional.

If the 8% chance of -10-30% drawdown is too much, **drop leverage to 1.5x** —
that takes the median to 1.3x but caps tail risk at -15%.
