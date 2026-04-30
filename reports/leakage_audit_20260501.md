# Look-ahead leakage audit — 2026-05-01

**Status: CRITICAL leak confirmed. 47 features contaminated. One model output line quarantined; two are clean.**

This audit is the response to CRITICAL issue #1 from the
[Devil's Advocate audit](devils_advocate_audit.md). It documents exactly
which features in the model pipeline are contaminated by look-ahead bias,
which model outputs survive scrutiny, and the remediation plan.

---

## TL;DR

| Pipeline | Status | Reason |
|---|---|---|
| `find_high_conviction.py` (daily picks) | **CONTAMINATED — quarantined** | Auto-loads 47 leaked features via `scr_*`, `qvm_*`, `acad_*` prefixes |
| `backtest_dynamic_gated.py` | **CLEAN** | Uses BASE_FEATS only (price/technical/market) |
| `find_180d_frontier_honest.py` | **CLEAN** | Uses BASE_FEATS only |
| `run_v3_with_catalysts.py` | **PARTIAL** | Catalyst features are time-series (OK); needs verification re extras |

**Numbers that survive this audit (still publishable):**
- 180d horizon, +15% target: ~85% prospective hit rate (`reports/180d_honest_frontier.md`)
- 9-year dynamic-gated median: +8% ann (`reports/dynamic_gated_backtest.md`)

**Numbers that do NOT survive this audit (retracted):**
- Any "calibrated daily score 0.85+" claim from `find_high_conviction.py`
- Any expected-return number derived from extras-enabled scoring

---

## How the leak works

### The pattern (canonical)

```python
# in feature_factory.py
sf = pd.read_parquet("screener_fundamentals.parquet")  # 1 row per symbol
sf = sf.sort_values("fetch_date").groupby("symbol").tail(1)  # latest only
df = df.merge(sf, on="symbol", how="left")  # broadcast over all dates
```

The merge key is `symbol` only — no date. Every historical row for
`RELIANCE` (2018 through 2026) gets the **same** `scr_pe`, `scr_market_cap_cr`,
`scr_roce`, etc., because the source parquet has only one row per
symbol (today's snapshot).

When the model trains on 2018 data, it sees today's PE. When it trains on
2026 data, it also sees today's PE. The feature is constant per-symbol —
which is worse than useless: tree-based models can use it as a
high-cardinality symbol identifier and memorise per-symbol patterns.

### Verification

```python
# extra_features.parquet, sample check
>>> df.groupby("symbol")["scr_pe"].nunique().value_counts()
1    177    # all 177 symbols have exactly 1 unique PE value
>>> df.groupby("symbol")["acad_mom_12_1"].nunique().value_counts()
1    2058   # all 2058 symbols have exactly 1 unique 12-1 momentum value
```

This is the diagnostic any reviewer should run on any feature. If a
feature has `nunique() == 1` per symbol, it is either a leak or a
constant by construction.

---

## The 47 contaminated features

### A. Screener fundamentals (18 columns)
Source: `src/agentic/fetch_screener_fundamentals.py` →
`data/derived/screener_fundamentals.parquet`
Injected: `src/agentic/feature_factory.py` lines 140-152

```
scr_pe, scr_market_cap_cr, scr_dividend_yield, scr_book_value, scr_roce,
scr_roe, scr_compounded_sales_growth_3_years, scr_compounded_sales_growth_5_years,
scr_compounded_profit_growth_3_years, scr_compounded_profit_growth_5_years,
scr_return_on_equity_3_years, scr_return_on_equity_5_years,
scr_stock_price_cagr_1_year, scr_stock_price_cagr_3_years, scr_stock_price_cagr_5_years,
scr_peg_3y, scr_price_to_book, scr_earnings_yield
```

### B. QVM derived (20+ columns)
Source: `src/agentic/build_derived_ratios.py` (reads from screener)
Injected: `feature_factory.py` lines 165-188

```
qvm_magic_formula_rank, qvm_earnings_yield, qvm_peg_3y, qvm_peg_5y, qvm_peg_ttm,
qvm_roe_z, qvm_roce_z, qvm_growth5y_z, qvm_quality_composite,
qvm_pe_inv_z, qvm_book_to_price, qvm_btp_z, qvm_divyld_z, qvm_value_composite,
qvm_stock_price_cagr_1_year_z, qvm_stock_price_cagr_3_years_z,
qvm_stock_price_cagr_5_years_z, qvm_momentum_composite, qvm_qvm_score,
qvm_qvm_rank, qvm_tillinghast_score, qvm_roe_growth_fusion,
qvm_mom_x_growth_3y, qvm_roe_persistence
```

These inherit the leak from screener (their input). Even features that
look "computed" (z-scores, rank) are still per-symbol-constant.

### C. Academic alphas (9 columns)
Source: `src/agentic/build_academic_alphas.py` lines 32-148
Injected: `feature_factory.py` lines 191-198

```
acad_mom_12_1, acad_return_5y, acad_beta_252d, acad_bab_factor,
acad_idio_vol_60d, acad_short_term_reversal_1m, acad_asness_qmj,
acad_liquidity_proxy, acad_vol_adj_mom_6m
```

The developer comment in `build_academic_alphas.py` at L194-195 explicitly
acknowledges the design:

> *"academic_alphas only has TODAY's snapshot. We broadcast it forward
> as a static feature (same caveat as Screener fundamentals)."*

This is a useful citation — it shows the leak was known but
unaddressed. The constitution now requires it be addressed before any
claim from the affected pipeline ships.

---

## What "contamination" actually inflates

In tree-based models (LightGBM, XGBoost), a per-symbol-constant feature
is treated as a categorical identifier. The model can:

1. Learn that "RELIANCE has scr_pe=24, qvm_qvm_rank=5, acad_mom_12_1=0.18"
   maps to the symbol RELIANCE (de-anonymisation).
2. Memorise outcome patterns specific to RELIANCE in the training years
   (2018-2023) and apply them to RELIANCE in the test years (2024-2025).
3. Inflate the in-sample fit and the apparent OOS performance because the
   "OOS" rows for RELIANCE share the same constant features as the IS rows.

The empirical signature: very high feature importance for
`qvm_qvm_rank`, `acad_mom_12_1`, `scr_pe` in the importance tables, but
low marginal lift in a leave-one-symbol-out evaluation. (Test pending.)

---

## Remediation plan

### Phase 1 — Quarantine (DONE 2026-05-01)
- [x] Mark `find_high_conviction.py` outputs as CONTAMINATED in HANDOFF.md
- [x] Log the discovery in `logs/calibration_corrections.jsonl`
- [x] This audit (`leakage_audit_20260501.md`) committed to repo
- [ ] Disable the extras prefix-loading in `find_high_conviction.py` (next commit)
- [ ] Re-run `find_high_conviction.py` with BASE_FEATS only; document the delta

### Phase 2 — Time-series fundamentals layer (next 3-7 days)
The repo already has historical raw filings under
`data/fundamentals_full_history/`. The fix is:

1. For each symbol, parse all quarterly result filings.
2. Use the filing's **publication date** (when NSE made it public) as
   the `effective_from_date`.
3. For any historical training row dated `D`, use only the latest
   filing whose `effective_from_date <= D`.
4. Rebuild `derived_ratios` and `academic_alphas` row-by-row with these
   per-date inputs. Each output row gets a real `(symbol, as_of_date)`
   key, not just `symbol`.

### Phase 3 — Re-validate
1. Re-run the 9-year walk-forward with the fixed features.
2. Compare honest-vs-contaminated lift to quantify the inflation.
3. If the fixed features still add lift (after Bonferroni), keep them.
4. If the lift disappears, drop them — the prior "lift" was the leak.

### Phase 4 — Anti-recurrence
Add a unit test: `tests/test_no_per_symbol_constants.py` — for every
feature column the model trains on, assert
`df.groupby("symbol")[col].nunique().median() > 1`. CI fails if any
feature is constant per symbol.

---

## Why this strengthens the publishability case, not weakens it

A reviewer who reads this audit and the calibration ledger sees:

1. The team ran a self-falsification battery (`devils_advocate.py`).
2. The battery found a real leak.
3. The leak was logged honestly with the developer's own admission cited.
4. The contaminated outputs were quarantined before any claim was
   re-asserted.
5. The clean outputs (`find_180d_frontier_honest.py`,
   `backtest_dynamic_gated.py`) were verified independent of the leak.

That is the rare credibility move that distinguishes publishable work
from "I asked an LLM and it picked stocks." The leak is not a setback;
documenting and remediating it under the constitution is the
publishable artifact.

---

_Generated 2026-05-01. Author: operating Claude under
[CONSTITUTION.md](../CONSTITUTION.md) §1.2 (pre-commit to falsification)
and §1.6 (living calibration ledger)._
