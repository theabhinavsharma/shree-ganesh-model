# Research-code audit — 2026-09-27

Workflow `audit-sgm-research-code` (run wf_b29fab98-db7): one reviewer per script across 7 bug lenses, every finding put to 3 independent skeptics told to refute it (confirmed = at least 2 of 3 upheld), plus a completeness critic. 247 agents.
**73 confirmed** (13 high, 23 medium, 37 low); 6 refuted. Full evidence and skeptic notes: `audit_20260927_research_code.json`.

## Root causes behind the high-severity findings
1. **Panel lacks BE/BZ (trade-for-trade) sessions before 2025**: windows counted in rows stretch over 100-300+ sessions, and returns across gaps are zeroed by fillna(0). Affects anatomy labels, leader-cell holds, all portfolio NAVs.
2. **Back-adjusted close used as a price level** (`close > 50`, log_px, PE): it encodes future splits/bonuses, so it's lookahead in universe filters and model features.
3. **mcap_pit construction**: share counts lag splits by months, the screener backfill can use another entity's shares after renames, and nothing is point-in-time before 2018-04.
4. **Order amount = largest figure in the text**: picks revenue/order-book boilerplate; USD converted at a fabricated 83.0 before Feb-2024; the order regex misses and mislabels categories.
5. **Corporate actions beyond split/bonus are unadjusted** (demergers, rights, capital reductions): fake one-day crashes and jumps in the adjusted panel (critic).

## Confirmed findings

### 1. [HIGH] src/agentic/anatomy_1p5x.py — lines 65-67, 210-212, 261-262  (3/3 upheld)
- **Lens:** LOOKAHEAD / PLAIN LOGIC (label definition, fold leakage)
- **Claim:** The labels y95, y63 and s95 are built over the next 95 panel rows per symbol, not the next 95 sessions. Before 2025 the panel is missing sessions for many small names (the known BE/T2T gap, which was backfilled only for 2025+). For those names a '95-session' window stretches over 100 to 300+ sessions, which inflates positives. The same stretched windows also carry training labels past the walk-forward cut into the test year.
- **Impact:** This overstates the headline results. Recomputed without the gappy rows, conf-era lifts fall below the 1.5x bar for uc20 (1.54→1.37), ret126 (1.54→1.42), px_sma200 (1.55→1.45) and log_px d1 (1.53→1.21). The log_mcap bottom decile goes from 1.46x to 0.92x in conf. So 3-4 of the 7 'both-era' Part 1 buckets fail. Model: in conf, 30.8% of the top-1% predictions are gappy rows, against a 5.4% base. Excluding them, conf AUC drops 0.640→0.609 and top-1% lift drops 2.34x→1.79x; disc AUC drops 0.580→0.561. The model is partly learning 'will have missing sessions' (T2T surveillance after a run-up). Part 3 is less affected because core rows are only 1.4% gappy.

### 2. [HIGH] src/agentic/anatomy_1p5x.py — lines 68, 76-77, 86, 123-124, 301  (3/3 upheld)
- **Lens:** LOOKAHEAD / UNITS (adjusted price vs as-reported values)
- **Claim:** The adjusted close is back-adjusted for splits and bonuses that happen after the entry date. Several features and filters use it directly: log_px, the core filter `close > 50`, and pe = adjusted close / as-reported eps_ttm. So they carry future corporate-action information, and pe mixes an adjusted price with unadjusted EPS. The file has `price_adjustment_factor_to_present`, and build_mcap_pit.py already uses it to undo this, but the anatomy does not.
- **Impact:** Two of the seven headline Part 1 'both-era' findings (pe d1, log_px d1) are largely artefacts and fail the 1.5x bar in conf once price is on the then-current basis. log_px is the #4 out-of-sample permutation feature (AUC drop 0.0138) and carries future split information into the model. The core misclassification removes later winners from every Part 3 arm, from the EW benchmark and from the mkt_breadth denominator. That is a small downward bias shared by all arms.

### 3. [HIGH] src/agentic/build_mcap_pit.py — lines 40, 46-53 (esp. 48 rolling median, 51-53 merge_asof + mcap = raw_px x shares)  (3/3 upheld)
- **Lens:** UNITS / data contract (share-count basis vs price basis) + PLAIN LOGIC
- **Claim:** On every split or bonus, pnl_implied market cap drops by the split factor and stays wrong for months. raw_px switches to the new share basis on the ex-date, but shares (PAT/EPS from filings) stay on the old basis until new filings take over the 4-quarter rolling median, which needs 2-3 post-split filings. When the latest filings have NaN shares, qq.dropna drops them and merge_asof keeps the stale pre-split count for up to 400 days.
- **Impact:** Market cap is understated 1.5-10x (median 2x) for about 8 months after each split or bonus. 2,110 rows fall below the Rs 50cr bar only because of this (false exclusions). The anatomy_1p5x `log_mcap` feature is off by about 0.3 log10 on ~70k rows, and those rows are not random: bonuses and splits usually follow strong price run-ups, so the error is correlated with momentum and future-return labels, and 70% of it sits in the conf era. Fix: rebase each quarter's shares to the present basis before the median (sh / factor_at(filing_dt)), then multiply by adjusted close.

### 4. [HIGH] src/agentic/build_mcap_pit.py — lines 65-73 (backfill shares x old symbol's adjusted close); producer src/agentic/fetch_screener_mcap_backfill.py:1-8,35-45  (3/3 upheld)
- **Lens:** LOOKAHEAD / UNITS (wrong entity's share count)
- **Claim:** screener_backfill sizes 276 symbols with another company's current share count: renamed successors, merger acquirers and re-listed post-insolvency entities, because the fetcher follows renames. The old symbol's price factor is 1.0 (the successor's later corporate actions are not carried back), so present successor shares are multiplied by an unadjusted old price. Line 70 never checks that the page belongs to the symbol.
- **Impact:** Mostly overstatement: 3,548 rows are falsely included in the Rs 50cr universe and 139 falsely excluded, from the rename pairs alone. Size features for these names are wrong by 2-23x across their pre-rename history, which falls mostly in the disc era (2015-2022). Fix: require slug == symbol, or rebase with the successor's corporate actions and check against the successor's first-day mcap; otherwise leave NULL.

### 5. [HIGH] src/agentic/build_mcap_pit.py — lines 1-13 (docstring), 56-63 (backcast), 65-73 (backfill), 77-92 (validation + manifest)  (3/3 upheld)
- **Lens:** LOOKAHEAD / STATISTICAL VALIDITY (era asymmetry)
- **Claim:** The output is not point-in-time before 2018-04, and it is much less point-in-time in the disc era than in the conf era. pnl_quarterly filing_dt starts 2018-04-02, so 2015-2017 has zero pnl_implied rows and every sized row uses 2026 (or last-trade) shares times adjusted close. That share count already includes later QIPs, preferential issues, mergers and debt conversions, and the error runs one way. The manifest calls the dataset 'point-in-time' and only validates the latest pnl_implied row per symbol, so it never tests backcast, backfill or history.
- **Impact:** Size features and the Rs 50cr filter use share counts from the future in the disc era, which overstates names that later raised equity (often later winners) and biases disc-era size results. Measurement quality also differs by era, so 'holds in both eras' comparisons partly compare construction methods rather than markets. Downstream consumers (anatomy_1p5x, sim_leader_cell_v2, sim_leader_portfolio_7x, event_materiality_study) ignore mcap_source.

### 6. [HIGH] src/agentic/event_materiality_study.py (parse_amount_cr) + src/agentic/fetch_order_fulltext.py — lines event_materiality_study.py 47-59; fetch_order_fulltext.py 76, 91  (3/3 upheld)
- **Lens:** 6. PARSING / REGEX (+3 units of meaning)
- **Claim:** parse_amount_cr returns the LARGEST amount anywhere in the text. On full attachment text that is usually boilerplate: company revenue, group size, order book or cumulative intake, not the order. Then `D["amount_cr"] = D["fulltext_amount_cr"].fillna(D["headline_amount_cr"])` lets that wrong full-text figure replace a correct headline amount.
- **Impact:** Boilerplate revenue makes amount/TTM revenue about 1.0, so every order press release from such filers is flagged 'material' (>=15% of revenue) even when the real order is under 1% of revenue (Siemens 78 cr vs ~13,000 cr revenue). Likely also hits LT, the #1 filer with 200 order filings, whose press releases carry a 'USD xx billion' boilerplate; this is my inference and not yet checked in the crawl. At the current ~39% error rate among OK rows, the material-order cell in test_hot_order_combo will be substantially made up of large caps with non-material orders. That dilutes (probably understates) any true material-order effect, and 'amount_cr' cannot be trusted without a context filter (skip about/revenue/order-book/group sentences, or prefer the headline amount).

### 7. [HIGH] src/agentic/model_bakeoff_1p5x.py — lines 5, 34, 48-49 (label built in src/agentic/anatomy_1p5x.py:65-66)  (3/3 upheld)
- **Lens:** LOOKAHEAD / train-test leakage + PLAIN LOGIC (label horizon)
- **Claim:** The claim that every training target window is closed (line 5) is false, and y95 does not mean 95 sessions for about 6% of rows. The label counts forward per-symbol ROWS: anatomy `fwd_max = ... s.shift(-1)[::-1].rolling(n, min_periods=76)...`. Before 2025 the panel has no BE/BZ-series sessions, so for names that moved to T2T/BE the 95-row window skips the gap and covers far more calendar time. The fixed `cut = Y-01-01 - 150 days` embargo (line 48) then does not close those windows, and the long-horizon rows carry inflated positive labels that the models learn and preferentially pick.
- **Impact:** The headline weekly top-10 tradable hit rate is inflated, mostly in the conf era because 2023-24 test years are affected: lgbm conf 28.70% falls to 26.40% without the long-span picks (about -2.3pp, lift 2.14x to about 1.97x); disc 26.54% falls to 25.94%. It affects all models and probably favours whichever one best exploits gap-driven tape features. A small amount of test-period outcome information also leaks into training (0.3-0.9% of training rows per fold). The conf-era model comparison mixes a labelling regime that ends in 2025.

### 8. [HIGH] src/agentic/sim_allin_matrix.py — lines 28-30, 71  (3/3 upheld)
- **Lens:** 4. PORTFOLIO ARITHMETIC (also 3. data contract)
- **Claim:** Every missing session in a held name's price series silently deletes the whole price move across the gap. The script pivots onto the union date index and calls `pct_change(fill_method=None)`, so both the missing day and the first day back come out NaN. `R.iloc[...][names].fillna(0)` then books them as 0% returns. The panel still has no BE/T2T-series sessions before 2025 (the BE-series gap incident; backfill was 2025+ only). Momentum leaders are exactly the names NSE moves into T2T with 5% bands, so the rule approaches lose their surveillance-period crashes.
- **Impact:** Rule-approach CAGRs are overstated by up to ~4.4pp at h=90 and ~1-2pp at h=30, much less at h=250. Worst-phase numbers are overstated by up to 6pp. The bias is selective: rules hit surveillance names far more than the model or EW, so it changes the h=90 ranking. As reported, BASELINE (27.5) beats MODEL top-10 (24.2); corrected, BASELINE is 22.6-23.2 against MODEL 23.7-25.2.

### 9. [HIGH] src/agentic/sim_leader_cell_v2.py — lines 72-76, 80, 83-86 (ret60/ret252 via pct_change(60/252); exit_i = start + min(pos+HOLD, n-1); hi_f/lo_f rolling(HOLD) on symbol rows)  (3/3 upheld)
- **Lens:** PLAIN LOGIC / PORTFOLIO ARITHMETIC (row-count windows over data gaps)
- **Claim:** The '126td' hold, the forward high/low window, and the ret60/ret252 lookbacks all count a symbol's own rows, not market sessions. Wherever the panel has gaps, a 126-row hold runs much longer than 126 market days: pre-2025 BE/T2T sessions are still missing from the panel (IZMO has a 300-day hole from Jun 2023 to Apr 2024), and sparse or suspended names do the same. The longer holds carry much larger returns, so mean/trade, P2x and P50 are inflated. This is worst in the mcap50 and low-ADV arms.
- **Impact:** Returns are overstated, mostly in the conf era and in illiquid slices: about +2.3pp/trade for core A1/B conf, about +6pp for mcap50 A1 conf, and +13 to +17pp for the mcap50 ADV<5cr slices. The committed claim that the mcap50 leader cell earns +22.6/+19.6%/trade, and the ADV-bucket ranking, largely come from this effect. P2x/P50 and worst-cohort figures are also computed over the stretched windows.

### 10. [HIGH] src/agentic/sim_leader_portfolio_7x.py — lines 48-50, 65, 107 (also 159)  (3/3 upheld)
- **Lens:** 4 PORTFOLIO ARITHMETIC / 3 data contract
- **Claim:** Missing sessions inside a holding window turn into zero returns, so the whole price move across the gap is lost. The panel has no BE/BZ (trade-for-trade) rows before 2025, so this hits momentum leaders in the disc era. It inflates disc-era CAGR by about 3 to 4pp and makes the two eras unequal.
- **Impact:** Reported disc-era CAGRs are overstated by about 3.2pp (baseline) to 3.7pp (H, G+H); conf-era figures barely move. After correction, baseline full-period CAGR is 18.6%, about equal to the reported EW benchmark (18.2%), so the '+20.6% vs +18.2%' edge mostly disappears. The disc-vs-conf comparisons behind the both-era verdicts are biased toward disc. The docstring's 'frozen' treatment is meant for delisted names but is also applied to mid-life gaps. Fix: forward-fill closes within each name's listing life before computing returns, and backfill BE sessions for 2015-2024.

### 11. [HIGH] src/agentic/sim_leader_portfolio_7x.py — lines 61 (also 70-71 breadth, 159 benchmark)  (3/3 upheld)
- **Lens:** 1 LOOKAHEAD / 3 adjusted vs raw prices
- **Claim:** The core-band penny filter `close > 50` uses the back-adjusted close, which already reflects future splits and bonuses. Stocks that will split later fail the filter in their earlier history even though they traded above Rs50 at the time.
- **Impact:** The baseline is understated by about 3pp. H's registered disc-era edge over baseline shrinks from +9.9pp to +5.1pp, and its conf-era gap widens from -0.7 to -1.1pp. The G breadth gate and the EW benchmark carry the same future-split bias. No verdict flips in the arms tested, but every level and lever gap in the log moves by several pp.

### 12. [HIGH] src/agentic/test_hot_order_combo.py — lines 25-31, 35, 45, 49-65, 112-117  (3/3 upheld)
- **Lens:** LOOKAHEAD/PIT coverage + ERA SPLIT + PORTFOLIO (2016+ window)
- **Claim:** No PIT TTM revenue exists before late 2018, so `mat_order` is always False for 2016 through 2018. The disc-era C cell is really a 2019-2022 cell, and in the 2016+ Part B window the HARD arm sits in forced cash through the 2018 small-cap crash.
- **Impact:** Part A disc era: C can only draw rows from about Nov 2018 onward. The disc base is 20.12% over all years but 22.32% for 2018-11..2022, and hot&up is 26.13% overall but 28.41% for 2018-11+. That gives C a mechanical tailwind of about 11% on the 1.5x-base bar and about 9% on the 1.2x-control bar, biasing the disc verdict toward PASS. Control and base also absorb all 2016-18 material orders unflagged. Part B 2016+: the HARD arm is 0% invested until about end-2018, which dilutes CAGR and removes the 2018 crash from its drawdown. Reference B+H worst DD is -77.5% for 2016+ vs -61.8% for 2019+, so HARD's 2016+ maxDD looks much better purely from missing data. TIE equals AS-IS before 2019, so the 2016+ drawdown comparison is not a test of the rule.

### 13. [HIGH] src/agentic/test_hot_order_combo.py — lines 23-24, 32-35 (amount_cr from fetch_order_fulltext.py:76,91 via event_materiality_study.py:47-59)  (3/3 upheld)
- **Lens:** PARSING/REGEX
- **Claim:** `amount_cr` is the largest Rs/USD figure anywhere in the attachment, and it overrides a correct headline amount. It routinely picks company-revenue or group-size boilerplate and order-book totals, which then pass the >=15%-of-TTM materiality bar.
- **Impact:** Systematic false 'material order' flags, concentrated in large caps that issue press releases with revenue boilerplate. Siemens order releases come out at about 1.0x TTM, HCL about 1.3x, Bharat Forge about 3x, HGS about 1.2x. These pollute C and pull it toward the control, biasing Part A toward FAIL, and make the HARD/TIE arms buy the wrong names. The size of the contamination scales with how often boilerplate issuers file orders, and one issuer can add dozens of flagged stock-weeks.

### 14. [MEDIUM] src/agentic/anatomy_1p5x.py — lines 108-118  (3/3 upheld)
- **Lens:** UNITS / PLAIN LOGIC (fundamentals basis and quarter alignment)
- **Claim:** The pnl_quarterly key is (symbol, quarter_end, basis). The code dedups on (symbol, quarter_end) only, keeping whichever of consolidated or standalone was filed last, sometimes a minute apart. As a result eps_ttm, eps_yoy, sales_yoy, loss_to_profit and pe mix consolidated and standalone figures. shift(4) and rolling(4) also count rows rather than quarters, so missing quarters misalign 'year-ago'. EPS YoY is also computed across splits because EPS is as-reported.
- **Impact:** eps_yoy and sales_yoy (#7 and #8 permutation features, AUC drop 0.0051 each), pe, pe_ind and loss_to_profit are noisy, and many of their extreme values are basis or split artefacts rather than real surprises. The Part 1 lifts and model importance for these features are unreliable. The same dedup feeds rev_ttm_cr in event_materiality_study, so last_order_to_rev_90d is affected too.

### 15. [MEDIUM] src/agentic/anatomy_1p5x.py — lines 3, 69-71, 78-79, 83, 86  (3/3 upheld)
- **Lens:** LOOKAHEAD / SURVIVORSHIP (universe membership and log_mcap)
- **Claim:** The docstring says the universe is 'PIT market cap >= Rs50cr'. In fact 37.7% of sampled rows take mcap from the screener backcast or backfill: today's (or last-trade) share count times the adjusted close, which ignores dilution after t. Membership in the universe and the log_mcap feature are therefore not point-in-time for exactly the small caps near the 50cr threshold.
- **Impact:** Names that later raised equity (QIP, preferential or merger shares, often after a rally) are sized with future share counts. That can admit them to the ≥50cr universe early and shifts log_mcap, the #1 permutation feature (AUC drop 0.0328) and 9.2% of gain. The direction per name is unknown, but the model's top feature and the universe edge both rest on non-PIT data for most of the small-cap band, mainly in disc. 12.9% of Part 3 model top-10 picks are screener-sized.

### 16. [MEDIUM] src/agentic/build_mcap_pit.py — lines 56-73 (row-level `miss` fill from a symbol-level backfill); src/agentic/fetch_screener_mcap_backfill.py:35-37 (`todo = eq - have`)  (3/3 upheld)
- **Lens:** SURVIVORSHIP / UNIVERSE coverage
- **Claim:** The fallback cascade fills misses row by row, but screener_mcap_backfill was fetched only for symbols with no sized row at all. Symbols that have pnl rows but no screener_fundamentals row therefore stay NULL before their first filing, and NULL means excluded from the Rs 50cr universe. This removes about 316 large, live names from 2015 to mid-2018 and hides the first weeks of many new listings.
- **Impact:** About 23% of the true 2016-17 Rs 50cr universe is missing, and the missing names are established mid and large caps. Breadth, heat and median-return context (anatomy mkt_med_ret20 and mkt_ew_ret60, broad-industry heat) are computed on a truncated universe until mid-2018 and then jump, which is a structural break inside the disc era. IPO early windows are dropped for 22% of new listings.

### 17. [MEDIUM] src/agentic/event_materiality_study.py — lines 33 (CATS['order'] 'awarded'); used by fetch_order_fulltext.py:38-45  (3/3 upheld)
- **Lens:** PARSING/REGEX (category false positives)
- **Claim:** The bare `awarded` alternative in the 'order' category pulls non-orders into the crawl population: recognition awards, arbitration and tribunal awards, and credit ratings. Their largest in-text amount is then treated as the order value.
- **Impact:** Roughly 5% of the order population (~330 filings) is not an order. Many carry boilerplate or arbitration amounts that clear 15% of TTM. This adds more false material flags on top of the largest-amount issue.

### 18. [MEDIUM] src/agentic/event_materiality_study.py (CATS) — lines 33  (3/3 upheld)
- **Lens:** 6. PARSING / REGEX (category false negatives) + 5. era validity
- **Claim:** The 'order' regex misses common and NSE-standard phrasings, and recall differs by era: 'Awarding of order(s)/contract(s)' (NSE subject line), plural 'orders worth', 'Receipt of order', and 'bags/wins/receives ... order'.
- **Impact:** The 6,340-filing crawl and the event study's order cell are under-inclusive: estimated recall is ~68% in disc (2,279 found vs ~1,000 missed) and ~87% in conf. Downstream, the test_hot_order_combo 'control = hot & >200DMA & no order' cell contains real material orders, which biases the C-vs-control lift toward 1. The era gap in recall also changes the cell's composition between disc and conf.

### 19. [MEDIUM] src/agentic/fetch_order_fulltext.py — lines 52-53, 65-67, 76 (consumed at test_hot_order_combo.py:32)  (3/3 upheld)
- **Lens:** UNITS
- **Claim:** USD order amounts are converted at a hard-coded 83.0 INR/USD for every filing before Feb 2024, because macro_panel.usdinr has no earlier data. The fallback is silent.
- **Impact:** USD orders in the disc era are inflated by about +24% (2016, ~67), +28% (2017, ~65), +19-22% (2018-19), +12% (2020-21) and +5% (2022). Orders worth about 12-13% of revenue then cross the 15% bar. Because the bias depends on era, it tilts disc-era C toward exporters and USD-contract names, and it breaks the like-for-like era comparison.

### 20. [MEDIUM] src/agentic/fetch_order_fulltext.py; src/agentic/ocr_order_filings.py; src/agentic/event_materiality_study.py — lines fetch 53, 65, 67, 76; ocr 57, 62, 70; event_materiality_study 110-113  (3/3 upheld)
- **Lens:** 3. UNITS / data contract
- **Claim:** macro_panel.usdinr is non-null only from 2024-02-19. Every USD amount from 2016 to Feb 2024 is therefore converted at a hard-coded 83.0, not 'that day's USDINR' as the manifest states. That fabricates an FX rate, contrary to AGENTS.md.
- **Impact:** USD-denominated amounts are overstated by about 8-28% in the disc era (2016 ~67, 2019 ~70, 2021 ~74) and by ~0-1% in 2023. The effect is era-asymmetric. In event_rows_20260924 it touches only 16 order events with a ratio (disc mean inflation 1.08x, max 1.18x), and 1 of them crosses the 15% materiality threshold. In the full-text crawl it is larger, because press releases quote USD figures more often, and with the largest-amount rule the inflated USD translation wins over the INR figure given alongside it.

### 21. [MEDIUM] src/agentic/model_bakeoff_1p5x.py — lines 82-83, 86-87  (3/3 upheld)
- **Lens:** STATISTICAL VALIDITY
- **Claim:** The per-era AUC and top-1% precision/lift pool four or four-plus test years with base rates from 9% to 38%. They therefore measure whether each model's score LEVEL follows the year's base rate, not ranking skill. For lgbm_rank the pooled numbers are arbitrary, because LambdaRank scores are only defined up to a shift per query (week).
- **Impact:** The AUC and top-1% columns in bakeoff.json cannot rank models. Examples are lgbm_rank's disc AUC of 0.687 as 'best', logit's disc top-1% lift of 0.69x, and lgbm's pooled disc AUC understating its within-period AUC by about 0.08. Only the within-week wk10 metric is valid for comparison. The recorded verdict relies on wk10, but the table as published is misleading.

### 22. [MEDIUM] src/agentic/model_bakeoff_1p5x.py — lines 61-64  (3/3 upheld)
- **Lens:** STATISTICAL VALIDITY (model comparison)
- **Claim:** The MLP's early stopping cannot catch overfitting. With `early_stopping=True`, sklearn holds out a RANDOM 10% of `sub` and monitors validation ACCURACY. Those rows are stock-weeks whose neighbours, with 94 of 95 label sessions shared, sit in the fit set. Validation accuracy keeps rising while out-of-sample AUC falls, so 'MLP worst' reflects the configuration, not the model family.
- **Impact:** The MLP rows (AUC 0.537/0.549; wk10 19.9%/22.7%) understate what a properly early-stopped network does. A 1-2 epoch MLP is roughly at logit/lgbm level. 'MLP worst' in the experiments.jsonl verdict is not supported, and the ensemble (line 75) is dragged down by including this overfit MLP.

### 23. [MEDIUM] src/agentic/model_bakeoff_1p5x.py — lines 31-32 (FEATS inherit log_px, pe, pe_ind from src/agentic/anatomy_1p5x.py:86, 123-124)  (3/3 upheld)
- **Lens:** LOOKAHEAD / UNITS (adjusted vs raw prices)
- **Claim:** FEATS includes every non-META column, including `log_px`, `pe` and `pe_ind`. These are computed from the back-adjusted `close` (the panel carries `price_adjustment_factor_to_present`). `pe` divides that adjusted price by as-reported `eps_basic`. As a result, any row before a later split or bonus has a deflated price and PE, and the model learns the future corporate action, which correlates with the rally.
- **Impact:** This inflates all models' out-of-sample skill, most for flexible learners that can use price level non-linearly. The conf-era lgbm picks over-select future-split names by about 1.9x. The absolute wk10 levels and any claim that 'features have hit the ceiling' include this leak.

### 24. [MEDIUM] src/agentic/model_bakeoff_1p5x.py — lines 84-86 (core defined in src/agentic/anatomy_1p5x.py:68)  (3/3 upheld)
- **Lens:** SURVIVORSHIP / universe (non-PIT tradable filter)
- **Claim:** The 'TRADABLE' universe for the headline wk10 metric is `E['core']`, and its `close > 50` test uses the back-adjusted close. Membership therefore depends on future splits and bonuses, and pre-split winners trading well above Rs 50 at the time are excluded.
- **Impact:** The wk10 numbers are not point-in-time. With the correct filter lgbm's hit rate is about 1.5pp (disc) and 2.8pp (conf) higher, but part of that gain comes from the log_px/pe leak above, so the true PIT figure cannot be read off either version. Models that lean on low price are affected differently, so the between-model wk10 ordering is also contaminated.

### 25. [MEDIUM] src/agentic/sim_allin_matrix.py — lines 28-30, 69-71  (3/3 upheld)
- **Lens:** 2. SURVIVORSHIP / 4. PORTFOLIO ARITHMETIC
- **Claim:** Symbol renames and merges are not stitched. When a held name is renamed mid-hold, its column goes NaN and fillna(0) freezes its value for the rest of the hold, although security_master.renamed_to maps the old symbol to the new one. Genuinely suspended or delisted names are also frozen at their last close rather than marked down, and the docstring does not disclose this.
- **Impact:** Errors run in both directions, from -1.6pp to +2.8pp median CAGR. The model arms are understated by up to ~2.8pp at long holds. The effect grows with hold length, because a rename is more likely to fall inside a longer hold.

### 26. [MEDIUM] src/agentic/sim_allin_matrix.py — lines 36, 42, 48, 54 (core from anatomy_1p5x.py L68)  (3/3 upheld)
- **Lens:** 1. LOOKAHEAD / 2. SURVIVORSHIP (universe)
- **Claim:** The tradable band `core` (ADV>=5cr & close>50) is tested on the back-adjusted close, which already divides out splits and bonuses that happen after the entry date. Names whose real traded price was above Rs 50 but whose later split pushes the adjusted price to Rs 50 or below are excluded from BASELINE heat, all rule picks, MODEL candidates and the EW benchmark, using future corporate-action information.
- **Impact:** The universe is chosen with hindsight and removes future winners, so EW and most rule cells are understated by roughly 0.4-2pp, with mixed sign at some holds. Fixing this alone blows up the MODEL arms (see the next finding), so the two fixes have to go in together.

### 27. [MEDIUM] src/agentic/sim_allin_matrix.py — lines 51-53 (pred from anatomy_1p5x.py L86, L123, L263-266)  (3/3 upheld)
- **Lens:** 1. LOOKAHEAD (model features)
- **Claim:** The MODEL arms use `pred`, a walk-forward LightGBM score that is not truly out of sample. It is trained on `log_px = log10(adjusted close)` and `pe = adjusted close / as-reported EPS`, and both encode future splits and bonuses: a low adjusted price marks a stock that will later split, which usually follows a rally. log_px is the model's 4th most important feature.
- **Impact:** In the current output the upward bias on MODEL is probably small, because the adjusted-close core filter happens to hide the worst-leaking rows. The model is still not point-in-time, and any correction to the universe produces spurious +5 to +11pp MODEL CAGRs. The fix is to use a raw-basis price for log_px and pe.

### 28. [MEDIUM] src/agentic/sim_leader_cell_v2.py — lines 151-154, 178, 182 (maxdd_proxy on cohort-mean endpoint returns)  (3/3 upheld)
- **Lens:** PORTFOLIO ARITHMETIC
- **Claim:** The '~maxDD' column spreads each cohort's 126-day endpoint return evenly over 26 steps and averages 26 cohorts. It never marks positions to market during the hold, and it skips weeks with no picks instead of counting them as cash. The drawdown it reports is about a third of the real NAV drawdown for the same cell.
- **Impact:** The drawdown and risk reported for every arm is understated by roughly 25pp (about 3x) for the production cell. Any arm ranking or tail gate built on '~maxDD' is not informative.

### 29. [MEDIUM] src/agentic/sim_leader_cell_v2.py — lines 96 (`in_uni = (px["adv"] >= 5) & (px["close"] > 50)`)  (3/3 upheld)
- **Lens:** LOOKAHEAD / SURVIVORSHIP (adjusted price in a universe filter)
- **Claim:** The core-band price floor is applied to the back-adjusted close. The adjustment depends on splits and bonuses that happen after t, so a name trading above Rs 50 at t is dropped if it later split or bonused. Membership therefore depends on the future, and the dropped names tend to be later winners, since splits and bonuses usually follow rallies.
- **Impact:** Core-universe results are understated by about 0.6-2.8pp/trade (larger in conf). Because it is lookahead, the bias depends on future corporate actions and is not a neutral filter. The mcap50 universe has no price floor and is not affected.

### 30. [MEDIUM] src/agentic/sim_leader_cell_v2.py — lines 92-94 (mcap50 universe from mcap_pit); producer build_mcap_pit.py lines 57-72  (3/3 upheld)
- **Lens:** LOOKAHEAD / UNITS (share count not point-in-time)
- **Claim:** The '--universe mcap50' filter treats mcap_pit as point-in-time. For about a third of universe rows, and more among picks, mcap_cr is present-day (or last-trade) share count x adjusted close. It ignores dilution that happened after t (QIP, preferential issues, mergers), so whether a name clears Rs 50cr at t uses the future share count.
- **Impact:** Universe membership near the 50cr threshold, and therefore industry heat and rank composition, is not point-in-time for most of the disc era. Direction is mixed but not small: sources differ by 10-16pp/trade. The mcap50 claims should be re-cut on pnl_implied rows only.

### 31. [MEDIUM] src/agentic/sim_leader_portfolio_7x.py — lines 35 (with 98-112)  (3/3 upheld)
- **Lens:** 4 PORTFOLIO ARITHMETIC / 7 LOGIC
- **Claim:** The --hold refactor in commit a8ad8d9 changed the default slot count from 26 to round(126/5)=25. The default run no longer reproduces the registered 126-session result. With 25 slots, each slot's next cohort enters one session before the previous cohort exits, so that day's return is compounded twice.
- **Impact:** Rerunning the script with defaults would overwrite portfolio_7x_nav.parquet with numbers that differ from the committed log by 1 to 2pp CAGR, which breaks the rule to preserve behavior when refactoring. The one-day double count adds about +0.3 to 0.4pp CAGR. Idle days per rollover also differ by hold: 4 of 130 at 126 (old code), 2 of 65 at 63, 1 of 85 at 84. That gives the 84-session variant roughly 2% more time invested in the hold-length comparison.

### 32. [MEDIUM] src/agentic/sim_leader_portfolio_7x.py — lines 159-160  (3/3 upheld)
- **Lens:** 1 LOOKAHEAD (benchmark)
- **Claim:** The EW benchmark counts a stock's day-t return only if the stock is in the core band on both t-1 and t. Stocks that leave the band on day t, mainly by closing through Rs50, are dropped on the day they fall. This inflates the benchmark.
- **Impact:** The benchmark is overstated by about 2.1pp CAGR (2.5pp in disc), so every 'vs EW market' comparison, including the commit's '+20.6% vs +18.2%', is off by that much. This works in the opposite direction to the gap finding above, so the two need fixing together before any claim against the market is made.

### 33. [MEDIUM] src/agentic/sim_leader_portfolio_7x.py — lines 94-113  (3/3 upheld)
- **Lens:** 5 STATISTICAL VALIDITY / 4 equal-weight drift
- **Claim:** Slots are never rebalanced back to 1/26, so the NAV ends up dominated by a few compounding chains. The reported CAGRs are one path-dependent draw, and varying the arbitrary slot count moves them by about 3pp.
- **Impact:** Lever-vs-baseline margins below about 3pp (B disc -0.8/conf +1.9, H conf -0.7, G+B conf -2.9) sit inside the phase noise and should not be read as effects. The 'no' verdicts survived every phase tested, but magnitude claims are not robust. Reporting the mean and spread over slot phases, or rebalancing slots, would fix this.

### 34. [MEDIUM] src/agentic/test_hot_order_combo.py — lines 30-35, 46, 52-53 (control definition)  (3/3 upheld)
- **Lens:** SURVIVORSHIP/UNIVERSE
- **Claim:** Only symbols covered by pnl_quarterly can ever be flagged, and that coverage is almost entirely survivors. C, HARD and TIE are therefore survivor-only populations, while the control and base keep non-survivors plus every unmeasurable order, including unparsed material ones.
- **Impact:** C and control are different populations. Uncovered names lift control P(+50%) by about 6% (≈0.793x24.1+0.207x30.9=25.5 vs 24.1), biasing the 1.2x-control bar toward FAIL. The survivor-only C biases s95 and mean-95 returns upward. 'No such order' in the control actually means 'no measurable material order'.

### 35. [MEDIUM] src/agentic/test_hot_order_combo.py — lines 41-46, 57-65  (2/3 upheld)
- **Lens:** STATISTICAL VALIDITY
- **Claim:** The registered n >= 50 per era counts overlapping stock-weeks, not independent orders. Each order flags about 8-9 consecutive weekly rows (60-day window), and their 95-session outcome windows overlap by more than 80%. No clustering by symbol or order episode is done.
- **Impact:** n=50 can be 6-15 independent episodes, dominated by a few names (for example Siemens-type boilerplate issuers). The PASS/FAIL gate and the 1.5x and 1.2x lifts have much wider uncertainty than the n implies, and one or two names can decide the verdict.

### 36. [MEDIUM] src/agentic/test_hot_order_combo.py — lines 79, 84-89  (3/3 upheld)
- **Lens:** PLAIN LOGIC
- **Claim:** Part B's '(1) as is' is not the B+H rule it claims to be. The reference rule (sim_allin_matrix.py:41-42) ranks top-10 by ret60 within the whole hot industry and then applies the core/ret252>0.5 filters. This script filters first and then takes head(10), so it admits names ranked as low as 65th. The HARD arm drops the top-10 cap entirely.
- **Impact:** The drawdown question is anchored on B+H, but the baseline arm differs from B+H by 2-5 pp in worst-phase maxDD and about 1 pp in CAGR. HARD and TIE are also built on the wider pool, so their deltas are not deltas versus the registered B+H rule.

### 37. [LOW] src/agentic/anatomy_1p5x.py — lines 117-120, 139, 144, 154-157, 162, 176 (and delivery_pct features on line 42)  (3/3 upheld)
- **Lens:** LOOKAHEAD (same-day after-close information)
- **Claim:** Several event dates are normalised to the calendar day and included at the entry close of that same day, although most are published after 15:30: results filing_dt, announcement sort_date and order-event d. PIT uses intimDt (the insider's notice to the company) rather than the exchange broadcast `date` column. NSE delivery data for day t is published after the close. Shareholding uses qe+21 rather than the available `submission` date.
- **Impact:** Small. It affects about 1% of rows for fundamentals, and window counts are diluted over 30-90 days. For those rows, though, the feature reflects news whose price reaction on t+1 falls inside the label window, which is a direct leak.

### 38. [LOW] src/agentic/anatomy_1p5x.py — lines 297, 335  (3/3 upheld)
- **Lens:** PORTFOLIO ARITHMETIC
- **Claim:** The Part 3 returns come from a wide pivot with pct_change(fill_method=None) followed by fillna(0). When a symbol misses sessions, both the return into the gap and the first return after it become NaN and then 0, so the whole price move across the gap disappears from the NAV.
- **Impact:** Model arms are overstated by about 0.2pp per 95-session period, well under 1pp CAGR. Rule arms and the EW benchmark carry the same bias, so the arm comparison is barely changed.

### 39. [LOW] src/agentic/anatomy_1p5x.py — lines 79  (3/3 upheld)
- **Lens:** PLAIN LOGIC / DATA ARTEFACTS
- **Claim:** mkt_ew_ret60 is a plain cross-sectional mean of ret60, so a handful of multi-thousand-percent rows dominate it. Some of those rows are relistings or gap artefacts.
- **Impact:** This is the #3 permutation feature (AUC drop 0.0153, 11.2% of gain), and on those dates it is partly measuring outliers rather than market regime. A median or trimmed mean would be robust.

### 40. [LOW] src/agentic/anatomy_1p5x.py — lines 164-166  (3/3 upheld)
- **Lens:** PLAIN LOGIC
- **Claim:** prom_sells90 is identically zero for the same reason as the known prom_buys90 bug: sellValue is the string '0' in almost every row. Any fix via acqMode/tdpTransactionType has to cover the sell side as well.
- **Impact:** The feature is dead in Part 1 and in the model. No bias, but the promoter-selling signal is absent from the results.

### 41. [LOW] src/agentic/anatomy_1p5x.py — lines 225  (3/3 upheld)
- **Lens:** PLAIN LOGIC
- **Claim:** prom_delta is a continuous signed change in promoter holding (percentage points). Because of the `prom_` prefix it falls into the binary rule, so every decrease is lumped with zero.
- **Impact:** Part 1 cannot show whether large promoter decreases or increases matter, even though prom_delta is the #11 permutation feature. This does not change the model.

### 42. [LOW] src/agentic/anatomy_1p5x.py — lines 64  (2/3 upheld)
- **Lens:** PLAIN LOGIC / STATISTICAL VALIDITY
- **Claim:** age_yrs is documented as 'years listed', but for any name listed before the 2015 panel floor it equals calendar time since 2015-01-01. For most rows it is therefore a time index that is always outside the training range in the walk-forward.
- **Impact:** The gain-share importance table is misleading: the top 'driver' is the model fitting year-level base rates in-sample. In walk-forward prediction it puts every test row into the latest-period leaf.

### 43. [LOW] src/agentic/anatomy_1p5x.py — lines 304-315  (3/3 upheld)
- **Lens:** PORTFOLIO ARITHMETIC / LOOKAHEAD
- **Claim:** The 'BASELINE rule' and 'G+H rule' arms are not the registered arms from sim_leader_portfolio_7x. The gate uses same-day close breadth instead of prior-session breadth, and BASELINE heat and rank are computed on S (mcap ≥ 50cr only) instead of the full core band.
- **Impact:** Small. The rule arms in Part 3 cannot be compared one-for-one with the earlier 7x results (e.g. G+H at a 63-session hold), and the gate uses the same close it trades at.

### 44. [LOW] src/agentic/anatomy_1p5x.py — lines 156-157, 201  (3/3 upheld)
- **Lens:** PLAIN LOGIC
- **Claim:** last_order_to_rev_90d is described as the 'largest recent order / TTM revenue (90d)', but merge_asof backward returns the most recent order within 90 days, not the largest.
- **Impact:** The feature and the manifest description disagree. Coverage is about 1% of rows, so Part 1 skips it (<5% non-null) and it has negligible model weight.

### 45. [LOW] src/agentic/anatomy_1p5x.py — lines 89-94  (3/3 upheld)
- **Lens:** LOOKAHEAD (industry label)
- **Claim:** Industry comes from a single 2026 screener.in classification (plus hand-labelled analog peers written 2026-09-23) and is applied back to 2016. Names that later pivoted into a hot industry are placed there retroactively. The industry_analyst_labels README requires backtests to report results with and without analyst_inference labels, and this script does not.
- **Impact:** Unquantified. It touches ind_heat, ind_heat_pct, rank_in_ind_pct and the rule arms (whose heat is industry-based). The industry features have a small OOS weight (ind_heat_pct AUC drop 0.0019).

### 46. [LOW] src/agentic/anatomy_1p5x.py — lines 89-94 (feeds `ind`, `ind_heat_pct` used at test_hot_order_combo.py:51, 74-77)  (2/3 upheld)
- **Lens:** LOOKAHEAD (industry label)
- **Claim:** Industry membership is one current (Sep-2026) screener label per symbol, applied to every historical week. It is not point-in-time, and labels are missing more often for names that no longer exist.
- **Impact:** Heat groups can include companies later re-labelled into a now-hot industry, which makes hot-industry membership and 'hot' cells slightly look-ahead. The size is not quantified here, and the missing-label share is small.

### 47. [LOW] src/agentic/build_mcap_pit.py — lines 49-54 (used via sim_leader_portfolio_7x.py 58-62, H lever)  (3/3 upheld)
- **Lens:** 3 UNITS (share basis) / 1 PIT
- **Claim:** Market cap for the H universe multiplies a raw-basis price by a rolling 4-quarter median of PAT/EPS share counts. After a split or bonus, the median stays on the pre-split count for several quarters while the price is already post-split, so mcap is understated by the split ratio. Separately, 46% of disc-era broad rows use the screener backcast (present-day shares), which ignores later dilution, yet the docstring calls the universe 'PIT'.
- **Impact:** This only affects H/B+H/G+H heat-universe membership near the Rs50cr floor: sub-Rs500cr names that split can drop out for months, and later-diluted names can enter early. Likely small; not quantified.

### 48. [LOW] src/agentic/build_mcap_pit.py — lines 42-43  (3/3 upheld)
- **Lens:** PLAIN LOGIC (dedupe ignores the `basis` key of the data contract)
- **Claim:** pnl_quarterly is keyed on (symbol, quarter_end, basis), but line 43 dedupes on (symbol, quarter_end) keeping the latest filing_dt. Which basis survives (standalone or consolidated) is decided by which was timestamped later, often minutes apart, and 2,613 exact ties are resolved arbitrarily by the non-stable sort. Consolidated PAT/EPS shares differ from standalone in a material share of quarters.
- **Impact:** Adds noise to pnl_implied mcap and can make split lag last longer (as with NYKAA). Of the 133 validation misses off by >20% (13.2%), 104 are not explained by recent splits, and basis switching plus EPS rounding are the main candidates. Prefer one basis per symbol (standalone for share count), or take the median across both bases.

### 49. [LOW] src/agentic/build_mcap_pit.py — lines 57-62  (3/3 upheld)
- **Lens:** UNITS (share basis vs adjustment date)
- **Claim:** Backcast shares come from screener at fetch_date (mostly 2026-08-30) but are multiplied by close adjusted to panel end (2026-09-23). A split or bonus between those dates leaves shares on the old basis against a new-basis adjusted price.
- **Impact:** Small: 924 rows, mostly PGIL's 2015-2018 history halved. Fix: divide sh_now by the adjustment factor at fetch_date.

### 50. [LOW] src/agentic/build_mcap_pit.py — lines 43, 48-52  (3/3 upheld)
- **Lens:** LOOKAHEAD (technical PIT violation)
- **Claim:** The rolling median runs in quarter_end order while each row is dated by its own filing_dt, and keep='last' can keep a later re-filing of an earlier quarter. The median known at quarter Q's date can therefore include a value first published after that date. Filing timestamps, many after the close, are also normalized to the date and used for that day's close.
- **Impact:** Negligible economically, because a share count carries almost no return information. Listed only because the manifest claims 'shares known at t'.

### 51. [LOW] src/agentic/build_security_master.py — lines 12-13, 68, 153 (manifest sources.isin)  (3/3 upheld)
- **Lens:** UNITS / data contract
- **Claim:** The docstring and manifest say ISINs come from 'raw cm*bhav.csv.zip 2016-2024'. The actual coverage is 2015-01-01..2019-12-31, so every symbol that lived only between 2020 and 2026-09 has no ISIN. Nothing discloses this.
- **Impact:** This is the root cause of finding 1: dead post-2019 ETFs rely on the name or symbol regex alone. Company names in the same 2020-2026 cohort lack the ISIN protection that live and pre-2020 names get, so a GOLD/NIFTY/ETF-looking symbol there would be silently dropped. None is today, having checked all 66. Consumers reading the manifest overstate ISIN coverage.

### 52. [LOW] src/agentic/build_security_master.py (+ src/agentic/generate_hybrid_basket.py non_equity) — lines build_security_master.py 106, 115-116, 126-128; generate_hybrid_basket.py 105, 140  (3/3 upheld)
- **Lens:** PARSING / REGEX + SURVIVORSHIP / universe
- **Claim:** Two dead ETFs and five rights-entitlement instruments pass both is_fund_unit and non_equity(), so they are treated as operating companies in every universe built from the panel. The ETFs are NETFIT (Nippon ETF Nifty IT, the predecessor of ITBEES) and NAVINIFTY. The rights entitlements are DUCON-RE1, JAYKAY-RE1, KSHITIJ-RE, RATNA-RE and VHLTD-RE1.
- **Impact:** The effect on reported core-band results is negligible. NETFIT closed at or below 31.6 on all 484 sessions, below the close>50 floor. NAVINIFTY's median turnover is 1.6 lakh. The rights entitlements have ADV under Rs 1 cr and lack the 252-day history the z-bands need. They are still counted in the manifest's 3,307 'equities'. They also enter all-panel computations that skip the core filter (e.g. the fetch_screener_mcap_backfill list, and non-core rows and cross-sections in anatomy/breadth). The rights-entitlement gap will recur with every rights issue now that BE is in the panel.

### 53. [LOW] src/agentic/event_materiality_study.py (AMT) — lines 44  (3/3 upheld)
- **Lens:** 6. PARSING / REGEX (amount false negatives / wrong currency)
- **Claim:** AMT requires a unit word and misses 'Crs', unit-less full-rupee Indian grouping, the HTML entity &#8377;, 'Rupees N crore', 'Rs.' followed by 2+ spaces, and figures written as 'Rs. 15.56 (Fifteen Crores...)'. It also has no word boundary before 'Rs' and treats any '$' (S$, SG$, A$) as USD.
- **Impact:** About 225 order filings (~3.5%) that state an amount get none, or a smaller or wrong one. Direction is mostly toward missing materiality, which is smaller than the largest-amount bias. The &#8377; misses sit almost entirely in conf.

### 54. [LOW] src/agentic/event_materiality_study.py (CATS) — lines 32-33  (3/3 upheld)
- **Lens:** 6. PARSING / REGEX (category false positives)
- **Claim:** 'awarded', `\bLoA\b` and `\bL-?1\b` pull in non-order filings: honours and trophies, credit ratings, NCD Letters of Allotment, and plant addresses.
- **Impact:** About 3% of 'order' events are not orders, which dilutes event-type cell A slightly. Combined with the largest-amount finding, an award press release carrying a boilerplate revenue figure becomes a 'material order'. The NCD rows get 0.1 cr from 'FVRS10LAC' (no word boundary before Rs).

### 55. [LOW] src/agentic/event_materiality_study.py (and replicated in test_hot_order_combo.py) — lines event_materiality_study 127  (3/3 upheld)
- **Lens:** 3. UNITS / 7. LOGIC (materiality denominator; adjacent to scope)
- **Claim:** rev_ttm_cr = rolling(4) over whatever quarter rows exist, with no check that they are 4 consecutive quarters. PIT revenue also starts only with 2018-04 filings, so order_to_rev is NaN for all 2016-2017 orders.
- **Impact:** The denominator is stale or mixed for ~9% of ratios, mostly understating revenue for growing firms and so overstating materiality. The disc era for any materiality test is effectively 2018/19-2022, so the full-text crawl of 2016-17 filings adds nothing to ratio-based tests.

### 56. [LOW] src/agentic/fetch_order_fulltext.py -> src/agentic/test_hot_order_combo.py — lines fetch 64-66; test_hot_order_combo ~39-43  (3/3 upheld)
- **Lens:** 1. LOOKAHEAD
- **Claim:** The producer stores only the filing DATE (`d=str(d.date())`) and drops sort_date's time. The consumer then counts orders with d <= trade_date (`searchsorted(..., side="right")`) and measures outcomes from the trade_date CLOSE, so orders filed after 15:30 on trade_date are used at that day's close.
- **Impact:** Small: only the weekly row whose trade_date equals the filing date is affected, roughly 1% of mat_order-flagged rows. For those rows the next-day reaction to a material order leaks into y95/fc95, which biases the material-order cell upward. The fix is to keep the timestamp and shift post-close filings to the next session.

### 57. [LOW] src/agentic/model_bakeoff_1p5x.py — lines 34, 49, 84-85  (3/3 upheld)
- **Lens:** SURVIVORSHIP
- **Claim:** Test and selection sets are restricted to rows with a y95 label, and a row gets one only if the symbol has at least 76 more panel rows. Rows for names that were renamed, merged or suspended soon after entry are silently dropped from the weekly top-10 candidates. The industry label, which comes from the current screener map, is also missing only for dead symbols.
- **Impact:** The wk10 bias is at most about ±1.4pp and its direction is mixed: renames such as ADANIGAS were large winners, JETAIRWAYS a failure. The effect is small but not point-in-time.

### 58. [LOW] src/agentic/model_bakeoff_1p5x.py — lines 75  (3/3 upheld)
- **Lens:** LOOKAHEAD (score transform)
- **Claim:** The ensemble averages `rank(pct=True)` computed over ALL test rows from 2019 to 2026 pooled. A row's ensemble score therefore depends on the future distribution of each model's scores, and the within-week ordering used by wk10 is not implementable in real time. The PIT version would rank within each trade_date.
- **Impact:** Likely small. It changes how the three models are weighted within a week according to their cross-year score levels, which vary a lot (e.g. lgbm mean pred ranges 0.06-0.32 by year), so the ensemble wk10 (25.0/27.5%) is not a clean PIT number. Not quantified: the per-model predictions are not saved.

### 59. [LOW] src/agentic/ocr_order_filings.py — lines 52-55, 75-78  (3/3 upheld)
- **Lens:** 7. PLAIN LOGIC BUG
- **Claim:** A transient download error in the OCR pass is permanent. The appended HTTP_/ERR_ record becomes the latest one, `todo` selects only latest status == 'NEEDS_OCR', and fetch_order_fulltext's `done` set still contains the key through the older NEEDS_OCR record, so neither script ever retries it. OCR_* statuses are also missing from the manifest's status list.
- **Impact:** Coverage loss only: amount_cr falls back to the headline amount. No bias beyond fewer full-text amounts.

### 60. [LOW] src/agentic/sim_allin_matrix.py — lines 43-44  (3/3 upheld)
- **Lens:** 1. LOOKAHEAD / 7. LOGIC (lever definition)
- **Claim:** The G gate reads same-day-close breadth (`mkt_breadth` at d, computed from the same closes the trade executes at). The registered G lever in sim_leader_portfolio_7x uses prior-session breadth (`breadth.shift(1)`), so the G, G+B, G+H and G+B+H rows here are not the same experiment and carry a same-close signal.
- **Impact:** Small, since about 4% of cohorts flip in both directions. Cross-experiment comparisons of G against the 7x factorial are nonetheless not like-for-like.

### 61. [LOW] src/agentic/sim_allin_matrix.py — lines 51-53 (pred features from anatomy_1p5x.py L117-120, L129-139)  (3/3 upheld)
- **Lens:** 1. LOOKAHEAD (same-day after-close filings)
- **Claim:** Results and announcement features that feed `pred` are keyed on the filing date normalised to midnight and joined backward inclusive of the entry date. Filings made after the 15:30 close on day d therefore count as known at the close of d, the entry close.
- **Impact:** Negligible for the matrix, below 0.1pp CAGR. It is still a point-in-time violation in the model features.

### 62. [LOW] src/agentic/sim_allin_matrix.py — lines 58-61, 82, 88-89  (2/3 upheld)
- **Lens:** 5. STATISTICAL VALIDITY
- **Claim:** 'Worst-phase' CAGR and drawdown are the minimum over h//5 phases, which ranges from 6 phases at h=30 to 50 at h=250. Longer holds therefore look worse from the extra draws alone, and each phase also covers a different window (phase starts up to 49 weeks apart). The two windows, 2016+ and 2019+, are nested and both include the conf era, so the output cannot show whether a result holds in both disc (2016-2022) and conf (2023+), which is the repo's trust criterion.
- **Impact:** The worst-phase columns (e.g. G+H h250 at -9.8%) cannot be compared across holds. No cell's result can be verified in both eras.

### 63. [LOW] src/agentic/sim_allin_matrix.py — lines 71  (2/3 upheld)
- **Lens:** 4. PORTFOLIO ARITHMETIC (costs)
- **Claim:** A 0.5% round trip is charged on 100% of capital every rotation, including for the EW-market benchmark and for names that stay in the basket. An equal-weight market portfolio has little turnover, so the benchmark's apparent improvement with longer holds is mostly cost drag.
- **Impact:** At short holds the EW benchmark is understated by about 3-4pp a year, which inflates every approach's excess over the market at h=30/60. This follows the 'full rotation' spec, but the benchmark comparison is biased.

### 64. [LOW] src/agentic/sim_allin_matrix.py — lines 25-26, 48, 54 (S universe from anatomy_1p5x.py L83)  (3/3 upheld)
- **Lens:** 2. SURVIVORSHIP / universe
- **Claim:** The universe only contains symbols with a PIT mcap row (anatomy L83 `mcap_cr >= 50`). Securities with no mcap_pit at all are overwhelmingly delisted or renamed, so they drop out of the EW benchmark and the rule heat, and cannot be picked.
- **Impact:** Small: about 0.33% of core rows. It mildly flatters EW and the rules by dropping collapses such as DHFL, and removes pre-rename history for renamed large caps.

### 65. [LOW] src/agentic/sim_allin_matrix.py — lines 36-42 (ind from anatomy_1p5x.py L89-94)  (3/3 upheld)
- **Lens:** 1. LOOKAHEAD (industry label)
- **Claim:** Industry labels come from one static 2026 screener crawl plus analyst analogs, applied to every date since 2016. Heat, hot-industry and within-industry rank therefore group stocks by their present-day classification, not the one that applied at the entry date.
- **Impact:** Direction and size not quantified; no historical labels are available to test against. Companies that later moved into a hot sector are grouped with it in hindsight, a mild upward bias for the heat-driven rule arms.

### 66. [LOW] src/agentic/sim_leader_cell_v2.py — lines 72-73, 78-84 (`delisted = last_dt < panel_end - 10d`; per-symbol pct_change)  (3/3 upheld)
- **Lens:** SURVIVORSHIP / PLAIN LOGIC (symbol renames treated as delistings)
- **Claim:** A symbol rename looks like a delisting: the old symbol exits at its last close with a truncated hold. The new symbol starts with no history, so for its first 60 rows it is outside heat and all arms, and for its first 252 rows it can never be EXT.
- **Impact:** Small on the measured exits (at most 0.4pp). There is an unmeasured selection loss: renamed names, often re-branded winners such as ZOMATO->ETERNAL and GET&D->GVT&D, are excluded from EXT arms for about a year after the rename.

### 67. [LOW] src/agentic/sim_leader_cell_v2.py — lines 47-59, 88, 98 (industry map; `px["ind"].notna()` filter)  (2/3 upheld)
- **Lens:** SURVIVORSHIP / LOOKAHEAD (industry labels)
- **Claim:** Rows without an industry label are dropped before heat and rank, and delisted names are far more likely to be unlabeled. This partly undoes the v2 survivorship fix under '--map nse' and, to a lesser degree, the production '--map nse4'. All maps also apply today's (2026) taxonomy to 2016 dates.
- **Impact:** For nse4 the measured effect is about +0.2pp. For the nse map it is not measurable but is a structural survivor tilt, so nse-map results are not comparable to nse4. The current-taxonomy lookahead is not quantified.

### 68. [LOW] src/agentic/sim_leader_cell_v2.py — lines 32, 101-109 (OO_RE regex; merge_asof backward with exact match on date-normalized sort_date)  (3/3 upheld)
- **Lens:** PARSING / LOOKAHEAD (takeover flag)
- **Claim:** The regex matches acquirer-side and non-offer filings, not only filings by takeover targets. Filings timestamped after the 15:30 close are also treated as known at that day's close.
- **Impact:** Negligible for the reported numbers: the no-takeover filter changes core A1 by only 1 row (disc) and 34 rows (conf), and means by at most 0.25pp. It matters only if the filter is later made stricter.

### 69. [LOW] src/agentic/sim_leader_portfolio_7x.py — lines 130 (also 125-128)  (3/3 upheld)
- **Lens:** 4 PORTFOLIO ARITHMETIC (metrics)
- **Claim:** Calendar-year returns are computed within each year's own rows, so the first session of every year (prior year-end close to first close) is left out. The era CAGRs likewise leave out the first session of 2023.
- **Impact:** The printed CALENDAR-YEAR table is off by up to 3.6pp per year for every arm and for the benchmark. The era-CAGR error is one session and negligible. Verdicts are unaffected.

### 70. [LOW] src/agentic/sim_leader_portfolio_7x.py — lines 42-46, 57  (3/3 upheld)
- **Lens:** 1 LOOKAHEAD (industry label) / data contract
- **Claim:** Industry labels come from a 2026-09 screener crawl plus analyst analog labels written 2026-09-23, and are applied to every date from 2016. They are not point-in-time. industry_analyst_labels.README.md says backtests must report results with and without the analyst labels, and this script reports only 'with'.
- **Impact:** Small: under 1% of picks rely on analyst analogs. Label drift for companies that changed business is not quantified. The with/without report required by the contract is missing.

### 71. [LOW] src/agentic/test_hot_order_combo.py — lines 25-29  (3/3 upheld)
- **Lens:** UNITS/PIT (TTM denominator)
- **Claim:** The TTM revenue denominator mixes standalone and consolidated bases and sums non-consecutive quarters. The dedup drops the `basis` key, and rolling(4) runs over rows, not calendar quarters. A few TTM rows also include a quarter that was filed after the TTM row's availability date.
- **Impact:** The denominator is off by up to tens of percent for groups with large subsidiaries. That misclassifies orders near the 15% bar; direction is mixed. A small mild-lookahead component is also present.

### 72. [LOW] src/agentic/test_hot_order_combo.py — lines 24, 41, 45 (d from fetch_order_fulltext.py:66 `str(d.date())`)  (3/3 upheld)
- **Lens:** LOOKAHEAD (same-day)
- **Claim:** Order filings disseminated after the 15:30 close on the entry date are counted as known at that close. `d` is truncated to a date, and `searchsorted(a, t, side='right')` includes a == trade_date.
- **Impact:** About 8-12% of orders get their first flagged week entered at the pre-announcement close, so the announcement reaction lands in y95/fc95. That is about 1-1.5% of flagged stock-weeks, and the bias is upward on C and on HARD/TIE entries. Small.

### 73. [LOW] src/agentic/test_hot_order_combo.py — lines 71, 103  (3/3 upheld)
- **Lens:** PORTFOLIO ARITHMETIC
- **Claim:** Returns across interior price gaps are silently set to zero. With pct_change(fill_method=None), both the missing day and the first day after the gap are NaN, and fillna(0) then drops the whole move across the gap.
- **Impact:** CAGR is overstated by roughly 0.1% per rotation (about 0.5 pp/yr) in all arms, a bit more for thin or T2T names. Relative comparisons between arms are mostly unaffected.

## Completeness critic

Five follow-up checks the audit did not cover, most important first. For each I name the files and what to look for. I ran quick read-only checks where noted and wrote nothing.

**1. The corporate-action adjuster only handles splits and bonuses**
- Files: `src/transform/corporate_actions.py` (`apply_split_bonus_adjustments`) and `src/agentic/repair_ca_adjustments.py`.
- In `data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet` none of these rows has an `adjustment_factor`: 101 demergers, 286 rights issues, 49 schemes of arrangement, 5 capital reductions, 2 consolidations (reverse splits) and 344 special dividends.
- The adjusted `close` still shows the unadjusted drops. Examples: CROMPGREAV -71.7% on 2016-03-15 and IIFL -60.6% on 2019-06-14, both demergers.
- Across 2016 and 2018-2021 there are 303 one-row moves below -45% or above +100%. 150 of them span a gap of more than 7 days, for example XPROINDIA +267% after a 104-day hole.
- What to check: how these reach the forward labels (`y95`, `fc95` minimum), `ret60`/`ret252`, the NAVs and the registered worst-cohort tail bar. Also list the factors the "empirical validation" policy dropped as parsed but cliffless.

**2. Feature data coverage differs by era, and "no data" is scored as "no event"**
- File: `src/agentic/anatomy_1p5x.py`, lines 158-176 (`window_count`, the shareholding merge).
- `pit_history`: about 320 rows in 2016-2018 against 13k-38k a year from 2019.
- `stock_shareholding`: 20-100 rows a year until 2020, then 2.6k-7k a year.
- `pnl_quarterly` starts 2018-04.
- `block_deals_history` ends 2026-04-29, and `announcements_historical` ends 2026-08-30, while the panel runs to 2026-09-23.
- Effect: the `prom_*`, `promoter_pct`, `prom_delta`, EPS/sales and block features act as era markers. That confounds the both-era comparison and the walk-forward model, and the last months of conf rows get zero block features.
- What to check: non-null and non-zero rate per feature per year in `rows.parquet`. Either drop the features that are not covered in both eras or add explicit coverage flags.

**3. Surveillance-series (BE/BZ) rows and the rolling features**
- Files: `src/agentic/repair_be_series_gaps.py` (T2T delivery imputation) and `src/features/indicators.py`.
- All 103k BE/BZ rows from 2025 on have `delivery_pct` exactly 1.0 (EQ median is 0.54). Before 2025 those sessions are missing entirely.
- So the delivery features in `anatomy_1p5x.py` line 42 (`delivery_pct`, `_vs_20d`, `_max_63d`, `_high_63d_flag`) work as an imputed surveillance flag in the conf era only.
- `avg_traded_value_20d` (which drives ADV and the core band), `sma_50`/`sma_200` and `return_1d` roll over rows rather than sessions. `return_1d` uses `pct_change`'s default pad fill.

**4. Backtest execution does not match the live contract**
- Files: `src/agentic/score_leader_sleeve.py` (docstring) and `src/agentic/screen_theme_leaders.py` line 186.
- Live entry is at the next session's open. Every audited simulation enters at the signal-day close.
- For the top-10 `ret60` leaders (core band, `ret252 > 0.5`), the median gap from close to next open is about +0.4% in both eras.
- 7.3% of those entries closed at the day's high after rising at least 2% (9.8% conf, 5.9% disc), so many are probably upper-circuit locks that cannot be bought at the close.
- A flat 0.5% round-trip cost is also applied to the ADV < 1cr and mcap50 arms in `sim_leader_cell_v2.py`.
- What to check: rerun the portfolio sims with next-open entry, skip entries locked at the upper circuit and exits locked at the lower circuit, and scale cost with ADV.

**5. The live screen's universe differs from the backtests (leader sleeve and 15D basket)**
- File: `src/agentic/screen_theme_leaders.py`, lines 64-65 and 83-90.
- Heat and the top-3 rank are computed on `adv >= 1.5 & close > 25` with industry groups of at least 5 names. `core_band` is only a flag at line 144, not a filter, and there is no market gate or takeover exclusion.
- No registered arm in `sim_leader_cell_v2.py` or `sim_leader_portfolio_7x.py` uses this universe, so the forward record in `outcomes.jsonl` is scoring an untested population.
- Separately, commit 4f6c34f changed `generate_hybrid_basket.non_equity` to the ISIN security master, which changes the production 15D basket universe. `backtest_10yr_15d5pct.py` does not import `non_equity`.
- What to check: the symbols that flipped, and whether any shipped 15D basket names or A/B results depend on them.

**Other gaps, lower priority:**
- The builders of `pit_history`, `stock_shareholding` and `block_deals_history` were never audited for date parsing. `pit_history` has `intimDt` values in 1922-1925 and 2027-2036.
- `sort_date` in `announcements_historical` is in IST, and 61% of announcements are timestamped at or after 15:30.
- The panel `stock_daily_facts_adjusted_2015plus.parquet` has no `.manifest.json`, which AGENTS.md requires for important outputs.