# Simplicity Audit — 2026-07-07T18:35:36

**Scanned**: 191 files · 45,159 LOC · **162 findings**

Policy: stdlib-first, simplest correct solution, no speculative abstraction.
Findings are candidates for deletion/simplification — audit never auto-rewrites.

## Dead functions (defined, never referenced anywhere) — 9

- `src/agentic/build_news_event_features.py` **load_news** — line 76
- `src/agentic/fetch_forex_macro.py` **stooq_csv** — line 51
- `src/ingest/fundamentals/interface.py` **load_fundamentals** — line 24
- `src/ingest/nse/normalize.py` **read_bhavcopy_csv_text** — line 57
- `src/ingest/sector_flow/interface.py` **load_sector_flow** — line 20
- `src/ingest/shareholding/interface.py` **load_shareholding** — line 23
- `src/master/stock_master.py` **build_stock_master_from_symbols** — line 40
- `src/transform/sector_flow_daily.py` **forward_fill_sector_flow_daily** — line 6
- `src/utils/validation.py` **assert_no_future_leakage** — line 12

## Dead classes — 0


## Unused imports — 76

- `src/agentic/analyze_superstar_alpha.py` **np** — from numpy
- `src/agentic/analyze_superstar_horizons.py` **np** — from numpy
- `src/agentic/backtest_10yr.py` **np** — from numpy
- `src/agentic/backtest_10yr_with_factors.py` **np** — from numpy
- `src/agentic/backtest_10yr_with_factors.py` **IsotonicRegression** — from sklearn
- `src/agentic/backtest_dynamic_gated.py` **np** — from numpy
- `src/agentic/backtest_event_window.py` **np** — from numpy
- `src/agentic/backtest_multibagger_strategy.py` **np** — from numpy
- `src/agentic/backtest_news_features.py` **json** — from json
- `src/agentic/backtest_regime_gated.py` **np** — from numpy
- `src/agentic/build_confluence_picks.py` **np** — from numpy
- `src/agentic/build_event_polarity.py` **np** — from numpy
- `src/agentic/build_handoff.py` **json** — from json
- `src/agentic/build_news_event_features.py` **np** — from numpy
- `src/agentic/build_status_dashboard.py` **subprocess** — from subprocess
- `src/agentic/compute_feature_importance.py` **np** — from numpy
- `src/agentic/data_completeness.py` **np** — from numpy
- `src/agentic/devils_advocate.py` **json** — from json
- `src/agentic/devils_advocate.py` **np** — from numpy
- `src/agentic/factor_evaluator.py` **stats** — from scipy
- `src/agentic/factor_registry.py` **field** — from dataclasses
- `src/agentic/fetch_amfi_mf_holdings.py` **time** — from time
- `src/agentic/fetch_block_deals.py` **io** — from io
- `src/agentic/fetch_broker_recos.py` **http** — from http
- `src/agentic/fetch_broker_recos.py` **ssl** — from ssl
- `src/agentic/fetch_fii_dii.py` **time** — from time
- `src/agentic/fetch_forex_macro.py` **gzip** — from gzip
- `src/agentic/fetch_forex_macro.py` **http** — from http
- `src/agentic/fetch_forex_macro.py` **ssl** — from ssl
- `src/agentic/fetch_fundamentals.py` **io** — from io
- `src/agentic/fetch_fundamentals.py` **sys** — from sys
- `src/agentic/fetch_global_macro.py` **timedelta** — from datetime
- `src/agentic/fetch_news_per_symbol.py` **re** — from re
- `src/agentic/fetch_pib_releases.py` **sys** — from sys
- `src/agentic/fetch_reddit.py` **hashlib** — from hashlib
- `src/agentic/fetch_screener_fundamentals.py` **timezone** — from datetime
- `src/agentic/fetch_screener_screens.py` **timezone** — from datetime
- `src/agentic/fetch_stock_fii_dii.py` **ET** — from xml
- `src/agentic/fetch_stock_fii_dii.py` **timezone** — from datetime
- `src/agentic/fetch_superstar_holdings.py` **timezone** — from datetime
- `src/agentic/fetch_youtube.py` **hashlib** — from hashlib
- `src/agentic/filter_cascade.py` **np** — from numpy
- `src/agentic/find_180d_frontier_honest.py` **np** — from numpy
- `src/agentic/find_achievable_frontier.py` **np** — from numpy
- `src/agentic/find_multibagger_today.py` **np** — from numpy
- `src/agentic/generate_event_driven_today.py` **timedelta** — from datetime
- `src/agentic/generate_event_driven_today.py` **np** — from numpy
- `src/agentic/generate_hybrid_basket.py` **np** — from numpy
- `src/agentic/generate_pro_brief.py` **np** — from numpy
- `src/agentic/generate_trade_plan.py` **np** — from numpy
- `src/agentic/hypothesis_agent.py` **pd** — from pandas
- `src/agentic/inspect_symbol.py` **sys** — from sys
- `src/agentic/joint_signal_analyzer.py` **np** — from numpy
- `src/agentic/miss_learner.py` **timedelta** — from datetime
- `src/agentic/miss_learner.py` **np** — from numpy
- `src/agentic/paper_trading_recorder.py` **json** — from json
- `src/agentic/portfolio_sizer.py` **np** — from numpy
- `src/agentic/risk_envelope.py` **np** — from numpy
- `src/agentic/run_multi_horizon.py` **json** — from json
- `src/agentic/run_multi_horizon.py` **np** — from numpy
- … and 16 more

## Single-method stateless classes (should be functions) — 0


## Trivial wrappers (single-call bodies) — 24

- `src/agentic/agent_loop.py` **load_registry** — line 52
- `src/agentic/backtest_10yr_15d5pct.py` **qc** — line 124
- `src/agentic/backtest_hybrid_15d5pct.py` **qc** — line 118
- `src/agentic/build_dashboard.py` **extract_mermaid** — line 65
- `src/agentic/build_html_viewer.py` **extract_mermaid_blocks** — line 23
- `src/agentic/fetch_announcements_historical.py` **has_chunk** — line 65
- `src/agentic/fetch_pib_releases.py` **has_shard** — line 120
- `src/analysis/week7_15pct_cluster_rerank_compare.py` **_make_relaxed_rule** — line 83
- `src/analysis/week7_15pct_random_forest_allnames.py` **_combine_focus_score** — line 55
- `src/analysis/week7_universe_contextual_bandit.py` **_bool_to_float** — line 60
- `src/analysis/weekly_run_gate_search.py` **_gate_columns_available** — line 130
- `src/ingest/events/nse.py` **_contains_any** — line 199
- `src/ingest/fundamentals/nse.py` **_all_positive** — line 293
- `src/ingest/nse/fetch_bhavcopy.py` **build_nse_delivery_url** — line 42
- `src/ingest/nse/io.py` **utc_now_iso** — line 42
- `src/ingest/public_fallback/groww.py` **_parse_groww_quarter_label** — line 292
- `src/ml/expert_pipeline.py` **_combine_focus_score** — line 624
- `src/ml/expert_pipeline.py` **_to_objective** — line 707
- `src/ml/metrics.py` **brier_score** — line 7
- `src/portfolio/state.py` **_empty_current_positions** — line 63
- `src/portfolio/state.py` **_empty_execution_ledger** — line 67
- `src/report/checklist.py` **_strictly_rising** — line 225
- `src/report/weekly_portfolio_report.py` **_default_target_date** — line 55
- `src/utils/data_catalog.py` **sidecar_manifest_path** — line 120

## Duplicated function bodies (shape-identical) — 32

- `2 copies` **build_panel** — src/agentic/ab_test_event_features.py:build_panel, src/agentic/ab_test_event_polarity.py:build_panel
- `2 copies` **build_panel** — src/agentic/backtest_10yr.py:build_panel, src/agentic/backtest_10yr_macro.py:build_panel
- `2 copies` **snap_features** — src/agentic/backtest_10yr_15d5pct.py:snap_features, src/agentic/backtest_hybrid_15d5pct.py:snap_features
- `2 copies` **load_oof** — src/agentic/backtest_event_driven.py:load_oof, src/agentic/backtest_event_window.py:load_oof
- `2 copies` **load_prices** — src/agentic/backtest_hybrid_15d5pct.py:load_prices, src/agentic/train_missed_winner_classifier.py:load_prices
- `4 copies` **build_panel** — src/agentic/backtest_multibagger_strategy.py:build_panel, src/agentic/find_achievable_targets.py:build_panel, src/agentic/find_multibagger_targets.py:build_panel, src/agentic/find_multibagger_today.py:build_panel
- `3 copies` **build_target** — src/agentic/backtest_multibagger_strategy.py:build_target, src/agentic/find_multibagger_targets.py:build_target, src/agentic/find_multibagger_today.py:build_target
- `2 copies` **fred_csv** — src/agentic/fetch_commodity_prices.py:fred_csv, src/agentic/fetch_global_rates.py:fred_csv
- `2 copies` **tag_symbols** — src/agentic/fetch_reddit.py:tag_symbols, src/agentic/fetch_youtube.py:tag_symbols
- `2 copies` **get_top_symbols** — src/agentic/fetch_screener_fundamentals.py:get_top_symbols, src/agentic/fetch_stock_fii_dii.py:get_top_symbols
- `2 copies` **main** — src/agentic/hypothesis_agent.py:main, src/agentic/hypothesis_agent_macro.py:main
- `2 copies` **_select_daily_top_n** — src/analysis/day1_5pct_model.py:_select_daily_top_n, src/analysis/day1_model_challenger.py:_select_daily_top_n
- `2 copies` **_evaluate_rf_metrics** — src/analysis/day1_random_forest_quick_compare.py:_evaluate_rf_metrics, src/analysis/week7_random_forest_quick_compare.py:_evaluate_rf_metrics
- `2 copies` **_fit_predict_classifier** — src/analysis/day1_random_forest_quick_compare.py:_fit_predict_classifier, src/analysis/week7_random_forest_quick_compare.py:_fit_predict_classifier
- `2 copies` **main** — src/analysis/day1_random_forest_quick_compare.py:main, src/analysis/week7_random_forest_quick_compare.py:main
- `2 copies` **_read_optional_table** — src/analysis/forward_return_study.py:_read_optional_table, src/analysis/threshold_study.py:_read_optional_table
- `2 copies` **_fit_predict_classifier** — src/analysis/week7_15pct_gbm_allnames.py:_fit_predict_classifier, src/analysis/week7_5pct_gbm_allnames_macro_veto.py:_fit_predict_classifier
- `3 copies` **_apply_calibration_15pct** — src/analysis/week7_15pct_random_forest_allnames.py:_apply_calibration_15pct, src/analysis/week7_5pct_gbm_allnames_macro_veto.py:_apply_calibration_5pct, src/analysis/week7_5pct_gbm_allnames_macro_veto.py:_apply_screened_calibration_5pct
- `2 copies` **_evaluate_daily_metrics** — src/analysis/week7_15pct_random_forest_allnames.py:_evaluate_daily_metrics, src/analysis/week7_5pct_gbm_allnames_macro_veto.py:_evaluate_daily_metrics
- `2 copies` **_evaluate_weekly_metrics** — src/analysis/week7_15pct_random_forest_allnames.py:_evaluate_weekly_metrics, src/analysis/week7_5pct_gbm_allnames_macro_veto.py:_evaluate_weekly_metrics
- `2 copies` **_load_champion_metrics** — src/analysis/week7_model_family_quick_compare.py:_load_champion_metrics, src/analysis/week7_random_forest_quick_compare.py:_load_champion_metrics
- `4 copies` **_iter_windows** — src/ingest/corporate_actions/nse.py:_iter_windows, src/ingest/events/nse.py:_iter_windows, src/ingest/events/nse_insider.py:_iter_windows, src/ingest/macro/nse_fred.py:iter_api_windows
- `2 copies` **_fetch_symbol_history** — src/ingest/fundamentals/nse.py:_fetch_symbol_history, src/ingest/shareholding/nse.py:_fetch_symbol_history
- `2 copies` **_to_number** — src/ingest/fundamentals/nse.py:_to_number, src/ingest/shareholding/nse.py:_to_number
- `2 copies` **_fetch_listing_rows** — src/ingest/fundamentals/nse_batched.py:_fetch_listing_rows, src/ingest/shareholding/nse_batched.py:_fetch_master_rows
- `2 copies` **_read_cached_listing_rows** — src/ingest/fundamentals/nse_batched.py:_read_cached_listing_rows, src/ingest/shareholding/nse_batched.py:_read_cached_master_rows
- `2 copies` **_dedupe_listing_rows** — src/ingest/fundamentals/nse_batched.py:_dedupe_listing_rows, src/ingest/shareholding/nse_batched.py:_dedupe_master_rows
- `2 copies` **_iter_quarter_windows** — src/ingest/fundamentals/nse_batched.py:_iter_quarter_windows, src/ingest/shareholding/nse_batched.py:_iter_quarter_windows
- `2 copies` **_thread_session** — src/ingest/fundamentals/nse_batched.py:_thread_session, src/ingest/shareholding/nse_batched.py:_thread_session
- `2 copies` **main** — src/ml/cli.py:main, src/ml/expert_cli.py:main
- `2 copies` **load_current_positions** — src/portfolio/state.py:load_current_positions, src/portfolio/state.py:load_execution_ledger
- `2 copies` **_suggested_allocation** — src/report/stateful_weekly_winners.py:_suggested_allocation, src/report/weekly_portfolio_report.py:_suggested_allocation

## Third-party deps with a stdlib equivalent — 13

- `src/agentic/fetch_news_rss.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_pib_releases.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_reddit.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_youtube.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/derivatives/nse_oi.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/events/nse_bulk_block.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/events/nse_insider.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/macro/nse_fred.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/nse/api.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/nse/api.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/nse/session.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/nse/session.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/ingest/public_fallback/groww.py` **requests** — urllib.request (already used by every fetcher in this repo)

## Files > 800 LOC (split candidates) — 5

- `src/agentic/build_dashboard.py` **** — 1114 LOC
- `src/analysis/week7_15pct_cluster_rerank_compare.py` **** — 886 LOC
- `src/analysis/week7_15pct_random_forest_allnames.py` **** — 923 LOC
- `src/analysis/week7_5pct_gbm_allnames_macro_veto.py` **** — 1054 LOC
- `src/report/production_weekly_run.py` **** — 989 LOC

## Functions nested > 5 deep — 3

- `src/agentic/miss_learner.py` **analyze_misses** — depth>5
- `src/agentic/simplicity_auditor.py` **audit** — depth>5
- `src/analysis/week7_15pct_meta_rerank_compare.py` **_run_single_model** — depth>5

## Debt ledger — 3 open / 4 total

- 2026-07-07 `src/agentic/generate_hybrid_basket.py` — ML>=0.85 penalty (-0.5 band_fit) routes around classifier overconfidence instead of recalibrating the classifier (why: ship corrected basket same day as the 10-yr backtest finding; loc 8; speed none)
- 2026-07-07 `src/agentic/backtest_10yr_15d5pct.py` — band-fit>=2 subset not yet re-run under day-by-day sequenced exits (why: day-by-day correction landed 2026-07-07; full rerun takes hours; loc 0; speed unknown until rerun)
- 2026-07-07 `src/agentic/fetch_global_macro.py` — nifty_50/bank_nifty/shcomp/gold have NO fallback source when Yahoo 429s (why: FRED has no NSE index series; alternates need research; loc 0; speed none)