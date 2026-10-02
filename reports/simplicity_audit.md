# Simplicity Audit — 2026-10-02T12:07:15

**Scanned**: 320 files · 73,429 LOC · **249 findings**

Policy: stdlib-first, simplest correct solution, no speculative abstraction.
Findings are candidates for deletion/simplification — audit never auto-rewrites.

## Dead functions (defined, never referenced anywhere) — 22

- `src/agentic/build_data_inventory_report.py` **year_counts** — line 9
- `src/agentic/build_mcap_pit.py` **rename_successors** — line 228
- `src/agentic/build_news_event_features.py` **load_news** — line 75
- `src/agentic/eval_checks.py` **ranking_lift** — line 178
- `src/agentic/eval_checks.py` **preregistered** — line 192
- `src/agentic/eval_checks.py` **drive_checksums** — line 282
- `src/agentic/eval_checks.py` **weekly_pick_review** — line 315
- `src/agentic/eval_checks.py` **no_unexplained_cliffs** — line 345
- `src/agentic/eval_checks.py` **renames_mapped** — line 359
- `src/agentic/eval_checks.py` **insider_fresh** — line 389
- `src/agentic/eval_checks.py` **message_grounded** — line 398
- `src/agentic/eval_checks.py` **claims_sourced** — line 407
- `src/agentic/eval_checks.py` **claude_golden** — line 432
- `src/agentic/eval_checks.py` **every_feed_guarded** — line 443
- `src/agentic/fetch_forex_macro.py` **stooq_csv** — line 51
- `src/ingest/fundamentals/interface.py` **load_fundamentals** — line 24
- `src/ingest/nse/normalize.py` **read_bhavcopy_csv_text** — line 57
- `src/ingest/sector_flow/interface.py` **load_sector_flow** — line 20
- `src/ingest/shareholding/interface.py` **load_shareholding** — line 23
- `src/master/stock_master.py` **build_stock_master_from_symbols** — line 40
- `src/transform/sector_flow_daily.py` **forward_fill_sector_flow_daily** — line 6
- `src/utils/validation.py` **assert_no_future_leakage** — line 12

## Dead classes — 0


## Unused imports — 92

- `src/agentic/ab_event_features_15d.py` **sys** — from sys
- `src/agentic/ab_event_rules_15d.py` **re** — from re
- `src/agentic/ab_event_rules_15d.py` **sys** — from sys
- `src/agentic/ab_fresh_leader.py` **np** — from numpy
- `src/agentic/ab_vol_gate.py` **sys** — from sys
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
- `src/agentic/build_status_dashboard.py` **subprocess** — from subprocess
- `src/agentic/build_symbol_isin_history.py` **sys** — from sys
- `src/agentic/compute_feature_importance.py` **np** — from numpy
- `src/agentic/data_completeness.py` **np** — from numpy
- `src/agentic/devils_advocate.py` **json** — from json
- `src/agentic/devils_advocate.py` **np** — from numpy
- `src/agentic/event_materiality_study.py` **re** — from re
- `src/agentic/explore_hitter_waves.py` **np** — from numpy
- `src/agentic/explore_pnl_winners.py` **sys** — from sys
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
- `src/agentic/fetch_pnl_history.py` **ET** — from xml
- `src/agentic/fetch_pnl_old_format.py` **shutil** — from shutil
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
- `src/agentic/generate_pro_brief.py` **np** — from numpy
- `src/agentic/generate_trade_plan.py` **np** — from numpy
- `src/agentic/hypothesis_agent.py` **pd** — from pandas
- `src/agentic/inspect_symbol.py` **sys** — from sys
- … and 32 more

## Single-method stateless classes (should be functions) — 0


## Trivial wrappers (single-call bodies) — 54

- `src/agentic/ab_event_features_15d.py` **had** — line 138
- `src/agentic/ab_valuation_3h.py` **fwd** — line 135
- `src/agentic/agent_loop.py` **load_registry** — line 52
- `src/agentic/anatomy_1p5x.py` **era_of** — line 247
- `src/agentic/autopsy_leader_winners.py` **B** — line 111
- `src/agentic/backfill_bulk_deals.py` **months** — line 37
- `src/agentic/backtest_10yr_15d5pct.py` **qc** — line 126
- `src/agentic/backtest_hybrid_15d5pct.py` **qc** — line 118
- `src/agentic/backtest_sleeve_walkforward.py` **fwd_max** — line 38
- `src/agentic/build_dashboard.py` **extract_mermaid** — line 65
- `src/agentic/build_html_viewer.py` **extract_mermaid_blocks** — line 23
- `src/agentic/build_industry_scores.py` **pct** — line 49
- `src/agentic/build_policy_scores.py` **pct** — line 34
- `src/agentic/engine_replay.py` **lgbm** — line 54
- `src/agentic/engine_replay.py` **xgbm** — line 60
- `src/agentic/eval_checks.py` **_res** — line 28
- `src/agentic/fetch_announcements_historical.py` **has_chunk** — line 65
- `src/agentic/fetch_budget_capex.py` **fy_start** — line 224
- `src/agentic/fetch_iip_core.py` **ym** — line 329
- `src/agentic/fetch_iip_core.py` **fnum** — line 333
- `src/agentic/fetch_iip_core.py` **norm_ws** — line 343
- `src/agentic/fetch_iip_core.py` **fy_month** — line 425
- `src/agentic/fetch_iip_core.py` **parse_iip_pdf** — line 742
- `src/agentic/fetch_pib_releases.py` **now_ist** — line 143
- `src/agentic/fetch_pib_releases.py` **_norm** — line 147
- `src/agentic/fetch_pib_releases.py` **_meta_path** — line 325
- `src/agentic/mine_doubler_ignition.py` **fwd** — line 40
- `src/agentic/mine_fast_double_conditional.py` **fwd** — line 52
- `src/agentic/mine_highvol_subcohorts.py` **fend** — line 51
- `src/agentic/mine_industry_contagion.py` **fwd** — line 37
- `src/agentic/pocket_search_leader.py` **cut** — line 102
- `src/agentic/research_queue.py` **has_code** — line 43
- `src/agentic/sim_allin_matrix.py` **fixed_offsets** — line 337
- `src/agentic/sim_leader_cell_v2.py` **_norm** — line 125
- `src/agentic/sim_leader_portfolio_7x.py` **_period** — line 478
- `src/agentic/sweep_horizon_2x_year.py` **fwd_max** — line 40
- `src/agentic/trust/data_ready.py` **qc** — line 61
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

## Duplicated function bodies (shape-identical) — 34

- `2 copies` **build_panel** — src/agentic/ab_test_event_features.py:build_panel, src/agentic/ab_test_event_polarity.py:build_panel
- `2 copies` **c2** — src/agentic/ab_vol_gate.py:c2, src/agentic/ab_zscore_bands.py:c2
- `2 copies` **build_panel** — src/agentic/backtest_10yr.py:build_panel, src/agentic/backtest_10yr_macro.py:build_panel
- `2 copies` **snap_features** — src/agentic/backtest_10yr_15d5pct.py:snap_features, src/agentic/backtest_hybrid_15d5pct.py:snap_features
- `2 copies` **load_oof** — src/agentic/backtest_event_driven.py:load_oof, src/agentic/backtest_event_window.py:load_oof
- `4 copies` **build_panel** — src/agentic/backtest_multibagger_strategy.py:build_panel, src/agentic/find_achievable_targets.py:build_panel, src/agentic/find_multibagger_targets.py:build_panel, src/agentic/find_multibagger_today.py:build_panel
- `3 copies` **build_target** — src/agentic/backtest_multibagger_strategy.py:build_target, src/agentic/find_multibagger_targets.py:build_target, src/agentic/find_multibagger_today.py:build_target
- `2 copies` **fred_csv** — src/agentic/fetch_commodity_prices.py:fred_csv, src/agentic/fetch_global_rates.py:fred_csv
- `2 copies` **tag_symbols** — src/agentic/fetch_reddit.py:tag_symbols, src/agentic/fetch_youtube.py:tag_symbols
- `2 copies` **get_top_symbols** — src/agentic/fetch_screener_fundamentals.py:get_top_symbols, src/agentic/fetch_stock_fii_dii.py:get_top_symbols
- `2 copies` **main** — src/agentic/hypothesis_agent.py:main, src/agentic/hypothesis_agent_macro.py:main
- `2 copies` **fading_only** — src/agentic/report_pick_stats.py:fading_only, src/agentic/report_v3_stats.py:fading_only
- `2 copies` **model_scores** — src/agentic/sim_screen_rank_exit.py:model_scores, src/agentic/test_v3_pnl.py:scores
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

## Third-party deps with a stdlib equivalent — 17

- `src/agentic/fetch_iip_core.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_news_rss.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_pib_releases.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_reddit.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_screener_industry.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_screener_mcap_backfill.py` **requests** — urllib.request (already used by every fetcher in this repo)
- `src/agentic/fetch_usdinr_history.py` **requests** — urllib.request (already used by every fetcher in this repo)
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

## Files > 800 LOC (split candidates) — 12

- `src/agentic/anatomy_1p5x.py` **** — 930 LOC
- `src/agentic/build_dashboard.py` **** — 1114 LOC
- `src/agentic/fetch_budget_capex.py` **** — 1003 LOC
- `src/agentic/fetch_iip_core.py` **** — 2001 LOC
- `src/agentic/fetch_pib_releases.py` **** — 902 LOC
- `src/agentic/model_bakeoff_1p5x.py` **** — 872 LOC
- `src/agentic/sim_leader_portfolio_7x.py` **** — 815 LOC
- `src/agentic/test_hot_order_combo.py` **** — 843 LOC
- `src/analysis/week7_15pct_cluster_rerank_compare.py` **** — 886 LOC
- `src/analysis/week7_15pct_random_forest_allnames.py` **** — 923 LOC
- `src/analysis/week7_5pct_gbm_allnames_macro_veto.py` **** — 1054 LOC
- `src/report/production_weekly_run.py` **** — 989 LOC

## Functions nested > 5 deep — 18

- `src/agentic/build_mcap_pit.py` **rename_links** — depth>5
- `src/agentic/build_price_only_ca_factors.py` **main** — depth>5
- `src/agentic/fetch_budget_capex.py` **parse_stat3** — depth>5
- `src/agentic/fetch_iip_core.py` **parse_iip_text** — depth>5
- `src/agentic/fetch_iip_core.py` **parse_ici_xlsx** — depth>5
- `src/agentic/fetch_iip_core.py` **build_iip** — depth>5
- `src/agentic/fetch_iip_core.py` **build_core** — depth>5
- `src/agentic/fetch_order_book.py` **crawl** — depth>5
- `src/agentic/fetch_order_fulltext.py` **main** — depth>5
- `src/agentic/fetch_pib_releases.py` **backfill** — depth>5
- `src/agentic/miss_learner.py` **analyze_misses** — depth>5
- `src/agentic/repair_ca_adjustments.py` **main** — depth>5
- `src/agentic/sim_leader_portfolio_7x.py` **run** — depth>5
- `src/agentic/sim_screen_rank_exit.py` **run_exit** — depth>5
- `src/agentic/simplicity_auditor.py` **audit** — depth>5
- `src/agentic/test_hot_order_combo.py` **simulate** — depth>5
- `src/agentic/trust/claim_check.py` **read_session** — depth>5
- `src/analysis/week7_15pct_meta_rerank_compare.py` **_run_single_model** — depth>5

## Debt ledger — 25 open / 27 total

- 2026-09-19 `src/agentic/verify_freshness.py MACRO_PANEL contract` — breadth_50/adv_decl_ratio (price-derived) not column-checked; 7 FRED spread columns dead since Apr-30 with no contract (why: found during 09-19 reconcile; adding contracts without fixing fetch_global_rates.py would hard-block every basket; loc 6; speed none)
- 2026-09-19 `src/agentic/render_basket_report.py _eta_days` — ETA is vol-implied median first-passage (0.45*(0.05/vol)^2), not calibrated on realized touch times (why: first report needed today; the 4,222-trade day-by-day backtest has the realized days-to-touch to calibrate against; loc 20; speed none)
- 2026-09-23 `src/agentic/sim_leader_sleeve.py maxdd_of_cohort_path (used by EXP-2026-09-23-extended-leader)` — maxDD is the sim's smoothed rolling-26 proxy, not a true overlapping-cohort daily NAV (why: 6e had to use the metric the registered +11.1/+19.3% leader result was judged on; changing it mid-experiment would be a second experiment; loc +40; speed +1 min)
- 2026-09-23 `src/agentic/screen_theme_leaders.py EXTENDED_ONLY filter` — filter validated on the core band (ADV>=5cr, close>50) but the live screen still lists expanded-band names (e.g. RSWM, band exp) (why: spec 6a keeps the band tag as evidence, not a filter; sim has no expanded-band arm; loc +10; speed 0)
- 2026-09-23 `src/agentic/screen_theme_leaders.py --asof replays` — industry map = modal smIndustry over all announcements (not point-in-time); SHP filtered by quarter_end<=asof without disclosure lag (why: forward screens (asof = panel max) are unaffected; only historical replays can leak; loc +8; speed 0)
- 2026-09-23 `logs/leader_sleeve/screen_20260908.json` — backfilled from reports/theme_leaders_20260908.md (the committed artifact), industry names truncated to 29 chars as printed; close/own252 recomputed from panel as of 2026-09-07 (why: the 09-08 screen predates the JSON output; re-running --asof 20260907 on today's (repaired, CA-rescaled) panel would not reproduce the issued list exactly; loc 0; speed 0)
- 2026-09-23 `announcements_historical.parquet smIndustry -> screen_theme_leaders/sim_leader_sleeve industry map` — symbols without an smIndustry row (e.g. SKYGOLD, SHANTIGOLD) are dropped from the leader cell entirely (why: industry map is the modal smIndustry of historical announcements; newer listings have none; loc +15; speed 0)
- 2026-09-23 `src/agentic/build_macro_panel.py step 8 (macro_sent__*)` — latest macro-sentiment snapshot is broadcast to EVERY historical row — constant columns, lookahead if any model trains on them (why: pre-existing; surfaced 2026-09-23 when the gold/crude sentiment columns looked 'frozen' (they are a broadcast, not stale); loc +10; speed 0)
- 2026-09-23 `CA store / adjusted panel for ETFs` — ETF unit splits are absent from the CA store (GOLDBEES adjusted close shows -94.5% 2016->2026) (why: CA ingest covers equities; build_gold_feed.py survives it via cross-ETF median + |r|>15% drop; loc +20; speed 0)
- 2026-09-23 `src/agentic/fetch_filing_text.py` — manual: run with SYMBOL:seq_id pairs; not scheduled (why: first use was 4 flagged filings on request; loc +25; speed +1 min/day)
- 2026-09-23 `src/agentic/backtest_10yr_15d5pct.py (EXP-2026-09-24-engines-count-sizing)` — engines_count A/B registered but not run; no engines_count in the 10y trade table (why: The five engines take no as-of date: each trains LGBM+XGB on the full panel and scores only the latest day (31 min serial for one as-of on 2026-09-23). A walk-forward engines_count needs a replay harness that refits all 5 engines at each cutoff and scores every weekly Monday since 2016 (~540 windows). Even with yearly refits and no OOF-CV that is ~10 refits x ~12 model pairs, an est. 2-4 h serial on a 24 GB Mac, plus porting each engine's panel/targets into cutoff-aware functions. Fidelity limits: extras exist only from 2023-06 (median-filled before, so pre-2023 hc/cs are different models from live); mh reads catalyst_features/sector_index_members, which may not be point-in-time. The backtest itself still uses the flat -3% SL (banned) and the non-canonical CA store (_incremental/normalized), so C2 (ab_vol_gate.c2) and the canonical CA path must go in first; loc +~350; speed one-off 2-4 h serial)
- 2026-09-23 `data/derived/extra_features.parquet / feature_factory.py / verify_freshness.py` — profiled but NOT wired into daily_data_layer.sh or the gate (update to 2026-09-19 entry) (why: Profile 2026-09-24 on the repaired panel: peak RSS 13.5 GB (14,475,296,768 B), 44 s wall, 1,862,326 rows x 184 cols, 2023-06-01..2026-09-23, so memory is fine. But cs AND hc both join this file, so refreshing it in place changes both engines immediately, and a 3-bd Contract blocks every basket until it is refreshed. The registered A/B (fresh vs incumbent median-filled extras, >=13 weekly windows) differs only after 2026-06-17; windows with a matured 15d label run 2026-06-22..2026-09-01 = 11 Mondays, so the >=13 bar can't be met before data through ~2026-10-13. Fresh file also adds 5 macro_gold_inr_* cols that pass the macro_ SAFE prefix: a feature-set change too; loc +5; speed +44 s per data-layer run)
- 2026-09-23 `reports/simplicity_audit.md` — audit 185 findings vs 162 at the 2026-07-07 baseline (+13 unused imports, +8 trivial wrappers, +1 dead func, +1 deep nesting), not zero (why: the delta sits in the 47 research scripts added since July (ab_*/mine_*/autopsy_*), not in the production files touched this session (only the unused numpy import in build_news_event_features.py, removed). Cleaning ~20 untracked research scripts is out of scope for a data-integrity session; loc -25; speed 0)
- 2026-09-23 `data/derived/news_feed.parquet (fetch_news_rss.py)` — RSS store begins Apr-2026 (July missing); symbol tagger matches tickers only (GRT/TBZ headline has symbols=[]) (why: free RSS has no history; tagger predates company-name matching; loc +15; speed 0)
- 2026-09-23 `screen_theme_leaders.py / sim_leader_sleeve.py group heat` — industry heat = MEAN own ret60 over >=5 names — one outlier (TBZ +182%) makes a flat group 'hot' (why: the validated cell was defined this way; changing it is a new experiment; loc +10; speed 0)
- 2026-09-23 `src/agentic/run_multi_horizon.py (mh engine) / verify_freshness.py` — data/derived/catalyst_features.parquet ends 2026-06-01 and is ungated; mh fillna(0.0)s every catalyst feature after that date, and its sector map is today's index membership (tmp/from_scratch_7d_run/alt2/sector_index_members.parquet, 1,726 rows), not point-in-time (why: found while building engine_replay.py 2026-09-24. Same class as extra_features (sgm-data-integrity #3): an 'optional' join silently going stale. Zero-filling is a silent model change, and it's worse than median-filling because 0 means 'no catalyst'; loc +5; speed unknown until profiled)
- 2026-09-24 `render_leader_report.PRIOR_2X / screen overlays` — LEADER&cheap and FRESH priors still from the v1 sim on the old map; screen PE overlay shows demerger/tiny-EPS artifacts (RAYMOND PE 1.4, KABRAEXTRU PE 5,235) (why: only the production cell was re-run on nse4 this session; loc +15; speed 0)
- 2026-09-24 `src/agentic/pocket_search_leader.py` — re-implements sim_leader_cell_v2 panel/path/heat machinery instead of importing it (sim is a top-level script) (why: one-session search; refactoring the sim mid-experiment risks changing registered numbers; loc -80; speed 0)
- 2026-09-27 `data/derived/stock_daily_facts_adjusted_2015plus.parquet` — symbols that only ever traded BE/BZ (never EQ) are not in the panel; 2015-16 coverage 98-99.5% of raw EQ/BE/BZ symbols (2017+ >= 99.57%) (why: repair_be_series_gaps inserts only missing sessions of symbols already in the panel (no new symbols); loc +20; speed 0)
- 2026-09-27 `data/corporate_actions_full_history (CA store)` — corporate actions missing from the NSE CA feed (e.g. CROMPGREAV demerger 2016-03-15) stay unadjusted (why: price-only factors are derived only for CA-store rows; auto-adjusting unexplained cliffs risks erasing real crashes; loc +40; speed 0)