#!/bin/bash
# Daily agentic pipeline — fetches alt data, refreshes catalyst features, retrains model, builds brief.
# Logs to logs/daily_pipeline_<date>.log

set -uo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
DATE_TAG=$(date +%Y%m%d_%H%M)
LOG="$ROOT/logs/daily_pipeline_${DATE_TAG}.log"
mkdir -p "$ROOT/logs"

PY="/usr/bin/python3"
export PYTHONPATH="$ROOT"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:$PATH"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_step() {
    local name="$1"; shift
    log ">>> START: $name"
    if "$@" >> "$LOG" 2>&1; then
        log "<<< OK:    $name"
    else
        local rc=$?
        log "<<< FAIL ($rc): $name (continuing)"
    fi
}

log "=========================================="
log "DAILY PIPELINE start  cwd=$ROOT"
log "=========================================="

run_step "refresh prices (bhavcopy + features)" $PY -m src.agentic.refresh_prices
run_step "refresh announcements + insider"  $PY -m src.agentic.refresh_announcements
run_step "tag announcements"                $PY -m src.agentic.catalyst_tagger \
    --in tmp/from_scratch_7d_run/alt/corp_announcements.parquet \
    --out tmp/from_scratch_7d_run/alt/announcements_tagged.parquet \
    --prices data/derived/stock_daily_facts_adjusted_2015plus.parquet
run_step "fetch block + bulk deals"         $PY src/agentic/fetch_block_deals.py
run_step "build catalyst features"          $PY -m src.agentic.build_catalyst_features \
    --ann tmp/from_scratch_7d_run/alt/announcements_tagged.parquet \
    --pit tmp/from_scratch_7d_run/alt/insider_trading_pit.parquet \
    --prices data/derived/stock_daily_facts_adjusted_2015plus.parquet \
    --out data/derived/catalyst_features.parquet

run_step "fetch news RSS"                   $PY src/agentic/fetch_news_rss.py
run_step "fetch per-symbol Google News (top-300+picks)" $PY src/agentic/fetch_news_per_symbol.py
run_step "fetch broker recommendations (Moneycontrol/ET/BS)" $PY src/agentic/fetch_broker_recos.py
run_step "fetch reddit"                     $PY src/agentic/fetch_reddit.py
run_step "fetch youtube"                    $PY src/agentic/fetch_youtube.py
run_step "score sentiment (news+reddit+yt, finance lexicon)" $PY src/agentic/score_sentiment.py
run_step "fetch options chain (F&O IV/OI)"  $PY src/agentic/fetch_options_chain.py
run_step "fetch fundamentals (NSE top-500)" $PY -m src.agentic.fetch_fundamentals
run_step "fetch macro time-series (USDINR/EUR/GBP/JPY via Frankfurter)" $PY src/agentic/fetch_forex_macro.py
run_step "fetch FII/DII daily flows (NSE)" $PY src/agentic/fetch_fii_dii.py
run_step "fetch Wikipedia pageviews (retail attention proxy)" $PY src/agentic/fetch_wiki_pageviews.py
run_step "fetch superstar holdings (Tickertape top-20)"     $PY src/agentic/fetch_superstar_holdings.py
run_step "fetch Screener.in curated screens (FII/DII)"     $PY src/agentic/fetch_screener_screens.py
run_step "fetch Screener fundamentals (40+ ratios)"        $PY src/agentic/fetch_screener_fundamentals.py --top-n 200
run_step "analyze superstar alpha (confluence vs baseline)" $PY src/agentic/analyze_superstar_alpha.py
run_step "analyze superstar alpha by horizon (7d to 252d)"  $PY src/agentic/analyze_superstar_horizons.py
run_step "run model diversity panel (5 ML families)"        $PY src/agentic/run_model_diversity.py

run_step "retrain v3 ensemble (long, 7d)"   $PY src/agentic/run_v3_with_catalysts.py
run_step "retrain short-side model"         $PY src/agentic/run_short_side.py
run_step "sector-weak short overlay (large-cap rotation)" $PY src/agentic/sector_weak_shorts.py
run_step "retrain multi-horizon (1d/7d/21d)" $PY src/agentic/run_multi_horizon.py
run_step "size portfolio (Kelly + regime)"  $PY src/agentic/portfolio_sizer.py
run_step "record paper-trading picks"       $PY src/agentic/paper_trading_recorder.py
run_step "data completeness audit (the gate)" $PY src/agentic/data_completeness.py
run_step "filter cascade (discipline layer)" $PY src/agentic/filter_cascade.py
run_step "generate daily brief"             $PY src/agentic/generate_daily_brief.py
run_step "generate pro brief (Bull/Base/Bear)" $PY src/agentic/generate_pro_brief.py
run_step "build workflow diagram"           $PY src/agentic/build_workflow_diagram.py
run_step "build status dashboard"           $PY src/agentic/build_status_dashboard.py
run_step "build HTML visualizer (developer)" $PY src/agentic/build_html_viewer.py
run_step "compute feature importance"       $PY src/agentic/compute_feature_importance.py
run_step "build human dashboard (user)"     $PY src/agentic/build_dashboard.py
run_step "find high-conviction (3-horizon, 80% bar)" $PY src/agentic/find_high_conviction.py
run_step "emit conviction alert"            $PY src/agentic/monitor_for_conviction.py
run_step "find today's multibagger candidates" $PY src/agentic/find_multibagger_today.py
run_step "track multibagger basket performance" $PY src/agentic/track_multibagger_basket.py
run_step "build confluence picks (7-layer aggregator)" $PY src/agentic/build_confluence_picks.py
run_step "regime check for multibagger deploy/wait" $PY src/agentic/analyze_regime_for_strategy.py
run_step "regime-gated strategy comparison"  $PY src/agentic/backtest_regime_gated.py
run_step "build chart signals (multi-timeframe technical)" $PY src/agentic/build_chart_signals.py
run_step "compute risk envelope (30% min / 2x max / -30% floor)" $PY src/agentic/risk_envelope.py
run_step "devils advocate audit (10-vector integrity check)" $PY src/agentic/devils_advocate.py
run_step "generate today's trade plan (consolidated action)" $PY src/agentic/generate_trade_plan.py

log "=========================================="
log "DAILY PIPELINE done"
log "Brief at: reports/daily_pro_brief_*.md"
log "Log:      $LOG"
log "=========================================="
