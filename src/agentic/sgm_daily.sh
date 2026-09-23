#!/usr/bin/env bash
# SGM DAILY — post-close, every trading day (cron; launchd is TCC-blocked on this Mac).
#   1. daily_data_layer.sh    every feed, CA before prices, panel rebuilt, loud status
#   2. render_basket_report   reports/daily_actions_<session>.md for the live basket
#      (BUY / HOLD / SELL PART / SELL / SELL? / SKIP / CLOSED / STANDBY) + basket report
#   2b. leader sleeve        score_leader_sleeve + render_leader_report (PAPER, non-fatal)
#   3. freshness dashboard    reports/freshness_status.md (never fails)
# Engines and baskets stay on run_weekly_pipeline.sh (Friday) — RL protocol, no churn.
# Log: logs/sgm_daily/<TS>.log     Status: logs/daily_data_layer_status.json
set -u
cd /Users/abhinavs./Documents/Zoom
mkdir -p logs/sgm_daily; TS=$(date +%Y%m%d_%H%M%S); LOG=logs/sgm_daily/$TS.log
exec > >(tee -a "$LOG") 2>&1
echo "═══ SGM DAILY $TS ═══"
bash src/agentic/daily_data_layer.sh; RC=$?
/usr/bin/python3 src/agentic/eval_panel_coverage.py --sessions 5 > logs/sgm_daily/${TS}_eval.log 2>&1 \
  && echo "✅ panel coverage eval" || { echo "❌ PANEL COVERAGE EVAL FAILED — see logs/sgm_daily/${TS}_eval.log"; tail -6 logs/sgm_daily/${TS}_eval.log; RC=1; }
/usr/bin/python3 src/agentic/render_basket_report.py > logs/sgm_daily/${TS}_reports.log 2>&1 \
  && echo "✅ daily actions: $(ls -t reports/daily_actions_*.md | head -1)" \
  || echo "❌ report render failed — see logs/sgm_daily/${TS}_reports.log"
# Leader sleeve (PAPER): score every immutable screen day by day, then render its report
/usr/bin/python3 src/agentic/score_leader_sleeve.py > logs/sgm_daily/${TS}_leader.log 2>&1 \
  && /usr/bin/python3 src/agentic/render_leader_report.py >> logs/sgm_daily/${TS}_leader.log 2>&1 \
  && echo "✅ leader sleeve: $(ls -t reports/leader_sleeve_*.md | head -1)" \
  || echo "⚠ leader sleeve score/render failed (paper, non-fatal) — see logs/sgm_daily/${TS}_leader.log"
/usr/bin/python3 src/agentic/emit_freshness_status.py > /dev/null 2>&1 && echo "✅ freshness dashboard"
echo "═══ DONE data_layer_rc=$RC ═══"
exit $RC
