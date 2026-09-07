#!/usr/bin/env bash
# Sep-8 basket, USER-AUTHORIZED max-memory mode (this run only): engines in
# 3+2 parallel batches (~50-60GB peak; 5-wide = the 90-100GB freeze config).
set -u
cd /Users/abhinavs./Documents/Zoom
while pgrep -f daily_data_layer.sh >/dev/null; do sleep 30; done
echo "=== layer done $(date) ==="
for ns in "cs:compare_short_horizons" "hc:find_high_conviction" "mb:find_multibagger_today"; do
  name="${ns%%:*}"; script="${ns#*:}"
  /usr/bin/python3 "src/agentic/${script}.py" > "logs/weekly_pipeline/20260908_e_${name}.log" 2>&1 &
done
wait
echo "=== batch1 done $(date) ==="
for ns in "mh:run_multi_horizon" "180d:find_180d_frontier_honest"; do
  name="${ns%%:*}"; script="${ns#*:}"
  /usr/bin/python3 "src/agentic/${script}.py" > "logs/weekly_pipeline/20260908_e_${name}.log" 2>&1 &
done
wait
echo "=== engines done $(date) ==="
/usr/bin/python3 src/agentic/generate_hybrid_basket.py > logs/weekly_pipeline/20260908_basket.log 2>&1
echo "=== BASKET EXIT: $? $(date) ==="
