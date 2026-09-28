#!/usr/bin/env bash
# 2026-09-27 post-audit reruns on the repaired panel (serial; watchdog kills python above 55 GB).
cd /Users/abhinavs./Documents/Zoom
L=logs/leader_sleeve/rerun_20260927
( while true; do kb=$(ps -axo rss,comm | awk '/[Pp]ython/{s+=$1} END{print s+0}'); if [ "$kb" -gt 57671680 ]; then echo "[$(date +%T)] WATCHDOG kill ${kb}KB" >> $L/chain.log; pkill -f "src/agentic/(anatomy|model_bakeoff|sim_)"; fi; sleep 20; done ) & WD=$!
run(){ n=$1; shift; echo "[$(date +%T)] start $n" >> $L/chain.log; nice -n 10 /usr/bin/python3 "$@" > $L/$n.log 2>&1; echo "[$(date +%T)] end $n rc=$?" >> $L/chain.log; }
run anatomy src/agentic/anatomy_1p5x.py
run bakeoff src/agentic/model_bakeoff_1p5x.py
run allin_close src/agentic/sim_allin_matrix.py --entry close
run allin_nextopen src/agentic/sim_allin_matrix.py --entry next_open
run ladder126_close src/agentic/sim_leader_portfolio_7x.py --hold 126 --entry close
run ladder63_close src/agentic/sim_leader_portfolio_7x.py --hold 63 --entry close
run ladder63_nextopen src/agentic/sim_leader_portfolio_7x.py --hold 63 --entry next_open
run cell_core src/agentic/sim_leader_cell_v2.py --map nse4 --universe core
run cell_mcap50 src/agentic/sim_leader_cell_v2.py --map nse4 --universe mcap50
run cell_mcap50_pit src/agentic/sim_leader_cell_v2.py --map nse4 --universe mcap50 --pit_only
run cell_core_nextopen src/agentic/sim_leader_cell_v2.py --map nse4 --universe core --entry next_open
kill $WD; echo "[$(date +%T)] RERUN CHAIN DONE" >> $L/chain.log
