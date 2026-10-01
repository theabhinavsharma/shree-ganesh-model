#!/usr/bin/env bash
# EXP-2026-09-30-v3-pnl-full (+ RERUN-enriched of EXP-2026-09-30-v3-pnl), chained after the old-format P&L fetch.
# Nothing here touches the live model, the live rows or the live P&L table: everything goes to logs/leader_sleeve/v3_pnl/.
set -u
cd /Users/abhinavs./Code/Zoom
PY=/usr/bin/python3; V=logs/leader_sleeve/v3_pnl; P="sales_yoy,eps_yoy,pe,pe_ind,profitable,loss_to_profit,days_since_results"
LOG=logs/pnl_old/chain_$(date +%Y%m%d_%H%M).log; exec > "$LOG" 2>&1
fail() { echo "FAILED: $1"; $PY -c "import sys; sys.path.insert(0,'src/agentic'); import notify; notify.send(notify.tidy(['❌ P&L A/B stopped · $(date +%b\ %d)', '', 'Step: $1', 'Log: $LOG']), 'research')"; exit 1; }

# 1. wait for fetch + normalize (fetch_pnl_old_format.py prints "normalized ..." at the end)
until grep -qh "^normalized" logs/pnl_old/fetch_*.log 2>/dev/null; do
  pgrep -f "fetch_pnl_old_format.py" > /dev/null || fail "old-format fetch ended without normalizing"
  sleep 60
done
grep -h "^normalized" logs/pnl_old/fetch_*.log

# 2. QC (2026-10-01): same-quarter overlaps are restatements (old row first filed, NEW row ~13 months later), so the check
#    is the old/new ratio median plus sales continuity across the format switch; spot check against a page read by eye
$PY - <<'EOF' || fail "QC"
import pandas as pd
O = pd.read_parquet("data/derived/pnl_quarterly_old.parquet"); L = pd.read_parquet("data/derived/pnl_quarterly.parquet")
L["quarter_end"] = pd.to_datetime(L["quarter_end"]); L["filing_dt"] = pd.to_datetime(L["filing_dt"]); L = L[L["source"] == "detail_api"]
m = O.merge(L, on=["symbol", "quarter_end", "basis"], suffixes=("_old", "_new"))
print(f"overlap {len(m)} quarters · old filed first {(m.filing_dt_old < m.filing_dt_new).mean():.0%} · median gap {(m.filing_dt_new - m.filing_dt_old).dt.days.median():.0f} days · "
      f"median old/new: sales {(m.net_sales_old / m.net_sales_new).median():.3f}, profit {(m.pat_old / m.pat_new).median():.3f}")
lo = O.sort_values("quarter_end").groupby(["symbol", "basis"]).tail(1); fn = L.sort_values("quarter_end").groupby(["symbol", "basis"]).head(1)
j = lo.merge(fn, on=["symbol", "basis"], suffixes=("_o", "_n")); j = j[(j.quarter_end_n - j.quarter_end_o).dt.days.between(80, 100) & (j.net_sales_o > 0)]
r = j.net_sales_n / j.net_sales_o
print(f"sales continuity across the switch: {len(j)} companies · median {r.median():.2f} · 20x jumps {int(((r > 20) | (r < .05)).sum())}")
s = O[(O["symbol"] == "20MICRONS") & (O["quarter_end"] == "2016-12-31") & (O["basis"] == "sa")]
print("spot check 20MICRONS 2016-12-31 sa (page: net sales 8498.69, net profit 244.96, EPS 0.69 · Rs lakh):", s[["net_sales", "pat", "eps_basic"]].to_dict("records"))
EOF

# 3. enriched P&L table (live table untouched)
$PY src/agentic/fetch_pnl_old_format.py --merge || fail "merge"

# 4. rows on the enriched P&L, in the test folder
mkdir -p $V/full/rows
$PY src/agentic/anatomy_1p5x.py --pnl data/derived/pnl_quarterly_enriched.parquet --outdir $V/full/rows > $V/full/rows/run.log 2>&1 || fail "rows rebuild"

# 5. models: full history (V3F, V3PF) and the 2019-trained rerun (V3T, V3P)
R=$V/full/rows/rows.parquet; Y="2020,2021,2022,2023,2024,2025,2026"
mkdir -p $V/full/model_F $V/full/model_PF $V/rerun_enriched/model_T $V/rerun_enriched/model_P
$PY src/agentic/model_bakeoff_1p5x.py --rows $R --out-dir $V/full/model_F --entry close --drop-features $P > $V/full/model_F/run.log 2>&1 || fail "model_F"
$PY src/agentic/model_bakeoff_1p5x.py --rows $R --out-dir $V/full/model_PF --entry close --add-features $P > $V/full/model_PF/run.log 2>&1 || fail "model_PF"
$PY src/agentic/model_bakeoff_1p5x.py --rows $R --out-dir $V/rerun_enriched/model_T --entry close --train-from 2019-01-01 --years $Y --drop-features $P > $V/rerun_enriched/model_T/run.log 2>&1 || fail "model_T rerun"
$PY src/agentic/model_bakeoff_1p5x.py --rows $R --out-dir $V/rerun_enriched/model_P --entry close --train-from 2019-01-01 --years $Y --add-features $P > $V/rerun_enriched/model_P/run.log 2>&1 || fail "model_P rerun"
grep -h "features [0-9]" $V/full/model_F/run.log $V/full/model_PF/run.log

# 6. the registered tests
SGM_PNL_FULL=1 $PY src/agentic/test_v3_pnl.py || fail "full test"
SGM_RERUN_TAG=enriched $PY src/agentic/test_v3_pnl.py || fail "rerun test"

# 7. phone
$PY - <<'EOF'
import json, sys; sys.path.insert(0, "src/agentic"); import notify
L = [json.loads(l) for l in open("logs/experiments.jsonl") if l.startswith("{")]
f = [e for e in L if e.get("id") == "EXP-2026-09-30-v3-pnl-full-RESULT"][-1]
r = [e for e in L if e.get("id") == "EXP-2026-09-30-v3-pnl-RERUN-enriched"][-1]
a = f["arms"]; line = lambda k: f"{k}: {a[k][0]}%/yr · 2019-22 {a[k][1]} · 2023+ {a[k][2]} · worst {a[k][3]}%"
ok = f["verdict"] != "no arm passes"
notify.send(notify.tidy([f"🧪 V3 + P&L A/B · {'✅ PASS' if ok else '❌ no pass'} · Sep 30 data", "",
    line("V3"), line("V3PF"), f"V3PF beats V3 in {f['phases']['V3P_vs_V3']} start weeks · P&L alone {f['phases']['V3P_vs_V3T']}", "",
    f"Rerun (2019-trained, enriched P&L): {'pass' if r['verdict'] != 'no arm passes' else 'no pass'}",
    "Live model unchanged" + (" · your go needed to adopt" if ok else "")]), "research")
EOF
echo "CHAIN DONE"
