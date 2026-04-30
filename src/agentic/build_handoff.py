"""Regenerate HANDOFF.md with current state snapshot.

Reads the latest reports + parquets and refreshes the §3 "today's known state"
section so a future Claude reads accurate info.

Run after every major pipeline run.
"""
from __future__ import annotations
import json
import re
from datetime import date, datetime
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
HANDOFF = ROOT / "HANDOFF.md"
HC_PRED = ROOT / "data/derived/high_conviction_predictions.parquet"
DYN_GATED = ROOT / "data/derived/dynamic_gated_backtest.parquet"
DA_AUDIT = ROOT / "data/derived/devils_advocate_audit.parquet"


def get_today_state() -> dict:
    state = {"date": str(date.today())}

    # high conviction
    if HC_PRED.exists():
        try:
            hc = pd.read_parquet(HC_PRED)
            cols = [c for c in hc.columns if c.startswith("score_") and c.endswith("_cal")]
            if cols:
                hc["best"] = hc[cols].max(axis=1)
                state["top_hc_score"] = round(float(hc["best"].max()), 3)
                state["n_above_095"] = int((hc["best"] >= 0.95).sum())
        except Exception:
            pass

    # dynamic gated
    if DYN_GATED.exists():
        try:
            dg = pd.read_parquet(DYN_GATED)
            state["dyn_gated_median_ann"] = round(float(dg["blended_ann_roi"].median()) * 100, 1)
            state["dyn_gated_min_ann"] = round(float(dg["blended_ann_roi"].min()) * 100, 1)
            state["dyn_gated_max_ann"] = round(float(dg["blended_ann_roi"].max()) * 100, 1)
        except Exception:
            pass

    # devil's advocate
    if DA_AUDIT.exists():
        try:
            da = pd.read_parquet(DA_AUDIT)
            state["da_critical"] = int((da["severity"] == "CRITICAL").sum())
            state["da_high"] = int((da["severity"] == "HIGH").sum())
        except Exception:
            pass

    return state


def main() -> None:
    state = get_today_state()
    text = HANDOFF.read_text()

    # update §3 block
    new_section = f"""## 3. Today's known state (snapshot at handoff time)

- Latest snapshot date: **{state.get('date', '—')}**
- Top high-conviction score (5%/7d, 10%/15d, 20%/30d max): **{state.get('top_hc_score', '—')}**
- Names ≥ 0.95 calibrated bar today: **{state.get('n_above_095', 0)}**
- Dynamic-gated backtest 9-year median ann ROI: **{state.get('dyn_gated_median_ann', '—')}%** (range {state.get('dyn_gated_min_ann','—')}% to {state.get('dyn_gated_max_ann','—')}%)
- Devil's advocate audit: **{state.get('da_critical', 0)} CRITICAL, {state.get('da_high', 0)} HIGH** issues open"""

    text = re.sub(
        r"## 3\. Today's known state.*?(?=\n## 4\.)",
        new_section + "\n\n",
        text, flags=re.DOTALL,
    )

    HANDOFF.write_text(text)
    print(f"updated {HANDOFF}: {state}")


if __name__ == "__main__":
    main()
