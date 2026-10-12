"""Insider / promoter disclosures 2026-03-01 .. 2026-05-02: close the gap in data/derived/pit_history.parquet (2026-10-12).

Plain English: the insider-trade history is rebuilt every day by fetch_pit_gg.merge_history from two pieces: the old NSE system's
history (pit_history_pre_gg.parquet, ends 2026-02-28) and NSE's new filing system (pit_gg.parquet, starts 2026-05-03). March and
April 2026 were in neither, so for those months every stock looked like it had no insider trades at all (found while setting up
EXP-2026-10-12-promoter-in-model). The old system's live store (refresh_announcements.py, same NSE api/corporates-pit, fetched
2026-05-06) holds those weeks. This script copies them, unchanged, into data/raw/pit_gap_2026/, which merge_history now adds.
Idempotent: re-running rewrites the same file.
Source (official): https://www.nseindia.com/api/corporates-pit?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "tmp/from_scratch_7d_run/alt/insider_trading_pit.parquet"
PRE = ROOT / "data/derived/pit_history_pre_gg.parquet"
GG = ROOT / "data/derived/pit_gg.parquet"
OUT = ROOT / "data/raw/pit_gap_2026/pit_corporates_pit_gap.parquet"


def _d(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d-%b-%Y %H:%M", errors="coerce")


def main() -> None:
    lo = _d(pd.read_parquet(PRE, columns=["date"])["date"]).max()
    hi = _d(pd.read_parquet(GG, columns=["date"])["date"]).min()
    L = pd.read_parquet(LIVE)
    d = _d(L["date"])
    G = L[(d > lo) & (d < hi)].drop(columns=["acq_from_parsed", "delta_pct", "intim_dt_parsed"], errors="ignore").copy()
    for c in G.columns:
        if G[c].dtype == object:
            G[c] = G[c].astype("string")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    G.to_parquet(OUT, index=False)
    by_month = _d(G["date"]).dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
    (OUT.with_name(OUT.name + ".manifest.json")).write_text(json.dumps(dict(
        dataset=OUT.name, producer="src/agentic/repair_pit_gap_2026.py", source="NSE api/corporates-pit (old insider-trading system), "
        "copied unchanged from the live store tmp/from_scratch_7d_run/alt/insider_trading_pit.parquet (fetched 2026-05-06)",
        window=dict(after=str(lo), before=str(hi), why="after the last row of pit_history_pre_gg, before the first row of pit_gg"),
        rows=len(G), rows_by_month=by_month, columns="pit_history columns (date = NSE broadcast time, '%d-%b-%Y %H:%M')",
        used_by="fetch_pit_gg.merge_history", updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUT.parent / "README.md").write_text("# Insider disclosures, Mar 1 - May 2 2026\n\nThe weeks between the old NSE insider system's history "
                                          "and the new system's first filing. Copied unchanged from the old system's live store; added to "
                                          "data/derived/pit_history.parquet by fetch_pit_gg.merge_history. See the manifest.\n")
    print(f"gap rows {len(G)} ({lo} .. {hi}) by month {by_month} -> {OUT}")


if __name__ == "__main__":
    main()
