"""Backfill BE/BZ-series gap rows (2025-01-01 onward) into stock_daily_facts.

One-shot repair run 2026-08-28 (see reports/be_series_gap_repair_20260828.md);
kept for reproducibility. Idempotent: re-running inserts nothing once the
panel is complete. Back up the parquet before running.

The old EQ-only series filter in build_daily_facts dropped every session a
symbol spent in the trade-to-trade surveillance series (BE/BZ), leaving
multi-week holes (DEEDEV: 60 sessions 2026-05-19..2026-08-10). This script:
  1. rebuilds normalized rows from raw bhavcopy partitions >= 2025-01-01 using
     the FIXED series filter (EQ|BE|BZ, T2T delivery imputation),
  2. inserts only (symbol, trade_date) pairs missing from the parquet, for
     symbols already in the panel (no new symbols introduced),
  3. CA-adjusts the inserted rows with the production adjuster,
  4. recomputes all rolling/derived features for affected symbols over their
     full history,
  5. writes the parquet (after a .bak copy made by the caller).

Scope: sessions >= START (argv[1], default 2025-01-01). 2026-09-27 run: START=2015-01-01.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.transform.build_daily_facts import EQUITY_SERIES, T2T_SERIES  # noqa: E402
from src.transform.corporate_actions import apply_split_bonus_adjustments  # noqa: E402
from src.features.indicators import add_daily_price_features  # noqa: E402
from src.ingest.nse.fetch_bhavcopy import build_nse_bhavcopy_url, build_nse_delivery_url  # noqa: E402
from src.ingest.nse.normalize import normalize_trade_date_directory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
RAW = ROOT / "data/raw/nse_full_history_official"
CA_PATH = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"
# 2026-09-27: generalised to any start (audit root cause 1: pre-2025 BE/BZ sessions still missing).
# Usage: repair_be_series_gaps.py [YYYY-MM-DD]  (default 2025-01-01 reproduces the 2026-08-28 run)
START = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2025-01-01")

FEATURE_COLS = [
    "sma_20", "sma_50", "sma_100", "sma_200", "ema_20", "ema_50", "ema_200",
    "rsi_14_daily", "return_1d", "return_20d", "avg_vol_20d", "avg_vol_60d",
    "vol_max_63d", "volume_vs_20d", "volume_vs_60d", "volume_high_63d_flag",
    "avg_traded_value_20d", "avg_traded_value_60d", "traded_value_vs_20d",
    "traded_value_vs_60d", "avg_delivery_qty_20d", "delivery_qty_vs_20d",
    "avg_delivery_pct_5d", "avg_delivery_pct_20d", "delivery_pct_max_63d",
    "delivery_pct_vs_5d", "delivery_pct_vs_20d", "delivery_above_5d_avg_flag",
    "delivery_pct_high_63d_flag", "rsi_14_weekly", "rsi_14_monthly",
]

print("loading existing parquet ...", flush=True)
old = pd.read_parquet(PARQUET)
old["trade_date"] = pd.to_datetime(old["trade_date"])
panel_symbols = set(old["symbol"].unique())
existing_keys = set(zip(old["symbol"].to_numpy(), old["trade_date"].to_numpy()))
print(f"existing: {len(old):,} rows, {len(panel_symbols):,} symbols", flush=True)

# --- 1. normalize raw partitions 2025+ with the fixed filter --------------
parts = []
part_dirs = [p for p in sorted(RAW.glob("trade_date=*"))
             if pd.Timestamp(p.name.split("=", 1)[1]) >= START]
print(f"normalizing {len(part_dirs)} raw partitions ...", flush=True)
for i, trade_dir in enumerate(part_dirs):
    td = pd.Timestamp(trade_dir.name.split("=", 1)[1]).date()
    df = normalize_trade_date_directory(
        trade_dir, td,
        market_source_url=build_nse_bhavcopy_url(td),
        delivery_source_url=build_nse_delivery_url(td),
    )
    df = df[df["symbol"].notna()]
    df = df[df["symbol"].isin(panel_symbols)]
    ser = df["series"].fillna("").str.upper()
    df = df[ser.isin(EQUITY_SERIES) | df["series"].isna()].copy()
    parts.append(df)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(part_dirs)}", flush=True)

fresh = pd.concat(parts, ignore_index=True)
fresh["trade_date"] = pd.to_datetime(fresh["trade_date"])

# T2T delivery imputation (same rule as the fixed build_daily_facts)
ser = fresh["series"].fillna("").str.upper()
t2t = ser.isin(T2T_SERIES)
fresh.loc[t2t & fresh["delivery_pct"].isna(), "delivery_pct"] = 1.0
qmask = t2t & fresh["deliverable_qty"].isna()
fresh.loc[qmask, "deliverable_qty"] = fresh.loc[qmask, "total_traded_qty"]

# --- 2. keep only missing (symbol, trade_date) pairs ----------------------
key_mask = [
    (s, d) not in existing_keys
    for s, d in zip(fresh["symbol"].to_numpy(), fresh["trade_date"].to_numpy())
]
inserts = fresh[np.array(key_mask)].copy()
del fresh
print(f"missing rows to insert: {len(inserts):,} across "
      f"{inserts['symbol'].nunique():,} symbols", flush=True)
print(inserts["series"].str.strip().value_counts().to_string(), flush=True)

# --- 3. CA-adjust inserted rows -------------------------------------------
ca = pd.read_parquet(CA_PATH) if CA_PATH.exists() else pd.DataFrame()
inserts = apply_split_bonus_adjustments(inserts, ca)
n_adjusted = int((inserts["price_adjustment_factor_to_present"] != 1.0).sum())
print(f"inserted rows carrying a CA adjustment: {n_adjusted:,}", flush=True)

# --- 4. splice + recompute features for affected symbols ------------------
inserts = inserts.reindex(columns=old.columns)  # align; feature cols -> NA
affected = sorted(set(inserts["symbol"].unique()))
combined = pd.concat([old, inserts], ignore_index=True)
del old, inserts
dup = combined.duplicated(["symbol", "trade_date"]).sum()
assert dup == 0, f"duplicate keys after splice: {dup}"

aff_mask = combined["symbol"].isin(affected)
aff = combined[aff_mask].copy()
rest = combined[~aff_mask]
print(f"recomputing features for {len(affected)} symbols ({len(aff):,} rows) ...",
      flush=True)
drop_cols = [c for c in FEATURE_COLS if c in aff.columns]
aff = add_daily_price_features(aff.drop(columns=drop_cols))
missing_cols = set(rest.columns) - set(aff.columns)
assert not missing_cols, f"feature recompute lost columns: {missing_cols}"
aff = aff[rest.columns]

out = pd.concat([rest, aff], ignore_index=True)
out = out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
print(f"final rows: {len(out):,}", flush=True)

out.to_parquet(PARQUET, index=False)
print(f"wrote {PARQUET}", flush=True)

# --- 5. quick DEEDEV verification -----------------------------------------
d = out[out["symbol"] == "DEEDEV"]
w = d[(d["trade_date"] >= "2026-05-15") & (d["trade_date"] <= "2026-08-12")]
print(f"DEEDEV rows 2026-05-15..2026-08-12: {len(w)}")
print(w[["trade_date", "series", "close", "return_1d", "delivery_pct",
         "volume_vs_20d"]].head(6).to_string(index=False))
print(w[["trade_date", "series", "close", "return_1d", "delivery_pct",
         "volume_vs_20d"]].tail(3).to_string(index=False))
