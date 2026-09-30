"""Remove NSE holiday-copy rows from the panel and re-date special sessions (2026-09-27).

NSE's archive serves the previous session's bhavcopy under a holiday's file name; the ingest dated rows by
the folder, so 76 holiday dates (2020-2026) carry a full copy of the prior session (identical OHLC, volume,
delivery) and 3 special Saturday sessions (Muhurat 2020-11-14, 2024-01-20, 2024-05-18) sit on the following
Monday. src/ingest/nse/normalize._reconcile_file_date now prevents both at ingest; this repairs rows already
in the panel, using each row's own file date (trade_date_source):
  * trade_date_source != trade_date and the symbol has a row on the file date -> drop (copy)
  * trade_date_source != trade_date and no row on the file date -> re-date to the file date
Then recomputes rolling features for every affected symbol. Back up the panel first (the caller does).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT))
from src.features.indicators import add_daily_price_features  # noqa: E402

PANEL = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
ROLL_PREFIX = ("sma_", "ema_", "rsi_", "return_", "volume_vs_", "traded_value_vs_", "avg_traded_value_", "avg_vol_",
               "vol_max_", "volume_high_", "avg_delivery", "delivery_qty_vs", "delivery_pct_max", "delivery_pct_vs",
               "delivery_above", "delivery_pct_high")


def main() -> None:
    p = pd.read_parquet(PANEL)
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    n0 = len(p)
    src = pd.to_datetime(p["trade_date_source"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
    mis = src.notna() & (src != p["trade_date"])
    keys = set(zip(p["symbol"].to_numpy(), p["trade_date"].to_numpy()))
    has_real = pd.Series([(s, d) in keys for s, d in zip(p.loc[mis, "symbol"].to_numpy(), src[mis].to_numpy())], index=p.index[mis])
    drop_idx = has_real[has_real].index
    redate_idx = has_real[~has_real].index
    print(f"mismatched rows {int(mis.sum()):,}: copies to drop {len(drop_idx):,} · special-session rows to re-date {len(redate_idx):,}", flush=True)
    print("re-dated sessions:", pd.DataFrame({"from": p.loc[redate_idx, "trade_date"], "to": src[redate_idx]}).drop_duplicates().to_string(index=False), flush=True)
    affected = set(p.loc[mis, "symbol"])
    p.loc[redate_idx, "trade_date"] = src[redate_idx]
    p = p.drop(index=drop_idx)
    assert not p.duplicated(["symbol", "trade_date"]).any(), "duplicate keys after repair"
    part = p[p["symbol"].isin(affected)].copy()
    rest = p[~p["symbol"].isin(affected)]
    roll = [c for c in part.columns if c.startswith(ROLL_PREFIX)]
    print(f"recomputing rolling features for {len(affected):,} symbols ({len(part):,} rows)", flush=True)
    part = add_daily_price_features(part.drop(columns=roll)).reindex(columns=p.columns)
    out = pd.concat([rest, part], ignore_index=True).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    out.to_parquet(PANEL, index=False)
    print(f"rows {n0:,} -> {len(out):,}; wrote {PANEL}", flush=True)
    chk = out[(out["symbol"] == "TATASTEEL") & out["trade_date"].between("2020-02-18", "2020-02-25")][["trade_date", "close", "return_1d"]]
    print("TATASTEEL around 2020-02-21 (holiday) :\n" + chk.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
