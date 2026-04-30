"""Consolidate ALL macro/aggregate features into a single daily panel.

Combines:
  • macro_timeseries.parquet     (FX: USDINR, EURINR, GBPINR, JPYINR)
  • commodity_prices.parquet     (brent, gold, copper, etc.)
  • global_rates.parquet         (US 10y, DXY, VIX, SPX, EEM, etc.)
  • market_breadth_panel.parquet (NSE breadth, dispersion, new highs/lows)
  • industry_panel.parquet       (sector rotation aggregates — wide-pivoted)
  • amfi_mf_aum.parquet          (MF AUM monthly)
  • global_macro_sentiment.parquet (macro news tone)

Builds derived features (the actual signals):
  • brent_5d_pct, brent_20d_pct, brent_60d_pct
  • gold_brent_ratio, gold_5d_pct
  • copper_5d_pct, copper_60d_pct (Dr. Copper)
  • dxy_5d_pct (INR pressure)
  • us_10y_5d_chg, us_10y_60d_chg
  • us_vix_z_60d
  • spx_5d_pct, eem_relative_to_spx
  • inr_5d_pct, inr_20d_pct
  • breadth_50, breadth_50_5d_chg, breadth_50_20d_chg
  • new_high_low_diff
  • dispersion_z_60d
  • smid_lcap_breadth_diff (rotation: small/mid vs large)
  • mf_equity_aum_yoy (institutional flow)
  • macro_sentiment_avg (avg across all topics)

Output: data/derived/macro_panel.parquet
  one row per trade_date (forward-filled where data is sparse)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
DERIVED = ROOT / "data/derived"
OUT = DERIVED / "macro_panel.parquet"


def safe_read(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df
    except Exception as e:
        print(f"  read fail {p.name}: {e}")
        return pd.DataFrame()


def main() -> None:
    print("== build_macro_panel ==")

    # 1. trade-date spine from prices
    spine = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                             columns=["trade_date"]).drop_duplicates()
    spine["trade_date"] = pd.to_datetime(spine["trade_date"])
    spine = spine.sort_values("trade_date").drop_duplicates()
    spine = spine[spine["trade_date"] >= "2018-01-01"]
    print(f"  spine: {len(spine):,} trade days")

    panel = spine.copy()

    # 2. FX panel
    fx = safe_read(DERIVED / "macro_timeseries.parquet")
    if not fx.empty:
        panel = panel.merge(fx, on="trade_date", how="left")
        print(f"  + FX: {len(fx)} rows ({list(c for c in fx.columns if c != 'trade_date')})")

    # 3. commodities
    com = safe_read(DERIVED / "commodity_prices.parquet")
    if not com.empty:
        panel = panel.merge(com, on="trade_date", how="left")
        print(f"  + commodities: {len(com)} rows ({len(com.columns)-1} commodities)")

    # 4. global rates
    glb = safe_read(DERIVED / "global_rates.parquet")
    if not glb.empty:
        panel = panel.merge(glb, on="trade_date", how="left")
        print(f"  + global_rates: {len(glb)} rows ({len(glb.columns)-1} indicators)")

    # 5. NSE breadth
    br = safe_read(DERIVED / "market_breadth_panel.parquet")
    if not br.empty:
        panel = panel.merge(br, on="trade_date", how="left", suffixes=("", "_dup"))
        print(f"  + breadth: {len(br)} rows ({len(br.columns)-1} metrics)")

    # 6. industry pivot — only wide-cast a few key columns
    ind = safe_read(DERIVED / "industry_panel.parquet")
    if not ind.empty and "sector" in ind.columns:
        for metric in ["rs_60d", "sector_breadth_50", "sector_dispersion_20d"]:
            if metric in ind.columns:
                wide = ind.pivot_table(index="trade_date", columns="sector", values=metric, aggfunc="first")
                wide.columns = [f"{metric}__{s}" for s in wide.columns]
                wide = wide.reset_index()
                panel = panel.merge(wide, on="trade_date", how="left")
        print(f"  + industry: {ind['sector'].nunique()} sectors × 3 metrics pivoted")

    # 7. MF AUM (monthly → forward-fill to daily)
    mf = safe_read(DERIVED / "amfi_mf_aum.parquet")
    if not mf.empty and "month_end" in mf.columns:
        mf["trade_date"] = pd.to_datetime(mf["month_end"])
        mf_keep = ["trade_date", "equity_aum_cr", "total_aum_cr", "sip_inflow_cr",
                   "equity_aum_yoy_pct", "total_aum_yoy_pct", "sip_inflow_yoy_pct",
                   "equity_aum_mom_pct"]
        mf_keep = [c for c in mf_keep if c in mf.columns]
        panel = panel.merge(mf[mf_keep], on="trade_date", how="left")
        # forward-fill MF columns (they're monthly)
        for c in mf_keep:
            if c == "trade_date": continue
            panel[c] = panel[c].ffill()
        print(f"  + MF AUM: {len(mf)} months  (ffill'd to daily)")

    # 8. macro sentiment (latest snapshot — pivot topic → wide)
    snt = safe_read(DERIVED / "global_macro_sentiment.parquet")
    if not snt.empty and "topic" in snt.columns:
        snt_latest = snt.sort_values("as_of").groupby("topic").tail(1)
        snt_pivot = snt_latest.pivot_table(index=None, columns="topic",
                                             values="sentiment_7d", aggfunc="first")
        # broadcast scalar latest sentiment to all rows (it's a 'today snapshot')
        for col in snt_pivot.columns:
            panel[f"macro_sent__{col}"] = float(snt_pivot[col].iloc[0])
        # avg macro sentiment
        sent_cols = [c for c in panel.columns if c.startswith("macro_sent__")]
        if sent_cols:
            panel["macro_sent_avg"] = panel[sent_cols].mean(axis=1)
        print(f"  + macro sentiment: {len(snt_latest)} topics broadcast")

    # 9. derived features
    panel = panel.sort_values("trade_date")
    if "brent" in panel.columns:
        panel["brent_5d_pct"]  = panel["brent"].pct_change(5)
        panel["brent_20d_pct"] = panel["brent"].pct_change(20)
        panel["brent_60d_pct"] = panel["brent"].pct_change(60)
    if "gold" in panel.columns:
        panel["gold_5d_pct"]   = panel["gold"].pct_change(5)
        panel["gold_60d_pct"]  = panel["gold"].pct_change(60)
        if "brent" in panel.columns:
            panel["gold_brent_ratio"] = panel["gold"] / panel["brent"].replace(0, np.nan)
    if "copper" in panel.columns:
        panel["copper_5d_pct"] = panel["copper"].pct_change(5)
        panel["copper_60d_pct"]= panel["copper"].pct_change(60)
    if "dxy" in panel.columns:
        panel["dxy_5d_pct"]    = panel["dxy"].pct_change(5)
        panel["dxy_20d_pct"]   = panel["dxy"].pct_change(20)
    if "us_10y" in panel.columns:
        panel["us_10y_5d_chg"]  = panel["us_10y"].diff(5)
        panel["us_10y_60d_chg"] = panel["us_10y"].diff(60)
    if "us_vix" in panel.columns:
        panel["us_vix_z_60d"] = (panel["us_vix"] - panel["us_vix"].rolling(60).mean()) \
                                / panel["us_vix"].rolling(60).std()
    if "spx" in panel.columns:
        panel["spx_5d_pct"]   = panel["spx"].pct_change(5)
        panel["spx_60d_pct"]  = panel["spx"].pct_change(60)
        if "eem" in panel.columns:
            panel["eem_5d_pct"] = panel["eem"].pct_change(5)
            panel["eem_rs_to_spx_5d"] = panel["eem_5d_pct"] - panel["spx_5d_pct"]
    if "usdinr" in panel.columns:
        panel["inr_5d_pct"]  = panel["usdinr"].pct_change(5)
        panel["inr_20d_pct"] = panel["usdinr"].pct_change(20)

    # crude × INR (importer pain index)
    if "brent" in panel.columns and "usdinr" in panel.columns:
        panel["brent_inr"] = panel["brent"] * panel["usdinr"]
        panel["brent_inr_5d_pct"] = panel["brent_inr"].pct_change(5)

    # forward-fill commodity / global rate columns onto Indian holidays (markets closed locally but global open)
    ffill_cols = [c for c in panel.columns
                  if c.startswith(("brent","gold","silver","copper","aluminum","zinc","nickel",
                                     "wheat","corn","cotton","sugar","natgas",
                                     "us_10y","us_3m","dxy","us_vix","spx","eem","usdjpy","btc",
                                     "ftse","n225","hsi","gld","wti",
                                     "usdinr","eurinr","gbpinr","jpyinr"))]
    for c in ffill_cols:
        panel[c] = panel[c].ffill(limit=5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    n_cols = len(panel.columns) - 1
    print(f"\nwrote {OUT}: {len(panel):,} days × {n_cols} features")
    last = panel.iloc[-1]
    snap_cols = ["trade_date","usdinr","brent","gold","copper","us_10y","dxy","us_vix",
                 "breadth_50","new_52w_highs","new_52w_lows","equity_aum_yoy_pct"]
    snap = {c: last.get(c) for c in snap_cols if c in panel.columns}
    print("\n  Latest snapshot:")
    for k, v in snap.items():
        if pd.notna(v):
            if isinstance(v, (pd.Timestamp,)):
                print(f"    {k:<24} = {v:%Y-%m-%d}")
            elif isinstance(v, (int, np.integer)):
                print(f"    {k:<24} = {v}")
            else:
                print(f"    {k:<24} = {float(v):.3f}")


if __name__ == "__main__":
    main()
