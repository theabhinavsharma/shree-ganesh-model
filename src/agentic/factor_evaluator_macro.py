"""Macro-aware factor evaluator — for date-level (NOT cross-sectional) features.

Why this exists:
  factor_evaluator.py uses CROSS-SECTIONAL Spearman IC (rank stocks within day).
  Macro features (brent, dxy, breadth_50, etc.) are IDENTICAL for every stock on
  a given date. Cross-sectional IC is mathematically undefined → every macro
  feature came back INSUFFICIENT in the standard evaluator.

The right tests for macro / aggregate features:
  1. TIME-SERIES IC: corr(feature[t], market_fwd_ret_7d[t])
       Where market_fwd_ret_7d is the median 7d forward return across the
       liquid universe. Does the feature predict the MARKET, not individual
       stocks?

  2. REGIME-CONDITIONAL SPLIT: rank dates into top vs bottom tertile of the
       feature; compute mean market_fwd_ret_7d in each. The spread is the
       "regime alpha." A feature with no regime alpha is useless macro info.

  3. TIME-SERIES IR: spread.mean() / spread.std() × sqrt(252) — annualized
       regime-spread information ratio.

Verdict:
  KEEP        if |ts_ic| >= 0.05 AND |regime_spread| >= 0.005 (50 bps)
  WATCHLIST   if 0.025 <= |ts_ic| < 0.05
  DROP        otherwise

Output:
  reports/factor_evaluation_macro.md
  registry update: macro hypotheses get lift_ic + verdict written back
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
EXTRA = ROOT / "data/derived/extra_features.parquet"
REGISTRY = ROOT / "data/derived/factor_registry.json"
OUT_REPORT = ROOT / "reports/factor_evaluation_macro.md"

H = 7  # forward horizon

IC_KEEP = 0.05      # time-series IC threshold (much higher than cross-sec — fewer obs)
SPREAD_KEEP = 0.005 # 50 bp regime spread
WATCHLIST_IC = 0.025


def main() -> None:
    print("== factor_evaluator_macro (TIME-SERIES + REGIME-SPLIT) ==")

    # 1. build market forward 7d return (the target variable for macro signals)
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close",
                                            "avg_traded_value_20d", "series"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px[px["series"] == "EQ"]
    px = px[px["avg_traded_value_20d"] / 1e7 >= 1.0]   # liquid universe only

    px = px.sort_values(["symbol", "trade_date"])
    px["fwd_close_7"] = px.groupby("symbol")["close"].shift(-H)
    px["fwd_ret_7"] = px["fwd_close_7"] / px["close"] - 1
    px = px.dropna(subset=["fwd_ret_7"])

    # market forward return = MEDIAN across liquid universe per date
    market = px.groupby("trade_date").agg(
        market_fwd_ret_7=("fwd_ret_7", "median"),
        market_fwd_ret_7_p25=("fwd_ret_7", lambda s: s.quantile(0.25)),
        market_fwd_ret_7_p75=("fwd_ret_7", lambda s: s.quantile(0.75)),
        market_fwd_ret_7_mean=("fwd_ret_7", "mean"),
        n_stocks=("symbol", "count"),
    ).reset_index()
    print(f"  market panel: {len(market):,} dates  ({market['trade_date'].min():%Y-%m-%d} → {market['trade_date'].max():%Y-%m-%d})")

    # 2. load engineered features and pick MACRO ones (date-level: same value for all stocks)
    ex = pd.read_parquet(EXTRA)
    ex["trade_date"] = pd.to_datetime(ex["trade_date"])
    feat_cols = [c for c in ex.columns
                 if c not in ("symbol", "trade_date")
                 and (c.startswith(("macro_", "sec_"))
                      or c.startswith(("usdinr", "eurinr", "gbpinr", "jpyinr"))
                      or c.endswith(("_5d_chg", "_20d_chg")))
                 and pd.api.types.is_numeric_dtype(ex[c])]
    print(f"  evaluating {len(feat_cols)} macro/aggregate features\n")

    # collapse to date-level: median value per date (since macro is constant-per-date)
    # for sec_* features (per-symbol via sector lookup), use the cross-sectional median per date
    macro_panel = ex[["trade_date"] + feat_cols].groupby("trade_date").median().reset_index()
    macro_panel = macro_panel.merge(market, on="trade_date", how="inner")
    print(f"  joined macro × market panel: {len(macro_panel):,} dates")

    rows = []
    for f in feat_cols:
        sub = macro_panel[["trade_date", f, "market_fwd_ret_7"]].dropna()
        if len(sub) < 30:
            rows.append({"feature": f, "n_dates": len(sub), "ts_ic": None, "ts_ic_t": None,
                         "regime_spread": None, "regime_ir": None, "verdict": "INSUFFICIENT"})
            continue

        # ── A. time-series IC (Spearman, rolling subseries to suppress trend) ──
        # use de-trended feature: (feat - rolling 252-day mean) for stationarity
        sub = sub.sort_values("trade_date").reset_index(drop=True)
        sub["feat_detrend"] = sub[f] - sub[f].rolling(252, min_periods=60).mean()
        sub["mkt_detrend"]  = sub["market_fwd_ret_7"] - sub["market_fwd_ret_7"].rolling(252, min_periods=60).mean()
        clean = sub[["feat_detrend", "mkt_detrend"]].dropna()
        if len(clean) < 30:
            ts_ic = np.nan
            ts_ic_t = np.nan
        else:
            ts_ic = float(clean.corr(method="spearman").iloc[0, 1])
            # rough t-stat: r * sqrt(n-2) / sqrt(1-r^2)
            n = len(clean)
            ts_ic_t = float(ts_ic * np.sqrt(n - 2) / np.sqrt(max(1 - ts_ic ** 2, 1e-9))) if abs(ts_ic) < 0.99 else np.nan

        # ── B. regime split (top vs bottom tertile of feature) ──
        try:
            sub["regime"] = pd.qcut(sub[f].rank(method="first"), 3,
                                     labels=["bot", "mid", "top"], duplicates="drop")
            top_ret = sub.loc[sub["regime"] == "top", "market_fwd_ret_7"]
            bot_ret = sub.loc[sub["regime"] == "bot", "market_fwd_ret_7"]
            spread = float(top_ret.mean() - bot_ret.mean())
            spread_std = float(np.sqrt(top_ret.var() / max(len(top_ret), 1) + bot_ret.var() / max(len(bot_ret), 1)))
            spread_t = spread / spread_std if spread_std > 0 else np.nan
            # annualized IR: assume rotated weekly (52 turns)
            regime_ir = (spread / spread_std * np.sqrt(52)) if spread_std > 0 else np.nan
        except Exception:
            spread, spread_t, regime_ir = np.nan, np.nan, np.nan

        # verdict
        ic_ok = (not pd.isna(ts_ic)) and abs(ts_ic) >= IC_KEEP
        sp_ok = (not pd.isna(spread)) and abs(spread) >= SPREAD_KEEP
        ic_watch = (not pd.isna(ts_ic)) and abs(ts_ic) >= WATCHLIST_IC

        if ic_ok and sp_ok:
            verdict = "KEEP"
        elif ic_watch and sp_ok:
            verdict = "WATCHLIST"
        elif pd.isna(ts_ic) or pd.isna(spread):
            verdict = "INSUFFICIENT"
        else:
            verdict = "DROP"

        rows.append({
            "feature": f, "n_dates": len(sub),
            "ts_ic": round(ts_ic, 4) if not pd.isna(ts_ic) else None,
            "ts_ic_t": round(ts_ic_t, 2) if not pd.isna(ts_ic_t) else None,
            "regime_spread": round(spread, 5) if not pd.isna(spread) else None,
            "regime_spread_pct": round(spread * 100, 3) if not pd.isna(spread) else None,
            "regime_ir": round(regime_ir, 2) if not pd.isna(regime_ir) else None,
            "verdict": verdict,
        })

    res = pd.DataFrame(rows)
    res = res.sort_values(by=["verdict", "ts_ic"],
                            key=lambda c: c if c.name != "verdict" else c.map({"KEEP":0,"WATCHLIST":1,"INSUFFICIENT":3,"DROP":2}),
                            ascending=[True, False],
                            na_position="last")

    keep = (res["verdict"] == "KEEP").sum()
    watch = (res["verdict"] == "WATCHLIST").sum()
    drop = (res["verdict"] == "DROP").sum()
    insf = (res["verdict"] == "INSUFFICIENT").sum()
    print(f"  KEEP: {keep}   WATCHLIST: {watch}   DROP: {drop}   INSUFFICIENT: {insf}\n")

    print("=== TOP 25 BY |ts_ic| ===")
    top = res[res["ts_ic"].notna()].sort_values("ts_ic", key=lambda s: s.abs(), ascending=False).head(25)
    print(top[["feature", "n_dates", "ts_ic", "ts_ic_t", "regime_spread_pct", "regime_ir", "verdict"]].to_string(index=False))

    # write report
    md = [
        f"# Macro Factor Evaluation — {pd.Timestamp.utcnow():%Y-%m-%d}", "",
        "Time-series IC + regime-split test for date-level features (commodities, "
        "global rates, breadth, MF AUM, macro sentiment).", "",
        f"Method:",
        f"- Target: median 7d forward return across liquid universe (NIFTY-equivalent broad market).",
        f"- IC: Spearman correlation of (feature − 252d rolling mean) vs (market_fwd_ret_7 − 252d rolling mean).",
        f"- Regime split: top tertile vs bottom tertile of feature → mean market 7d return spread.",
        "",
        f"Thresholds: KEEP ≥ |IC|={IC_KEEP} AND |spread|={SPREAD_KEEP*100:.1f}%; WATCHLIST ≥ |IC|={WATCHLIST_IC} AND spread.", "",
        f"## Summary: KEEP={keep}  WATCHLIST={watch}  DROP={drop}  INSUFFICIENT={insf}", "",
        "## All features (sorted by verdict then |ts_ic|)", "",
        "| feature | n_dates | ts_ic | ts_ic_t | regime spread | regime IR | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in res.iterrows():
        sp = f"{r['regime_spread_pct']:+.2f}%" if r['regime_spread_pct'] is not None else "—"
        md.append(
            f"| `{r['feature']}` | {r['n_dates']} | "
            f"{r['ts_ic'] if r['ts_ic'] is not None else '—'} | "
            f"{r['ts_ic_t'] if r['ts_ic_t'] is not None else '—'} | "
            f"{sp} | "
            f"{r['regime_ir'] if r['regime_ir'] is not None else '—'} | "
            f"**{r['verdict']}** |"
        )
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nreport → {OUT_REPORT}")

    # 3. update registry
    if REGISTRY.exists():
        with open(REGISTRY) as f:
            reg = json.load(f)
        feat_lookup = {r["feature"]: r for r in rows}
        updated = 0
        for h in reg:
            for fname, vrec in feat_lookup.items():
                hid = h["id"].lower()
                fname_l = fname.lower()
                # match if hypothesis id is contained in feature name (e.g., macro_brent_5d in macro_brent_5d_pct)
                if hid in fname_l or fname_l.endswith(hid) or any(token in fname_l for token in hid.split("_") if len(token) >= 4):
                    h["lift_ic"] = vrec["ts_ic"]
                    h["state"] = vrec["verdict"]
                    h["notes"] = (h.get("notes") or "") + f" | macro_eval ts_ic={vrec['ts_ic']} spread={vrec['regime_spread_pct']}% n={vrec['n_dates']}"
                    updated += 1
                    break
        with open(REGISTRY, "w") as f:
            json.dump(reg, f, indent=2)
        print(f"updated {updated} registry entries → {REGISTRY}")


if __name__ == "__main__":
    main()
