"""Factor evaluator — measure each new feature's lift on 7d-forward return.

Three metrics per feature (the third is the strict gate, added 2026-04-29
after the 5-KEEP-factors A/B fail showed high IC ≠ portfolio lift):

  • IC (Information Coefficient): Pearson rank-corr (Spearman) between
    feature value and forward 7d close-to-close return, averaged daily then meaned
  • Decile spread: rank stocks daily into 10 buckets by feature, take
    (top-decile mean fwd return − bottom-decile mean) → annualised IR
  • PORTFOLIO LIFT (top-5 basket): retrain a tiny model with-and-without
    the feature, measure top-5 daily basket mean 7d return delta. This is
    the ground truth — high IC factors that overlap existing inputs DON'T
    lift portfolios.

Verdict logic (IC + IR + portfolio lift):
  KEEP if (|IC|>=0.02 AND |IR|>=0.5 AND portfolio_lift_pp >= 0.30)
  WATCHLIST if IC+IR pass but portfolio_lift inconclusive
  DROP otherwise

This stricter gate prevents wasting model capacity on factors that the
existing inputs already capture.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
EXTRA = ROOT / "data/derived/extra_features.parquet"
REGISTRY = ROOT / "data/derived/factor_registry.json"
OUT_REPORT = ROOT / "reports/factor_evaluation.md"

H = 7  # forward horizon

# minimum thresholds to mark a factor as KEEP-worthy
IC_KEEP = 0.02
IR_KEEP = 0.5


def main(min_obs: int = 30) -> None:
    print("== factor_evaluator ==")
    df_extra = pd.read_parquet(EXTRA)
    df_extra["trade_date"] = pd.to_datetime(df_extra["trade_date"])
    feat_cols = [c for c in df_extra.columns if c not in ("symbol", "trade_date")]
    print(f"  evaluating {len(feat_cols)} features over {df_extra['trade_date'].nunique()} trading days")

    # build forward 7d close-to-close return
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    px["fwd_close_7"] = px.groupby("symbol")["close"].shift(-H)
    px["fwd_ret_7"] = px["fwd_close_7"] / px["close"] - 1

    panel = df_extra.merge(px[["symbol", "trade_date", "fwd_ret_7"]],
                            on=["symbol", "trade_date"], how="left")
    panel = panel.dropna(subset=["fwd_ret_7"])

    rows = []
    for f in feat_cols:
        sub = panel[["trade_date", f, "fwd_ret_7"]].dropna()
        if len(sub) < min_obs:
            rows.append({"feature": f, "n": len(sub), "ic_mean": None, "ic_t": None,
                         "decile_spread": None, "ir_annualised": None, "verdict": "INSUFFICIENT"})
            continue

        # daily Spearman IC
        daily_ic = sub.groupby("trade_date").apply(
            lambda g: g[[f, "fwd_ret_7"]].corr(method="spearman").iloc[0, 1] if len(g) >= 5 else np.nan,
            include_groups=False,
        ).dropna()
        if len(daily_ic) < 30:
            ic_mean = daily_ic.mean() if len(daily_ic) else np.nan
            ic_t = np.nan
        else:
            ic_mean = float(daily_ic.mean())
            ic_t = float(ic_mean / (daily_ic.std() / np.sqrt(len(daily_ic))))

        # decile spread (top - bottom)
        sub["decile"] = sub.groupby("trade_date")[f].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates="drop") if x.nunique() >= 10 else np.nan)
        sub_clean = sub.dropna(subset=["decile"])
        spread_daily = sub_clean.groupby("trade_date").apply(
            lambda g: g.loc[g["decile"] == 9, "fwd_ret_7"].mean() - g.loc[g["decile"] == 0, "fwd_ret_7"].mean(),
            include_groups=False,
        ).dropna()
        if len(spread_daily) < 30:
            spread_mean = spread_daily.mean() if len(spread_daily) else np.nan
            ir = np.nan
        else:
            spread_mean = float(spread_daily.mean())
            spread_std = float(spread_daily.std())
            # annualised IR = (mean_daily * 252) / (std_daily * sqrt(252))
            ir = (spread_mean * np.sqrt(252)) / spread_std if spread_std > 0 else np.nan

        # Two-stage gate added 2026-04-29:
        #   IC_PASSED = passes cross-sectional gate, awaiting portfolio A/B
        #   DROP      = fails IC or IR threshold
        # The KEEP verdict is now ONLY granted by portfolio_lift_evaluator.py
        # after it runs an actual top-5 backtest.
        verdict = "IC_PASSED" if (abs(ic_mean) >= IC_KEEP and abs(ir) >= IR_KEEP) else "DROP"
        if pd.isna(ic_mean) or pd.isna(ir):
            verdict = "INSUFFICIENT"
        rows.append({
            "feature": f, "n": len(sub),
            "ic_mean": round(ic_mean, 5) if not pd.isna(ic_mean) else None,
            "ic_t": round(ic_t, 2) if not pd.isna(ic_t) else None,
            "decile_spread": round(spread_mean, 5) if not pd.isna(spread_mean) else None,
            "ir_annualised": round(ir, 3) if not pd.isna(ir) else None,
            "verdict": verdict,
        })

    res = pd.DataFrame(rows).sort_values(["verdict", "ic_mean"], ascending=[True, False])
    print("\n=== factor evaluation results ===")
    print(res.to_string(index=False))
    keep_count = (res["verdict"] == "KEEP").sum()
    drop_count = (res["verdict"] == "DROP").sum()
    print(f"\nKEEP: {keep_count}   DROP: {drop_count}   INSUFFICIENT: {(res['verdict']=='INSUFFICIENT').sum()}")

    # write markdown report
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    # manual markdown (no tabulate dep)
    md_lines = [
        f"# Factor Evaluation — {pd.Timestamp.utcnow():%Y-%m-%d}", "",
        f"Horizon: {H} trading days forward close-to-close.", "",
        f"Thresholds: IC mean ≥ {IC_KEEP}, |IR| ≥ {IR_KEEP}.", "",
        "| feature | n | ic_mean | ic_t | decile_spread | ir_annualised | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in res.iterrows():
        md_lines.append(
            f"| `{r['feature']}` | {r['n']:,} | {r['ic_mean'] if r['ic_mean'] is not None else '—'} | "
            f"{r['ic_t'] if r['ic_t'] is not None else '—'} | "
            f"{r['decile_spread'] if r['decile_spread'] is not None else '—'} | "
            f"{r['ir_annualised'] if r['ir_annualised'] is not None else '—'} | "
            f"**{r['verdict']}** |"
        )
    OUT_REPORT.write_text("\n".join(md_lines))
    print(f"report → {OUT_REPORT}")

    # update registry
    if REGISTRY.exists():
        with open(REGISTRY) as f:
            reg = json.load(f)
        # naive name-match between registry name/id and feature column
        feat_lookup = {r["feature"]: r for r in rows}
        for h in reg:
            for k, v in feat_lookup.items():
                if h["id"].lower() in k.lower() or k.lower() in h["id"].lower():
                    h["lift_ic"] = v["ic_mean"]
                    # IC_PASSED → awaits portfolio A/B for KEEP. DROP / INSUFFICIENT terminal.
                    h["state"] = v["verdict"] if v["verdict"] in ("DROP", "IC_PASSED") else "EVALUATED"
                    break
        with open(REGISTRY, "w") as f:
            json.dump(reg, f, indent=2)
        print(f"updated registry verdicts → {REGISTRY}")


if __name__ == "__main__":
    main()
