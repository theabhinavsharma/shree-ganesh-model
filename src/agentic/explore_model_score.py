"""Exploration (not a registered test; changes no rule): what does the model score tell us about the achievers?

The score (bakeoff_preds 'ensemble') = average of the within-week percentile ranks of three walk-forward models (LightGBM,
XGBoost, a small neural net) trained to answer: will this stock's high reach 1.5x its close within the next 95 sessions?
So 0.98 means 'ranked above 98% of stocks that week', not '98% likely'. The three models' raw outputs are probabilities.
Outcome here: y95_next_open (bought at the next open, high reaches +50% within the 95-session window), weeks with a full
window only. Liquid = 20-day traded value >= Rs 1 cr and price > Rs 50 (V3.1's bar).
  A  score bucket -> share of achievers, all liquid stocks and inside the V3.1 candidate pool, by era and for 2025
  B  V3.1 picks: do the three models agreeing (lowest of the three ranks) or the score rising over 4 weeks pick achievers?
  C  the models' own probability level by year (average of LightGBM / XGBoost / net for the top-ranked stocks) vs the
     achiever rate those stocks actually had: does the absolute level warn of a weak year like 2025?
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
OUT = ROOT / "logs/leader_sleeve/model_score_dive"
PREDS = ROOT / "logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet"
PICKS = ROOT / "logs/leader_sleeve/v31_hits_by_year/v31_picks.csv"
BUCKETS = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0001]
LABELS = ["<50", "50-80", "80-90", "90-95", "95-98", "98-99", "99+"]


def main() -> None:
    p = pd.read_parquet(PREDS, columns=["symbol", "trade_date", "adv", "raw_close", "y95_next_open", "lgbm", "xgb", "mlp", "ensemble"])
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    adv_cr = p["adv"] / 1e7 if p["adv"].median() > 1e5 else p["adv"]           # the file stores rupees or crore; normalise
    p["liquid"] = (adv_cr >= 1.0) & (p["raw_close"] > 50)
    for m in ("lgbm", "xgb", "mlp"):
        p[f"{m}_rk"] = p.groupby("trade_date")[m].rank(pct=True)
    p["agree"] = p[["lgbm_rk", "xgb_rk", "mlp_rk"]].min(axis=1)
    p["prob"] = p[["lgbm", "xgb", "mlp"]].mean(axis=1)
    p["year"] = p["trade_date"].dt.year
    p["era"] = np.where(p["year"] <= 2022, "2019-22", "2023+")
    lab = p[p["y95_next_open"].notna()].copy()
    lab["b"] = pd.cut(lab["ensemble"], BUCKETS, labels=LABELS, right=False)
    res = {}

    print("=== A. score bucket -> share that hit +50% within 95 sessions (liquid stocks) ===")
    L = lab[lab["liquid"]]
    A = L.groupby(["era", "b"], observed=True)["y95_next_open"].agg(["size", "mean"]).unstack(0)
    A25 = L[L["year"] == 2025].groupby("b", observed=True)["y95_next_open"].agg(["size", "mean"])
    tab = pd.DataFrame({"2019-22 n": A[("size", "2019-22")], "2019-22 hit%": (100 * A[("mean", "2019-22")]).round(1),
                        "2023+ n": A[("size", "2023+")], "2023+ hit%": (100 * A[("mean", "2023+")]).round(1),
                        "2025 n": A25["size"], "2025 hit%": (100 * A25["mean"]).round(1)})
    print(tab.to_string()); res["A_liquid"] = tab.reset_index().to_dict("records")

    pk = pd.read_csv(PICKS, parse_dates=["trade_date"])
    pool_keys = None
    if "pool_n" in pk:
        pass
    # the V3.1 pool itself is not stored per stock; use the picks (the pool's top 9) for B, and the liquid universe for A

    print("\n=== B. V3.1 picks (phase 0): does agreement or a rising score pick achievers? ===")
    q = pk.merge(p[["symbol", "trade_date", "y95_next_open", "agree", "prob", "lgbm_rk", "xgb_rk", "mlp_rk", "ensemble"]], on=["symbol", "trade_date"], how="left")
    prev = p[["symbol", "trade_date", "ensemble"]].rename(columns={"trade_date": "d_prev", "ensemble": "ens_4w"}).sort_values("d_prev")
    q["d_look"] = q["trade_date"] - pd.Timedelta(days=28)
    q = pd.merge_asof(q.sort_values("d_look"), prev, left_on="d_look", right_on="d_prev", by="symbol", direction="backward", tolerance=pd.Timedelta(days=10))
    q["rise4w"] = q["ensemble"] - q["ens_4w"]
    q["era"] = np.where(q["trade_date"].dt.year <= 2022, "2019-22", "2023+")
    f = q[q["full"] & q["y95_next_open"].notna()].copy()
    out_b = {}
    for lab_, col, cuts, names in (("all three models agree (lowest of the three ranks)", "agree", [0, 0.90, 0.95, 1.01], ["<0.90", "0.90-0.95", "0.95+"]),
                                   ("score change over the last 4 weeks", "rise4w", [-9, -0.02, 0.02, 9], ["falling", "flat", "rising"]),
                                   ("the models' own probability (avg of 3)", "prob", None, None)):
        if cuts is None:
            f["g"] = pd.qcut(f[col], 3, labels=["lowest third", "middle", "highest third"])
        else:
            f["g"] = pd.cut(f[col], cuts, labels=names, right=False)
        g = f.groupby(["era", "g"], observed=True).agg(picks=("hit", "size"), hit126=("hit", "mean"), y95=("y95_next_open", "mean"), avg6m=("ret", "mean"))
        g[["hit126", "y95", "avg6m"]] = (100 * g[["hit126", "y95", "avg6m"]]).round(1)
        print(f"-- {lab_}:\n{g.to_string()}"); out_b[col] = g.reset_index().astype({"g": str}).to_dict("records")
        if col == "prob":
            print("   probability thirds cut at", np.round(f[col].quantile([1 / 3, 2 / 3]).to_numpy(), 3))
    res["B_picks"] = out_b

    print("\n=== C. the models' own probability level by year vs what the top-ranked liquid stocks did ===")
    top = lab[lab["liquid"] & (lab["ensemble"] >= 0.95)]
    C = top.groupby("year").agg(stock_weeks=("prob", "size"), avg_model_prob=("prob", "mean"), actual_hit=("y95_next_open", "mean"))
    base = lab[lab["liquid"]].groupby("year")["y95_next_open"].mean().rename("all_liquid_hit")
    C = C.join(base); C[["avg_model_prob", "actual_hit", "all_liquid_hit"]] = (100 * C[["avg_model_prob", "actual_hit", "all_liquid_hit"]]).round(1)
    print(C.to_string()); res["C_by_year"] = C.reset_index().to_dict("records")
    now = p[p["liquid"] & (p["trade_date"] == p["trade_date"].max()) & (p["ensemble"] >= 0.95)]
    print(f"latest week {p['trade_date'].max().date()}: top-ranked liquid stocks' average model probability {100 * now['prob'].mean():.1f}% ({len(now)} stocks)")
    m = p[p["liquid"] & (p["ensemble"] >= 0.95)].groupby(p["trade_date"].dt.to_period("Q"))["prob"].mean().tail(12)
    print("by quarter (last 12):", {str(k): round(100 * v, 1) for k, v in m.items()})
    res["C_latest"] = dict(week=str(p["trade_date"].max().date()), avg_prob=round(100 * float(now["prob"].mean()), 1), n=int(len(now)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "results.json.manifest.json").write_text(json.dumps(dict(dataset="results.json", producer="src/agentic/explore_model_score.py", kind="exploration",
        definitions=__doc__, units=dict(hit="percent of stock-weeks / picks whose high reached +50% (y95 = 95 sessions, hit126 = 126 sessions)",
                                        prob="percent, average of the three models' predicted probabilities", ensemble="within-week percentile 0-1"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUT / "README.md").write_text("# What the model score says about achievers\n\nExploration. results.json: A = score bucket vs achiever share, "
                                   "B = V3.1 picks by model agreement / score trend / probability level, C = model probability level by year.\n")


if __name__ == "__main__":
    main()
