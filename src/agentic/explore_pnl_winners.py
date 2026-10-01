"""Do P&L numbers separate winners? (2026-09-30, EXPLORATION, 2019-2022 only; 2023+ stays sealed for a registered test.)

Rows: logs/leader_sleeve/anatomy_1p5x/rows.parquet (weekly samples, every 5th session; P&L values point in time:
only quarters filed by that date). Winner = y95: the high reached 1.5x the entry close within the next 95 sessions.
Populations: ALL = tradable core band (ADV >= Rs 5 cr, then-traded close > Rs 50);
             POOL = V3-like pool: core + industry heat percentile >= 0.70 + trend test (>= 1.5x the 52-week low,
             12-month return >= 30%, above the 200-day average, 50-day above 200-day). V3's budget veto, finance
             drop and fading-theme drop are not applied (not in these rows).
Per feature: winner rate and lift by bucket (lift = bucket winner rate / the population's winner rate), and the
within-week AUC (0.50 = no separation; above 0.50 = higher values win more), averaged over weeks, overall and by year.
Units (rows manifest): growth and ratios are fractions (0.30 = +30%); pe is NaN when trailing EPS <= 0.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COLS = ["symbol", "trade_date", "core", "y95", "off_low", "ret252", "px_sma200", "sma50_200", "ind_heat_pct",
        "sales_yoy", "eps_yoy", "pe", "pe_ind", "profitable", "loss_to_profit", "days_since_results"]
R = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=COLS)
R["trade_date"] = pd.to_datetime(R["trade_date"])
R = R[(R["trade_date"] >= "2019-01-01") & (R["trade_date"] <= "2022-12-31") & R["y95"].notna() & R["core"].astype(bool)].copy()
# price-vs-average columns: ratio (centred on 1) or difference (centred on 0)? decide from the data, do not assume
c200 = 1.0 if R["px_sma200"].median() > 0.5 else 0.0
c50 = 1.0 if R["sma50_200"].median() > 0.5 else 0.0
print(f"px_sma200 median {R['px_sma200'].median():.3f}, sma50_200 median {R['sma50_200'].median():.3f} -> threshold {c200:g} / {c50:g}")
pool = (R["ind_heat_pct"] >= 0.70) & (R["off_low"] >= 0.5) & (R["ret252"] >= 0.30) & (R["px_sma200"] > c200) & (R["sma50_200"] > c50)
POPS = {"ALL": R, "POOL": R[pool]}

B = {
    "sales_yoy": ([-np.inf, 0, 0.15, 0.30, 0.60, np.inf], ["fell", "0-15%", "15-30%", "30-60%", ">60%"]),
    "eps_yoy": ([-np.inf, 0, 0.25, 0.50, 1.0, np.inf], ["fell", "0-25%", "25-50%", "50-100%", ">100%"]),
    "pe": ([0, 15, 30, 60, np.inf], ["<15", "15-30", "30-60", ">60"]),
    "pe_ind": ([0, 0.5, 1, 2, np.inf], ["<0.5x ind", "0.5-1x", "1-2x", ">2x ind"]),
    "days_since_results": ([-1, 30, 60, 90, np.inf], ["<=30d", "31-60d", "61-90d", ">90d"]),
}


def bucket(x, f):
    if f in B:
        edges, labs = B[f]
        b = pd.cut(x[f], edges, labels=labs, right=False).astype(object)
        if f == "pe":
            b = b.where(x[f].notna(), np.where(x["profitable"] == 0, "loss-making", "missing"))
        return b.where(x[f].notna() | (f == "pe"), "missing")
    return x[f].map({1: "yes", 1.0: "yes", True: "yes", 0: "no", 0.0: "no", False: "no"}).fillna("missing")


def wk_auc(x, f):
    out = {}
    for d, g in x[["trade_date", f, "y95"]].dropna().groupby("trade_date"):
        if len(g) >= 20 and 2 <= g["y95"].sum() <= len(g) - 2:
            r = g[f].rank()
            n1 = g["y95"].sum(); n0 = len(g) - n1
            out[d] = (r[g["y95"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    s = pd.Series(out)
    return s.mean(), s.groupby(s.index.year).mean() if len(s) else pd.Series(dtype=float), len(s)


log = {}
for name, x in POPS.items():
    base = x["y95"].mean()
    print(f"\n==================== {name} · rows {len(x):,} · weeks {x['trade_date'].nunique()} · winner rate {base:.1%} ====================")
    log[name] = dict(rows=len(x), base=round(base, 4))
    for f in ["sales_yoy", "eps_yoy", "profitable", "loss_to_profit", "pe", "pe_ind", "days_since_results"]:
        t = x.assign(b=bucket(x, f)).groupby("b")["y95"].agg(["size", "mean"])
        t["lift"] = t["mean"] / base
        a, ay, nw = wk_auc(x, f) if f not in ("profitable", "loss_to_profit") else wk_auc(x.assign(**{f: x[f].astype(float)}), f)
        print(f"\n{f}: within-week AUC {a:.3f} ({nw} weeks) · by year " + " ".join(f"{y}:{v:.3f}" for y, v in ay.items()))
        print("   " + " | ".join(f"{k}: {r['mean']:.1%} ({r['lift']:.2f}x, n={int(r['size']):,})" for k, r in t.iterrows()))
        log[name][f] = dict(auc=round(float(a), 3), auc_by_year={int(k): round(float(v), 3) for k, v in ay.items()},
                            buckets={str(k): dict(n=int(r["size"]), rate=round(float(r["mean"]), 4)) for k, r in t.iterrows()})
    both = (x["sales_yoy"] >= 0.30) & (x["eps_yoy"] >= 0.50)
    known = x["sales_yoy"].notna() & x["eps_yoy"].notna()
    print(f"\nsales >= +30% AND EPS >= +50%: {x.loc[both, 'y95'].mean():.1%} (n={int(both.sum()):,}) · "
          f"other known: {x.loc[known & ~both, 'y95'].mean():.1%} (n={int((known & ~both).sum()):,}) · P&L missing: {x.loc[~known, 'y95'].mean():.1%} (n={int((~known).sum()):,})")

with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-09-30-pnl-winners-EXPLORATION",
                             status="EXPLORATION 2019-2022 only (2023+ sealed); no rule changed", producer="src/agentic/explore_pnl_winners.py",
                             winner="y95 (1.5x within 95 sessions)", results=log)) + "\n")
