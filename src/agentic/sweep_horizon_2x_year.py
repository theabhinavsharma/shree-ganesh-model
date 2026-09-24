"""EXP-2026-09-24-horizon-sweep-2x-year — which (horizon, target) bucket doubles capital
within a year most reliably?

Same machinery for every bucket (registered in logs/experiments.jsonl before this ran):
walk-forward LightGBM per bucket on the engines' 19 price features, yearly refit with a
label embargo, weekly top-8, enter next open, sell at +X% or at day-H close, 0.30% RT,
capital laddered across H/5 weekly tranches. Headline: P(12-month multiple >= 2x) over
every rolling 52-week window, per era.

Writes reports/horizon_sweep_2x_year.md and appends the result to logs/experiments.jsonl.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import find_multibagger_today as E  # noqa: E402  (panel + the engines' price features)

EXP_ID = "EXP-2026-09-24-horizon-sweep-2x-year"
BUCKETS = [(15, 0.05), (30, 0.10), (60, 0.20), (126, 0.40), (252, 1.00)]
COST, TOP, YEARS = 0.003, 8, range(2018, 2027)
FEATS = E.BASE_FEATS

print("panel…", flush=True)
df = E.build_panel().dropna(subset=FEATS).reset_index(drop=True)
df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
cal = {d: i for i, d in enumerate(sorted(df["trade_date"].unique()))}
df["td"] = df["trade_date"].map(cal).astype(int)
g = df.groupby("symbol", sort=False)
df["open1"] = g["open"].shift(-1)


def fwd_max(s, h):
    return s.iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1].shift(-1)


days = sorted(cal)
weekly = set(days[::5])
df["weekly"] = df["trade_date"].isin(weekly)
df["investable"] = (df["adv_20d_cr"] >= 5) & (df["close"] > 50)
wd = sorted(d for d in weekly if d >= pd.Timestamp("2018-01-01"))
widx = {d: i for i, d in enumerate(wd)}

results, cohorts_all = {}, {}
for H, X in BUCKETS:
    name = f"{H}d/{int(X*100)}%"
    print(f"\n== {name} ==", flush=True)
    df["fh"] = g["high"].transform(lambda s: fwd_max(s, H))
    df["cH"] = g["close"].shift(-H)
    df["label"] = (df["fh"] / df["close"] - 1 >= X).astype(float).where(df["fh"].notna())
    # outcome from the next open: +X% if reached, else close of day H
    hit = df["fh"] / df["open1"] - 1 >= X
    df["ret"] = np.where(hit, X, df["cH"] / df["open1"] - 1) - COST
    df.loc[df["cH"].isna() | df["open1"].isna(), "ret"] = np.nan
    df["hit"] = hit & df["ret"].notna()

    train_base = df[df["weekly"] & df["label"].notna() & (df["adv_20d_cr"] >= 1.0)]
    picks = []
    for Y in YEARS:
        ents = [d for d in wd if d.year == Y]
        if not ents:
            continue
        tr = train_base[train_base["td"] + H < cal[ents[0]]]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64, min_child_samples=200,
                               feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                               random_state=42, verbose=-1, n_jobs=-1).fit(tr[FEATS], tr["label"].astype(int))
        sc = df[df["trade_date"].isin(ents) & df["investable"]].copy()
        sc["p"] = m.predict_proba(sc[FEATS])[:, 1]
        picks.append(sc.sort_values(["trade_date", "p"], ascending=[True, False]).groupby("trade_date").head(TOP))
        print(f"  {Y}: train {len(tr):,} rows (pos {tr['label'].mean()*100:.1f}%)", flush=True)
    P = pd.concat(picks)
    coh = P.groupby("trade_date").agg(ret=("ret", "mean"), hit=("hit", "mean"), n=("ret", "count"))
    coh = coh[coh["n"] >= 4]                                     # outcome known for the cohort
    base = df[df["weekly"] & df["investable"] & df["trade_date"].isin(wd)].groupby("trade_date")["ret"].mean().dropna()

    def ladder(cret: pd.Series) -> pd.Series:
        """Realized equity on the weekly grid; H/5 tranches, cash when a cohort is missing."""
        N = max(1, round(H / 5)); T = len(wd)
        val = np.ones((N, T + N + 1))
        for j in range(N):
            v = 1.0
            for s in range(j, T, N):
                r = cret.get(wd[s], np.nan)
                v *= 1 + (0.0 if np.isnan(r) else r)
                e = min(s + N, T + N)
                val[j, e:] = v
        return pd.Series(val.mean(axis=0)[:T], index=wd)

    def roll12(eq: pd.Series, last_valid: pd.Timestamp) -> pd.DataFrame:
        r = []
        for i, d in enumerate(eq.index):
            if i + 52 >= len(eq) or eq.index[i + 52] > last_valid:
                break
            r.append((d, eq.iloc[i + 52] / eq.iloc[i]))
        return pd.DataFrame(r, columns=["start", "mult"])

    last_valid = coh.index.max()
    out = {}
    for label, cret in (("model", coh["ret"]), ("baseline", base)):
        rr = roll12(ladder(cret), last_valid)
        for era, m_ in (("disc", rr["start"].dt.year <= 2022), ("conf", rr["start"].dt.year >= 2023)):
            s = rr[m_]["mult"]
            ce = cret[(cret.index.year <= 2022) if era == "disc" else (cret.index.year >= 2023)]
            out[(label, era)] = {"windows": int(len(s)),
                                 "p_2x": round(float((s >= 2).mean() * 100), 1) if len(s) else None,
                                 "p_1_5x": round(float((s >= 1.5).mean() * 100), 1) if len(s) else None,
                                 "med_mult": round(float(s.median()), 2) if len(s) else None,
                                 "p10_mult": round(float(s.quantile(0.1)), 2) if len(s) else None,
                                 "cohort_mean_pct": round(float(ce.mean() * 100), 2) if len(ce) else None,
                                 "cohorts": int(len(ce))}
        if label == "model":
            for era in ("disc", "conf"):
                cm = coh[(coh.index.year <= 2022) if era == "disc" else (coh.index.year >= 2023)]
                out[("model", era)]["hit_pct"] = round(float(cm["hit"].mean() * 100), 1)
    results[name] = {f"{a}|{b}": v for (a, b), v in out.items()}
    cohorts_all[name] = coh
    print(json.dumps(results[name]), flush=True)

# ---------------- report
need = {H: (2 ** (H / 252) - 1) * 100 for H, _ in BUCKETS}
md = [f"# Horizon sweep — which bucket doubles capital within a year? ({EXP_ID})", "",
      f"_generated {datetime.now():%Y-%m-%d %H:%M} · weekly entries 2018-01..(last with a complete outcome) · walk-forward LGBM per bucket, "
      f"top-{TOP}/week, next-open entry, sell at +X% or day-H close, 0.30% RT, no stop, H/5-tranche ladder, realized-only equity_", "",
      "| bucket | era | cohorts | hit % | cohort net % | needed/cycle for 2x/yr % | 12m windows | **P(≥2x in 12m) %** | P(≥1.5x) % | median 12m × | 10th pct 12m × | baseline P(2x) % | baseline median × |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for H, X in BUCKETS:
    name = f"{H}d/{int(X*100)}%"
    for era in ("disc", "conf"):
        m, b = results[name][f"model|{era}"], results[name][f"baseline|{era}"]
        md.append(f"| {name} | {era} | {m['cohorts']} | {m.get('hit_pct')} | {m['cohort_mean_pct']} | {need[H]:.1f} | {m['windows']} | "
                  f"**{m['p_2x']}** | {m['p_1_5x']} | {m['med_mult']} | {m['p10_mult']} | {b['p_2x']} | {b['med_mult']} |")
rank = sorted(results, key=lambda k: (min(results[k]["model|disc"]["p_2x"] or 0, results[k]["model|conf"]["p_2x"] or 0),
                                      min(results[k]["model|disc"]["p10_mult"] or 0, results[k]["model|conf"]["p10_mult"] or 0)), reverse=True)
md += ["", f"**Ranking (registered rule: P(2x/12m) in the weaker era, tie → higher 10th pct):** " + " > ".join(rank), "",
       "## Caveats", "",
       "- Realized-only equity: open positions are not marked to market, so drawdowns are understated.",
       "- Gap-ups through the target are filled at the target (conservative); no stop-loss in any bucket.",
       "- Price-only model: the engines' 19 tape features. 252d windows starting after ~Sep 2024 can't close yet, so conf-era long buckets have fewer windows.",
       "- 5 buckets tested; all reported."]
(ROOT / "reports/horizon_sweep_2x_year.md").write_text("\n".join(md) + "\n")
with open(ROOT / "logs/experiments.jsonl", "a") as fh:
    fh.write(json.dumps({"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "id": EXP_ID, "status": "MEASURED",
                         "ranking": rank, "results": results}) + "\n")
print("\n".join(md))
