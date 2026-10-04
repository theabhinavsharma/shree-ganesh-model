"""Does the hot-industry list shift in particular months? (2026-10-04, EXPLORATION for Abhinav's question; no rule changed)

hot = heat_pct >= 0.70: cross-industry percentile of the mean 60-session return of an industry's core stocks (V3's definition,
src/agentic/build_industry_scores.py). Month-end snapshots from data/derived/industry_scores_policy.parquet.
  turnover   share of last month-end's hot industries that are no longer hot at this month-end
  rank_corr  rank correlation of heat_pct across industries between consecutive month-ends (1 = same order, 0 = reshuffled)
  p_corr     the same for P_pct, the budget + activity score behind V3's veto (P_pct < 0.30)
Budget dates: Union Budget publication dates from data/derived/budget_capex.parquet; for each, how many industries enter or
leave V3's veto (P_pct < 0.30) over the next 5 sessions, and how much of the hot list drops out over the next 20, vs any day.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet", columns=["date", "industry", "heat_pct", "P_pct", "E"])
S["date"] = pd.to_datetime(S["date"])


def rcorr(a, b):
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1]) if ok.sum() > 5 else np.nan


me = S.groupby(S.date.dt.to_period("M")).date.max()
snap = {p: S[S.date == d].set_index("industry") for p, d in me.items()}
rows = []
for a, b in zip(list(me.index)[:-1], list(me.index)[1:]):
    A, B = snap[a], snap[b]; com = A.index.intersection(B.index)
    hA = set(com[A.loc[com, "heat_pct"] >= 0.70]); hB = set(com[B.loc[com, "heat_pct"] >= 0.70])
    rows.append(dict(year=b.year, month=b.month, turnover=len(hA - hB) / max(len(hA), 1), hot=len(hB),
                     rank_corr=rcorr(A.loc[com, "heat_pct"], B.loc[com, "heat_pct"]), p_corr=rcorr(A.loc[com, "P_pct"], B.loc[com, "P_pct"])))
M = pd.DataFrame(rows)
by = M.groupby("month").agg(years=("year", "size"), turnover=("turnover", "mean"), lo=("turnover", "min"), hi=("turnover", "max"),
                            rank_corr=("rank_corr", "mean"), p_corr=("p_corr", "mean"))
print(f"month-ends {me.iloc[0].date()} .. {me.iloc[-1].date()} · average hot industries {M.hot.mean():.0f}")
print(f"all months: turnover {M.turnover.mean():.0%} · heat rank_corr {M.rank_corr.mean():.2f} · P rank_corr {M.p_corr.mean():.2f}\n")
print("month  years  hot list turnover (lowest-highest year)  heat rank_corr  P rank_corr")
for m, r in by.iterrows():
    print(f" {pd.Timestamp(2000, m, 1):%b}    {r.years:2.0f}     {r.turnover:4.0%} ({r.lo:3.0%}-{r.hi:3.0%})                    {r.rank_corr:5.2f}          {r.p_corr:5.2f}")

P = S.pivot_table(index="date", columns="industry", values="P_pct"); Hh = S.pivot_table(index="date", columns="industry", values="heat_pct")


def moved(X, i, k, thr, below):   # industries entering / leaving a set between session i-1 and session i-1+k
    a, b = X.iloc[i - 1], X.iloc[min(i - 1 + k, len(X) - 1)]
    sa, sb = (set(a.index[a < thr]), set(b.index[b < thr])) if below else (set(a.index[a >= thr]), set(b.index[b >= thr]))
    return len(sb - sa), len(sa - sb), len(sa), rcorr(a, b)


base_v = np.mean([sum(moved(P, i, 5, 0.30, True)[:2]) for i in range(1, len(P) - 5, 5)])
base_h = np.mean([moved(Hh, i, 20, 0.70, False)[1] / max(moved(Hh, i, 20, 0.70, False)[2], 1) for i in range(1, len(Hh) - 20, 20)])
pubs = sorted(pd.to_datetime(pd.read_parquet(ROOT / "data/derived/budget_capex.parquet", columns=["pub_date"])["pub_date"]).unique())
pubs = [p for p in pubs if p >= P.index[0]]
print(f"\nUnion Budget dates (data/derived/budget_capex.parquet pub_date); the budget score switches on the session after")
print(f"  baseline on any day: veto list {base_v:.1f} industries in+out per 5 sessions · hot list {base_h:.0%} turnover per 20 sessions")
out = dict(by_month=by.round(3).reset_index().to_dict("records"), budget=[], baseline=dict(veto_moves_5s=round(float(base_v), 2), hot_turnover_20s=round(float(base_h), 3)))
for p in pubs:
    i = int(P.index.searchsorted(p, side="right"))   # first session strictly after publication
    if i >= len(P): continue
    e, l, n0, rc = moved(P, i, 5, 0.30, True); he, hl, hn, hrc = moved(Hh, i, 20, 0.70, False)
    print(f"  {pd.Timestamp(p).date()}: veto list {n0} -> in {e}, out {l} (P rank_corr {rc:.2f}) · hot list 20 sessions later: {hl}/{hn} dropped ({hl / max(hn, 1):.0%}), heat rank_corr {hrc:.2f}")
    out["budget"].append(dict(date=str(pd.Timestamp(p).date()), veto_in=e, veto_out=l, p_corr=round(rc, 3), hot_dropped=hl, hot_n=hn))
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-04-heat-seasonality-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_heat_seasonality.py", results=out)) + "\n")
