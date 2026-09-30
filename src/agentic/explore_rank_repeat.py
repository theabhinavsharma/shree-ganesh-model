"""Exploration (discovery era 2019-2022 ONLY; 2023+ kept sealed for a registered test):
hit rate of +50% (high >= 1.5x next-open entry within 126 sessions) by pool rank and by repeat count, Sri Lakshmi G1 pool."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/abhinavs./Code/Zoom"); sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, test_industry_fundamentals as tif
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
cal = D["cal"]; PX = rp.load_panel(["open", "high", "close"]); O = rp.wide(PX, "open", cal); H = rp.wide(PX, "high", cal); C = rp.wide(PX, "close", cal); del PX
wk = [d for d in sp.weekly_grid(cal, 0) if pd.Timestamp(tif.START) <= d < pd.Timestamp("2023-01-01")]
sre.TOPN = 100000
pool = tif.select_elig(X["F"], wk, imap, P, S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)])
rows, sel_hist = [], []
for w, d in enumerate(wk):
    names = pool.get(d, []); i0 = cal.get_loc(d) + 1
    prior = sel_hist[-25:]
    for r, s in enumerate(names, 1):
        e = O[s].iloc[i0] if s in O.columns and i0 < len(cal) else np.nan
        if not np.isfinite(e) or i0 + 126 > len(cal):
            continue
        hit = float(np.nanmax(H[s].iloc[i0:i0 + 126].to_numpy())) >= 1.5 * e
        ret = C[s].iloc[i0 + 125] / e - 1
        rep = sum(s in p for p in prior)
        streak = 0
        for p in reversed(prior):
            if s in p: streak += 1
            else: break
        rows.append(dict(week=d, sym=s, rank=r, n_pool=len(names), hit=hit, ret=ret, rep=rep, streak=streak))
    sel_hist.append(set(names[:9]))
R = pd.DataFrame(rows); T = R[R["rank"] <= 9]
print(f"DISCOVERY ERA {wk[0].date()}..{wk[-1].date()} · {len(wk)} weeks · pool rows {len(R)} · picked rows {len(T)} · median pool size {int(R.groupby('week')['n_pool'].first().median())}")
print(f"base rate: picked {T['hit'].mean():.1%} hit +50% · whole pool {R['hit'].mean():.1%}")
def tab(df, col, bins, labels, title):
    g = df.assign(b=pd.cut(df[col], bins, labels=labels)).groupby("b", observed=True)
    t = pd.DataFrame(dict(n=g.size(), hit=g["hit"].mean().mul(100).round(1), med_ret=g["ret"].median().mul(100).round(1),
                          share_of_hitters=(g["hit"].sum() / df["hit"].sum() * 100).round(1)))
    print(f"\n{title}\n" + t.to_string())
tab(R, "rank", [0, 1, 2, 3, 6, 9, 15, 25, 1e9], ["1", "2", "3", "4-6", "7-9", "10-15", "16-25", "26+"], "BY RANK IN THE WEEKLY POOL (1-9 = bought)")
tab(T, "rep", [-1, 0, 2, 5, 10, 17, 25], ["0 (new)", "1-2", "3-5", "6-10", "11-17", "18-25"], "PICKED: BY TIMES PICKED IN THE PRIOR 25 WEEKS")
tab(T, "streak", [-1, 0, 1, 3, 7, 12, 25], ["0", "1", "2-3", "4-7", "8-12", "13-25"], "PICKED: BY CONSECUTIVE PRIOR WEEKS PICKED")
print("\nbest cell (n >= 30):", end=" ")
cells = T.groupby([pd.cut(T["rank"], [0, 3, 6, 9]), pd.cut(T["rep"], [-1, 0, 5, 25])], observed=True)["hit"].agg(["size", "mean"])
print(cells[cells["size"] >= 30].sort_values("mean").tail(3).round(3).to_string())

# 2026-09-30 follow-up ("between 1-9, is there a rank set with higher odds? fewer stocks, more certainty?")
# Buy only the top k each week (equal weight), discovery era only. Per stock: odds of +50%, of any gain, of -20% or
# worse at the 126-session close. Per weekly batch: how often the batch ends up, and how bad the bad batches are.
print("\nTOP-k SUBSETS (discovery era only)")
out = []
for k in (1, 2, 3, 4, 5, 6, 9):
    t = R[R["rank"] <= k]
    b = t.groupby("week")["ret"].mean()
    se = (t["hit"].mean() * (1 - t["hit"].mean()) / len(t)) ** 0.5
    out.append(dict(k=k, stocks=len(t), hit50=round(100 * t["hit"].mean(), 1), pm2se=round(200 * se, 1),
                    any_gain=round(100 * (t["ret"] > 0).mean(), 1), loss20=round(100 * (t["ret"] <= -0.2).mean(), 1),
                    med_ret=round(100 * t["ret"].median(), 1), batches=len(b), batch_up=round(100 * (b > 0).mean(), 1),
                    batch_mean=round(100 * b.mean(), 1), batch_p10=round(100 * b.quantile(0.1), 1), batch_worst=round(100 * b.min(), 1)))
print(pd.DataFrame(out).to_string(index=False))
