"""Exploration (discovery era 2019-2022 ONLY; 2023+ sealed): are there pockets (industry heat level, industry size,
broad sector) where Sri Lakshmi's picks hit +50% disproportionately often? For each group: picks, +50% hit rate,
median 126-session return, share that lost 20%+, and in how many of the 4 years the group beat that year's
overall hit rate. A pocket worth a registered test beats the base in 4/4 years with enough picks.
Hit = high >= 1.5x next-open entry within 126 sessions; ret = close at session 126 / entry - 1.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, test_industry_fundamentals as tif
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
cal = D["cal"]; PX = rp.load_panel(["open", "high", "close"])
O, H, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
wk = [d for d in sp.weekly_grid(cal, 0) if pd.Timestamp(tif.START) <= d < pd.Timestamp("2023-01-01")]
sre.TOPN = 100000
pool = tif.select_elig(X["F"], wk, imap, P, S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)])
rows = []
for d in wk:
    i0 = cal.get_loc(d) + 1
    for r, s in enumerate(pool.get(d, []), 1):
        e = O[s].iloc[i0] if s in O.columns else np.nan
        if np.isfinite(e) and i0 + 126 <= len(cal):
            rows.append(dict(week=d, symbol=s, rank=r, hit=float(np.nanmax(H[s].iloc[i0:i0 + 126].to_numpy())) >= 1.5 * e,
                             ret=C[s].iloc[i0 + 125] / e - 1))
R = pd.DataFrame(rows)
R["industry"] = R["symbol"].map(imap)
sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"])
sc["industry"] = sc["industry"].map(__import__("html").unescape)
sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
R["sector"] = R["industry"].map(sector).fillna("unknown")
R = R.merge(S[["date", "industry", "heat_pct", "n_core"]].rename(columns={"date": "week"}), on=["week", "industry"], how="left")
R["heat"] = pd.cut(R["heat_pct"], [0.699, 0.8, 0.9, 0.95, 1.0], labels=["0.70-0.80", "0.80-0.90", "0.90-0.95", "0.95-1.00"])
R["ind_size"] = pd.cut(R["n_core"], [4, 7, 12, 24, 999], labels=["5-7 names", "8-12", "13-24", "25+"])
R["year"] = R["week"].dt.year
T = R[R["rank"] <= 9]
base_y = T.groupby("year")["hit"].mean()
print(f"DISCOVERY ERA · picks {len(T)} · base hit {T['hit'].mean():.1%} · by year {base_y.round(3).to_dict()}")

def table(df, col, min_n=40):
    g = df.groupby(col, observed=True)
    t = pd.DataFrame(dict(picks=g.size(), hit=(g["hit"].mean() * 100).round(1), median_ret=(g["ret"].median() * 100).round(1),
                          lost20=(g["ret"].apply(lambda x: (x <= -0.2).mean()) * 100).round(1)))
    yrs = df.groupby([col, "year"], observed=True)["hit"].mean().unstack()
    t["years_above_base"] = [int((yrs.loc[k].dropna() > base_y.reindex(yrs.loc[k].dropna().index)).sum()) if k in yrs.index else 0 for k in t.index]
    t["years_with_picks"] = [int(yrs.loc[k].notna().sum()) if k in yrs.index else 0 for k in t.index]
    return t[t["picks"] >= min_n].sort_values("hit", ascending=False)

pd.set_option("display.width", 200)
for col, title in (("heat", "BY INDUSTRY HEAT"), ("ind_size", "BY INDUSTRY SIZE (core-band names)"), ("sector", "BY BROAD SECTOR (>= 40 picks)")):
    print(f"\n{title}\n" + table(T, col).to_string())
print("\nBY INDUSTRY (>= 40 picks)\n" + table(T, "industry").head(12).to_string())
