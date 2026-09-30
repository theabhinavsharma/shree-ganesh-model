"""Exploration (discovery era 2019-2022 ONLY; 2023+ sealed): WHY do some sectors give far more +50% hitters?
Candidate reasons, measured per industry on the screen date from the stocks in it (core band), point-in-time:
  swing        median 60-session volatility of daily returns (a +50% move needs room to move)
  size         median market cap, Rs cr (small companies can re-rate further)
  valuation    median P/E (cheap industries re-rate; always-expensive ones cannot)
  op_leverage  median EPS growth minus sales growth (profits growing faster than sales = margins widening)
  profit_grow  median EPS growth year on year
  heat_age     weeks in a row the industry has been hot (heat pct >= 0.70): young vs long-running themes
For each: hit rate of Sri Lakshmi picks by the trait's fifths, then whether the trait explains the sector gap
(sector hit rates within the high and low half of the trait). Units: rows.parquet manifest; delivery-like fractions.
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
vol60 = np.log(C).diff().rolling(60, min_periods=40).std() * np.sqrt(252)
wk = [d for d in sp.weekly_grid(cal, 0) if pd.Timestamp(tif.START) <= d < pd.Timestamp("2023-01-01")]
sre.TOPN = 9
picks = tif.select_elig(X["F"], wk, imap, P, S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)])
R = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "mcap_cr", "pe", "eps_yoy", "sales_yoy"])
R = R.sort_values("trade_date")
sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"]); sc["industry"] = sc["industry"].map(__import__("html").unescape)
sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
Sh = S.pivot(index="date", columns="industry", values="heat_pct").sort_index()
hot = (Sh >= 0.70).astype(int)
wk_all = [d for d in sp.weekly_grid(cal, 0)]
Hw = hot.reindex(wk_all).fillna(0)
age = Hw.copy() * 0
for i in range(len(Hw)):
    age.iloc[i] = (age.iloc[i - 1] + 1) * Hw.iloc[i] if i else Hw.iloc[i]
rows = []
for d in wk:
    F = X["F"][(X["F"]["trade_date"] == d) & X["F"]["core"]][["symbol"]].copy()
    F["industry"] = F["symbol"].map(imap)
    F["vol"] = vol60.loc[d].reindex(F["symbol"]).to_numpy()
    f = R[R["trade_date"] <= d].groupby("symbol").tail(1).set_index("symbol")
    F = F.join(f[["mcap_cr", "pe", "eps_yoy", "sales_yoy"]], on="symbol")
    F["opl"] = F["eps_yoy"] - F["sales_yoy"]
    I = F.groupby("industry").agg(swing=("vol", "median"), size=("mcap_cr", "median"), valuation=("pe", "median"),
                                  op_leverage=("opl", "median"), profit_grow=("eps_yoy", "median"))
    i0 = cal.get_loc(d) + 1
    for s in picks.get(d, []):
        e = O[s].iloc[i0] if s in O.columns else np.nan
        ind = imap.get(s)
        if not np.isfinite(e) or i0 + 126 > len(cal) or ind not in I.index:
            continue
        rows.append(dict(week=d, symbol=s, industry=ind, sector=sector.get(ind, "unknown"), **I.loc[ind].to_dict(),
                         heat_age=float(age.loc[d, ind]) if ind in age.columns and d in age.index else np.nan,
                         hit=float(np.nanmax(H[s].iloc[i0:i0 + 126].to_numpy())) >= 1.5 * e, ret=C[s].iloc[i0 + 125] / e - 1))
T = pd.DataFrame(rows); T["year"] = T["week"].dt.year
print(f"DISCOVERY ERA · picks {len(T)} · base hit {T['hit'].mean():.1%}")
traits = ["swing", "size", "valuation", "op_leverage", "profit_grow", "heat_age"]
print("\n1) HIT RATE BY FIFTHS OF EACH INDUSTRY TRAIT (low -> high) · spread = top fifth minus bottom · years the top beat the bottom")
for t in traits:
    x = T.dropna(subset=[t]); q = pd.qcut(x[t].rank(method="first"), 5, labels=False)
    h = x.groupby(q)["hit"].mean().mul(100).round(1).tolist()
    edges = x.groupby(q)[t].median().round(2).tolist()
    yr = sum((x[(q == 4) & (x["year"] == y)]["hit"].mean() > x[(q == 0) & (x["year"] == y)]["hit"].mean()) for y in (2019, 2020, 2021, 2022))
    print(f"{t:12s} hit by fifth {h} · spread {h[-1] - h[0]:+.1f} · top>bottom in {yr}/4 years · fifth medians {edges}")
good, bad = ["Industrials", "Commodities", "Healthcare", "Utilities", "Information Technology"], ["Financial Services", "Fast Moving Consumer Goods", "Energy"]
T["grp"] = np.where(T["sector"].isin(good), "winning sectors", np.where(T["sector"].isin(bad), "losing sectors", "other"))
print("\n2) TRAIT PROFILE: winning vs losing sectors (medians at pick time)")
print(T.groupby("grp")[traits + ["hit"]].median().round(2).to_string())
print("\n3) DOES THE TRAIT EXPLAIN THE SECTOR GAP? hit % of winning vs losing sectors inside the low / high half of each trait")
for t in traits:
    x = T.dropna(subset=[t]); hi = x[t] >= x[t].median()
    cell = x[x["grp"] != "other"].groupby([hi[x["grp"] != "other"].map({True: "high", False: "low"}), "grp"])["hit"].agg(["mean", "size"])
    print(f"{t:12s} " + " · ".join(f"{a}/{b.split()[0]} {m * 100:.0f}% (n {n})" for (a, b), (m, n) in cell.iterrows()))
