"""Exploration (2026-09-30): which industries produced unusually many +50% winners, year by year, 2016 to mid-2022.
Used to build the theme list for the theme engine from data instead of guesses. Entries up to 2022-06-30 only, so
every 126-session window ends inside 2022; 2023+ stays unseen for the registered theme test.
Universe: every core-band stock (20d ADV >= Rs 5 cr, price > Rs 50) on each weekly screen date. Hit = high >= 1.5x the
next open within 126 sessions. Lift = industry hit rate / that year's hit rate. A "wave" = lift >= 1.8 with >= 20 hits
from >= 4 different stocks.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]
PX = rp.load_panel(["open", "high"]); O, H = rp.wide(PX, "open", cal), rp.wide(PX, "high", cal); del PX
fmax = H[::-1].rolling(126, min_periods=100).max()[::-1].shift(-1)          # max high over the next 126 sessions
hitw = fmax >= 1.5 * O.shift(-1)
known = fmax.notna() & O.shift(-1).notna()
wk = [d for d in sp.weekly_grid(cal, 0) if pd.Timestamp("2016-01-01") <= d <= pd.Timestamp("2022-06-30")]
F = X["F"][X["F"]["trade_date"].isin(wk) & X["F"]["core"]][["trade_date", "symbol"]]
F = F[F["symbol"].isin(H.columns)]
di = {d: i for i, d in enumerate(cal)}; ci = {s: j for j, s in enumerate(H.columns)}
ii = F["trade_date"].map(di).to_numpy(); jj = F["symbol"].map(ci).to_numpy()
F["hit"] = hitw.to_numpy()[ii, jj]; F["known"] = known.to_numpy()[ii, jj]
F = F[F["known"]]; F["industry"] = F["symbol"].map(imap); F["year"] = F["trade_date"].dt.year
F = F.dropna(subset=["industry"])
yr = F.groupby("year")["hit"].mean()
print("core stock-weeks", len(F), "· hit rate by year", (yr * 100).round(1).to_dict())
g = F.groupby(["year", "industry"]).agg(n=("hit", "size"), hits=("hit", "sum"), stocks=("symbol", "nunique"),
                                         hit_stocks=("symbol", lambda s: s[F.loc[s.index, "hit"]].nunique()))
g["rate"] = g["hits"] / g["n"]; g["lift"] = g["rate"] / yr.reindex(g.index.get_level_values(0)).to_numpy()
W = g[(g["lift"] >= 1.8) & (g["hits"] >= 20) & (g["hit_stocks"] >= 4)].reset_index().sort_values(["year", "lift"], ascending=[True, False])
pd.set_option("display.width", 200)
print("\nWAVES (lift >= 1.8, >= 20 hits from >= 4 stocks):")
for y, w in W.groupby("year"):
    print(f"{y}: " + " · ".join(f"{r.industry} x{r.lift:.1f} ({int(r.hit_stocks)} stocks)" for r in w.head(8).itertuples()))
rep = W.groupby("industry").agg(years=("year", lambda x: sorted(x.tolist())), best_lift=("lift", "max")).sort_values("best_lift", ascending=False)
print(f"\nINDUSTRIES WITH AT LEAST ONE WAVE ({len(rep)}):")
print(rep.to_string())
W.to_csv(ROOT / "logs/leader_sleeve/hitter_waves_2016_2022.csv", index=False)
