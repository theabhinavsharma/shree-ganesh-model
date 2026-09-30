"""Exploration (discovery era 2019-2022 ONLY; 2023+ sealed): what did Sri Lakshmi v2's 2x winners look like on the day
they were picked? Non-linear: a shallow decision tree finds COMBINATIONS of traits (not one trait at a time) that
preceded a 2x move (high >= 2x the next open within 126 sessions). Then: profile of the top 15 winners vs all picks,
and the best combinations' 2x rate year by year (a real pattern holds in most years).
Traits at pick time: rows.parquet features (units in its manifest) + pool rank, times picked in the prior 25 weeks,
industry heat, weeks the industry has been hot, budget/activity pct, PIB attention, rising/fading themes tied to the
company, promoter market purchases in 90 days.
"""
import html, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, test_industry_fundamentals as tif
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet"); sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"])
sc["industry"] = sc["industry"].map(html.unescape); sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
fin = lambda s: sector.get(imap.get(s)) == "Financial Services"  # noqa: E731
wk = [d for d in sp.weekly_grid(cal, 0) if pd.Timestamp(tif.START) <= d < pd.Timestamp("2023-01-01")]
sre.TOPN = 100000
full = tif.select_elig(X["F"], wk, imap, P, S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)])
PX = rp.load_panel(["open", "high"]); O, H = rp.wide(PX, "open", cal), rp.wide(PX, "high", cal); del PX
I = pd.read_parquet(ROOT / "data/derived/theme_intensity.parquet"); I["date"] = pd.to_datetime(I["date"])
state = {(d, t): st for d, t, st in zip(I["date"], I["theme"], I["state"])}
E = pd.read_parquet(ROOT / "data/derived/theme_exposure.parquet"); E["an_dt"] = pd.to_datetime(E["an_dt"])
ex = {(s, t): np.sort(g["an_dt"].to_numpy()) for (s, t), g in E.groupby(["symbol", "theme"])}
tof = {}
for (s, t) in ex: tof.setdefault(s, []).append(t)
pit = pd.read_parquet(ROOT / "data/derived/pit_history.parquet", columns=["symbol", "personCategory", "acqMode", "date"])
pit = pit[pit["personCategory"].fillna("").str.contains("Promoter", case=False) & (pit["acqMode"].fillna("").str.strip() == "Market Purchase")]
pit["d"] = pd.to_datetime(pit["date"], format="%d-%b-%Y %H:%M", errors="coerce"); pb = {s: np.sort(g["d"].dropna().to_numpy()) for s, g in pit.groupby("symbol")}
hot = S.assign(h=S["heat_pct"] >= 0.70).pivot(index="date", columns="industry", values="h").sort_index()
cnt = lambda a, d, days: 0 if a is None else int(np.searchsorted(a, np.datetime64(d), side="right") - np.searchsorted(a, np.datetime64(d - pd.Timedelta(days=days)), side="right"))  # noqa: E731
rows, hist = [], []
for d in wk:
    names = full.get(d, []); v2 = [s for s in names[:9] if not fin(s)]; i0 = cal.get_loc(d) + 1
    hd = hot[hot.index <= d]
    for s in v2:
        e = O[s].iloc[i0] if s in O.columns else np.nan
        if not np.isfinite(e) or i0 + 126 > len(cal):
            continue
        ind = imap.get(s); hs = hd[ind] if ind in hd.columns else pd.Series(dtype=bool)
        age = int((hs[::-1].cumprod()).sum()) if len(hs) else 0
        st = [state.get((d, t), "n/a") for t in tof.get(s, []) if cnt(ex[(s, t)], d, 365) > 0]
        srow = S[(S["date"] == d) & (S["industry"] == ind)]
        rows.append(dict(week=d, symbol=s, pool_rank=names.index(s) + 1, pool_size=len(names),
                         picked_before_25w=sum(s in h for h in hist[-25:]), ind_heat=float(srow["heat_pct"].iloc[0]) if len(srow) else np.nan,
                         ind_hot_days=age, budget_activity_pct=float(srow["P_pct"].iloc[0]) if len(srow) else np.nan,
                         pib_attention=float(srow["H_raw"].iloc[0]) if len(srow) else np.nan,
                         themes_rising=st.count("rising"), themes_fading=st.count("fading"), promoter_buys_90d=cnt(pb.get(s), d, 90),
                         peak=float(np.nanmax(H[s].iloc[i0:i0 + 126].to_numpy())) / e - 1))
    hist.append(set(v2))
T = pd.DataFrame(rows)
R = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet").sort_values("trade_date")
drop = {"era", "ind", "core", "y95", "y63", "s95", "fh95", "fc95", "pred", "mcap_source", "ind_src", "close", "age_yrs"}
feat = [c for c in R.columns if c not in drop | {"symbol", "trade_date"} and pd.api.types.is_numeric_dtype(R[c])]
T = pd.merge_asof(T.sort_values("week"), R[["symbol", "trade_date"] + feat], left_on="week", right_on="trade_date", by="symbol",
                  direction="backward", tolerance=pd.Timedelta(days=7)).drop(columns="trade_date")
T["big"] = T["peak"] >= 1.0; T["year"] = T["week"].dt.year
cols = [c for c in T.columns if c not in ("week", "symbol", "peak", "big", "year")]
print(f"DISCOVERY ERA v2 picks {len(T)} · 2x winners {int(T['big'].sum())} ({T['big'].mean():.1%}) · traits {len(cols)}")
Xf = T[cols].astype(float).fillna(T[cols].astype(float).median())
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40, criterion="entropy", random_state=0).fit(Xf, T["big"])
print("\nDECISION TREE (2x rate in each leaf):\n" + export_text(tree, feature_names=cols, decimals=2, show_weights=True))
T["leaf"] = tree.apply(Xf)
L = T.groupby("leaf").agg(picks=("big", "size"), rate=("big", "mean")).sort_values("rate", ascending=False)
base_y = T.groupby("year")["big"].mean()
for leaf in L.index[:3]:
    y = T[T["leaf"] == leaf].groupby("year")["big"].agg(["mean", "size"])
    print(f"leaf {leaf}: {L.loc[leaf, 'picks']} picks · 2x rate {L.loc[leaf, 'rate']:.0%} (base {T['big'].mean():.0%}) · by year " +
          " ".join(f"{yr}:{m:.0%}(n{n})/base {base_y[yr]:.0%}" for yr, (m, n) in y.iterrows()))
top = T.nlargest(15, "peak")
key = ["peak", "pool_rank", "picked_before_25w", "ind_hot_days", "themes_rising", "themes_fading", "promoter_buys_90d", "mcap_cr",
       "avg_delivery_pct_20d", "off_high", "ret60", "ret252", "volume_vs_60d", "uc20", "eps_yoy", "sales_yoy", "pe", "days_since_results"]
key = [k for k in key if k in T.columns]
print("\nTOP 15 WINNERS vs ALL PICKS (medians):")
prof = pd.DataFrame({"top 15": top[key].median(), "all picks": T[key].median()}).round(2)
print(prof.to_string()); print("\ntop 15:", ", ".join(f"{r.symbol} {r.peak:+.0%} ({r.week:%b %y})" for r in top.itertuples()))
W = T[T["big"]]
print(f"\n2x picks {len(W)} came from {W['symbol'].nunique()} different stocks · by year: " +
      str(W.groupby('year')['symbol'].agg(lambda s: f"{len(s)} picks / {s.nunique()} stocks").to_dict()))
first = W.sort_values("week").drop_duplicates("symbol")
allfirst = T.sort_values("week").drop_duplicates("symbol")
print("FIRST time each 2x stock was picked vs first pick of every stock (medians):")
print(pd.DataFrame({"2x stocks": first[key[1:]].median(), "all stocks": allfirst[key[1:]].median()}).round(2).to_string())
T.to_parquet("/private/tmp/claude-501/-Users-abhinavs--Documents-Zoom/f1d31fd0-e296-4e81-a7bd-b84642862799/scratchpad/big_winners_T.parquet", index=False)
