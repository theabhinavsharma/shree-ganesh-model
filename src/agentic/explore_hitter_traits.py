"""Exploration (discovery era 2019-2022 ONLY; 2023+ sealed): which traits separate +50% hitters from the rest INSIDE the
Sri Lakshmi (G1) pool, at the same model rank? A trait the model already uses shows little spread once rank is held
fixed; a trait it is missing keeps a spread. Hit = high >= 1.5x next-open entry within 126 sessions.
Features: logs/leader_sleeve/anatomy_1p5x/rows.parquet (units in its manifest), as-of the screen date (<= 7 days old).
Per feature: hit rate in the pool's top vs bottom fifth, within 3 rank bands (1-9 bought, 10-25, 26+), averaged;
the same spread per year 2019..2022 (how many of 4 years agree); n and a 2-SE band. Many features are tested, so a
few will look good by chance: only spreads well past 2 SE and consistent in 4 of 4 years are worth a registered test.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/abhinavs./Code/Zoom"); sys.path.insert(0, str(ROOT / "src/agentic"))
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
R = pd.DataFrame(rows).sort_values("week")
F = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet").sort_values("trade_date")
skip = {"symbol", "trade_date", "era", "ind", "core", "y95", "y63", "s95", "fh95", "fc95", "pred", "mcap_source", "ind_src", "close"}
feats = [c for c in F.columns if c not in skip and pd.api.types.is_numeric_dtype(F[c])]
M = pd.merge_asof(R, F[["symbol", "trade_date"] + feats].rename(columns={"trade_date": "fdate"}), left_on="week", right_on="fdate",
                  by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
M = M.merge(S[["date", "industry", "E", "G", "P_pct", "heat_pct"]].rename(columns={"date": "week", "industry": "ind_s"}),
            left_on=["week", M["symbol"].map(imap).rename("ind_s")], right_on=["week", "ind_s"], how="left")
feats += ["E", "G", "P_pct"]
M["band"] = pd.cut(M["rank"], [0, 9, 25, 1e9], labels=["1-9", "10-25", "26+"])
M["year"] = M["week"].dt.year
print(f"DISCOVERY ERA · pool rows {len(M)} · hitters {int(M['hit'].sum())} ({M['hit'].mean():.1%}) · features {len(feats)}")

def spread(df, f):
    x = df[[f, "hit"]].dropna()
    if len(x) < 100 or x[f].nunique() < 3:
        return np.nan, 0
    if x[f].nunique() <= 3 or (x[f] == 0).mean() > 0.6:          # sparse / event counts: present vs absent
        hi, lo = x[x[f] > 0], x[x[f] <= 0]
    else:
        q = x[f].rank(pct=True); hi, lo = x[q > 0.8], x[q <= 0.2]
    if min(len(hi), len(lo)) < 30:
        return np.nan, 0
    return 100 * (hi["hit"].mean() - lo["hit"].mean()), min(len(hi), len(lo))

out = []
for f in feats:
    by_band = [spread(M[M["band"] == b], f) for b in ("1-9", "10-25", "26+")]
    vals = [v for v, n in by_band if np.isfinite(v)]
    if len(vals) < 2:
        continue
    yrs = [spread(M[M["year"] == y], f)[0] for y in (2019, 2020, 2021, 2022)]
    avg = float(np.mean(vals)); sgn = np.sign(avg)
    n_small = min(n for v, n in by_band if np.isfinite(v))
    se2 = 200 * np.sqrt(0.3 * 0.7 * 2 / max(n_small, 1))
    out.append(dict(feature=f, cov=round(100 * M[f].notna().mean()), spread_within_rank=round(avg, 1),
                    bought_1_9=round(by_band[0][0], 1), r10_25=round(by_band[1][0], 1), r26plus=round(by_band[2][0], 1),
                    years_agree=int(sum(np.sign(y) == sgn for y in yrs if np.isfinite(y))), two_se=round(se2, 1)))
T = pd.DataFrame(out).assign(a=lambda t: t["spread_within_rank"].abs()).sort_values("a", ascending=False).drop(columns="a")
pd.set_option("display.width", 200)
print("\nSpread = hit-rate points, top fifth (or event present) minus bottom fifth (or absent), same rank band")
print(T.head(18).to_string(index=False))
T.to_csv(ROOT / "logs/leader_sleeve/hitter_traits_disc.csv", index=False)

# 2026-09-30 follow-up: achievers vs non-achievers side by side, and is it only volatility (do the same traits also
# crash more)? ret = close at session 126 / next-open entry - 1.
M["prom_buy_any"] = (M["prom_buys90"] > 0).astype(float)
prof = ["mcap_cr", "adv", "avg_delivery_pct_20d", "off_low", "ret252", "px_sma200", "promoter_pct", "prom_buy_any",
        "pe", "eps_yoy", "sales_yoy", "rsi_14_daily", "ind_heat_pct", "volume_vs_60d", "days_since_results", "rank"]
for name, df in (("WHOLE POOL", M), ("BOUGHT (rank 1-9)", M[M["rank"] <= 9])):
    g = df.groupby("hit")[prof].median().T
    g.columns = ["non-achievers", "achievers"]
    g.loc["prom_buy_any"] = df.groupby("hit")["prom_buy_any"].mean().values
    print(f"\nPROFILE, {name}: medians (prom_buy_any = share with a promoter market purchase in 90d) · n "
          f"{int((~df['hit']).sum())} non / {int(df['hit'].sum())} achievers")
    print(g.round(3).to_string())
print("\nIS IT JUST SWING? same trait groups: +50% touch vs a -20% loss at the 126-session close")
for f, hi_is in (("mcap_cr", "small"), ("avg_delivery_pct_20d", "high"), ("off_low", "high"), ("ret252", "high"), ("prom_buy_any", "yes")):
    for band, df in (("bought", M[M["rank"] <= 9]), ("pool", M)):
        x = df.dropna(subset=[f])
        if f == "prom_buy_any":
            a, b = x[x[f] > 0], x[x[f] <= 0]
        else:
            q = x[f].rank(pct=True)
            a, b = (x[q <= 0.2], x[q > 0.8]) if hi_is == "small" else (x[q > 0.8], x[q <= 0.2])
        row = lambda y: f"n {len(y):4d} · +50% {100 * y['hit'].mean():5.1f}% · -20% loss {100 * (y['ret'] <= -0.2).mean():5.1f}% · median {100 * y['ret'].median():+6.1f}% · mean {100 * y['ret'].mean():+6.1f}%"
        print(f"{f:22s} {band:7s} {hi_is:>5s}: {row(a)}")
        print(f"{'':22s} {band:7s} {'other':>5s}: {row(b)}")

# 2026-09-30 follow-up 2 ("delivery % ka trend bhi hoga"): longer delivery trends from the daily panel, as of the screen
# date's close (the screen runs after the close; entry is the next open):
#   del_trend60   20-day average delivery % now minus 60 sessions earlier (more of the volume being taken home)
#   delqty_trend  20-day average delivered quantity now / 60 sessions earlier (more shares being taken home)
#   del_updown60  average delivery % on up-close days minus on down-close days, last 60 sessions (buyers keep, sellers trade)
#   op_leverage   EPS growth minus sales growth (profit growing faster than sales = margins widening)
DP = rp.load_panel(["close", "delivery_pct", "avg_delivery_pct_20d", "avg_delivery_qty_20d"])
dp, a20, q20, cl = (rp.wide(DP, c, cal) for c in ("delivery_pct", "avg_delivery_pct_20d", "avg_delivery_qty_20d", "close")); del DP
up = cl.pct_change(fill_method=None) > 0; dn = cl.pct_change(fill_method=None) < 0
upd = dp.where(up).rolling(60, min_periods=20).mean() - dp.where(dn).rolling(60, min_periods=20).mean()
new = dict(del_trend60=a20 - a20.shift(60), delqty_trend=q20 / q20.shift(60), del_updown60=upd)
idx = M["week"].map(lambda d: cal.get_loc(d)).to_numpy()
for k, W in new.items():
    col = {s: j for j, s in enumerate(W.columns)}
    A = W.to_numpy(); M[k] = [A[i, col[s]] if s in col else np.nan for i, s in zip(idx, M["symbol"])]
M["op_leverage"] = M["eps_yoy"] - M["sales_yoy"]
print("\nDELIVERY TRENDS + OPERATING LEVERAGE (same method: spread within rank band; years agreeing of 4)")
for f in ["del_trend60", "delqty_trend", "del_updown60", "op_leverage"]:
    by_band = [spread(M[M["band"] == b], f) for b in ("1-9", "10-25", "26+")]
    yrs = [spread(M[M["year"] == y], f)[0] for y in (2019, 2020, 2021, 2022)]
    avg = np.nanmean([v for v, n in by_band]); sgn = np.sign(avg)
    print(f"{f:14s} cov {100 * M[f].notna().mean():3.0f}% · within-rank spread {avg:+5.1f} pts (bought {by_band[0][0]:+5.1f}, "
          f"10-25 {by_band[1][0]:+5.1f}, 26+ {by_band[2][0]:+5.1f}) · years agree {sum(np.sign(y) == sgn for y in yrs if np.isfinite(y))}/4")
    x = M[M["rank"] <= 9].dropna(subset=[f]); q = x[f].rank(pct=True); a, b = x[q > 0.8], x[q <= 0.2]
    row = lambda y: f"+50% {100 * y['hit'].mean():5.1f}% · -20% loss {100 * (y['ret'] <= -0.2).mean():5.1f}% · median {100 * y['ret'].median():+6.1f}%"
    print(f"{'':14s} bought, top fifth: {row(a)}   |   bottom fifth: {row(b)}")
print("\ncorrelation with the model's rank inside the pool (Spearman; ~0 = the model is not using it here):")
print(M[["rank", "avg_delivery_pct_20d", "del_trend60", "del_updown60", "mcap_cr", "prom_buy_any", "op_leverage"]]
      .corr(method="spearman")["rank"].round(2).drop("rank").to_string())
