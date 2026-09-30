"""Per-pick outcome stats for Sri Lakshmi versions (2026-09-30): G1 (old), V2 (financials dropped), V3 (V2 + skip
stocks tied only to fading themes). Weekly picks 2019+, phase 0, entry = next open, 126-session hold.
Per pick: peak = best high in the hold / entry - 1; final = close at session 126 / entry - 1; trough = worst low / entry - 1.
Only picks whose 126 sessions are complete are counted. Returns are before costs.
"""
import html, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, test_industry_fundamentals as tif
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet"); sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"])
sc["industry"] = sc["industry"].map(html.unescape); sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
fin = lambda s: sector.get(imap.get(s)) == "Financial Services"  # noqa: E731
I = pd.read_parquet(ROOT / "data/derived/theme_intensity.parquet"); I["date"] = pd.to_datetime(I["date"])
state = {(d, t): st for d, t, st in zip(I["date"], I["theme"], I["state"])}
E = pd.read_parquet(ROOT / "data/derived/theme_exposure.parquet"); E["an_dt"] = pd.to_datetime(E["an_dt"])
ex = {(s, t): np.sort(g["an_dt"].to_numpy()) for (s, t), g in E.groupby(["symbol", "theme"])}
tof = {}
for (s, t) in ex: tof.setdefault(s, []).append(t)
def fading_only(s, d):
    d64 = np.datetime64(d + pd.Timedelta(hours=23, minutes=59)); st = []
    for t in tof.get(s, []):
        a = ex[(s, t)]
        if np.searchsorted(a, d64, side="right") - np.searchsorted(a, d64 - np.timedelta64(365, "D"), side="right") > 0:
            st.append(state.get((d, t), "n/a"))
    return "fading" in st and "rising" not in st
wk = [d for d in sp.weekly_grid(cal, 0) if d >= pd.Timestamp(tif.START)]
g1 = tif.select_elig(X["F"], wk, imap, P, S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)])
arms = {"G1 (old)": g1, "V2": {d: [s for s in g1.get(d, []) if not fin(s)] for d in wk}}
arms["V3"] = {d: [s for s in arms["V2"][d] if not fading_only(s, d)] for d in wk}
PX = rp.load_panel(["open", "high", "low", "close"]); O, H, L, C = (rp.wide(PX, c, cal) for c in ("open", "high", "low", "close")); del PX
rows = []
for arm, pk in arms.items():
    for d in wk:
        i0 = cal.get_loc(d) + 1
        if i0 + 126 > len(cal):
            continue
        for s in pk.get(d, []):
            e = O[s].iloc[i0] if s in O.columns else np.nan
            if not np.isfinite(e):
                continue
            hh, ll = H[s].iloc[i0:i0 + 126].to_numpy(), L[s].iloc[i0:i0 + 126].to_numpy()
            rows.append(dict(arm=arm, week=d, symbol=s, peak=np.nanmax(hh) / e - 1, trough=np.nanmin(ll) / e - 1, final=C[s].iloc[i0 + 125] / e - 1))
R = pd.DataFrame(rows); R["era"] = np.where(R["week"] < pd.Timestamp("2023-01-01"), "2019-22", "2023+")
def stats(x):
    b = x.loc[x["peak"].idxmax()]; w = x.loc[x["final"].idxmin()]; bf = x.loc[x["final"].idxmax()]
    return pd.Series({"picks": len(x), "hit +50%": f"{(x['peak'] >= 0.5).mean():.0%}", "hit 2x": f"{(x['peak'] >= 1.0).mean():.0%}",
                      "ended +50%": f"{(x['final'] >= 0.5).mean():.0%}", "ended in loss": f"{(x['final'] < 0).mean():.0%}",
                      "ended -20% or worse": f"{(x['final'] <= -0.2).mean():.0%}", "median final": f"{x['final'].median():+.0%}",
                      "average final": f"{x['final'].mean():+.0%}",
                      "best peak": f"{b['symbol']} {b['peak']:+.0%} ({b['week']:%b %Y})", "best final": f"{bf['symbol']} {bf['final']:+.0%}",
                      "worst final": f"{w['symbol']} {w['final']:+.0%}"})
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 40)
for era in ("2019-22", "2023+", "all"):
    x = R if era == "all" else R[R["era"] == era]
    print(f"\n=== {era} ===\n" + x.groupby("arm", sort=False).apply(stats).T.to_string())
