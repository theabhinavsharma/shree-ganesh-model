"""Exploration (discovery era 2019-2022 ONLY; 2023+ sealed): is there a loss level where a Sri Lakshmi v2 pick is
practically dead? For each level X (close <= entry x (1 - X) at any point in the 126-session hold): how many picks got
there, how many still reached +50% over entry AFTER that day, how many still ended in profit, and the average of
"hold to session 126" vs "sell at that close" (0.5% cost either way). Also: how many eventual 2x winners the stop
would have thrown out. Entry = next open; closes / highs from the adjusted panel.
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
wk = [d for d in sp.weekly_grid(cal, 0) if pd.Timestamp(tif.START) <= d < pd.Timestamp("2023-01-01")]
g1 = tif.select_elig(X["F"], wk, imap, P, S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)])
PX = rp.load_panel(["open", "high", "close"]); O, H, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
paths = []
for d in wk:
    i0 = cal.get_loc(d) + 1
    if i0 + 126 > len(cal):
        continue
    for s in g1.get(d, []):
        if sector.get(imap.get(s)) == "Financial Services" or s not in O.columns:
            continue
        e = O[s].iloc[i0]
        if np.isfinite(e):
            paths.append((s, d, C[s].iloc[i0:i0 + 126].to_numpy() / e, H[s].iloc[i0:i0 + 126].to_numpy() / e))
print(f"v2 picks 2019-2022 with a full 126-session path: {len(paths)}")
rows = []
for X_ in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
    hit_after, profit, hold, sell, trig, killed_2x, day = 0, 0, [], [], 0, 0, []
    total_2x = 0
    for s, d, c, h in paths:
        big = np.nanmax(h) >= 2.0
        total_2x += big
        idx = np.where(c <= 1 - X_)[0]
        if len(idx) == 0:
            continue
        k = idx[0]; trig += 1; day.append(k + 1)
        hit_after += np.nanmax(h[k + 1:]) >= 1.5 if k + 1 < len(h) else False
        final = c[-1] if np.isfinite(c[-1]) else c[np.isfinite(c)][-1]
        profit += final >= 1.0
        hold.append(final - 1 - 0.005); sell.append(c[k] - 1 - 0.005)
        killed_2x += big
    rows.append({"stop at": f"-{X_:.0%}", "picks that got there": f"{trig} ({trig / len(paths):.0%})",
                 "typical day it happened": int(np.median(day)) if day else None,
                 "still hit +50% later": f"{hit_after / max(trig, 1):.1%}", "still ended in profit": f"{profit / max(trig, 1):.0%}",
                 "avg if held": f"{np.mean(hold):+.1%}", "avg if sold": f"{np.mean(sell):+.1%}",
                 "2x winners thrown out": f"{killed_2x} of {total_2x}"})
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).to_string(index=False))
