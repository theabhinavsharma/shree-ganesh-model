"""Sri Lakshmi V3: bought in early October, how did it look around Diwali? (2026-09-30; insight only, no rule.)

Batches: V3 picks (v3_rule.py, the same code as the live screen) for every session Oct 1-14 of 2019-2025
(all five weekly phase offsets, so ~10 overlapping batches a year). Entry = next session open.
Marked at the last session on or before Diwali + 0 / 7 / 14 / 21 / 28 calendar days (Diwali = Lakshmi Puja; 2025 = NSE
Muhurat trading day). Batch return = equal-weight average of its picks, before costs.
Typical stock = median return of every core-band stock over the same entry -> mark window (no index series in the repo).
"""
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import v3_rule  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

DIWALI = {2019: "2019-10-27", 2020: "2020-11-14", 2021: "2021-11-04", 2022: "2022-10-24",
          2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-21"}
MARKS = (0, 7, 14, 21, 28)

D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
ctx = v3_rule.context(imap); fin, fading_only = ctx["fin"], ctx["fading_only"]   # shared V3 rule
days = [d for d in cal if d.month == 10 and d.day <= 14 and d.year in DIWALI]
g1 = tif.select_elig(X["F"], days, imap, P, S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)])
v3 = {d: [s for s in g1.get(d, []) if not fin(s) and not fading_only(s, d)] for d in days}
PX = rp.load_panel(["open", "close"]); O, C = rp.wide(PX, "open", cal), rp.wide(PX, "close", cal); del PX
F = X["F"]; core = F[F["core"]].groupby("trade_date")["symbol"].apply(list)

rows = []
for d in days:
    i0 = cal.get_loc(d) + 1
    dw = pd.Timestamp(DIWALI[d.year])
    for k in MARKS:
        j = cal.searchsorted(dw + pd.Timedelta(days=k), side="right") - 1
        if j < i0:
            continue
        r = [C[s].iloc[j] / O[s].iloc[i0] - 1 for s in v3[d] if s in O.columns and np.isfinite(O[s].iloc[i0])]
        u = [C[s].iloc[j] / O[s].iloc[i0] - 1 for s in core.get(d, []) if s in O.columns and np.isfinite(O[s].iloc[i0]) and O[s].iloc[i0] > 0]
        u = [x for x in u if np.isfinite(x)]
        if r:
            rows.append(dict(year=d.year, entry=d, mark=k, v3=np.nanmean(r), up=np.mean([x > 0 for x in r]), typical=np.median(u) if u else np.nan,
                             mark_date=cal[j]))
R = pd.DataFrame(rows)
lbl = {0: "Diwali", 7: "+1 wk", 14: "+2 wk", 21: "+3 wk", 28: "+4 wk"}
pct = lambda v: f"{v:+.0%}"  # noqa: E731
print(f"early-October entries: {R['entry'].nunique()} batch dates, {R.groupby('entry').size().shape[0]} batches · avg picks per batch "
      f"{np.mean([len(v3[d]) for d in days]):.1f}\n")
Y = R.groupby(["year", "mark"])["v3"].mean().unstack().rename(columns=lbl)
T = R.groupby(["year", "mark"])["typical"].mean().unstack().rename(columns=lbl)
print("V3 batch return from early-Oct entry (average of that year's batches):")
print(Y.to_string(float_format=pct))
print("\nTypical stock over the same windows:")
print(T.to_string(float_format=pct))
A = R.groupby("mark").agg(v3_avg=("v3", "mean"), v3_median=("v3", "median"), typical=("typical", "mean"), picks_up=("up", "mean"),
                          years_positive=("v3", lambda v: 0)).rename(index=lbl)
A["years_positive"] = [f"{(Y[c] > 0).sum()}/{Y[c].notna().sum()}" for c in A.index]
A["years_beat_typical"] = [f"{(Y[c] > T[c]).sum()}/{Y[c].notna().sum()}" for c in A.index]
print("\nAll 7 years:")
print(A.to_string(float_format=pct))
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-09-30-v3-diwali-window-EXPLORATION",
                             status="EXPLORATION for insight only (no rule); uses 2019-2025", producer="src/agentic/report_v3_diwali_window.py",
                             by_year={int(y): {c: round(float(v), 3) for c, v in r.items() if pd.notna(v)} for y, r in Y.iterrows()})) + "\n")
