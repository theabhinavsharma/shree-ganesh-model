"""V3 streak names week by week (8, 9, 10, ... separately), every year (2019-10-10, EXPLORATION for Abhinav: "8 plus mat kar,
check for 8 9 10 11 12 13 14 15 ... median average hits +50% ... har year ke liye"; no rule changed).

Weekly lists: v3_rule.picks on the last session of every ISO week (the live Friday list). Streak = consecutive weekly lists.
For every (list date, stock): weeks on the list k, gain since its first-week entry (first-week entry open -> this list's close),
and the next 126 sessions from the next open: return, touched +50% (max high >= 1.5x entry), ended <= -30%. Before costs.
Rows per year: k = 8, 9, ..., the longest streak; plus a baseline row "1-7" (all picks in weeks 1 to 7 that year).
Output: logs/explorations/v3_streak_by_week/by_week_year.csv (+ manifest, README) and printed tables.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402

H = 126
OUT = ROOT / "logs/explorations/v3_streak_by_week"
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)
s = pd.Series(cal, index=cal); s = s[s >= "2018-09-01"]
fri = sorted(d for d in (g.iloc[-1] for _, g in s.groupby([s.dt.isocalendar().year, s.dt.isocalendar().week])) if d >= pd.Timestamp("2019-01-01"))
V = v3_rule.picks(X["F"], fri, imap, P, ctx)
PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
On, Hn, Cn = O.to_numpy(), Hh.to_numpy(), C.ffill(limit=300).to_numpy(); col = {x: i for i, x in enumerate(O.columns)}

streak, first, rows = {}, {}, []
for d in fri:
    cur = V.get(d, [])
    new = {x: streak.get(x, 0) + 1 for x in cur}
    first = {x: (first[x] if x in first and streak.get(x, 0) > 0 else d) for x in cur}
    streak = new
    di = cal.get_loc(d); i0 = di + 1
    for x in cur:
        c = col.get(x)
        if c is None or i0 + H > len(cal) or not np.isfinite(On[i0, c]) or On[i0, c] <= 0 or not np.isfinite(Cn[i0 + H - 1, c]):
            continue
        f0 = cal.get_loc(first[x]) + 1; e1 = On[f0, c]
        e = On[i0, c]
        rows.append(dict(d=d, year=d.year, s=x, k=streak[x], run=f"{x}|{first[x].date()}", r=Cn[i0 + H - 1, c] / e - 1,
                         hit=bool(np.nanmax(Hn[i0:i0 + H, c]) >= 1.5 * e), gained=(Cn[di, c] / e1 - 1) if np.isfinite(e1) and e1 > 0 else np.nan))
R = pd.DataFrame(rows)
R["wk"] = np.where(R.k <= 7, "1-7", R.k.astype(str))


def stats(g):
    return pd.Series(dict(picks=len(g), stocks=g.s.nunique(), avg=g.r.mean() * 100, median=g.r.median() * 100, hit50=g.hit.mean() * 100,
                          down30=(g.r <= -0.3).mean() * 100, gained_so_far=g.gained.median() * 100 if g.k.min() > 1 else np.nan))


order = ["1-7"] + [str(k) for k in range(8, int(R.k.max()) + 1)]
tabs = {}
for lab, g in [("all years", R), ("2019-22", R[R.year < 2023]), ("2023+", R[R.year >= 2023])] + [(str(y), R[R.year == y]) for y in sorted(R.year.unique())]:
    t = g.groupby("wk").apply(stats).reindex([o for o in order if o in set(g.wk)])
    tabs[lab] = t
    print(f"\n=== {lab} · next 6 months by week on the list (weeks 1-7 = baseline) ===")
    print(t.round(1).to_string())
OUT.mkdir(parents=True, exist_ok=True)
flat = pd.concat({k: v for k, v in tabs.items()}, names=["period", "week_on_list"]).reset_index()
flat.to_csv(OUT / "by_week_year.csv", index=False)
(OUT / "by_week_year.csv.manifest.json").write_text(json.dumps(dict(dataset="by_week_year.csv", producer="src/agentic/explore_v3_streak_by_week.py", status="EXPLORATION (no rule changed)",
    definitions=__doc__, columns=dict(period="all years / 2019-22 / 2023+ / a calendar year (of the list date)", week_on_list="consecutive weeks on the V3 list at the list date; 1-7 = all early weeks",
    picks="list-date x stock rows with a finished 126-session hold", stocks="distinct stocks", avg="mean next-126-session return, percent", median="median return, percent",
    hit50="percent touching +50% within 126 sessions", down30="percent ending <= -30%", gained_so_far="median gain from the stock's first-week entry open to this list's close, percent"),
    caveats=["overlapping trades: a stock in week 8 is usually also in week 9, 10", "small cells: many week x year cells have 0-5 picks", "V3 in-sample; before costs"],
    updated=datetime.now().isoformat(timespec="seconds")), indent=1))
(OUT / "README.md").write_text("# V3 streak names, week by week and year by year (exploration)\n\nNext-6-month outcomes of V3 picks by how many consecutive weekly lists the "
                               "stock had been on (8, 9, 10, ... separately, with weeks 1-7 as the baseline), for every year 2019-2026. by_week_year.csv; definitions in its manifest.\n")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-10-v3-streak-by-week-EXPLORATION", status="EXPLORATION (no rule changed)",
                             producer="src/agentic/explore_v3_streak_by_week.py", output=str(OUT.relative_to(ROOT)),
                             all_years={k: {c: round(float(v), 1) for c, v in r.items()} for k, r in tabs["all years"].iterrows()}), default=str) + "\n")
