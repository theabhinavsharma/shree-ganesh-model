"""V3 at different hold lengths, with 2023+ shown separately (2026-10-04, EXPLORATION for Abhinav: "v3 post 2023 kaisa hai
for diff hold periods?"; no rule changed — V3's registered hold is 126 sessions).

Portfolio: the registered ladder engine (sim_screen_rank_exit.run_exit, rule E0 = time exit) with HOLD set to 63 / 126 /
189 / 252 / 378 / 504 sessions; the engine sizes the ladder to the hold (sp.slots_for: weekly slots = hold / 5), so each
weekly batch gets 1/slots of the money. Phase 0 and the average over the 5 weekly entry phases.
Per pick: V3 picks entered in 2023 or later (phase 0, next open): average and median return at the hold's last close,
share touching +50% at any time within the hold, share ending <= -30%, and the equal-weight average of all stocks
priced on the entry day over the same dates (the market). Before costs. A pick counts only if its hold has ended.
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

D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)
HOLDS = {"3m": 63, "6m": 126, "9m": 189, "12m": 252, "18m": 378, "24m": 504}
picks = {o: (lambda wk: (wk, v3_rule.picks(X["F"], wk, imap, P, ctx)))([d for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp("2019-01-01")]) for o in range(5)}
out = {"portfolio": {}, "per_pick_2023plus": {}, "calendar_years": {}, "per_pick_by_year": {}}
YEARS = {}
print("=== V3 as a portfolio (ladder sized to the hold) · CAGR %/yr: all · 2019-22 · 2023+ · max drawdown · phase 0 | 5-phase average ===")
for h, n in HOLDS.items():
    sre.HOLD = n
    rs, navs = [], []
    for o in range(5):
        wk, pk = picks[o]
        nav, _ = sre.run_exit(D, X, pk, wk, "E0"); rs.append(sp.metrics(nav)); navs.append(nav / nav.iloc[0])
    m0 = rs[0]; av = {k: float(np.mean([r[k] for r in rs])) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}
    yr = pd.DataFrame([(lambda yl: (yl / yl.shift(1).fillna(1.0) - 1) * 100)(nv.groupby(nv.index.year).last()) for nv in navs]).mean()
    YEARS[h] = yr
    out["portfolio"][h] = dict(phase0={k: round(float(m0[k]), 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, avg5={k: round(v, 1) for k, v in av.items()})
    print(f" {h:>3s}: {m0['cagr']:5.1f} · {m0['cagr_disc']:5.1f} · {m0['cagr_conf']:5.1f} · {m0['maxdd']:6.1f}   |   {av['cagr']:5.1f} · {av['cagr_disc']:5.1f} · {av['cagr_conf']:5.1f} · {av['maxdd']:6.1f}")
print("\n=== V3 portfolio, calendar-year return % by hold (average of the 5 weekly entry phases; 2026 = to date) ===")
print(" hold " + "".join(f"{y:>8d}" for y in range(2019, 2027)))
for h, yr in YEARS.items():
    out["calendar_years"][h] = {int(y): round(float(v), 1) for y, v in yr.items()}
    print(f" {h:>4s} " + "".join(f"{yr.get(y, np.nan):+8.0f}" for y in range(2019, 2027)))
sre.HOLD = 126
PX = rp.load_panel(["open", "high", "close"]); O = rp.wide(PX, "open", cal); cols = {s: i for i, s in enumerate(O.columns)}
On, Hn, Cn = O.to_numpy(), rp.wide(PX, "high", cal).to_numpy(), rp.wide(PX, "close", cal).ffill(limit=300).to_numpy(); del PX
mk = {}
print("\n=== V3 picks entered 2023+ (phase 0) · avg (median) · touched +50% · ended <= -30% · market avg · [n] ===")
wk0, pk0 = picks[0]
for h, n in HOLDS.items():
    r, hit = [], []; mm = []
    for d in wk0:
        if d < pd.Timestamp("2023-01-01"):
            continue
        i0 = cal.get_loc(d) + 1
        if i0 + n > len(cal):
            continue
        if (i0, n) not in mk:
            e, x = On[i0], Cn[i0 + n - 1]; ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
            mk[(i0, n)] = float(np.mean(np.clip(x[ok] / e[ok] - 1, -1, 10)))
        for s in pk0[d]:
            c = cols.get(s)
            if c is None or not np.isfinite(On[i0, c]) or On[i0, c] <= 0 or not np.isfinite(Cn[i0 + n - 1, c]):
                continue
            e = On[i0, c]; r.append(Cn[i0 + n - 1, c] / e - 1); hit.append(np.nanmax(Hn[i0:i0 + n, c]) >= 1.5 * e); mm.append(mk[(i0, n)])
    r = np.array(r)
    if len(r):
        out["per_pick_2023plus"][h] = dict(n=len(r), avg=round(float(r.mean()), 3), median=round(float(np.median(r)), 3), hit50=round(float(np.mean(hit)), 3),
                                          lost30=round(float((r <= -0.3).mean()), 3), market=round(float(np.mean(mm)), 3))
        print(f" {h:>3s}: {r.mean():+5.0%} ({np.median(r):+5.0%}) · {np.mean(hit):4.0%} · {(r <= -0.3).mean():4.0%} · market {np.mean(mm):+5.0%} · [{len(r)}]")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-04-v3-holds-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_v3_holds.py", results=out)) + "\n")

print("\n=== V3 picks by ENTRY YEAR (phase 0) · avg (median) · touched +50% · ended <= -30% · market avg · [n] — a pick counts once its hold has ended ===")
for y in (2023, 2024, 2025, 2026):
    print(f" entered {y}:")
    for h, n in HOLDS.items():
        r, hit, mm = [], [], []
        for d in wk0:
            if d.year != y:
                continue
            i0 = cal.get_loc(d) + 1
            if i0 + n > len(cal):
                continue
            if (i0, n) not in mk:
                e, x = On[i0], Cn[i0 + n - 1]; ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
                mk[(i0, n)] = float(np.mean(np.clip(x[ok] / e[ok] - 1, -1, 10)))
            for s in pk0[d]:
                c = cols.get(s)
                if c is None or not np.isfinite(On[i0, c]) or On[i0, c] <= 0 or not np.isfinite(Cn[i0 + n - 1, c]):
                    continue
                e = On[i0, c]; r.append(Cn[i0 + n - 1, c] / e - 1); hit.append(np.nanmax(Hn[i0:i0 + n, c]) >= 1.5 * e); mm.append(mk[(i0, n)])
        r = np.array(r)
        if len(r):
            out["per_pick_by_year"][f"{y}|{h}"] = dict(n=len(r), avg=round(float(r.mean()), 3), median=round(float(np.median(r)), 3),
                                                       hit50=round(float(np.mean(hit)), 3), lost30=round(float((r <= -0.3).mean()), 3), market=round(float(np.mean(mm)), 3))
            print(f"   {h:>3s}: {r.mean():+5.0%} ({np.median(r):+5.0%}) · {np.mean(hit):4.0%} · {(r <= -0.3).mean():4.0%} · market {np.mean(mm):+5.0%} · [{len(r)}]")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-04-v3-holds-by-year-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_v3_holds.py", results=out)) + "\n")
