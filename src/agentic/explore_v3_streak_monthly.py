"""V3 long-streak names, tested the way they would be used live (2026-10-10, EXPLORATION for Abhinav: "go, run 1 aur 2 and run it
for 20 different starting scenarios of 1st of every month"; no rule changed).

Weekly lists: v3_rule.picks on the last session of every ISO week (the live Friday list; Thursday when Friday is a holiday).
Streak = consecutive weekly lists a stock has been on, counted at each list.
1  Stress test of "week 8+ returns +41% in 2023+": week-8+ picks by entry year, distinct runs, share of the average from the top
   5 stocks, and a 95% range from a bootstrap that resamples whole runs (a stock's consecutive weeks move together).
2  Today's candidates: the latest list (data through the last session) and every stock's current streak, gain since its
   first week (first-week entry open -> latest close), industry, and whether it was in the Oct 1 / Oct 9 live batches.
3  Monthly starts: on the first session of each month, take the latest weekly list known before it; buy at that session's
   open the names on a current streak of 8-12 weeks (STREAK), separately the names on weeks 1-7 (FRESH) and 13+ (LONG);
   hold 126 sessions; equal money per name. Market = average of every priced stock over the same dates. Before costs.
Output: printed tables + an EXPLORATION line in logs/experiments.jsonl.
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
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)
s = pd.Series(cal, index=cal)
fri = [g.iloc[-1] for _, g in s[s >= "2018-09-01"].groupby([s[s >= "2018-09-01"].dt.isocalendar().year, s[s >= "2018-09-01"].dt.isocalendar().week])]
fri = sorted(d for d in fri if d >= pd.Timestamp("2019-01-01"))
V = v3_rule.picks(X["F"], fri, imap, P, ctx)
live = {}
for f in sorted((ROOT / "logs/sri_lakshmi").glob("screen_2026*.json")):
    j = json.loads(f.read_text())
    names = list(j.get("names") or [])
    live[str(j.get("data_through"))[:10]] = names
for dt, names in live.items():
    if names and pd.Timestamp(dt) in V:
        print(f"wiring check {dt}: weekly list {'equals' if V[pd.Timestamp(dt)] == names else 'DIFFERS from'} the live batch ({len(names)} names)")

PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
On, Hn, Cn = O.to_numpy(), Hh.to_numpy(), C.ffill(limit=300).to_numpy(); col = {s_: i for i, s_ in enumerate(O.columns)}
allret = {}


def fwd(c, i0):
    if c is None or i0 + H > len(cal) or not np.isfinite(On[i0, c]) or On[i0, c] <= 0 or not np.isfinite(Cn[i0 + H - 1, c]):
        return None
    e = On[i0, c]; return Cn[i0 + H - 1, c] / e - 1, bool(np.nanmax(Hn[i0:i0 + H, c]) >= 1.5 * e)


def market(i0):
    if i0 not in allret:
        e, x = On[i0], Cn[i0 + H - 1]; ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
        allret[i0] = float(np.mean(np.clip(x[ok] / e[ok] - 1, -1, 10)))
    return allret[i0]


# streaks per weekly list
streak, first, ST = {}, {}, {}
for d in fri:
    cur = V.get(d, [])
    new = {x: streak.get(x, 0) + 1 for x in cur}
    first = {x: (first[x] if x in first and streak.get(x, 0) > 0 else d) for x in cur}
    streak = new; ST[d] = {x: (streak[x], first[x]) for x in cur}

# ---- 1 stress test
rows = []
for d in fri:
    i0 = cal.get_loc(d) + 1
    for x, (k, f0) in ST[d].items():
        r = fwd(col.get(x), i0) if i0 < len(cal) else None
        if r and k >= 8:
            rows.append(dict(d=d, s=x, k=k, first=f0, r=r[0], hit=r[1]))
W = pd.DataFrame(rows); W["year"] = W.d.dt.year; W["run"] = W.s + "|" + W["first"].astype(str)
print("\n=== 1) week-8+ picks (weekly Friday lists), next 6 months from next open, by entry year")
for y, g in W.groupby("year"):
    print(f"  {y}: picks {len(g):3d} · separate streaks {g.run.nunique():2d} · stocks {g.s.nunique():2d} · avg {g.r.mean()*100:+6.1f}% · median {g.r.median()*100:+6.1f}% · "
          f"hit +50% {g.hit.mean()*100:4.0f}% · down 30%+ {(g.r <= -0.3).mean()*100:4.0f}%")
rng = np.random.default_rng(7)
for lab, g in (("2023+", W[W.year >= 2023]), ("2019-22", W[W.year < 2023])):
    runs = g.groupby("run").r.apply(list).tolist()
    boots = [np.mean(np.concatenate([runs[i] for i in rng.integers(0, len(runs), len(runs))])) for _ in range(4000)]
    by = g.groupby("s").r.sum().sort_values(ascending=False)
    top5 = by.head(5)
    print(f"  {lab}: avg {g.r.mean()*100:+.1f}% · 95% range (resampling whole streaks) {np.percentile(boots, 2.5)*100:+.1f}% .. {np.percentile(boots, 97.5)*100:+.1f}% · "
          f"top 5 stocks give {top5.sum() / g.r.sum() * 100:.0f}% of the total gain ({', '.join(top5.index)}) · avg without them {g[~g.s.isin(top5.index)].r.mean()*100:+.1f}%")

# ---- 2 today's candidates
last = fri[-1]
print(f"\n=== 2) today's list (data through {last.date()}) and every name's current streak")
inb = {k: set(v) for k, v in live.items()}
for x, (k, f0) in sorted(ST[last].items(), key=lambda kv: -kv[1][0]):
    c = col.get(x); i_f = cal.get_loc(f0) + 1
    g = Cn[cal.get_loc(last), c] / On[i_f, c] - 1 if c is not None and i_f < len(cal) and np.isfinite(On[i_f, c]) else np.nan
    bat = [b for b, v in inb.items() if x in v]
    print(f"  {x:12s} week {k:2d} on the list (first {f0.date()}) · gained since first-week entry {g*100:+6.1f}% · {imap.get(x, '-')[:34]:34s} · live batches {bat or '-'}")
recent = pd.DataFrame([dict(d=d, s=x, k=k) for d in fri[-16:] for x, (k, _) in ST[d].items()])
peak = recent.groupby("s").k.max().sort_values(ascending=False)
print("  longest streaks seen in the last 16 weekly lists: " + ", ".join(f"{x} {k}" for x, k in peak.head(12).items()))

# ---- 3 monthly starts
months = pd.date_range("2019-03-01", cal[-1], freq="MS")
out = []
for m in months:
    i_dec = int(cal.searchsorted(m))
    if i_dec + H > len(cal):
        continue
    known = [d for d in fri if d < cal[i_dec]]
    if not known:
        continue
    L = ST[known[-1]]
    mk = market(i_dec)
    row = dict(month=m.strftime("%Y-%m"), market=mk * 100)
    for lab, cond in (("STREAK", lambda k: 8 <= k <= 12), ("FRESH", lambda k: k <= 7), ("LONG", lambda k: k >= 13)):
        rs = [fwd(col.get(x), i_dec) for x, (k, _) in L.items() if cond(k)]
        rs = [r for r in rs if r]
        row[lab + "_n"] = len(rs)
        row[lab + "_avg"] = np.mean([r for r, _ in rs]) * 100 if rs else np.nan
        row[lab + "_hit"] = np.mean([h for _, h in rs]) * 100 if rs else np.nan
        row[lab + "_names"] = ",".join(x for x, (k, _) in L.items() if cond(k)) if lab == "STREAK" else ""
    out.append(row)
M = pd.DataFrame(out)
last20 = M.tail(20)
print("\n=== 3) the 20 latest monthly starts with a finished 6-month hold · next 6 months, equal money per name, before costs")
print(f" {'start':8s} {'streak 8-12: n':>15s} {'avg':>7s} {'hit':>5s}   {'fresh 1-7: n':>13s} {'avg':>7s}   {'13+: n':>7s} {'avg':>7s}   {'market':>7s}   streak names")
for r in last20.itertuples():
    print(f" {r.month:8s} {r.STREAK_n:15d} {r.STREAK_avg:+7.1f} {r.STREAK_hit:5.0f}   {r.FRESH_n:13d} {r.FRESH_avg:+7.1f}   {r.LONG_n:7d} {r.LONG_avg:+7.1f}   {r.market:+7.1f}   {r.STREAK_names}")
for lab, g in (("last 20 starts", last20), ("all starts 2019-", M), ("starts 2023-", M[M.month >= "2023"]), ("starts 2019-22", M[M.month < "2023"])):
    s_ok = g.dropna(subset=["STREAK_avg"])
    print(f" {lab:17s}: months with streak names {len(s_ok)}/{len(g)} · streak avg {s_ok.STREAK_avg.mean():+.1f}% (median month {s_ok.STREAK_avg.median():+.1f}%) · "
          f"beat fresh in {(s_ok.STREAK_avg > s_ok.FRESH_avg).mean()*100:.0f}% · beat market in {(s_ok.STREAK_avg > s_ok.market).mean()*100:.0f}% · lost money in {(s_ok.STREAK_avg < 0).mean()*100:.0f}% · "
          f"fresh avg {g.FRESH_avg.mean():+.1f}% · market avg {g.market.mean():+.1f}%")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-10-v3-streak-monthly-EXPLORATION", status="EXPLORATION (no rule changed)",
                             producer="src/agentic/explore_v3_streak_monthly.py", by_year={int(y): dict(picks=len(g), avg=round(g.r.mean() * 100, 1), median=round(g.r.median() * 100, 1)) for y, g in W.groupby("year")},
                             last20=last20[["month", "STREAK_n", "STREAK_avg", "FRESH_avg", "market"]].round(1).to_dict(orient="records")), default=str) + "\n")
