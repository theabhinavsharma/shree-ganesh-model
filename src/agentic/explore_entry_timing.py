"""Exploration (2026-10-12): how much of the V3.1 / V3.2 backtest comes from buying at the signal-day (Friday) close, which
the live process cannot do? Re-runs test_v3_l1_newlist_cost.run_cost with the first day replaced by the next open
(D["first_fac"]) and names locked at the upper band at that open skipped. Result 2026-10-12: V3.1 49.6 vs 51.8 %/yr,
V3.2 54.1 vs 56.7 (2023+ 61.0 vs 64.2); V3.2 still beats V3.1 at the open.
"""
import sys; sys.path.insert(0,'src/agentic')
import numpy as np, pandas as pd, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule, research_panel as rp
import test_v3_l1_newlist_cost as T
src = open('src/agentic/test_v3_l1_newlist_cost.py').read()
fn = src[src.index('def run_cost('):src.index('def main()')]
fn = fn.replace('def run_cost(D, X, picks, weekly, rule, costs_of):', 'def run_cost_open(D, X, picks, weekly, rule, costs_of):')
a = '                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])'
fn = fn.replace(a, a + '\n                seg[0] = np.nan_to_num(D["first_fac"][i0, cols] - 1.0)   # bought at the next open, not the signal-day close')
a2 = '            names = [n for n in picks.get(d, []) if n in cidx]'
fn = fn.replace(a2, a2 + '\n            names = [n for n in names if not D["locked"][i0, cidx[n]] and np.isfinite(D["first_fac"][i0, cidx[n]])]   # locked at the upper band at the open: not bought')
ns = dict(np=np, pd=pd, sp=sp, sre=sre); exec(fn, ns); run_open = ns['run_cost_open']
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
L = rp.load_panel(["avg_traded_value_20d", "series"]); L["adv"] = L["avg_traded_value_20d"] / 1e7; L['t2t'] = L.series.isin(['BE','BZ']).astype(float)
ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); TW = rp.wide(L, 't2t', cal).ffill(); tcol = {s: i for i, s in enumerate(TW.columns)}; TWn = TW.to_numpy(); del L
cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))
YEARS = (2023, 2024, 2025, 2026)
def m(nav):
    mm = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
    return [mm['cagr'], mm['cagr_disc'], mm['cagr_conf'], mm['maxdd']] + [float(r.get(y, np.nan)) for y in YEARS]
res = {}; lockinfo = []
for o in range(5):
    wk = [d for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp("2019-01-01")]
    l31 = v3_rule.picks_v31(F1, wk, imap, P, ctx); pos = D['dpos']
    l32 = {d: [s for s in v if not (s in tcol and TWn[pos[d], tcol[s]] == 1.0)] for d, v in l31.items()}
    if o == 0:
        for lab, lst in (('V3.1', l31), ('V3.2', l32)):
            n = sum(len(v) for v in lst.values()); lk = sum(1 for d, v in lst.items() for s in v if s in D['cidx'] and D['locked'][pos[d], D['cidx'][s]])
            gaps = [D['first_fac'][pos[d], D['cidx'][s]] for d, v in lst.items() for s in v if s in D['cidx']]
            lockinfo.append(f"{lab}: {lk} of {n} picks locked at the upper band at the next open")
    for lab, lst in (('V3.1', l31), ('V3.2', l32)):
        res.setdefault((lab, 'Friday close (backtest)'), []).append(m(T.run_cost(D, X, lst, wk, "E0", cost)[0]))
        res.setdefault((lab, 'Monday open (real)'), []).append(m(run_open(D, X, lst, wk, "E0", cost)[0]))
print("; ".join(lockinfo))
print("arm | entry | CAGR | 2019-22 | 2023+ | maxDD (schedule 0) | years 2023/2024/2025/2026 (avg of 5) | CAGR range")
for (lab, e), v in res.items():
    v0 = v[0]; yrs = np.nanmean(np.array(v)[:, 4:], axis=0)
    print(f"{lab} | {e} | {v0[0]:.1f} | {v0[1]:.1f} | {v0[2]:.1f} | {v0[3]:.1f} | " + " / ".join(f"{x:+.0f}" for x in yrs) + f" | {min(x[0] for x in v):.1f}..{max(x[0] for x in v):.1f}")
