"""Block-bootstrap confidence ranges for the gap between two arms (daily NAV log-return differences, 126-session blocks)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src/agentic")
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule, test_industry_fundamentals as tif
from test_v3_l1_newlist_cost import run_cost
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
PC = pd.read_parquet("logs/leader_sleeve/model_real_goal/model_C/bakeoff_preds.parquet", columns=["symbol", "trade_date", "ensemble"])
PC["trade_date"] = pd.to_datetime(PC["trade_date"]).astype("datetime64[ns]"); PC = PC.dropna().sort_values("trade_date")
L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy()
t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0
cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))
def nav(picks, wk): return run_cost(D, X, picks, wk, "E0", cost)[0]
def boot(a, b, block=126, n=4000, seed=7):
    d = (np.log(a).diff() - np.log(b).diff()).dropna().to_numpy(); T = len(d)
    spy = T / ((a.index[-1] - a.index[0]).days / 365.25); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block))
    out = np.array([d[((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T].mean() * spy for _ in range(n)]) * 100
    obs = d.mean() * spy * 100
    return obs, np.percentile(out, 2.5), np.percentile(out, 97.5), (out > 0).mean(), T / block
rows = []
for o in range(5):
    w19 = [d for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp("2019-04-01")]; w21 = [d for d in w19 if d >= pd.Timestamp("2021-01-01")]
    v31 = v3_rule.picks_v31(F1, w19, imap, P, ctx); v32 = {d: [s for s in l if not t2t(s, d)] for d, l in v31.items()}
    keep = sre.TOPN; sre.TOPN = 9; base = tif.select_elig(X["F"], w19, imap, P, ctx["G1"]); sre.TOPN = keep
    base = {d: base.get(d, []) for d in w19}
    c32 = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, w21, imap, PC, ctx).items()}
    n32, n31, nb = nav(v32, w19), nav(v31, w19), nav(base, w19)
    n32b, nc = nav({d: v32[d] for d in w21}, w21), nav(c32, w21)
    for name, a, b in (("V3.2 vs no extra rules (G1), Apr 2019+", n32, nb), ("V3.2 vs V3.1 (trade-for-trade drop), Apr 2019+", n32, n31),
                       ("2019-trained model vs live model, both V3.2 rules, 2021+", nc, n32b)):
        obs, lo, hi, pos, nper = boot(a, b); rows.append(dict(cmp=name, o=o, gap=obs, lo=lo, hi=hi, pos=pos, periods=nper))
R = pd.DataFrame(rows)
for name, g in R.groupby("cmp", sort=False):
    g0 = g[g.o == 0].iloc[0]
    print(f"{name}\n   start week 0: gap {g0.gap:+.1f} pts/yr · 95% range {g0.lo:+.1f} to {g0.hi:+.1f} · bootstrap draws > 0: {g0.pos:.0%} · ~{g0.periods:.0f} independent 6-month periods"
          f"\n   all 5 start weeks: gap {g.gap.min():+.1f}..{g.gap.max():+.1f} · draws > 0: {g.pos.min():.0%}..{g.pos.max():.0%} · 95% range excludes 0 in {int((g.lo > 0).sum() + (g.hi < 0).sum())}/5")
