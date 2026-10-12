"""Does a V3.2 year show itself early? (2026-10-13; Abhinav: 'by week 13-15 and by June-July we get to know how the year is going
to perform?'). EXPLORATION, no rule changed.

A fresh start at EVERY weekly list from Apr 2019 to a year before the data ends (schedule 3): Rs 1 lakh into each list for 26
lists, lots sold after 126 sessions and rolled (as explore_v32_yearly_growth.py). Return on money put in at weeks 13, 15, 26, 30
and 52; how well the early return predicts week 52 (Spearman rank correlation; outcomes grouped by the early return). Starts one
week apart overlap almost entirely, so the effective sample is about one per year: read the numbers as a guide.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
CHECK = (13, 15, 26, 30, 52)


def main():
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy()
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")].replace("return pd.Series(nav, index=cal[start_i:]),", "return V, start_i, slots,")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(fn, ns); run_slots = ns["run_cost"]
    grid = [d for d in sp.weekly_grid(cal, 3) if d >= pd.Timestamp("2019-04-01")]
    picks = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, grid, imap, P, ctx).items()}
    rows = []
    for k0 in range(0, len(grid) - 52):
        wk = grid[k0:k0 + 52]
        V, si, slots, _ = run_slots(D, X, {d: picks[d] for d in wk}, wk, "E0", cost)
        first = {s: tpos[wk[s]] for s in range(slots)}
        r = dict(start=wk[0])
        for w in CHECK:
            i = tpos[wk[w - 1]] if w < 52 else tpos[grid[k0 + 52]]          # week 52 = one full year after the start
            inv = [s for s, f in first.items() if f <= i]
            r[f"w{w}"] = (sum(V[s, i] for s in inv) / len(inv) - 1) * 100
        rows.append(r)
    R = pd.DataFrame(rows)
    print(f"{len(R)} fresh starts, {R.start.min().date()}..{R.start.max().date()} · ends at week 52: mean {R.w52.mean():.0f}% · lost money {(R.w52 < 0).mean():.0%} · worst {R.w52.min():.0f}% · best {R.w52.max():.0f}%")
    for w in CHECK[:-1]:
        rho = R[f"w{w}"].rank().corr(R["w52"].rank())
        print(f"week {w}: rank correlation with week 52 = {rho:.2f}")
    for w, cuts in ((13, [-100, 0, 10, 1000]), (26, [-100, 0, 15, 30, 1000]), (30, [-100, 0, 15, 30, 1000])):
        g = R.groupby(pd.cut(R[f"w{w}"], cuts), observed=True)["w52"]
        print(f"\nweek {w} return -> week 52 return")
        for k, s in g:
            print(f"  week {w} {str(k):14s} {len(s):3d} starts ({s.index.map(lambda i: R.start[i].year).nunique()} start years) · week 52: mean {s.mean():5.0f}% · median {s.median():5.0f}% · worst {s.min():5.0f}% · lost money {(s < 0).mean():.0%}")
    R.to_csv(ROOT / "reports/v32_yearly_growth/early_signal.csv", index=False)


if __name__ == "__main__":
    main()
