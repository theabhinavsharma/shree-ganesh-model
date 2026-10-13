"""Rs 45 lakh from Apr 2019 with tax paid ONCE A YEAR on the financial year's net gains (2026-10-13; Abhinav: 'tax is paid every
1 year on yearly gain'). EXPLORATION, no rule changed.

The pre-tax ladder (test_v3_l1_newlist_cost.run_cost / its next-open variant) records every lot's realised gain. Each financial
year (Apr-Mar): gains and losses of all lots sold in it are netted (short-term 20.8%, long-term 13.0% incl. cess; a net loss is
carried forward), and the tax is paid at the last session of March out of the portfolio (every holding trimmed in proportion; the
small gain that trimming itself realises is ignored). On the last date every open lot is treated as sold and the year-to-date tax
is paid. Arms: V3.2 (list-day close / next open), hindsight-free chain (bare G1 2019-20, the walk-forward chooser's sets after),
equal-weight liquid market; average of the 5 weekly schedules.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule, test_industry_fundamentals as tif  # noqa: E402
ST, LT, START = 0.208, 0.130, pd.Timestamp("2019-04-01")


def recorders():
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")].replace("def run_cost(", "def run_rec(")
    a = "                cap = cap * float(path[-1]); names_n.append(len(cols))"
    assert fn.count(a) == 1
    fn = fn.replace(a, "                cap0 = cap; cap = cap * float(path[-1]); names_n.append(len(cols))\n                REC.append((end, (cap - cap0) / slots, (cal[end] - cal[i0]).days))")
    fo = fn.replace("def run_rec(", "def run_rec_open(")
    fo = fo.replace("                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])", "                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])\n                seg[0] = np.nan_to_num(D[\"first_fac\"][i0, cols] - 1.0)")
    fo = fo.replace("            names = [n for n in picks.get(d, []) if n in cidx]", "            names = [n for n in picks.get(d, []) if n in cidx and not D[\"locked\"][i0, cidx[n]] and np.isfinite(D[\"first_fac\"][i0, cidx[n]])]")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return"), REC=[]); exec(fn, ns); exec(fo, ns)
    def run(which, *args):
        ns["REC"].clear(); nav = ns[which](*args)[0]; return nav, list(ns["REC"])
    return run


def yearly_tax(nav, rec, cal):
    """After-tax NAV path (tax paid at each financial year's last March session) and the money in hand on the last date."""
    T = len(cal); fy_end = sorted({cal[cal <= pd.Timestamp(f"{y}-03-31")][-1] for y in range(cal[0].year + 1, cal[-1].year + 1) if (cal <= pd.Timestamp(f"{y}-03-31")).any()})
    pos = {d: i for i, d in enumerate(cal)}; i_nav0 = pos[nav.index[0]]
    f = 1.0; carry = 0.0; fser = pd.Series(1.0, index=nav.index); prev = -1; paid = 0.0
    def tax_on(g_st, g_lt, carry):
        net = g_st + g_lt
        if net <= 0:
            return 0.0, carry - net
        use = min(carry, net); carry -= use
        lt_part = max(0.0, min(g_lt, net - use)); st_part = net - use - lt_part      # losses first offset short-term, then long-term
        return ST * st_part + LT * lt_part, carry
    for d in fy_end + [cal[-1]]:
        i = pos[d]
        if i < i_nav0:
            continue
        last = d == cal[-1]
        g = [(gg, days) for e, gg, days in rec if prev < e <= i and (e < T - 1 or last)]
        g_st = f * sum(x for x, days in g if days <= 365); g_lt = f * sum(x for x, days in g if days > 365)
        tx, carry = tax_on(g_st, g_lt, carry)
        w = f * nav.loc[d]; paid += tx
        if last:
            return fser, w - tx, paid
        f *= (1 - tx / w); fser[fser.index > d] = f; prev = i


def main():
    run = recorders()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); F1["core_p10"] = (F1["adv"] >= 1.0) & (F1["raw"] > 10)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    import explore_v32_lumpsum as LS                                                          # its ranked / wf helpers are module level
    liquid = F1[F1["core"].fillna(False).astype(bool) & F1["symbol"].isin(D["cidx"])].groupby("trade_date")["symbol"].apply(list).to_dict()
    yrs = (cal[-1] - START).days / 365.25
    out = []
    for o in range(5):
        wk = [d for d in sp.weekly_grid(cal, o) if d >= START]
        v32 = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, wk, imap, P, ctx).items()}
        keep = sre.TOPN; sre.TOPN = 9; base = tif.select_elig(X["F"], wk, imap, P, ctx["G1"]); sre.TOPN = keep; base = {d: base.get(d, []) for d in wk}
        r = {}
        for k, pk, eng in (("V3.2 close", v32, "run_rec"), ("V3.2 open", v32, "run_rec_open"), ("base", base, "run_rec_open"),
                           ("wf0", LS.wf(wk, False), "run_rec_open"), ("wf1", LS.wf(wk, True), "run_rec_open"), ("market", {d: liquid.get(d, []) for d in wk}, "run_rec_open")):
            nav, rec = run(eng, D, X, pk, wk, "E0", cost)
            fser, final, paid = yearly_tax(nav, rec, cal)
            r[k] = dict(pre=nav.iloc[-1] / nav.iloc[0], post=final / nav.iloc[0], paid=paid / nav.iloc[0], path=(fser * nav) / nav.iloc[0], liq=final / (fser.iloc[-1] * nav.iloc[-1]))
        seg = lambda k, a, b: r[k]["path"][r[k]["path"].index <= b].iloc[-1] / (r[k]["path"][r[k]["path"].index <= a].iloc[-1] if (r[k]["path"].index <= a).any() else 1.0)  # noqa: E731
        hf = seg("base", pd.Timestamp("2000-01-01"), pd.Timestamp("2020-12-31")) * seg("wf0", pd.Timestamp("2020-12-31"), pd.Timestamp("2022-12-31")) * seg("wf1", pd.Timestamp("2022-12-31"), cal[-1]) * r["wf1"]["liq"]
        out.append({"o": o, **{k: r[k]["post"] for k in ("V3.2 close", "V3.2 open", "market")}, "hindsight-free": hf, "V3.2 pre-tax": r["V3.2 close"]["pre"], "tax paid (V3.2 open)": r["V3.2 open"]["paid"]})
        print({k: round(float(v), 2) for k, v in out[-1].items()}, flush=True)
    R = pd.DataFrame(out)
    for k in ("V3.2 pre-tax", "V3.2 close", "V3.2 open", "hindsight-free", "market"):
        m = R[k].mean()
        print(f"{k:16s} Rs 45 L -> Rs {45 * m:,.0f} L ({45 * R[k].min():,.0f}-{45 * R[k].max():,.0f}) · x{m:.1f} · {(m ** (1 / yrs) - 1) * 100:.1f}%/yr")
    print(f"tax paid over the years (V3.2, Monday open): Rs {45 * R['tax paid (V3.2 open)'].mean():,.0f} L")


if __name__ == "__main__":
    main()
