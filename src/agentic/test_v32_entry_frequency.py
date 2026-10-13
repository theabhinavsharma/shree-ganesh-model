"""EXP-2026-10-14-v32-entry-frequency (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: same V3.2 lists and same total money, but buy every 2 weeks (13 bigger slices) or every 4 weeks (7 slices) instead
of every week (26 slices). Does it change the return? Engine: test_v3_l1_newlist_cost.run_cost with slices = 126 // (5 x step) + 1;
pass vs W1 pre-tax (both eras, drawdown, schedule 0 and >= 4 of 5); after tax (20.8% per sale) + next open reported.
"""
import json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v32_hold_after_tax import engines, boot, ST, LT  # noqa: E402
EXP_ID = "EXP-2026-10-14-v32-entry-frequency"; OUTDIR = ROOT / "logs/leader_sleeve/v32_entry_frequency"
START, YEARS, STEPS = pd.Timestamp("2019-04-01"), (2023, 2024, 2025, 2026), {"W1": 1, "W2": 2, "W4": 4}


def build():
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")]
    a = "T, slots = len(cal), sp.slots_for(sre.HOLD)"; assert fn.count(a) == 1
    fn = fn.replace(a, "T, slots = len(cal), sre.HOLD // (5 * STEP) + 1").replace("def run_cost(D, X, picks, weekly, rule, costs_of):", "def run_step(D, X, picks, weekly, rule, costs_of, STEP):")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(fn, ns)
    return ns["run_step"], fn


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    run_step, fn = build()
    # after-tax next-open engine with the slice change (built the same way as test_v32_hold_after_tax.engines)
    import test_v32_hold_after_tax as H
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    f2 = src_[src_.index("def run_cost("):src_.index("def main()")].replace("def run_cost(D, X, picks, weekly, rule, costs_of):", "def run_tax_open(D, X, picks, weekly, rule, costs_of, st, lt, STEP):")
    for a, b in [("T, slots = len(cal), sp.slots_for(sre.HOLD)", "T, slots = len(cal), sre.HOLD // (5 * STEP) + 1"),
                 ("        cap = 1.0; v = V[s]; v[start_i:] = cap; late = np.zeros(T); pend = []", "        cap = 1.0; v = V[s]; v[start_i:] = cap; late = np.zeros(T); pend = []; bank = 0.0"),
                 ("                cap = cap * float(path[-1]); names_n.append(len(cols))",
                  "                cap0 = cap; cap = cap * float(path[-1]); names_n.append(len(cols))\n                gain = cap - cap0; rate = lt if (cal[end] - cal[i0]).days > 365 else st\n"
                  "                if gain < 0:\n                    bank += -gain; tx = 0.0\n                else:\n                    taxable = max(0.0, gain - bank); bank = max(0.0, bank - gain); tx = rate * taxable\n"
                  "                cap -= tx\n                if end == T - 1:\n                    v[end] -= tx"),
                 ("                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])", "                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])\n                seg[0] = np.nan_to_num(D[\"first_fac\"][i0, cols] - 1.0)"),
                 ("            names = [n for n in picks.get(d, []) if n in cidx]", "            names = [n for n in picks.get(d, []) if n in cidx and not D[\"locked\"][i0, cidx[n]] and np.isfinite(D[\"first_fac\"][i0, cidx[n]])]")]:
        assert f2.count(a) == 1, a[:50]; f2 = f2.replace(a, b)
    ns2 = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(f2, ns2); run_real = ns2["run_tax_open"]

    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10))]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}

    lists = {o: {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, [d for d in sp.weekly_grid(cal, o) if d >= START], imap, P, ctx).items()} for o in range(5)}
    # wiring: STEP=1 engine == run_cost
    from test_v3_l1_newlist_cost import run_cost
    w0 = list(lists[0]); a = run_cost(D, X, lists[0], w0, "E0", cost)[0]; b = run_step(D, X, lists[0], w0, "E0", cost, 1)[0]
    same = bool(np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)); print(f"wiring check: every-week engine == run_cost: {same}", flush=True)
    if not same:
        raise SystemExit("wiring check FAILED — no result reported")
    res, real, ph, navs, extra = {k: [] for k in STEPS}, {k: [] for k in STEPS}, {k: [] for k in ("W2", "W4")}, {}, {k: [] for k in ("W2", "W4")}
    for o in range(5):
        wk = list(lists[o])
        for k, st in STEPS.items():
            sub = wk[::st]
            n = run_step(D, X, {d: lists[o][d] for d in sub}, sub, "E0", cost, st)[0]; navs[(k, o)] = n
            res[k].append(met(n)); real[k].append(met(run_real(D, X, {d: lists[o][d] for d in sub}, sub, "E0", cost, ST, LT, st)[0]))
            for ph_ in range(1, st):                            # other phases (diagnostic range)
                s2 = wk[ph_::st]; extra[k].append(met(run_step(D, X, {d: lists[o][d] for d in s2}, s2, "E0", cost, st)[0])["cagr"])
        for k in ("W2", "W4"):
            ph[k].append(bool(sp.beats(res[k][o], res["W1"][o])))
        print(f"schedule {o}: " + " · ".join(f"{k} {res[k][o]['cagr_disc']:.1f}/{res[k][o]['cagr_conf']:.1f}/{res[k][o]['maxdd']:.1f}" for k in STEPS), flush=True)
    passed = {k: bool(v[0] and sum(v) >= 4) for k, v in ph.items()}
    bs = {k: [boot(navs[(k, o)], navs[("W1", o)]) for o in range(5)] for k in ("W2", "W4")}
    yrs = {k: {y: round(float(np.mean([x[y] for x in res[k]])), 1) for y in YEARS} for k in STEPS}
    print(f"\n=== {EXP_ID} · entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for k in STEPS:
        r, q = res[k][0], real[k][0]; allc = [x["cagr"] for x in res[k]] + extra.get(k, [])
        print(f"{k} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · all phases {min(allc):.1f}..{max(allc):.1f} ({len(allc)})"
              + " · years " + " ".join(f"{y}:{yrs[k][y]:+.0f}" for y in YEARS) + f" · after tax + next open {q['cagr']:.1f} (maxDD {q['maxdd']:.1f})")
    for k in ("W2", "W4"):
        print(f"{k} vs W1: beats in {sum(ph[k])}/5 · gap {bs[k][0][0]:+.1f} pts/yr (95% {bs[k][0][1]:+.1f}..{bs[k][0][2]:+.1f}) · {'PASS' if passed[k] else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(arm=k, **{c: round(res[k][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs[k][y] for y in YEARS},
                       after_tax_next_open=round(real[k][0]["cagr"], 1)) for k in STEPS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v32_entry_frequency.py", definitions=__doc__,
        phases={k: f"{sum(v)}/5" for k, v in ph.items()}, bootstrap=bs, units=dict(cagr="percent a year", maxdd="percent"), updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nEvery week vs every 2 / 4 weeks (same money, bigger lots). W2: {'PASS' if passed['W2'] else 'no pass'}; W4: {'PASS' if passed['W4'] else 'no pass'}.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=[k for k, v in passed.items() if v] or "no pass",
                                 arms={k: [round(res[k][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for k in STEPS},
                                 after_tax_next_open={k: round(real[k][0]["cagr"], 1) for k in STEPS}, phases={k: f"{sum(v)}/5" for k, v in ph.items()},
                                 bootstrap=bs, calendar_years=yrs, cols="CAGR, 2019-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
