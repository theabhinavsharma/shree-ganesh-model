"""EXP-2026-10-13-v32-hold-after-tax (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: V3.2 sells every lot after 6 months, so every gain is short-term and taxed at 20.8%. Does holding 9 months, or past
12 months (13.0% long-term tax), leave more money after tax?
Arms: H6 (126 sessions, live), H9 (189), H12 (260 sessions, ~12.5 months). Same V3.2 lists; test_v3_l1_newlist_cost.run_cost with
sre.HOLD per arm (weekly ladder of HOLD//5+1 slices), dividends counted, payouts as cash, liquidity-scaled costs, list-day close.
Tax (today's rates, every year): held <= 365 days 20.8%, > 365 days 13.0% (cess included, no exemption, no surcharge), charged on
each lot's gain at its sale and taken from the money that rolls on; a lot's loss offsets later gains in its slice; on the last
date open lots are treated as sold. ADOPTION per arm (H9, H12), after tax: beats H6 in both eras, max drawdown not worse by > 2
points, at schedule 0 and >= 4 of 5. SGM_REPRO=1: log nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402

EXP_ID = "EXP-2026-10-13-v32-hold-after-tax"
OUTDIR = ROOT / "logs/leader_sleeve/v32_hold_after_tax"
START, YEARS = pd.Timestamp("2019-04-01"), (2023, 2024, 2025, 2026)
HOLDS = {"H6": 126, "H9": 189, "H12": 260}
ST, LT = 0.208, 0.130


def engines():
    """run_cost with tax at each sale (and its next-open variant), built from the registered engine's own source."""
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")].replace("def run_cost(D, X, picks, weekly, rule, costs_of):", "def run_tax(D, X, picks, weekly, rule, costs_of, st, lt):")
    reps = [("        cap = 1.0; v = V[s]; v[start_i:] = cap; late = np.zeros(T); pend = []",
             "        cap = 1.0; v = V[s]; v[start_i:] = cap; late = np.zeros(T); pend = []; bank = 0.0"),
            ("                cap = cap * float(path[-1]); names_n.append(len(cols))",
             "                cap0 = cap; cap = cap * float(path[-1]); names_n.append(len(cols))\n"
             "                gain = cap - cap0; rate = lt if (cal[end] - cal[i0]).days > 365 else st\n"
             "                if gain < 0:\n"
             "                    bank += -gain; tx = 0.0\n"
             "                else:\n"
             "                    taxable = max(0.0, gain - bank); bank = max(0.0, bank - gain); tx = rate * taxable\n"
             "                cap -= tx\n"
             "                if end == T - 1:\n"
             "                    v[end] -= tx")]
    for a, b in reps:
        assert fn.count(a) == 1, a[:60]
        fn = fn.replace(a, b)
    fo = fn.replace("def run_tax(", "def run_tax_open(")
    fo = fo.replace("                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])", "                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])\n                seg[0] = np.nan_to_num(D[\"first_fac\"][i0, cols] - 1.0)")
    fo = fo.replace("            names = [n for n in picks.get(d, []) if n in cidx]", "            names = [n for n in picks.get(d, []) if n in cidx and not D[\"locked\"][i0, cidx[n]] and np.isfinite(D[\"first_fac\"][i0, cidx[n]])]")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(fn, ns); exec(fo, ns)
    return ns["run_tax"], ns["run_tax_open"]


def boot(a, b, block=126, n=4000, seed=7):
    d = (np.log(a).diff() - np.log(b).diff()).dropna().to_numpy(); T = len(d)
    spy = T / ((a.index[-1] - a.index[0]).days / 365.25); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block))
    out = np.array([d[((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T].mean() * spy for _ in range(n)]) * 100
    return round(float(d.mean() * spy * 100), 1), round(float(np.percentile(out, 2.5)), 1), round(float(np.percentile(out, 97.5)), 1)


def main() -> None:
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
    run_tax, run_tax_open = engines()
    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    lists = {o: {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[o], imap, P, ctx).items()} for o in range(5)}

    # ---- data gate (no outcomes yet)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10))]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}

    # ---- wiring: H6 pre-tax == the promoter-buyers V3.2 code on the same data; tax engine at 0% == run_cost
    sre.HOLD = 126
    nav6 = run_cost(D, X, lists[0], weeks[0], "E0", cost)[0]
    got = [round(met(nav6)[k], 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
    rr = subprocess.run(["/usr/bin/python3", "src/agentic/test_v32_promoter_buyers.py"], cwd=ROOT, env=dict(os.environ, SGM_REPRO="1"), capture_output=True, text=True)
    if rr.returncode != 0:
        raise SystemExit("wiring reference run failed — no result reported")
    rv = pd.read_csv(ROOT / "logs/leader_sleeve/v32_promoter_buyers/repro/results.csv").set_index("arm").loc["V3.2"]
    want = [round(float(rv[k]), 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
    zero = run_tax(D, X, lists[0], weeks[0], "E0", cost, 0.0, 0.0)[0]
    same0 = bool(np.allclose(zero.to_numpy(), nav6.to_numpy(), rtol=0, atol=1e-12))
    print(f"wiring check: H6 pre-tax {got} vs promoter-buyers code {want} · tax engine at 0% == run_cost: {same0}", flush=True)
    if got != want or not same0:
        raise SystemExit("wiring check FAILED — no result reported")

    held = {}
    for a, h in HOLDS.items():
        ends = [(cal[tpos[d] + h] - d).days for d in weeks[0] if tpos[d] + h <= len(cal) - 1]
        held[a] = dict(min_days=int(min(ends)), max_days=int(max(ends)), lots_long_term=f"{sum(x > 365 for x in ends)}/{len(ends)}")
    print(f"days held per lot (schedule 0): {held}", flush=True)

    res = {k: {a: [] for a in HOLDS} for k in ("pre", "post", "post_open")}
    navs = {}
    for o in range(5):
        for a, h in HOLDS.items():
            sre.HOLD = h
            n_pre = run_cost(D, X, lists[o], weeks[o], "E0", cost)[0]
            n_post = run_tax(D, X, lists[o], weeks[o], "E0", cost, ST, LT)[0]
            n_open = run_tax_open(D, X, lists[o], weeks[o], "E0", cost, ST, LT)[0]
            res["pre"][a].append(met(n_pre)); res["post"][a].append(met(n_post)); res["post_open"][a].append(met(n_open))
            navs[(a, o)] = n_post
        print(f"schedule {o}: after tax " + " · ".join(f"{a} {res['post'][a][o]['cagr_disc']:.1f}/{res['post'][a][o]['cagr_conf']:.1f}/{res['post'][a][o]['maxdd']:.1f}" for a in HOLDS)
              + " · pre-tax " + " · ".join(f"{a} {res['pre'][a][o]['cagr']:.1f}" for a in HOLDS), flush=True)
    sre.HOLD = 126
    ph = {a: [bool(sp.beats(res["post"][a][o], res["post"]["H6"][o])) for o in range(5)] for a in ("H9", "H12")}
    passed = {a: bool(v[0] and sum(v) >= 4) for a, v in ph.items()}
    bs = {a: [boot(navs[(a, o)], navs[("H6", o)]) for o in range(5)] for a in ("H9", "H12")}
    yrs = {k: {a: {y: round(float(np.mean([x[y] for x in res[k][a]])), 1) for y in YEARS} for a in HOLDS} for k in ("pre", "post")}

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in HOLDS:
        r, p, ro = res["post"][a][0], res["pre"][a][0], res["post_open"][a][0]
        print(f"{a:4s} AFTER TAX CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res['post'][a]):.1f}..{max(x['cagr'] for x in res['post'][a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs['post'][a][y]:+.0f}" for y in YEARS) + f" · next open {ro['cagr']:.1f} || pre-tax {p['cagr']:.1f} ({p['cagr_disc']:.1f}/{p['cagr_conf']:.1f}/{p['maxdd']:.1f})")
    for a in ("H9", "H12"):
        print(f"{a} vs H6 after tax: beats in {sum(ph[a])}/5 (schedule 0 {'yes' if ph[a][0] else 'no'}) · growth gap {bs[a][0][0]:+.1f} pts/yr, 95% range {bs[a][0][1]:+.1f}..{bs[a][0][2]:+.1f} (schedule 0) · range excludes 0 in {sum(lo > 0 or hi < 0 for _, lo, hi in bs[a])}/5 · {'PASS' if passed[a] else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, basis=k, **{c: round(res[k][a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs[k][a][y] for y in YEARS} if k in yrs else {})
                  for k in ("pre", "post", "post_open") for a in HOLDS]).to_csv(OUTDIR / "results.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v32_hold_after_tax.py", definitions=__doc__,
        basis=dict(pre="before tax", post="after tax, list-day close entry", post_open="after tax, bought at the next open"), phases={a: f"{sum(v)}/5" for a, v in ph.items()},
        bootstrap_after_tax_gap_vs_H6=bs, days_held=held, units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules"), updated=now), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nHold 6 / 9 / 12+ months on V3.2, compared after tax. H9: {'PASS' if passed['H9'] else 'no pass'}; H12: {'PASS' if passed['H12'] else 'no pass'}. results.csv + manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=[a for a, v in passed.items() if v] or "no pass",
                                 after_tax={a: [round(res["post"][a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in HOLDS},
                                 pre_tax={a: [round(res["pre"][a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in HOLDS},
                                 next_open_after_tax={a: round(res["post_open"][a][0]["cagr"], 1) for a in HOLDS},
                                 phases={a: f"{sum(v)}/5" for a, v in ph.items()}, bootstrap=bs, calendar_years_after_tax=yrs["post"], days_held=held,
                                 cols="CAGR, 2019-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
