"""EXP-2026-10-14-v32-rule-significance (registered before any outcome was computed). MEASUREMENT.

Plain English: which of V3.2's rules are statistically proven? Remove one rule at a time and measure how much V3.2 loses, with a
block-bootstrap 95% range (126-session blocks) per weekly schedule. Grades: PROVEN (range above 0 in >= 4 of 5 schedules), LIKELY
(>= 80% of draws above 0 in >= 4 of 5), UNPROVEN, HURTS (range below 0 in >= 4 of 5).
"""
import json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
EXP_ID = "EXP-2026-10-14-v32-rule-significance"; OUTDIR = ROOT / "logs/leader_sleeve/v32_rule_significance"; START = pd.Timestamp("2019-04-01")
VARIANTS = ["policy_veto", "hot_industry", "financials", "fading", "liquidity_1cr", "new_listings", "t2t"]


def boot(a, b, block=126, n=4000, seed=7):
    d = (np.log(a).diff() - np.log(b).diff()).dropna().to_numpy(); T = len(d)
    spy = T / ((a.index[-1] - a.index[0]).days / 365.25); rng = np.random.default_rng(seed); nb = int(np.ceil(T / block))
    out = np.array([d[((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T].mean() * spy for _ in range(n)]) * 100
    return dict(gap=float(d.mean() * spy * 100), lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)), pos=float((out > 0).mean()))


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
    ctx = v3_rule.context(imap); F0 = X["F"].copy(); F0["core_old"] = F0["core"].astype(bool); F1 = v3_rule.frame_v31(F0, D)
    Sc = ctx["scores"]; ELIG = {"g1": ctx["G1"], "heat": Sc[Sc["heat_pct"] >= 0.70], "any": None}
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731

    def ranked(weeks, core_col, young_on, elig):
        W = F1[F1["trade_date"].isin(set(weeks)) & F1[core_col].fillna(False).astype(bool)].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        if elig is not None:
            W = W.merge(elig[["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = (W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])
        if young_on:
            tr = tr | W["young"].fillna(False).astype(bool)
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g = Pool.groupby("trade_date").head(9).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: g.get(d, []) for d in weeks}

    def lists(weeks, drop=None):
        core = "core_old" if drop == "liquidity_1cr" else "core"; young = drop != "new_listings"
        elig = ELIG["heat"] if drop == "policy_veto" else ELIG["any"] if drop == "hot_industry" else ELIG["g1"]
        R = ranked(weeks, core, young, elig)
        return {d: [s for s in R[d] if not (drop != "financials" and ctx["fin"](s)) and not (drop != "fading" and ctx["fading_only"](s, d))
                    and not (drop != "t2t" and t2t(s, d))] for d in weeks}

    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.span("industry scores", Sc["date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10))]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px
    w0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, w0, imap, P, ctx).items()}
    base0 = lists(w0); same = sum(base0[d] == ref[d] for d in w0)
    print(f"wiring check: rebuilt V3.2 == v3_rule V3.1 minus trade-for-trade on {same}/{len(w0)} weeks", flush=True)
    if same < len(w0):
        raise SystemExit("wiring check FAILED — no result reported")
    res = {v: [] for v in VARIANTS}; cg = {v: [] for v in VARIANTS + ["V3.2"]}; changed = {}
    for o in range(5):
        wk = [d for d in sp.weekly_grid(cal, o) if d >= START]
        b = lists(wk); nb = run_cost(D, X, b, wk, "E0", cost)[0]; cg["V3.2"].append(float(sp.metrics(nb)["cagr"]))
        for v in VARIANTS:
            lv = lists(wk, v); nv = run_cost(D, X, lv, wk, "E0", cost)[0]
            res[v].append(boot(nb, nv)); cg[v].append(float(sp.metrics(nv)["cagr"]))
            if o == 0:
                changed[v] = int(sum(lv[d] != b[d] for d in wk))
        print(f"schedule {o} done", flush=True)
    grade = {}
    for v in VARIANTS:
        r = res[v]; up = sum(x["lo"] > 0 for x in r); down = sum(x["hi"] < 0 for x in r); lik = sum(x["pos"] >= 0.8 for x in r)
        grade[v] = "PROVEN" if up >= 4 else "HURTS" if down >= 4 else "LIKELY" if lik >= 4 else "UNPROVEN"
    print(f"\n=== {EXP_ID} · V3.2 CAGR {np.mean(cg['V3.2']):.1f}%/yr (avg of 5 schedules) ===")
    for v in VARIANTS:
        r = res[v]
        print(f"{v:14s} {grade[v]:9s} · without it {np.mean(cg[v]):5.1f}%/yr · gap {min(x['gap'] for x in r):+5.1f}..{max(x['gap'] for x in r):+5.1f} pts/yr · "
              f"95% range above 0 in {sum(x['lo'] > 0 for x in r)}/5 · draws > 0 {min(x['pos'] for x in r):.0%}..{max(x['pos'] for x in r):.0%} · week-0 range {r[0]['lo']:+.1f}..{r[0]['hi']:+.1f} · lists changed {changed[v]}/{len(w0)}")
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(rule=v, grade=grade[v], cagr_without=round(np.mean(cg[v]), 1), gap_min=round(min(x["gap"] for x in res[v]), 1), gap_max=round(max(x["gap"] for x in res[v]), 1),
                       proven_schedules=sum(x["lo"] > 0 for x in res[v]), lists_changed=changed[v]) for v in VARIANTS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v32_rule_significance.py", definitions=__doc__,
        v32_cagr=round(np.mean(cg["V3.2"]), 1), units=dict(cagr_without="percent a year, avg of 5 schedules", gap="V3.2 minus variant, annualised growth points"), updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nLeave-one-out significance of V3.2's rules: " + ", ".join(f"{v} {g}" for v, g in grade.items()) + ".\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", grades=grade, v32_cagr=round(np.mean(cg["V3.2"]), 1),
                                 without={v: round(np.mean(cg[v]), 1) for v in VARIANTS}, bootstrap={v: [{k: round(x[k], 2) for k in x} for x in res[v]] for v in VARIANTS},
                                 lists_changed=changed)) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
