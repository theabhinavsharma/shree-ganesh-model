"""EXP-2026-10-15-promoter-pref-universe (registered in logs/experiments.jsonl before any outcome of these rules was computed).

Plain English: rank from the start with 'promoter took a preferential allotment in the last 90 days' (PREF) as the first key.
U1 = PREF names among ALL liquid stocks first (any industry, no trend test), then V3.2's ranked pool; U3 = PREF first within V3.2's
whole candidate pool; S = PREF names only (top 9 by model score, cash if none). Then V3.2's filters, no refill.
PASS per arm vs V3.2: both eras, max drawdown not worse by > 2 points, schedule 0 and >= 4 of 5.
"""
import json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
from test_v32_hold_after_tax import engines, boot  # noqa: E402
from test_famous_overlay import promoter_flags  # noqa: E402
EXP_ID = "EXP-2026-10-15-promoter-pref-universe"; OUTDIR = ROOT / "logs/leader_sleeve/famous/promoter_pref_universe"
START, YEARS = pd.Timestamp("2019-04-01"), (2023, 2024, 2025, 2026)


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); tpos = {d: i for i, d in enumerate(cal)}
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    keep = lambda s, d: ctx["keep"](s, d) and not t2t(s, d)  # noqa: E731
    _, run_open = engines()
    flags, info, m = promoter_flags(cal, tpos); pref = flags["PREF"]
    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}

    def universe_ranked(wk):
        """{date: PREF names among all liquid stocks, by model score (latest within 7 days; unscored last)}."""
        W = F1[F1["trade_date"].isin(set(wk)) & F1["core"].fillna(False).astype(bool)][["symbol", "trade_date"]].copy()
        W = W[[pref(s, d) for s, d in zip(W["symbol"], W["trade_date"])]]
        Lx = W.reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        mm = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        W["score"] = mm.set_index("index")["ensemble"].reindex(W.index).fillna(-np.inf)
        W = W.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g = W.groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: g.get(d, []) for d in wk}

    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    U0 = universe_ranked(weeks[0]); Kc = pd.DataFrame([dict(d=d, known=(1.0 if U0[d] else np.nan)) for d in weeks[0]])
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.span("promoter preferential disclosures", m["d"], pd.Timestamp("2019-04-30"), cal[-1] - pd.Timedelta(days=60)),
              dr.coverage("weeks with at least one liquid PREF name", Kc, "d", ["known"], range(START.year, cal[-1].year + 1), min_share=0.5)]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    def lists(o, arm):
        pool = v3_rule.ranked_v31(F1, weeks[o], imap, P, ctx["G1"], 10000)
        U = U0 if o == 0 else universe_ranked(weeks[o]); out = {}
        for d in weeks[o]:
            pl = pool[d]
            if arm == "V3.2":
                top = pl[:9]
            elif arm == "U1":
                top = (U[d] + [s for s in pl if s not in set(U[d])])[:9]
            elif arm == "U3":
                f = [s for s in pl if pref(s, d)]; top = (f + [s for s in pl if s not in set(f)])[:9]
            else:
                top = U[d][:9]
            out[d] = [s for s in top if keep(s, d)]
        return out
    ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[0], imap, P, ctx).items()}
    b0 = lists(0, "V3.2"); same = sum(b0[d] == ref[d] for d in weeks[0]); print(f"wiring check: V3.2 == v3_rule on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")

    def met(nav):
        mt = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(mt[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}
    ARMS = ("V3.2", "U1", "U3", "S")
    res, opn, navs, ph, sz = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {}, {a: [] for a in ARMS[1:]}, {}
    for o in range(5):
        for a in ARMS:
            pk = lists(o, a); n = run_cost(D, X, pk, weeks[o], "E0", cost)[0]; navs[(a, o)] = n; res[a].append(met(n))
            opn[a].append(met(run_open(D, X, pk, weeks[o], "E0", cost, 0.0, 0.0)[0]))
            if o == 0:
                ln = [len(v) for v in pk.values()]; pr = [np.mean([pref(s, d) for s in v]) for d, v in pk.items() if v]
                sz[a] = dict(names=round(float(np.mean(ln)), 1), empty=int(sum(x == 0 for x in ln)), pref_share=round(float(np.mean(pr)) * 100, 1) if pr else 0.0)
        for a in ph:
            ph[a].append(bool(sp.beats(res[a][o], res["V3.2"][o])))
        print(f"schedule {o}: " + " · ".join(f"{a} {res[a][o]['cagr_disc']:.1f}/{res[a][o]['cagr_conf']:.1f}/{res[a][o]['maxdd']:.1f}" for a in ARMS), flush=True)
    passed = {a: bool(v[0] and sum(v) >= 4) for a, v in ph.items()}
    bs = {a: [boot(navs[(a, o)], navs[("V3.2", o)]) for o in range(5)] for a in ph}
    yrs = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}
    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r = res[a][0]
        print(f"{a:5s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs[a][y]:+.0f}" for y in YEARS) + f" · next open {opn[a][0]['cagr']:.1f} · {sz[a]}")
    for a in ph:
        print(f"{a} vs V3.2: beats in {sum(ph[a])}/5 · gap {bs[a][0][0]:+.1f} pts/yr (95% {bs[a][0][1]:+.1f}..{bs[a][0][2]:+.1f}) · range excludes 0 in {sum(lo > 0 or hi < 0 for _, lo, hi in bs[a])}/5 · {'PASS' if passed[a] else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(arm=a, **{c: round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs[a][y] for y in YEARS},
                       next_open=round(opn[a][0]["cagr"], 1), **sz[a]) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_promoter_pref_universe.py", definitions=__doc__,
        phases={a: f"{sum(v)}/5" for a, v in ph.items()}, bootstrap=bs, units=dict(cagr="percent a year", maxdd="percent", pref_share="percent of a list's names that are PREF"), updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nPREF as the first ranking key over the whole universe / pool / alone: " + ", ".join(f"{a} {'PASS' if v else 'no pass'}" for a, v in passed.items()) + ".\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=[a for a, v in passed.items() if v] or "no pass",
                                 arms={a: [round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS}, phases={a: f"{sum(v)}/5" for a, v in ph.items()},
                                 bootstrap=bs, calendar_years=yrs, next_open={a: round(opn[a][0]["cagr"], 1) for a in ARMS}, lists=sz, cols="CAGR, 2019-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
