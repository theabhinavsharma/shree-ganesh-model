"""EXP-2026-10-15-v32-blended-score (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: instead of 'hot industries only, then the best model scores', rank every trend-passing liquid stock by
B = industry heat x stock model score. B9 = top 9 by B; BT63 = every stock with B >= 0.63 (0.70 x 0.90). Then V3.2's filters, no
refill. PASS per arm vs V3.2: both eras, max drawdown not worse by > 2 points, schedule 0 and >= 4 of 5.
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
EXP_ID = "EXP-2026-10-15-v32-blended-score"; OUTDIR = ROOT / "logs/leader_sleeve/v32_blended_score"
START, HOLD, YEARS, TAU = pd.Timestamp("2019-04-01"), 126, (2023, 2024, 2025, 2026), 0.63


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); tpos = {d: i for i, d in enumerate(cal)}
    Sc = ctx["scores"]; ALL = Sc[~(Sc["P_pct"] < 0.30)]
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    keep = lambda s, d: ctx["keep"](s, d) and not t2t(s, d)  # noqa: E731
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)]); vr = D.get("vr", {})
    _, run_open = engines()

    def pool(weeks, elig):
        """v3_rule.ranked_v31's pool (same steps), keeping score and industry heat."""
        W = F1[F1["trade_date"].isin(set(weeks)) & F1["core"]].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W.merge(elig[["date", "industry", "heat_pct"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])) | W["young"]
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool["B"] = Pool["heat_pct"] * Pool["score"].where(np.isfinite(Pool["score"]), np.nan)
        return Pool

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    pools = {o: pool(weeks[o], ALL) for o in range(5)}

    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    pz = pools[0]; K = pd.DataFrame(dict(d=pz["trade_date"], known=np.where(pz["B"].notna(), 1.0, np.nan)))
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.span("industry scores", Sc["date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.coverage("blended score known for pool candidates (heat and model score)", K, "d", ["known"], range(START.year, cal[-1].year + 1), min_share=0.9)]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    def lists(o, arm):
        Pz = pools[o]; out = {}
        if arm == "V3.2":
            G = Pz.merge(ctx["G1"][["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
            G = G.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort"); g = G.groupby("trade_date").head(9)
        else:
            G = Pz.dropna(subset=["B"]).sort_values(["trade_date", "B", "symbol"], ascending=[True, False, True], kind="mergesort")
            g = G.groupby("trade_date").head(9) if arm == "B9" else G[G["B"] >= TAU]
        gl = g.groupby("trade_date")["symbol"].apply(list).to_dict()
        for d in weeks[o]:
            out[d] = [s for s in gl.get(d, []) if keep(s, d)]
        return out
    ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[0], imap, P, ctx).items()}
    base = lists(0, "V3.2"); same = sum(base[d] == ref[d] for d in weeks[0])
    print(f"wiring check: V3.2 via the pool builder == v3_rule on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}
    ARMS = ("V3.2", "B9", "BT63")
    res, opn, navs, ph, sizes = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {}, {"B9": [], "BT63": []}, {}
    for o in range(5):
        for a in ARMS:
            pk = lists(o, a); n = run_cost(D, X, pk, weeks[o], "E0", cost)[0]; navs[(a, o)] = n; res[a].append(met(n))
            opn[a].append(met(run_open(D, X, pk, weeks[o], "E0", cost, 0.0, 0.0)[0]))
            if o == 0:
                sz = [len(v) for v in pk.values()]; sizes[a] = dict(avg=round(float(np.mean(sz)), 1), median=float(np.median(sz)), max=int(max(sz)), empty_weeks=int(sum(x == 0 for x in sz)))
        for a in ph:
            ph[a].append(bool(sp.beats(res[a][o], res["V3.2"][o])))
        print(f"schedule {o}: " + " · ".join(f"{a} {res[a][o]['cagr_disc']:.1f}/{res[a][o]['cagr_conf']:.1f}/{res[a][o]['maxdd']:.1f}" for a in ARMS), flush=True)
    passed = {a: bool(v[0] and sum(v) >= 4) for a, v in ph.items()}
    bs = {a: [boot(navs[(a, o)], navs[("V3.2", o)]) for o in range(5)] for a in ph}
    yrs = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}

    # ---- where B9's picks come from (schedule 0, after filters) and how they did
    def r6(s, d):
        q = D["cidx"].get(s); i0 = tpos[d]
        if q is None or i0 + HOLD > T - 1:
            return np.nan
        g = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
        for k, f in vr.get(q, ()):
            if i0 + 1 <= k <= i0 + HOLD - 1:
                g = f * g + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
        return ((1 - float(cost(i0, np.array([q]))[0])) * g - 1) * 100
    hp = pools[0].set_index(["trade_date", "symbol"])["heat_pct"]
    tier = lambda h: "hot (>= 0.70)" if h >= 0.70 else "mid (0.40-0.70)" if h >= 0.40 else "cold (< 0.40)"  # noqa: E731
    rows = [dict(tier=tier(float(hp.loc[(d, s)])), r=r6(s, d)) for d, l in lists(0, "B9").items() for s in l if (d, s) in hp.index]
    TR = pd.DataFrame(rows).dropna().groupby("tier")["r"].agg(picks="size", avg="mean", hit50=lambda s: (s >= 50).mean() * 100, lost=lambda s: (s < 0).mean() * 100).round(1)

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r = res[a][0]
        print(f"{a:5s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs[a][y]:+.0f}" for y in YEARS) + f" · next open {opn[a][0]['cagr']:.1f} · names/list {sizes[a]}")
    for a in ph:
        print(f"{a} vs V3.2: beats in {sum(ph[a])}/5 · gap {bs[a][0][0]:+.1f} pts/yr (95% {bs[a][0][1]:+.1f}..{bs[a][0][2]:+.1f}) · range excludes 0 in {sum(lo > 0 or hi < 0 for _, lo, hi in bs[a])}/5 · {'PASS' if passed[a] else 'no pass'}")
    print("\nB9 picks by industry heat (schedule 0, after filters, 6-month return after costs):"); print(TR.to_string())
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(arm=a, **{c: round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs[a][y] for y in YEARS},
                       next_open=round(opn[a][0]["cagr"], 1), names_avg=sizes[a]["avg"]) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    TR.to_csv(OUTDIR / "b9_by_heat.csv")
    for f, what in (("results.csv", "one row per arm, schedule 0; years = average of 5 schedules"), ("b9_by_heat.csv", "B9 picks by industry heat tier, 6-month return after costs, schedule 0")):
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_v32_blended_score.py", definitions=__doc__, contents=what, updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nIndustry heat x stock score: " + ", ".join(f"{a} {'PASS' if v else 'no pass'}" for a, v in passed.items()) + ". results.csv, b9_by_heat.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=[a for a, v in passed.items() if v] or "no pass",
                                 arms={a: [round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS}, phases={a: f"{sum(v)}/5" for a, v in ph.items()},
                                 bootstrap=bs, calendar_years=yrs, next_open={a: round(opn[a][0]["cagr"], 1) for a in ARMS}, names_per_list=sizes,
                                 b9_by_heat=TR.reset_index().to_dict("records"), cols="CAGR, 2019-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
