"""EXP-2026-10-17-small-industries (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: an industry gets a heat score only with >= 5 names trading Rs 5 cr a day, so stocks in small niche
industries can never pass V3.2's hot-industry gate. Two fixes are tested against V3.2:
  SMALLHEAT  an unscored industry's heat = mean 60-session return of its members liquid on V3.1's bar (Rs 1 cr, > Rs 50),
             however few, placed against that day's scored industries; eligible at >= 0.70 like G1 (no policy veto)
  SMALLFREE  every unscored industry eligible, no heat test
Everything else as V3.2 (trend test / new-listing exemption, model rank top 9, financial / fading / trade-for-trade drops,
no refill). PASS: both eras, maxDD not worse by > 2 points, schedule 0 and >= 4 of 5. Also: picks from unscored industries.
Mode `parent` (EXP-2026-10-18-parent-heat): PARENTHEAT = an unscored industry takes the heat percentile of its nearest
scored parent in NSE's hierarchy (broad_industry, then sector, then broad_sector; groups with >= 5 core names at Rs 5 cr,
mean 60-session return, percentile among that level's groups that day); eligible at >= 0.70.
Usage: test_small_industries.py [small | parent]
"""
import html, json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
from test_v32_hold_after_tax import engines, boot  # noqa: E402
MODE = sys.argv[1] if len(sys.argv) > 1 else "small"
EXP_ID = {"small": "EXP-2026-10-17-small-industries", "parent": "EXP-2026-10-18-parent-heat"}[MODE]
OUTDIR = ROOT / "logs/leader_sleeve" / {"small": "small_industries", "parent": "parent_heat"}[MODE]
LEVELS = ("broad_industry", "sector", "broad_sector")
START, HOLD, YEARS = pd.Timestamp("2019-04-01"), 126, (2023, 2024, 2025, 2026)


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); tpos = {d: i for i, d in enumerate(cal)}
    Sc = ctx["scores"]; G1 = ctx["G1"][["date", "industry"]]
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    keep = lambda s, d: ctx["keep"](s, d) and not t2t(s, d)  # noqa: E731
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)]); vr = D.get("vr", {})
    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    alld = sorted(set().union(*weeks.values()))

    # ---- unscored industries and their small-industry heat per list date
    W = F1[F1["trade_date"].isin(set(alld)) & F1["core"] & F1["ret60"].notna()][["symbol", "trade_date", "ret60"]].copy()
    W["industry"] = W["symbol"].map(imap); W = W.dropna(subset=["industry"])
    scored = set(zip(Sc["date"], Sc["industry"]))
    W["scored"] = [(d, i) in scored for d, i in zip(W["trade_date"], W["industry"])]
    U = W[~W["scored"]].groupby(["trade_date", "industry"])["ret60"].agg(heat="mean", n="size").reset_index()
    ref_heat = {d: np.sort(g["heat_mean"].dropna().to_numpy()) for d, g in Sc.groupby("date")}
    U["pct"] = [np.searchsorted(ref_heat[d], h, side="right") / len(ref_heat[d]) if d in ref_heat and len(ref_heat[d]) else np.nan for d, h in zip(U["trade_date"], U["heat"])]
    ADD = {"SMALLHEAT": U[U["pct"] >= 0.70][["trade_date", "industry"]].rename(columns={"trade_date": "date"}),
           "SMALLFREE": U[["trade_date", "industry"]].rename(columns={"trade_date": "date"})}
    if MODE == "parent":
        H = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet"); H = H[H["status"].str.startswith("OK")].dropna(subset=["industry"])
        H["industry"] = H["industry"].map(html.unescape)
        par = {lv: H.dropna(subset=[lv]).groupby("industry")[lv].agg(lambda x: x.mode().iat[0]) for lv in LEVELS}
        C5 = X["F"][X["F"]["trade_date"].isin(set(alld)) & X["F"]["core"].fillna(False).astype(bool) & X["F"]["ret60"].notna()][["symbol", "trade_date", "ret60"]].copy()
        C5["industry"] = C5["symbol"].map(imap); C5 = C5.dropna(subset=["industry"]); lp = {}
        for lv in LEVELS:
            C5[lv] = C5["industry"].map(par[lv]); g = C5.dropna(subset=[lv]).groupby(["trade_date", lv])["ret60"].agg(["mean", "size"])
            g = g[g["size"] >= 5].copy(); g["pct"] = g.groupby(level=0)["mean"].rank(pct=True); lp[lv] = g["pct"].to_dict()
        pp, pl = [], []
        for d, ind in zip(U["trade_date"], U["industry"]):
            v, l = np.nan, None
            for lv in LEVELS:
                k = (d, par[lv].get(ind))
                if k in lp[lv]:
                    v, l = lp[lv][k], lv; break
            pp.append(v); pl.append(l)
        U["parent_pct"], U["parent_level"] = pp, pl
        ADD = {"PARENTHEAT": U[U["parent_pct"] >= 0.70][["trade_date", "industry"]].rename(columns={"trade_date": "date"})}
        print(f"unscored industry-dates with a parent score: {U['parent_pct'].notna().mean() * 100:.1f}% · decided by " + ", ".join(f"{k} {v}" for k, v in U["parent_level"].value_counts().items())
              + f" · hot parent (>= 0.70): {(U['parent_pct'] >= 0.70).mean() * 100:.1f}%", flush=True)
    w0 = W[W["trade_date"].isin(set(weeks[0]))]
    print(f"liquid stock-weeks (schedule 0) in unscored industries: {int((~w0['scored']).sum()):,} of {len(w0):,} ({(~w0['scored']).mean() * 100:.1f}%) · "
          f"unscored industries per week: median {int(U[U['trade_date'].isin(set(weeks[0]))].groupby('trade_date').size().median())} · "
          f"members: median {int(U['n'].median())}, one-name {(U['n'] == 1).mean() * 100:.0f}%", flush=True)

    # ---- data gate (no outcomes yet)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    sd = set(Sc["date"]); wk = pd.DataFrame(dict(d=weeks[0], scored=[1.0 if d in sd else np.nan for d in weeks[0]]))
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.span("industry scores", Sc["date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.coverage("weekly list dates with industry scores", wk, "d", ["scored"], range(START.year, cal[-1].year + 1), min_share=0.95),
              dr.qc("liquid stocks with an industry label on list dates (share)", float(F1[F1["trade_date"].isin(set(weeks[0])) & F1["core"]]["symbol"].map(imap).notna().mean()), 0.9)]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=os.environ.get("SGM_RERUN_TAG") or "RESULT")
    del Px

    def lists(o, arm):
        el = G1 if arm == "V3.2" else pd.concat([G1, ADD[arm]]).drop_duplicates(["date", "industry"])
        return {d: [s for s in l if keep(s, d)] for d, l in v3_rule.ranked_v31(F1, weeks[o], imap, P, el, 9).items()}
    ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[0], imap, P, ctx).items()}
    base = lists(0, "V3.2"); same = sum(base[d] == ref[d] for d in weeks[0])
    print(f"wiring check: nothing added == v3_rule V3.2 on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}
    _, run_open = engines(); ARMS = ("V3.2", "SMALLHEAT", "SMALLFREE") if MODE == "small" else ("V3.2", "PARENTHEAT")
    res, opn, navs, ph, pk0 = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {}, {a: [] for a in ARMS[1:]}, {}
    for o in range(5):
        for a in ARMS:
            pk = base if (o == 0 and a == "V3.2") else lists(o, a); pk0[(a, o)] = pk
            n = run_cost(D, X, pk, weeks[o], "E0", cost)[0]; navs[(a, o)] = n; res[a].append(met(n))
            opn[a].append(met(run_open(D, X, pk, weeks[o], "E0", cost, 0.0, 0.0)[0]))
        for a in ARMS[1:]:
            ph[a].append(bool(sp.beats(res[a][o], res["V3.2"][o])))
        print(f"schedule {o}: " + " · ".join(f"{a} {res[a][o]['cagr_disc']:.1f}/{res[a][o]['cagr_conf']:.1f}/{res[a][o]['maxdd']:.1f}" for a in ARMS), flush=True)
    passed = {a: bool(v[0] and sum(v) >= 4) for a, v in ph.items()}
    bs = {a: [boot(navs[(a, o)], navs[("V3.2", o)]) for o in range(5)] for a in ARMS[1:]}
    yrs = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}

    def r6(s, d):
        q = D["cidx"].get(s); i0 = tpos[d]
        if q is None or i0 + HOLD > T - 1:
            return np.nan
        g = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
        for k, f in vr.get(q, ()):
            if i0 + 1 <= k <= i0 + HOLD - 1:
                g = f * g + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
        return ((1 - float(cost(i0, np.array([q]))[0])) * g - 1) * 100
    WI = {}
    V = pk0[("V3.2", 0)]
    for a in ARMS[1:]:
        E = pk0[(a, 0)]; added = set(zip(ADD[a]["date"], ADD[a]["industry"]))
        newp = [(d, s) for d in weeks[0] for s in E[d] if (d, imap.get(s)) in added]; outp = [(d, s) for d in weeks[0] for s in V[d] if s not in E[d]]
        sm = lambda lst: dict(picks=len(lst), avg6m=round(float(np.nanmean([r6(s, d) for d, s in lst])), 1) if lst else None,  # noqa: E731
                              hit50=round(float(np.nanmean([r6(s, d) >= 50 for d, s in lst if np.isfinite(r6(s, d))])) * 100, 1) if lst else None)
        WI[a] = dict(lists_changed=sum(E[d] != V[d] for d in weeks[0]), picks_from_small=sm(newp), picks_pushed_out=sm(outp),
                     share_from_small=round(len(newp) / max(1, sum(len(E[d]) for d in weeks[0])) * 100, 1),
                     top_small_names=pd.Series([s for _, s in newp]).value_counts().head(8).to_dict())
    WI["v32_all_picks"] = dict(picks=sum(len(V[d]) for d in weeks[0]), avg6m=round(float(np.nanmean([r6(s, d) for d in weeks[0] for s in V[d]])), 1))

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r = res[a][0]
        print(f"{a:9s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs[a][y]:+.0f}" for y in YEARS) + f" · next open {opn[a][0]['cagr']:.1f}")
    for a in ARMS[1:]:
        print(f"{a} vs V3.2: beats in {sum(ph[a])}/5 · gap {bs[a][0][0]:+.1f} pts/yr (95% {bs[a][0][1]:+.1f}..{bs[a][0][2]:+.1f}) · {'PASS' if passed[a] else 'no pass'}")
        print(f"   what changed: {json.dumps(WI[a])}")
    print(f"V3.2 all picks: {WI['v32_all_picks']}")
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(arm=a, **{c: round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs[a][y] for y in YEARS},
                       next_open=round(opn[a][0]["cagr"], 1)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    U.to_csv(OUTDIR / "unscored_industries.csv", index=False)
    for f, what, units in (("results.csv", "one row per arm, schedule 0; years = average of 5 schedules", dict(cagr="percent a year", maxdd="percent")),
                           ("unscored_industries.csv", "per list date: industries without a heat score, their liquid members' mean 60-session return and its place among scored industries",
                            dict(heat="fraction (0.1 = 10%)", n="liquid members on V3.1's bar", pct="share of scored industries at or below, 0-1"))):
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer=f"src/agentic/test_small_industries.py {MODE}", definitions=__doc__, contents=what, units=units, updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\n{'Small-industry stocks' if MODE == 'small' else 'Small-industry stocks with a hot parent group'} through the hot-industry gate: " + ", ".join(f"{a} {'PASS' if v else 'no pass'}" for a, v in passed.items())
                                      + ". results.csv, unscored_industries.csv, run.log.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + ("-RESULT" if not os.environ.get("SGM_RERUN_TAG") else "-RERUN-" + os.environ["SGM_RERUN_TAG"]),
                                 verdict=[a for a, v in passed.items() if v] or "no pass",
                                 arms={a: [round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph[a])}/5"] if a in ph else []) for a in ARMS},
                                 bootstrap=bs, calendar_years=yrs, next_open={a: round(opn[a][0]["cagr"], 1) for a in ARMS}, what_changed=WI,
                                 cols="CAGR, 2019-22, 2023+, maxDD, phases"), default=str) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
