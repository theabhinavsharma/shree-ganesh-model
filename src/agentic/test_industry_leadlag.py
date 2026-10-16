"""EXP-2026-10-16-industry-leadlag (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: do some industries' moves predict OTHER industries' next month, beyond those industries' own momentum and
the market? Each January a lead-lag map is found from earlier years only (every ordered industry pair, regression of the
follower's next 4 weeks on the leader's past 4 weeks, controlling for the follower's own past 4 weeks and the market's;
Benjamini-Hochberg false discovery rate q <= 0.10). Through that year the map gives each follower a lead score
(sum of beta x leader's past-4-week return).
Part A (evidence): weekly rank correlation (IC) of lead score vs the industry's realised next 4 weeks, out of sample.
Part B (adoption): E1A = V3.2 with industries 'about to get hot' (lead score top 10%, heat_pct < 0.70, not policy-vetoed)
added to the G1 gate. PASS vs V3.2: both eras, maxDD not worse by > 2 points, schedule 0 and >= 4 of 5.
Industry return = equal-weight total return of its liquid members (Rs 1 cr ADV, price > Rs 50) at the window start,
industries with >= 5 such members; market = equal-weight all liquid stocks. Industry map = today's (as V3.2's).
"""
import json, math, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
from test_v32_hold_after_tax import engines, boot  # noqa: E402
EXP_ID = "EXP-2026-10-16-industry-leadlag"; OUTDIR = ROOT / "logs/leader_sleeve/industry_leadlag"
START, HOLD, YEARS = pd.Timestamp("2019-04-01"), 126, (2023, 2024, 2025, 2026)
EST0, PER, MINMEM, MINOBS, Q, TOP = pd.Timestamp("2016-01-01"), 20, 5, 24, 0.10, 0.10


def bh(p, q):
    """Benjamini-Hochberg discoveries at false discovery rate q."""
    p = np.asarray(p); n = len(p); m = np.zeros(n, bool)
    if n:
        o = np.argsort(p); ok = p[o] <= q * np.arange(1, n + 1) / n
        if ok.any():
            m[o[:np.nonzero(ok)[0].max() + 1]] = True
    return m


def mean_boot(x, block=13, n=2000, seed=0):
    """Mean of a weekly series with a moving-block bootstrap 95% range."""
    x = np.asarray(x, float); T = len(x); rng = np.random.default_rng(seed); k = math.ceil(T / block); out = []
    for _ in range(n):
        st = rng.integers(0, T - block + 1, k); out.append(np.concatenate([x[s:s + block] for s in st])[:T].mean())
    return float(x.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); tpos = {d: i for i, d in enumerate(cal)}; syms = D["syms"]
    Sc = ctx["scores"]; SI = {(d, i): (h, q) for d, i, h, q in Sc[["date", "industry", "heat_pct", "P_pct"]].itertuples(index=False)}
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    L["liq"] = ((L["adv"] >= v3_rule.V31_ADV_CR) & (rp.raw_price(L, "close") > v3_rule.V31_PRICE)).astype(float)
    ADV = rp.wide(L, "adv", cal).reindex(columns=syms).to_numpy(); LIQ = rp.wide(L, "liq", cal).reindex(columns=syms).fillna(0).to_numpy() > 0; del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    keep = lambda s, d: ctx["keep"](s, d) and not t2t(s, d)  # noqa: E731
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)]); vr = D.get("vr", {})
    inds = sorted(imap.dropna().unique()); nI = len(inds); icode = {s: i for i, s in enumerate(inds)}
    g = np.array([icode.get(imap.get(s), -1) for s in syms])

    def ind_ret(i0, i1):
        """(industry returns close i0 -> close i1, NaN under MINMEM liquid members at i0; market return)."""
        r = np.exp(lg[i1 + 1] - lg[i0 + 1]) - 1; m = LIQ[i0] & (g >= 0)
        cnt = np.bincount(g[m], minlength=nI); sm = np.bincount(g[m], weights=r[m], minlength=nI)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(cnt >= MINMEM, sm / cnt, np.nan)
        return out, float(r[LIQ[i0]].mean()), cnt

    # ---- non-overlapping 4-week periods from 2016 for the maps
    b = list(range(int(np.searchsorted(np.asarray(cal), np.datetime64(EST0))), T, PER))
    RR, MM, NN = [], [], []
    for j in range(len(b) - 1):
        r, m, c = ind_ret(b[j], b[j + 1]); RR.append(r); MM.append(m); NN.append(int((c >= MINMEM).sum()))
    RR, MM = np.array(RR), np.array(MM)
    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}

    # ---- data gate (no outcomes yet)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh_rows = dr.bhav_rows()
    per = pd.DataFrame(dict(d=[cal[b[j + 1]] for j in range(len(b) - 1)], ind20=[n if n >= 20 else np.nan for n in NN]))
    sd = set(Sc["date"]); wk = pd.DataFrame(dict(d=weeks[0], scored=[1.0 if d in sd else np.nan for d in weeks[0]]))
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh_rows.groupby(bh_rows.index.year).sum(), range(2016, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.span("industry scores", Sc["date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.coverage("4-week periods with >= 20 industries of >= 5 liquid members", per, "d", ["ind20"], range(2016, cal[-1].year + 1), min_share=0.9),
              dr.coverage("weekly list dates with industry scores", wk, "d", ["scored"], range(START.year, cal[-1].year + 1), min_share=0.95)]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px
    print(f"periods {len(b) - 1} · industries {nI} · industries with >= {MINMEM} liquid members per period: median {int(np.median(NN))}", flush=True)

    # ---- yearly maps, walk-forward (only periods whose target ends before Jan 1 of the year)
    maps, linkrows = {}, []
    for Y in range(START.year, cal[-1].year + 1):
        J = np.array([j for j in range(len(b) - 2) if cal[b[j + 2]] < pd.Timestamp(f"{Y}-01-01")])
        X0, Y1, m = RR[J], RR[J + 1], MM[J]; rows = []
        for B in range(nI):
            yb, ob = Y1[:, B], X0[:, B]; base = ~np.isnan(yb) & ~np.isnan(ob)
            if base.sum() < MINOBS:
                continue
            for A in range(nI):
                if A == B:
                    continue
                ok = base & ~np.isnan(X0[:, A]); n = int(ok.sum())
                if n < MINOBS:
                    continue
                Z = np.column_stack([np.ones(n), X0[ok, A], ob[ok], m[ok]]); ZZ = Z.T @ Z
                if np.linalg.matrix_rank(ZZ) < 4:
                    continue
                iZ = np.linalg.inv(ZZ); be = iZ @ Z.T @ yb[ok]; e = yb[ok] - Z @ be; se = math.sqrt(max(e @ e / (n - 4) * iZ[1, 1], 1e-300))
                rows.append((A, B, be[1], be[1] / se, 2 * stats.t.sf(abs(be[1] / se), n - 4), n))
        R = pd.DataFrame(rows, columns=["A", "B", "beta", "t", "p", "n"]); R["sig"] = bh(R["p"].to_numpy(), Q)
        sig = R[R["sig"]]; maps[Y] = {B: list(zip(grp["A"], grp["beta"])) for B, grp in sig.groupby("B")}
        linkrows += [dict(year=Y, leader=inds[a], follower=inds[bb], beta=round(be_, 4), t=round(t_, 2), p=p_, n=n_) for a, bb, be_, t_, p_, n_, _ in sig.itertuples(index=False)]
        print(f"map {Y}: periods {len(J)} · pairs tested {len(R)} · significant links {len(sig)} (followers {sig['B'].nunique()}, + {int((sig['beta'] > 0).sum())} / - {int((sig['beta'] < 0).sum())})", flush=True)
    LK = pd.DataFrame(linkrows)
    pers = {}
    for Y in range(START.year + 1, cal[-1].year + 1):
        a = {(r.leader, r.follower) for r in LK[LK["year"] == Y - 1].itertuples()} if len(LK) else set()
        c = {(r.leader, r.follower) for r in LK[LK["year"] == Y].itertuples()} if len(LK) else set()
        pers[Y] = dict(links=len(c), kept_from_prev=len(a & c), jaccard=round(len(a & c) / len(a | c), 3) if a | c else None)

    sc_cache = {}

    def lead(d):
        """{industry code: lead score} at list date d (map of d's year); past 4 weeks = close d-20 -> close d."""
        if d not in sc_cache:
            i = tpos[d]; past, _, _ = ind_ret(i - PER, i); mp = maps.get(d.year, {}); out = {}
            for B, lks in mp.items():
                v = [be * past[A] for A, be in lks if not np.isnan(past[A])]
                if v:
                    out[B] = float(np.sum(v))
            sc_cache[d] = (out, past)
        return sc_cache[d]

    # ---- Part A: out-of-sample weekly IC (schedule 0)
    ics = []
    for d in weeks[0]:
        i = tpos[d]
        if i + PER > T - 1:
            continue
        sc, past = lead(d); nxt, _, _ = ind_ret(i, i + PER)
        ks = [k for k in sc if not np.isnan(nxt[k])]
        own = [k for k in range(nI) if not np.isnan(past[k]) and not np.isnan(nxt[k])]
        row = dict(d=d, n_scored=len(ks), ic=np.nan, own_ic_same=np.nan, own_ic_all=np.nan)
        if len(ks) >= 5:
            row["ic"] = stats.spearmanr([sc[k] for k in ks], nxt[ks]).statistic
            row["own_ic_same"] = stats.spearmanr(past[ks], nxt[ks]).statistic
        if len(own) >= 5:
            row["own_ic_all"] = stats.spearmanr(past[own], nxt[own]).statistic
        ics.append(row)
    IC = pd.DataFrame(ics); ICv = IC.dropna(subset=["ic"])
    A_all = mean_boot(ICv["ic"]) if len(ICv) >= 26 else (np.nan, np.nan, np.nan)
    era = {e: float(ICv.loc[(ICv["d"].dt.year <= 2022) == (e == "2019-22"), "ic"].mean()) for e in ("2019-22", "2023+")}
    ic_y = ICv.groupby(ICv["d"].dt.year)["ic"].agg(["mean", "size"]).round(3)
    A_pass = bool(len(ICv) >= 26 and A_all[0] > 0 and A_all[1] > 0 and era["2019-22"] > 0 and era["2023+"] > 0)

    # ---- Part B: widened gate
    G1 = ctx["G1"][["date", "industry"]]; add_rows = []

    def added(d):
        sc, _ = lead(d)
        if not sc:
            return []
        k = math.ceil(TOP * len(sc)); top = sorted(sc, key=lambda x: (-sc[x], inds[x]))[:k]; out = []
        for B in top:
            key = (d, inds[B])
            if key in SI:
                h, pp = SI[key]
                if h < 0.70 and not (pp < 0.30):
                    out.append((inds[B], sc[B], float(h)))
        return out
    elig, addset = {}, {}
    for o in range(5):
        ex = []
        for d in weeks[o]:
            for ind, s_, h in added(d):
                ex.append(dict(date=d, industry=ind)); addset.setdefault(d, set()).add(ind)
                if o == 0:
                    add_rows.append(dict(date=d, industry=ind, lead_score=round(s_, 4), heat_pct=round(h, 3)))
        elig[o] = pd.concat([G1, pd.DataFrame(ex, columns=["date", "industry"])]).drop_duplicates(["date", "industry"]) if ex else G1
    AD = pd.DataFrame(add_rows, columns=["date", "industry", "lead_score", "heat_pct"])

    def lists(o, el):
        return {d: [s for s in l if keep(s, d)] for d, l in v3_rule.ranked_v31(F1, weeks[o], imap, P, el, 9).items()}
    ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[0], imap, P, ctx).items()}
    base = lists(0, G1); same = sum(base[d] == ref[d] for d in weeks[0])
    print(f"wiring check: empty link set (G1 only) == v3_rule V3.1 minus trade-for-trade on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}
    _, run_open = engines(); ARMS = ("V3.2", "E1A")
    res, opn, navs, ph, pk0 = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {}, [], {}
    for o in range(5):
        for a in ARMS:
            pk = lists(o, G1 if a == "V3.2" else elig[o]); pk0[(a, o)] = pk
            n = run_cost(D, X, pk, weeks[o], "E0", cost)[0]; navs[(a, o)] = n; res[a].append(met(n))
            opn[a].append(met(run_open(D, X, pk, weeks[o], "E0", cost, 0.0, 0.0)[0]))
        ph.append(bool(sp.beats(res["E1A"][o], res["V3.2"][o])))
        print(f"schedule {o}: " + " · ".join(f"{a} {res[a][o]['cagr_disc']:.1f}/{res[a][o]['cagr_conf']:.1f}/{res[a][o]['maxdd']:.1f}" for a in ARMS), flush=True)
    passed = bool(ph[0] and sum(ph) >= 4)
    bs = [boot(navs[("E1A", o)], navs[("V3.2", o)]) for o in range(5)]
    yrs = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}

    # ---- what changed (schedule 0): picks from added industries and their 6-month returns
    def r6(s, d):
        q = D["cidx"].get(s); i0 = tpos[d]
        if q is None or i0 + HOLD > T - 1:
            return np.nan
        gg = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
        for k, f in vr.get(q, ()):
            if i0 + 1 <= k <= i0 + HOLD - 1:
                gg = f * gg + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
        return ((1 - float(cost(i0, np.array([q]))[0])) * gg - 1) * 100
    E, V = pk0[("E1A", 0)], pk0[("V3.2", 0)]
    changed = sum(E[d] != V[d] for d in weeks[0])
    newp = [(d, s) for d in weeks[0] for s in E[d] if imap.get(s) in addset.get(d, set())]
    outp = [(d, s) for d in weeks[0] for s in V[d] if s not in E[d]]
    summ = lambda lst: dict(picks=len(lst), avg6m=round(float(np.nanmean([r6(s, d) for d, s in lst])), 1) if lst else None)  # noqa: E731
    allv = summ([(d, s) for d in weeks[0] for s in V[d]])
    WI = dict(weeks=len(weeks[0]), weeks_with_added_industry=sum(1 for d in weeks[0] if addset.get(d)), lists_changed=changed,
              picks_from_added=summ(newp), picks_pushed_out=summ(outp), v32_all_picks=allv,
              share_of_E1A_picks_from_added=round(len(newp) / max(1, sum(len(E[d]) for d in weeks[0])) * 100, 1),
              added_with_negative_score=int((AD["lead_score"] < 0).sum()) if len(AD) else 0, added_rows=len(AD))

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} ===")
    print("Part A · lead score vs next 4 weeks, rank correlation per week (out of sample):")
    print(f"  weeks with >= 5 scored industries {len(ICv)}/{len(IC)} · mean IC {A_all[0]:+.3f} (95% {A_all[1]:+.3f}..{A_all[2]:+.3f}) · 2019-22 {era['2019-22']:+.3f} · 2023+ {era['2023+']:+.3f} · {'PASS' if A_pass else 'no pass'}")
    print(f"  own 4-week momentum IC, same industries {ICv['own_ic_same'].mean():+.3f} · all industries {IC['own_ic_all'].mean():+.3f}")
    print("  by year:\n" + ic_y.to_string())
    print("  link persistence: " + " · ".join(f"{y}: {v['links']} links, {v['kept_from_prev']} kept" for y, v in pers.items()))
    print("Part B · E1A (G1 + about-to-get-hot industries) vs V3.2, schedule 0 (years = avg of 5):")
    for a in ARMS:
        r = res[a][0]
        print(f"  {a:4s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs[a][y]:+.0f}" for y in YEARS) + f" · next open {opn[a][0]['cagr']:.1f}")
    print(f"  E1A vs V3.2: beats in {sum(ph)}/5 · gap {bs[0][0]:+.1f} pts/yr (95% {bs[0][1]:+.1f}..{bs[0][2]:+.1f}) · range excludes 0 in {sum(lo > 0 or hi < 0 for _, lo, hi in bs)}/5 · {'PASS' if passed else 'no pass'}")
    print("  what changed: " + json.dumps(WI))
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(arm=a, **{c: round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs[a][y] for y in YEARS},
                       next_open=round(opn[a][0]["cagr"], 1)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    LK.to_csv(OUTDIR / "links.csv", index=False); IC.to_csv(OUTDIR / "ic_weekly.csv", index=False); AD.to_csv(OUTDIR / "added_industries.csv", index=False)
    files = {"results.csv": ("one row per arm, schedule 0; years = average of 5 schedules", dict(cagr="percent a year", maxdd="percent")),
             "links.csv": ("significant lead-lag links per map year (estimated on earlier years only)", dict(beta="follower next-4-week return per unit of leader past-4-week return (fractions)", p="two-sided p-value before FDR")),
             "ic_weekly.csv": ("weekly Spearman rank correlation, schedule 0", dict(ic="lead score vs next-4-week industry return", own_ic_same="own past 4 weeks vs next 4 weeks, same industries", own_ic_all="same, all industries")),
             "added_industries.csv": ("industries added to the G1 gate per list date, schedule 0", dict(lead_score="fraction (sum of beta x leader past-4-week return)", heat_pct="percentile 0-1"))}
    for f, (what, units) in files.items():
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_industry_leadlag.py", definitions=__doc__, contents=what, units=units, updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nIndustry lead-lag map, walk-forward. Part A (does the map predict industries' next 4 weeks?): {'PASS' if A_pass else 'no pass'}, "
                                      f"mean weekly IC {A_all[0]:+.3f}. Part B (add 'about to get hot' industries to V3.2's gate): {'PASS' if passed else 'no pass'}. "
                                      "Files: results.csv, links.csv, ic_weekly.csv, added_industries.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=dict(part_A="PASS" if A_pass else "no pass", part_B="PASS" if passed else "no pass"),
                                 part_A=dict(weeks=len(ICv), mean_ic=round(A_all[0], 4), ci95=[round(A_all[1], 4), round(A_all[2], 4)], eras={k: round(v, 4) for k, v in era.items()},
                                             own_ic_same=round(float(ICv["own_ic_same"].mean()), 4), own_ic_all=round(float(IC["own_ic_all"].mean()), 4),
                                             by_year=ic_y.reset_index().to_dict("records"), persistence=pers),
                                 arms={a: [round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS}, phases=f"{sum(ph)}/5",
                                 bootstrap=bs, calendar_years=yrs, next_open={a: round(opn[a][0]["cagr"], 1) for a in ARMS}, what_changed=WI,
                                 cols="CAGR, 2019-22, 2023+, maxDD"), default=str) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
