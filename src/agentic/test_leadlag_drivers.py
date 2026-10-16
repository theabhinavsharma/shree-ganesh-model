"""EXP-2026-10-16-commodity-leadlag (E3) and EXP-2026-10-16-cluster-leadlag (E6), registered in logs/experiments.jsonl first.

Usage: /usr/bin/python3 src/agentic/test_leadlag_drivers.py commodity | cluster
Plain English: the industry lead-lag method of test_industry_leadlag.py (E1) with different leaders / followers.
commodity: leaders = 12 commodity and currency series (FRED, point in time by known_from), followers = industries; 'about to get
  hot' industries (lead score top 10%, heat_pct < 0.70, not policy-vetoed) join V3.2's G1 gate.
cluster: each January, liquid stocks are grouped into 30 clusters by how they moved together over the previous 2 years
  (Ward clustering of z-scored residual weekly returns); leaders and followers = clusters; members of the top-10% clusters by lead
  score join V3.2's eligible set (except stocks whose industry is policy-vetoed that date).
Maps are found each January from earlier years only (4-week periods from 2016, OLS controlling for the follower's own past 4
weeks and the market's, Benjamini-Hochberg q <= 0.10). Part A = weekly rank correlation of lead score vs next 4 weeks; Part B =
the widened V3.2 vs V3.2: both eras, maxDD not worse by > 2 points, schedule 0 and >= 4 of 5.
"""
import json, math, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
from test_v32_hold_after_tax import engines, boot  # noqa: E402
from test_industry_leadlag import bh, mean_boot, START, HOLD, YEARS, EST0, PER, MINMEM, MINOBS, Q, TOP  # noqa: E402
MODES = {"commodity": "EXP-2026-10-16-commodity-leadlag", "cluster": "EXP-2026-10-16-cluster-leadlag"}
DRIVERS = ["brent", "henry_hub_gas", "usdinr", "broad_dollar", "copper", "aluminium", "zinc", "nickel", "iron_ore", "sugar", "cotton", "wheat"]
K_CL, CL_MIN, CL_LIQ, WK_N, WK_MIN = 30, 5, 3, 104, 90


def fit_pairs(Xl, Yf, Xf, m, same):
    """OLS of follower next period on leader past period + follower own past + market past, every (leader, follower) pair.
    Xl: leader past (periods x nL); Yf / Xf: follower next / own past (periods x nF); same(a, b) -> skip pair."""
    rows = []
    for B in range(Yf.shape[1]):
        yb, ob = Yf[:, B], Xf[:, B]; base = ~np.isnan(yb) & ~np.isnan(ob) & ~np.isnan(m)
        if base.sum() < MINOBS:
            continue
        for A in range(Xl.shape[1]):
            if same(A, B):
                continue
            ok = base & ~np.isnan(Xl[:, A]); n = int(ok.sum())
            if n < MINOBS:
                continue
            Z = np.column_stack([np.ones(n), Xl[ok, A], ob[ok], m[ok]]); ZZ = Z.T @ Z
            if np.linalg.matrix_rank(ZZ) < 4:
                continue
            iZ = np.linalg.inv(ZZ); be = iZ @ Z.T @ yb[ok]; e = yb[ok] - Z @ be; se = math.sqrt(max(e @ e / (n - 4) * iZ[1, 1], 1e-300))
            rows.append((A, B, be[1], be[1] / se, 2 * stats.t.sf(abs(be[1] / se), n - 4), n))
    R = pd.DataFrame(rows, columns=["A", "B", "beta", "t", "p", "n"]); R["sig"] = bh(R["p"].to_numpy(), Q)
    return R


def main(mode):
    t0 = time.time(); EXP_ID = MODES[mode]; OUTDIR = ROOT / "logs/leader_sleeve" / f"{mode}_leadlag"
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); tpos = {d: i for i, d in enumerate(cal)}; syms = D["syms"]; acal = np.asarray(cal)
    Sc = ctx["scores"]; SI = {(d, i): (h, q) for d, i, h, q in Sc[["date", "industry", "heat_pct", "P_pct"]].itertuples(index=False)}
    VET = set(zip(ctx["vetoed"]["date"], ctx["vetoed"]["industry"]))
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    L["liq"] = ((L["adv"] >= v3_rule.V31_ADV_CR) & (rp.raw_price(L, "close") > v3_rule.V31_PRICE)).astype(float)
    ADV = rp.wide(L, "adv", cal).reindex(columns=syms).to_numpy(); LIQ = rp.wide(L, "liq", cal).reindex(columns=syms).fillna(0).to_numpy() > 0; del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    keep = lambda s, d: ctx["keep"](s, d) and not t2t(s, d)  # noqa: E731
    Rv = D["Rv"]; lg = np.vstack([np.zeros((1, Rv.shape[1])), np.cumsum(np.log1p(np.nan_to_num(Rv)), axis=0)]); vr = D.get("vr", {})
    inds = sorted(imap.dropna().unique()); nI = len(inds); icode = {s: i for i, s in enumerate(inds)}
    g = np.array([icode.get(imap.get(s), -1) for s in syms])

    def grp_ret(i0, i1, code, n, minm):
        """(group returns close i0 -> close i1 for group codes 0..n-1 of members liquid at i0, NaN under minm; market return)."""
        r = np.exp(lg[i1 + 1] - lg[i0 + 1]) - 1; mk = LIQ[i0] & (code >= 0)
        cnt = np.bincount(code[mk], minlength=n); sm = np.bincount(code[mk], weights=r[mk], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt >= minm, sm / cnt, np.nan), float(r[LIQ[i0]].mean()), cnt
    b = list(range(int(np.searchsorted(acal, np.datetime64(EST0))), T, PER)); nP = len(b) - 1
    RR, MM, NN = [], [], []
    for j in range(nP):
        r, m, c = grp_ret(b[j], b[j + 1], g, nI, MINMEM); RR.append(r); MM.append(m); NN.append(int((c >= MINMEM).sum()))
    RR, MM = np.array(RR), np.array(MM)
    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    Jy = lambda Y: np.array([j for j in range(nP - 1) if cal[b[j + 2]] < pd.Timestamp(f"{Y}-01-01")])  # noqa: E731
    years = range(START.year, cal[-1].year + 1)

    # ---- data gate (no outcomes yet)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh_rows = dr.bhav_rows()
    per = pd.DataFrame(dict(d=[cal[b[j + 1]] for j in range(nP)], ind20=[n if n >= 20 else np.nan for n in NN]))
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh_rows.groupby(bh_rows.index.year).sum(), range(2016, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.span("industry scores", Sc["date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.coverage("4-week periods with >= 20 industries of >= 5 liquid members", per, "d", ["ind20"], range(2016, cal[-1].year + 1), min_share=0.9)]
    if mode == "commodity":
        F = pd.read_parquet(ROOT / "data/derived/macro_drivers_fred.parquet"); F["known_from"] = pd.to_datetime(F["known_from"])
        DV = {}
        for k in DRIVERS:
            f = F[F["series"] == k].dropna(subset=["value"]).sort_values("known_from"); f = f[f["value"] > 0]
            DV[k] = (f["known_from"].to_numpy(), f["value"].to_numpy(float)); daily = f["freq"].iloc[-1] == "D"
            checks.append(dr.span(f"{k} ({'daily' if daily else 'monthly'}, known_from)", f["known_from"], EST0 - pd.Timedelta(days=60), cal[-1] - pd.Timedelta(days=10 if daily else 75)))
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px
    print(f"periods {nP} · industries {nI} (median with >= {MINMEM} liquid members {int(np.median(NN))})", flush=True)

    # ---- leaders / followers per mode
    maps, linkrows, info = {}, [], {}
    if mode == "commodity":
        def dval(k, d):
            a, v = DV[k]; i = np.searchsorted(a, np.datetime64(d), side="right") - 1
            return v[i] if i >= 0 else np.nan
        def dchg(i0, i1):
            return np.array([math.log(dval(k, cal[i1]) / dval(k, cal[i0])) if np.isfinite(dval(k, cal[i0])) and np.isfinite(dval(k, cal[i1])) else np.nan for k in DRIVERS])
        DC = np.array([dchg(b[j], b[j + 1]) for j in range(nP)])
        lname = lambda a: DRIVERS[a]; fname = lambda f, Y: inds[f]  # noqa: E731
        for Y in years:
            J = Jy(Y); R = fit_pairs(DC[J], RR[J + 1], RR[J], MM[J], lambda a, bb: False); maps[Y] = R
        def lead(d):
            i = tpos[d]; past = dchg(i - PER, i); out = {}
            for B, gr in maps[d.year][maps[d.year]["sig"]].groupby("B"):
                v = [be * past[A] for A, be in zip(gr["A"], gr["beta"]) if np.isfinite(past[A])]
                if v:
                    out[B] = float(np.sum(v))
            return out
        def fnext(d, Y):
            i = tpos[d]; return grp_ret(i, i + PER, g, nI, MINMEM)[0], grp_ret(i - PER, i, g, nI, MINMEM)[0]
    else:
        cl = {}
        cv = np.vstack([np.zeros((1, Rv.shape[1])), np.cumsum(~np.isnan(Rv), axis=0)])
        purity = {}
        for Y in years:
            iY = int(np.searchsorted(acal, np.datetime64(f"{Y}-01-01"))) - 1; w = [iY - 5 * k for k in range(WK_N + 1)][::-1]
            sel = np.nonzero(LIQ[iY] & (g >= 0))[0]
            W = np.array([lg[w[k + 1] + 1, sel] - lg[w[k] + 1, sel] for k in range(WK_N)])            # weeks x stocks, log returns
            pres = np.array([(cv[w[k + 1] + 1, sel] - cv[w[k] + 1, sel]) >= 4 for k in range(WK_N)])
            okc = pres.sum(0) >= WK_MIN; sel, W, pres = sel[okc], W[:, okc], pres[:, okc]
            W = np.where(pres, W, np.nan); res = W - np.nanmean(W, axis=1, keepdims=True)
            sd = np.nanstd(res, 0); Z = np.nan_to_num((res - np.nanmean(res, 0)) / np.where(sd > 0, sd, np.nan))
            lab = fcluster(linkage(Z.T, "ward"), K_CL, criterion="maxclust") - 1
            sizes = np.bincount(lab, minlength=K_CL); keepc = np.nonzero(sizes >= CL_MIN)[0]; remap = {c: k for k, c in enumerate(keepc)}
            code = np.full(len(syms), -1); code[sel] = [remap.get(c, -1) for c in lab]; nC = len(keepc)
            CR = np.array([grp_ret(b[j], b[j + 1], code, nC, CL_LIQ)[0] for j in range(nP)])
            J = Jy(Y); maps[Y] = fit_pairs(CR[J], CR[J + 1], CR[J], MM[J], lambda a, bb: a == bb); cl[Y] = (code, nC)
            pur = [pd.Series(g[code == k]).value_counts(normalize=True).iloc[0] for k in range(nC)]
            purity[Y] = dict(stocks=int(len(sel)), clusters=nC, median_size=int(np.median(sizes[keepc])), purity_median=round(float(np.median(pur)), 2))
        info["clusters"] = purity
        lname = lambda a: f"C{a}"; fname = lambda f, Y: f"C{f}"  # noqa: E731
        def lead(d):
            i = tpos[d]; code, nC = cl[d.year]; past = grp_ret(i - PER, i, code, nC, CL_LIQ)[0]; out = {}
            for B, gr in maps[d.year][maps[d.year]["sig"]].groupby("B"):
                v = [be * past[A] for A, be in zip(gr["A"], gr["beta"]) if np.isfinite(past[A])]
                if v:
                    out[B] = float(np.sum(v))
            return out
        def fnext(d, Y):
            i = tpos[d]; code, nC = cl[Y]; return grp_ret(i, i + PER, code, nC, CL_LIQ)[0], grp_ret(i - PER, i, code, nC, CL_LIQ)[0]
    for Y in years:
        R = maps[Y]; sg = R[R["sig"]]
        linkrows += [dict(year=Y, leader=lname(a), follower=fname(bb, Y), beta=round(be, 4), t=round(t, 2), p=p, n=n) for a, bb, be, t, p, n, _ in sg.itertuples(index=False)]
        info.setdefault("maps", {})[Y] = dict(pairs=len(R), links=len(sg), p05_share=round(float((R["p"] < 0.05).mean()) * 100, 1) if len(R) else float("nan"),
                                              min_p=float(R["p"].min()) if len(R) else float("nan"))
        print(f"map {Y}: periods {len(Jy(Y))} · pairs tested {len(R)} · significant links {len(sg)} · p<0.05 share {info['maps'][Y]['p05_share']}% · smallest p {info['maps'][Y]['min_p']:.1e}"
              + (f" · clusters {info['clusters'][Y]}" if mode == "cluster" else ""), flush=True)
    LK = pd.DataFrame(linkrows, columns=["year", "leader", "follower", "beta", "t", "p", "n"])
    sc_cache = {}
    def leadc(d):
        if d not in sc_cache:
            sc_cache[d] = lead(d)
        return sc_cache[d]

    # ---- Part A: weekly IC (schedule 0), out of sample
    ics = []
    for d in weeks[0]:
        if tpos[d] + PER > T - 1:
            continue
        sc = leadc(d); nxt, past = fnext(d, d.year); ks = [k for k in sc if np.isfinite(nxt[k])]
        row = dict(d=d, n_scored=len(ks), ic=np.nan, own_ic_same=np.nan)
        if len(ks) >= 5:
            row["ic"] = stats.spearmanr([sc[k] for k in ks], nxt[ks]).statistic; row["own_ic_same"] = stats.spearmanr(past[ks], nxt[ks]).statistic
        ics.append(row)
    IC = pd.DataFrame(ics); ICv = IC.dropna(subset=["ic"])
    A_all = mean_boot(ICv["ic"]) if len(ICv) >= 26 else (np.nan, np.nan, np.nan)
    era = {e: float(ICv.loc[(ICv["d"].dt.year <= 2022) == (e == "2019-22"), "ic"].mean()) if len(ICv) else np.nan for e in ("2019-22", "2023+")}
    A_pass = bool(len(ICv) >= 26 and A_all[0] > 0 and A_all[1] > 0 and era["2019-22"] > 0 and era["2023+"] > 0)

    # ---- Part B: widened eligibility
    G1 = ctx["G1"][["date", "industry"]]; add_rows = []

    def ranked_ext(dates, extra, topn=9):
        """v3_rule.ranked_v31 with eligibility = G1 industries OR (date, symbol) in extra (stock level)."""
        W = F1[F1["trade_date"].isin(set(dates)) & F1["core"]].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W.merge(G1.rename(columns={"date": "trade_date", "industry": "ind"}).assign(_g1=True), on=["trade_date", "ind"], how="left")
        W = W[W["_g1"].fillna(False).astype(bool) | pd.Series([(d, s) in extra for d, s in zip(W["trade_date"], W["symbol"])], index=W.index)]
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])) | W["young"]
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        gg = Pool.groupby("trade_date").head(topn).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: gg.get(d, []) for d in dates}

    def added(d):
        sc = leadc(d)
        if not sc:
            return []
        top = sorted(sc, key=lambda x: (-sc[x], x))[:math.ceil(TOP * len(sc))]
        if mode == "commodity":
            out = []
            for B in top:
                hp = SI.get((d, inds[B]))
                if hp is not None and hp[0] < 0.70 and not (hp[1] < 0.30):
                    out.append((inds[B], sc[B]))
            return out
        code, _ = cl[d.year]
        return [(syms[q], sc[code[q]]) for B in top for q in np.nonzero(code == B)[0] if (d, imap.get(syms[q])) not in VET]

    def lists(o, widen):
        if mode == "commodity":
            ex = [dict(date=d, industry=i) for d in weeks[o] for i, _ in (added(d) if widen else [])]
            el = pd.concat([G1, pd.DataFrame(ex, columns=["date", "industry"])]).drop_duplicates(["date", "industry"]) if ex else G1
            R = v3_rule.ranked_v31(F1, weeks[o], imap, P, el, 9)
        else:
            R = ranked_ext(weeks[o], {(d, s) for d in weeks[o] for s, _ in (added(d) if widen else [])})
        return {d: [s for s in l if keep(s, d)] for d, l in R.items()}
    for d in weeks[0]:
        for k, s_ in added(d):
            add_rows.append(dict(date=d, added=k, lead_score=round(s_, 4)))
    AD = pd.DataFrame(add_rows, columns=["date", "added", "lead_score"])
    ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[0], imap, P, ctx).items()}
    base = lists(0, False) if mode == "commodity" else {d: [s for s in l if keep(s, d)] for d, l in ranked_ext(weeks[0], set()).items()}
    same = sum(base[d] == ref[d] for d in weeks[0])
    print(f"wiring check: nothing added == v3_rule V3.1 minus trade-for-trade on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}
    _, run_open = engines(); arm = "E3A" if mode == "commodity" else "E6A"; ARMS = ("V3.2", arm)
    res, opn, navs, ph, pk0 = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {}, [], {}
    for o in range(5):
        for a in ARMS:
            pk = lists(o, a != "V3.2"); pk0[(a, o)] = pk
            n = run_cost(D, X, pk, weeks[o], "E0", cost)[0]; navs[(a, o)] = n; res[a].append(met(n))
            opn[a].append(met(run_open(D, X, pk, weeks[o], "E0", cost, 0.0, 0.0)[0]))
        ph.append(bool(sp.beats(res[arm][o], res["V3.2"][o])))
        print(f"schedule {o}: " + " · ".join(f"{a} {res[a][o]['cagr_disc']:.1f}/{res[a][o]['cagr_conf']:.1f}/{res[a][o]['maxdd']:.1f}" for a in ARMS), flush=True)
    passed = bool(ph[0] and sum(ph) >= 4)
    bs = [boot(navs[(arm, o)], navs[("V3.2", o)]) for o in range(5)]
    yrs = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}

    def r6(s, d):
        q = D["cidx"].get(s); i0 = tpos[d]
        if q is None or i0 + HOLD > T - 1:
            return np.nan
        gg = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
        for k, f in vr.get(q, ()):
            if i0 + 1 <= k <= i0 + HOLD - 1:
                gg = f * gg + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
        return ((1 - float(cost(i0, np.array([q]))[0])) * gg - 1) * 100
    E, V = pk0[(arm, 0)], pk0[("V3.2", 0)]
    newp = [(d, s) for d in weeks[0] for s in E[d] if s not in V[d]]; outp = [(d, s) for d in weeks[0] for s in V[d] if s not in E[d]]
    summ = lambda lst: dict(picks=len(lst), avg6m=round(float(np.nanmean([r6(s, d) for d, s in lst])), 1) if lst else None)  # noqa: E731
    WI = dict(weeks=len(weeks[0]), weeks_with_additions=int(AD["date"].nunique()) if len(AD) else 0, lists_changed=sum(E[d] != V[d] for d in weeks[0]),
              picks_new=summ(newp), picks_pushed_out=summ(outp), v32_all_picks=summ([(d, s) for d in weeks[0] for s in V[d]]))

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} ===")
    print(f"Part A · weeks with >= 5 scored {'industries' if mode == 'commodity' else 'clusters'} {len(ICv)}/{len(IC)} · mean IC {A_all[0]:+.3f} (95% {A_all[1]:+.3f}..{A_all[2]:+.3f}) · "
          f"2019-22 {era['2019-22']:+.3f} · 2023+ {era['2023+']:+.3f} · own momentum IC (same) {ICv['own_ic_same'].mean() if len(ICv) else np.nan:+.3f} · {'PASS' if A_pass else 'no pass'}")
    if len(ICv):
        print("  by year: " + " · ".join(f"{y}: {v:+.3f}" for y, v in ICv.groupby(ICv['d'].dt.year)['ic'].mean().items()))
    for a in ARMS:
        r = res[a][0]
        print(f"  {a:4s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs[a][y]:+.0f}" for y in YEARS) + f" · next open {opn[a][0]['cagr']:.1f}")
    print(f"  {arm} vs V3.2: beats in {sum(ph)}/5 · gap {bs[0][0]:+.1f} pts/yr (95% {bs[0][1]:+.1f}..{bs[0][2]:+.1f}) · range excludes 0 in {sum(lo > 0 or hi < 0 for _, lo, hi in bs)}/5 · {'PASS' if passed else 'no pass'}")
    print("  what changed: " + json.dumps(WI))
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(arm=a, **{c: round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs[a][y] for y in YEARS},
                       next_open=round(opn[a][0]["cagr"], 1)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    LK.to_csv(OUTDIR / "links.csv", index=False); IC.to_csv(OUTDIR / "ic_weekly.csv", index=False); AD.to_csv(OUTDIR / "added.csv", index=False)
    files = {"results.csv": ("one row per arm, schedule 0; years = average of 5 schedules", dict(cagr="percent a year", maxdd="percent")),
             "links.csv": ("significant lead-lag links per map year (estimated on earlier years only)", dict(beta="follower next-4-week return per unit of leader past-4-week change (fractions; commodity changes are log changes)", p="two-sided p-value before FDR")),
             "ic_weekly.csv": ("weekly Spearman rank correlation, schedule 0", dict(ic="lead score vs next-4-week return", own_ic_same="own past 4 weeks vs next 4 weeks, same groups")),
             "added.csv": ("industries (commodity) or stocks (cluster) added to V3.2's eligible set per list date, schedule 0", dict(lead_score="fraction"))}
    for f, (what, units) in files.items():
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer=f"src/agentic/test_leadlag_drivers.py {mode}", definitions=__doc__, contents=what, units=units, updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nPart A (lead score predicts the next 4 weeks?): {'PASS' if A_pass else 'no pass'}, mean weekly IC {A_all[0]:+.3f} over {len(ICv)} weeks. "
                                      f"Part B ({arm} vs V3.2): {'PASS' if passed else 'no pass'}. Links per year: " + ", ".join(f"{y} {v['links']}" for y, v in info["maps"].items())
                                      + ". Files: results.csv, links.csv, ic_weekly.csv, added.csv; run log run.log.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=dict(part_A="PASS" if A_pass else "no pass", part_B="PASS" if passed else "no pass"),
                                 part_A=dict(weeks=len(ICv), mean_ic=A_all[0], ci95=[A_all[1], A_all[2]], eras=era), info=info,
                                 arms={a: [round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS}, phases=f"{sum(ph)}/5",
                                 bootstrap=bs, calendar_years=yrs, next_open={a: round(opn[a][0]["cagr"], 1) for a in ARMS}, what_changed=WI,
                                 cols="CAGR, 2019-22, 2023+, maxDD"), default=str) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
