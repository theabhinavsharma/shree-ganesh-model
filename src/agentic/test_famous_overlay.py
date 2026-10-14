"""Famous-hypothesis tests that flag stocks (configs/famous_hypotheses.yaml; each REGISTERED in logs/experiments.jsonl first).

Usage: /usr/bin/python3 src/agentic/test_famous_overlay.py promoter_pref_premium | kedia_small_size | promoter_pref_any | orderbook_mcap
Plain English: for a flag (e.g. 'promoter took preferential shares at a premium in the last 90 days', 'market cap <= Rs 1,000 cr'):
A (evidence) - each stock's 6-month return after costs on every weekly list since Apr 2019, flagged vs not, in all liquid stocks,
the V3.1 pool and the pool's top 25; verdict = top-25 weekly gap with a block-bootstrap (26-week blocks) 95% range above 0.
B (adoption) - flagged names first within the pool's top 25, then V3.2 (filters, trade-for-trade drop, no refill); sp.beats vs V3.2
at schedule 0 and >= 4 of 5 (test_v3_l1_newlist_cost.run_cost). The registration holds each flag's exact definition.
"""
import json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, eval_checks, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
START, HOLD = pd.Timestamp("2019-04-01"), 126
HYP = {"promoter_pref_premium": dict(exp="EXP-2026-10-15-promoter-pref-premium", main="PREM", extra="PREF"),
       "kedia_small_size": dict(exp="EXP-2026-10-15-kedia-small-size", main="SMALL", extra="SMALL500"),
       "promoter_pref_any": dict(exp="EXP-2026-10-15-promoter-pref-any", main="PREF", extra="PREM"),
       "orderbook_mcap": dict(exp="EXP-2026-10-15-orderbook-to-mcap", main="OBM1", extra="OBM2", pool=True)}


def promoter_flags(cal, tpos):
    p = pd.read_parquet(ROOT / "data/derived/pit_history.parquet")
    pf = p[p["personCategory"].fillna("").str.contains("Promoter", case=False) & p["acqMode"].fillna("").str.contains("Preferential", case=False)].copy()
    pf["symbol"] = pf["symbol"].astype(str)
    pf["d"] = pd.to_datetime(pf["date"], format="%d-%b-%Y %H:%M", errors="coerce").dt.normalize()
    pf["px"] = pd.to_numeric(pf["secVal"], errors="coerce") / pd.to_numeric(pf["secAcq"], errors="coerce").replace(0, np.nan)
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "close", "price_adjustment_factor_to_present"], filters=[("symbol", "in", pf["symbol"].unique().tolist())])
    S["symbol"] = S["symbol"].astype(str); S["trade_date"] = pd.to_datetime(S["trade_date"]); S["raw"] = rp.raw_price(S, "close")
    m = pd.merge_asof(pf.dropna(subset=["d"]).sort_values("d"), S[["symbol", "trade_date", "raw"]].sort_values("trade_date"), left_on="d", right_on="trade_date",
                      by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    m["ratio"] = m["px"] / m["raw"]
    ev = {"PREF": m, "PREM": m[(m["ratio"] >= 1.0) & (m["ratio"] <= 2.0)]}
    out = {k: {s: np.sort(g["d"].unique()) for s, g in v.groupby("symbol")} for k, v in ev.items()}
    info = dict(rows=len(m), price_usable=float(m["ratio"].notna().mean()), prem_rows=len(ev["PREM"]), prem_companies=int(ev["PREM"]["symbol"].nunique()),
                first=str(m["d"].min().date()), last=str(m["d"].max().date()))
    def flag(kind):
        E = out[kind]
        def f(s, d):
            a = E.get(s)
            if a is None:
                return False
            dd = np.datetime64(d); i = np.searchsorted(a, dd, side="right")
            return i > 0 and (dd - a[i - 1]) <= np.timedelta64(90, "D")
        return f
    return {"PREM": flag("PREM"), "PREF": flag("PREF")}, info, m


def orderbook_flags(mc):
    """OBM = latest stated order book (median of statements within 45 days of the latest one, filed before 15:30 IST on the
    list date, latest within 120 days) / point-in-time market cap. OBM > 20 = extraction error (NaN, counted)."""
    B = pd.read_parquet(ROOT / "data/derived/order_book_filings.parquet", columns=["symbol", "filed", "backlog_cr"])
    B["filed"] = pd.to_datetime(B["filed"], errors="coerce"); B = B[(B["backlog_cr"] >= 1) & B["filed"].notna()].sort_values("filed")
    per = {s: (g["filed"].values, g["backlog_cr"].to_numpy(float)) for s, g in B.groupby("symbol")}
    cache, err = {}, [0, 0]
    def obm(s, d):
        k = (s, d)
        if k in cache:
            return cache[k]
        a = per.get(s); r = np.nan
        if a is not None:
            cut = np.datetime64(pd.Timestamp(d) + pd.Timedelta(hours=15, minutes=30)); i = np.searchsorted(a[0], cut, side="left")
            if i > 0 and a[0][i - 1] >= np.datetime64(pd.Timestamp(d) - pd.Timedelta(days=120)):
                j = np.searchsorted(a[0], a[0][i - 1] - np.timedelta64(45, "D"), side="right"); book = float(np.median(a[1][j:i])); m = mc(s, d)
                if np.isfinite(m) and m > 0:
                    r = book / m; err[1] += 1
                    if r > 20:
                        err[0] += 1; r = np.nan
        cache[k] = r
        return r
    info = dict(statements=len(B), companies=int(B["symbol"].nunique()), first=str(B["filed"].min().date()), last=str(B["filed"].max().date()))
    return {"OBM1": lambda s, d: obm(s, d) >= 1, "OBM2": lambda s, d: obm(s, d) >= 2}, info, err, B


def size_flags():
    A = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "mcap_cr"])
    A["trade_date"] = pd.to_datetime(A["trade_date"]); A = A.dropna(subset=["mcap_cr"])
    M = {s: (g["trade_date"].values, g["mcap_cr"].values) for s, g in A.sort_values("trade_date").groupby("symbol")}
    def mc(s, d):
        a = M.get(s)
        if a is None:
            return np.nan
        i = np.searchsorted(a[0], np.datetime64(d), side="right") - 1
        return a[1][i] if i >= 0 and (np.datetime64(d) - a[0][i]) <= np.timedelta64(7, "D") else np.nan
    return {"SMALL": lambda s, d: mc(s, d) <= 1000, "SMALL500": lambda s, d: mc(s, d) <= 500}, mc


def main(key):
    t0 = time.time(); H = HYP[key]; EXP_ID = H["exp"]; OUTDIR = ROOT / "logs/leader_sleeve/famous" / key
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); tpos = {d: i for i, d in enumerate(cal)}; cidx = D["cidx"]
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)]); vr = D.get("vr", {})
    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    liq = F1[F1["core"].fillna(False).astype(bool) & F1["trade_date"].isin(set(weeks[0]))][["symbol", "trade_date"]]

    # ---- flags + data gate (no outcomes yet)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10))]
    if key.startswith("promoter_pref"):
        flags, info, m = promoter_flags(cal, tpos)
        em = eval_checks.insider_no_empty_month({}); nmon = len(pd.period_range("2019-01", pd.Timestamp.now().to_period("M") - 1, freq="M"))
        checks += [dr.qc("insider history: months 2019-01..last month with disclosures (share)", 1.0 - float(em["value"]) / nmon, 1.0),
                   dr.qc("promoter preferential rows with a usable implied price (share)", info["price_usable"], 0.7),
                   dr.span("promoter preferential disclosures", m["d"], pd.Timestamp("2019-04-30"), cal[-1] - pd.Timedelta(days=60))]
    elif key == "orderbook_mcap":
        _, mc = size_flags(); flags, info, err, Bk = orderbook_flags(mc)
        for d in weeks[0][::4]:                                       # sample list dates for the extraction-error share
            for s in liq.loc[liq["trade_date"] == d, "symbol"]:
                flags["OBM1"](s, d)
        info["error_share"] = round(err[0] / max(1, err[1]), 4); info["valued"] = err[1]
        checks += [dr.span("order-book statements", Bk["filed"], pd.Timestamp("2016-06-30"), cal[-1] - pd.Timedelta(days=60)),
                   dr.qc("order books that are plausible (<= 20x market cap) among valued stock-dates", 1 - info["error_share"], 0.9)]
    else:
        flags, mc = size_flags(); info = {}
        K = pd.DataFrame([dict(d=d, known=(1.0 if pd.notna(mc(s, d)) else np.nan)) for s, d in zip(liq["symbol"], liq["trade_date"])])
        checks += [dr.coverage("market cap known for liquid stocks on list dates", K, "d", ["known"], range(START.year, cal[-1].year + 1))]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px
    main_f, extra_f = flags[H["main"]], flags[H["extra"]]

    def r6(s, d):
        q = cidx.get(s); i0 = tpos[d]
        if q is None or i0 + HOLD > T - 1:
            return np.nan
        g = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
        for k, f in vr.get(q, ()):
            if i0 + 1 <= k <= i0 + HOLD - 1:
                g = f * g + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
        return ((1 - float(cost(i0, np.array([q]))[0])) * g - 1) * 100

    pool0 = v3_rule.ranked_v31(F1, weeks[0], imap, P, ctx["G1"], 10000)
    rows = []
    for d in weeks[0]:
        top25, pool = set(pool0[d][:25]), set(pool0[d])
        for s in liq.loc[liq["trade_date"] == d, "symbol"]:
            r = r6(s, d)
            if not np.isnan(r):
                rows.append(dict(d=d, s=s, f=bool(main_f(s, d)), x=bool(extra_f(s, d)), r=r, pool=s in pool, top25=s in top25))
    A = pd.DataFrame(rows)

    def wk_boot(sub, col, block=26, n=4000, seed=7):
        g = sub.groupby(["d", col])["r"].mean().unstack().dropna()
        if True not in g.columns or len(g) < 2 * block:      # fewer weeks than two blocks: the resample cannot vary (2026-10-15 fix)
            return [np.nan, np.nan, np.nan, len(g)]
        x = (g[True] - g[False]).to_numpy(); Tn = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(Tn / block))
        out = np.array([x[((rng.integers(0, Tn, nb)[:, None] + np.arange(block)[None, :]).ravel()[:Tn]) % Tn].mean() for _ in range(n)])
        return [round(float(x.mean()), 2), round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2), int(Tn)]
    evid = {}
    for col, lab in (("f", H["main"]), ("x", H["extra"])):
        for name, msk in (("all liquid stocks", A["s"].notna()), ("V3.1 pool", A["pool"]), ("pool top 25", A["top25"])):
            sub = A[msk]; v, nv = sub[sub[col]], sub[~sub[col]]
            e = dict(n=len(v), share=round(len(v) / max(1, len(sub)) * 100, 2), avg=round(v.r.mean(), 1), med=round(v.r.median(), 1), hit50=round((v.r >= 50).mean() * 100, 1),
                     lost=round((v.r < 0).mean() * 100, 1), o_avg=round(nv.r.mean(), 1), o_med=round(nv.r.median(), 1), o_hit50=round((nv.r >= 50).mean() * 100, 1), o_lost=round((nv.r < 0).mean() * 100, 1))
            if name == "pool top 25":
                e["weekly_gap"] = wk_boot(sub, col)
            evid[f"{lab} · {name}"] = e
    g = evid[f"{H['main']} · pool top 25"]["weekly_gap"]; a_pass = bool(np.isfinite(g[1]) and g[1] > 0)
    a_text = "YES" if a_pass else ("not enough data (fewer than 52 weeks with a flagged name)" if not np.isfinite(g[1]) else "no")

    def lists(o, first_on):
        R = pool0 if o == 0 else v3_rule.ranked_v31(F1, weeks[o], imap, P, ctx["G1"], 25)
        out = {}
        for d in weeks[o]:
            pool = R[d][:25]
            if first_on:
                fl = [s for s in pool if main_f(s, d)]; pool = fl + [s for s in pool if s not in set(fl)]
            out[d] = [s for s in pool[:9] if ctx["keep"](s, d) and not t2t(s, d)]
        return out
    L0 = lists(0, False); ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[0], imap, P, ctx).items()}
    same = sum(L0[d] == ref[d] for d in weeks[0]); print(f"wiring check: V3.2 rebuilt == v3_rule on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")
    def lists_pool(o):
        R = v3_rule.ranked_v31(F1, weeks[o], imap, P, ctx["G1"], 10000); out = {}
        for d in weeks[o]:
            pool = R[d]; fl = [s for s in pool if main_f(s, d)]; pool = fl + [s for s in pool if s not in set(fl)]
            out[d] = [s for s in pool[:9] if ctx["keep"](s, d) and not t2t(s, d)]
        return out
    arm = f"V3.2_{H['main']}"; arm2 = f"V3.2_{H['main']}_POOL"
    res, ph, changed = {"V3.2": [], arm: [], **({arm2: []} if H.get("pool") else {})}, [], 0
    ph2 = []
    for o in range(5):
        a, b = lists(o, False), lists(o, True)
        if o == 0:
            changed = int(sum(a[d] != b[d] for d in weeks[o]))
        arms_o = [("V3.2", a), (arm, b)] + ([(arm2, lists_pool(o))] if H.get("pool") else [])
        for k, pk in arms_o:
            mt = sp.metrics(run_cost(D, X, pk, weeks[o], "E0", cost)[0]); res[k].append({c: float(mt[c]) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")})
        ph.append(bool(sp.beats(res[arm][o], res["V3.2"][o])))
        if H.get("pool"):
            ph2.append(bool(sp.beats(res[arm2][o], res["V3.2"][o])))
    b_pass = bool(ph[0] and sum(ph) >= 4); b2_pass = bool(ph2 and ph2[0] and sum(ph2) >= 4)

    print(f"\n=== {EXP_ID} · weekly lists {START.date()}..{cal[-1].date()} · {info} ===")
    for k, e in evid.items():
        print(f"{k:28s} flagged {e['n']:6d} ({e['share']}%) avg {e['avg']:+.1f}% med {e['med']:+.1f}% +50% {e['hit50']}% lost {e['lost']}%  ||  others avg {e['o_avg']:+.1f}% med {e['o_med']:+.1f}% +50% {e['o_hit50']}% lost {e['o_lost']}%"
              + (f"  ||  weekly gap {e['weekly_gap'][0]:+} (95% {e['weekly_gap'][1]}..{e['weekly_gap'][2]}, {e['weekly_gap'][3]} weeks)" if "weekly_gap" in e else ""))
    print(f"A (evidence, {H['main']} in the pool's top 25): {a_text}")
    for k in res:
        r = res[k][0]; print(f"{k:12s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[k]):.1f}..{max(x['cagr'] for x in res[k]):.1f}")
    print(f"B: {arm} beats V3.2 in {sum(ph)}/5 · lists changed {changed}/{len(weeks[0])} · {'PASS' if b_pass else 'no pass'}")
    if H.get("pool"):
        print(f"B2: {arm2} beats V3.2 in {sum(ph2)}/5 · {'PASS' if b2_pass else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(group=k, **{kk: vv for kk, vv in e.items() if kk != "weekly_gap"}) for k, e in evid.items()]).to_csv(OUTDIR / "evidence.csv", index=False)
    (OUTDIR / "evidence.csv.manifest.json").write_text(json.dumps(dict(dataset="evidence.csv", experiment=EXP_ID, producer="src/agentic/test_famous_overlay.py " + key, definitions=__doc__,
        units=dict(avg="6-month return after costs, percent", hit50="percent of stocks at +50% or more after 6 months", lost="percent below 0"), v32=res, phases=f"{sum(ph)}/5", info=info, updated=now), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nEvidence ({H['main']}, pool top 25): {a_text}. V3.2 overlay: {'PASS' if b_pass else 'no pass'}. evidence.csv + manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + ("-RESULT" if not os.environ.get("SGM_RERUN_TAG") else "-RERUN-" + os.environ["SGM_RERUN_TAG"]), evidence_pass=a_pass, evidence_text=a_text, verdict=([arm] if b_pass else []) + ([arm2] if b2_pass else []) or "no pass", evidence=evid, info=info, phases_pool=(f"{sum(ph2)}/5" if ph2 else None),
                                 arms={k: [round(res[k][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for k in res}, phases=f"{sum(ph)}/5", lists_changed=changed), default=str) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
