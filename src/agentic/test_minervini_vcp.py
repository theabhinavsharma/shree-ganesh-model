"""EXP-2026-10-15-minervini-vcp (registered in logs/experiments.jsonl before any outcome was computed). Daily famous-hypothesis #1.

Plain English: Mark Minervini's Volatility Contraction Pattern - a stock near its high whose pullbacks keep getting shallower while
volume dries up. Does it pick multibaggers, and does it improve V3.2?
VCP (fixed): last 60 sessions in 3 blocks of 20, range = (max high - min low) / max high; range1 > range2 > range3, range3 <= 10%,
close >= 85% of the 252-session high, mean volume last 10 <= 0.8 x mean last 50 (adjusted panel prices / volumes).
A (evidence): 6-month return after costs, VCP vs not, in all liquid stocks / the V3.1 pool / the pool's top 25; criterion = top-25
weekly mean difference with a block-bootstrap (26-week blocks) 95% range above 0.
B (adoption): VCP names first within the pool's top 25, then V3.2 (filters, trade-for-trade drop, no refill); sp.beats vs V3.2 at
schedule 0 and >= 4 of 5. SGM_REPRO=1: log nothing.
"""
import json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
EXP_ID = "EXP-2026-10-15-minervini-vcp"; OUTDIR = ROOT / "logs/leader_sleeve/minervini_vcp"; START, HOLD = pd.Timestamp("2019-04-01"), 126


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    L = rp.load_panel(["high", "low", "close", "total_traded_qty", "avg_traded_value_20d"])
    W = {c: rp.wide(L, c, cal).reindex(columns=D["syms"]) for c in ("high", "low", "close", "total_traded_qty")}
    ADV = (rp.wide(L, "avg_traded_value_20d", cal).reindex(columns=D["syms"]) / 1e7).to_numpy(); del L
    hi20, lo20 = W["high"].rolling(20, min_periods=15).max(), W["low"].rolling(20, min_periods=15).min()
    rng = (hi20 - lo20) / hi20
    r1, r2, r3 = rng.shift(40), rng.shift(20), rng
    near = W["close"] / W["high"].rolling(252, min_periods=200).max()
    vol = W["total_traded_qty"].rolling(10, min_periods=8).mean() / W["total_traded_qty"].rolling(50, min_periods=40).mean()
    VCP = ((r1 > r2) & (r2 > r3) & (r3 <= 0.10) & (near >= 0.85) & (vol <= 0.8)).to_numpy()
    KNOWN = (r1.notna() & r3.notna() & near.notna() & vol.notna()).to_numpy()
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)]); vr = D.get("vr", {})
    cidx = D["cidx"]
    isv = lambda s, d: (s in cidx) and bool(VCP[tpos[d], cidx[s]])  # noqa: E731

    def r6(s, d):
        q = cidx.get(s); i0 = tpos[d]
        if q is None or i0 + HOLD > T - 1:
            return np.nan
        g = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
        for k, f in vr.get(q, ()):
            if i0 + 1 <= k <= i0 + HOLD - 1:
                g = f * g + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
        return ((1 - float(cost(i0, np.array([q]))[0])) * g - 1) * 100

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    pool0 = v3_rule.ranked_v31(F1, weeks[0], imap, P, ctx["G1"], 10000)

    # ---- data gate (no outcomes yet)
    liq = F1[F1["core"].fillna(False).astype(bool) & F1["trade_date"].isin(set(weeks[0]))][["symbol", "trade_date"]]
    K = pd.DataFrame([dict(d=d, known=(1.0 if (s in cidx and KNOWN[tpos[d], cidx[s]]) else np.nan)) for s, d in zip(liq["symbol"], liq["trade_date"])])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.coverage("VCP inputs (60-session ranges, 252-session high, volumes) known for liquid stocks on list dates", K, "d", ["known"], range(START.year, cal[-1].year + 1)),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10))]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    # ---- A: evidence
    rows = []
    for d in weeks[0]:
        top25 = set(pool0[d][:25]); pool = set(pool0[d])
        for s in liq.loc[liq["trade_date"] == d, "symbol"]:
            r = r6(s, d)
            if np.isnan(r):
                continue
            rows.append(dict(d=d, s=s, vcp=isv(s, d), r=r, pool=s in pool, top25=s in top25))
    A = pd.DataFrame(rows)

    def wk_boot(sub, block=26, n=4000, seed=7):
        g = sub.groupby(["d", "vcp"])["r"].mean().unstack(); g = g.dropna()
        x = (g[True] - g[False]).to_numpy(); Tn = len(x); rngb = np.random.default_rng(seed); nb = int(np.ceil(Tn / block))
        out = np.array([x[((rngb.integers(0, Tn, nb)[:, None] + np.arange(block)[None, :]).ravel()[:Tn]) % Tn].mean() for _ in range(n)])
        return float(x.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), int(Tn)
    evid = {}
    for name, m in (("all liquid stocks", A["s"].notna()), ("V3.1 pool", A["pool"]), ("pool top 25", A["top25"])):
        sub = A[m]; v, nv = sub[sub.vcp], sub[~sub.vcp]
        evid[name] = dict(vcp_n=len(v), other_n=len(nv), vcp_share=round(len(v) / max(1, len(sub)) * 100, 1),
                          vcp_avg=round(v.r.mean(), 1), other_avg=round(nv.r.mean(), 1), vcp_med=round(v.r.median(), 1), other_med=round(nv.r.median(), 1),
                          vcp_50=round((v.r >= 50).mean() * 100, 1), other_50=round((nv.r >= 50).mean() * 100, 1),
                          vcp_lost=round((v.r < 0).mean() * 100, 1), other_lost=round((nv.r < 0).mean() * 100, 1))
        if name == "pool top 25":
            evid[name]["weekly_gap"] = [round(x, 2) if isinstance(x, float) else x for x in wk_boot(sub)]
    a_pass = evid["pool top 25"]["weekly_gap"][1] > 0

    # ---- B: V3.2 overlay
    def lists(o, vcp_first):
        R = pool0 if o == 0 else v3_rule.ranked_v31(F1, weeks[o], imap, P, ctx["G1"], 25)
        out = {}
        for d in weeks[o]:
            pool = R[d][:25] if o == 0 else R[d]
            if vcp_first:
                first = [s for s in pool if isv(s, d)]; pool = first + [s for s in pool if s not in set(first)]
            out[d] = [s for s in pool[:9] if ctx["keep"](s, d) and not t2t(s, d)]
        return out
    L0 = lists(0, False); ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[0], imap, P, ctx).items()}
    same = sum(L0[d] == ref[d] for d in weeks[0]); print(f"wiring check: V3.2 rebuilt == v3_rule on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")
    res, ph, changed = {"V3.2": [], "V3.2_VCP": []}, [], 0
    for o in range(5):
        a, b = lists(o, False), lists(o, True)
        if o == 0:
            changed = int(sum(a[d] != b[d] for d in weeks[o]))
        for k, pk in (("V3.2", a), ("V3.2_VCP", b)):
            m = sp.metrics(run_cost(D, X, pk, weeks[o], "E0", cost)[0]); res[k].append({c: float(m[c]) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")})
        ph.append(bool(sp.beats(res["V3.2_VCP"][o], res["V3.2"][o])))
    b_pass = bool(ph[0] and sum(ph) >= 4)

    print(f"\n=== {EXP_ID} · weekly lists {START.date()}..{cal[-1].date()} ===")
    for name, e in evid.items():
        print(f"{name:18s} VCP {e['vcp_n']:6d} ({e['vcp_share']}%) avg {e['vcp_avg']:+.1f}% med {e['vcp_med']:+.1f}% · +50%: {e['vcp_50']}% · lost {e['vcp_lost']}%  ||  "
              f"others avg {e['other_avg']:+.1f}% med {e['other_med']:+.1f}% · +50%: {e['other_50']}% · lost {e['other_lost']}%" + (f"  ||  weekly gap {e['weekly_gap'][0]:+.1f} pts (95% {e['weekly_gap'][1]:+.1f}..{e['weekly_gap'][2]:+.1f}, {e['weekly_gap'][3]} weeks)" if "weekly_gap" in e else ""))
    print(f"A (evidence in the pool's top 25): {'YES' if a_pass else 'no'}")
    for k in res:
        r = res[k][0]; print(f"{k:9s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f}")
    print(f"B: V3.2_VCP beats V3.2 in {sum(ph)}/5 · lists changed {changed}/{len(weeks[0])} · {'PASS' if b_pass else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(group=k, **v) for k, v in evid.items()]).to_csv(OUTDIR / "evidence.csv", index=False)
    (OUTDIR / "evidence.csv.manifest.json").write_text(json.dumps(dict(dataset="evidence.csv", experiment=EXP_ID, producer="src/agentic/test_minervini_vcp.py", definitions=__doc__,
        units=dict(avg="6-month return after costs, percent", _50="percent of stocks reaching +50% in 6 months", lost="percent of stocks below 0"), v32=res, phases=f"{sum(ph)}/5", updated=now), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nMinervini VCP. Evidence (pool top 25): {'YES' if a_pass else 'no'}. V3.2 overlay: {'PASS' if b_pass else 'no pass'}. evidence.csv + manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", evidence_pass=bool(a_pass), verdict=["V3.2_VCP"] if b_pass else "no pass", evidence=evid,
                                 arms={k: [round(res[k][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for k in res}, phases=f"{sum(ph)}/5", lists_changed=changed)) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
