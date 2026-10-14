"""EXP-2026-10-15-v32-heat-mix (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: V3.2 gives all 9 weekly slots to hot industries. Give mid- and cold-heat industries a few slots for their best stocks:
7-2-0, 6-2-1, 5-3-1 (hot-mid-cold). Tiers by industry heat_pct that week: HOT = G1 (>= 0.70, policy pillar not < 0.30), MID 0.40-0.70,
COLD < 0.40 (policy veto everywhere). Within a tier: liquid stocks (Rs 1 cr, > Rs 50) passing the trend test (or new-listing
exemption), ranked by model score, top-N; unfilled mid / cold slots go to the next hot names; then V3.2's filters, no refill.
PASS per arm vs V3.2: both eras, max drawdown not worse by > 2 points, schedule 0 and >= 4 of 5. Also: slot-level 6-month returns.
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
EXP_ID = "EXP-2026-10-15-v32-heat-mix"; OUTDIR = ROOT / "logs/leader_sleeve/v32_heat_mix"
START, HOLD, YEARS = pd.Timestamp("2019-04-01"), 126, (2023, 2024, 2025, 2026)
MIX = {"V3.2": (9, 0, 0), "M720": (7, 2, 0), "M621": (6, 2, 1), "M531": (5, 3, 1)}


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); tpos = {d: i for i, d in enumerate(cal)}
    Sc = ctx["scores"]; ok = ~(Sc["P_pct"] < 0.30)
    TIERS = {"hot": ctx["G1"], "mid": Sc[ok & (Sc["heat_pct"] >= 0.40) & (Sc["heat_pct"] < 0.70)], "cold": Sc[ok & (Sc["heat_pct"] < 0.40)]}
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

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    ranked = {o: {t: v3_rule.ranked_v31(F1, weeks[o], imap, P, el, 30) for t, el in TIERS.items()} for o in range(5)}

    # ---- data gate (no outcomes yet)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    cnt = pd.DataFrame([dict(d=d, mid=(1.0 if ranked[0]["mid"][d] else np.nan), cold=(1.0 if ranked[0]["cold"][d] else np.nan)) for d in weeks[0]])
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.span("industry scores", Sc["date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.coverage("weeks with at least one mid / cold candidate", cnt, "d", ["mid", "cold"], range(START.year, cal[-1].year + 1), min_share=0.5)]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    def lists(o, mix):
        h, m, c = mix; out = {}
        for d in weeks[o]:
            H, M, C = ranked[o]["hot"][d], ranked[o]["mid"][d], ranked[o]["cold"][d]
            mm, cc = M[:m], C[:c]; hh = H[:h + (m - len(mm)) + (c - len(cc))]
            out[d] = [s for s in hh + mm + cc if keep(s, d)]
        return out
    ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks[0], imap, P, ctx).items()}
    base = lists(0, MIX["V3.2"]); same = sum(base[d] == ref[d] for d in weeks[0])
    print(f"wiring check: 9-0-0 via tiers == v3_rule V3.1 minus trade-for-trade on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}
    res, opn, navs, ph = {a: [] for a in MIX}, {a: [] for a in MIX}, {}, {a: [] for a in MIX if a != "V3.2"}
    for o in range(5):
        for a, mix in MIX.items():
            pk = lists(o, mix); n = run_cost(D, X, pk, weeks[o], "E0", cost)[0]; navs[(a, o)] = n; res[a].append(met(n))
            opn[a].append(met(run_open(D, X, pk, weeks[o], "E0", cost, 0.0, 0.0)[0]))
        for a in ph:
            ph[a].append(bool(sp.beats(res[a][o], res["V3.2"][o])))
        print(f"schedule {o}: " + " · ".join(f"{a} {res[a][o]['cagr_disc']:.1f}/{res[a][o]['cagr_conf']:.1f}/{res[a][o]['maxdd']:.1f}" for a in MIX), flush=True)
    passed = {a: bool(v[0] and sum(v) >= 4) for a, v in ph.items()}
    bs = {a: [boot(navs[(a, o)], navs[("V3.2", o)]) for o in range(5)] for a in ph}
    yrs = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in MIX}

    # ---- slot-level 6-month returns (schedule 0, after V3.2's filters)
    def r6(s, d):
        q = D["cidx"].get(s); i0 = tpos[d]
        if q is None or i0 + HOLD > T - 1:
            return np.nan
        g = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
        for k, f in vr.get(q, ()):
            if i0 + 1 <= k <= i0 + HOLD - 1:
                g = f * g + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
        return ((1 - float(cost(i0, np.array([q]))[0])) * g - 1) * 100
    slots = {"hot #1-3": ("hot", 0, 3), "hot #4-6": ("hot", 3, 6), "hot #7-9": ("hot", 6, 9), "hot #10-12": ("hot", 9, 12),
             "mid #1-2": ("mid", 0, 2), "mid #3": ("mid", 2, 3), "cold #1": ("cold", 0, 1)}
    srows = []
    for d in weeks[0]:
        for lab, (t, a, b) in slots.items():
            for s in ranked[0][t][d][a:b]:
                if keep(s, d):
                    r = r6(s, d)
                    if not np.isnan(r):
                        srows.append(dict(slot=lab, r=r))
    SR = pd.DataFrame(srows).groupby("slot", sort=False)["r"].agg(picks="size", avg="mean", median="median", hit50=lambda s: (s >= 50).mean() * 100, lost=lambda s: (s < 0).mean() * 100).round(1)

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in MIX:
        r = res[a][0]
        print(f"{a:5s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs[a][y]:+.0f}" for y in YEARS) + f" · next open {opn[a][0]['cagr']:.1f}")
    for a in ph:
        print(f"{a} vs V3.2: beats in {sum(ph[a])}/5 · gap {bs[a][0][0]:+.1f} pts/yr (95% {bs[a][0][1]:+.1f}..{bs[a][0][2]:+.1f}) · range excludes 0 in {sum(lo > 0 or hi < 0 for _, lo, hi in bs[a])}/5 · {'PASS' if passed[a] else 'no pass'}")
    print("\nslot-level 6-month returns after costs (schedule 0, names kept by V3.2's filters):"); print(SR.to_string())
    if os.environ.get("SGM_REPRO") == "1":
        return
    OUTDIR.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame([dict(arm=a, mix="-".join(map(str, MIX[a])), **{c: round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs[a][y] for y in YEARS},
                       next_open=round(opn[a][0]["cagr"], 1)) for a in MIX]).to_csv(OUTDIR / "results.csv", index=False)
    SR.to_csv(OUTDIR / "slots.csv")
    for f, what in (("results.csv", "one row per mix, schedule 0; years = average of 5 schedules"), ("slots.csv", "6-month return after costs by tier-rank slot (schedule 0, names kept by V3.2's filters)")):
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_v32_heat_mix.py", definitions=__doc__, contents=what,
            units=dict(cagr="percent a year", maxdd="percent", avg="percent", hit50="percent of picks at +50% or more", lost="percent of picks below 0"), updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nHeat mixes 7-2-0 / 6-2-1 / 5-3-1 vs V3.2 (9-0-0): " + ", ".join(f"{a} {'PASS' if v else 'no pass'}" for a, v in passed.items()) + ". results.csv, slots.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=[a for a, v in passed.items() if v] or "no pass",
                                 arms={a: [round(res[a][0][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in MIX}, phases={a: f"{sum(v)}/5" for a, v in ph.items()},
                                 bootstrap=bs, calendar_years=yrs, next_open={a: round(opn[a][0]["cagr"], 1) for a in MIX}, slots=SR.reset_index().to_dict("records"),
                                 cols="CAGR, 2019-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
