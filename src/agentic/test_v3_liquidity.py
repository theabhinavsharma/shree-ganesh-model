"""EXP-2026-10-11-v3-liquidity-1cr (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: V3 only buys stocks trading >= Rs 5 cr a day on average (20 days) with a price above Rs 50. Does a Rs 1 cr bar
(price floor kept) pick better?
  V3     live rule (bar Rs 5 cr)
  V3_L1  same rule with the bar at Rs 1 cr; industry heat / G1 eligibility, trend test, model rank, top 9 and the financial /
         fading filters unchanged
Window: weekly entries from 2019-01-01; eras 2019-22 / 2023+; phases 0-4; run_exit("E0"), 126-session hold, 0.5% round trip.
PASS = sp.beats(V3_L1, V3) at phase 0 and in >= 4 of 5. Also reported: calendar years 2023-2026 averaged over 5 schedules.
Before any outcome: trust/data_ready.gate(). SGM_REPRO=1: log nothing. SGM_RERUN_TAG=<tag>: log -RERUN-<tag>.
"""
from __future__ import annotations

import json
import os
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

EXP_ID = "EXP-2026-10-11-v3-liquidity-1cr"
OUTDIR = ROOT / "logs/leader_sleeve/v3_liquidity_1cr"
START, YEARS = pd.Timestamp("2019-01-01"), (2023, 2024, 2025, 2026)
ARMS = ("V3", "V3_L1")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    L = L[["symbol", "trade_date", "raw", "adv"]]

    def F_with(bar):
        F = X["F"].merge(L, on=["symbol", "trade_date"], how="left")
        F["core"] = (F["adv"] >= bar) & (F["raw"] > 50)
        return F.drop(columns=["raw", "adv"])
    F5, F1 = F_with(5.0), F_with(1.0)

    # ---- data-readiness gate
    wk0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    new = F1[F1["core"] & ~F5["core"] & F1["trade_date"].isin(set(wk0))][["symbol", "trade_date"]].sort_values("trade_date")
    mm = pd.merge_asof(new.astype({"trade_date": "datetime64[ns]"}), P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    new["scored"] = mm["ensemble"].notna().astype(float).where(lambda x: x > 0).to_numpy()
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.coverage("model score known for the newly eligible Rs 1-5 cr stocks on list dates", new, "trade_date", ["scored"], range(START.year, cal[-1].year + 1)),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring check: the rebuilt Rs 5 cr bar reproduces the live rule exactly
    live = v3_rule.picks(X["F"], wk0, imap, P, ctx)
    reb = v3_rule.picks(F5, wk0, imap, P, ctx)
    same = sum(reb[d] == live.get(d, []) for d in wk0)
    print(f"wiring check: rebuilt Rs 5 cr bar equals v3_rule.picks on {same}/{len(wk0)} weeks", flush=True)
    if same < len(wk0):
        raise SystemExit("wiring check FAILED — no result reported")

    def arms(wk):
        return {"V3": v3_rule.picks(X["F"], wk, imap, P, ctx), "V3_L1": v3_rule.picks(F1, wk, imap, P, ctx)}

    sel0 = arms(wk0)
    small = set(zip(new["symbol"], new["trade_date"]))
    sm = [(s, d) for d in wk0 for s in sel0["V3_L1"][d] if (s, d) in small]
    print(f"V3_L1: picks from the Rs 1-5 cr band {len(sm)} of {sum(len(v) for v in sel0['V3_L1'].values())} ({len(set(s for s, _ in sm))} stocks)", flush=True)

    # ---- outcomes
    res, ph, yrs = {}, [], {a: [] for a in ARMS}
    for o in range(5):
        wk = wk0 if o == 0 else [d for d in sp.weekly_grid(cal, o) if d >= START]
        sel = sel0 if o == 0 else arms(wk)
        R = {}
        for a in ARMS:
            nav, info = sre.run_exit(D, X, sel[a], wk, "E0")
            m = sp.metrics(nav); R[a] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, avg_names=info["avg_names"])
            n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
            yrs[a].append({y: float(r.get(y, np.nan)) for y in YEARS})
        ph.append(bool(sp.beats(R["V3_L1"], R["V3"])))
        if o == 0:
            res = R
        print(f"phase {o}: V3 {R['V3']['cagr_disc']:.1f}/{R['V3']['cagr_conf']:.1f}/{R['V3']['maxdd']:.1f} · V3_L1 {R['V3_L1']['cagr_disc']:.1f}/{R['V3_L1']['cagr_conf']:.1f}/{R['V3_L1']['maxdd']:.1f} · beats {ph[-1]}", flush=True)
    passed = ph[0] and sum(ph) >= 4
    ya = {a: {y: round(float(np.mean([x[y] for x in yrs[a]])), 1) for y in YEARS} for a in ARMS}

    # ---- the small names vs the rest (phase 0)
    PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
    On, Hn, Cn = O.to_numpy(), Hh.to_numpy(), C.ffill(limit=300).to_numpy(); col = {s: i for i, s in enumerate(O.columns)}

    def outcome(s, d):
        i0 = cal.get_loc(d) + 1; c = col.get(s)
        if c is None or i0 + 126 > len(cal) or not np.isfinite(On[i0, c]) or On[i0, c] <= 0 or not np.isfinite(Cn[i0 + 125, c]):
            return None
        e = On[i0, c]; return Cn[i0 + 125, c] / e - 1, bool(np.nanmax(Hn[i0:i0 + 126, c]) >= 1.5 * e)
    f = lambda v: dict(n=len(v), avg=round(float(np.mean([r for r, _ in v])), 3) if v else None, median=round(float(np.median([r for r, _ in v])), 3) if v else None,  # noqa: E731
                       hit50=round(float(np.mean([h for _, h in v])), 3) if v else None, lost30=round(float(np.mean([r <= -0.3 for r, _ in v])), 3) if v else None)
    a_small, a_rest = [], []
    for d in wk0:
        for s in sel0["V3_L1"][d]:
            x = outcome(s, d)
            if x:
                (a_small if (s, d) in small else a_rest).append(x)
    diag = dict(small_band_picks=f(a_small), other_picks=f(a_rest))
    print(f"V3_L1 picks from Rs 1-5 cr: {diag['small_band_picks']} · other picks: {diag['other_picks']}")
    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · phase 0 ===")
    for a in ARMS:
        r = res[a]
        print(f"{a:6s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}"
              + (f" · beats V3 in {sum(ph)}/5 phases (phase 0 {'yes' if ph[0] else 'no'})" if a == "V3_L1" else ""))
    print("calendar years (avg of 5 schedules), V3 / V3_L1: " + " · ".join(f"{y}: {ya['V3'][y]:+.1f} / {ya['V3_L1'][y]:+.1f}" for y in YEARS)
          + f" · V3_L1 beats V3 in {sum(ya['V3_L1'][y] > ya['V3'][y] for y in YEARS)}/4 years")
    print(f"VERDICT: {'V3_L1 PASS' if passed else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **res[a], **{str(y): ya[a][y] for y in YEARS}, phases_beaten=(f"{sum(ph)}/5" if a == "V3_L1" else None), PASS=(passed if a == "V3_L1" else None)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_liquidity.py", definitions=__doc__,
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules"), small_vs_other=diag, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nV3 with a Rs 1 cr liquidity bar instead of Rs 5 cr. Verdict: {'V3_L1 PASS' if passed else 'no pass'}. Numbers and calendar years in results.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=["V3_L1"] if passed else "no arm passes",
                                 arms={a: [round(res[a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph)}/5"] if a == "V3_L1" else [None]) for a in ARMS},
                                 calendar_years=ya, small_vs_other=diag, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
