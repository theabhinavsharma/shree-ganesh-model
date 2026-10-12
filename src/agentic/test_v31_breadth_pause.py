"""EXP-2026-10-11-v31-breadth-pause (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: skip the new weekly V3.1 list whenever the market has visibly turned: fewer than half of liquid stocks are
above their 200-day average AND the equal-weight index of liquid stocks is 10% or more below its running peak (both turned
in mid-Nov 2024). Universe each day = Rs 1 cr average traded value and raw price > Rs 50 on the previous day; both signs
are read at the list date's close. A skipped week's slot holds cash until its next turn (26 weeks, the engine's ladder);
held positions are not sold.
Arms: V3.1 (frozen copy of the live selection) vs V3.1_PAUSE. Engine: test_v3_l1_newlist_cost.run_cost, weekly entries
2019+, schedules 0-4. PASS: CAGR higher in both eras and max drawdown not worse by > 2 points, at schedule 0 and >= 4 of 5.
SGM_REPRO=1: log nothing. SGM_RERUN_TAG=<tag>: log -RERUN-<tag>.
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
import nse_symbols  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402

EXP_ID = "EXP-2026-10-11-v31-breadth-pause"
OUTDIR = ROOT / "logs/leader_sleeve/v31_breadth_pause"
START, YEARS = pd.Timestamp("2019-01-01"), (2023, 2024, 2025, 2026)
BREADTH_MAX, DD_MAX = 0.50, -0.10

def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy()
    F = X["F"].merge(L[["symbol", "trade_date", "raw", "adv"]], on=["symbol", "trade_date"], how="left").sort_values(["symbol", "trade_date"])
    del L
    firstd = D["px"].groupby("symbol")["trade_date"].min(); ren = set(nse_symbols.now_map().values())
    F["age"] = F.groupby("symbol").cumcount()
    F["young"] = F["symbol"].map(firstd).gt(pd.Timestamp("2015-01-10")) & ~F["symbol"].isin(ren) & F["age"].between(63, 251)
    hot = set(zip(pd.to_datetime(ctx["G1"]["date"]), ctx["G1"]["industry"]))

    def v31_lists(weeks: list) -> dict:
        """{date: V3.1 picks in model order} (frozen copy)."""
        W = F[F["trade_date"].isin(set(weeks)) & (F["adv"] >= 1.0) & (F["raw"] > 50)].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W[[(d, i) in hot for d, i in zip(W["trade_date"], W["ind"])]]
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])) | W["young"]
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g = Pool.groupby("trade_date").head(9).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: [s for s in g.get(d, []) if ctx["keep"](s, d)] for d in weeks}

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    lists = {o: v31_lists(weeks[o]) for o in range(5)}

    # ---- market signals (known at each close)
    Lm = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    Lm["raw"] = rp.raw_price(Lm, "close"); Lm["adv"] = Lm["avg_traded_value_20d"] / 1e7
    Cf = rp.wide(Lm, "close", cal).ffill(limit=5); RAW = rp.wide(Lm, "raw", cal); ADVm = rp.wide(Lm, "adv", cal); del Lm
    uni = ((ADVm >= 1.0) & (RAW > 50)).shift(1, fill_value=False).astype(bool)
    ew = (1 + Cf.pct_change(fill_method=None).clip(-0.5, 1.0).where(uni).mean(axis=1).fillna(0.0)).cumprod()
    sma200 = Cf.rolling(200, min_periods=200).mean(); known = uni & sma200.notna()
    nk = known.sum(axis=1).astype(float).replace(0.0, np.nan)
    sig = pd.DataFrame({"breadth": ((Cf > sma200) & known).sum(axis=1).astype(float) / nk, "ew_dd": ew / ew.cummax() - 1, "n_known": nk}, index=cal)
    del Cf, RAW, ADVm, sma200, known
    sig["pause"] = (sig["breadth"] < BREADTH_MAX) & (sig["ew_dd"] <= DD_MAX)

    # ---- data gate (no outcomes yet)
    wk_all = sorted(set(d for o in range(5) for d in weeks[o]))
    S = sig.reindex(wk_all).reset_index().rename(columns={"index": "date"})
    S["signal_known"] = np.where(S["breadth"].notna() & (S["n_known"] >= 300), 1.0, np.nan)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("breadth / index signal known (>= 300 stocks with a 200-day average) on every list date", S, "date", ["signal_known"], yrs, min_share=0.99),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2015, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731

    def run(sel: dict, o: int):
        nav, info = run_cost(D, X, sel, weeks[o], "E0", cost)
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}, nav

    res, ph, navs = {"V3.1": [], "V3.1_PAUSE": []}, [], {}
    for o in range(5):
        paused = {d for d in weeks[o] if bool(sig["pause"].get(d, False))}
        a, nav_a = run(lists[o], o)
        b, nav_b = run({d: ([] if d in paused else lists[o][d]) for d in weeks[o]}, o)
        if o == 0:
            want = tuple(round(x, 1) for x in __import__("reproduce").logged("EXP-2026-10-11-v3-l1-newlist-cost")["arms"]["V3_L1Nc"][:4]); got = tuple(round(a[k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd"))
            print(f"wiring check: V3.1 schedule 0 = {got}, registered V3_L1Nc = {want}", flush=True)
            if got != want:
                raise SystemExit("wiring check FAILED — no result reported")
            navs = {"V3.1": nav_a, "V3.1_PAUSE": nav_b}; paused0 = sorted(paused)
        res["V3.1"].append(a); res["V3.1_PAUSE"].append(b)
        ph.append(bool(sp.beats(b, a)))
        print(f"schedule {o}: paused {len(paused)}/{len(weeks[o])} weeks · V3.1 {a['cagr_disc']:.1f}/{a['cagr_conf']:.1f}/{a['maxdd']:.1f} · PAUSE {b['cagr_disc']:.1f}/{b['cagr_conf']:.1f}/{b['maxdd']:.1f} · beats {ph[-1]}", flush=True)
    passed = ph[0] and sum(ph) >= 4

    # paused stretches (schedule 0) and what those skipped lists went on to do (6-month, from the explore table if present)
    runs, cur = [], []
    for d in paused0:
        if cur and (d - cur[-1]).days > 10:
            runs.append(cur); cur = []
        cur.append(d)
    if cur:
        runs.append(cur)
    pk = ROOT / "logs/leader_sleeve/v31_hits_by_year/v31_picks.csv"
    skipped = None
    if pk.exists():
        P6 = pd.read_csv(pk, parse_dates=["trade_date"]); P6 = P6[P6["full"]]
        skipped = {f"{r[0].date()}..{r[-1].date()}": dict(weeks=len(r), picks=int(P6["trade_date"].isin(r).sum()),
                   avg6m=round(100 * float(P6[P6["trade_date"].isin(r)]["ret"].mean()), 1) if P6["trade_date"].isin(r).any() else None) for r in runs}
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in res}
    nv = {a: navs[a] / navs[a].iloc[0] for a in navs}
    dd_path = {a: round(float((nv[a] / nv[a].cummax() - 1).loc["2024-09-01":"2025-06-30"].min()) * 100, 1) for a in nv}

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in res:
        r = res[a][0]
        print(f"{a:11s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · CAGR range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS) + (f" · beats V3.1 in {sum(ph)}/5 (schedule 0 {'yes' if ph[0] else 'no'})" if a != "V3.1" else ""))
    print(f"worst fall Sep 2024 - Jun 2025: {dd_path}")
    print(f"paused stretches (schedule 0): {skipped if skipped is not None else [f'{r[0].date()}..{r[-1].date()} ({len(r)}w)' for r in runs]}")
    print(f"VERDICT: {'V3.1_PAUSE PASS' if passed else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       phases_beaten=(f"{sum(ph)}/5" if a != "V3.1" else None), PASS=(passed if a != "V3.1" else None)) for a in res]).to_csv(OUTDIR / "results.csv", index=False)
    sig.loc[sig.index >= START, ["breadth", "ew_dd", "pause"]].to_csv(OUTDIR / "signal_daily.csv")
    for f, extra in (("results.csv", dict(paused_stretches=skipped)), ("signal_daily.csv", {})):
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_v31_breadth_pause.py", definitions=__doc__,
            units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules", breadth="share 0-1", ew_dd="fraction below running peak"),
            updated=datetime.now().isoformat(timespec="seconds"), **extra), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nPause new V3.1 lists when breadth < 50% and the equal-weight index is 10%+ off its peak. Verdict: {'V3.1_PAUSE PASS' if passed else 'no pass'}. "
                                      "results.csv = both arms; signal_daily.csv = the daily signal.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=["V3.1_PAUSE"] if passed else "no pass",
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph)}/5"] if a != "V3.1" else [None]) for a in res},
                                 calendar_years=yrs_avg, paused_stretches=skipped, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
