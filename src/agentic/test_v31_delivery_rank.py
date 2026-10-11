"""EXP-2026-10-11-v31-delivery-rank (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: re-rank each weekly V3.1 list by delivery (how much of the traded volume buyers took home), as a level or as
"recent delivery vs its own past", and buy only the top X.
Data: delivery_pct in the price panel (NSE MTO / sec_bhavdata_full, official). Only EQ-series sessions count (BE/BZ
trade-for-trade sessions have compulsory 100% delivery = unknown). Lagged one session (published after the close). A window
is known if >= 60% of its sessions are known EQ sessions; a 1-session value only if that session was EQ.
Parameters (all higher first, fixed before any outcome): dl20 = 20-session average; dSvL = average over the last S sessions
divided by the average over the last L sessions, S in 1 / 5 / 20, L in 20 / 60 / 120 / 250 (S < L).
Slot-keeping re-rank (unknown values keep their model-order slot), top X = 3, 4, 5, 6 vs model-order top X (M_X).
Verdict only for parameters known for >= 75% of phase-0 picks in every year 2019-2026.
PASS per parameter (X = 5): vs M_5, CAGR higher in both eras and max drawdown not worse by > 2 points, at schedule 0 and in
>= 4 of 5 schedules. Frozen V3.1 selection and engine as test_v31_inbasket_rank_known.py.
SGM_REPRO=1: log nothing. SGM_RERUN_TAG=<tag>: log -RERUN-<tag>.
"""
from __future__ import annotations

import json
import math
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

EXP_ID = "EXP-2026-10-11-v31-delivery-rank"
OUTDIR = ROOT / "logs/leader_sleeve/v31_delivery_rank"
START, SPLIT, HOLD = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), 126
YEARS, XS, PRIMARY, KNOWN = (2023, 2024, 2025, 2026), (3, 4, 5, 6), 5, 0.60
PARAMS = ["dl20"] + [f"d{s}v{l}" for s in (1, 5, 20) for l in (20, 60, 120, 250) if s < l]

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

    # ---- delivery features from the panel (EQ sessions only, lagged one session)
    Dl = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series", "delivery_pct"])
    Dl["trade_date"] = pd.to_datetime(Dl["trade_date"])
    eq = Dl["series"] == "EQ"
    yr = Dl["trade_date"].dt.year
    deq_share = Dl[eq]["delivery_pct"].notna().groupby(yr[eq]).mean()
    Dl["deq"] = Dl["delivery_pct"].where(eq)
    Wd = rp.wide(Dl, "deq", cal).shift(1); del Dl
    avg = {w: Wd.rolling(w, min_periods=(1 if w == 1 else math.ceil(KNOWN * w))).mean().to_numpy(dtype="float32") for w in (1, 5, 20, 60, 120, 250)}
    FA = {"dl20": avg[20]}
    for p in PARAMS[1:]:
        s, l = (int(x) for x in p[1:].split("v"))
        with np.errstate(invalid="ignore", divide="ignore"):
            FA[p] = avg[s] / np.where(avg[l] > 0, avg[l], np.nan)
    fcol = {s: i for i, s in enumerate(Wd.columns)}; fpos = {d: i for i, d in enumerate(cal)}; del avg, Wd

    def pick_frame(o: int) -> pd.DataFrame:
        Pk = pd.DataFrame([dict(trade_date=d, symbol=s, pos=i + 1) for d in weeks[o] for i, s in enumerate(lists[o][d])])
        r = Pk["trade_date"].map(fpos).to_numpy(); c = Pk["symbol"].map(fcol).fillna(-1).astype(int).to_numpy()
        for p in PARAMS:
            Pk[p] = np.where(c >= 0, FA[p][r, np.maximum(c, 0)], np.nan)
        return Pk
    PF = {o: pick_frame(o) for o in range(5)}

    # ---- data gate (no outcomes yet)
    yrs = range(START.year, cal[-1].year + 1)
    cov = {p: {int(y): round(float(v), 2) for y, v in PF[0][p].notna().groupby(PF[0]["trade_date"].dt.year).mean().items()} for p in PARAMS}
    bar = dr.BAR["coverage_min"]
    verdict_ok = [p for p in PARAMS if all(cov[p].get(y, 0) >= bar for y in yrs)]
    print("known share of phase-0 picks by year: " + json.dumps(cov), flush=True)
    print(f"verdict parameters (>= {bar:.0%} known every year 2019-2026): {verdict_ok}; pick-level only: {[p for p in PARAMS if p not in verdict_ok]}", flush=True)
    allp = pd.concat([PF[o] for o in range(5)])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.completeness("delivery known on EQ stock-days (share of EQ rows)", deq_share, pd.Series(1.0, index=deq_share.index), range(2018, cal[-1].year + 1)),
              dr.coverage("verdict parameters known for V3.1 picks, all 5 schedules", allp, "trade_date", verdict_ok, yrs),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731

    def run(sel: dict, o: int) -> dict:
        nav, info = run_cost(D, X, sel, weeks[o], "E0", cost)
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, "names": info["avg_names"], **{y: float(r.get(y, np.nan)) for y in YEARS}}
    m9 = run(lists[0], 0)
    want = (51.8, 48.9, 54.9, -40.0); got = tuple(round(m9[k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd"))
    print(f"wiring check: M_9 schedule 0 = {got}, registered V3_L1Nc = {want}", flush=True)
    if got != want:
        raise SystemExit("wiring check FAILED — no result reported")

    def reorder(df: pd.DataFrame, p: str) -> dict:
        """Slot-keeping re-rank: unknown values keep their model-order slot; known values (higher first) fill the known slots."""
        out = {}
        for d, g in df.groupby("trade_date"):
            g = g.sort_values("pos"); syms = g["symbol"].tolist(); v = g[p].to_numpy(dtype=float)
            kn = np.where(np.isfinite(v))[0]
            order = kn[np.lexsort((g["pos"].to_numpy()[kn], -v[kn]))]
            new = list(syms)
            for slot, src_i in zip(kn, order):
                new[slot] = syms[src_i]
            out[d] = new
        return out

    res = {}
    for o in range(5):
        res.setdefault("V3.1 (all)", []).append(m9 if o == 0 else run(lists[o], o))
        RO = {p: reorder(PF[o], p) for p in PARAMS}
        for x in XS:
            res.setdefault(f"M_{x}", []).append(run({d: lists[o][d][:x] for d in weeks[o]}, o))
            for p in PARAMS:
                res.setdefault(f"{p}_{x}", []).append(run({d: RO[p].get(d, [])[:x] for d in weeks[o]}, o))
        print(f"schedule {o} done", flush=True)

    def beats(a: dict, b: dict) -> bool:
        return a["cagr_disc"] > b["cagr_disc"] and a["cagr_conf"] > b["cagr_conf"] and a["maxdd"] >= b["maxdd"] - 2.0
    rows, verdict = [], []
    for name, v in res.items():
        p, x = name.rsplit("_", 1) if name != "V3.1 (all)" else (name, None)
        ref = res.get(f"M_{x}") if p in PARAMS else None
        ph = [beats(v[o], ref[o]) for o in range(5)] if ref else None
        rows.append(dict(arm=name, cagr=round(v[0]["cagr"], 1), era_2019_22=round(v[0]["cagr_disc"], 1), era_2023p=round(v[0]["cagr_conf"], 1), maxdd=round(v[0]["maxdd"], 1),
                         cagr_range=f"{min(r['cagr'] for r in v):.1f}..{max(r['cagr'] for r in v):.1f}", **{str(y): round(float(np.mean([r[y] for r in v])), 1) for y in YEARS},
                         beats_model_order=(f"{sum(ph)}/5" + (" (sched 0 yes)" if ph[0] else " (sched 0 no)")) if ph else None,
                         verdict_eligible=(p in verdict_ok) if p in PARAMS else None))
        if ref and int(x) == PRIMARY and p in verdict_ok and ph[0] and sum(ph) >= 4:
            verdict.append(name)

    # ---- pick level (phase 0, full windows, known values only)
    PXo = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PXo, c, cal) for c in ("open", "high", "close")); del PXo
    On, Cn = O.to_numpy(), C.ffill(limit=300).to_numpy(); Hf = Hh.iloc[::-1].rolling(HOLD, min_periods=1).max().iloc[::-1].to_numpy()
    col = {s: i for i, s in enumerate(O.columns)}; pos = {d: i for i, d in enumerate(cal)}; T = len(cal)
    P0 = PF[0].copy()
    e = P0["trade_date"].map(pos).to_numpy() + 1; c = P0["symbol"].map(col).fillna(-1).astype(int).to_numpy()
    ok = (c >= 0) & (e + HOLD - 1 <= T - 1); e2, c2 = np.where(ok, e, 0), np.where(ok, c, 0)
    entry = np.where(ok, On[e2, c2], np.nan); good = ok & (entry > 0)
    P0["ret"] = np.where(good, Cn[np.minimum(e2 + HOLD - 1, T - 1), c2] / entry - 1, np.nan)
    P0["hit"] = np.where(good, (Hf[e2, c2] / entry - 1 >= 0.5).astype(float), np.nan)
    P0["era"] = np.where(P0["trade_date"] <= SPLIT, "2019-22", "2023+")

    def ic(df: pd.DataFrame, k: str) -> tuple:
        v = []
        for _, g in df.groupby("trade_date"):
            g = g[[k, "ret"]].dropna()
            if len(g) >= 4 and g[k].nunique() > 1:
                v.append(g[k].rank().corr(g["ret"].rank()))
        return (round(float(np.nanmean(v)), 3) if v else None, len(v))
    pick, half = {}, {}
    print("\npick level: within-list correlation with 6m return (+ = higher delivery did better), lists with >= 4 known values")
    for p in PARAMS:
        pick[p] = {era: ic(P0[P0["era"] == era], p) for era in ("2019-22", "2023+")}
        pick[p].update({str(y): ic(P0[P0["trade_date"].dt.year == y], p) for y in (2023, 2024, 2025)})
        q = P0[P0[p].notna() & P0["ret"].notna()].copy(); q["top"] = q.groupby("trade_date")[p].rank(pct=True) > 0.5
        half[p] = q.groupby(["era", "top"]).agg(n=("ret", "size"), avg6m=("ret", "mean"), hit=("hit", "mean")).round(3).reset_index().to_dict("records")
        hs = {(r["era"], r["top"]): r for r in half[p]}
        print(f"  {p:8s} " + " · ".join(f"{k} {v[0]:+.3f}" if v[0] is not None else f"{k} n/a" for k, v in pick[p].items())
              + " | upper vs lower half 2023+: " + (f"hit {100 * hs[('2023+', True)]['hit']:.0f}% vs {100 * hs[('2023+', False)]['hit']:.0f}%, avg {100 * hs[('2023+', True)]['avg6m']:+.0f}% vs {100 * hs[('2023+', False)]['avg6m']:+.0f}%" if ("2023+", True) in hs and ("2023+", False) in hs else "n/a"))

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for r in rows:
        print(json.dumps(r))
    print(f"VERDICT (X = {PRIMARY}, verdict parameters {verdict_ok}): {', '.join(verdict) + ' PASS' if verdict else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v31_delivery_rank.py", definitions=__doc__,
        coverage=cov, verdict_parameters=verdict_ok, pick_level_ic=pick, upper_vs_lower_half=half, delivery_known_on_eq_rows={int(k): round(float(v), 4) for k, v in deq_share.items()},
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules", ic="mean within-list Spearman correlation with 6-month return", delivery="fraction of traded quantity marked for delivery (0-1)"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nRe-rank the weekly V3.1 list by delivery level or recent-vs-past delivery (EQ sessions only), buy the top X. "
                                      f"Verdict (X = {PRIMARY}): {', '.join(verdict) + ' PASS' if verdict else 'no pass'}. All arms in results.csv; pick-level numbers in the manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no pass", verdict_parameters=verdict_ok,
                                 arms={r["arm"]: [r["cagr"], r["era_2019_22"], r["era_2023p"], r["maxdd"], r["beats_model_order"]] for r in rows},
                                 calendar_years={r["arm"]: {y: r[str(y)] for y in YEARS} for r in rows}, pick_level_ic=pick, cols="CAGR, 2019-22, 2023+, maxDD, beats model order")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
