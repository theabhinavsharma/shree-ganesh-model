"""EXP-2026-10-11-v31-inbasket-rank (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: within each weekly V3.1 list (up to 9 names, model-ordered), is there one simple parameter (or two) that
orders the names better than the model score, so that buying only the top X after re-ranking earns more?
Candidates: the 1.5x model's own point-in-time inputs (logs/leader_sleeve/anatomy_1p5x/rows.parquet, phase-0 weekly dates;
other schedules take the latest earlier row within 7 days). A candidate qualifies only if known for >= 75% of V3.1 picks
in every year 2019-2026.
Choice, on 2019-22 lists only: mean within-list Spearman correlation with the 6-month return (next-open entry, 126
sessions). R1 = the best single parameter, R2 = average rank of the best two. Unknown values rank last; ties by model order.
Arms, X = 3, 4, 5, 6: M_X model-order top X, R1_X, R2_X; plus V3.1 as is. Engine: test_v3_l1_newlist_cost.run_cost
(liquidity-scaled costs), weekly entries from 2019, schedules 0-4.
PASS (primary X = 5): vs M_5, 2023+ CAGR higher and max drawdown not worse by > 2 points, at schedule 0 and in >= 4 of 5.
Wiring: M_9 at schedule 0 reproduces V3_L1Nc of EXP-2026-10-11-v3-l1-newlist-cost (51.8 / 48.9 / 54.9 / -40.0).
The V3.1 selection below is a frozen copy (same steps as explore_v31_hits_by_year / test_v3_l1_newlist_cost.nl_select).
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

EXP_ID = "EXP-2026-10-11-v31-inbasket-rank"
OUTDIR = ROOT / "logs/leader_sleeve/v31_inbasket_rank"
FEATS = ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet"
START, SEL_END, HOLD = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), 126
YEARS, XS, PRIMARY = (2023, 2024, 2025, 2026), (3, 4, 5, 6), 5
CANDS = {"mcap_cr": "company size (market cap)", "adv": "daily traded value", "ret60": "3-month return", "ret252": "1-year return",
         "off_high": "distance from 52-week high", "off_low": "distance from 52-week low", "ind_heat_pct": "industry heat",
         "rank_in_ind_pct": "rank within its industry", "pe": "PE", "sales_yoy": "sales growth (yoy)", "eps_yoy": "EPS growth (yoy)",
         "promoter_pct": "promoter holding", "avg_delivery_pct_20d": "delivery % (20-day avg)", "rsi_14_weekly": "weekly RSI",
         "dvol60": "volatility (60-day)", "traded_value_vs_60d": "traded value vs 60-day avg"}


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
    R = pd.read_parquet(FEATS, columns=["symbol", "trade_date", *CANDS]); R["trade_date"] = pd.to_datetime(R["trade_date"]).astype("datetime64[ns]")
    R = R.sort_values("trade_date")

    def pick_frame(o: int) -> pd.DataFrame:
        Pk = pd.DataFrame([dict(trade_date=d, symbol=s, pos=i + 1) for d in weeks[o] for i, s in enumerate(lists[o][d])])
        Pk["trade_date"] = Pk["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Pk.reset_index().sort_values("trade_date"), R, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        return m.set_index("index").sort_index()
    PF = {o: pick_frame(o) for o in range(5)}

    # ---- qualify candidates (coverage only, no outcomes), then the data gate
    yrs = range(START.year, cal[-1].year + 1)
    cov0 = dr.coverage("candidate parameters known for phase-0 V3.1 picks", PF[0], "trade_date", list(CANDS), yrs)
    bar = dr.BAR["coverage_min"]
    qual = [c for c in CANDS if all(v >= bar for v in cov0["by_feature"][c].values())] if "by_feature" in cov0 else None
    if qual is None:
        s0 = {c: PF[0][c].notna().groupby(PF[0]["trade_date"].dt.year).mean() for c in CANDS}
        qual = [c for c in CANDS if all(float(s0[c].get(y, 0)) >= bar for y in yrs)]
    excluded = {c: {int(y): round(float(v), 2) for y, v in PF[0][c].notna().groupby(PF[0]["trade_date"].dt.year).mean().items()} for c in CANDS if c not in qual}
    print(f"qualifying parameters ({len(qual)}): {qual}\nexcluded (known for < {bar:.0%} of picks in some year): {excluded}", flush=True)
    allp = pd.concat([PF[o] for o in range(5)])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.coverage("qualifying parameters known for V3.1 picks, all 5 schedules", allp, "trade_date", qual, yrs),
              dr.span("model-input feature table dates", R["trade_date"], START, cal[-1]),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring: V3 through the frozen selection steps at Rs 5 cr equals v3_rule.picks is covered by the M_9 reproduction below
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731

    def run(sel: dict, o: int) -> dict:
        nav, info = run_cost(D, X, sel, weeks[o], "E0", cost)
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, "names": info["avg_names"], **{y: float(r.get(y, np.nan)) for y in YEARS}}
    m9 = run(lists[0], 0)
    want = tuple(round(x, 1) for x in __import__("reproduce").logged("EXP-2026-10-11-v3-l1-newlist-cost")["arms"]["V3_L1Nc"][:4]); got = tuple(round(m9[k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd"))
    print(f"wiring check: M_9 schedule 0 = {got}, registered V3_L1Nc = {want}", flush=True)
    if got != want:
        raise SystemExit("wiring check FAILED — no result reported")

    # ---- selection on 2019-22 (phase 0): within-list Spearman with the 6-month return
    PXo = rp.load_panel(["open", "close"]); O, C = (rp.wide(PXo, c, cal) for c in ("open", "close")); del PXo
    On, Cn = O.to_numpy(), C.ffill(limit=300).to_numpy(); col = {s: i for i, s in enumerate(O.columns)}; pos = {d: i for i, d in enumerate(cal)}; T = len(cal)
    P0 = PF[0].copy()
    e = P0["trade_date"].map(pos).to_numpy() + 1; c = P0["symbol"].map(col).fillna(-1).astype(int).to_numpy()
    ok = (c >= 0) & (e + HOLD - 1 <= T - 1); e2, c2 = np.where(ok, e, 0), np.where(ok, c, 0)
    entry = np.where(ok, On[e2, c2], np.nan)
    P0["ret"] = np.where(ok & (entry > 0), Cn[np.minimum(e2 + HOLD - 1, T - 1), c2] / entry - 1, np.nan)

    def ic(df: pd.DataFrame, p: str) -> float:
        v = []
        for _, g in df.groupby("trade_date"):
            g = g[[p, "ret"]].dropna()
            if len(g) >= 4 and g[p].nunique() > 1:
                v.append(g[p].rank().corr(g["ret"].rank()))
        return float(np.nanmean(v)) if v else np.nan
    sel_rows = P0[P0["trade_date"] <= SEL_END]
    ics = {p: ic(sel_rows, p) for p in qual + ["pos"]}
    order = sorted(qual, key=lambda p: -abs(ics[p]) if np.isfinite(ics[p]) else 0)
    r1, r2 = order[0], order[:2]
    sign = {p: (1 if ics[p] > 0 else -1) for p in qual}
    print("2019-22 within-list correlation with 6m return (+ = higher value did better; 'pos' = model order, + would mean lower-ranked did better):")
    print("  " + " · ".join(f"{p} {ics[p]:+.3f}" for p in order + ["pos"]))
    print(f"chosen: R1 = {r1} ({'higher' if sign[r1] > 0 else 'lower'} first) · R2 = {r2[0]} + {r2[1]}", flush=True)
    ic_post = {p: ic(P0[P0["trade_date"] > SEL_END], p) for p in qual + ["pos"]}

    def reorder(df: pd.DataFrame, params: list) -> dict:
        out = {}
        for d, g in df.groupby("trade_date"):
            g = g.sort_values("pos")
            sc = np.zeros(len(g))
            for p in params:
                v = sign[p] * g[p].to_numpy(dtype=float)
                rk = pd.Series(v).rank(pct=True).fillna(0).to_numpy()  # unknown ranks last
                sc += rk
            idx = np.lexsort((g["pos"].to_numpy(), -sc))
            out[d] = g["symbol"].to_numpy()[idx].tolist()
        return out

    # ---- arms
    res = {}
    for o in range(5):
        L1, L2 = reorder(PF[o], [r1]), reorder(PF[o], r2)
        res.setdefault("V3.1 (all)", []).append(m9 if o == 0 else run(lists[o], o))
        for x in XS:
            res.setdefault(f"M_{x}", []).append(run({d: lists[o][d][:x] for d in weeks[o]}, o))
            res.setdefault(f"R1_{x}", []).append(run({d: L1.get(d, [])[:x] for d in weeks[o]}, o))
            res.setdefault(f"R2_{x}", []).append(run({d: L2.get(d, [])[:x] for d in weeks[o]}, o))
        print(f"schedule {o} done", flush=True)

    def beats(a: dict, b: dict) -> bool:
        return a["cagr_conf"] > b["cagr_conf"] and a["maxdd"] >= b["maxdd"] - 2.0
    rows, verdict = [], []
    for name, v in res.items():
        x = name.split("_")[-1]
        ref = res.get(f"M_{x}") if name[0] == "R" else None
        ph = [beats(v[o], ref[o]) for o in range(5)] if ref else None
        rows.append(dict(arm=name, cagr=round(v[0]["cagr"], 1), era_2019_22=round(v[0]["cagr_disc"], 1), era_2023p=round(v[0]["cagr_conf"], 1), maxdd=round(v[0]["maxdd"], 1),
                         names=round(v[0]["names"], 1), cagr_range=f"{min(r['cagr'] for r in v):.1f}..{max(r['cagr'] for r in v):.1f}",
                         **{str(y): round(float(np.mean([r[y] for r in v])), 1) for y in YEARS},
                         beats_model_order=(f"{sum(ph)}/5" + (" (sched 0 yes)" if ph[0] else " (sched 0 no)")) if ph else None))
        if ref and int(x) == PRIMARY and ph[0] and sum(ph) >= 4:
            verdict.append(name)

    # ---- hit / return by re-ranked position, 2023+ (phase 0)
    H = rp.load_panel(["high"]); Hh = rp.wide(H, "high", cal).reindex(columns=O.columns); del H
    Hf = Hh.iloc[::-1].rolling(HOLD, min_periods=1).max().iloc[::-1].to_numpy()
    P0["hit"] = np.where(ok & (entry > 0), Hf[e2, c2] / entry - 1 >= 0.5, np.nan)
    L1 = reorder(PF[0], [r1]); rp1 = {(d, s): i + 1 for d, v in L1.items() for i, s in enumerate(v)}
    P0["r1pos"] = [rp1.get((d, s)) for d, s in zip(P0["trade_date"], P0["symbol"])]
    post = P0[(P0["trade_date"] > SEL_END) & P0["ret"].notna()]
    grad = {lab: post.groupby(k).agg(n=("ret", "size"), hit=("hit", "mean"), avg6m=("ret", "mean")).round(3).reset_index().to_dict("records") for lab, k in (("model order", "pos"), (f"R1 ({r1})", "r1pos"))}

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for r in rows:
        print(json.dumps(r))
    print("2023+ within-list correlation (out of sample): " + " · ".join(f"{p} {ic_post[p]:+.3f}" for p in order + ["pos"]))
    for lab, g in grad.items():
        print(f"2023+ by position, {lab}: " + " · ".join(f"#{int(r[list(r)[0]])} hit {100 * r['hit']:.0f}% avg {100 * r['avg6m']:+.0f}% (n {r['n']})" for r in g))
    print(f"VERDICT (primary X = {PRIMARY}): {', '.join(verdict) + ' PASS' if verdict else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v31_inbasket_rank.py", definitions=__doc__,
        chosen=dict(R1=r1, R2=r2, direction={p: sign[p] for p in r2}), ic_2019_22=ics, ic_2023p=ic_post, excluded=excluded,
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules", ic="mean within-list Spearman correlation with 6-month return"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nRe-rank the weekly V3.1 list by one simple parameter (R1 = {r1}) or two (R2 = {' + '.join(r2)}), chosen on 2019-22, and buy the top X. "
                                      f"Verdict (X = {PRIMARY}): {', '.join(verdict) + ' PASS' if verdict else 'no pass'}. All arms in results.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no pass",
                                 chosen=dict(R1=r1, R2=r2), ic_2019_22={p: round(v, 3) for p, v in ics.items()}, ic_2023p={p: round(v, 3) for p, v in ic_post.items()},
                                 arms={r["arm"]: [r["cagr"], r["era_2019_22"], r["era_2023p"], r["maxdd"], r["beats_model_order"]] for r in rows},
                                 calendar_years={r["arm"]: {y: r[str(y)] for y in YEARS} for r in rows}, cols="CAGR, 2019-22, 2023+, maxDD, beats model order")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
