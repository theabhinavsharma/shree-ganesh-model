"""EXP-2026-10-12-v31-fpi-sizing (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: put more money into a week's V3.1 list when foreign investors (FPIs) have been net buyers, less when they
have been net sellers, and see whether that beats equal weekly amounts. Several tilt sizes and two ways of using the cash.
Signal: FPI equity stock-exchange net flow over the last 60 NSDL reporting dates / their gross turnover over those dates
(data/derived/nsdl_fpi_daily.parquet, rows used from known_from on). Its percentile on the list date is measured against all
earlier values since 2015 (expanding window). Tilts (low / mid / high): T_050_150 0.5/1/1.5, T_075_125 0.75/1/1.25,
T_000_200 0/1/2, T_050_100 0.5/1/1, T_100_150 1/1/1.5 (terciles), Q_050_150 0.5/1/1.5 on bottom / top fifths.
Engine: one pooled cash account. On each list date invest multiplier x equity/26 (never more than the cash), equal split over
the names after the liquidity-scaled round-trip cost, hold 126 sessions on the panel's daily returns, then back to cash.
Cash: IDLE (waits) or CATCHUP (each list date also invests 1/13 of the cash above one normal week's amount).
PASS per arm vs FLAT (multiplier 1, same cash rule): CAGR higher in both eras and max drawdown not worse by > 2 points, at
schedule 0 and in >= 4 of 5. FAMILY verdict: sizing is real only if >= 6 of the 12 arms pass.
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

EXP_ID = "EXP-2026-10-12-v31-fpi-sizing"
OUTDIR = ROOT / "logs/leader_sleeve/v31_fpi_sizing"
START, YEARS, HOLD = pd.Timestamp("2019-01-01"), (2023, 2024, 2025, 2026), 126
TILTS = {"FLAT": (1.0, 1.0, 1.0, 1 / 3), "T_050_150": (0.5, 1.0, 1.5, 1 / 3), "T_075_125": (0.75, 1.0, 1.25, 1 / 3), "T_000_200": (0.0, 1.0, 2.0, 1 / 3),
         "T_050_100": (0.5, 1.0, 1.0, 1 / 3), "T_100_150": (1.0, 1.0, 1.5, 1 / 3), "Q_050_150": (0.5, 1.0, 1.5, 0.2)}   # low, mid, high, tail share
CASH = ("IDLE", "CATCHUP")

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

    # ---- FPI signal, point in time
    Fp = pd.read_parquet(ROOT / "data/derived/nsdl_fpi_daily.parquet")
    se = Fp[(Fp["asset"] == "Equity") & (Fp["route"] == "Stock Exchange")].copy()
    se["reporting_date"] = pd.to_datetime(se["reporting_date"]); se["known_from"] = pd.to_datetime(se["known_from"])
    se = se.sort_values("reporting_date").reset_index(drop=True)
    gross = se["gross_purchases_cr"] + se["gross_sales_cr"]
    se["sig"] = se["net_cr"].rolling(60, min_periods=48).sum() / gross.rolling(60, min_periods=48).sum()
    se = se.dropna(subset=["sig"])
    kf, sv = se["known_from"].to_numpy(), se["sig"].to_numpy()

    def pct_on(d) -> float:
        k = np.searchsorted(kf, np.datetime64(d), side="right")          # rows known on or before d
        if k < 250:
            return np.nan
        return float((sv[:k - 1] < sv[k - 1]).mean())                    # today's reading vs all earlier ones
    all_weeks = sorted(set(d for o in range(5) for d in weeks[o]))
    PCT = {d: pct_on(d) for d in all_weeks}

    def mult(d, tilt) -> float:
        lo, mid, hi, tail = TILTS[tilt]; p = PCT.get(d, np.nan)
        if not np.isfinite(p):
            return 1.0
        return lo if p < tail else (hi if p >= 1 - tail else mid)

    # ---- data gate (no outcomes yet)
    S = pd.DataFrame({"d": all_weeks, "known": [1.0 if np.isfinite(PCT[d]) else np.nan for d in all_weeks]})
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("FPI 60-day flow percentile known on every list date (NSDL, point in time)", S, "d", ["known"], yrs, min_share=0.99),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    Rv, dpos, cidx = D["Rv"], D["dpos"], D["cidx"]; T = len(cal)

    def metrics(nav: pd.Series) -> dict:
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}

    def pooled(o: int, tilt: str, cash_rule: str) -> tuple:
        wk = weeks[o]; on = {dpos[d]: d for d in wk}; start = dpos[wk[0]]
        cash, book, nav, deployed, idle = 1.0, [], np.full(T, np.nan), [], []
        for i in range(start, T):
            keep = []
            for p in book:
                if p["i0"] < i <= p["end"]:
                    p["val"] = p["val"] * (1 + np.nan_to_num(Rv[i, p["cols"]]))
                    for q, cq in enumerate(p["cols"]):                  # 2026-10-12: big payout / demerger -> cash at the ex-session
                        for k, f in D.get("vr", {}).get(int(cq), ()):
                            if k == i and i < p["end"]:
                                cash += float(p["val"][q]) * (1 - f); p["val"][q] *= f
                if i >= p["end"]:
                    cash += float(p["val"].sum())
                else:
                    keep.append(p)
            book = keep
            if i in on:
                d = on[i]; names = [n for n in lists[o].get(d, []) if n in cidx]
                eq = cash + sum(float(p["val"].sum()) for p in book)
                base = eq / 26; amt = mult(d, tilt) * base
                if cash_rule == "CATCHUP":
                    amt += max(0.0, cash - amt - base) / 13
                amt = min(amt, cash)
                if names and amt > 0:
                    cols = np.array([cidx[n] for n in names])
                    book.append(dict(cols=cols, val=amt * (1 - cost(i, cols)) / len(cols), i0=i, end=min(i + HOLD, T - 1)))
                    cash -= amt
                deployed.append(amt / eq if eq > 0 else 0.0)
            nav[i] = cash + sum(float(p["val"].sum()) for p in book)
            if i in on:
                idle.append(cash / nav[i])
        s = pd.Series(nav[start:], index=cal[start:])
        return metrics(s), dict(avg_idle_cash=float(np.mean(idle)), max_idle_cash=float(np.max(idle)))

    # ---- wiring
    lad = metrics(run_cost(D, X, lists[0], weeks[0], "E0", cost)[0])
    got = tuple(round(lad[k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd"))
    flat0, _ = pooled(0, "FLAT", "IDLE")
    print(f"wiring check 1: V3.1 ladder schedule 0 = {got} (registered 51.8 / 48.9 / 54.9 / -40.0) · wiring check 2: pooled FLAT CAGR {flat0['cagr']:.1f} "
          f"(2019-22 {flat0['cagr_disc']:.1f}, 2023+ {flat0['cagr_conf']:.1f}, maxDD {flat0['maxdd']:.1f})", flush=True)
    if got != tuple(round(x, 1) for x in __import__("reproduce").logged("EXP-2026-10-11-v3-l1-newlist-cost")["arms"]["V3_L1Nc"][:4]) or abs(flat0["cagr"] - lad["cagr"]) > 2.0:
        raise SystemExit("wiring check FAILED — no result reported")

    lowshare = {lab: round(float(np.mean([mult(d, t) < 1 for d in weeks[0]])), 2) for lab, t in (("tercile low weeks", "T_050_150"), ("fifth low weeks", "Q_050_150"))}
    print(f"share of schedule-0 list weeks marked low: {lowshare}", flush=True)
    res, cashinfo = {}, {}
    for o in range(5):
        for c in CASH:
            for t in TILTS:
                r, ci = (flat0, None) if (o == 0 and c == "IDLE" and t == "FLAT") else pooled(o, t, c)
                if ci is None:
                    _, ci = pooled(o, t, c)
                res.setdefault((t, c), []).append(r); cashinfo.setdefault((t, c), []).append(ci)
        print(f"schedule {o} done", flush=True)
    rows, passed = [], []
    for (t, c), v in res.items():
        ref = res[("FLAT", c)]
        ph = [bool(sp.beats(v[o], ref[o])) for o in range(5)] if t != "FLAT" else None
        ok = bool(ph and ph[0] and sum(ph) >= 4)
        if ok:
            passed.append(f"{t}/{c}")
        r0 = v[0]
        rows.append(dict(arm=t, cash=c, cagr=round(r0["cagr"], 1), era_2019_22=round(r0["cagr_disc"], 1), era_2023p=round(r0["cagr_conf"], 1), maxdd=round(r0["maxdd"], 1),
                         cagr_range=f"{min(x['cagr'] for x in v):.1f}..{max(x['cagr'] for x in v):.1f}", **{str(y): round(float(np.mean([x[y] for x in v])), 1) for y in YEARS},
                         avg_idle_cash_pct=round(100 * float(np.mean([x["avg_idle_cash"] for x in cashinfo[(t, c)]])), 1),
                         beats_flat=(f"{sum(ph)}/5" + (" (sched 0 yes)" if ph[0] else " (sched 0 no)")) if ph else None, PASS=ok if ph else None))
    family = len(passed) >= 6
    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for r in rows:
        print(json.dumps(r))
    print(f"arms passing: {len(passed)}/12 {passed}")
    print(f"FAMILY VERDICT: {'FPI sizing PASS' if family else 'no pass (fewer than 6 of 12 arms pass)'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v31_fpi_sizing.py", definitions=__doc__,
        arms_passing=passed, family_pass=family, low_week_share=lowshare, units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules",
        avg_idle_cash_pct="average cash left uninvested on list dates, percent of equity"), updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nWeekly V3.1 amounts tilted by FPI 60-day flows; 6 tilts x 2 cash rules vs equal amounts. Arms passing {len(passed)}/12; "
                                      f"family verdict {'PASS' if family else 'no pass'}. Every arm in results.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"),
                                 verdict=("FPI sizing family PASS: " + ", ".join(passed)) if family else f"no pass ({len(passed)}/12 arms: {passed})",
                                 arms={f"{r['arm']}/{r['cash']}": [r["cagr"], r["era_2019_22"], r["era_2023p"], r["maxdd"], r["beats_flat"]] for r in rows},
                                 calendar_years={f"{r['arm']}/{r['cash']}": {y: r[str(y)] for y in YEARS} for r in rows}, cols="CAGR, 2019-22, 2023+, maxDD, beats FLAT")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
