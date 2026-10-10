"""EXP-2026-10-11-v3-new-listings (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: V3 cannot see a new listing for about a year (its trend test needs a 1-year return and a 200-day average).
Let liquid new listings in after 3 months of trading — do the lists get better?
New listing: first panel session after 2015-01-10, not the new name of a renamed company (nse_symbols chains), aged 63-251
sessions on the list date (from 252 sessions on, V3's normal test applies).
  V3         live rule (v3_rule.picks)
  V3_NL      a new listing passes the trend test measured since listing: close >= 1.5x its lowest low since listing (lo252),
             close >= 1.30x its first close (replaces 1-year return >= 30%), close > its 50-day average (replaces the 200-day
             tests); everything else as V3 (hot-industry eligibility, liquidity, model rank, top 9, financial / fading filters)
  V3_NLOPEN  new listings skip the trend test (liquid + hot industry + model rank only)
Window: weekly entries from 2019-01-01; eras 2019-22 / 2023+; phases 0-4; run_exit("E0"), 126-session hold.
PASS per arm = sp.beats vs V3 at phase 0 and in >= 4 of 5. Before any outcome: trust/data_ready.gate().
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

EXP_ID = "EXP-2026-10-11-v3-new-listings"
OUTDIR = ROOT / "logs/leader_sleeve/v3_new_listings"
START, AGE_MIN, AGE_MAX = pd.Timestamp("2019-01-01"), 63, 251
ARMS = ("V3", "V3_NL", "V3_NLOPEN")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    F = X["F"].sort_values(["symbol", "trade_date"]).copy()
    first = D["px"].groupby("symbol")["trade_date"].min()
    ren = set(nse_symbols.now_map().values())
    F["age"] = F.groupby("symbol").cumcount()
    F["first_close"] = F.groupby("symbol")["close"].transform("first")
    F["newlist"] = F["symbol"].map(first).gt(pd.Timestamp("2015-01-10")) & ~F["symbol"].isin(ren)
    F["young"] = F["newlist"] & F["age"].between(AGE_MIN, AGE_MAX)

    def select(weekly, mode, top=9):
        W = F[F["trade_date"].isin(set(weekly)) & F["core"]].copy()
        W["ind"] = W["symbol"].map(imap)
        W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W.merge(ctx["G1"][["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
        with np.errstate(invalid="ignore", divide="ignore"):
            trend = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"]))
            if mode == "NL":
                trend = trend | (W["young"] & (W["close"] / W["lo252"] - 1 >= 0.5) & (W["close"] / W["first_close"] - 1 >= 0.30) & (W["close"] > W["sma_50"]))
            elif mode == "NLOPEN":
                trend = trend | W["young"]
        Pool = W[trend].copy()
        L = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date")
        L["trade_date"] = L["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(L, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g1 = Pool.groupby("trade_date").head(top).groupby("trade_date")["symbol"].apply(list).to_dict()
        young = set(zip(Pool.loc[Pool["young"], "symbol"], Pool.loc[Pool["young"], "trade_date"]))
        return {d: [s for s in g1.get(d, []) if ctx["keep"](s, d)] for d in weekly}, young

    # ---- data-readiness gate
    wk0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    Y = F[F["young"] & F["core"] & F["trade_date"].isin(set(wk0))][["symbol", "trade_date"]].copy()
    Y["ind"] = Y["symbol"].map(imap).notna().astype(float).where(lambda x: x > 0)
    mm = pd.merge_asof(Y.sort_values("trade_date").astype({"trade_date": "datetime64[ns]"}), P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    Y = Y.sort_values("trade_date"); Y["scored"] = mm["ensemble"].notna().astype(float).where(lambda x: x > 0).to_numpy()
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("industry known for liquid new listings aged 3-12 months on list dates", Y, "trade_date", ["ind"], yrs),
              dr.coverage("model score known for liquid new listings aged 3-12 months on list dates", Y, "trade_date", ["scored"], yrs),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring check
    live = v3_rule.picks(X["F"], wk0, imap, P, ctx)
    base, _ = select(wk0, None)
    same = sum(base[d] == live.get(d, []) for d in wk0)
    print(f"wiring check: selection with new listings off equals v3_rule.picks on {same}/{len(wk0)} weeks", flush=True)
    if same < len(wk0):
        raise SystemExit("wiring check FAILED — no result reported")

    def arms(wk):
        out, yset = {"V3": v3_rule.picks(X["F"], wk, imap, P, ctx)}, {}
        for a, mode in (("V3_NL", "NL"), ("V3_NLOPEN", "NLOPEN")):
            out[a], yset[a] = select(wk, mode)
        return out, yset

    sel0, young0 = arms(wk0)
    for a in ARMS[1:]:
        nl = [(s, d) for d in wk0 for s in sel0[a][d] if (s, d) in young0[a]]
        top = pd.Series([s for s, _ in nl]).value_counts().head(12)
        print(f"{a}: new-listing picks {len(nl)} ({len(set(s for s, _ in nl))} stocks) · most frequent: {', '.join(f'{k} {v}' for k, v in top.items())}", flush=True)
        for s in ("KAYNES", "NETWEB"):
            ds = [d for d in wk0 if s in sel0[a][d]]
            print(f"   {s}: first on the list {ds[0].date() if ds else 'never'} ({len(ds)} weeks)")

    # ---- outcomes
    res, ph = {}, {a: [] for a in ARMS[1:]}
    for o in range(5):
        wk = wk0 if o == 0 else [d for d in sp.weekly_grid(cal, o) if d >= START]
        sel = sel0 if o == 0 else arms(wk)[0]
        R = {}
        for a in ARMS:
            nav, info = sre.run_exit(D, X, sel[a], wk, "E0")
            m = sp.metrics(nav); R[a] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, avg_names=info["avg_names"])
        for a in ARMS[1:]:
            ph[a].append(bool(sp.beats(R[a], R["V3"])))
        if o == 0:
            res = R
        print(f"phase {o}: " + " · ".join(f"{a} {R[a]['cagr_disc']:.1f}/{R[a]['cagr_conf']:.1f}/{R[a]['maxdd']:.1f}" for a in ARMS), flush=True)
    verdict = [a for a in ARMS[1:] if ph[a][0] and sum(ph[a]) >= 4]

    # ---- new-listing picks vs the rest (phase 0, 126 sessions from next open)
    PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
    On, Hn, Cn = O.to_numpy(), Hh.to_numpy(), C.ffill(limit=300).to_numpy(); col = {s: i for i, s in enumerate(O.columns)}

    def outcome(s, d):
        i0 = cal.get_loc(d) + 1; c = col.get(s)
        if c is None or i0 + 126 > len(cal) or not np.isfinite(On[i0, c]) or On[i0, c] <= 0 or not np.isfinite(Cn[i0 + 125, c]):
            return None
        e = On[i0, c]; return Cn[i0 + 125, c] / e - 1, bool(np.nanmax(Hn[i0:i0 + 126, c]) >= 1.5 * e)
    f = lambda v: dict(n=len(v), avg=round(float(np.mean([r for r, _ in v])), 3) if v else None, median=round(float(np.median([r for r, _ in v])), 3) if v else None,  # noqa: E731
                       hit50=round(float(np.mean([h for _, h in v])), 3) if v else None, lost30=round(float(np.mean([r <= -0.3 for r, _ in v])), 3) if v else None)
    diag = {}
    for a in ARMS[1:]:
        nl, rest = [], []
        for d in wk0:
            for s in sel0[a][d]:
                x = outcome(s, d)
                if x:
                    (nl if (s, d) in young0[a] else rest).append(x)
        diag[a] = dict(new_listing_picks=f(nl), other_picks=f(rest))
        print(f"{a}: new-listing picks {diag[a]['new_listing_picks']} · other picks {diag[a]['other_picks']}")
    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · phase 0 ===")
    for a in ARMS:
        r = res[a]
        print(f"{a:9s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}"
              + (f" · beats V3 in {sum(ph[a])}/5 phases (phase 0 {'yes' if ph[a][0] else 'no'})" if a != "V3" else ""))
    print(f"VERDICT: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **res[a], phases_beaten=(f"{sum(ph[a])}/5" if a in ph else None), PASS=(a in verdict) if a in ph else None) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_new_listings.py", definitions=__doc__,
        units=dict(cagr="percent a year", maxdd="percent"), new_listing_vs_other=diag, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nLet liquid new listings into V3 after 3 months instead of 12? Verdict: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}. "
                                      "Numbers in results.csv; new-listing picks vs the rest in its manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no arm passes",
                                 arms={a: [round(res[a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [f"{sum(ph[a])}/5" if a in ph else None] for a in ARMS},
                                 new_listing_vs_other=diag, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
