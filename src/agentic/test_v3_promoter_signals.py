"""EXP-2026-10-04-v3-promoter-signals (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: inside V3's G1 pool, does moving recent promoter buyers and pledge releasers to the front (within the top 25
by model) pick better than V3?
Arms (each: full G1 pool by model rank -> reorder -> top 9 -> V3's filters (Financial Services, fading-theme-only), no refill):
  V3       no reorder (= v3_rule.picks; wiring check)
  V3_PB    names with prom_buys90 > 0 (promoter PIT 'Market Purchase' in the prior 90 days; anatomy rows as of the list date,
           <= 7 days old) AND pool rank <= 25 go first, in model order
  V3_PR    same with promoter pledges RELEASED >= 1% of share capital in the prior 90 days (NSE SAST Reg 31 'Release',
           encumbrance type Pledge, by broadcast time; today's ticker via nse_symbols)
  V3_PBPR  either signal
Window: weekly entries from 2019-04-01 (promoter-buying data starts April 2019); eras 2019-22 and 2023+; phases 0-4.
Engine: sim_screen_rank_exit.run_exit(..., "E0"), 126-session hold. PASS (per arm vs V3) = sp.beats at phase 0 and in >= 4 of 5.
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
import nse_symbols  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402
import v3_rule  # noqa: E402

EXP_ID = "EXP-2026-10-04-v3-promoter-signals"
OUTDIR = ROOT / "logs/leader_sleeve/v3_promoter_signals"
START, BOOST_RANK, REL_MIN = pd.Timestamp("2019-04-01"), 25, 1.0
ARMS = ("V3", "V3_PB", "V3_PR", "V3_PBPR")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    A = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "prom_buys90"])
    A["trade_date"] = pd.to_datetime(A["trade_date"])
    PB = {s: (g["trade_date"].values, g["prom_buys90"].values) for s, g in A.sort_values("trade_date").groupby("symbol")}

    def pb(s, d):
        a = PB.get(s)
        if a is None:
            return np.nan
        i = np.searchsorted(a[0], np.datetime64(d), side="right") - 1
        return a[1][i] if i >= 0 and (np.datetime64(d) - a[0][i]) <= np.timedelta64(7, "D") else np.nan
    E = pd.read_parquet(ROOT / "data/derived/pledge_events.parquet")
    E = E[(E["event"] == "Release") & E["encumbrance"].str.contains("pledge", case=False, na=False)]
    ER = {s: (g["broadcast_dt"].values, g["pct_of_capital"].clip(0, 100).values) for s, g in E.sort_values("broadcast_dt").groupby("symbol_now")}

    def pr(s, d):
        a = ER.get(nse_symbols.now(s))
        if a is None:
            return 0.0
        m = (a[0] < np.datetime64(d)) & (a[0] >= np.datetime64(d - pd.Timedelta(days=90)))
        return float(a[1][m].sum())
    SIG = {"V3_PB": lambda s, d: pb(s, d) > 0, "V3_PR": lambda s, d: pr(s, d) >= REL_MIN,
           "V3_PBPR": lambda s, d: pb(s, d) > 0 or pr(s, d) >= REL_MIN}

    def pools(o):
        wk = [d for d in sp.weekly_grid(cal, o) if d >= START]
        top = sre.TOPN; sre.TOPN = 100000
        full = tif.select_elig(X["F"], wk, imap, P, ctx["G1"]); sre.TOPN = top
        return wk, full

    def arms(wk, full):
        out = {}
        for a in ARMS:
            sel = {}
            for d in wk:
                pool = full.get(d, [])
                if a != "V3":
                    first = [s for s in pool[:BOOST_RANK] if SIG[a](s, d)]
                    pool = first + [s for s in pool if s not in set(first)]
                sel[d] = [s for s in pool[:9] if ctx["keep"](s, d)]
            out[a] = sel
        return out

    # ---- data-readiness gate (before any outcome)
    wk0, full0 = pools(0)
    rows = pd.DataFrame([dict(d=d, known=(1.0 if pd.notna(pb(s, d)) else np.nan)) for d in wk0 for s in full0.get(d, [])[:BOOST_RANK]])
    raw = sorted((ROOT / "data/raw/pledge_events").glob("*.json"))
    have = pd.Series([int(p.stem[:4]) for p in raw]).value_counts()
    want = pd.Series([m.year for m in pd.date_range("2016-01-01", pd.Timestamp.now(), freq="MS")]).value_counts()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("promoter-buying signal known for G1-pool names (top 25) on list dates", rows, "d", ["known"], yrs),
              dr.completeness("pledge-event months on file vs months since 2016 (NSE Reg 31)", have, want, range(2016, cal[-1].year + 1)),
              dr.span("pledge events", E["broadcast_dt"], START - pd.Timedelta(days=90), cal[-1] - pd.Timedelta(days=10)),
              dr.span("anatomy rows (promoter buying)", A["trade_date"], START, cal[-1] - pd.Timedelta(days=10))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring check: the V3 arm equals the live rule
    sel0 = arms(wk0, full0)
    live = v3_rule.picks(X["F"], wk0, imap, P, ctx)
    same = sum(sel0["V3"][d] == live.get(d, []) for d in wk0)
    print(f"wiring check: V3 arm equals v3_rule.picks on {same}/{len(wk0)} weeks", flush=True)
    if same < len(wk0):
        raise SystemExit("wiring check FAILED — no result reported")

    # ---- outcomes
    res, ph = {}, {a: [] for a in ARMS[1:]}
    for o in range(5):
        wk, full = (wk0, full0) if o == 0 else pools(o)
        sel = sel0 if o == 0 else arms(wk, full)
        R = {}
        for a in ARMS:
            nav, info = sre.run_exit(D, X, sel[a], wk, "E0")
            m = sp.metrics(nav); R[a] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, avg_names=info["avg_names"])
        for a in ARMS[1:]:
            ph[a].append(bool(sp.beats(R[a], R["V3"])))
        if o == 0:
            res = R
    verdict = [a for a in ARMS[1:] if ph[a][0] and sum(ph[a]) >= 4]

    # ---- moved in vs moved out (phase 0)
    PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX

    def outcome(s, d):
        i0 = cal.get_loc(d) + 1
        if i0 + 126 > len(cal) or s not in O.columns or not np.isfinite(O[s].iloc[i0]):
            return None
        e = O[s].iloc[i0]; return C[s].iloc[i0 + 125] / e - 1, bool((Hh[s].iloc[i0:i0 + 126] >= 1.5 * e).any())
    diag = {}
    print(f"\n=== {EXP_ID} · weekly entries from {START.date()} ===")
    for a in ARMS[1:]:
        inn, out = [], []
        for d in wk0:
            for s in set(sel0[a][d]) - set(sel0["V3"][d]):
                x = outcome(s, d); inn += [x] if x else []
            for s in set(sel0["V3"][d]) - set(sel0[a][d]):
                x = outcome(s, d); out += [x] if x else []
        f = lambda v: dict(n=len(v), avg=round(float(np.mean([r for r, _ in v])), 3) if v else None, hit50=round(float(np.mean([h for _, h in v])), 3) if v else None,  # noqa: E731
                           lost30=round(float(np.mean([r <= -0.3 for r, _ in v])), 3) if v else None)
        diag[a] = dict(moved_in=f(inn), moved_out=f(out))
        print(f"{a}: moved in {diag[a]['moved_in']} · moved out {diag[a]['moved_out']}")
    for a in ARMS:
        r = res[a]
        print(f"{a:7s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}"
              + (f" · beats V3 in {sum(ph[a])}/5 phases (phase 0 {'yes' if ph[a][0] else 'no'})" if a != "V3" else ""))
    print(f"VERDICT: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **res[a], phases_beaten=(f"{sum(ph[a])}/5" if a in ph else None), PASS=(a in verdict) if a in ph else None) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_promoter_signals.py",
        definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent"), moved=diag, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nDo recent promoter buyers / pledge releasers, moved ahead inside V3's pool, pick better than V3? "
                                      f"Verdict: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}. Numbers in results.csv; definitions and moved-in/out stats in its manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no arm passes",
                                 arms={a: [round(res[a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [f"{sum(ph[a])}/5" if a in ph else None] for a in ARMS},
                                 moved=diag, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
