"""EXP-2026-10-10-v3-new-pledge-veto (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: when a promoter newly pledges shares, skip that V3 pick. Does V3 do better without those picks?
Arms (each drops names from V3's list, no refill):
  V3      live rule (v3_rule.picks)
  V3_NP   minus picks whose promoter created any new pledge (NSE SAST Reg 31 'Creation', encumbrance Pledge; today's ticker
          via nse_symbols) broadcast in the 90 calendar days up to the list date's 15:30 close
  V3_NP1  same, only when those new pledges sum to >= 1% of share capital
Window: weekly entries from 2019-01-01; eras 2019-22 and 2023+; phases 0-4. Engine: sim_screen_rank_exit.run_exit("E0"),
126-session hold. PASS per arm vs V3 = sp.beats at phase 0 and in >= 4 of 5. Before any outcome: trust/data_ready.gate().
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

EXP_ID = "EXP-2026-10-10-v3-new-pledge-veto"
OUTDIR = ROOT / "logs/leader_sleeve/v3_new_pledge_veto"
START, LOOKBACK, CLOSE = pd.Timestamp("2019-01-01"), pd.Timedelta(days=90), pd.Timedelta(hours=15, minutes=30)
ARMS = ("V3", "V3_NP", "V3_NP1")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    E = pd.read_parquet(ROOT / "data/derived/pledge_events.parquet")
    E = E[(E["event"].str.lower() == "creation") & E["encumbrance"].str.contains("pledge", case=False, na=False)].copy()
    E["broadcast_dt"] = pd.to_datetime(E["broadcast_dt"])
    EC = {s: (g["broadcast_dt"].values, g["pct_of_capital"].fillna(0).clip(0, 100).values) for s, g in E.sort_values("broadcast_dt").groupby("symbol_now")}

    def new_pct(s, d):
        a = EC.get(nse_symbols.now(s))
        if a is None:
            return None
        hi, lo = np.datetime64(d + CLOSE), np.datetime64(d + CLOSE - LOOKBACK)
        m = (a[0] <= hi) & (a[0] > lo)
        return float(a[1][m].sum()) if m.any() else None

    def arms(wk):
        v3 = v3_rule.picks(X["F"], wk, imap, P, ctx)
        out = {"V3": v3, "V3_NP": {}, "V3_NP1": {}}
        for d in wk:
            flags = {s: new_pct(s, d) for s in v3.get(d, [])}
            out["V3_NP"][d] = [s for s in v3.get(d, []) if flags[s] is None]
            out["V3_NP1"][d] = [s for s in v3.get(d, []) if flags[s] is None or flags[s] < 1.0]
        return out

    # ---- data-readiness gate (before any outcome)
    raw = sorted((ROOT / "data/raw/pledge_events").glob("*.json"))
    have = pd.Series([int(p.stem[:4]) for p in raw]).value_counts()
    want = pd.Series([m.year for m in pd.date_range("2016-01-01", pd.Timestamp.now(), freq="MS")]).value_counts()
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.completeness("pledge-event months on file vs months since 2016 (NSE Reg 31)", have, want, range(2016, cal[-1].year + 1)),
              dr.span("pledge creations", E["broadcast_dt"], START - LOOKBACK, cal[-1] - pd.Timedelta(days=10)),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), yrs)]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring check
    wk0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    sel0 = arms(wk0)
    live = v3_rule.picks(X["F"], wk0, imap, P, ctx)
    same = sum(sel0["V3"][d] == live.get(d, []) for d in wk0)
    print(f"wiring check: V3 arm equals v3_rule.picks on {same}/{len(wk0)} weeks", flush=True)
    if same < len(wk0):
        raise SystemExit("wiring check FAILED — no result reported")
    for a in ARMS[1:]:
        print(f"{a}: drops {sum(len(sel0['V3'][d]) - len(sel0[a][d]) for d in wk0)} of {sum(len(v) for v in sel0['V3'].values())} V3 picks (phase 0)", flush=True)

    # ---- outcomes
    res, ph = {}, {a: [] for a in ARMS[1:]}
    for o in range(5):
        wk = wk0 if o == 0 else [d for d in sp.weekly_grid(cal, o) if d >= START]
        sel = sel0 if o == 0 else arms(wk)
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

    # ---- dropped vs kept (phase 0)
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
        drop, keep, names = [], [], {}
        for d in wk0:
            for s in sel0["V3"][d]:
                x = outcome(s, d)
                if x:
                    if s in sel0[a][d]:
                        keep.append(x)
                    else:
                        drop.append(x); names[s] = names.get(s, 0) + 1
        diag[a] = dict(dropped=f(drop), kept=f(keep), dropped_companies=len(names), top=dict(sorted(names.items(), key=lambda kv: -kv[1])[:6]))
        print(f"{a}: dropped {diag[a]['dropped']} from {len(names)} companies · kept {diag[a]['kept']}")
    print(f"\n=== {EXP_ID} · weekly entries from {START.date()} to {cal[-1].date()} · phase 0 ===")
    for a in ARMS:
        r = res[a]
        print(f"{a:7s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}"
              + (f" · beats V3 in {sum(ph[a])}/5 phases (phase 0 {'yes' if ph[a][0] else 'no'})" if a != "V3" else ""))
    print(f"VERDICT: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **res[a], phases_beaten=(f"{sum(ph[a])}/5" if a in ph else None), PASS=(a in verdict) if a in ph else None) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_new_pledge_veto.py", definitions=__doc__,
        units=dict(cagr="percent a year", maxdd="percent"), dropped_vs_kept=diag, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nSkip V3 picks whose promoter newly pledged shares in the prior 90 days? Verdict: "
                                      f"{', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}. Numbers in results.csv (phase 0); dropped-vs-kept stats in its manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no arm passes",
                                 arms={a: [round(res[a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [f"{sum(ph[a])}/5" if a in ph else None] for a in ARMS},
                                 dropped_vs_kept=diag, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
