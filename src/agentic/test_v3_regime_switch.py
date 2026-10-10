"""EXP-2026-10-10-v3-regime-switch (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: V3 falls 1.6-1.8x the market in small-cap sell-offs. Does holding cash while the broad market is in a downtrend
cut the worst falls without giving up the return?
Signal: risk-off on a session = equal-weight index of every priced stock (average daily return, each clipped to [-50%, +100%])
closes below its own 200-session average (known at that close).
Arms:
  V3        live rule, time exit (E0)
  V3_RSKIP  no new weekly buy when the list date is risk-off (that slot's money stays in cash until its next scheduled batch)
  V3_REXIT  V3_RSKIP plus: a held position is sold at the next open after the first risk-off session in its holding window
            (exit rule E3 fed the market risk-off flag instead of the stock's own 200-day flag); cash waits for the slot's next batch
Window: weekly entries from 2019-01-01; eras 2019-22 / 2023+; phases 0-4; 126-session hold, 26-slot ladder.
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
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402

EXP_ID = "EXP-2026-10-10-v3-regime-switch"
OUTDIR = ROOT / "logs/leader_sleeve/v3_regime_switch"
START, SMA = pd.Timestamp("2019-01-01"), 200
ARMS = ("V3", "V3_RSKIP", "V3_REXIT")
WINDOWS = {"Jan-Feb 2025": ("2024-12-31", "2025-02-28"), "Jan 2026": ("2025-12-31", "2026-01-31"),
           "Mar 2026": ("2026-02-27", "2026-03-31"), "Apr 2026": ("2026-03-31", "2026-04-30")}


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    C = rp.wide(rp.load_panel(["close"]), "close", cal)
    mret = (C / C.shift(1) - 1).clip(-0.5, 1.0).mean(axis=1).fillna(0.0)
    idx = (1 + mret).cumprod()
    off = (idx < idx.rolling(SMA, min_periods=SMA).mean()).to_numpy()
    OFF = pd.Series(off, index=cal)

    # ---- data-readiness gate
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(),
                              bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], START, cal[-1] - pd.Timedelta(days=10))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    y = OFF[OFF.index >= START]
    print("share of sessions risk-off by year: " + " · ".join(f"{k} {v*100:.0f}%" for k, v in y.groupby(y.index.year).mean().items()))
    spells, cur = [], None
    for d, v in y.items():
        if v and cur is None:
            cur = d
        elif not v and cur is not None:
            spells.append((cur, prev)); cur = None
        prev = d
    if cur is not None:
        spells.append((cur, prev))
    long = [(a, b) for a, b in spells if (b - a).days >= 20]
    print(f"risk-off spells since 2019: {len(spells)} ({len(long)} lasting 20+ days): " + ", ".join(f"{a:%b %Y}–{b:%b %Y}" for a, b in long))

    Xoff = dict(X, below=np.broadcast_to(off[:, None], X["below"].shape))

    def arms(wk):
        v3 = v3_rule.picks(X["F"], wk, imap, P, ctx)
        skip = {d: ([] if OFF.get(d, False) else v3.get(d, [])) for d in wk}
        return {"V3": (v3, X, "E0"), "V3_RSKIP": (skip, X, "E0"), "V3_REXIT": (skip, Xoff, "E3")}

    # ---- wiring check
    wk0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    A0 = arms(wk0)
    live = v3_rule.picks(X["F"], wk0, imap, P, ctx)
    same = sum(A0["V3"][0][d] == live.get(d, []) for d in wk0)
    skipped = sum(bool(OFF.get(d, False)) for d in wk0)
    print(f"wiring check: V3 arm equals v3_rule.picks on {same}/{len(wk0)} weeks · weekly buys skipped when risk-off: {skipped}/{len(wk0)}", flush=True)
    if same < len(wk0):
        raise SystemExit("wiring check FAILED — no result reported")

    # ---- outcomes
    res, navs, ph = {}, {}, {a: [] for a in ARMS[1:]}
    for o in range(5):
        wk = wk0 if o == 0 else [d for d in sp.weekly_grid(cal, o) if d >= START]
        A = A0 if o == 0 else arms(wk)
        R = {}
        for a in ARMS:
            sel, x, rule = A[a]
            nav, info = sre.run_exit(D, x, sel, wk, rule)
            m = sp.metrics(nav)
            R[a] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, avg_names=info["avg_names"], sold_early_pct=info["pct_sold_early"])
            if o == 0:
                navs[a] = nav
        for a in ARMS[1:]:
            ph[a].append(bool(sp.beats(R[a], R["V3"])))
        if o == 0:
            res = R
        print(f"phase {o}: " + " · ".join(f"{a} {R[a]['cagr_disc']:.1f}/{R[a]['cagr_conf']:.1f}/{R[a]['maxdd']:.1f}" for a in ARMS), flush=True)
    verdict = [a for a in ARMS[1:] if ph[a][0] and sum(ph[a]) >= 4]

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · phase 0 ===")
    for a in ARMS:
        r = res[a]
        print(f"{a:9s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}"
              + (f" · sold early {r['sold_early_pct']:.0f}% of positions" if a == "V3_REXIT" else "")
              + (f" · beats V3 in {sum(ph[a])}/5 phases (phase 0 {'yes' if ph[a][0] else 'no'})" if a != "V3" else ""))
    cy = {a: (lambda n: {int(k): round(float(v), 1) for k, v in ((n.groupby(n.index.year).last() / n.groupby(n.index.year).last().shift(1).fillna(1.0) - 1) * 100).items()})(navs[a] / navs[a].iloc[0]) for a in ARMS}
    print("\ncalendar years % (V3 / skip / skip+exit): " + " · ".join(f"{y}: {cy['V3'][y]:+.0f} / {cy['V3_RSKIP'][y]:+.0f} / {cy['V3_REXIT'][y]:+.0f}" for y in cy["V3"]))
    win = {}
    for lab, (a_, b_) in WINDOWS.items():
        win[lab] = {a: round(float(navs[a][navs[a].index <= b_].iloc[-1] / navs[a][navs[a].index <= a_].iloc[-1] * 100 - 100), 1) for a in ARMS}
    print("stress windows % (V3 / skip / skip+exit): " + " · ".join(f"{k}: {v['V3']:+.1f} / {v['V3_RSKIP']:+.1f} / {v['V3_REXIT']:+.1f}" for k, v in win.items()))
    print(f"VERDICT: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **res[a], phases_beaten=(f"{sum(ph[a])}/5" if a in ph else None), PASS=(a in verdict) if a in ph else None) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_regime_switch.py", definitions=__doc__,
        units=dict(cagr="percent a year", maxdd="percent", sold_early_pct="percent of positions"), calendar_years=cy, stress_windows=win,
        risk_off_share_by_year={int(k): round(float(v), 3) for k, v in y.groupby(y.index.year).mean().items()}, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nCash while the broad market is below its 200-day average: does it cut V3's worst falls without costing return? "
                                      f"Verdict: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}. Numbers in results.csv; years, stress windows and risk-off share in its manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no arm passes",
                                 arms={a: [round(res[a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [f"{sum(ph[a])}/5" if a in ph else None] for a in ARMS},
                                 calendar_years=cy, stress_windows=win, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
