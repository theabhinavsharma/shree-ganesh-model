"""EXP-2026-09-30-v3-pnl (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: does Sri Lakshmi V3 pick better when its ranking model can also see the company's P&L (sales growth,
EPS growth, P/E, P/E vs industry, profitable, turned profitable, days since results)?
Arms (same V3 rule: G1 top 9 minus Financial Services minus stocks tied only to fading themes, no refill):
  V3   live model scores (logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet; 50 features, trained on rows from 2016)
  V3T  model retrained on the same 50 features from rows dated 2019-01-01 on (logs/leader_sleeve/v3_pnl/model_T)
  V3P  model retrained on the 50 features + 7 P&L features from rows dated 2019-01-01 on (logs/leader_sleeve/v3_pnl/model_P)
Window: weekly entries from 2020-01-01 (the first year a 2019-trained model can score); eras 2020-2022 and 2023+.
ADOPTION PASS = sp.beats(V3P, V3) at phase 0 and in >= 4 of 5 phases. DIAGNOSTIC: V3P vs V3T (the P&L effect alone).
SGM_PNL_FULL=1: EXP-2026-09-30-v3-pnl-full — V3 vs V3F (rows rebuilt on the P&L backfill, 50 features, all years) vs V3PF
(same rows, + the 7 P&L features, all years), entries from 2019-01-01; adoption = V3PF beats V3 (same pass rule).
Before any outcome: trust/data_ready.gate() (configs/data_bar.json). Below the bar -> DATA-READY NOT READY, exit 3, no
result. SGM_GATE_ONLY=1: run the gate and stop.
SGM_RERUN_TAG=<tag>: same registered test on re-built model scores (e.g. after the old-format P&L backfill); writes
v3_pnl/rerun_<tag>/ and logs EXP-2026-09-30-v3-pnl-RERUN-<tag>. SGM_REPRO=1: run, write repro/, log nothing.
"""
from __future__ import annotations

import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import build_themes  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

FULL = os.environ.get("SGM_PNL_FULL") == "1"   # EXP-2026-09-30-v3-pnl-full: P&L back to 2015 (old-format backfill), all years
EXP_ID = "EXP-2026-09-30-v3-pnl-full" if FULL else "EXP-2026-09-30-v3-pnl"
OUTDIR = ROOT / "logs/leader_sleeve/v3_pnl"
START = pd.Timestamp("2019-01-01" if FULL else "2020-01-01")
NAME = {"V3": "V3", "V3T": "V3F" if FULL else "V3T", "V3P": "V3PF" if FULL else "V3P"}


def scores(path: Path) -> pd.DataFrame:
    P = pd.read_parquet(path, columns=["symbol", "trade_date", "ensemble"])
    P["trade_date"] = pd.to_datetime(P["trade_date"]).astype("datetime64[ns]")
    return P.dropna(subset=["ensemble"]).sort_values("trade_date")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]
    base = OUTDIR / "full" if FULL else (OUTDIR / f"rerun_{os.environ['SGM_RERUN_TAG']}" if os.environ.get("SGM_RERUN_TAG") else OUTDIR)
    tdir, pdir = ("model_F", "model_PF") if FULL else ("model_T", "model_P")
    PS = {"V3": sre.model_scores(), "V3T": scores(base / tdir / "bakeoff_preds.parquet"), "V3P": scores(base / pdir / "bakeoff_preds.parquet")}
    for k, P in PS.items():
        print(f"{NAME[k]}: scores {P['trade_date'].min().date()}..{P['trade_date'].max().date()} · {len(P):,} rows", flush=True)
    S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
    G1 = S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)]
    sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"]); sc["industry"] = sc["industry"].map(html.unescape)
    sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
    fin = lambda s: sector.get(imap.get(s)) == "Financial Services"  # noqa: E731
    fading_only, _, _ = build_themes.fading_checker()
    grid = lambda o: [d for d in sp.weekly_grid(D["cal"], o) if d >= START]  # noqa: E731

    # ---- data-readiness gate (2026-10-02, GIGO): no result on P&L data below configs/data_bar.json
    sys.path.insert(0, str(ROOT / "src/agentic/trust"))
    import data_ready as dr
    PNL_F = ["sales_yoy", "eps_yoy", "profitable", "loss_to_profit", "days_since_results"]   # pe / pe_ind blank for loss-makers by
    rows_path = OUTDIR / "full/rows/rows.parquet" if FULL else ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet"  # definition:
    pnl_path = ROOT / ("data/derived/pnl_quarterly_enriched.parquet" if FULL else "data/derived/pnl_quarterly.parquet")  # 'profitable'
    R = pd.read_parquet(rows_path, columns=["trade_date", "core"] + PNL_F); R = R[R["core"].astype(bool)]  # stands in for them
    Q = pd.read_parquet(pnl_path, columns=["symbol", "quarter_end"]); Q["y"] = pd.to_datetime(Q["quarter_end"]).dt.year
    C = pd.read_parquet(ROOT / "data/derived/results_calendar_full.parquet", columns=["symbol", "toDate", "period"])
    C = C[C["period"].astype(str).str.lower() == "quarterly"]; C["y"] = pd.to_datetime(C["toDate"], format="%d-%b-%Y", errors="coerce").dt.year
    have = Q.drop_duplicates(["symbol", "quarter_end"]).groupby("y").size()
    # 2026-10-04: NSE's own count = quarterly results calendar + integrated filings (2025+ results are filed only as
    # integrated filings, so the calendar alone left 2025-26 "official count missing")
    I = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet", columns=["symbol", "period_to"])
    I["qe"] = pd.to_datetime(I["period_to"], format="%d-%b-%Y", errors="coerce")
    C["qe"] = pd.to_datetime(C["toDate"], format="%d-%b-%Y", errors="coerce")
    off = pd.concat([C[["symbol", "qe"]], I[["symbol", "qe"]]]).dropna().drop_duplicates()
    official = off.groupby(off["qe"].dt.year).size()
    train_from = 2016 if FULL else 2019
    end = D["cal"][-1]
    checks = [dr.completeness("P&L company-quarters vs NSE's quarterly results calendar", have, official, range(train_from - 1, end.year + 1)),
              dr.coverage("P&L features filled in the model rows (liquid stocks)", R, "trade_date", PNL_F, range(train_from, end.year + 1))]
    checks += [dr.span(f"{NAME[k]} model scores", P["trade_date"], START, end) for k, P in PS.items()]
    dr.gate(EXP_ID, checks, run=os.environ.get("SGM_RERUN_TAG") or "RESULT")
    if os.environ.get("SGM_GATE_ONLY") == "1":
        return

    def v3(P, wk):
        top9 = tif.select_elig(X["F"], wk, imap, P, G1)
        return {d: [s for s in top9.get(d, []) if not fin(s) and not fading_only(s, d)] for d in wk}

    def run_phase(o: int) -> dict:
        wk = grid(o)
        sels = {k: v3(P, wk) for k, P in PS.items()}
        if o == 0:
            top = sre.TOPN; sre.TOPN = 100000
            full = tif.select_elig(X["F"], wk, imap, PS["V3"], G1); sre.TOPN = top
            alt = {d: [s for s in full.get(d, [])[:9] if not fin(s) and not fading_only(s, d)] for d in wk}
            same = sum(alt[d] == sels["V3"][d] for d in wk)
            print(f"wiring check: V3 from the full pool equals V3 from the top 9 on {same}/{len(wk)} weeks", flush=True)
            if same < len(wk):
                raise SystemExit("wiring check FAILED — no result reported")
            for k in ("V3T", "V3P"):
                print(f"weeks where {NAME[k]} picks differ from V3: {sum(sels[k][d] != sels['V3'][d] for d in wk)}/{len(wk)} · "
                      f"avg names {sum(len(v) for v in sels[k].values()) / len(wk):.1f}", flush=True)
        res = {}
        for k, pk in sels.items():
            nav, info = sre.run_exit(D, X, pk, wk, "E0")
            m = sp.metrics(nav)
            res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")}, avg_names=info["avg_names"])
        return res

    R0 = run_phase(0)
    ph = {"V3P_vs_V3": [bool(sp.beats(R0["V3P"], R0["V3"]))], "V3P_vs_V3T": [bool(sp.beats(R0["V3P"], R0["V3T"]))],
          "V3T_vs_V3": [bool(sp.beats(R0["V3T"], R0["V3"]))]}
    for o in (1, 2, 3, 4):
        Ro = run_phase(o)
        ph["V3P_vs_V3"].append(bool(sp.beats(Ro["V3P"], Ro["V3"]))); ph["V3P_vs_V3T"].append(bool(sp.beats(Ro["V3P"], Ro["V3T"])))
        ph["V3T_vs_V3"].append(bool(sp.beats(Ro["V3T"], Ro["V3"])))
    passed = bool(ph["V3P_vs_V3"][0] and sum(ph["V3P_vs_V3"]) >= 4)
    print(f"\n=== {EXP_ID} · entries from {START.date()} ===")
    for k in ("V3", "V3T", "V3P"):
        r = R0[k]
        print(f"{NAME[k]:4s} CAGR {r['cagr']:5.1f} · {START.year}-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}")
    for k, v in ph.items():
        print(f"{k.replace('V3P', NAME['V3P']).replace('V3T', NAME['V3T'])}: beats in {sum(v)}/5 phases (phase 0 {'yes' if v[0] else 'no'})")
    print(f"ADOPTION ({NAME['V3P']} vs V3): {'PASS' if passed else 'no pass'}")
    tag = os.environ.get("SGM_RERUN_TAG", "")
    out = (OUTDIR / "full" if FULL else OUTDIR) / ("repro" if os.environ.get("SGM_REPRO") == "1" else (f"rerun_{tag}" if tag else "results"))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=NAME[k], **r) for k, r in R0.items()]).to_csv(out / "results.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    (out / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_pnl.py",
        definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent"), phases={k: f"{sum(v)}/5" for k, v in ph.items()}, updated=now), indent=1))
    (out / "README.md").write_text(f"# {EXP_ID}\n\nDoes V3 pick better when its model also sees P&L? Adoption verdict (V3P vs V3): "
                                   f"{'PASS' if passed else 'no pass'}. See results.csv + manifest.\n")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=[NAME["V3P"]] if passed else "no arm passes",
                                 arms={NAME[k]: [round(R0[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for k in R0},
                                 phases={k: f"{sum(v)}/5" for k, v in ph.items()}, cols="CAGR, 2020-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
