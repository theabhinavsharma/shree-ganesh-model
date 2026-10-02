"""EXP-2026-10-02-weekly7d-6m-hold — the Mar/Apr 2026 weekly 7-day-winners logic, held 6 months.

Picks come from the model's OWN yearly walk-forward (src.ml.expert_pipeline, focus horizon
week_7), re-run with training ending 2019 so every test year 2020..2025 is out-of-sample:
  data/ml/expert_runs/wf2020_week7_oof.parquet
Each week's first trading day, universe mid_small: target-zone names (pred 7d return in
[5%,10%]) first, then 0.55*prob_5pct + 0.20*prob_10pct + 0.75*clip(pred_return) — the live
ranking_score without calibration (monotone) and the 0.10 aux-horizon weight; top 12.
Outcome: equal weight, next open -> close of trading day 126, 0.30% RT.

Writes reports/backtest_weekly7d_6m_hold.md and appends to logs/experiments.jsonl.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
OOF = ROOT / "data/ml/expert_runs/wf2020_week7_oof.parquet"
OLD_OOF = ROOT / "data/ml/expert_runs/20260408T090057Z/week_7_oof_predictions.parquet"
EXP_ID = "EXP-2026-10-02-weekly7d-6m-hold"
TOP, HOLD, COST = 12, 126, 0.30


def weekly_picks(oof: pd.DataFrame, universe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = oof[oof["universe_name"] == universe].copy()
    u["trade_date"] = pd.to_datetime(u["trade_date"])
    days = np.array(sorted(u["trade_date"].unique()))
    first = sorted({pd.Timestamp(days[np.searchsorted(days, np.datetime64(w))])
                    for w in pd.date_range(days[0], days[-1], freq="W-MON") if np.searchsorted(days, np.datetime64(w)) < len(days)})
    u = u[u["trade_date"].isin(first)]
    u["zone"] = u["pred_return"].between(0.05, 0.10)
    u["score"] = 0.55 * u["prob_5pct"] + 0.20 * u["prob_10pct"] + 0.75 * u["pred_return"].clip(-0.20, 0.20)
    u = u.sort_values(["trade_date", "zone", "score", "symbol"], ascending=[True, False, False, True])
    return u.groupby("trade_date").head(TOP).copy(), u


def six_month(rows: pd.DataFrame, px: dict) -> pd.Series:
    out = []
    for r in rows[["symbol", "trade_date"]].itertuples(index=False):
        g = px.get(r.symbol)
        if g is None:
            out.append(np.nan); continue
        i = np.searchsorted(g["d"], np.datetime64(r.trade_date), side="right")   # next session
        j = i + HOLD - 1
        if i >= len(g["d"]) or j >= len(g["d"]) or np.isnan(g["o"][i]):
            out.append(np.nan); continue
        out.append((g["c"][j] / g["o"][i] - 1) * 100 - COST)
    return pd.Series(out, index=rows.index)


def summarise(picks: pd.DataFrame, univ: pd.DataFrame) -> pd.DataFrame:
    wk = picks.groupby("trade_date").agg(port=("r6m", "mean"), n=("r6m", "count"), up=("r6m", lambda s: (s > 0).mean()))
    wk = wk[wk["n"] >= 6]
    wk["univ_med"] = univ.groupby("trade_date")["r6m"].median().reindex(wk.index)
    wk["beat"] = wk["port"] > wk["univ_med"]
    wk["year"] = wk.index.year
    rows = []
    for y, s in list(wk.groupby("year")) + [("all", wk)]:
        rows.append({"entry_year": y, "weeks": len(s), "mean_6m": s["port"].mean(), "median_6m": s["port"].median(),
                     "worst_week": s["port"].min(), "best_week": s["port"].max(), "pct_picks_up": s["up"].mean() * 100,
                     "univ_median_6m": s["univ_med"].median(), "pct_weeks_beat_univ": s["beat"].mean() * 100})
    return pd.DataFrame(rows)


def main():
    print("prices…", flush=True)
    p = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["symbol", "trade_date", "open", "close"])
    p["trade_date"] = pd.to_datetime(p["trade_date"]); p = p.sort_values(["symbol", "trade_date"])
    px = {s: {"d": g["trade_date"].values, "o": g["open"].values, "c": g["close"].values} for s, g in p.groupby("symbol")}
    last_close = p["trade_date"].max()
    oof = pd.read_parquet(OOF)
    md = [f"# Weekly 7-day-winners logic, held 6 months — {EXP_ID}", "",
          f"_generated {datetime.now():%Y-%m-%d %H:%M} · picks: {OOF.relative_to(ROOT)} (walk-forward, train ≤ year-1, test 2020-2025) · "
          f"prices: data/derived/stock_daily_facts_adjusted_2015plus.parquet through {last_close.date()} · top {TOP}/week, next open → close of day {HOLD}, 0.30% RT_", ""]
    res = {}
    for uni in ("mid_small", "liquid_5cr_plus"):
        picks, all_u = weekly_picks(oof, uni)
        picks["r6m"] = six_month(picks, px)
        base = all_u.groupby("trade_date").sample(n=min(150, all_u.groupby("trade_date").size().min()), random_state=0) \
            if len(all_u) > 400000 else all_u
        base = base.copy(); base["r6m"] = six_month(base, px)
        t = summarise(picks, base); res[uni] = t.round(2).to_dict(orient="records")
        md += [f"## Universe `{uni}`{' (the one the 8-Apr live run used)' if uni == 'mid_small' else ' (secondary)'}", "",
               "| entry year | weeks | avg 6M % | median 6M % | worst week % | best week % | % picks up | universe median 6M % | % weeks beating universe |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in t.itertuples():
            md.append(f"| {r.entry_year} | {r.weeks} | {r.mean_6m:+.1f} | {r.median_6m:+.1f} | {r.worst_week:+.1f} | {r.best_week:+.1f} | "
                      f"{r.pct_picks_up:.0f} | {r.univ_median_6m:+.1f} | {r.pct_weeks_beat_univ:.0f} |")
        md.append("")
    # fidelity: overlap with the saved 8-Apr walk-forward (same yearly folds 2023-25, older panel)
    if OLD_OOF.exists():
        a, _ = weekly_picks(oof[pd.to_datetime(oof["trade_date"]).dt.year >= 2023], "mid_small")
        b, _ = weekly_picks(pd.read_parquet(OLD_OOF), "mid_small")
        ov = [len(set(x["symbol"]) & set(b[b["trade_date"] == d]["symbol"])) for d, x in a.groupby("trade_date") if d in set(b["trade_date"])]
        md += [f"Fidelity check: 2023-25 weekly top-{TOP} overlap with the saved 8-Apr run (same folds, pre-repair panel): "
               f"mean {np.mean(ov):.1f}/{TOP} over {len(ov)} weeks.", ""]
    md += ["## Caveats", "",
           "- The live run only produced a shortlist when its regime gate was active; this invests every week.",
           "- Ranking drops the isotonic calibration (monotone) and the 0.10 weight on the 1-day/15-day models, which the walk-forward file doesn't carry.",
           "- 12-name weekly cohorts overlap heavily week to week; weeks are not independent samples.",
           "- Universe median is computed on a random 150 names per week when the universe is large."]
    (ROOT / "reports/backtest_weekly7d_6m_hold.md").write_text("\n".join(md) + "\n")
    with open(ROOT / "logs/experiments.jsonl", "a") as f:
        f.write(json.dumps({"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "id": EXP_ID, "status": "MEASURED", "result": res}) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
