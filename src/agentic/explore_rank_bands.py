"""Which slice of the model ranking does best, bought in April or October and held 3-24 months? (2026-10-08, EXPLORATION
for Abhinav: "what % of the top if held for 3/6/9/12/18/24 months gives the highest return? check in 5% intervals and go
granular if you find it wise. Pick 2 scenarios of Oct & April as the month where we invest"; no rule changed).

Ranking: the walk-forward model score (logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet, column ensemble; each test
year scored by models trained only on rows whose labels ended before that year) of every scored stock on the first
weekly score date of April and of October, 2019-2026. Scored universe: about 1,300 stocks in 2019, 2,400 in 2026.
This is the model alone across all stocks, not V3 (V3 = model rank inside hot industries + trend filter, top 9).
Slices: rank percentile bands 0-5%, 5-10%, ..., 95-100% (0 = highest score), 1% bands inside the top 20%, and
cumulative top X% (2, 5, 10, ..., 50, 100).
Trade: buy at the next session's open after the score date, sell at the close H sessions later (H = 63 / 126 / 189 / 252
/ 378 / 504 sessions = 3 / 6 / 9 / 12 / 18 / 24 months), equal money in every stock of the slice. A stock that stops
trading keeps its last close (up to 300 sessions). Per-stock return clipped to [-100%, +1000%]. Before costs.
A cohort (one April or one October) counts for a hold only once the hold has ended.
Reported: average over cohorts of the slice's average stock return (each cohort counts once), the same minus the
all-scored average (excess), and how many cohorts the slice beat the all-scored average in.
Output: logs/explorations/rank_bands/ (cohorts.csv + manifest + README) and an EXPLORATION line in logs/experiments.jsonl.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402

OUT = ROOT / "logs/explorations/rank_bands"
HOLDS = {"3m": 63, "6m": 126, "9m": 189, "12m": 252, "18m": 378, "24m": 504}
MONTHS = {"Apr": 4, "Oct": 10}
BANDS5 = [(i * 5, i * 5 + 5) for i in range(20)]
BANDS1 = [(i, i + 1) for i in range(20)]
CUM = [2, 5, 10, 15, 20, 25, 30, 40, 50, 100]

P = sre.model_scores()
PX = rp.load_panel(["open", "close"]); cal = rp.session_calendar(PX)
O = rp.wide(PX, "open", cal); C = rp.wide(PX, "close", cal).ffill(limit=300); del PX
On, Cn, col = O.to_numpy(), C.to_numpy(), {s: i for i, s in enumerate(O.columns)}
dates = pd.DatetimeIndex(sorted(P["trade_date"].unique()))

rows = []
for y in range(2019, cal[-1].year + 1):
    for mname, m in MONTHS.items():
        ds = dates[(dates.year == y) & (dates.month == m)]
        if not len(ds):
            continue
        d = ds[0]; i0 = int(cal.searchsorted(d, side="right"))
        S = P[P["trade_date"] == d][["symbol", "ensemble"]].copy()
        S["pct"] = S["ensemble"].rank(ascending=False, method="first") / len(S) * 100   # (0, 100], 0 = best
        S["c"] = S["symbol"].map(col)
        S = S.dropna(subset=["c"]); S["c"] = S["c"].astype(int)
        for h, n in HOLDS.items():
            if i0 + n > len(cal):
                continue
            e, x = On[i0, S["c"].to_numpy()], Cn[i0 + n - 1, S["c"].to_numpy()]
            ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
            r = pd.Series(np.where(ok, np.clip(x / np.where(ok, e, 1) - 1, -1, 10), np.nan), index=S.index)
            T = S.assign(r=r).dropna(subset=["r"])
            allavg = T["r"].mean()
            base = dict(month=mname, year=y, score_date=str(d.date()), entry=str(cal[i0].date()), hold=h, all_avg=allavg, all_n=len(T))
            for lo, hi in BANDS5:
                g = T[(T["pct"] > lo) & (T["pct"] <= hi)]
                rows.append(dict(base, kind="band5", slice=f"{lo}-{hi}%", avg=g["r"].mean(), median=g["r"].median(), n=len(g)))
            for lo, hi in BANDS1:
                g = T[(T["pct"] > lo) & (T["pct"] <= hi)]
                rows.append(dict(base, kind="band1", slice=f"{lo}-{hi}%", avg=g["r"].mean(), median=g["r"].median(), n=len(g)))
            for k in CUM:
                g = T[T["pct"] <= k]
                rows.append(dict(base, kind="cum", slice=f"top {k}%", avg=g["r"].mean(), median=g["r"].median(), n=len(g)))
R = pd.DataFrame(rows); R["excess"] = R["avg"] - R["all_avg"]
OUT.mkdir(parents=True, exist_ok=True)
R.to_csv(OUT / "cohorts.csv", index=False)

order = {k: list(dict.fromkeys(R.loc[R["kind"] == k, "slice"])) for k in ("band5", "band1", "cum")}
summary = {}


def table(kind: str, months: list, title: str) -> None:
    X = R[(R["kind"] == kind) & R["month"].isin(months)]
    avg = X.pivot_table(index="slice", columns="hold", values="avg", aggfunc="mean").reindex(index=order[kind], columns=list(HOLDS)) * 100
    exc = X.pivot_table(index="slice", columns="hold", values="excess", aggfunc="mean").reindex(index=order[kind], columns=list(HOLDS)) * 100
    won = X.assign(w=X["excess"] > 0).pivot_table(index="slice", columns="hold", values="w", aggfunc="sum").reindex(index=order[kind], columns=list(HOLDS))
    ncoh = X.groupby("hold")["year"].nunique().reindex(list(HOLDS)) if len(months) == 1 else X.groupby("hold").apply(lambda g: len(g[["month", "year"]].drop_duplicates())).reindex(list(HOLDS))
    allv = X.drop_duplicates(["month", "year", "hold"]).groupby("hold")["all_avg"].mean().reindex(list(HOLDS)) * 100
    print(f"\n=== {title} · average return % per cohort (excess over all scored stocks) [cohorts beaten / cohorts] ===")
    print(f" {'slice':10s}" + "".join(f"{h:>22s}" for h in HOLDS))
    for s in order[kind]:
        print(f" {s:10s}" + "".join(f"{avg.at[s, h]:+8.0f} ({exc.at[s, h]:+5.0f}) [{int(won.at[s, h])}/{int(ncoh[h])}]" if np.isfinite(avg.at[s, h]) else f"{'':>22s}" for h in HOLDS))
    print(f" {'all scored':10s}" + "".join(f"{allv[h]:+8.0f}{'':14s}" for h in HOLDS))
    best = {h: (avg[h].idxmax(), round(float(avg[h].max()), 1)) for h in HOLDS if avg[h].notna().any()}
    print(" best slice by hold: " + " · ".join(f"{h} {b[0]} ({b[1]:+.0f}%)" for h, b in best.items()))
    summary[f"{kind}|{'+'.join(months)}"] = dict(best=best, avg=avg.round(1).to_dict(), excess=exc.round(1).to_dict(), cohorts=ncoh.to_dict(), all_scored=allv.round(1).to_dict())


for months, lab in ((["Apr"], "APRIL entries"), (["Oct"], "OCTOBER entries"), (["Apr", "Oct"], "APRIL + OCTOBER")):
    table("band5", months, f"{lab} · 5% bands of the model rank")
    table("cum", months, f"{lab} · cumulative top X%")
table("band1", ["Apr", "Oct"], "APRIL + OCTOBER · 1% bands inside the top 20%")

print("\n=== by entry year: top 5% band vs all scored · average return % (12m hold; 6m where 12m not ended) ===")
for mname in MONTHS:
    X = R[(R["kind"] == "band5") & (R["slice"] == "0-5%") & (R["month"] == mname)]
    for y, g in X.groupby("year"):
        h = "12m" if (g["hold"] == "12m").any() else "6m"
        r = g[g["hold"] == h].iloc[0]
        print(f"  {mname} {y} ({h}): top 5% {r['avg'] * 100:+6.0f} · all {r['all_avg'] * 100:+6.0f} · n {int(r['n'])}")

(OUT / "cohorts.csv.manifest.json").write_text(json.dumps(dict(
    dataset="cohorts.csv", producer="src/agentic/explore_rank_bands.py", status="EXPLORATION (no rule changed)", definitions=__doc__,
    key=["month", "year", "hold", "kind", "slice"],
    columns=dict(month="entry month scenario (Apr / Oct)", year="calendar year of the entry", score_date="model score date (session close)",
                 entry="buy session (open)", hold="hold label; sessions in HOLDS", kind="band5 = 5% rank bands, band1 = 1% bands in the top 20%, cum = top X% cumulative",
                 slice="rank percentile range, 0 = highest score", avg="average stock return over the hold, fraction (0.10 = +10%)",
                 median="median stock return, fraction", n="stocks in the slice with an entry price", all_avg="average over every scored stock, fraction",
                 all_n="scored stocks with an entry price", excess="avg - all_avg, fraction"),
    holds_sessions=HOLDS, model_scores=str(sre.PREDS.relative_to(ROOT)), price_panel=str(rp.PANEL.relative_to(ROOT)),
    caveats=["few cohorts: 8 Aprils (2019-2026) and 7 Octobers (2019-2025) at most, fewer for long holds; 18m and 24m cohorts overlap",
             "before costs; the scored universe includes illiquid small stocks that cannot take real money at size",
             "a stock that stops trading keeps its last close (delistings at a loss may be flattered)"],
    updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
(OUT / "README.md").write_text("# Model rank bands, April / October entries (exploration)\n\nWhich slice of the model's ranking (top 5%, 5-10%, ...) "
                               "returned most when bought in April or October and held 3 to 24 months. One row per cohort x hold x slice in "
                               "cohorts.csv; definitions and caveats in its manifest. Exploration only; no rule changed.\n")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-08-rank-bands-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_rank_bands.py", results=summary), default=str) + "\n")
print("done")
