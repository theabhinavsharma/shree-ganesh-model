"""Sri Lakshmi V3 pick stats + time of year (2026-09-30; EXPLORATION, no rule is changed by this).

V3 = G1 top 9 minus Financial Services minus stocks tied only to fading themes, no refill (production from 2026-10-02).
Weekly picks 2019+, phase 0, entry = next session open, 126-session hold, returns before costs.
Per pick: peak = best high in the hold / entry - 1; final = close of session 126 / entry - 1;
days to +50% / 2x = sessions (and calendar days) from the entry session to the first high at 1.5x / 2x the entry.
Part 1 (pick stats) covers every era: it describes a rule whose returns are already reported for both eras.
Part 2 (entry month, pre-Diwali) slices 2019-2022 ONLY: slicing by season is idea-hunting, and 2023+ stays sealed so a
seasonal rule, if ever proposed, can still be tested cleanly there. --unseal-2023 (Abhinav asked for insight, not a rule,
2026-09-30) also slices 2023-2025; after that run 2023+ is NO LONGER clean confirmation for any seasonal rule (logged). Diwali = Lakshmi Puja date; "pre-Diwali" = entry
within the 42 calendar days before it.
"""
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402
import v3_rule  # noqa: E402

DIWALI = {2019: "2019-10-27", 2020: "2020-11-14", 2021: "2021-11-04", 2022: "2022-10-24",
          2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-21"}   # 2025 = NSE Muhurat trading day

D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)                                  # the V3 rule, shared with the live screen
wk = [d for d in sp.weekly_grid(cal, 0) if d >= pd.Timestamp(tif.START)]
v3 = v3_rule.picks(X["F"], wk, imap, P, ctx)
PX = rp.load_panel(["open", "high", "low", "close"]); O, H, L, C = (rp.wide(PX, c, cal) for c in ("open", "high", "low", "close")); del PX
rows = []
for d in wk:
    i0 = cal.get_loc(d) + 1
    if i0 + 126 > len(cal):
        continue
    for s in v3.get(d, []):
        e = O[s].iloc[i0] if s in O.columns else np.nan
        if not np.isfinite(e):
            continue
        hh = H[s].iloc[i0:i0 + 126].to_numpy()
        hit = lambda m: (lambda j: (int(j[0]) + 1, (cal[i0 + int(j[0])] - cal[i0]).days) if len(j) else (np.nan, np.nan))(np.flatnonzero(hh >= m * e))  # noqa: E731
        (s50, c50), (s100, c100) = hit(1.5), hit(2.0)
        rows.append(dict(week=d, symbol=s, peak=np.nanmax(hh) / e - 1, final=C[s].iloc[i0 + 125] / e - 1,
                         s50=s50, c50=c50, s100=s100, c100=c100))
R = pd.DataFrame(rows); R["era"] = np.where(R["week"] < pd.Timestamp("2023-01-01"), "2019-22", "2023+")


def stats(x):
    h, h2 = x[x["s50"].notna()], x[x["s100"].notna()]
    return pd.Series({
        "picks": len(x), "batches": x["week"].nunique(),
        "hit +50% (any time in hold)": f"{len(h) / len(x):.0%}", "hit 2x": f"{len(h2) / len(x):.0%}",
        "days to +50%: median / avg (calendar)": f"{h['c50'].median():.0f} / {h['c50'].mean():.0f}" if len(h) else "-",
        "days to +50%: fastest quarter hit by": f"{h['c50'].quantile(0.25):.0f}" if len(h) else "-",
        "hitters that got there in 3 months": f"{(h['c50'] <= 91).mean():.0%}" if len(h) else "-",
        "days to 2x: median (calendar)": f"{h2['c100'].median():.0f}" if len(h2) else "-",
        "ended (day 126) +50% or more": f"{(x['final'] >= 0.5).mean():.0%}", "ended in a loss": f"{(x['final'] < 0).mean():.0%}",
        "median / average final": f"{x['final'].median():+.0%} / {x['final'].mean():+.0%}",
        "hitters that ended below +50%": f"{(h['final'] < 0.5).mean():.0%}" if len(h) else "-"})


print("=== PART 1 · V3 picks (all eras) ===")
print(pd.concat({e: stats(R if e == "all" else R[R["era"] == e]) for e in ("2019-22", "2023+", "all")}, axis=1).to_string())

def season(label, x):
    B = x.groupby("week").agg(batch_final=("final", "mean"), hit=("s50", lambda v: v.notna().mean()), n=("symbol", "size")).reset_index()
    B["month"] = B["week"].dt.month
    B["pre_diwali"] = B["week"].apply(lambda w: 0 <= (pd.Timestamp(DIWALI.get(w.year, "1900-01-01")) - w).days <= 42)
    fmt = lambda v: f"{v:+.0%}" if abs(v) < 5 else f"{v:.0f}"  # noqa: E731
    print(f"\n=== PART 2 · {label} · by entry month (batch = equal-weight average of its picks, before costs) ===")
    M = B.groupby("month").agg(batches=("week", "size"), avg_batch_final=("batch_final", "mean"), median_batch_final=("batch_final", "median"),
                                hit_rate=("hit", "mean"), losing_batches=("batch_final", lambda v: (v < 0).mean()))
    M.index = [pd.Timestamp(2000, m, 1).strftime("%b") for m in M.index]
    print(M.to_string(float_format=fmt))
    Y = B.assign(year=B["week"].dt.year).groupby(["year", "pre_diwali"]).agg(batches=("week", "size"), avg_batch_final=("batch_final", "mean"),
                                                                             hit_rate=("hit", "mean")).unstack()
    print(f"\n=== pre-Diwali (entry 0-42 days before Diwali) vs rest of the year, {label} ===")
    print(Y.to_string(float_format=fmt))
    pdw, rest = B[B["pre_diwali"]], B[~B["pre_diwali"]]
    print(f"\npre-Diwali batches {len(pdw)}: avg {pdw['batch_final'].mean():+.0%}, hit {pdw['hit'].mean():.0%} · rest {len(rest)}: avg {rest['batch_final'].mean():+.0%}, hit {rest['hit'].mean():.0%}")
    return dict(pre_diwali=dict(batches=len(pdw), avg_final=round(pdw["batch_final"].mean(), 3), hit=round(pdw["hit"].mean(), 3)),
                rest=dict(batches=len(rest), avg_final=round(rest["batch_final"].mean(), 3), hit=round(rest["hit"].mean(), 3)))


UNSEAL = "--unseal-2023" in sys.argv
out = {"2019-22": season("2019-22", R[R["era"] == "2019-22"])}
if UNSEAL:
    out["2023-25"] = season("2023-25", R[(R["week"] >= "2023-01-01") & (R["week"] < "2026-01-01")])

if __import__("os").environ.get("SGM_NO_LOG") == "1":   # verification runs (e.g. after a refactor) log nothing
    raise SystemExit(0)
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"),
        id="EXP-2026-09-30-v3-season" + ("-2023-EXPLORATION" if UNSEAL else "-EXPLORATION"),
        status=("EXPLORATION for insight only (no rule). 2023-2025 sliced by season at Abhinav's request: 2023+ is no longer "
                "clean confirmation for any seasonal rule" if UNSEAL else "EXPLORATION (no rule changed; 2023+ not sliced by season)"),
        producer="src/agentic/report_v3_stats.py",
        v3_all=dict(picks=len(R), hit50=round(R["s50"].notna().mean(), 3), hit2x=round(R["s100"].notna().mean(), 3),
                    median_days_to_50=float(R["c50"].median())), season=out)) + "\n")
