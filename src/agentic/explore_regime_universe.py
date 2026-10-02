"""Does V3 do worse when mid/small caps are weak, and would larger caps have held up? (2026-10-02, EXPLORATION, 2019-2022 only.)

Idea (Abhinav): Sept-Oct entries are held through Oct-April, which was bad for mid/small caps in 2024 and 2025; maybe switch
V3's universe when that happens. A calendar rule cannot be tested fairly (2023+ already sliced by season), so this looks at
the MARKET STATE at entry instead, on 2019-2022 only (2023+ stays the clean era for a registered test).
Regime at entry = share of tradable (core-band) stocks whose close is above their 200-day average; WEAK if below 50%.
Selections per weekly batch (phase 0, entry next open, exit close of session 126, 0.5% round-trip cost):
  V3        live rule (G1 top 9 minus financials minus fading-only themes, no refill)
  V3_LARGE  same rule on the pool restricted to market cap >= Rs 5,000 cr at entry (fixed round number, not tuned)
  TYPICAL   median return of all core-band stocks entered the same day
Batch return = equal-weight average of its picks. Thresholds (50%, Rs 5,000 cr) were fixed before looking at any outcome.
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
import build_themes  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

COST = 0.005
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
G1 = S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)]
sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet"); sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"])
sc["industry"] = sc["industry"].map(html.unescape); sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
fin = lambda s: sector.get(imap.get(s)) == "Financial Services"  # noqa: E731
fading_only, _, _ = build_themes.fading_checker()
wk = [d for d in sp.weekly_grid(cal, 0) if pd.Timestamp("2019-01-01") <= d <= pd.Timestamp("2022-12-31")]
top = sre.TOPN; sre.TOPN = 100000
full = tif.select_elig(X["F"], wk, imap, P, G1); sre.TOPN = top
M = pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet", columns=["symbol", "trade_date", "mcap_cr"])
M["trade_date"] = pd.to_datetime(M["trade_date"]); M = M[M["trade_date"].isin(set(wk))]
mcap = {(s, d): v for s, d, v in zip(M["symbol"], M["trade_date"], M["mcap_cr"])}
F = X["F"]; Fw = F[F["trade_date"].isin(set(wk)) & F["core"]]
breadth = (Fw["close"] > Fw["sma_200"]).groupby(Fw["trade_date"]).mean()
core_syms = Fw.groupby("trade_date")["symbol"].apply(list)
PX = rp.load_panel(["open", "close"]); O, C = rp.wide(PX, "open", cal), rp.wide(PX, "close", cal); del PX


def ret(s, i0):
    if s not in O.columns:
        return np.nan
    e, x = O[s].iloc[i0], C[s].iloc[i0 + 125]
    return x / e - 1 - COST if np.isfinite(e) and e > 0 and np.isfinite(x) else np.nan


rows, cover = [], []
for d in wk:
    i0 = cal.get_loc(d) + 1
    if i0 + 126 > len(cal):
        continue
    v3 = [s for s in full.get(d, [])[:9] if not fin(s) and not fading_only(s, d)]
    big = [s for s in full.get(d, []) if mcap.get((s, d), 0) >= 5000]
    v3l = [s for s in big[:9] if not fin(s) and not fading_only(s, d)]
    cover.append(np.mean([(s, d) in mcap for s in full.get(d, [])[:30]]) if full.get(d) else np.nan)
    uni = [ret(s, i0) for s in core_syms.get(d, [])]
    rows.append(dict(week=d, breadth=breadth.get(d, np.nan), n_v3=len(v3), n_large=len(v3l),
                     V3=np.nanmean([ret(s, i0) for s in v3]) if v3 else np.nan,
                     V3_LARGE=np.nanmean([ret(s, i0) for s in v3l]) if v3l else np.nan,
                     TYPICAL=np.nanmedian(uni) if uni else np.nan,
                     mcap_v3=np.nanmedian([mcap.get((s, d), np.nan) for s in v3]) if v3 else np.nan))
R = pd.DataFrame(rows); R["regime"] = np.where(R["breadth"] < 0.5, "WEAK", "STRONG")
print(f"market-cap known for the top-30 ranked names: {np.nanmean(cover):.1%} of name-weeks (data check)")
print(f"weeks {len(R)} · weak {int((R.regime == 'WEAK').sum())} · strong {int((R.regime == 'STRONG').sum())} · "
      f"V3 picks' median market cap Rs {R['mcap_v3'].median():,.0f} cr · large-cap list empty in {int((R.n_large == 0).sum())} weeks")
pct = lambda v: f"{v:+.1%}"  # noqa: E731
for g, x in R.groupby("regime"):
    print(f"\n{g} ({len(x)} weeks):")
    for k in ("V3", "V3_LARGE", "TYPICAL"):
        y = x[k].dropna()
        print(f"  {k:9s} avg {pct(y.mean())} · median {pct(y.median())} · losing batches {(y < 0).mean():.0%} (n={len(y)})")
print("\nWEAK weeks by year (avg batch return):")
w = R[R.regime == "WEAK"]
print(w.groupby(w.week.dt.year)[["V3", "V3_LARGE", "TYPICAL"]].agg(["mean", "count"]).round(3).to_string())
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-02-regime-universe-EXPLORATION",
        status="EXPLORATION 2019-2022 only (2023+ untouched for this idea); no rule changed", producer="src/agentic/explore_regime_universe.py",
        results={g: {k: dict(avg=round(float(x[k].mean()), 4), median=round(float(x[k].median()), 4), n=int(x[k].notna().sum()))
                     for k in ("V3", "V3_LARGE", "TYPICAL")} for g, x in R.groupby("regime")})) + "\n")
