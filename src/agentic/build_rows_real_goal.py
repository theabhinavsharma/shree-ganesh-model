"""Model rows with the real goal as the target and V3.2's rule ingredients as inputs (EXP-2026-10-13-real-goal-model, 2026-10-13).

Plain English: the live model learns 'did the stock TOUCH +50% within 95 sessions'. What we earn is the 6-month (126-session)
return from the list-day close, with dividends, big payouts / demergers as cash, after liquidity-scaled costs. This copies the
live anatomy rows and adds:
  y126c50          1 if that 6-month return after costs is >= +50%, 0 if not, blank if the 126 sessions have not passed
  fwd126_net       the 6-month return itself (fraction; diagnostic, never an input)
  t2t              1 if the stock trades in the trade-for-trade segment (BE / BZ series) on the row date
  fin_sector       1 if its industry's broad sector is Financial Services (screener industry map, as V3's filter)
  fading_only      1 if its filings of the last 365 days tie it only to fading themes (build_themes.fading_checker, as V3)
  g1_elig          1 if its industry is G1-eligible that week (heat_pct >= 0.70 and policy pillar P_pct not < 0.30)
  ind_policy_pct   the industry's policy-pillar percentile P_pct that week (blank where the industry has no score)
  trend_ok         1 if it passes V3's trend test (>= 50% above the 52-week low, 1-year return >= 30%, close above the
                   200-day average, 50-day above 200-day)
  young_listing    1 if it listed after Jan 2015 (not a rename) and is 63-251 sessions old (V3.1's new-listing exemption)
The live rows, model and scores are not touched. Output: logs/leader_sleeve/model_real_goal/rows.parquet (+ manifest).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import total_return as tr  # noqa: E402
import v3_rule  # noqa: E402

SRC = ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet"
OUT = ROOT / "logs/leader_sleeve/model_real_goal/rows.parquet"
HOLD, GOAL = 126, 0.50
NEW = ["t2t", "fin_sector", "fading_only", "g1_elig", "ind_policy_pct", "trend_ok", "young_listing"]


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]
    ctx = v3_rule.context(imap)
    R = pd.read_parquet(SRC); R["trade_date"] = pd.to_datetime(R["trade_date"])
    T = len(cal); tpos = pd.Series(np.arange(T), index=cal)
    i0 = tpos.reindex(R["trade_date"]).to_numpy()
    c = R["symbol"].map(D["cidx"]).to_numpy(dtype=float)
    ok = np.isfinite(i0) & np.isfinite(c); i0c = np.where(ok, i0, 0).astype(int); cc = np.where(ok, c, 0).astype(int)

    # ---- target: 126-session return from the row-date close, dividends counted, payouts / demergers as cash, after costs
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)])   # lg[i] = log growth to session i-1
    end = i0c + HOLD; done = ok & (end <= T - 1)
    g_end = np.where(done, np.exp(lg[np.minimum(end, T - 1) + 1, cc] - lg[i0c + 1, cc]), np.nan)
    vr = D.get("vr", {})
    for q, events in vr.items():                       # first value-removal event inside the hold -> that share becomes cash
        rows = np.where(done & (cc == q))[0]
        for k, f in events:
            hit = rows[(i0c[rows] + 1 <= k) & (k <= i0c[rows] + HOLD - 1)]
            hit = hit[~np.isin(hit, np.where(np.isnan(g_end))[0])]
            if len(hit):
                base = np.exp(lg[k + 1, q] - lg[i0c[hit] + 1, q])
                g_end[hit] = f * g_end[hit] + (1 - f) * base
                rows = np.setdiff1d(rows, hit)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    adv0 = np.where(ok, ADV[i0c, cc], np.nan)
    cost = np.where(adv0 >= 5, 0.005, np.where(adv0 >= 1, 0.010, 0.020))
    R["fwd126_net"] = (1 - cost) * g_end - 1
    R["y126c50"] = np.where(np.isfinite(R["fwd126_net"]), (R["fwd126_net"] >= GOAL).astype(float), np.nan)
    print(f"[{time.time() - t0:.0f}s] target: {np.isfinite(R['fwd126_net']).sum():,} rows labelled · base rate {R['y126c50'].mean():.3f}", flush=True)

    # ---- rule ingredients
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill().reindex(columns=D["syms"]).to_numpy(); del S
    R["t2t"] = np.where(ok, np.nan_to_num(TW[i0c, cc]), np.nan); del TW
    R["fin_sector"] = R["symbol"].map(lambda s: float(ctx["fin"](s)))
    fo = ctx["fading_only"]
    R["fading_only"] = [float(fo(s, d)) for s, d in zip(R["symbol"], R["trade_date"])]
    print(f"[{time.time() - t0:.0f}s] t2t / sector / fading done", flush=True)
    Sc = ctx["scores"][["date", "industry", "heat_pct", "P_pct"]].rename(columns={"date": "trade_date"})
    R["_ind"] = R["symbol"].map(imap)
    R = R.merge(Sc.rename(columns={"industry": "_ind"}), on=["trade_date", "_ind"], how="left")
    R["g1_elig"] = ((R["heat_pct"] >= 0.70) & ~(R["P_pct"] < 0.30)).astype(float)
    R = R.rename(columns={"P_pct": "ind_policy_pct"}).drop(columns=["heat_pct"])
    F1 = v3_rule.frame_v31(X["F"], D)[["symbol", "trade_date", "close", "lo252", "ret252", "sma_50", "sma_200", "young"]]
    F1 = F1.rename(columns={"close": "_c", "lo252": "_lo", "ret252": "_r252", "sma_50": "_s50", "sma_200": "_s200", "young": "_young"})
    R = R.merge(F1, on=["symbol", "trade_date"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        R["trend_ok"] = ((R["_c"] / R["_lo"] - 1 >= 0.5) & (R["_r252"] >= 0.30) & (R["_c"] > R["_s200"]) & (R["_s50"] > R["_s200"])).astype(float)
    R["young_listing"] = R["_young"].fillna(False).astype(float)
    R = R.drop(columns=["_ind", "_c", "_lo", "_r252", "_s50", "_s200", "_young"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_parquet(OUT, index=False)
    man = json.loads(SRC.with_name(SRC.name + ".manifest.json").read_text())
    man.update(dataset=OUT.name, producer="src/agentic/build_rows_real_goal.py", experiment="EXP-2026-10-13-real-goal-model",
               copied_from=str(SRC.relative_to(ROOT)), new_columns={k: v for k, v in
               [ln.strip().split(None, 1) for ln in __doc__.split("adds:")[1].split("The live rows")[0].strip().splitlines() if ln.strip() and not ln.startswith("  " * 9)] if k in NEW + ["y126c50", "fwd126_net"]},
               target=dict(column="y126c50", horizon_sessions=HOLD, threshold=GOAL, costs="0.5% ADV >= Rs 5 cr, 1% >= Rs 1 cr, 2% below (round trip)"),
               updated=datetime.now().isoformat(timespec="seconds"))
    OUT.with_name(OUT.name + ".manifest.json").write_text(json.dumps(man, indent=1, default=str))
    cov = R.groupby(R["trade_date"].dt.year).agg(n=("symbol", "size"), labelled=("y126c50", lambda s: s.notna().mean()),
                                                 policy=("ind_policy_pct", lambda s: s.notna().mean()), t2t=("t2t", "mean"), g1=("g1_elig", "mean"),
                                                 trend=("trend_ok", "mean"), fading=("fading_only", "mean"), fin=("fin_sector", "mean"))
    print(cov.round(3).to_string()); print(f"rows {len(R):,} -> {OUT} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
