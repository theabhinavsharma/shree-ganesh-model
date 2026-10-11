"""Exploration (not a registered test; changes no rule): did the 2024-25 fall show up early, and where do those signs stand now?

Universe each day = stocks trading >= Rs 1 cr a day (20-day average) at a raw price > Rs 50 the day before (V3.1's bar).
Signals, all known on the day (no look-ahead):
  ew_vs_200   equal-weight index of the universe (daily returns clipped to -50%..+100%) vs its 200-day average
  ew_dd       equal-weight index fall from its running peak
  above200    share of universe stocks above their own 200-day average
  trend_pass  share of universe stocks passing V3.1's trend test (50% over 52w low, 1y return >= 30%, above 200d, 50d > 200d)
  pool        V3.1 candidate pool on the weekly list date (liquid, hot industry, trend or new listing; phase 0, from
              logs/leader_sleeve/v31_hits_by_year/v31_picks.csv)
  held13      average return so far of V3.1 picks bought in the previous 13 weekly lists (what the account shows)
  t2t         share of that week's V3.1 picks in the trade-for-trade (BE/BZ) series
Episodes: every fall of the equal-weight index of 20% or more from a peak. For each signal: value 6 / 3 / 0 months before the
peak, 3 months after, and today, plus the first date after each peak a simple line was crossed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

OUT = ROOT / "logs/leader_sleeve/regime_warning"


def main() -> None:
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d", "series"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(L["trade_date"]).unique()))
    C = rp.wide(L, "close", cal); RAW = rp.wide(L, "raw", cal); ADV = rp.wide(L, "adv", cal)
    S = L[["symbol", "trade_date", "series"]].copy(); S["trade_date"] = pd.to_datetime(S["trade_date"]); del L
    Cf = C.ffill(limit=5)
    uni = ((ADV >= 1.0) & (RAW > 50)).shift(1, fill_value=False).astype(bool)
    r = Cf.pct_change(fill_method=None).clip(-0.5, 1.0)
    ew_ret = r.where(uni).mean(axis=1).fillna(0.0)
    ew = (1 + ew_ret).cumprod()
    sig = pd.DataFrame(index=cal)
    sig["ew"] = ew
    sig["ew_vs_200"] = ew / ew.rolling(200).mean() - 1
    sig["ew_dd"] = ew / ew.cummax() - 1
    sma200, sma50 = Cf.rolling(200, min_periods=200).mean(), Cf.rolling(50, min_periods=50).mean()
    lo252 = Cf.rolling(252, min_periods=200).min()
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = (Cf / lo252 - 1 >= 0.5) & (Cf / Cf.shift(252) - 1 >= 0.30) & (Cf > sma200) & (sma50 > sma200)
    known = uni & sma200.notna()
    nk = known.sum(axis=1).astype(float).replace(0.0, np.nan)
    sig["above200"] = ((Cf > sma200) & known).sum(axis=1).astype(float) / nk
    sig["trend_pass"] = (trend & known).sum(axis=1).astype(float) / nk
    sig["n_universe"] = uni.sum(axis=1)

    pk = pd.read_csv(ROOT / "logs/leader_sleeve/v31_hits_by_year/v31_picks.csv", parse_dates=["trade_date"])
    pk = pk.merge(S, left_on=["symbol", "trade_date"], right_on=["symbol", "trade_date"], how="left")
    wk = pk.groupby("trade_date").agg(pool=("pool_n", "first"), t2t=("series", lambda x: (x != "EQ").mean()))
    pos = {d: i for i, d in enumerate(cal)}; col = {s: i for i, s in enumerate(Cf.columns)}; Cn = Cf.to_numpy()
    pk["e"] = pk["trade_date"].map(pos) + 1; pk["c"] = pk["symbol"].map(col)
    pk = pk[pk["c"].notna() & (pk["e"] < len(cal))]
    pk["entry"] = Cn[pk["e"].astype(int), pk["c"].astype(int)]  # close of the entry day (open not loaded; close is close enough here)
    wdates = list(wk.index)
    held = {}
    for i, d in enumerate(wdates):
        prev = pk[pk["trade_date"].isin(wdates[max(0, i - 13):i])]
        t = pos[d]
        v = Cn[t, prev["c"].astype(int)] / prev["entry"].to_numpy() - 1
        held[d] = float(np.nanmean(v)) if len(prev) else np.nan
    wk["held13"] = pd.Series(held)
    W = sig.reindex(wk.index).join(wk)
    W["pool_share"] = W["pool"] / W["n_universe"]
    W.index.name = "date"

    # episodes: falls of >= 20% in the equal-weight index
    eps, peak, in_dd = [], None, False
    dd = sig["ew_dd"]
    for d, v in dd.items():
        if v == 0:
            if in_dd and trough_v <= -0.20:
                eps.append((peak, trough, trough_v))
            peak, in_dd, trough_v, trough = d, False, 0.0, d
        else:
            in_dd = True
            if v < trough_v:
                trough_v, trough = v, d
    if in_dd and trough_v <= -0.20:
        eps.append((peak, trough, trough_v))
    eps = [e for e in eps if e[0] >= pd.Timestamp("2016-06-01")]

    def at(series: pd.Series, d) -> float:
        s = series.dropna(); s = s[s.index <= d]
        return float(s.iloc[-1]) if len(s) else np.nan
    cols = ["ew_vs_200", "above200", "trend_pass", "pool", "pool_share", "held13", "t2t"]
    src = lambda c: sig[c] if c in sig else W[c]  # noqa: E731
    rows = []
    print("=== falls of 20%+ in the equal-weight Rs 1 cr universe ===")
    for p, t, v in eps:
        print(f"peak {p.date()} -> low {t.date()}: {100 * v:.0f}%")
        for lab, d in (("6m before peak", p - pd.DateOffset(months=6)), ("3m before peak", p - pd.DateOffset(months=3)), ("at peak", p), ("3m after peak", p + pd.DateOffset(months=3))):
            rows.append(dict(episode=str(p.date()), when=lab, **{c: round(at(src(c), d), 3) for c in cols}))
        first = {}
        after = sig[sig.index > p]
        for lab, s, cond in (("index below its 200d avg", after["ew_vs_200"], lambda x: x < 0), ("under half of stocks above 200d", after["above200"], lambda x: x < 0.5),
                             ("index 10% off peak", after["ew_dd"], lambda x: x <= -0.10)):
            hit = s[cond(s)]
            first[lab] = f"{hit.index[0].date()} ({(hit.index[0] - p).days} days after peak)" if len(hit) else "never"
        rows[-1]["first_flips"] = first
        print("   first flips:", first)
    now = W.index[-1]
    rows.append(dict(episode="NOW", when=str(now.date()), **{c: round(at(src(c), now), 3) for c in cols}))
    T = pd.DataFrame(rows)
    print("\n=== signals around each peak, and now ===")
    print(T.drop(columns=["first_flips"], errors="ignore").to_string(index=False))

    print("\n=== 2024-25 and the last 12 months, monthly (last list date of each month) ===")
    M = W[(W.index >= "2024-01-01")].groupby(W[(W.index >= "2024-01-01")].index.to_period("M")).last()
    M["ew_dd"] = [at(sig["ew_dd"], d) for d in W[(W.index >= "2024-01-01")].groupby(W[(W.index >= "2024-01-01")].index.to_period("M")).apply(lambda x: x.index[-1])]
    print(M[["ew_vs_200", "ew_dd", "above200", "trend_pass", "pool", "pool_share", "held13", "t2t"]].round(3).to_string())

    pct = {c: round(float((src(c)[src(c).index >= "2016-06-01"].dropna() <= at(src(c), now)).mean()), 2) for c in cols}
    print("\nwhere today sits in its own history since 2016/2019 (share of weeks at or below today's value):", pct)
    OUT.mkdir(parents=True, exist_ok=True)
    W.to_csv(OUT / "weekly_signals.csv")
    (OUT / "weekly_signals.csv.manifest.json").write_text(json.dumps(dict(dataset="weekly_signals.csv", producer="src/agentic/explore_regime_warning.py", kind="exploration",
        definitions=__doc__, units=dict(ew_vs_200="fraction above(+)/below(-) the 200-day average", ew_dd="fraction below running peak", above200="share 0-1", trend_pass="share 0-1",
                                        pool="stocks", held13="fraction, average return so far", t2t="share 0-1"), episodes=[[str(a.date()), str(b.date()), round(c, 3)] for a, b, c in eps],
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUT / "README.md").write_text("# Early-warning signals vs the 2024-25 fall\n\nExploration. weekly_signals.csv = market breadth, equal-weight index trend, V3.1 pool size, "
                                   "how recent picks are doing, trade-for-trade share, per weekly list date. Definitions in the manifest.\n")


if __name__ == "__main__":
    main()
