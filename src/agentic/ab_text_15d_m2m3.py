"""M2 (landmine veto) + M3 (tone-of-quiet) at 15d — universe-wide per the
2026-09-08 law (ALL stocks, ALL variables, ALL ranked). Registered 2026-09-07.

Features per (symbol, week), point-in-time from filing_scores.parquet:
  n_neg90 / n_pos90 : count of catdir -1 / +1 filings trailing 90d
  tone_neg30        : mean FinBERT negative prob over trailing-30d filings
  landmine          : n_neg90 >= 1 OR tone_neg30 in own top decile that week

M2 cells: universe + pool cells, landmine-score decile sweep, and the ship-shape
test — top-8 C2 baskets with landmine veto (reserves absorb) vs baseline.
Bar: veto basket +2pp touch AND -20% relative SL-hit rate, BOTH eras.
M3 cells: pool x tone terciles. Bar: >=3pp touch spread, BOTH eras.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")

print("filing scores…", flush=True)
fs = pd.read_parquet(ROOT / "data/derived/filing_scores.parquet")
fs["event_dt"] = pd.to_datetime(fs["event_dt"]).dt.normalize()
fs = fs.sort_values("event_dt")

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "open", "high", "low", "close", "return_1d",
             "rsi_14_daily", "return_20d", "volume_vs_20d", "delivery_pct",
             "avg_delivery_pct_20d", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["ret5"] = g["close"].pct_change(5)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
mu = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).mean())
sd = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).std())
px["rsi_z"] = (px["rsi_14_daily"] - mu) / sd
px["r20_z"] = px["return_20d"] / (px["dvol"] * np.sqrt(20))
px["r5_z"] = px["ret5"] / (px["dvol"] * np.sqrt(5))
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]
px["hi15"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(15, min_periods=8).max()[::-1])
px["lo15"] = g["low"].transform(lambda s: s.shift(-1)[::-1].rolling(15, min_periods=8).min()[::-1])

days = sorted(px["trade_date"].unique())
weekly = set(days[::5])
INV = ((px["adv"] >= 5) & (px["close"] > 50) & px["trade_date"].isin(weekly)
       & (px["trade_date"] >= "2016-06-01") & px["hi15"].notna())
uni = px[INV][["symbol", "trade_date", "close", "hi15", "lo15", "rsi_z", "r20_z",
               "r5_z", "volume_vs_20d", "dlv"]].copy()
uni["touch"] = uni["hi15"] / uni["close"] - 1 >= 0.05
uni["big"] = uni["hi15"] / uni["close"] - 1 >= 0.10
uni["deep"] = uni["lo15"] / uni["close"] - 1 <= -0.08
uni["year"] = uni["trade_date"].dt.year
uni["in_pool"] = (uni["rsi_z"].between(-1.0, 0.25) & uni["r20_z"].between(-1.0, 0.75)
                  & uni["r5_z"].between(-1.0, 1.0) & (uni["volume_vs_20d"] < 2))

print("trailing filing features…", flush=True)
neg = fs[fs["catdir"] == -1].groupby(["symbol", "event_dt"]).size().rename("n").reset_index()
pos = fs[fs["catdir"] == 1].groupby(["symbol", "event_dt"]).size().rename("n").reset_index()
tone = fs.dropna(subset=["tone_neg"]).groupby(["symbol", "event_dt"])["tone_neg"].mean().reset_index()


def trailing_count(events, grid, days_back):
    ev = {s: gg["event_dt"].values.astype("datetime64[ns]") for s, gg in events.groupby("symbol")}
    t64 = grid["trade_date"].values.astype("datetime64[ns]")
    out = np.zeros(len(grid))
    syms = grid["symbol"].values
    for i in range(len(grid)):
        a = ev.get(syms[i])
        if a is None:
            continue
        hi = np.searchsorted(a, t64[i], "right")
        lo = np.searchsorted(a, t64[i] - np.timedelta64(days_back, "D"), "right")
        out[i] = hi - lo
    return out


uni = uni.sort_values("trade_date")
uni["n_neg90"] = trailing_count(neg, uni, 90)
uni["n_pos90"] = trailing_count(pos, uni, 90)
# trailing-30d mean tone_neg via asof on per-symbol 30d rolling of filing tone
tone = tone.sort_values(["symbol", "event_dt"]).reset_index(drop=True)
tone["tone_neg30"] = (tone.groupby("symbol", group_keys=False)
                          .apply(lambda gg: gg.rolling("30D", on="event_dt")["tone_neg"].mean()))
tone = tone.sort_values("event_dt")
uni = pd.merge_asof(uni, tone[["symbol", "event_dt", "tone_neg30"]]
                    .rename(columns={"event_dt": "trade_date"}).sort_values("trade_date"),
                    on="trade_date", by="symbol", direction="backward",
                    tolerance=pd.Timedelta(days=30))
uni["landmine"] = (uni["n_neg90"] >= 1)
print(f"grid {len(uni):,} · landmine rate {uni['landmine'].mean()*100:.1f}% · "
      f"tone30 known {uni['tone_neg30'].notna().mean()*100:.0f}%", flush=True)

ERAS = [("disc<=22", uni["year"] <= 2022), ("conf>=23", uni["year"] >= 2023)]


def row(name, s, base_t, base_d):
    if len(s) < 80:
        return f"{name:<40s}| {len(s):>7,} | too few"
    return (f"{name:<40s}| {len(s):>7,} | {s['touch'].mean()*100:5.1f}% | {s['big'].mean()*100:5.1f}% | "
            f"{s['deep'].mean()*100:5.1f}% | {(s['touch'].mean()-base_t)*100:+4.1f}pp | "
            f"{(s['deep'].mean()/base_d-1)*100:+5.0f}%dp")


print(f"\n===== M2 UNIVERSE cells =====\n{'CELL':<40s}| {'n':>7s} | touch | big10 | deep8 | vsT | vsDeep")
for era, em in ERAS:
    U = uni[em]; bt, bd = U["touch"].mean(), U["deep"].mean()
    print(row(f"UNIVERSE [{era}]", U, bt, bd), flush=True)
    print(row(f"clean (no neg cat 90d) [{era}]", U[~U["landmine"]], bt, bd), flush=True)
    print(row(f"LANDMINE cat 90d [{era}]", U[U["landmine"]], bt, bd), flush=True)
    print(row(f"landmine x2+ [{era}]", U[U["n_neg90"] >= 2], bt, bd), flush=True)
    print(row(f"booster cat 90d [{era}]", U[U["n_pos90"] >= 1], bt, bd), flush=True)
    tq = U.dropna(subset=["tone_neg30"])
    if len(tq) > 1000:
        d9 = tq["tone_neg30"].quantile(0.9)
        print(row(f"tone_neg30 top decile [{era}]", tq[tq["tone_neg30"] >= d9], bt, bd), flush=True)

print(f"\n===== M3 pool x tone =====")
pool = uni[uni["in_pool"]]
for era, yr in [("disc<=22", lambda y: y <= 2022), ("conf>=23", lambda y: y >= 2023)]:
    P = pool[yr(pool["year"])]
    bt, bd = P["touch"].mean(), P["deep"].mean()
    tq = P.dropna(subset=["tone_neg30"])
    if len(tq) < 500:
        print(f"[{era}] tone known n={len(tq)} too thin"); continue
    t1, t2 = tq["tone_neg30"].quantile([1 / 3, 2 / 3])
    print(row(f"pool tone CALM third [{era}]", tq[tq["tone_neg30"] <= t1], bt, bd), flush=True)
    print(row(f"pool tone MID third [{era}]", tq[(tq["tone_neg30"] > t1) & (tq["tone_neg30"] <= t2)], bt, bd), flush=True)
    print(row(f"pool tone DARK third [{era}]", tq[tq["tone_neg30"] > t2], bt, bd), flush=True)
    print(row(f"pool no filings 30d [{era}]", P[P["tone_neg30"].isna()], bt, bd), flush=True)

print("\n===== M2 ship-shape: top-8 C2 with landmine veto =====", flush=True)
sf = {s: gg.reset_index(drop=True) for s, gg in
      px[["symbol", "trade_date", "open", "high", "low", "close", "dvol"]].groupby("symbol")}


def c2(sym, d):
    gg = sf[sym]
    sig = gg[gg["trade_date"] == d]; fut = gg[gg["trade_date"] > d].head(15)
    if len(sig) == 0 or len(fut) < 8 or pd.isna(fut.iloc[0]["open"]):
        return None
    ep = float(fut.iloc[0]["open"])
    dv = float(sig.iloc[0]["dvol"]) if pd.notna(sig.iloc[0]["dvol"]) else 0.02
    tgt, sl, half, trail = ep * 1.05, ep * (1 - np.clip(3 * dv, 0.03, 0.12)), False, None
    touched, slhit = False, False
    for i in range(len(fut)):
        r = fut.iloc[i]; lo, hi = r["low"], r["high"]
        if pd.isna(lo):
            continue
        if pd.notna(hi) and hi >= tgt:
            touched = True
        if not half:
            if lo <= sl:
                return dict(ret=(sl / ep - 1) * 100, touch=touched, sl=True)
            if pd.notna(hi) and hi >= tgt:
                half = True; trail = ep * 1.025; continue
        else:
            if lo <= trail:
                return dict(ret=((0.05 + (trail / ep - 1)) / 2) * 100, touch=True, sl=False)
    last = float(fut.iloc[-1]["close"])
    return dict(ret=((0.05 + (last / ep - 1)) / 2) * 100 if half else (last / ep - 1) * 100,
                touch=touched, sl=False)


res = {"BASE top8": [], "VETO landmine top8": []}
for d, wkp in pool.dropna(subset=["dlv"]).groupby("trade_date"):
    if len(wkp) < 8:
        continue
    base8 = wkp.sort_values("dlv", ascending=False).head(8)
    veto8 = wkp[~wkp["landmine"]].sort_values("dlv", ascending=False).head(8)
    for name, bk in [("BASE top8", base8), ("VETO landmine top8", veto8)]:
        out = [c2(r["symbol"], d) for _, r in bk.iterrows()]
        out = [o for o in out if o]
        if len(out) >= 4:
            res[name].append(dict(d=d, year=d.year, ret=np.mean([o["ret"] for o in out]) - 0.30,
                                  touch=np.mean([o["touch"] for o in out]),
                                  sl=np.mean([o["sl"] for o in out])))
print(f"{'VARIANT':<22s}| era      | weeks | touch% | SL-hit% | mean wk", flush=True)
for name, rr in res.items():
    W = pd.DataFrame(rr)
    for era, yr in [("disc<=22", W["year"] <= 2022), ("conf>=23", W["year"] >= 2023)]:
        s = W[yr]
        print(f"{name:<22s}| {era} | {len(s):>5} | {s['touch'].mean()*100:5.1f}% | "
              f"{s['sl'].mean()*100:6.1f}% | {s['ret'].mean():+5.2f}%", flush=True)
print("\nM2+M3 COMPLETE", flush=True)
