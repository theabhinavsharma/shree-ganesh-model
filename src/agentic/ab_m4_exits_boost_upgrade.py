"""M4 (valuation-shaped exits) + BOOST-UPGRADE (SUE-confirmed good print).
Registered 2026-09-05 / 2026-09-08. One panel load; z-band pool frame; C2 sims.

M4 variants on top-8-by-delivery baskets:
  C2 BASE   : vol-scaled SL, half@+5%, trail lock +2.5%, day-15 timeout
  C2 VALEXIT: pe_ind<1  -> half@+5%, trail lock +1.0% (room to run)
              pe_ind>=1 or unknown -> FULL exit at +5% (take the target)
  Bar: beat BASE mean weekly net in BOTH eras.

BOOST-UPGRADE on top-24 -> top-8 construction:
  BASE-BOOST : prefer names with price-only good print <=90d (rx>=5% on 2x vol)
  SUE-BOOST  : prefer only prints that ALSO have SUE top-tercile (EPS accel)
  Bar: SUE-BOOST beats BASE-BOOST top-8 touch in BOTH eras.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")

# ---------- TTM PE (same machinery as ab_valuation_3h) ----------
print("ttm/pe…", flush=True)
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
_o = pd.Timestamp("1900-01-01")
_c = lastq["con"] if "con" in lastq else pd.Series(_o, index=lastq.index)
_s = lastq["sa"] if "sa" in lastq else pd.Series(_o, index=lastq.index)
bmap = pd.Series(np.where(_c.fillna(_o) >= _s.fillna(_o), "con", "sa"), index=lastq.index)
q = q[q["basis"] == q["symbol"].map(bmap)].copy()
q["filing_dt"] = pd.to_datetime(q["filing_dt"]).dt.normalize()

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "open", "high", "low", "close", "return_1d",
             "rsi_14_daily", "return_20d", "volume_vs_20d", "delivery_pct",
             "avg_delivery_pct_20d", "avg_traded_value_20d", "price_adjustment_factor_to_present"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")

fac = px[["symbol", "trade_date", "price_adjustment_factor_to_present", "close",
          "return_1d", "volume_vs_20d"]].rename(columns={"trade_date": "filing_dt"})
q = pd.merge_asof(q.sort_values("filing_dt"), fac.sort_values("filing_dt"), on="filing_dt",
                  by="symbol", direction="forward", tolerance=pd.Timedelta(days=5))
q["eps_adj"] = q["eps_basic"] * q["price_adjustment_factor_to_present"].fillna(1.0)
q = q.sort_values(["symbol", "quarter_end"])
q["sue"] = q.groupby("symbol")["eps_adj"].diff(4) / q["close"]
q["cohort"] = q["filing_dt"].dt.to_period("Q")
q["sue_ter"] = q.groupby("cohort")["sue"].transform(
    lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
q["loud_up"] = (q["return_1d"] >= 0.05) & (q["volume_vs_20d"] >= 2)

rows = []
for sym, gg in q.dropna(subset=["eps_adj"]).groupby("symbol"):
    qe = gg["quarter_end"].values; ea = gg["eps_adj"].values
    fd = pd.to_datetime(gg["filing_dt"]).values
    for i in range(3, len(gg)):
        if (qe[i] - qe[i - 3]).astype("timedelta64[D]").astype(int) > 380:
            continue
        rows.append((sym, max(fd[i - 3:i + 1]), ea[i - 3:i + 1].sum()))
ttm = pd.DataFrame(rows, columns=["symbol", "known", "ttm"]).sort_values("known")

# ---------- weekly pool ----------
print("pool…", flush=True)
px["ret5"] = g["close"].pct_change(5)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
mu = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).mean())
sd = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).std())
rsi_z = (px["rsi_14_daily"] - mu) / sd
r20_z = px["return_20d"] / (px["dvol"] * np.sqrt(20))
r5_z = px["ret5"] / (px["dvol"] * np.sqrt(5))
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]
days = sorted(px["trade_date"].unique())
Z = (rsi_z.between(-1.0, 0.25) & r20_z.between(-1.0, 0.75) & r5_z.between(-1.0, 1.0)
     & (px["adv"] >= 5) & (px["close"] > 50) & (px["volume_vs_20d"] < 2)
     & px["trade_date"].isin(set(days[::5])) & (px["trade_date"] >= "2018-06-01"))
pool = px[Z][["symbol", "trade_date", "close", "dlv"]].dropna(subset=["dlv"]).copy()

# pe_ind at signal date
pool = pd.merge_asof(pool.sort_values("trade_date"),
                     ttm.rename(columns={"known": "trade_date"}).sort_values("trade_date"),
                     on="trade_date", by="symbol", direction="backward",
                     tolerance=pd.Timedelta(days=200))
pool["pe"] = np.where(pool["ttm"] > 0, pool["close"] / pool["ttm"], np.nan)
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
pool["ind"] = pool["symbol"].map(imap)
med = (pool.dropna(subset=["pe", "ind"]).groupby(["trade_date", "ind"])["pe"]
           .agg(["median", "size"]).reset_index().rename(columns={"median": "ind_pe"}))
pool = pool.merge(med[med["size"] >= 5][["trade_date", "ind", "ind_pe"]],
                  on=["trade_date", "ind"], how="left")
pool["cheap"] = (pool["pe"] / pool["ind_pe"] < 1).fillna(False)

# boost flags: latest print within 90d
pr = q[["symbol", "filing_dt", "sue_ter", "loud_up"]].rename(
    columns={"filing_dt": "trade_date"}).sort_values("trade_date")
pool = pd.merge_asof(pool.sort_values("trade_date"), pr, on="trade_date", by="symbol",
                     direction="backward", tolerance=pd.Timedelta(days=90))
pool["boost_price"] = pool["loud_up"].fillna(False)
pool["boost_sue"] = (pool["loud_up"].fillna(False)) & (pool["sue_ter"] == 2)
print(f"pool {len(pool):,} · cheap {pool['cheap'].mean()*100:.0f}% · "
      f"boost_price {pool['boost_price'].mean()*100:.1f}% · boost_sue {pool['boost_sue'].mean()*100:.1f}%", flush=True)

# ---------- C2 sim with exit variants ----------
sf = {s: gg.reset_index(drop=True) for s, gg in
      px[["symbol", "trade_date", "open", "high", "low", "close", "dvol"]].groupby("symbol")}


def c2(sym, d, mode="base", cheap=False):
    gg = sf[sym]
    sig = gg[gg["trade_date"] == d]; fut = gg[gg["trade_date"] > d].head(15)
    if len(sig) == 0 or len(fut) < 8 or pd.isna(fut.iloc[0]["open"]):
        return None
    ep = float(fut.iloc[0]["open"])
    dv = float(sig.iloc[0]["dvol"]) if pd.notna(sig.iloc[0]["dvol"]) else 0.02
    tgt = ep * 1.05
    sl = ep * (1 - np.clip(3 * dv, 0.03, 0.12))
    trail_lock = 1.025 if (mode == "base" or cheap is False) else 1.01
    if mode == "valexit" and cheap:
        trail_lock = 1.01
    half, trail, touched = False, None, False
    for i in range(len(fut)):
        r = fut.iloc[i]; lo, hi = r["low"], r["high"]
        if pd.isna(lo):
            continue
        if pd.notna(hi) and hi >= tgt:
            touched = True
        if not half:
            if lo <= sl:
                return dict(ret=(sl / ep - 1) * 100, touch=touched)
            if pd.notna(hi) and hi >= tgt:
                if mode == "valexit" and not cheap:
                    return dict(ret=5.0, touch=True)     # full exit at target
                half = True; trail = ep * trail_lock; continue
        else:
            if lo <= trail:
                return dict(ret=((0.05 + (trail / ep - 1)) / 2) * 100, touch=True)
    last = float(fut.iloc[-1]["close"])
    return dict(ret=((0.05 + (last / ep - 1)) / 2) * 100 if half else (last / ep - 1) * 100,
                touch=touched)


def run_baskets(select, mode):
    out = []
    for d, wkp in pool.groupby("trade_date"):
        bk = select(wkp)
        if bk is None or len(bk) < 8:
            continue
        rr = [c2(r["symbol"], d, mode, bool(r["cheap"])) for _, r in bk.iterrows()]
        rr = [x for x in rr if x]
        if len(rr) >= 4:
            out.append(dict(year=d.year, ret=np.mean([x["ret"] for x in rr]) - 0.30,
                            touch=np.mean([x["touch"] for x in rr])))
    return pd.DataFrame(out)


top8 = lambda w: w.sort_values("dlv", ascending=False).head(8)  # noqa: E731
boostp = lambda w: w.sort_values(["boost_price", "dlv"], ascending=[False, False]).head(8)  # noqa: E731
boosts = lambda w: w.sort_values(["boost_sue", "dlv"], ascending=[False, False]).head(8)  # noqa: E731

print("\n===== M4 exits + boost upgrade (top-8 C2) =====", flush=True)
print(f"{'VARIANT':<26s}| era      | weeks | touch% | mean wk net", flush=True)
for name, sel, mode in [("BASE C2", top8, "base"), ("VALEXIT C2", top8, "valexit"),
                        ("BOOST price-only", boostp, "base"), ("BOOST SUE-confirmed", boosts, "base")]:
    W = run_baskets(sel, mode)
    for era, m in [("disc<=22", W["year"] <= 2022), ("conf>=23", W["year"] >= 2023)]:
        s = W[m]
        print(f"{name:<26s}| {era} | {len(s):>5} | {s['touch'].mean()*100:5.1f}% | "
              f"{s['ret'].mean():+5.2f}%", flush=True)
print("\nM4 + BOOST-UPGRADE COMPLETE", flush=True)
