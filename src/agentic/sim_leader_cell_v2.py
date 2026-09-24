"""LEADER CELL v2 — the industry-leader philosophy rebuilt from scratch (EXP-2026-09-23b).

What changed vs sim_leader_sleeve.py (2026-09-08), each found on 2026-09-23:
  1. SURVIVORSHIP: v1 required a full 126td future (hi126/cl126 notna) BEFORE ranking, so
     names that later delisted/suspended were never selectable. v2 ranks every core-band
     name at t; a name that stops trading inside 126td exits at its last close.
  2. UNIVERSE: fund units removed by ISIN (security_master INF*), not by symbol regex.
  3. INDUSTRY MAP: v1 used modal smIndustry (covers ~40% of core names). v2 runs on
     --map nse (same labels), --map nse4 (NSE 4-level 'Industry' via screener.in for
     every equity — one taxonomy, ~all core names) and --map nse4_full (+ analyst analog
     labels for the ~14 liquid names no feed labels).
  4. HEAT: v1 = MEAN own ret60 of the group (one takeover stock made jewellery 'hot').
     v2 reports MEAN and MEDIAN heat.
  5. TAKEOVERS: names with an open-offer / detailed-public-statement filing in the prior
     180 days (announcements_historical, 2016+) can be excluded.
Arms (registered): A0 mean-heat top-3 · A1 = A0 + EXTENDED (ret252>0.5, production) ·
B = median heat + EXTENDED + no takeover targets (primary) · plus single-change arms and
A1 as originally computed (v1 survivorship) for the size of the bias.
Hold 126td close-to-close, 0.5% round trip, weekly cohorts 2016-06+, core band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
COST, HOLD = 0.5, 126
OO_RE = r"open offer|detailed public statement"

ap = argparse.ArgumentParser()
ap.add_argument("--map", choices=["nse", "nse4", "nse4_full"], default="nse")
args = ap.parse_args()

# ---------- industry map ----------
# nse       : NSE smIndustry (old 74-industry list; ~1,213 equities — what v1 saw)
# nse4      : NSE 4-level 'Industry' via screener.in for EVERY equity it covers (one taxonomy)
# nse4_full : nse4 + analyst analog labels (industry_analyst_labels.csv) for the residual
import html as _html
sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
fund = set(sm.loc[sm["is_fund_unit"], "symbol"])
if args.map == "nse":
    imap = sm.dropna(subset=["industry"]).set_index("symbol")["industry"]
else:
    sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry"])
    imap = sc.set_index("symbol")["industry"].map(_html.unescape)
    if args.map == "nse4_full":
        al = pd.read_csv(ROOT / "data/derived/industry_analyst_labels.csv")
        al = al[~al["symbol"].isin(imap.index)]
        add = al.set_index("symbol")["analog_symbol"].map(imap).dropna()
        imap = pd.concat([imap, add])
        print(f"analyst analog labels: +{len(add)} (of {len(al)})", flush=True)
imap = imap[~imap.index.isin(fund)]
print(f"industry labels: {len(imap):,} equities, {imap.nunique()} industries ({args.map})", flush=True)

# ---------- panel + forward paths (delisting-aware) ----------
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                     columns=["symbol", "trade_date", "high", "low", "close", "avg_traded_value_20d"])
import sys as _sys
_sys.path.insert(0, str(ROOT / "src/agentic"))
from generate_hybrid_basket import non_equity  # noqa: E402  (ISIN master + exclusion file + regex)
px = px[~px["symbol"].isin(fund) & ~non_equity(px["symbol"])]
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60, fill_method=None)
px["ret252"] = g["close"].pct_change(252, fill_method=None)
pos = g.cumcount().to_numpy(); n = g["close"].transform("size").to_numpy()
start = np.arange(len(px)) - pos
exit_i = start + np.minimum(pos + HOLD, n - 1)
panel_end = px["trade_date"].max()
last_dt = g["trade_date"].transform("max")
delisted = (last_dt < panel_end - pd.Timedelta(days=10)).to_numpy()
full = pos + HOLD <= n - 1
px["valid"] = full | delisted                                   # censor only still-trading names
cl = px["close"].to_numpy()
px["exit_close"] = cl[exit_i]
px["held"] = exit_i - np.arange(len(px))
px["hi_f"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(HOLD, min_periods=1).max()[::-1])
px["lo_f"] = g["low"].transform(lambda s: s.shift(-1)[::-1].rolling(HOLD, min_periods=1).min()[::-1])
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["ind"] = px["symbol"].map(imap)

days = sorted(px["trade_date"].unique())
wk = px[(px["adv"] >= 5) & (px["close"] > 50) & px["trade_date"].isin(set(days[::5]))
        & (px["trade_date"] >= "2016-06-01") & px["ind"].notna() & px["ret60"].notna()].copy()

# ---------- takeover targets (point-in-time, 180d) ----------
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "desc", "attchmntText", "sort_date"])
t = ah["desc"].fillna("") + " " + ah["attchmntText"].fillna("")
oo = ah[t.str.contains(OO_RE, case=False, regex=True)].copy()
oo["trade_date"] = pd.to_datetime(oo["sort_date"], errors="coerce").dt.normalize()
oo = oo.dropna(subset=["trade_date"]).sort_values("trade_date")[["symbol", "trade_date"]].drop_duplicates()
oo["oo_dt"] = oo["trade_date"]
wk = pd.merge_asof(wk.sort_values("trade_date"), oo, on="trade_date", by="symbol",
                   direction="backward", tolerance=pd.Timedelta(days=180))
wk["takeover"] = wk["oo_dt"].notna()
print(f"open-offer filings {len(oo):,} on {oo['symbol'].nunique()} symbols · weekly rows {len(wk):,}", flush=True)

# ---------- heat (mean + median), leaders ----------
def add_heat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nm"] = df.groupby(["trade_date", "ind"])["ret60"].transform("size")
    df = df[df["nm"] >= 5].copy()
    for how in ("mean", "median"):
        h = df.groupby(["trade_date", "ind"])["ret60"].agg(how).rename("h").reset_index()
        h[f"hot_{how}"] = h.groupby("trade_date")["h"].rank(pct=True) >= 0.9
        df = df.merge(h[["trade_date", "ind", f"hot_{how}"]], on=["trade_date", "ind"])
    df["rk"] = df.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
    df["ext"] = df["ret252"] > 0.50
    return df


# v1 method: drop every name without a full 126td future BEFORE heat/rank (survivorship)
v1 = add_heat(wk[wk["held"] >= HOLD])
wk = add_heat(wk)

# outcomes (valid rows only)
wk["net"] = (wk["exit_close"] / wk["close"] - 1) * 100 - COST
wk["x2"] = wk["hi_f"] / wk["close"] - 1 >= 1.0
wk["x50"] = wk["hi_f"] / wk["close"] - 1 >= 0.5
wk["trough"] = (wk["lo_f"] / wk["close"] - 1) * 100
wk["year"] = wk["trade_date"].dt.year
top3 = wk["rk"] <= 3

ARMS = {
    "A0 mean-heat top3": wk["hot_mean"] & top3,
    "A1 PRODUCTION (A0+EXT)": wk["hot_mean"] & top3 & wk["ext"],
    "  A1 + median heat only": wk["hot_median"] & top3 & wk["ext"],
    "  A1 + no-takeover only": wk["hot_mean"] & top3 & wk["ext"] & ~wk["takeover"],
    "B PRIMARY (median+EXT+no-TO)": wk["hot_median"] & top3 & wk["ext"] & ~wk["takeover"],
    "  B without EXT": wk["hot_median"] & top3 & ~wk["takeover"],
}


def maxdd_proxy(coh: pd.Series) -> float:        # identical to sim_leader_sleeve.maxdd_of_cohort_path
    step = (coh.sort_index() / 100.0) / 26.0
    nav = (1 + step.rolling(26, min_periods=1).mean()).cumprod()
    return float((nav / nav.cummax() - 1).min() * 100)


def outcomes(df):
    df["net"] = (df["exit_close"] / df["close"] - 1) * 100 - COST
    df["x2"] = df["hi_f"] / df["close"] - 1 >= 1.0
    df["x50"] = df["hi_f"] / df["close"] - 1 >= 0.5
    df["trough"] = (df["lo_f"] / df["close"] - 1) * 100
    df["year"] = df["trade_date"].dt.year
    return df


v1 = outcomes(v1)
ARMS_DF = {k: (wk, m) for k, m in ARMS.items()}
ARMS_DF["A1 as v1 (survivorship ref)"] = (v1, v1["hot_mean"] & (v1["rk"] <= 3) & v1["ext"])

out = {}
print(f"\n{'ARM':<32s}| era      |     n | tr/yr | mean/tr | med/tr | win% | P2x  | P50  | medTrough | worst coh | ~maxDD | yrs+ ", flush=True)
for name, (D, mask) in ARMS_DF.items():
    V = D[mask & D["valid"]]
    for era, em in (("disc<=22", V["year"] <= 2022), ("conf>=23", V["year"] >= 2023)):
        S = V[em]
        if len(S) < 30:
            print(f"{name:<32s}| {era} | {len(S):>5,} | too few", flush=True); continue
        coh = S.groupby("trade_date")["net"].mean()
        yrs = max((S["trade_date"].max() - S["trade_date"].min()).days / 365.25, 0.1)
        yp = S.groupby("year")["net"].mean()
        r = dict(n=len(S), tr_yr=len(S) / yrs, mean=S["net"].mean(), med=S["net"].median(), win=(S["net"] > 0).mean() * 100,
                 p2x=S["x2"].mean() * 100, p50=S["x50"].mean() * 100, trough=S["trough"].median(), worst=coh.min(),
                 dd=maxdd_proxy(coh), yrs=f"{(yp > 0).sum()}/{len(yp)}", delisted_exits=int((S["held"] < HOLD).sum()))
        out[f"{name.strip()}|{era}"] = r
        print(f"{name:<32s}| {era} | {r['n']:>5,} | {r['tr_yr']:>5.0f} | {r['mean']:>+6.2f}% | {r['med']:>+6.2f}% | {r['win']:>4.1f} | "
              f"{r['p2x']:>4.1f} | {r['p50']:>4.1f} | {r['trough']:>+8.1f}% | {r['worst']:>+8.1f}% | {r['dd']:>+6.1f}% | {r['yrs']:>5s}", flush=True)

Path(ROOT / f"logs/leader_sleeve/sim_v2_{args.map}.json").write_text(json.dumps(out, indent=1, default=float))
print("LEADER CELL V2 COMPLETE", flush=True)
