"""EXP-2026-09-24-2x-qual-at-15d — do the 2x model's qualitative indicators help at 15D/+5%?

Flags (from screen_theme_leaders.py overlays; definitions registered in logs/experiments.jsonl
before this script existed): IND_HOT, HOT_LEADER, EXTENDED, LOSS, CHEAP_IND, PROM_UP, ABOVE_200.
SUE / landmine filings / FinBERT tone are NOT retested — each already failed its own 15d A/B.

A (descriptive) universe-wide: flag vs rest — touch, C2 net, lift, recall — both eras.
B (decides) z-band pool: BASE top-8 by delivery ratio vs TILT_<flag> (flag first, then
  delivery ratio). PASS iff touch +4pp in BOTH eras, C2 net/week >= BASE in both eras, and
  pooled paired weekly touch-diff p < 0.05/7.

Writes reports/ab_2x_qual_15d.md and appends the verdict to logs/experiments.jsonl.
"""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
EXP_ID = "EXP-2026-09-24-2x-qual-at-15d"
FLAGS = ["IND_HOT", "HOT_LEADER", "EXTENDED", "LOSS", "CHEAP_IND", "PROM_UP", "ABOVE_200"]
H, COST, BAR_PP, ALPHA = 15, 0.30, 4.0, 0.05 / 7

# ---------------- TTM EPS (point-in-time, CA-adjusted) — same construction as ab_valuation_3h
print("eps…", flush=True)
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
_o = pd.Timestamp("1900-01-01")
_c = lastq["con"] if "con" in lastq else pd.Series(_o, index=lastq.index)
_s = lastq["sa"] if "sa" in lastq else pd.Series(_o, index=lastq.index)
bmap = pd.Series(np.where(_c.fillna(_o) >= _s.fillna(_o), "con", "sa"), index=lastq.index)
q = q[q["basis"] == q["symbol"].map(bmap)].copy()

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "open", "high", "low", "close", "return_1d", "rsi_14_daily",
             "return_20d", "volume_vs_20d", "delivery_pct", "avg_delivery_pct_20d",
             "avg_traded_value_20d", "sma_200", "price_adjustment_factor_to_present"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

fac = px[["symbol", "trade_date", "price_adjustment_factor_to_present"]].rename(
    columns={"trade_date": "filing_dt", "price_adjustment_factor_to_present": "fac"})
q["filing_dt"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
q = pd.merge_asof(q.sort_values("filing_dt"), fac.sort_values("filing_dt"), on="filing_dt", by="symbol",
                  direction="backward", tolerance=pd.Timedelta(days=30))
q["eps_adj"] = q["eps_basic"] * q["fac"].fillna(1.0)
rows = []
for sym, gq in q.sort_values("quarter_end").groupby("symbol"):
    qe = gq["quarter_end"].values; ea = gq["eps_adj"].values; fd = pd.to_datetime(gq["filing_dt"]).values
    for i in range(3, len(gq)):
        if (qe[i] - qe[i - 3]).astype("timedelta64[D]").astype(int) <= 380:
            rows.append((sym, max(fd[i - 3:i + 1]), ea[i - 3:i + 1].sum()))
ttm = pd.DataFrame(rows, columns=["symbol", "known", "ttm"]).sort_values("known")

# ---------------- per-symbol features + forward path
print("features…", flush=True)
g = px.groupby("symbol")
px["ret5"] = g["close"].pct_change(5)
px["ret60"] = g["close"].pct_change(60)
px["ret252"] = g["close"].pct_change(252)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
mu = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).mean())
sd = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).std())
px["rsi_z"] = (px["rsi_14_daily"] - mu) / sd
px["r20_z"] = px["return_20d"] / (px["dvol"] * np.sqrt(20))
px["r5_z"] = px["ret5"] / (px["dvol"] * np.sqrt(5))
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]
for k in range(1, H + 1):
    for c in ("open", "high", "low", "close"):
        if c == "open" and k > 1:
            continue
        px[f"{c[0]}{k}"] = g[c].shift(-k).astype("float32")

days = sorted(px["trade_date"].unique())
weekly = set(days[::5])
wk = px[px["trade_date"].isin(weekly) & (px["adv"] >= 5) & (px["close"] > 50)
        & (px["trade_date"] >= "2018-06-01")].copy()
del px
wk = wk[wk[f"c{H}"].notna() | (wk[[f"c{k}" for k in range(1, H + 1)]].notna().sum(axis=1) >= 8)]

# ---------------- flags
print("flags…", flush=True)
wk = pd.merge_asof(wk.sort_values("trade_date"), ttm.rename(columns={"known": "trade_date"}),
                   on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
shp["prom_delta"] = shp.groupby("symbol")["promoter_pct"].diff()
shp["trade_date"] = shp["qe"] + pd.Timedelta(days=45)
wk = pd.merge_asof(wk.sort_values("trade_date"),
                   shp.dropna(subset=["prom_delta"]).sort_values("trade_date")[["symbol", "trade_date", "prom_delta"]],
                   on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
wk["ind"] = wk["symbol"].map(imap)
grp = (wk.dropna(subset=["ind", "ret60"]).groupby(["trade_date", "ind"])["ret60"]
         .agg(grp60="mean", n="size").reset_index().query("n >= 5"))
grp["hot"] = grp["grp60"] >= grp.groupby("trade_date")["grp60"].transform(lambda s: s.quantile(0.9))
wk = wk.merge(grp[["trade_date", "ind", "hot"]], on=["trade_date", "ind"], how="left")
wk["own_rk"] = wk.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False)
wk["pe"] = np.where(wk["ttm"] > 0, wk["close"] / wk["ttm"], np.nan)
med = (wk.dropna(subset=["pe", "ind"]).groupby(["trade_date", "ind"])["pe"]
         .agg(ind_pe="median", ind_n="size").reset_index())
wk = wk.merge(med[med["ind_n"] >= 5], on=["trade_date", "ind"], how="left")

wk["IND_HOT"] = wk["hot"].fillna(False).astype(bool)
wk["HOT_LEADER"] = wk["IND_HOT"] & (wk["own_rk"] <= 3)
wk["EXTENDED"] = wk["ret252"] > 0.50
wk["LOSS"] = wk["ttm"] <= 0
wk["CHEAP_IND"] = (wk["pe"] / wk["ind_pe"]) < 1
wk["PROM_UP"] = wk["prom_delta"] > 0
wk["ABOVE_200"] = wk["close"] > wk["sma_200"]
# a flag is only compared among names where its input is known (coverage gap != "no")
for f, k in {"LOSS": wk["ttm"].notna(), "CHEAP_IND": wk["ind_pe"].notna() & wk["pe"].notna(),
             "PROM_UP": wk["prom_delta"].notna(), "ABOVE_200": wk["sma_200"].notna(),
             "EXTENDED": wk["ret252"].notna(), "IND_HOT": wk["ind"].notna(), "HOT_LEADER": wk["ind"].notna()}.items():
    wk[f"known_{f}"] = k

# ---------------- outcomes: touch (vs signal close) + vectorised C2 from the next open
print("outcomes…", flush=True)
hi = np.column_stack([wk[f"h{k}"] for k in range(1, H + 1)]).astype("float64")
lo = np.column_stack([wk[f"l{k}"] for k in range(1, H + 1)]).astype("float64")
cl = np.column_stack([wk[f"c{k}"] for k in range(1, H + 1)]).astype("float64")
close0 = wk["close"].values
wk["touch"] = np.nanmax(hi, axis=1) / close0 - 1 >= 0.05
ep = wk["o1"].values.astype("float64")
dv = np.nan_to_num(wk["dvol"].values, nan=0.02)
sl = ep * (1 - np.clip(3 * dv, 0.03, 0.12)); tgt = ep * 1.05; trail = ep * 1.025
done = np.zeros(len(wk), bool); half = np.zeros(len(wk), bool); ret = np.full(len(wk), np.nan)
for k in range(H):
    l, h = lo[:, k], hi[:, k]; ok = ~done & ~np.isnan(l); was_half = half.copy()
    m_sl = ok & ~was_half & (l <= sl)
    ret[m_sl] = sl[m_sl] / ep[m_sl] - 1; done |= m_sl
    half |= ok & ~was_half & ~m_sl & (h >= tgt)
    m_tr = ok & was_half & (l <= trail)
    ret[m_tr] = (0.05 + 0.025) / 2; done |= m_tr
last = pd.DataFrame(cl).ffill(axis=1).iloc[:, -1].values
rest = ~done
ret[rest] = np.where(half[rest], (0.05 + last[rest] / ep[rest] - 1) / 2, last[rest] / ep[rest] - 1)
wk["c2"] = ret * 100 - COST
wk = wk[~np.isnan(ep) & ~np.isnan(wk["c2"].values)].copy()
wk["era"] = np.where(wk["trade_date"].dt.year <= 2022, "disc", "conf")
print(f"  grid {len(wk):,} stock-weeks · {wk['trade_date'].nunique()} weeks · "
      f"{wk['trade_date'].min().date()}..{wk['trade_date'].max().date()}", flush=True)


# ---------------- A. universe-wide
def uni(f):
    out = {}
    for era in ("disc", "conf"):
        e = wk[(wk["era"] == era) & wk[f"known_{f}"]]
        a, b = e[e[f]], e[~e[f]]
        tt = e["touch"].sum()
        out[era] = {"n_flag": int(len(a)), "n_rest": int(len(b)),
                    "touch_flag": round(a["touch"].mean() * 100, 1), "touch_rest": round(b["touch"].mean() * 100, 1),
                    "lift_pp": round((a["touch"].mean() - b["touch"].mean()) * 100, 1),
                    "c2_flag": round(a["c2"].mean(), 2), "c2_rest": round(b["c2"].mean(), 2),
                    "recall": round(a["touch"].sum() / tt * 100, 1) if tt else None,
                    "share": round(len(a) / len(e) * 100, 1)}
    return out


A = {f: uni(f) for f in FLAGS}

# ---------------- B. z-band pool overlay (decides)
Z = (wk["rsi_z"].between(-1.0, 0.25) & wk["r20_z"].between(-1.0, 0.75)
     & wk["r5_z"].between(-1.0, 1.0) & (wk["volume_vs_20d"] < 2))
pool = wk[Z].dropna(subset=["dlv"])
W = []
for d, p in pool.groupby("trade_date"):
    if len(p) < 8:
        continue
    base = p.sort_values("dlv", ascending=False).head(8)
    row = {"d": d, "era": p["era"].iloc[0], "BASE_touch": base["touch"].mean(), "BASE_c2": base["c2"].mean()}
    for f in FLAGS:
        t = p.assign(_f=p[f].fillna(False).astype(int)).sort_values(["_f", "dlv"], ascending=False).head(8)
        row[f"{f}_touch"] = t["touch"].mean(); row[f"{f}_c2"] = t["c2"].mean(); row[f"{f}_nflag"] = int(t["_f"].sum())
    W.append(row)
W = pd.DataFrame(W)


def paired_p(x):
    x = x.dropna(); n = len(x)
    if n < 10 or x.std() == 0:
        return None
    t = x.mean() / (x.std(ddof=1) / math.sqrt(n))
    return math.erfc(abs(t) / math.sqrt(2))           # two-sided, normal approx (n in the hundreds)


B, verdicts = {}, {}
for f in FLAGS:
    r = {}
    for era in ("disc", "conf"):
        e = W[W["era"] == era]
        r[era] = {"weeks": int(len(e)), "base_touch": round(e["BASE_touch"].mean() * 100, 1),
                  "tilt_touch": round(e[f"{f}_touch"].mean() * 100, 1),
                  "gap_pp": round((e[f"{f}_touch"] - e["BASE_touch"]).mean() * 100, 1),
                  "base_c2": round(e["BASE_c2"].mean(), 2), "tilt_c2": round(e[f"{f}_c2"].mean(), 2),
                  "avg_flagged_in_8": round(e[f"{f}_nflag"].mean(), 1)}
    p = paired_p(W[f"{f}_touch"] - W["BASE_touch"])
    ok = all(r[e]["gap_pp"] >= BAR_PP and r[e]["tilt_c2"] >= r[e]["base_c2"] for e in ("disc", "conf")) and p is not None and p < ALPHA
    r["p_pooled"] = None if p is None else float(f"{p:.2g}")
    B[f] = r; verdicts[f] = "PASS" if ok else "FAIL"

# ---------------- report
rng = f"{wk['trade_date'].min().date()}..{wk['trade_date'].max().date()}"
md = [f"# 2x-model qual indicators at 15D/+5% — {EXP_ID}", "",
      f"_generated {datetime.now():%Y-%m-%d %H:%M} · weekly grid {rng}, investable ADV>=5cr & close>50, "
      f"{len(wk):,} stock-weeks · touch = 15d high >= +5% over signal close · C2 net from next open, 0.30% RT_", "",
      f"**Verdict:** " + ", ".join(f"{f} {v}" for f, v in verdicts.items()), "",
      f"Bar (test B): tilt beats BASE top-8 by >= {BAR_PP}pp weekly touch in BOTH eras, C2 net/week not lower, pooled paired p < {ALPHA:.4f}.", "",
      "## B — z-band pool overlay (decides)", "",
      "| flag | era | weeks | BASE touch % | TILT touch % | gap pp | BASE C2/wk % | TILT C2/wk % | flagged in top-8 | p pooled |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(FLAGS, key=lambda f: -min(B[f]["disc"]["gap_pp"], B[f]["conf"]["gap_pp"])):
    for era in ("disc", "conf"):
        r = B[f][era]
        md.append(f"| {f} | {era} | {r['weeks']} | {r['base_touch']} | {r['tilt_touch']} | {r['gap_pp']:+} | "
                  f"{r['base_c2']} | {r['tilt_c2']} | {r['avg_flagged_in_8']} | {B[f]['p_pooled'] if era == 'disc' else ''} |")
md += ["", "## A — universe-wide (descriptive): flag vs rest, ranked by the weaker era's lift", "",
       "| flag | era | n flag | share % | touch flag % | touch rest % | lift pp | C2 flag % | C2 rest % | recall % |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(FLAGS, key=lambda f: -min(A[f]["disc"]["lift_pp"], A[f]["conf"]["lift_pp"])):
    for era in ("disc", "conf"):
        r = A[f][era]
        md.append(f"| {f} | {era} | {r['n_flag']:,} | {r['share']} | {r['touch_flag']} | {r['touch_rest']} | {r['lift_pp']:+} | "
                  f"{r['c2_flag']} | {r['c2_rest']} | {r['recall']} |")
md += ["", "## Caveats", "",
       "- Industry = modal smIndustry over all announcements (not point-in-time), as in the 2x model.",
       "- TTM needs 4 filed quarters (pnl_quarterly); PROM_UP needs two SHP quarters, known at quarter_end + 45d.",
       "- A raw touch counts a +5% high even if C2 would have stopped out first; C2 net is the tradeable number.",
       "- SUE, landmine filings and FinBERT tone were not retested: each failed its own registered 15d A/B (4b66c60)."]
(ROOT / "reports/ab_2x_qual_15d.md").write_text("\n".join(md) + "\n")
with open(ROOT / "logs/experiments.jsonl", "a") as fh:
    fh.write(json.dumps({"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "id": EXP_ID,
                         "status": "PASS" if "PASS" in verdicts.values() else "FAIL",
                         "verdicts": verdicts, "test_B": B, "test_A": A, "grid": rng, "n": int(len(wk))}) + "\n")
print("\n".join(md))
