"""LIVE SCREEN — winner theme > hot industry > right-to-win names (2026-09-08).

Applies the only both-era 2x survivor (industry contagion miner, 2026-09-03):
hot industry (group 60d return top decile) x top-3 leaders by own ret60
= 2.1-2.4x lift on P(2x/126td); conf-era LEADER & cheap-vs-industry = 12.4%.

Overlays reported per name (evidence, not filters):
  pe_ind (cheap<1 preferred per conf era) · TTM sign · latest SUE tercile ·
  landmine filings 90d · promoter delta (latest known SHP) · 200DMA position ·
  ADV band (core >=5cr/50 vs expanded >=1.5cr/25).
Output: ranked names + reports/theme_leaders_YYYYMMDD.md. Paper/venture-sleeve
candidates ONLY — historical P(2x/6mo) ~8-12%, median troughs -14 to -19%.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
TODAY = "20260908"

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "close", "avg_traded_value_20d",
             "price_adjustment_factor_to_present"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60)
px["dma200"] = g["close"].transform(lambda s: s.rolling(200, min_periods=120).mean())
last_dt = px["trade_date"].max()
snap = px[px["trade_date"] == last_dt].copy()
snap["adv"] = snap["avg_traded_value_20d"] / 1e7
snap = snap[(snap["adv"] >= 1.5) & (snap["close"] > 25)].dropna(subset=["ret60"])
print(f"as-of {last_dt.date()} · investable(expanded) {len(snap):,}", flush=True)

ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
snap["ind"] = snap["symbol"].map(imap)
snap = snap.dropna(subset=["ind"])
grp = (snap.groupby("ind").agg(grp_ret60=("ret60", "mean"), n=("ret60", "size"))
           .query("n >= 5").sort_values("grp_ret60", ascending=False))
cut = grp["grp_ret60"].quantile(0.9)
hot = grp[grp["grp_ret60"] >= cut]
print(f"industries {len(grp)} · HOT (top decile): {len(hot)}", flush=True)
for i, r in hot.iterrows():
    print(f"  🔥 {i:<38s} grp60 {r.grp_ret60*100:+6.1f}%  ({int(r.n)} names)", flush=True)

lead = snap[snap["ind"].isin(hot.index)].copy()
lead["rk"] = lead.groupby("ind")["ret60"].rank(ascending=False)
lead = lead[lead["rk"] <= 3].sort_values(["ind", "rk"])

# ---- overlays ----
print("overlays…", flush=True)
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
_o = pd.Timestamp("1900-01-01")
_c = lastq["con"] if "con" in lastq else pd.Series(_o, index=lastq.index)
_s = lastq["sa"] if "sa" in lastq else pd.Series(_o, index=lastq.index)
bmap = pd.Series(np.where(_c.fillna(_o) >= _s.fillna(_o), "con", "sa"), index=lastq.index)
q = q[q["basis"] == q["symbol"].map(bmap)]

def ttm_sue(sym):
    gg = q[q["symbol"] == sym].sort_values("quarter_end")
    if len(gg) < 4:
        return np.nan, np.nan
    t = gg["eps_basic"].tail(4).sum()
    sue = (gg["eps_basic"].iloc[-1] - gg["eps_basic"].iloc[-5]) if len(gg) >= 5 else np.nan
    return t, sue

vals = {s: ttm_sue(s) for s in lead["symbol"]}
lead["ttm"] = lead["symbol"].map(lambda s: vals[s][0])
lead["pe"] = np.where(lead["ttm"] > 0, lead["close"] / lead["ttm"], np.nan)
ipe = (snap.merge(q.groupby("symbol")["eps_basic"].apply(lambda s: s.tail(4).sum()).rename("t4"),
                  on="symbol", how="left"))
ipe["pe"] = np.where(ipe["t4"] > 0, ipe["close"] / ipe["t4"], np.nan)
ind_med = ipe.dropna(subset=["pe"]).groupby("ind")["pe"].median()
lead["pe_ind"] = lead.apply(lambda r: r["pe"] / ind_med.get(r["ind"], np.nan), axis=1)

fs = pd.read_parquet(ROOT / "data/derived/filing_scores.parquet")
fs["event_dt"] = pd.to_datetime(fs["event_dt"])
recent = fs[fs["event_dt"] >= last_dt - pd.Timedelta(days=90)]
nneg = recent[recent["catdir"] == -1].groupby("symbol").size()
npos = recent[recent["catdir"] == 1].groupby("symbol").size()
lead["neg90"] = lead["symbol"].map(nneg).fillna(0).astype(int)
lead["pos90"] = lead["symbol"].map(npos).fillna(0).astype(int)

try:
    shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
    shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
    shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
    shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
    shp["d"] = shp.groupby("symbol")["promoter_pct"].diff()
    pd_map = shp.groupby("symbol")["d"].last()
    lead["prom_d"] = lead["symbol"].map(pd_map)
except Exception:
    lead["prom_d"] = np.nan

lead["above200"] = lead["close"] > lead["dma200"]
lead["core_band"] = (lead["adv"] >= 5) & (lead["close"] > 50)
lead["grp60"] = lead["ind"].map(grp["grp_ret60"])

# right-to-win score: leader rank + conf-era-validated cheap + clean overlays
lead["rtw"] = ((4 - lead["rk"])                       # leadership strength
               + (lead["pe_ind"] < 1).fillna(False) * 2   # conf-era 12.4% cell
               + (lead["ttm"] > 0).fillna(False) * 1
               + lead["above200"] * 1
               + (lead["prom_d"] > 0).fillna(False) * 1
               - (lead["neg90"] > 0) * 1)
lead = lead.sort_values(["rtw", "grp60"], ascending=False)

cols_hdr = f"{'SYM':<14}{'INDUSTRY':<30}{'grp60':>7} {'own60':>7} {'PE':>7} {'peInd':>6} {'TTM':>5} {'promΔ':>6} {'neg':>4}{'pos':>4} {'200d':>5} {'band':>5} {'RTW':>4}"
print("\n" + cols_hdr, flush=True)
out_lines = [cols_hdr]
for _, r in lead.iterrows():
    line = (f"{r['symbol']:<14}{r['ind'][:29]:<30}{r['grp60']*100:>+6.1f}% {r['ret60']*100:>+6.1f}% "
            f"{r['pe'] if pd.notna(r['pe']) else float('nan'):>7.1f} "
            f"{r['pe_ind'] if pd.notna(r['pe_ind']) else float('nan'):>6.2f} "
            f"{'PROF' if r['ttm'] and r['ttm'] > 0 else 'LOSS' if pd.notna(r['ttm']) else '?':>5} "
            f"{r['prom_d'] if pd.notna(r['prom_d']) else float('nan'):>+6.2f} "
            f"{r['neg90']:>4}{r['pos90']:>4} {'✓' if r['above200'] else '✗':>5} "
            f"{'core' if r['core_band'] else 'exp':>5} {r['rtw']:>4.0f}")
    print(line, flush=True)
    out_lines.append(line)

rep = ROOT / f"reports/theme_leaders_{TODAY}.md"
rep.write_text(f"# Theme > Industry > Right-to-Win leaders — {last_dt.date()}\n\n"
               "Cell evidence: hot-industry top-3 leaders = 2.1-2.4x P(2x/126td) both eras; "
               "conf-era LEADER & pe_ind<1 = 12.4% P(2x), +18.4% mean 6-mo. Venture/paper "
               "sizing only; median troughs -14 to -19%.\n\n```\n" + "\n".join(out_lines) + "\n```\n")
print(f"\nwrote {rep}", flush=True)
print("THEME LEADERS SCREEN COMPLETE", flush=True)
