"""Aggregate confluence picker — find stocks where MULTIPLE independent
signals align. Each layer is independently verified (90%+ OOS hit rate
where applicable). When 2+ layers agree, conviction multiplies.

Layers checked per stock:
  1. Multibagger model (100%/180d, score ≥ 0.86)
  2. Superstar holdings (held by ≥ 2 of Tickertape's top-20 investors)
  3. Screener FII/DII buying screen (curated)
  4. Daily 7d model (5%/7d at calibrated score ≥ 0.95 — strict bar)
  5. Sector tailwind (sector_5d_ret ≥ +1%)
  6. Fundamental quality (ROE ≥ 18 OR ROCE ≥ 20)
  7. Reasonable technical (40 ≤ RSI ≤ 70, ADV ≥ ₹50cr/day)

Output: data/derived/confluence_picks.parquet
        reports/confluence_picks.md

A stock with 4+ layers checked = highest conviction. Size 8% per name.
A stock with 3 layers = solid conviction. Size 5% per name.
A stock with 2 layers = monitor only.
A stock with 1 layer = noise.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
MULTI = ROOT / "data/derived/multibagger_today_predictions.parquet"
SUPERSTAR = ROOT / "data/derived/superstar_holdings.parquet"
FII_SCREEN = ROOT / "data/derived/screener_screens.parquet"
HC_PRED = ROOT / "data/derived/high_conviction_predictions.parquet"
FUND = ROOT / "data/derived/screener_fundamentals.parquet"
SECT_MEMBERS = ROOT / "tmp/from_scratch_7d_run/alt2/sector_index_members.parquet"

OUT = ROOT / "data/derived/confluence_picks.parquet"
OUT_REPORT = ROOT / "reports/confluence_picks.md"


def main() -> None:
    print("== build_confluence_picks ==")

    # Layer 0: liquid universe with technical context (today)
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "rsi_14_daily",
                                            "return_20d", "avg_traded_value_20d", "series",
                                            "sma_50"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    latest = px["trade_date"].max()
    today = px[(px["trade_date"] == latest) & (px["series"] == "EQ")].copy()
    today["adv_cr"] = today["avg_traded_value_20d"] / 1e7
    today["above_sma50"] = (today["close"] > today["sma_50"]).astype(int)
    print(f"  liquid universe today: {len(today)} stocks")

    # Layer 1: multibagger picks (100%/180d at 0.86)
    if MULTI.exists():
        m = pd.read_parquet(MULTI)
        m["multibagger_180"] = (m.get("score_100pct_180d", 0) >= 0.86).astype(int)
        m["multibagger_252"] = (m.get("score_100pct_252d", 0) >= 0.84).astype(int)
        m["multibagger_378"] = (m.get("score_100pct_378d", 0) >= 0.77).astype(int)
        m["multibagger_any"] = (m["multibagger_180"] | m["multibagger_252"] | m["multibagger_378"]).astype(int)
        m["multibagger_score"] = m[["score_100pct_180d", "score_100pct_252d", "score_100pct_378d"]].max(axis=1)
        today = today.merge(m[["symbol", "multibagger_180", "multibagger_252", "multibagger_378",
                                 "multibagger_any", "multibagger_score"]], on="symbol", how="left")
    today["multibagger_any"] = today.get("multibagger_any", 0).fillna(0).astype(int)

    # Layer 2: superstar holdings ≥ 2 investors
    if SUPERSTAR.exists():
        ss = pd.read_parquet(SUPERSTAR)
        ss["fetch_date"] = pd.to_datetime(ss["fetch_date"]).dt.date
        ss = ss[ss["fetch_date"] == ss["fetch_date"].max()]
        confluence_count = ss.groupby("symbol")["investor_tag"].nunique().rename("n_superstars")
        today = today.merge(confluence_count, on="symbol", how="left")
        today["n_superstars"] = today["n_superstars"].fillna(0).astype(int)
        # exclude noise (HDFCBANK/RELIANCE/TCS-style appears in all 17 = page header artifact)
        today["superstar_2plus"] = ((today["n_superstars"] >= 2) & (today["n_superstars"] <= 10)).astype(int)
    else:
        today["n_superstars"] = 0
        today["superstar_2plus"] = 0

    # Layer 3: Screener.in FII/DII buying screen
    if FII_SCREEN.exists():
        fii = pd.read_parquet(FII_SCREEN)
        fii_today = fii[fii["screen_tag"] == "FII_DII_BUYING"]
        fii_syms = set(fii_today["symbol"].astype(str))
        today["fii_dii_screen"] = today["symbol"].isin(fii_syms).astype(int)
    else:
        today["fii_dii_screen"] = 0

    # Layer 4: 7d model at 0.95 calibrated (strict)
    if HC_PRED.exists():
        hc = pd.read_parquet(HC_PRED)
        hc["hc_5pct_7d"] = (hc.get("score_5pct_7d_cal", 0) >= 0.95).astype(int)
        hc["hc_10pct_15d"] = (hc.get("score_10pct_15d_cal", 0) >= 0.95).astype(int)
        hc["hc_short_horizon"] = (hc["hc_5pct_7d"] | hc["hc_10pct_15d"]).astype(int)
        today = today.merge(hc[["symbol", "hc_short_horizon",
                                  "score_5pct_7d_cal", "score_10pct_15d_cal"]], on="symbol", how="left")
    today["hc_short_horizon"] = today.get("hc_short_horizon", 0).fillna(0).astype(int)

    # Layer 5: sector tailwind (sector 5d return ≥ +1%)
    if SECT_MEMBERS.exists():
        sm = pd.read_parquet(SECT_MEMBERS)
        SECT_PRIORITY = ["NIFTY IT","NIFTY BANK","NIFTY AUTO","NIFTY METAL","NIFTY PHARMA",
                         "NIFTY FMCG","NIFTY REALTY","NIFTY ENERGY","NIFTY MEDIA","NIFTY PSE",
                         "NIFTY PVT BANK","NIFTY FINANCIAL SERVICES","NIFTY CONSUMER DURABLES",
                         "NIFTY OIL & GAS","NIFTY INFRA"]
        sm["pri"] = sm["index_name"].map({n:i for i,n in enumerate(SECT_PRIORITY)}).fillna(99)
        sec_map = sm.sort_values("pri").drop_duplicates("symbol")[["symbol","index_name"]].rename(columns={"index_name":"sector"})

        # compute sector 5d return on the full panel
        full = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "return_1d"])
        full["trade_date"] = pd.to_datetime(full["trade_date"])
        full = full.merge(sec_map, on="symbol", how="left")
        full = full[full["sector"].isin(SECT_PRIORITY)]
        sd = full.groupby(["trade_date", "sector"])["return_1d"].median().reset_index()
        sd = sd.sort_values(["sector", "trade_date"])
        sd["s_5d"] = sd.groupby("sector")["return_1d"].transform(lambda s: s.rolling(5).sum())
        sd_latest = sd[sd["trade_date"] == latest][["sector", "s_5d"]]
        today = today.merge(sec_map, on="symbol", how="left")
        today = today.merge(sd_latest, on="sector", how="left")
        today["sector_tailwind"] = (today["s_5d"].fillna(0) >= 0.01).astype(int)
    else:
        today["sector"] = "OTHER"
        today["s_5d"] = 0
        today["sector_tailwind"] = 0

    # Layer 6: fundamental quality (ROE ≥ 18 OR ROCE ≥ 20)
    if FUND.exists():
        f = pd.read_parquet(FUND)
        f["fetch_date"] = pd.to_datetime(f["fetch_date"])
        f = f.sort_values("fetch_date").groupby("symbol").tail(1)
        keep = ["symbol"]
        if "roe" in f.columns: keep.append("roe")
        if "roce" in f.columns: keep.append("roce")
        if "pe" in f.columns: keep.append("pe")
        today = today.merge(f[keep], on="symbol", how="left")
        roe_pass = today.get("roe", pd.Series(0, index=today.index)).fillna(0) >= 18
        roce_pass = today.get("roce", pd.Series(0, index=today.index)).fillna(0) >= 20
        today["fundamental_quality"] = (roe_pass | roce_pass).astype(int)
    else:
        today["fundamental_quality"] = 0

    # Layer 7: reasonable technical (40 ≤ RSI ≤ 70, ADV ≥ ₹50cr/day)
    today["technical_clean"] = (
        (today["rsi_14_daily"].between(40, 70)) &
        (today["adv_cr"] >= 50.0)
    ).astype(int)

    # Confluence score
    layer_cols = ["multibagger_any", "superstar_2plus", "fii_dii_screen",
                   "hc_short_horizon", "sector_tailwind", "fundamental_quality",
                   "technical_clean"]
    today["confluence_count"] = today[layer_cols].sum(axis=1)

    # rank
    today_sorted = today.sort_values(["confluence_count", "multibagger_score", "adv_cr"],
                                       ascending=[False, False, False]).reset_index(drop=True)

    # save full
    out_cols = ["symbol", "close", "adv_cr", "rsi_14_daily", "return_20d",
                "n_superstars",
                "multibagger_180", "multibagger_252", "multibagger_378", "multibagger_any",
                "multibagger_score",
                "superstar_2plus", "fii_dii_screen", "hc_short_horizon",
                "sector", "s_5d", "sector_tailwind",
                "roe", "roce", "fundamental_quality", "technical_clean",
                "confluence_count"]
    out_cols = [c for c in out_cols if c in today_sorted.columns]
    today_out = today_sorted[out_cols].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    today_out.to_parquet(OUT, index=False)

    # ── REPORT ──
    md = [f"# Confluence picks — {latest:%Y-%m-%d}", "",
          "Cross-checks 7 independent signals. Stocks with multiple alignments = highest conviction.", "",
          "**The 7 layers checked per stock:**", "",
          "1. Multibagger model: 100% in 180/252/378d at score ≥ 0.86/0.84/0.77",
          "2. Superstar holdings: held by ≥ 2 of Tickertape top-20 investors",
          "3. Screener FII/DII buying screen (curated)",
          "4. 7-day model: score ≥ 0.95 on 5%/7d or 10%/15d (strict)",
          "5. Sector tailwind: sector 5d ≥ +1%",
          "6. Fundamental quality: ROE ≥ 18 OR ROCE ≥ 20",
          "7. Technical clean: 40 ≤ RSI ≤ 70 AND ADV ≥ ₹50cr/day",
          "",
          "## Distribution of confluence counts", ""]
    counts = today_sorted["confluence_count"].value_counts().sort_index(ascending=False)
    md.append("| Count | # stocks |")
    md.append("|---:|---:|")
    for k, v in counts.items():
        md.append(f"| {int(k)} | {int(v)} |")
    md.append("")

    # top names by confluence
    top = today_sorted[today_sorted["confluence_count"] >= 3].head(30)
    md.append(f"## Names with ≥ 3 layer alignments ({len(today_sorted[today_sorted['confluence_count'] >= 3])} total, top 30)")
    md.append("")
    if len(top):
        md.append("| Symbol | Sector | Close | ADV cr | RSI | 20d% | MB | SS | FII | HC | Sec+ | Fund | Tech | **Total** |")
        md.append("|---|---|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
        for _, r in top.iterrows():
            mb = "✓" if r.get("multibagger_any", 0) else ""
            ss = f"⭐{int(r['n_superstars'])}" if r.get("superstar_2plus", 0) else ""
            fii = "✓" if r.get("fii_dii_screen", 0) else ""
            hc = "✓" if r.get("hc_short_horizon", 0) else ""
            sec = "✓" if r.get("sector_tailwind", 0) else ""
            fund = "✓" if r.get("fundamental_quality", 0) else ""
            tech = "✓" if r.get("technical_clean", 0) else ""
            md.append(f"| **{r['symbol']}** | {r.get('sector','—')} | ₹{r['close']:.2f} | "
                      f"{r['adv_cr']:.0f} | {r['rsi_14_daily']:.0f} | "
                      f"{r['return_20d']*100:+.1f}% | {mb} | {ss} | {fii} | {hc} | {sec} | "
                      f"{fund} | {tech} | **{int(r['confluence_count'])}** |")
    else:
        md.append("_No stocks today have ≥3 layer alignments._")
    md.append("")
    md.append("## How to size by confluence count")
    md.append("")
    md.append("- **6-7 layers**: 8% sizing (highest conviction; rare)")
    md.append("- **4-5 layers**: 6% sizing")
    md.append("- **3 layers**: 4% sizing (monitor closely)")
    md.append("- **≤2 layers**: noise, skip")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))

    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT}")
    print(f"\nConfluence distribution:")
    print(counts.to_string())
    print(f"\n=== Names with ≥ 3 alignments ===")
    if len(top):
        cols_show = ["symbol", "close", "rsi_14_daily", "n_superstars",
                      "multibagger_any", "fii_dii_screen", "fundamental_quality",
                      "technical_clean", "confluence_count"]
        cols_show = [c for c in cols_show if c in top.columns]
        print(top[cols_show].to_string(index=False))
    else:
        print("  none")


if __name__ == "__main__":
    main()
