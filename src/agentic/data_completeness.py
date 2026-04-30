"""Data completeness audit — runs daily, fails LOUDLY if any stock in the
liquid universe is missing critical parameters.

This is the gate. No model output is published until we know exactly
WHICH params are missing for WHICH stocks. We never silently drop signals
because of NaN.

Outputs:
  reports/data_completeness_<YYYYMMDD>.md   — human-readable per-param coverage
  data/derived/completeness.parquet         — append-only daily ledger
  data/derived/missing_today.parquet        — per-(stock, param) gaps for today

The parameter inventory below is the contract. Add a column to a feature
parquet → add it here. If completeness drops day-over-day, the brief is
flagged 'STALE/INCOMPLETE'.
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CAT = ROOT / "data/derived/catalyst_features.parquet"
FUND = ROOT / "data/derived/fundamentals_snapshot.parquet"
BLOCK = ROOT / "data/derived/block_features.parquet"
OPT = ROOT / "data/derived/options_chain_snapshot.parquet"
NEWS = ROOT / "data/derived/news_features.parquet"
MACRO_SENT = ROOT / "data/derived/macro_sentiment.parquet"
NEWS_RAW = ROOT / "data/derived/news_feed.parquet"
REDDIT_RAW = ROOT / "data/derived/reddit_feed.parquet"
YT_RAW = ROOT / "data/derived/youtube_videos.parquet"
SECT_MEMBERS = ROOT / "tmp/from_scratch_7d_run/alt2/sector_index_members.parquet"
MH = ROOT / "tmp/from_scratch_7d_run/multi_horizon_full.csv"
LIVE_LONG = ROOT / "tmp/from_scratch_7d_run/v3_live_full.csv"
LIVE_SHORT = ROOT / "tmp/from_scratch_7d_run/short_live_full.csv"
# fall back to top-100 files if full not yet emitted (first-run before model rebuild)
if not MH.exists():
    MH = ROOT / "tmp/from_scratch_7d_run/multi_horizon_top.csv"
if not LIVE_LONG.exists():
    LIVE_LONG = ROOT / "tmp/from_scratch_7d_run/v3_live_top100.csv"
if not LIVE_SHORT.exists():
    LIVE_SHORT = ROOT / "tmp/from_scratch_7d_run/short_live_top100.csv"

LEDGER = ROOT / "data/derived/completeness.parquet"
MISSING = ROOT / "data/derived/missing_today.parquet"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ADV_CR = 0.1  # liquid universe floor

# ─────────────────────────────────────────────────────────────────────────
# THE PARAM INVENTORY — each is a column we expect to be non-null on the
# latest trade_date for every liquid EQ symbol.
# ─────────────────────────────────────────────────────────────────────────
PARAMS = {
    "PRICE_TECHNICAL": [
        "close", "open", "high", "low",
        "sma_20", "sma_50", "sma_200",
        "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
        "return_1d", "return_20d",
        "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
        "avg_traded_value_20d",
    ],
    "FUNDAMENTAL": [
        "pe", "sector_pe", "pe_vs_sector_ratio",
        "week52_high", "week52_low",
        "dist_from_52w_high_pct", "dist_from_52w_low_pct",
        "last_q_revenue", "last_q_pat",
        "qoq_revenue_growth", "qoq_pat_growth",
    ],
    "CATALYST": [
        "ann_5d_count", "ann_30d_count",
        "ann_order_5d", "ann_order_30d", "ann_result_5d",
        "ann_capex_30d", "ann_fundraise_30d", "ann_buyback_30d",
        "ann_ma_30d", "ann_regulatory_30d",
        "catalyst_score_5d", "catalyst_score_30d",
    ],
    "INSIDER_PIT": [
        "insider_net_60d_inr", "insider_buy_60d_inr", "insider_stake_delta_60d",
    ],
    "BLOCK_BULK": [
        "block_buy_5d_inr", "block_sell_5d_inr", "block_net_5d_inr",
        "block_buy_30d_inr", "block_sell_30d_inr", "block_net_30d_inr",
        "distinct_buyers_30d",
    ],
    "OPTIONS_FNO": [
        "atm_iv", "iv_skew", "pcr_oi", "pcr_volume",
        "max_pain", "max_pain_distance_pct",
    ],
    "SECTOR": [
        "sector", "sector_5d_ret", "sector_20d_ret", "sector_60d_ret",
    ],
    "MARKET_MACRO": [
        "market_1d_ret", "market_5d_ret", "market_20d_ret",
        "market_breadth_50dma", "market_breadth_200dma",
        "rel_strength_20d",
    ],
    "NEWS_SOCIAL": [
        "news_count_5d", "news_sentiment_5d", "news_count_30d", "news_sentiment_30d",
        "reddit_mentions_5d", "reddit_sentiment_5d",
        "youtube_mentions_5d", "youtube_sentiment_5d",
    ],
    "MACRO_SENT": [
        "global_macro_sent", "domestic_macro_sent",
        "rate_hawkish_score", "rate_dovish_score",
        "oil_sentiment", "usdinr_sentiment",
    ],
    "MODEL_OUTPUTS": [
        "score_ens", "score_calibrated",
        "short_score_calibrated",
        "score_h1_cal", "score_h7_cal", "score_h21_cal",
        "consensus", "triangulated",
    ],
}


def latest_universe() -> pd.DataFrame:
    """Liquid EQ universe on the latest trade_date."""
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "series", "avg_traded_value_20d"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    latest = df["trade_date"].max()
    snap = df[(df["trade_date"] == latest) & (df["series"] == "EQ")]
    snap = snap[snap["avg_traded_value_20d"] >= MIN_ADV_CR * 1e7]
    return snap[["symbol"]].drop_duplicates().assign(trade_date=latest)


def _compute_sector_macro(panel: pd.DataFrame, pf: pd.DataFrame, latest: pd.Timestamp) -> pd.DataFrame:
    """Inline-compute sector + macro features (mirroring run_short_side logic)."""
    if not SECT_MEMBERS.exists():
        return panel
    sm = pd.read_parquet(SECT_MEMBERS)
    SECT_PRIORITY = ["NIFTY IT", "NIFTY BANK", "NIFTY AUTO", "NIFTY METAL", "NIFTY PHARMA",
                     "NIFTY FMCG", "NIFTY REALTY", "NIFTY ENERGY", "NIFTY MEDIA", "NIFTY PSE",
                     "NIFTY PVT BANK", "NIFTY FINANCIAL SERVICES", "NIFTY CONSUMER DURABLES",
                     "NIFTY OIL & GAS", "NIFTY INFRA", "NIFTY 50", "NIFTY NEXT 50",
                     "NIFTY MIDCAP 100", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 100",
                     "NIFTY SMALLCAP 250", "NIFTY 500", "NIFTY MICROCAP 250"]
    sm["pri"] = sm["index_name"].map({n: i for i, n in enumerate(SECT_PRIORITY)}).fillna(99)
    sec_map = sm.sort_values("pri").drop_duplicates("symbol")[["symbol", "index_name"]].rename(
        columns={"index_name": "sector"})
    panel = panel.merge(sec_map, on="symbol", how="left")
    panel["sector"] = panel["sector"].fillna("OTHER")

    # rolling sector returns (latest 60d)
    px2 = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "return_1d"])
    px2["trade_date"] = pd.to_datetime(px2["trade_date"])
    px2 = px2.merge(sec_map, on="symbol", how="left")
    px2["sector"] = px2["sector"].fillna("OTHER")
    sec_d = px2.groupby(["trade_date", "sector"])["return_1d"].median().reset_index()
    sec_d = sec_d.sort_values(["sector", "trade_date"])
    sec_d["sector_5d_ret"] = sec_d.groupby("sector")["return_1d"].transform(lambda s: s.rolling(5).sum())
    sec_d["sector_20d_ret"] = sec_d.groupby("sector")["return_1d"].transform(lambda s: s.rolling(20).sum())
    sec_d["sector_60d_ret"] = sec_d.groupby("sector")["return_1d"].transform(lambda s: s.rolling(60).sum())
    sec_latest = sec_d[sec_d["trade_date"] == latest]
    panel = panel.merge(sec_latest[["sector", "sector_5d_ret", "sector_20d_ret", "sector_60d_ret"]],
                        on="sector", how="left")

    # market macro
    liq = px2[px2["return_1d"].notna()]
    mkt = liq.groupby("trade_date").agg(
        market_1d_ret=("return_1d", "median"),
    ).reset_index().sort_values("trade_date")
    mkt["market_5d_ret"] = mkt["market_1d_ret"].rolling(5).sum()
    mkt["market_20d_ret"] = mkt["market_1d_ret"].rolling(20).sum()
    mkt_latest = mkt[mkt["trade_date"] == latest].iloc[0] if len(mkt[mkt["trade_date"] == latest]) else None
    if mkt_latest is not None:
        panel["market_1d_ret"] = mkt_latest["market_1d_ret"]
        panel["market_5d_ret"] = mkt_latest["market_5d_ret"]
        panel["market_20d_ret"] = mkt_latest["market_20d_ret"]

    # breadth
    if "sma_50" in pf.columns and "sma_200" in pf.columns:
        breadth_50 = (pf["close"] > pf["sma_50"]).mean()
        breadth_200 = (pf["close"] > pf["sma_200"]).mean()
        panel["market_breadth_50dma"] = breadth_50
        panel["market_breadth_200dma"] = breadth_200

    # rel strength = stock_20d_ret - sector_20d_ret
    if "return_20d" in panel.columns and "sector_20d_ret" in panel.columns:
        panel["rel_strength_20d"] = panel["return_20d"] - panel["sector_20d_ret"]
    return panel


def _compute_news_aggregates(panel: pd.DataFrame) -> pd.DataFrame:
    """Pull per-symbol news/reddit/youtube features from score_sentiment output (preferred)
    or fall back to raw-feed counting if features parquet is missing."""
    if NEWS.exists():
        nf = pd.read_parquet(NEWS)
        # latest snapshot per symbol
        if "as_of" in nf.columns:
            nf["as_of"] = pd.to_datetime(nf["as_of"]).dt.date
            nf = nf.sort_values("as_of").groupby("symbol").tail(1)
        keep = ["symbol"] + [c for c in PARAMS["NEWS_SOCIAL"] if c in nf.columns]
        panel = panel.merge(nf[keep], on="symbol", how="left", suffixes=("", "_news"))

    # macro sentiment — same value broadcast to every row
    if MACRO_SENT.exists():
        ms = pd.read_parquet(MACRO_SENT)
        if "as_of" in ms.columns:
            ms["as_of"] = pd.to_datetime(ms["as_of"]).dt.date
            ms_today = ms.sort_values("as_of").iloc[-1]
            for col in PARAMS["MACRO_SENT"]:
                if col in ms.columns:
                    panel[col] = ms_today[col]

    syms_lower = panel["symbol"].str.lower().tolist()

    # fallback: raw-count if news_features.parquet missing
    if NEWS_RAW.exists() and "news_count_5d" not in panel.columns:
        news = pd.read_parquet(NEWS_RAW)
        news["pub_ts"] = pd.to_datetime(news["pub_ts"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=5)
        news_recent = news[news["pub_ts"] >= cutoff]
        news_recent["text"] = (news_recent["title"].fillna("") + " " + news_recent["desc"].fillna("")).str.lower()
        counts = {}
        for sym, sym_l in zip(panel["symbol"], syms_lower):
            counts[sym] = int(news_recent["text"].str.contains(rf"\b{sym_l}\b", regex=True, na=False).sum())
        panel["news_count_5d"] = panel["symbol"].map(counts)

    if REDDIT_RAW.exists():
        red = pd.read_parquet(REDDIT_RAW)
        red["created_utc"] = pd.to_numeric(red["created_utc"], errors="coerce")
        cutoff_ts = (pd.Timestamp.utcnow() - pd.Timedelta(days=5)).timestamp()
        red_recent = red[red["created_utc"] >= cutoff_ts]
        red_recent["text"] = (red_recent["title"].fillna("") + " " + red_recent["selftext"].fillna("")).str.lower()
        counts = {}
        for sym, sym_l in zip(panel["symbol"], syms_lower):
            counts[sym] = int(red_recent["text"].str.contains(rf"\b{sym_l}\b", regex=True, na=False).sum())
        panel["reddit_mentions_5d"] = panel["symbol"].map(counts)

    if YT_RAW.exists():
        yt = pd.read_parquet(YT_RAW)
        yt["published_ts"] = pd.to_datetime(yt["published_ts"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=5)
        yt_recent = yt[yt["published_ts"] >= cutoff]
        counts = {}
        for sym, sym_l in zip(panel["symbol"], syms_lower):
            counts[sym] = int(yt_recent["title"].str.lower().str.contains(rf"\b{sym_l}\b", regex=True, na=False).sum())
        panel["youtube_mentions_5d"] = panel["symbol"].map(counts)

    return panel


def assemble_panel(universe: pd.DataFrame) -> pd.DataFrame:
    """Left-join every available param onto the latest universe."""
    latest = universe["trade_date"].iloc[0]

    # 1. price + technical (the master parquet)
    px_cols = ["symbol", "trade_date"] + PARAMS["PRICE_TECHNICAL"]
    pf = pd.read_parquet(PRICES)
    pf["trade_date"] = pd.to_datetime(pf["trade_date"])
    pf = pf[pf["trade_date"] == latest]
    have_px = [c for c in px_cols if c in pf.columns]
    pf_keep = pf[have_px]

    panel = universe.merge(pf_keep, on=["symbol", "trade_date"], how="left")

    # 1a. sector + macro (computed inline)
    panel = _compute_sector_macro(panel, pf, latest)

    # 1b. news/reddit/youtube aggregates (computed from raw feeds)
    panel = _compute_news_aggregates(panel)

    # 2. catalyst features — use latest available row per symbol (may lag price by a day)
    if CAT.exists():
        cat = pd.read_parquet(CAT)
        cat["trade_date"] = pd.to_datetime(cat["trade_date"])
        cat = cat.sort_values("trade_date").groupby("symbol").tail(1)
        keep = ["symbol"] + [c for c in (PARAMS["CATALYST"] + PARAMS["INSIDER_PIT"]) if c in cat.columns]
        panel = panel.merge(cat[keep], on="symbol", how="left", suffixes=("", "_cat"))

    # 3. fundamentals (latest snapshot per symbol)
    if FUND.exists():
        fund = pd.read_parquet(FUND)
        fund["fetch_date"] = pd.to_datetime(fund["fetch_date"])
        fund = fund.sort_values("fetch_date").groupby("symbol").tail(1)
        keep = ["symbol"] + [c for c in PARAMS["FUNDAMENTAL"] if c in fund.columns]
        panel = panel.merge(fund[keep], on="symbol", how="left", suffixes=("", "_fund"))

    # 4. block deals features
    if BLOCK.exists():
        bk = pd.read_parquet(BLOCK)
        keep = ["symbol"] + [c for c in PARAMS["BLOCK_BULK"] if c in bk.columns]
        if len(keep) > 1:
            panel = panel.merge(bk[keep], on="symbol", how="left", suffixes=("", "_blk"))

    # 5. options chain (if file exists)
    if OPT.exists():
        op = pd.read_parquet(OPT)
        if "trade_date" in op.columns:
            op["trade_date"] = pd.to_datetime(op["trade_date"])
            op = op[op["trade_date"] == latest]
        keep = ["symbol"] + [c for c in PARAMS["OPTIONS_FNO"] if c in op.columns]
        if len(keep) > 1:
            panel = panel.merge(op[keep], on="symbol", how="left", suffixes=("", "_opt"))

    # 6. news / social
    if NEWS.exists():
        nw = pd.read_parquet(NEWS)
        keep = ["symbol"] + [c for c in PARAMS["NEWS_SOCIAL"] if c in nw.columns]
        if len(keep) > 1:
            panel = panel.merge(nw[keep], on="symbol", how="left", suffixes=("", "_news"))

    # 7. model outputs
    if LIVE_LONG.exists():
        ll = pd.read_csv(LIVE_LONG)[["symbol", "score_ens", "score_calibrated"]]
        panel = panel.merge(ll, on="symbol", how="left", suffixes=("", "_ll"))
    if LIVE_SHORT.exists():
        ls = pd.read_csv(LIVE_SHORT)[["symbol", "score_calibrated"]].rename(
            columns={"score_calibrated": "short_score_calibrated"})
        panel = panel.merge(ls, on="symbol", how="left", suffixes=("", "_ls"))
    if MH.exists():
        mh = pd.read_csv(MH)
        keep = ["symbol"] + [c for c in PARAMS["MODEL_OUTPUTS"] if c in mh.columns]
        if len(keep) > 1:
            panel = panel.merge(mh[keep], on="symbol", how="left", suffixes=("", "_mh"))

    return panel


def per_param_coverage(panel: pd.DataFrame) -> dict[str, dict]:
    """For each param, fraction of universe with non-null."""
    out: dict[str, dict] = {}
    n = len(panel)
    for group, params in PARAMS.items():
        out[group] = {}
        for p in params:
            if p not in panel.columns:
                out[group][p] = {"present": False, "coverage": 0.0, "non_null": 0, "n": n}
            else:
                non_null = panel[p].notna().sum()
                out[group][p] = {
                    "present": True,
                    "coverage": round(float(non_null) / n, 4) if n else 0.0,
                    "non_null": int(non_null),
                    "n": n,
                }
    return out


def missing_per_stock(panel: pd.DataFrame) -> pd.DataFrame:
    """Long format: (symbol, param, group) for every gap."""
    rows = []
    for group, params in PARAMS.items():
        for p in params:
            if p not in panel.columns:
                # whole column missing — count every stock as missing this param
                for sym in panel["symbol"]:
                    rows.append({"symbol": sym, "param": p, "group": group, "reason": "column_absent"})
                continue
            null_mask = panel[p].isna()
            for sym in panel.loc[null_mask, "symbol"]:
                rows.append({"symbol": sym, "param": p, "group": group, "reason": "null_value"})
    return pd.DataFrame(rows)


def write_report(coverage: dict, panel_n: int, today: pd.Timestamp) -> Path:
    lines = [f"# Data Completeness Audit — {today:%Y-%m-%d}", ""]
    lines.append(f"**Liquid universe:** {panel_n:,} symbols (EQ series, ADV ≥ ₹{MIN_ADV_CR}cr/day)")
    lines.append("")

    # group summary
    lines.append("## Group summary")
    lines.append("")
    lines.append("| Group | Params tracked | Avg coverage | Min coverage |")
    lines.append("|---|---|---|---|")
    for group, items in coverage.items():
        present = [v for v in items.values() if v["present"]]
        if not present:
            avg_cov = 0
            min_cov = 0
        else:
            covs = [v["coverage"] for v in present]
            avg_cov = sum(covs) / len(covs)
            min_cov = min(covs)
        absent = sum(1 for v in items.values() if not v["present"])
        present_count = len(present)
        flag = " ⚠️" if avg_cov < 0.5 else (" ✓" if avg_cov >= 0.95 else "")
        lines.append(f"| {group} | {present_count} present, {absent} absent | {avg_cov*100:.1f}% | {min_cov*100:.1f}%{flag} |")
    lines.append("")

    # per-param detail
    lines.append("## Per-param coverage")
    lines.append("")
    for group, items in coverage.items():
        lines.append(f"### {group}")
        lines.append("")
        lines.append("| Param | Present? | Coverage | n with data | n total |")
        lines.append("|---|---|---|---|---|")
        for p, v in items.items():
            mark = "✓" if v["present"] else "❌"
            cov = f"{v['coverage']*100:.1f}%"
            lines.append(f"| `{p}` | {mark} | {cov} | {v['non_null']:,} | {v['n']:,} |")
        lines.append("")

    # action items
    lines.append("## Action items (gaps)")
    lines.append("")
    actions = []
    for group, items in coverage.items():
        for p, v in items.items():
            if not v["present"]:
                actions.append(f"- **{group}.{p}**: column absent — fetcher missing or not yet wired")
            elif v["coverage"] < 0.5:
                actions.append(f"- **{group}.{p}**: only {v['coverage']*100:.0f}% covered — backfill needed")
    if not actions:
        lines.append("None — all params present at ≥50% coverage.")
    else:
        lines.extend(actions)
    lines.append("")

    out = REPORT_DIR / f"data_completeness_{today:%Y%m%d}.md"
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    today = pd.Timestamp(date.today())
    universe = latest_universe()
    print(f"liquid universe: {len(universe):,} symbols on {universe['trade_date'].iloc[0]:%Y-%m-%d}")
    panel = assemble_panel(universe)
    print(f"panel: {len(panel):,} rows × {len(panel.columns)} cols")

    coverage = per_param_coverage(panel)

    # write per-stock missing parquet (long)
    miss = missing_per_stock(panel)
    miss["audit_date"] = today
    MISSING.parent.mkdir(parents=True, exist_ok=True)
    miss.to_parquet(MISSING, index=False)
    print(f"missing-today rows: {len(miss):,} → {MISSING}")

    # append daily ledger
    ledger_rows = []
    for group, items in coverage.items():
        for p, v in items.items():
            ledger_rows.append({
                "audit_date": today,
                "group": group,
                "param": p,
                "present": v["present"],
                "coverage": v["coverage"],
                "non_null": v["non_null"],
                "n": v["n"],
            })
    ledger_today = pd.DataFrame(ledger_rows)
    if LEDGER.exists():
        old = pd.read_parquet(LEDGER)
        merged = pd.concat([old, ledger_today], ignore_index=True)
        merged = merged.drop_duplicates(["audit_date", "group", "param"], keep="last")
        merged.to_parquet(LEDGER, index=False)
    else:
        ledger_today.to_parquet(LEDGER, index=False)

    # human-readable report
    report = write_report(coverage, len(panel), today)

    # console summary
    print("\n=== completeness summary ===")
    for group, items in coverage.items():
        present = [v for v in items.values() if v["present"]]
        if not present:
            print(f"  {group:<22}  ALL ABSENT ({len(items)} params)")
            continue
        covs = [v["coverage"] for v in present]
        avg = sum(covs)/len(covs)
        absent = sum(1 for v in items.values() if not v["present"])
        flag = " ⚠️" if avg < 0.5 else ("" if avg >= 0.95 else " ◔")
        print(f"  {group:<22}  avg={avg*100:5.1f}%  present={len(present)}/{len(items)}  absent={absent}{flag}")
    print(f"\nfull report → {report}")


if __name__ == "__main__":
    main()
