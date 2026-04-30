"""Filter cascade — show exactly what gets dropped at each gate, with counts.

Discipline layer: the difference between 2,137 liquid stocks and a 5-name
actionable list is many filters. Make every drop visible so we can audit
why a name we expected to see isn't there.

Cascade order:
  1. Liquid universe (ADV >= 0.1cr, EQ series)
  2. Has minimum feature completeness (>= 80% non-null on PRICE_TECHNICAL)
  3. Macro overlay — if RISK-OFF, only keep score >= 0.75 names
  4. Patience filter — score_calibrated >= 0.65 (configurable)
  5. RSI sanity — drop RSI > 90 (climax) and RSI < 20 (knife-falling)
  6. Liquidity-sized — ADV >= 5cr/day for 8% sizing, else half-size
  7. Sector concentration cap — max 25% per sector
  8. Triangulation bonus — multi-horizon agreement preferred
  9. Top-N selection (default 8)

Output:
  reports/filter_cascade_<YYYYMMDD>.md
  tmp/from_scratch_7d_run/actionable_today.csv
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
LIVE_LONG_FULL = ROOT / "tmp/from_scratch_7d_run/v3_live_full.csv"
LIVE_LONG_TOP = ROOT / "tmp/from_scratch_7d_run/v3_live_top100.csv"
MH_FULL = ROOT / "tmp/from_scratch_7d_run/multi_horizon_full.csv"
MH_TOP = ROOT / "tmp/from_scratch_7d_run/multi_horizon_top.csv"
MACRO_SENT = ROOT / "data/derived/macro_sentiment.parquet"
NEWS_FEAT = ROOT / "data/derived/news_features.parquet"

REPORT_DIR = ROOT / "reports"
OUT_CSV = ROOT / "tmp/from_scratch_7d_run/actionable_today.csv"

# tunables
MIN_ADV_CR = 0.1
HALF_SIZE_ADV_CR = 5.0  # below this → half-size
RISK_OFF_THRESHOLD = -0.3
RISK_ON_THRESHOLD = 0.3
PATIENCE_FLOOR = 0.65
RISK_OFF_FLOOR = 0.75
MAX_PER_NAME = 0.08
MAX_PER_SECTOR = 0.25
TOP_N_FINAL = 8


def load_macro() -> tuple[float, str]:
    if not MACRO_SENT.exists():
        return 0.0, "NEUTRAL"
    ms = pd.read_parquet(MACRO_SENT)
    ms["as_of"] = pd.to_datetime(ms["as_of"]).dt.date
    row = ms.sort_values("as_of").iloc[-1]
    overall = (float(row.get("global_macro_sent", 0)) + float(row.get("domestic_macro_sent", 0))) / 2
    if overall <= RISK_OFF_THRESHOLD:
        return overall, "RISK_OFF"
    if overall >= RISK_ON_THRESHOLD:
        return overall, "RISK_ON"
    return overall, "NEUTRAL"


def main() -> None:
    today = pd.Timestamp(date.today())
    cascade: list[dict] = []
    drops: dict[str, list[str]] = {}

    # 1. Liquid universe
    px = pd.read_parquet(PRICES)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    latest = px["trade_date"].max()
    snap = px[(px["trade_date"] == latest) & (px["series"] == "EQ")].copy()
    snap["adv_cr"] = snap["avg_traded_value_20d"] / 1e7
    snap = snap[snap["adv_cr"] >= MIN_ADV_CR]
    cascade.append({"step": "1. Liquid universe (ADV ≥ 0.1cr, EQ)", "n_after": len(snap)})
    universe = set(snap["symbol"].astype(str))

    # 2. Feature completeness (proxy: has all of close, sma_50, rsi_14_daily, return_20d)
    must_have = ["close", "sma_50", "rsi_14_daily", "return_20d", "volume_vs_20d"]
    complete_mask = snap[must_have].notna().all(axis=1)
    n_dropped_complete = (~complete_mask).sum()
    snap = snap[complete_mask]
    cascade.append({"step": "2. Feature completeness (≥80% non-null core features)",
                    "n_after": len(snap), "dropped": int(n_dropped_complete)})

    # 3. Pull model scores
    if LIVE_LONG_FULL.exists():
        long_full = pd.read_csv(LIVE_LONG_FULL)
    elif LIVE_LONG_TOP.exists():
        long_full = pd.read_csv(LIVE_LONG_TOP)
    else:
        print("no live long picks — abort")
        return
    long_full = long_full[["symbol", "score_ens", "score_calibrated"]].copy()
    snap = snap.merge(long_full, on="symbol", how="left")
    n_dropped_score = snap["score_calibrated"].isna().sum()
    snap = snap.dropna(subset=["score_calibrated"])
    cascade.append({"step": "3. Has model score (long ensemble)",
                    "n_after": len(snap), "dropped": int(n_dropped_score)})

    # 4. Macro overlay
    macro_score, macro_state = load_macro()
    floor = RISK_OFF_FLOOR if macro_state == "RISK_OFF" else PATIENCE_FLOOR
    n_before = len(snap)
    snap_macro = snap[snap["score_calibrated"] >= floor]
    cascade.append({
        "step": f"4. Macro overlay (state={macro_state}, floor={floor:.2f})",
        "n_after": len(snap_macro), "dropped": n_before - len(snap_macro),
        "note": f"macro_score={macro_score:+.2f}",
    })
    snap = snap_macro

    # 5. RSI sanity
    if len(snap):
        rsi_ok = (snap["rsi_14_daily"] >= 20) & (snap["rsi_14_daily"] <= 90)
        snap_rsi = snap[rsi_ok]
        cascade.append({"step": "5. RSI sanity (20–90 band)",
                        "n_after": len(snap_rsi), "dropped": len(snap) - len(snap_rsi)})
        snap = snap_rsi

    # 6. Triangulation bonus (preferred; not a hard filter)
    if MH_FULL.exists() or MH_TOP.exists():
        mh_path = MH_FULL if MH_FULL.exists() else MH_TOP
        mh = pd.read_csv(mh_path)
        if "triangulated" in mh.columns:
            tri_set = set(mh.loc[mh["triangulated"] == True, "symbol"].astype(str))
            snap["triangulated"] = snap["symbol"].astype(str).isin(tri_set)

    # 7. Per-name and per-sector cap (greedy pack)
    snap = snap.sort_values("score_calibrated", ascending=False)
    sector_alloc: dict[str, float] = {}
    chosen_rows = []
    total = 0.0
    cap_reached = []
    sector_skipped: list[str] = []
    for _, r in snap.iterrows():
        sec = "OTHER"
        if "sector" in r.index and pd.notna(r["sector"]):
            sec = str(r["sector"])
        # liquidity-sized
        sz = MAX_PER_NAME if r["adv_cr"] >= HALF_SIZE_ADV_CR else MAX_PER_NAME / 2
        if sector_alloc.get(sec, 0) + sz > MAX_PER_SECTOR:
            sector_skipped.append(f"{r['symbol']}({sec})")
            continue
        chosen_rows.append({**r.to_dict(), "alloc_pct": sz, "sector_pick": sec})
        sector_alloc[sec] = sector_alloc.get(sec, 0) + sz
        total += sz
        if len(chosen_rows) >= TOP_N_FINAL:
            break

    cascade.append({"step": f"6. Liquidity sizing (8% if ADV≥5cr else 4%) + 25% sector cap",
                    "n_after": len(chosen_rows),
                    "dropped": len(snap) - len(chosen_rows) - len(sector_skipped),
                    "sector_skipped": sector_skipped[:10]})

    cascade.append({"step": f"7. Top-{TOP_N_FINAL} final selection", "n_after": len(chosen_rows)})

    # ── output
    final_df = pd.DataFrame(chosen_rows)
    if len(final_df):
        cols = ["symbol", "sector_pick", "close", "score_calibrated", "score_ens",
                "rsi_14_daily", "return_20d", "adv_cr", "alloc_pct", "triangulated"]
        cols = [c for c in cols if c in final_df.columns]
        final_df = final_df[cols].rename(columns={"sector_pick": "sector", "adv_cr": "adv_cr_per_day"})
        final_df["sl_price"] = (final_df["close"] * 0.95).round(2)
        final_df["t1_price"] = (final_df["close"] * 1.05).round(2)
        final_df["t2_price"] = (final_df["close"] * 1.15).round(2)
        final_df.to_csv(OUT_CSV, index=False)

    # report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rep = REPORT_DIR / f"filter_cascade_{today:%Y%m%d}.md"
    lines = [f"# Filter Cascade — {today:%Y-%m-%d}", ""]
    lines.append(f"**Macro state:** {macro_state} (score={macro_score:+.2f})")
    lines.append(f"**Patience floor:** {floor:.2f}  •  Per-name cap: {MAX_PER_NAME*100:.0f}%  •  Sector cap: {MAX_PER_SECTOR*100:.0f}%")
    lines.append("")
    lines.append("| Step | Stocks remaining | Dropped this step | Notes |")
    lines.append("|---|---:|---:|---|")
    for c in cascade:
        notes = c.get("note", "")
        if c.get("sector_skipped"):
            notes += " sector-cap dropped: " + ", ".join(c["sector_skipped"][:5])
        lines.append(f"| {c['step']} | {c['n_after']:,} | {c.get('dropped', '—'):>4} | {notes} |")
    lines.append("")

    if len(final_df):
        lines.append("## Final actionable list")
        lines.append("")
        lines.append(final_df.to_markdown(index=False))
    else:
        lines.append("## Final actionable list")
        lines.append("")
        lines.append("⚠️ **No names cleared all gates today.** Park in LIQUIDPLUS.")

    rep.write_text("\n".join(lines))
    print(f"\nFilter cascade summary:")
    for c in cascade:
        notes = ""
        if c.get("note"):
            notes = f"  ({c['note']})"
        print(f"  {c['step']:<60} → {c['n_after']:>5,} stocks  (dropped {c.get('dropped','—')}){notes}")
    print(f"\nfinal: {len(chosen_rows)} actionable names")
    print(f"report → {rep}")
    print(f"csv    → {OUT_CSV}")


if __name__ == "__main__":
    main()
