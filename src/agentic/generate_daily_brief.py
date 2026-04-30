"""
Synthesize the daily brief from latest model outputs + alt-data signals.

Inputs (latest snapshots):
  - tmp/from_scratch_7d_run/v3_live_top100.csv     (model)
  - tmp/from_scratch_7d_run/live_top100_with_levels.csv  (entry/target/SL quantiles)
  - data/derived/news_feed.parquet
  - data/derived/reddit_feed.parquet
  - data/derived/youtube_videos.parquet
  - tmp/from_scratch_7d_run/alt/announcements_tagged.parquet
  - tmp/from_scratch_7d_run/alt/insider_trading_pit.parquet

Output: reports/daily_brief_<as_of>.md
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
TMP = ROOT / "tmp/from_scratch_7d_run"
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    v3 = load(TMP / "v3_live_top100.csv")
    levels = load(TMP / "live_top100_with_levels.csv")
    news = load(ROOT / "data/derived/news_feed.parquet")
    reddit = load(ROOT / "data/derived/reddit_feed.parquet")
    yt = load(ROOT / "data/derived/youtube_videos.parquet")
    ann = load(TMP / "alt/announcements_tagged.parquet")
    pit = load(TMP / "alt/insider_trading_pit.parquet")

    if v3 is None:
        print("v3_live_top100.csv missing — run pipeline first")
        return

    as_of = pd.to_datetime(v3["trade_date"]).iloc[0].date() if "trade_date" in v3.columns else datetime.now().date()
    out_path = OUT_DIR / f"daily_brief_{as_of}.md"

    # merge model levels into v3 picks
    if levels is not None and "symbol" in levels.columns:
        v3 = v3.merge(levels[[c for c in [
            "symbol", "pred_entry_low_q25", "pred_entry_high_q50",
            "pred_target_q50", "pred_target_q75", "pred_sl_q25",
            "pred_argmax_day", "calibrated_hit_5pct_7td",
        ] if c in levels.columns]], on="symbol", how="left")

    v3 = v3[v3["adv_20d_cr"] >= 1.0].copy() if "adv_20d_cr" in v3.columns else v3
    v3 = v3.sort_values("score_ens", ascending=False).head(20).reset_index(drop=True)

    # alt-signal merges
    if news is not None and not news.empty:
        recent_news = news[news["pub_ts"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)] if "pub_ts" in news.columns else news.tail(0)
        news_counts = recent_news.explode("symbols").groupby("symbols").size().rename("news_48h_count")
        v3 = v3.merge(news_counts.reset_index().rename(columns={"symbols": "symbol"}), on="symbol", how="left")
    if reddit is not None and not reddit.empty:
        rr = reddit.copy()
        if "created_ts" in rr.columns:
            rr = rr[rr["created_ts"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)]
        rg = rr.explode("symbols").groupby("symbols").agg(
            reddit_48h_count=("id", "count"),
            reddit_score=("score", "sum"),
        ).reset_index().rename(columns={"symbols": "symbol"})
        v3 = v3.merge(rg, on="symbol", how="left")
    if yt is not None and not yt.empty and "symbols" in yt.columns:
        yt_counts = yt.explode("symbols").groupby("symbols").size().rename("yt_count")
        v3 = v3.merge(yt_counts.reset_index().rename(columns={"symbols": "symbol"}), on="symbol", how="left")
    if pit is not None:
        pit2 = pit.copy()
        pit2["intim_dt"] = pd.to_datetime(pit2["intimDt"].astype(str).str.split(" ").str[0],
                                          format="%d-%b-%Y", errors="coerce")
        pit2 = pit2[(pit2["intim_dt"] > pd.Timestamp(as_of) - pd.Timedelta(days=60)) & (pit2["delta_pct"].abs() < 5.0)]
        pit2["net_inr"] = pit2["buyValue"].fillna(0) - pit2["sellValue"].fillna(0)
        ag = pit2.groupby("symbol").agg(insider_60d_inr=("net_inr", "sum")).reset_index()
        v3 = v3.merge(ag, on="symbol", how="left")
    if ann is not None:
        ann2 = ann.copy()
        ann2 = ann2[ann2["ann_date"] > pd.Timestamp(as_of) - pd.Timedelta(days=14)]
        # latest catalyst per symbol
        ann2 = ann2.sort_values("ann_date", ascending=False)
        latest_cat = ann2.groupby("symbol").first().reset_index()[["symbol", "catalyst_cat", "catalyst_score", "desc"]]
        latest_cat = latest_cat.rename(columns={"desc": "latest_ann_subject"})
        v3 = v3.merge(latest_cat, on="symbol", how="left")

    for c in ["news_48h_count", "reddit_48h_count", "reddit_score", "yt_count", "insider_60d_inr"]:
        if c in v3.columns:
            v3[c] = v3[c].fillna(0)

    # build markdown
    lines = []
    lines.append(f"# Daily Brief — {as_of}")
    lines.append(f"_generated {datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("## Top 20 model picks (v3 ensemble + alt-signal overlay)\n")
    lines.append("| # | Symbol | Sector | Close | Score | Pwin* | Entry lo-hi | Tgt 50/75 | SL | News/RD/YT | Insider₹cr | Catalyst |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in v3.iterrows():
        c = r.get("close", float("nan"))
        e_lo = r.get("pred_entry_low_q25"); e_hi = r.get("pred_entry_high_q50")
        t50 = r.get("pred_target_q50"); t75 = r.get("pred_target_q75"); sl = r.get("pred_sl_q25")
        e_str = f"{c*(1+e_lo):.1f}-{c*(1+e_hi):.1f}" if pd.notna(e_lo) and pd.notna(e_hi) and pd.notna(c) else "-"
        t_str = f"{c*(1+t50):.1f}/{c*(1+t75):.1f}" if pd.notna(t50) and pd.notna(t75) and pd.notna(c) else "-"
        sl_str = f"{c*(1+sl):.1f}" if pd.notna(sl) and pd.notna(c) else "-"
        pwin = r.get("score_calibrated", r.get("score_ens", 0)) * 100
        alt = f"{int(r.get('news_48h_count',0))}/{int(r.get('reddit_48h_count',0))}/{int(r.get('yt_count',0))}"
        ins = r.get("insider_60d_inr", 0) / 1e7 if pd.notna(r.get("insider_60d_inr", 0)) else 0
        cat = r.get("catalyst_cat", "")
        cat_str = (cat if isinstance(cat, str) else "") + (f" ({r.get('latest_ann_subject','')[:60]})" if isinstance(r.get("latest_ann_subject"), str) else "")
        lines.append(f"| {i+1} | {r['symbol']} | {r.get('sector','')} | {c:.1f} | {r['score_ens']:.3f} | {pwin:.1f}% | {e_str} | {t_str} | {sl_str} | {alt} | {ins:+.1f} | {cat_str} |")

    lines.append("\n*Pwin = isotonic-calibrated ensemble probability of ≥+5% in any of next 7 trading days.*\n")
    lines.append("\n_News/RD/YT_ = mention counts in news (48h), reddit (48h), YouTube (recent).\n")

    # market regime block
    lines.append("\n## Market regime\n")
    if "market_breadth_50dma" in v3.columns:
        b50 = v3["market_breadth_50dma"].iloc[0]
        b200 = v3["market_breadth_200dma"].iloc[0]
        lines.append(f"- Breadth above 50dma: **{b50*100:.1f}%**, above 200dma: **{b200*100:.1f}%**")

    # top reddit/news themes (raw)
    if reddit is not None and "title" in reddit.columns:
        rr = reddit.copy()
        rr = rr.sort_values("score", ascending=False)
        top_posts = rr.head(8)
        lines.append("\n## Top reddit posts (by score)\n")
        for _, row in top_posts.iterrows():
            lines.append(f"- **r/{row['sub']}** [{int(row.get('score',0))} pts, {int(row.get('num_comments',0))}c] — {row['title']}")

    if news is not None and "title" in news.columns:
        nn = news.copy()
        if "pub_ts" in nn.columns:
            nn = nn.sort_values("pub_ts", ascending=False)
        lines.append("\n## Latest news headlines\n")
        for _, row in nn.head(10).iterrows():
            lines.append(f"- [{row['source']}] {row['title']}")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
