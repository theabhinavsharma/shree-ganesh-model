"""ENGINE ATTRIBUTION — archive every engine's flags weekly, score them against reality.

Born 2026-08-29 from the doubler audit: the mb engine ran for months claiming
"90% calibrated P(2x)" with zero outcome accountability (its point-in-time flags
survived only by accident in git history). Rule going forward: every engine's
top-N is archived every run, and every archived snapshot old enough to judge
gets scored. An engine that isn't earning its flags shows up here within weeks.

Outputs:
  logs/engine_flags/YYYY-MM-DD.json      (today's top-N per engine — the archive)
  logs/engine_attribution.jsonl          (scored snapshots, appended)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
FLAGS = ROOT / "logs/engine_flags"
FLAGS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "logs/engine_attribution.jsonl"

ENGINES = {
    "mb_best100": ("data/derived/multibagger_today_predictions.parquet", "best_score_100pct", 20),
    "hc_20pct30d": ("data/derived/high_conviction_predictions.parquet", "score_20pct_30d_cal", 20),
    "hc_10pct15d": ("data/derived/high_conviction_predictions.parquet", "score_10pct_15d_cal", 20),
    "f180_15pct": ("data/derived/180d_today_predictions.parquet", "score_15pct", 20),
    "f180_50pct": ("data/derived/180d_today_predictions.parquet", "score_50pct", 20),
    "cs_5pct15d": ("data/derived/compare_short_horizons.parquet", "score_5pct_15d", 20),
}


def main() -> None:
    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                         columns=["symbol", "trade_date", "close", "high", "avg_traded_value_20d"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    today = px["trade_date"].max()
    snap = px[px["trade_date"] == today].set_index("symbol")
    inv = set(snap[(snap["avg_traded_value_20d"] / 1e7 >= 5) & (snap["close"] > 50)].index)

    # ---- 1. archive today's flags ----
    flags = {"as_of": str(today.date())}
    for name, (path, col, n) in ENGINES.items():
        fp = ROOT / path
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if col not in df.columns or "symbol" not in df.columns:
            continue
        flags[name] = df[df["symbol"].isin(inv)].nlargest(n, col)["symbol"].tolist()
    (FLAGS / f"{today.date()}.json").write_text(json.dumps(flags, indent=1))
    print(f"archived flags for {sum(1 for k in flags if k != 'as_of')} engines → logs/engine_flags/{today.date()}.json")

    # ---- 2. score archived snapshots old enough to judge ----
    scored_keys = set()
    if OUT.exists():
        scored_keys = {(r["as_of"], r["engine"], r["horizon_td"]) for r in
                       (json.loads(l) for l in OUT.read_text().splitlines() if l.strip())}
    days = sorted(px["trade_date"].unique())

    def fwd_stats(sym: str, d0: pd.Timestamp, n_td: int):
        g = px[(px["symbol"] == sym) & (px["trade_date"] > d0)].head(n_td)
        if len(g) < min(n_td, 20):
            return None
        base = px[(px["symbol"] == sym) & (px["trade_date"] == d0)]
        if base.empty:
            return None
        c0 = float(base.iloc[0]["close"])
        return dict(peak=(g["high"].max() / c0 - 1) * 100, end=(float(g.iloc[-1]["close"]) / c0 - 1) * 100)

    new_rows = []
    for f in sorted(FLAGS.glob("*.json")):
        rec = json.loads(f.read_text())
        d0 = pd.Timestamp(rec["as_of"])
        age_td = sum(1 for d in days if d > d0)
        for horizon in (30, 100):
            if age_td < horizon:
                continue
            for eng, syms in rec.items():
                if eng == "as_of" or (rec["as_of"], eng, horizon) in scored_keys:
                    continue
                stats = [s for s in (fwd_stats(x, d0, horizon) for x in syms) if s]
                if len(stats) < 5:
                    continue
                peaks = [s["peak"] for s in stats]; ends = [s["end"] for s in stats]
                row = dict(as_of=rec["as_of"], engine=eng, horizon_td=horizon, n=len(stats),
                           touch25=round(sum(p >= 25 for p in peaks) / len(peaks) * 100, 1),
                           touch2x=round(sum(p >= 100 for p in peaks) / len(peaks) * 100, 1),
                           mean_end=round(sum(ends) / len(ends), 2),
                           scored_on=str(date.today()))
                new_rows.append(row)
    if new_rows:
        with OUT.open("a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
    print(f"scored {len(new_rows)} (snapshot, engine, horizon) cells")
    for r in new_rows:
        print(f"  {r['as_of']} {r['engine']:14s} {r['horizon_td']}td: touch+25% {r['touch25']}% · 2x {r['touch2x']}% · mean end {r['mean_end']:+.1f}% (n={r['n']})")


if __name__ == "__main__":
    main()
