"""Score our own past baskets weekly — the self-scoring half of the RL loop.

Finds every committed basket in live_predictions/*_15d5pct.json whose 15-session
window has completed (or is in progress) and is not yet scored, simulates the
PUBLISHED exit rules day-by-day (entry next-open, SL-first on both-touch), and
appends one row per basket to logs/basket_outcomes.jsonl:
  - per-pick: entry, exit reason/date, realized %, touched_5pct (any day in window)
  - basket: realized P&L, touch rate (the 90% goal metric), SL rate

Wired into run_weekly_pipeline.sh step 3.4 — runs BEFORE miss_learner so the
weekly RL cycle sees our own hits/losses first.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
LOG = ROOT / "logs/basket_outcomes.jsonl"


def score_basket(basket: dict, px: pd.DataFrame) -> dict | None:
    entry_after = pd.Timestamp(basket["data_through"])
    picks_out, n_touch, n_sl = [], 0, 0
    complete = True
    for p in basket["picks"]:
        g = px[(px["symbol"] == p["symbol"]) & (px["trade_date"] > entry_after)].head(15)
        if len(g) == 0 or pd.isna(g.iloc[0]["open"]):
            picks_out.append({"symbol": p["symbol"], "status": "NO_DATA"}); continue
        if len(g) < 15:
            complete = False
        ep = float(g.iloc[0]["open"])
        tgt, sl = ep * 1.05, ep * 0.97
        status, ret, exit_day = "OPEN", None, None
        half, trail, touched = False, None, False
        for i in range(len(g)):
            d = g.iloc[i]; lo, hi = d["low"], d["high"]
            if pd.isna(lo): continue
            if hi >= tgt: touched = True
            if not half:
                if lo <= sl:
                    status, ret, exit_day = "SL", -3.0, str(d["trade_date"].date()); break
                if hi >= tgt:
                    half, trail = True, ep * 1.025; continue
            else:
                if lo <= trail:
                    status, ret, exit_day = "TGT_TRAIL", 3.75, str(d["trade_date"].date()); break
        if status == "OPEN":
            mtm = (float(g.iloc[-1]["close"]) / ep - 1) * 100
            ret = (5.0 + mtm) / 2 if half else mtm
            status = "TIMEOUT" if len(g) >= 15 else ("TRAILING" if half else "OPEN")
        n_touch += touched
        n_sl += status == "SL"
        picks_out.append({"symbol": p["symbol"], "entry": round(ep, 2), "status": status,
                          "ret_pct": round(ret, 2), "exit_day": exit_day, "touched_5pct": bool(touched),
                          "rank": p.get("rank"), "confidence": p.get("confidence")})
    scored = [x for x in picks_out if "ret_pct" in x]
    if not scored: return None
    return {
        "basket": basket["as_of_date"], "data_through": basket["data_through"],
        "window_complete": complete, "n_picks": len(scored),
        "basket_pnl_pct": round(sum(x["ret_pct"] for x in scored) / len(scored), 2),
        "touch_rate": round(n_touch / len(scored), 3),          # ← the 90% goal metric
        "sl_rate": round(n_sl / len(scored), 3),
        "picks": picks_out,
    }


def main() -> None:
    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                         columns=["symbol", "trade_date", "open", "high", "low", "close"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])

    seen = set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("window_complete"): seen.add(r["basket"])

    latest = px["trade_date"].max()
    rows = []
    for f in sorted((ROOT / "live_predictions").glob("*_15d5pct.json")):
        b = json.loads(f.read_text())
        if b["as_of_date"] in seen: continue
        if pd.Timestamp(b["data_through"]) >= latest: continue   # no forward data yet
        r = score_basket(b, px)
        if r: rows.append(r)

    if not rows:
        print("no unscored baskets"); return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    for r in rows:
        tag = "final" if r["window_complete"] else f"in-progress"
        print(f"  {r['basket']} [{tag}]: P&L {r['basket_pnl_pct']:+.2f}%  touch {r['touch_rate']*100:.0f}%  "
              f"SL {r['sl_rate']*100:.0f}%  ({r['n_picks']} picks)")
    # goal tracker
    finals = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    finals = [r for r in finals if r.get("window_complete")]
    if finals:
        tr = sum(r["touch_rate"] for r in finals) / len(finals)
        print(f"  GOAL TRACKER: mean touch rate {tr*100:.0f}% across {len(finals)} completed baskets (target 90%)")
        # RANK VALIDATION: does confidence rank actually predict outcomes?
        ranked = [(x["rank"], x["touched_5pct"]) for r in finals for x in r["picks"]
                  if x.get("rank") is not None and "touched_5pct" in x]
        if len(ranked) >= 16:
            top = [t for rk, t in ranked if rk <= 4]; bot = [t for rk, t in ranked if rk >= 5]
            print(f"  RANK CHECK: ranks 1-4 touch {sum(top)/len(top)*100:.0f}% vs ranks 5-8 {sum(bot)/len(bot)*100:.0f}% "
                  f"— {'ranking earns its place' if sum(top)/len(top) > sum(bot)/len(bot) else '⚠️ ranking NOT predictive, revisit weights'}")


if __name__ == "__main__":
    main()
