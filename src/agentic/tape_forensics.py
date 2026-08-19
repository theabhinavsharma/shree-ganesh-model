"""Tape forensics — the deep per-name scrutiny, applied to EVERY basket name, every run.

Born 2026-08-18: the user caught RSYSTEMS sinking by eye; forensics then found the same
flags on LXCHEM and RACLGEAR that nobody had looked at. Rule: if one name deserves the
tape-forensics treatment, they all do — automatically, before entry, not after a question.

For each pick + reserve in the latest basket:
  • DISTRIBUTION days   — return <= -3% on volume > 1.5x (institutional selling)
  • POST-RESULTS DUMP   — results filing followed by <= -3% next session on >1.5x volume
                          (informed selling we cannot read — we don't parse financials)
  • Structure           — above/below 50DMA & 200DMA, last-15-session trend slope
  • Accumulation        — share of sessions with delivery > 1.1x of its 20d norm
  • Low-volume drift    — dumps on <0.8x volume are noted as drift, NOT distribution

Output: reports/tape_forensics_<date>.md + stdout summary with 🔴/🟢 per name.
Exit code always 0 (informational — the human judges swaps; verdicts belong to the analyst
layer). Wired into run_weekly_pipeline.sh step 4.3.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")


def forensics_for(sym: str, px: pd.DataFrame, ann: pd.DataFrame) -> dict:
    g = px[px["symbol"] == sym].sort_values("trade_date").tail(30).reset_index(drop=True)
    if len(g) < 15:
        return {"symbol": sym, "red": ["insufficient history"], "green": []}
    last = g.iloc[-1]
    g15 = g.tail(15).assign(dlv=lambda d: d["delivery_pct"] / d["avg_delivery_pct_20d"])

    dist = g15[(g15["volume_vs_20d"] > 1.5) & (g15["return_1d"] <= -0.03)]
    acc_share = float((g15["dlv"] > 1.1).mean())
    hv_up = g15[(g15["volume_vs_20d"] > 1.5) & (g15["return_1d"] >= 0.03)]
    slope = float(np.polyfit(range(len(g15)), g15["close"].values, 1)[0] / g15["close"].mean() * 100)
    d50 = float(last["close"] / last["sma_50"] - 1)
    d200 = float(last["close"] / last["sma_200"] - 1)

    a = ann[(ann["symbol"] == sym) & (ann["is_results_event"]) & (ann["event_date"] >= g["trade_date"].min())]
    dumps = []
    for _, r in a.drop_duplicates("event_date").iterrows():
        nxt = g[g["trade_date"] > r["event_date"]].head(1)
        if len(nxt) and float(nxt.iloc[0]["return_1d"]) <= -0.03:
            dumps.append((str(r["event_date"].date())[5:],
                          float(nxt.iloc[0]["return_1d"]) * 100,
                          float(nxt.iloc[0]["volume_vs_20d"])))

    red, green = [], []
    if len(dist):
        red.append(f"DISTRIBUTION x{len(dist)} ({', '.join(str(d.date())[5:] for d in dist['trade_date'])})")
    for d in dumps:
        sev = "HIGH-VOL" if d[2] > 1.5 else "low-vol drift"
        (red if d[2] > 1.5 else green if False else red).append(
            f"POST-RESULTS {'DUMP' if d[2] > 1.5 else 'drift'} ({d[0]}: {d[1]:+.1f}% on {d[2]:.1f}x, {sev})")
    if d200 < 0: red.append(f"BELOW 200DMA ({d200:+.1%})")
    if d50 < 0: red.append(f"below 50DMA ({d50:+.1%})")
    if slope < -0.3: red.append(f"downtrend last-15d ({slope:.2f}%/day)")
    if acc_share >= 0.6: green.append(f"delivery>1.1x in {acc_share*100:.0f}% of sessions")
    if len(hv_up): green.append(f"high-vol UP days x{len(hv_up)}")
    if slope > 0.15: green.append(f"uptrend last-15d ({slope:+.2f}%/day)")
    if d50 > 0 and d200 > 0: green.append("above both DMAs")
    # severity: 3 = high-vol post-results dump OR (distribution + below 200DMA);
    # 2 = distribution or high-vol dump alone; 1 = structural reds only; 0 = clean
    hv_dump = any(d[2] > 1.5 for d in dumps)
    if (hv_dump and d200 < 0) or (len(dist) and d200 < 0 and not green): sev = 3
    elif hv_dump or len(dist): sev = 2
    elif red: sev = 1
    else: sev = 0
    return {"symbol": sym, "severity": sev, "red": red, "green": green}


def main() -> None:
    baskets = sorted((ROOT / "live_predictions").glob("*_15d5pct.json"))
    b = json.loads(baskets[-1].read_text())
    names = [(str(p["rank"]), p["symbol"]) for p in b["picks"]] + \
            [(str(r["rank"]), r["symbol"]) for r in b.get("reserves", [])]

    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
        columns=["symbol", "trade_date", "close", "return_1d", "volume_vs_20d",
                 "delivery_pct", "avg_delivery_pct_20d", "sma_50", "sma_200"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    ann = pd.read_parquet(ROOT / "data/events_full_history/normalized/stock_announcements.parquet",
        columns=["symbol", "event_date", "is_results_event"])
    ann["event_date"] = pd.to_datetime(ann["event_date"], errors="coerce")

    lines = [f"# Tape Forensics — {b['as_of_date']} basket", ""]
    worst = []
    for rank, sym in names:
        f = forensics_for(sym, px, ann)
        sev = f.get("severity", 0)
        worst.append((sev, rank, sym))
        icon = ["🟢", "🟡", "🟠", "🔴"][min(sev, 3)]
        print(f"{icon} sev{sev} {rank:>2s} {sym:<12s} 🔴 {'; '.join(f['red']) or 'none'}")
        print(f"{'':>22s} 🟢 {'; '.join(f['green']) or 'none'}")
        lines += [f"## {icon} sev{sev} · rank {rank} · {sym}",
                  f"- RED: {'; '.join(f['red']) or 'none'}",
                  f"- GREEN: {'; '.join(f['green']) or 'none'}", ""]
    sev3 = [f"{r} {s}" for sev, r, s in worst if sev >= 3]
    if sev3:
        print(f"\n⚠️  SEVERITY-3 NAMES (candidates for reserve swap): {', '.join(sev3)}")
        lines.append(f"**SEVERITY-3 (swap candidates): {', '.join(sev3)}**")
    out = ROOT / f"reports/tape_forensics_{b['as_of_date'].replace('-','')}.md"
    out.write_text("\n".join(lines))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
