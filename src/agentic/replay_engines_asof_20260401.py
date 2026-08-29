"""APRIL-1 ENGINE REPLAY — what would mb / 180d / hc have flagged on 2026-04-01?

Method: truncate the (CA-repaired) prices parquet to trade_date <= 2026-04-01, patch
each engine module's PRICES/OUT_* constants, and run its main() UNCHANGED — the same
code, features, training cutoffs, and thresholds prod uses, just as-of April 1.
Training discipline is each engine's own (labels' forward windows complete before the
scoring date — no information after Apr-1 enters). Then score every flag list against
what actually happened Apr-1 → Aug-27.

Output: research/doubler_audit/replay/  (per-engine parquets + replay_eval.md)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/agentic"))
AS_OF = pd.Timestamp("2026-04-01")
RP = ROOT / "research/doubler_audit/replay"
RP.mkdir(parents=True, exist_ok=True)
TRUNC = RP / "prices_asof_20260401.parquet"
FULL = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"

if not TRUNC.exists():
    print("building truncated panel …", flush=True)
    px = pd.read_parquet(FULL)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px[px["trade_date"] <= AS_OF].to_parquet(TRUNC, index=False)
    print(f"  {TRUNC.name} written", flush=True)

ENGINES = [
    ("find_multibagger_today", {"PRICES": TRUNC, "OUT_PARQUET": RP / "mb_predictions.parquet",
                                 "OUT_REPORT": RP / "mb_report.md"}),
    ("find_180d_frontier_honest", {"PRICES": TRUNC, "OUT_PARQUET": RP / "f180_frontier.parquet",
                                    "OUT_TODAY": RP / "f180_today.parquet", "OUT_REPORT": RP / "f180_report.md"}),
    ("find_high_conviction", {"PRICES": TRUNC, "OUT_PREDICTIONS": RP / "hc_predictions.parquet",
                               "OUT_REPORT": RP / "hc_report.md"}),
]

for name, patches in ENGINES:
    marker = RP / f".done_{name}"
    if marker.exists():
        print(f"== {name}: cached, skipping ==", flush=True)
        continue
    print(f"\n════════ REPLAY {name} @ {AS_OF.date()} ════════", flush=True)
    mod = importlib.import_module(name)
    for attr, val in patches.items():
        assert hasattr(mod, attr), f"{name} lacks {attr}"
        setattr(mod, attr, val)
    mod.main()
    marker.touch()

# ---------------- evaluation ----------------
print("\n════════ EVALUATION vs Apr-1 → Aug-27 reality ════════", flush=True)
px = pd.read_parquet(FULL, columns=["symbol", "trade_date", "close", "high", "low", "open", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
since = px[px["trade_date"] >= AS_OF].sort_values(["symbol", "trade_date"])
base = since.groupby("symbol").first()
peak = since.groupby("symbol")["high"].max()
lastc = since.groupby("symbol")["close"].last()
lo = since.groupby("symbol")["low"].min()
mult_peak = (peak / base["close"]).rename("peak_mult")
mult_end = (lastc / base["close"]).rename("end_mult")
dd = (lo / base["close"] - 1).rename("max_dd")
inv_univ = base[(base["avg_traded_value_20d"] / 1e7 >= 5) & (base["close"] > 50)].index
doublers_A = set(mult_peak[mult_peak >= 2].index) & set(inv_univ)
base_rate = len(doublers_A) / len(inv_univ)
print(f"investable universe on Apr-1: {len(inv_univ)} · doublers (2x from Apr-1 close): {len(doublers_A)} · base rate {base_rate*100:.1f}%")

def evaluate(label: str, names: list[str]) -> str:
    names = [n for n in names if n in mult_peak.index]
    if not names:
        return f"| {label} | 0 | — | — | — | — | — | — |"
    s2x = [n for n in names if n in doublers_A or mult_peak.get(n, 0) >= 2]
    pm = mult_peak.reindex(names)
    em = mult_end.reindex(names)
    d = dd.reindex(names)
    prec = len(s2x) / len(names) * 100
    return (f"| {label} | {len(names)} | {len(s2x)} ({prec:.0f}%) | {prec/ (base_rate*100):.1f}x | "
            f"{(em.mean()-1)*100:+.1f}% | {(pm.mean()-1)*100:+.1f}% | {d.mean()*100:+.1f}% | "
            f"{(pm>=1.25).mean()*100:.0f}% |")

rows = ["| flag list | n | 2x hits | lift | mean end ret | mean peak ret | mean maxDD | ≥+25% |",
        "|---|---|---|---|---|---|---|---|"]

mb = pd.read_parquet(RP / "mb_predictions.parquet")
if "clears_any_100pct" in mb.columns:
    rows.append(evaluate("mb: clears any 100% bar", mb[mb["clears_any_100pct"]]["symbol"].tolist()))
if "best_score_100pct" in mb.columns:
    rows.append(evaluate("mb: top 20 by best 100% score", mb.nlargest(20, "best_score_100pct")["symbol"].tolist()))
for c in ["score_50pct_180d", "score_75pct_180d"]:
    if c in mb.columns:
        rows.append(evaluate(f"mb: top 20 by {c}", mb.nlargest(20, c)["symbol"].tolist()))

f180 = pd.read_parquet(RP / "f180_today.parquet")
sc = [c for c in f180.columns if c.startswith(("score_", "p_"))]
print("f180 score cols:", sc[:8])
for c in sc[:4]:
    rows.append(evaluate(f"180d: top 20 by {c}", f180.nlargest(20, c)["symbol"].tolist()))

hc = pd.read_parquet(RP / "hc_predictions.parquet")
sc = [c for c in hc.columns if c.startswith(("score", "p_", "prob"))]
print("hc score cols:", sc[:8])
for c in sc[:4]:
    rows.append(evaluate(f"hc: top 10 by {c}", hc.nlargest(10, c)["symbol"].tolist()))
    rows.append(evaluate(f"hc: top 20 by {c}", hc.nlargest(20, c)["symbol"].tolist()))

table = "\n".join(rows)
print("\n" + table)
(RP / "replay_eval.md").write_text(
    f"# April-1 engine replay — evaluation vs Apr-1→Aug-27\n\n"
    f"Universe {len(inv_univ)} investable · {len(doublers_A)} doubled from Apr-1 close · base rate {base_rate*100:.1f}%.\n"
    f"'mean end ret' = equal-weight buy Apr-1 close → Aug-27 close (the sleeve view).\n\n{table}\n")
print("\nREPLAY COMPLETE", flush=True)
