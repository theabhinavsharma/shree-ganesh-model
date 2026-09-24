"""EXP-2026-09-24-engines-count-sizing — does multi-engine consensus earn more under C2?

Inputs (both produced walk-forward, no peeking):
  data/derived/backtest_10yr_15d5pct.parquet          trade table (top-8/week, C2, 0.30% RT)
  data/derived/engine_replay/top30_<engine>.parquet   per-engine top-30 per entry day

engines_count(entry_date, symbol) = number of the 5 engines (cs, hc, mb, mh, 180d) with the
symbol in their top-30 that day — the same definition generate_hybrid_basket.py uses live.

PRIMARY (decides): trade table, Tier-1 (engines_count >= 2) vs Tier-2 (<= 1), C2 net per
trade, eras disc 2020-2022 / conf 2023+. Pass: Tier-1 beats Tier-2 by >= 0.5pp in BOTH eras
with n >= 200 per arm per era; else no sizing change.
SECONDARY (descriptive): the same split over every QC-clean name each window.

Writes reports/ab_engines_count.md and appends the verdict to logs/experiments.jsonl.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import backtest_10yr_15d5pct as B  # noqa: E402

TRADES = ROOT / "data/derived/backtest_10yr_15d5pct.parquet"
REPLAY = ROOT / "data/derived/engine_replay"
ENGINES = ["cs", "hc", "mb", "mh", "180d"]
EXP_ID = "EXP-2026-09-24-engines-count-sizing"
BAR_PP, MIN_N = 0.5, 200


def engines_table() -> pd.DataFrame:
    parts = []
    for e in ENGINES:
        t = pd.read_parquet(REPLAY / f"top30_{e}.parquet")[["trade_date", "symbol"]]
        parts.append(t.assign(**{f"in_{e}": 1}).set_index(["trade_date", "symbol"]))
    m = pd.concat(parts, axis=1).fillna(0).astype(int)
    m["engines_count"] = m.sum(axis=1)
    start = max(pd.read_parquet(REPLAY / f"top30_{e}.parquet")["trade_date"].min() for e in ENGINES)
    return m.reset_index(), start


def attach(trades: pd.DataFrame, eng: pd.DataFrame, start) -> pd.DataFrame:
    t = trades.copy(); t["entry_date"] = pd.to_datetime(t["entry_date"])
    t = t[t["entry_date"] >= start]
    t = t.merge(eng, left_on=["entry_date", "symbol"], right_on=["trade_date", "symbol"], how="left").drop(columns="trade_date")
    t["engines_count"] = t["engines_count"].fillna(0).astype(int)
    t["era"] = np.where(t["entry_date"].dt.year <= 2022, "disc", "conf")
    t["tier"] = np.where(t["engines_count"] >= 2, "tier1", "tier2")
    return t


def stats(g: pd.DataFrame) -> dict:
    return {"n": int(len(g)), "net_per_trade": round(float(g["realized_pct"].mean()), 2) if len(g) else None,
            "median": round(float(g["realized_pct"].median()), 2) if len(g) else None,
            "touch_rate": round(float(g["touched_5pct"].mean() * 100), 1) if len(g) else None,
            "sl_rate": round(float((g["exit_reason"] == "SL").mean() * 100), 1) if len(g) else None}


def split(t: pd.DataFrame) -> dict:
    out = {}
    for era in ("disc", "conf"):
        e = t[t["era"] == era]
        out[era] = {"tier1": stats(e[e["tier"] == "tier1"]), "tier2": stats(e[e["tier"] == "tier2"]),
                    "by_count": {str(k): stats(g) for k, g in e.groupby(e["engines_count"].clip(upper=3))}}
    return out


def verdict(res: dict) -> tuple[str, dict]:
    detail, ok = {}, True
    for era in ("disc", "conf"):
        a, b = res[era]["tier1"], res[era]["tier2"]
        gap = None if a["n"] == 0 or b["n"] == 0 else round(a["net_per_trade"] - b["net_per_trade"], 2)
        n_ok = a["n"] >= MIN_N and b["n"] >= MIN_N
        detail[era] = {"gap_pp": gap, "n_ok": n_ok}
        ok &= n_ok and gap is not None and gap >= BAR_PP
    return ("PASS" if ok else "FAIL"), detail


def universe_trades(start) -> pd.DataFrame:
    """Every QC-clean name per window (backtest QC), C2 day-by-day, 0.30% RT."""
    df = B.load_prices(); contam = B.contamination_set(df)
    days = sorted(df["trade_date"].unique())
    rows = []
    for w in B.generate_windows():
        i = np.searchsorted(days, np.datetime64(pd.Timestamp(w)))
        if i + B.HOLD_DAYS >= len(days) or days[i] < np.datetime64(start):
            continue
        d = pd.Timestamp(days[i])
        clean = B.qc(B.snap_features(df, d), contam)
        paths = B.forward_paths(df, d, B.HOLD_DAYS, set(clean["symbol"]))
        for _, r in clean.iterrows():
            o = B.c2_exit(paths.get(r["symbol"]), r["dvol_20d"])
            if o:
                rows.append({"entry_date": d, "symbol": r["symbol"], "realized_pct": o[0] - B.COST_RT_PCT,
                             "exit_reason": o[1], "touched_5pct": o[1] in B.TOUCH_REASONS})
    return pd.DataFrame(rows).drop_duplicates(["entry_date", "symbol"])


def table(res: dict) -> list[str]:
    md = ["| era | engines_count | n | C2 net/trade % | median % | touch % | SL % |", "|---|---|---|---|---|---|---|"]
    for era in ("disc", "conf"):
        for k, s in res[era]["by_count"].items():
            md.append(f"| {era} | {k if k != '3' else '3+'} | {s['n']} | {s['net_per_trade']} | {s['median']} | {s['touch_rate']} | {s['sl_rate']} |")
        for tier in ("tier1", "tier2"):
            s = res[era][tier]
            md.append(f"| {era} | **{tier}** ({'>=2' if tier == 'tier1' else '<=1'}) | {s['n']} | {s['net_per_trade']} | {s['median']} | {s['touch_rate']} | {s['sl_rate']} |")
    return md


def main():
    eng, start = engines_table()
    trades = attach(pd.read_parquet(TRADES), eng, start)
    prim = split(trades); v, detail = verdict(prim)
    uni = attach(universe_trades(start), eng, start); sec = split(uni)
    rng = f"{trades['entry_date'].min().date()}..{trades['entry_date'].max().date()}"
    md = [f"# engines_count sizing A/B — {EXP_ID}", "",
          f"_generated {datetime.now():%Y-%m-%d %H:%M} · engine sets: data/derived/engine_replay/top30_*.parquet (walk-forward, yearly refit) · "
          f"trades: data/derived/backtest_10yr_15d5pct.parquet {rng} · C2 exits, next-open entry, 0.30% RT_", "",
          f"**Verdict: {v}** — bar: Tier-1 beats Tier-2 on C2 net/trade by >= {BAR_PP}pp in both eras, n >= {MIN_N} per arm per era.", "",
          f"Gap (tier1 − tier2): disc {detail['disc']['gap_pp']}pp (n ok: {detail['disc']['n_ok']}), conf {detail['conf']['gap_pp']}pp (n ok: {detail['conf']['n_ok']}).", "",
          "## Primary — backtest trade table (top-8 per week)", ""] + table(prim) + [
          "", f"## Secondary (descriptive, no decision) — every QC-clean name per window, {uni['entry_date'].min().date()}..{uni['entry_date'].max().date()}", ""] + table(sec) + [
          "", "## Caveats", "",
          "- Engines refit once per scoring year (live refits every run); labels embargoed at the scoring day.",
          "- cs/hc extras exist only from 2023-06; median-filled before, exactly as live does.",
          "- mh uses today's sector-index membership (not point-in-time) and catalyst_features ends 2026-06-01, both as live.",
          "- disc era here is 2020-2022 only (engines_count needs 2018-19 history for mh)."]
    out = ROOT / "reports/ab_engines_count.md"; out.write_text("\n".join(md) + "\n")
    with open(ROOT / "logs/experiments.jsonl", "a") as f:
        f.write(json.dumps({"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "id": EXP_ID, "status": v,
                            "result": {"primary": prim, "secondary": sec, "gap": detail}, "trades_range": rng,
                            "verdict": f"{v}: see reports/ab_engines_count.md"}) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
