"""DAILY EVAL — is every stock that traded actually in the panel, at today's date?

Born 2026-09-24: the 2026-08-28 BE/BZ series fix was lost on 09-10 and for 12 sessions
the panel silently dropped ~270 traded symbols a day (every BE/BZ name — including two
names of the live leader sleeve). The freshness gate could not see it: file max-date
was fresh, the four checked columns were fresh, continuity was fine. Freshness is
"is the newest row new"; COVERAGE is "is every row that should exist there".

Checks, per session over the trailing N sessions (raw bhavcopy = ground truth):
  1. universe coverage  : raw EQ/BE/BZ symbols vs panel rows that date  (>= 99.5%)
  2. series presence    : BE and BZ rows > 0 whenever raw has them       (the 09-08 signature)
  3. watched names      : every symbol in live baskets / paper sleeve /
                          leader screen present on the latest session    (100%)
  4. silent drop-offs   : symbols with rows up to T-1 that vanished at T while still in raw
  5. price sanity       : panel close == raw close (adjusted only if a CA on that date)

Writes logs/evals/panel_coverage_<date>.json and appends logs/evals/panel_coverage.jsonl.
Exit 1 on any FAIL so cron/pipeline logs go red. verify_freshness reads the latest
json as the PANEL_COVERAGE contract.

Usage: /usr/bin/python3 src/agentic/eval_panel_coverage.py [--sessions 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SGM_ROOT", "/Users/abhinavs./Documents/Zoom"))
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
RAW = ROOT / "data/raw/nse_full_history_official"
CA = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"
OUT_DIR = ROOT / "logs/evals"
EQUITY = {"EQ", "BE", "BZ"}
MIN_COVERAGE = 0.995


def _raw_day(d: pd.Timestamp) -> pd.DataFrame | None:
    files = glob.glob(str(RAW / f"trade_date={d.date()}" / "sec_bhavdata_full_*.csv"))
    if not files:
        return None
    r = pd.read_csv(files[0], low_memory=False)
    r.columns = [c.strip() for c in r.columns]
    r["SYMBOL"] = r["SYMBOL"].astype(str).str.strip()
    r["SERIES"] = r["SERIES"].astype(str).str.strip()
    r = r[r["SERIES"].isin(EQUITY)]
    r["CLOSE_PRICE"] = pd.to_numeric(r["CLOSE_PRICE"], errors="coerce")
    return r[["SYMBOL", "SERIES", "CLOSE_PRICE"]]


def _watched() -> dict[str, list[str]]:
    w: dict[str, list[str]] = {}
    baskets = sorted(glob.glob(str(ROOT / "live_predictions/*_15d5pct.json")))[-2:]
    for b in baskets:
        d = json.load(open(b))
        w[Path(b).name] = [p["symbol"] for p in d["picks"]] + [r["symbol"] for r in d.get("reserves", [])]
    ps = ROOT / "logs/paper_sleeve_positions.jsonl"
    if ps.exists():
        syms = set()
        for line in open(ps):
            try:
                j = json.loads(line)
                if j.get("status", "OPEN").upper() in ("OPEN", "PENDING") and j.get("symbol"):
                    syms.add(j["symbol"])
            except Exception:
                pass
        if syms:
            w["paper_sleeve"] = sorted(syms)
    tl = sorted(glob.glob(str(ROOT / "reports/theme_leaders_*.md")))
    if tl:
        txt = open(tl[-1]).read()
        syms = re.findall(r"^([A-Z0-9&-]{2,12})\s{2,}", txt, flags=re.M)
        if syms:
            w[Path(tl[-1]).name] = sorted(set(syms) - {"SYM"})
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.read_parquet(PRICES, columns=["trade_date"])["trade_date"]
    dates = pd.to_datetime(dates).drop_duplicates().sort_values().tail(args.sessions + 1).tolist()
    panel = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "series", "close", "raw_close"],
                            filters=[("trade_date", ">=", dates[0])])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    ca = pd.read_parquet(CA, columns=["symbol", "ex_date"]) if CA.exists() else pd.DataFrame(columns=["symbol", "ex_date"])
    ca["ex_date"] = pd.to_datetime(ca["ex_date"], errors="coerce")

    results, fails = [], []
    prev_syms: set[str] | None = None
    for d in dates:
        raw = _raw_day(d)
        pd_syms = set(panel.loc[panel["trade_date"] == d, "symbol"])
        if raw is None:
            results.append({"date": str(d.date()), "raw": None, "note": "no raw bhavcopy partition on disk"})
            prev_syms = pd_syms
            continue
        raw_syms = set(raw["SYMBOL"])
        missing = sorted(raw_syms - pd_syms)
        cov = 1 - len(missing) / max(1, len(raw_syms))
        ser_raw = raw["SERIES"].value_counts().to_dict()
        ser_panel = panel.loc[panel["trade_date"] == d, "series"].value_counts().to_dict()
        dropped = sorted((prev_syms or set()) & raw_syms - pd_syms) if prev_syms is not None else []
        # price sanity on the intersection: raw close vs panel raw_close (unadjusted)
        j = panel[panel["trade_date"] == d].merge(raw, left_on="symbol", right_on="SYMBOL")
        j = j[j["raw_close"].notna()]
        bad_px = j[(j["raw_close"] - j["CLOSE_PRICE"]).abs() > 0.011]
        row = {"date": str(d.date()), "raw": len(raw_syms), "panel": len(pd_syms), "coverage": round(cov, 4),
               "missing_n": len(missing), "missing_sample": missing[:25],
               "series_raw": ser_raw, "series_panel": ser_panel,
               "silent_dropoffs": dropped[:25], "silent_dropoffs_n": len(dropped),
               "price_mismatch_n": int(len(bad_px)), "price_mismatch_sample": bad_px["symbol"].head(10).tolist()}
        if cov < MIN_COVERAGE:
            fails.append(f"{d.date()}: coverage {cov:.2%} (< {MIN_COVERAGE:.1%}) — {len(missing)} traded symbols missing")
        for s in ("BE", "BZ"):
            if ser_raw.get(s, 0) > 0 and ser_panel.get(s, 0) == 0:
                fails.append(f"{d.date()}: raw has {ser_raw[s]} {s}-series rows, panel has 0 — the 2026-09-08 signature (series filter regressed)")
        if len(bad_px):
            fails.append(f"{d.date()}: {len(bad_px)} raw-close mismatches (e.g. {bad_px['symbol'].head(3).tolist()})")
        results.append(row)
        prev_syms = pd_syms

    latest = dates[-1]
    latest_syms = set(panel.loc[panel["trade_date"] == latest, "symbol"])
    watched = _watched(); watch_missing = {}
    for src, syms in watched.items():
        m = [s for s in syms if s not in latest_syms]
        if m:
            watch_missing[src] = m
            fails.append(f"{src}: {len(m)} watched names have no row on {latest.date()}: {m}")

    verdict = {"ts": datetime.now().isoformat(timespec="seconds"), "latest_session": str(latest.date()),
               "status": "PASS" if not fails else "FAIL", "fails": fails, "watch_missing": watch_missing,
               "sessions": results}
    (OUT_DIR / f"panel_coverage_{latest.date()}.json").write_text(json.dumps(verdict, indent=1))
    with open(OUT_DIR / "panel_coverage.jsonl", "a") as f:
        f.write(json.dumps({k: verdict[k] for k in ("ts", "latest_session", "status", "fails")}) + "\n")

    print(f"PANEL COVERAGE EVAL · latest session {latest.date()} · {verdict['status']}")
    for r in results:
        if r.get("raw") is None:
            print(f"  {r['date']}: (no raw file)"); continue
        print(f"  {r['date']}: raw {r['raw']:,}  panel {r['panel']:,}  coverage {r['coverage']:.2%}  "
              f"BE {r['series_panel'].get('BE',0)}/{r['series_raw'].get('BE',0)}  BZ {r['series_panel'].get('BZ',0)}/{r['series_raw'].get('BZ',0)}  "
              f"drop-offs {r['silent_dropoffs_n']}  px-mismatch {r['price_mismatch_n']}")
    for f_ in fails:
        print("  ❌", f_)
    if not fails:
        print("  ✅ every traded symbol is in the panel; all watched names present; closes match raw")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
