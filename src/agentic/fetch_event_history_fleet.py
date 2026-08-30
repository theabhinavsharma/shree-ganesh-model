"""EVENT-HISTORY FLEET — the deep-history feeds for 10-year event studies.

Feeds (each checkpointed, resumable, polite):
  results_calendar : corporates-financial-results per symbol — every quarterly
                     filing's dates/links back to ~2005. One call per symbol.
                     → data/derived/results_calendar_history.parquet
  shareholding     : corporate-share-holdings-master per symbol — ~22 quarters of
                     promoter/public %. One call per symbol.
                     → data/derived/stock_shareholding.parquet
  pit              : corporates-pit in monthly windows 2019-01 → 2026-02 (live
                     store covers Feb-2026+). → data/derived/pit_history.parquet

Usage: python3 fetch_event_history_fleet.py [results_calendar|shareholding|pit|all]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/agentic"))
from src.ingest.nse.session import build_session  # noqa: E402
from src.ingest.nse.api import get_json  # noqa: E402

RAW = ROOT / "data/raw/event_history"
RAW.mkdir(parents=True, exist_ok=True)
SLEEP = 2.0


def universe() -> list[str]:
    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                         columns=["symbol", "trade_date", "avg_traded_value_20d"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    recent = px[px["trade_date"] >= px["trade_date"].max() - pd.Timedelta(days=400)]
    adv = recent.groupby("symbol")["avg_traded_value_20d"].max() / 1e7
    return sorted(adv[adv >= 2].index)   # every name that was ever near-investable


def run_symbol_feed(tag: str, url_fmt: str, parse) -> None:
    ckpt = RAW / f"_{tag}_done.jsonl"
    done = set()
    if ckpt.exists():
        done = {json.loads(l)["symbol"] for l in ckpt.read_text().splitlines() if l.strip()}
    syms = [s for s in universe() if s not in done]
    print(f"[{tag}] {len(syms)} symbols remaining (of {len(syms) + len(done)})", flush=True)
    s = build_session()
    shard_dir = RAW / tag
    shard_dir.mkdir(exist_ok=True)
    for i, sym in enumerate(syms):
        try:
            j = get_json(s, url_fmt.format(sym=sym))
            df = parse(sym, j)
            if df is not None and len(df):
                df.to_parquet(shard_dir / f"{sym}.parquet", index=False)
            with ckpt.open("a") as f:
                f.write(json.dumps({"symbol": sym, "rows": 0 if df is None else len(df)}) + "\n")
        except Exception as e:
            print(f"  [{sym}] ERROR {str(e)[:80]}", flush=True)
            s = build_session()   # refresh cookies and keep going
        if i % 50 == 0:
            print(f"  [{tag}] {i}/{len(syms)}", flush=True)
        time.sleep(SLEEP)
    # consolidate
    shards = list(shard_dir.glob("*.parquet"))
    if shards:
        out = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
        dest_name = {"results_calendar": "results_calendar_history",
                     "shareholding": "stock_shareholding",
                     "integrated": "results_calendar_integrated"}[tag]
        dest = ROOT / f"data/derived/{dest_name}.parquet"
        out.to_parquet(dest, index=False)
        print(f"[{tag}] consolidated {len(out):,} rows → {dest.name}", flush=True)


def parse_results(sym: str, j) -> pd.DataFrame | None:
    if not isinstance(j, list) or not j:
        return None
    rows = [{
        "symbol": sym,
        "period_from": r.get("fromDate"), "period_to": r.get("toDate"),
        "relating_to": r.get("relatingTo"), "filing_date": r.get("filingDate"),
        "broadcast": r.get("broadCastDate"), "exch_diss_time": r.get("exchdisstime"),
        "audited": r.get("audited"), "consolidated": r.get("consolidated"),
        "detail_link": r.get("resultDetailedDataLink"), "seq": r.get("seqNumber"),
    } for r in j]
    return pd.DataFrame(rows)


def parse_shp(sym: str, j) -> pd.DataFrame | None:
    import fetch_stock_fii_dii as shp_mod  # reuse its record extraction
    try:
        recs = shp_mod.extract_records(sym, j) if hasattr(shp_mod, "extract_records") else None
    except Exception:
        recs = None
    if recs is None:
        # fallback: fetch_one-equivalent local parse of the master json
        data = j.get("data", j) if isinstance(j, dict) else j
        if not isinstance(data, list):
            return None
        rows = []
        for r in data:
            rows.append({"symbol": sym, "quarter_end": r.get("date") or r.get("qtrEndDate"),
                         "promoter_pct": r.get("pr_and_prgrp"), "public_pct": r.get("public_val"),
                         "submission": r.get("submissionDate"), "xbrl": r.get("xbrl")})
        return pd.DataFrame(rows)
    return pd.DataFrame(recs)


def parse_integrated(sym: str, j) -> pd.DataFrame | None:
    d = j if isinstance(j, list) else (j.get("data", []) if isinstance(j, dict) else [])
    if not d:
        return None
    rows = [{
        "symbol": sym,
        "period_to": r.get("qe_Date") or r.get("period_ended") or r.get("toDate"),
        "relating_to": r.get("relatingTo"), "broadcast": r.get("broadcast_Date"),
        "audited": r.get("audited"), "consolidated": r.get("consolidated"),
        "detail_link": r.get("xbrl") or r.get("attchmntFile"), "source": "integrated",
    } for r in d]
    return pd.DataFrame(rows)


def run_pit() -> None:
    ckpt = RAW / "_pit_done.jsonl"
    done = {json.loads(l)["window"] for l in ckpt.read_text().splitlines()} if ckpt.exists() else set()
    months = pd.period_range("2019-01", "2026-02", freq="M")
    s = build_session()
    shard_dir = RAW / "pit"
    shard_dir.mkdir(exist_ok=True)
    for m in months:
        w = str(m)
        if w in done:
            continue
        frm = m.start_time.strftime("%d-%m-%Y")
        to = m.end_time.strftime("%d-%m-%Y")
        try:
            j = get_json(s, f"https://www.nseindia.com/api/corporates-pit?index=equities&from_date={frm}&to_date={to}")
            d = j.get("data", []) if isinstance(j, dict) else (j or [])
            if d:
                pd.DataFrame(d).to_parquet(shard_dir / f"{w}.parquet", index=False)
            with ckpt.open("a") as f:
                f.write(json.dumps({"window": w, "rows": len(d)}) + "\n")
            print(f"  [pit] {w}: {len(d)} rows", flush=True)
        except Exception as e:
            print(f"  [pit] {w} ERROR {str(e)[:80]}", flush=True)
            s = build_session()
        time.sleep(SLEEP + 1)
    shards = list(shard_dir.glob("*.parquet"))
    if shards:
        out = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
        out.to_parquet(ROOT / "data/derived/pit_history.parquet", index=False)
        print(f"[pit] consolidated {len(out):,} rows", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("results_calendar", "all"):
        run_symbol_feed("results_calendar",
                        "https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={sym}&period=Quarterly",
                        parse_results)
    if which in ("shareholding", "all"):
        run_symbol_feed("shareholding",
                        "https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={sym}",
                        parse_shp)
    if which in ("integrated", "all"):
        run_symbol_feed("integrated",
                        "https://www.nseindia.com/api/integrated-filing-results?index=equities&symbol={sym}&period_ended=Quarterly",
                        parse_integrated)
    if which in ("pit", "all"):
        run_pit()
    print("FLEET DONE", flush=True)
