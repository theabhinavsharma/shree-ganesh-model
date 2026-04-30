"""Fetch stock-level FII/DII holdings from NSE quarterly shareholding pattern.

Endpoint: https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol=X
Returns JSON with quarter-end records + XBRL URLs containing:
  - Promoter & promoter-group % (pr_and_prgrp)
  - Public % (public_val)
  - Foreign portfolio investors (in XBRL)
  - Domestic institutional investors (in XBRL)

We fetch the master JSON (lightweight, ~100KB per symbol) and extract:
  symbol, quarter_end_date, promoter_pct, public_pct, fii_pct (from XBRL parse),
  dii_pct (from XBRL parse), submission_date

Output: data/derived/stock_shareholding.parquet

To compute the alpha signal:
  fii_pct_delta_qoq = current_fii_pct - prior_quarter_fii_pct
  Stocks with +1pp FII increase historically lead next-60-90d returns.
"""
from __future__ import annotations
import gzip
import http.cookiejar
import json
import re
import ssl
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/stock_shareholding.parquet"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

DELAY = 1.5
LONG_BREAK_EVERY = 50
LONG_BREAK_SEC = 25


def _opener():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )


def _get(opener, url: str) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    })
    with opener.open(req, timeout=20) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw


def warm(opener):
    for url in ["https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"]:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "gzip, deflate",
            })
            opener.open(req, timeout=10).read()
        except Exception:
            pass


def parse_xbrl(xbrl_bytes: bytes) -> dict:
    """Best-effort: extract FII / DII / Promoter / Public % from XBRL.

    XBRL structure varies; we look for known XBRL tag names referenced in
    NSE shareholding-pattern schema."""
    try:
        text = xbrl_bytes.decode("utf-8", errors="replace")
    except Exception:
        return {}
    out = {}
    # heuristic: look for percentages associated with category keywords
    patterns = [
        ("fii_pct",       r'(?i)foreign\s*portfolio\s*investor[^>]*>([0-9.]+)<'),
        ("fii_pct_alt",   r'(?i)foreign\s*institutional\s*investor[^>]*>([0-9.]+)<'),
        ("dii_pct",       r'(?i)domestic\s*institutional\s*investor[^>]*>([0-9.]+)<'),
        ("dii_pct_mf",    r'(?i)mutual\s*fund[^>]*>([0-9.]+)<'),
        ("promoter_pct",  r'(?i)total\s*promoter\s*group[^>]*>([0-9.]+)<'),
        ("public_pct",    r'(?i)total\s*public[^>]*>([0-9.]+)<'),
    ]
    for key, pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def fetch_one(opener, sym: str, parse_xbrl_files: bool = False) -> list[dict]:
    """Fetch master JSON for a symbol, return list of quarterly records."""
    url = f"https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={sym}"
    try:
        raw = _get(opener, url)
        records = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    rows = []
    for rec in records:
        promoter_pct = None
        public_pct = None
        try:
            promoter_pct = float(rec.get("pr_and_prgrp")) if rec.get("pr_and_prgrp") else None
            public_pct = float(rec.get("public_val")) if rec.get("public_val") else None
        except (ValueError, TypeError):
            pass

        fii_pct = None
        dii_pct = None
        if parse_xbrl_files and rec.get("xbrl"):
            try:
                xbrl_bytes = _get(opener, rec["xbrl"])
                parsed = parse_xbrl(xbrl_bytes)
                fii_pct = parsed.get("fii_pct") or parsed.get("fii_pct_alt")
                dii_pct = parsed.get("dii_pct") or parsed.get("dii_pct_mf")
            except Exception as e:
                pass

        # parse the date "31-MAR-2026"
        d = rec.get("date") or ""
        try:
            qe = pd.to_datetime(d, format="%d-%b-%Y")
        except Exception:
            qe = pd.to_datetime(d, errors="coerce")

        rows.append({
            "symbol": sym,
            "quarter_end": qe,
            "promoter_pct": promoter_pct,
            "public_pct": public_pct,
            "fii_pct": fii_pct,
            "dii_pct": dii_pct,
            "xbrl_url": rec.get("xbrl"),
            "submission_date": rec.get("submissionDate"),
        })
    return rows


def get_top_symbols(n: int) -> list[str]:
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "series", "avg_traded_value_20d"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    latest = df["trade_date"].max()
    snap = df[(df["trade_date"] == latest) & (df["series"] == "EQ")]
    snap = snap.sort_values("avg_traded_value_20d", ascending=False).head(n)
    return snap["symbol"].astype(str).tolist()


def main(top_n: int = 200, parse_xbrl_files: bool = False) -> None:
    print(f"== fetch_stock_fii_dii  top-{top_n}  parse_xbrl={parse_xbrl_files} ==")
    print(f"  endpoint: corporate-share-holdings-master (verified 2026-04-29)")
    syms = get_top_symbols(top_n)
    print(f"  fetching {len(syms)} symbols")
    opener = _opener()
    warm(opener)
    rows = []
    started = time.time()
    for i, sym in enumerate(syms, 1):
        try:
            recs = fetch_one(opener, sym, parse_xbrl_files=parse_xbrl_files)
            rows.extend(recs)
            if i <= 3 or i % 25 == 0:
                elapsed = time.time() - started
                eta = (len(syms) - i) * (elapsed / i) / 60
                print(f"  [{i}/{len(syms)}] {sym:<12} → {len(recs)} quarters  elapsed={elapsed:.0f}s  eta={eta:.0f}min")
        except Exception as e:
            print(f"  [{i}/{len(syms)}] {sym} FAIL {type(e).__name__}: {str(e)[:120]}")
        if i % LONG_BREAK_EVERY == 0:
            print(f"    [break] {LONG_BREAK_SEC}s ...")
            time.sleep(LONG_BREAK_SEC)
        else:
            time.sleep(DELAY)

    if not rows:
        print("no rows")
        return

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["quarter_end"])
    df = df.sort_values(["symbol", "quarter_end"])
    # compute QoQ deltas if we have at least 2 quarters per symbol
    df["promoter_qoq"] = df.groupby("symbol")["promoter_pct"].diff()
    df["public_qoq"] = df.groupby("symbol")["public_pct"].diff()
    if "fii_pct" in df.columns:
        df["fii_qoq"] = df.groupby("symbol")["fii_pct"].diff()
    if "dii_pct" in df.columns:
        df["dii_qoq"] = df.groupby("symbol")["dii_pct"].diff()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        old["quarter_end"] = pd.to_datetime(old["quarter_end"])
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(["symbol", "quarter_end"], keep="last").sort_values(["symbol", "quarter_end"])
    else:
        merged = df
    merged.to_parquet(OUT, index=False)

    print(f"\nwrote {OUT}: {len(merged):,} rows ({merged['symbol'].nunique()} symbols × ~{len(merged)/max(1,merged['symbol'].nunique()):.0f} quarters)")
    print(f"\nlatest 10 with promoter_qoq + public_qoq filled:")
    latest = merged.dropna(subset=["promoter_qoq", "public_qoq"]).sort_values("quarter_end", ascending=False)
    print(latest[["symbol", "quarter_end", "promoter_pct", "public_pct", "promoter_qoq", "public_qoq"]].head(10).to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--parse-xbrl", action="store_true",
                    help="Also download + parse XBRL per quarter (3-4x slower, gives FII/DII%)")
    args = ap.parse_args()
    main(top_n=args.top_n, parse_xbrl_files=args.parse_xbrl)
