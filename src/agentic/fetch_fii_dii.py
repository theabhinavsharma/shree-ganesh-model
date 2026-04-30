"""Fetch FII / DII daily provisional flows from NSE.

Endpoint: https://www.nseindia.com/api/fiidiiTradeReact

This is the holy-grail India macro indicator — published end-of-day.
NSE serves a JSON list with last ~30 days. We fetch + append to parquet.

Output: data/derived/fii_dii_flows.parquet
  cols: trade_date, fii_buy_inr_cr, fii_sell_inr_cr, fii_net_inr_cr,
        dii_buy_inr_cr, dii_sell_inr_cr, dii_net_inr_cr,
        category (FII/DII)
"""
from __future__ import annotations
import gzip
import http.cookiejar
import json
import ssl
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/fii_dii_flows.parquet"

NSE_HOME = "https://www.nseindia.com"
ENDPOINT = "https://www.nseindia.com/api/fiidiiTradeReact"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")


def _opener():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )


def _get(opener, url, *, accept="application/json"):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": accept,
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with opener.open(req, timeout=20) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw


def warm(opener):
    try:
        _get(opener, NSE_HOME, accept="text/html,application/xhtml+xml")
    except Exception:
        pass
    try:
        _get(opener, "https://www.nseindia.com/reports/fii-dii", accept="text/html,application/xhtml+xml")
    except Exception:
        pass


def parse_payload(j) -> list[dict]:
    """NSE returns either a list of {category, date, buyValue, sellValue, netValue}
    or a dict with `data` key. Handle both."""
    if isinstance(j, dict):
        items = j.get("data") or j.get("FII_DII") or []
    elif isinstance(j, list):
        items = j
    else:
        return []
    rows = []
    for it in items:
        cat = (it.get("category") or it.get("type") or "").upper()
        d = it.get("date") or it.get("Date") or ""
        try:
            buy = float(it.get("buyValue") or it.get("buy") or 0)
            sell = float(it.get("sellValue") or it.get("sell") or 0)
            net = float(it.get("netValue") or it.get("net") or (buy - sell))
        except (TypeError, ValueError):
            continue
        # parse date (DD-MMM-YYYY)
        try:
            ts = pd.to_datetime(d, format="%d-%b-%Y", errors="coerce")
        except Exception:
            ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append({"trade_date": ts, "category": cat,
                     "buy_inr_cr": buy, "sell_inr_cr": sell, "net_inr_cr": net})
    return rows


def main() -> None:
    print(f"== fetch_fii_dii ==")
    op = _opener()
    warm(op)
    try:
        raw = _get(op, ENDPOINT, accept="application/json")
        j = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {str(exc)[:140]}")
        return

    rows = parse_payload(j)
    if not rows:
        print(f"  no data parsed; payload keys={list(j.keys()) if isinstance(j, dict) else 'list'}")
        return

    df = pd.DataFrame(rows)
    # pivot: per trade_date one row with FII + DII numbers
    pivot = df.pivot_table(index="trade_date", columns="category",
                           values=["buy_inr_cr", "sell_inr_cr", "net_inr_cr"]).reset_index()
    pivot.columns = ["_".join([c for c in col if c]).strip("_").lower() for col in pivot.columns]
    pivot.columns = [c.replace(" ", "_") for c in pivot.columns]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        old["trade_date"] = pd.to_datetime(old["trade_date"])
        merged = pd.concat([old, pivot], ignore_index=True)
        merged = merged.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    else:
        merged = pivot
    merged.to_parquet(OUT, index=False)
    print(f"  fetched {len(pivot)} new days, total {len(merged)} days in parquet")
    print(f"  columns: {merged.columns.tolist()}")
    if len(merged):
        print(f"  latest 5:")
        print(merged.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
