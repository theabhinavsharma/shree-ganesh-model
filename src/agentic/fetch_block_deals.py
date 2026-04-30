#!/usr/bin/env python3
"""
Fetch NSE block deals + bulk deals and turn them into a per-stock smart-money signal.

Outputs:
  data/derived/block_deals.parquet      append-only normalized deal rows
  data/derived/block_features.parquet   per-(symbol, trade_date) rolling features

Constraints:
  - stdlib + pandas + urllib only (no `requests`)
  - graceful on endpoint blocks (NSE 401/403/timeout) -- writes nothing new, exits 0

Usage:
  /usr/bin/python3 src/agentic/fetch_block_deals.py
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import time
import zlib
from datetime import date, datetime, timedelta
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DEALS = REPO_ROOT / "data" / "derived" / "block_deals.parquet"
OUT_FEATS = REPO_ROOT / "data" / "derived" / "block_features.parquet"
PRICES_PARQUET = REPO_ROOT / "data" / "derived" / "stock_daily_facts_adjusted_2015plus.parquet"

LARGE_DEALS_REF = "https://www.nseindia.com/market-data/large-deals"
# NSE has shuffled these paths over time; try each in order until one yields data.
BLOCK_API_CANDIDATES = (
    "https://www.nseindia.com/api/block-deal",
    "https://www.nseindia.com/api/snapshot-capital-market-largedeal",
    "https://www.nseindia.com/api/historical/securities/block-deals?from={frm}&to={to}",
)
BULK_API_CANDIDATES = (
    "https://www.nseindia.com/api/historical/cm/bulk?from={frm}&to={to}",
    "https://www.nseindia.com/api/historical/securities/bulk-deals?from={frm}&to={to}",
    "https://www.nseindia.com/api/historical/securities/bulk_deals_data?from={frm}&to={to}",
    "https://www.nseindia.com/api/historical/cm/bulk_deals_data?from={frm}&to={to}",
    "https://www.nseindia.com/api/snapshot-capital-market-largedeal",
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


def _build_opener():
    cj = CookieJar()
    return build_opener(HTTPCookieProcessor(cj))


def _decode(resp) -> bytes:
    raw = resp.read()
    enc = (resp.headers.get("Content-Encoding") or "").lower()
    if enc == "gzip":
        return gzip.decompress(raw)
    if enc == "deflate":
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw


def _doc_headers(referer: str | None = None) -> dict[str, str]:
    h = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    }
    if referer:
        h["Referer"] = referer
    return h


def _api_headers(referer: str) -> dict[str, str]:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Referer": referer,
        "Origin": "https://www.nseindia.com",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "X-Requested-With": "XMLHttpRequest",
    }


def warm(opener) -> None:
    """Mint cookies via the large-deals page."""
    # First touch home page (best-effort), then large-deals.
    for url in ("https://www.nseindia.com/", LARGE_DEALS_REF):
        try:
            req = Request(url, headers=_doc_headers())
            opener.open(req, timeout=20).read(2048)
        except (HTTPError, URLError):
            pass
        time.sleep(0.4)


def fetch_json(opener, url: str, *, referer: str = LARGE_DEALS_REF, attempts: int = 3):
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            req = Request(url, headers=_api_headers(referer))
            with opener.open(req, timeout=30) as resp:
                body = _decode(resp)
            text = body.decode("utf-8", errors="replace").strip()
            if not text:
                return None
            return json.loads(text)
        except HTTPError as e:
            last_err = e
            if e.code in (401, 403):
                warm(opener)
                time.sleep(0.6 * (i + 1))
                continue
            if e.code >= 500:
                time.sleep(0.8 * (i + 1))
                continue
            break
        except (URLError, json.JSONDecodeError, TimeoutError) as e:
            last_err = e
            time.sleep(0.8 * (i + 1))
    print(f"  [warn] fetch failed {url}: {str(last_err)[:140]}", file=sys.stderr)
    return None


def _f(x) -> float | None:
    if x is None:
        return None
    s = str(x).replace(",", "").strip()
    if not s or s in {"-", "NA", "N/A"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(s) -> pd.Timestamp | None:
    if s is None:
        return None
    s = str(s).strip()
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return None


def _pick(rec: dict, *names) -> object:
    """Case-insensitive multi-name lookup."""
    lc = {str(k).lower(): v for k, v in rec.items()}
    for n in names:
        v = rec.get(n)
        if v not in (None, ""):
            return v
        v = lc.get(n.lower())
        if v not in (None, ""):
            return v
    return None


def _normalize_record(rec: dict, deal_type: str) -> dict | None:
    """Map an NSE block/bulk JSON row to our normalized schema."""
    # Observed NSE field families:
    #   historical: BD_DT_DATE, BD_SYMBOL, BD_CLIENT_NAME, BD_BUY_SELL, BD_QTY_TRD, BD_TP_WATP
    #   snapshot:   date, symbol, clientName, buySell, qty, tradePrice (or wap)
    sym = _pick(rec, "BD_SYMBOL", "symbol", "SYMBOL")
    if not sym:
        return None
    dt_raw = _pick(rec, "BD_DT_DATE", "date", "tradeDate", "mkt")
    dt = _parse_date(dt_raw)
    if dt is None or pd.isna(dt):
        return None
    bs_raw = str(_pick(rec, "BD_BUY_SELL", "buySell", "buy_sell") or "").upper().strip()
    if bs_raw.startswith("B"):
        bs = "BUY"
    elif bs_raw.startswith("S"):
        bs = "SELL"
    else:
        return None
    qty = _f(_pick(rec, "BD_QTY_TRD", "qty", "quantity"))
    px = _f(_pick(rec, "BD_TP_WATP", "BD_PR_TRD", "tradePrice", "watp", "wap", "price"))
    client_raw = _pick(rec, "BD_CLIENT_NAME", "clientName", "client_name")
    client = str(client_raw or "").strip()
    if qty is None or px is None:
        return None
    return {
        "trade_date": pd.Timestamp(dt.date()),
        "symbol": str(sym).strip().upper(),
        "deal_type": deal_type,
        "client_name": client,
        "buy_sell": bs,
        "qty": qty,
        "traded_price": px,
        "value_inr": qty * px,
    }


def _extract_rows(j, prefer: str | None = None) -> list[dict]:
    """Walk top-level keys looking for a list of dicts (NSE wraps payloads variably).

    `prefer` selects between snapshot sections: "block" -> BLOCK_DEALS_DATA, "bulk" -> BULK_DEALS_DATA.
    """
    if isinstance(j, list):
        return [r for r in j if isinstance(r, dict)]
    if isinstance(j, dict):
        # Snapshot endpoint shape: {"BLOCK_DEALS_DATA": {"data": [...]}, "BULK_DEALS_DATA": {...}, ...}
        preferred_keys = []
        if prefer == "block":
            preferred_keys = ["BLOCK_DEALS_DATA"]
        elif prefer == "bulk":
            preferred_keys = ["BULK_DEALS_DATA"]
        keys = preferred_keys + ["data", "BLOCK_DEALS_DATA", "BULK_DEALS_DATA", "rows", "as_on_date"]
        for key in keys:
            val = j.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
            if isinstance(val, dict):
                inner = val.get("data")
                if isinstance(inner, list):
                    return [r for r in inner if isinstance(r, dict)]
        # Fallback: any value that's a list of dicts (only if no preference, to avoid mixing sections)
        if prefer is None:
            for v in j.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
                if isinstance(v, dict):
                    inner = v.get("data")
                    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                        return inner
    return []


def fetch_block_deals(opener) -> list[dict]:
    today = date.today()
    start = today - timedelta(days=14)
    ctx = {"frm": start.strftime("%d-%m-%Y"), "to": today.strftime("%d-%m-%Y")}
    for tmpl in BLOCK_API_CANDIDATES:
        url = tmpl.format(**ctx)
        j = fetch_json(opener, url)
        rows = _extract_rows(j, prefer="block")
        if rows:
            print(f"  block source: {url} ({len(rows)} raw)")
            out = [nr for r in rows if (nr := _normalize_record(r, "block"))]
            if not out and rows:
                print(f"  [debug] block sample keys: {sorted(rows[0].keys())[:12]}")
            if out:
                return out
    return []


def fetch_bulk_deals(opener, days: int = 14) -> list[dict]:
    today = date.today()
    start = today - timedelta(days=days)
    ctx = {"frm": start.strftime("%d-%m-%Y"), "to": today.strftime("%d-%m-%Y")}
    for tmpl in BULK_API_CANDIDATES:
        url = tmpl.format(**ctx)
        j = fetch_json(opener, url)
        rows = _extract_rows(j, prefer="bulk")
        if rows:
            print(f"  bulk  source: {url} ({len(rows)} raw)")
            out = [nr for r in rows if (nr := _normalize_record(r, "bulk"))]
            if not out and rows:
                print(f"  [debug] bulk sample keys: {sorted(rows[0].keys())[:12]}")
            if out:
                return out
    return []


def merge_deals(new_rows: list[dict]) -> pd.DataFrame:
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        if OUT_DEALS.exists():
            return pd.read_parquet(OUT_DEALS)
        return df_new
    if OUT_DEALS.exists():
        old = pd.read_parquet(OUT_DEALS)
        merged = pd.concat([old, df_new], ignore_index=True)
    else:
        merged = df_new
    # Dedup on symbol + trade_date + client_name + qty (per spec)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.normalize()
    merged = merged.drop_duplicates(
        subset=["symbol", "trade_date", "client_name", "qty"],
        keep="first",
    ).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    OUT_DEALS.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_DEALS, index=False)
    return merged


def build_features(deals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if deals.empty:
        return pd.DataFrame()

    deals = deals.copy()
    deals["trade_date"] = pd.to_datetime(deals["trade_date"]).dt.normalize()
    prices = prices.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.normalize()

    grid = (
        prices[["symbol", "trade_date"]]
        .drop_duplicates()
        .sort_values(["symbol", "trade_date"])
        .reset_index(drop=True)
    )

    # Aggregate deals to symbol+date+side
    agg = (
        deals.groupby(["symbol", "trade_date", "buy_sell"], as_index=False)
        .agg(value_inr=("value_inr", "sum"), client_name=("client_name", "nunique"))
    )
    buy = agg[agg.buy_sell == "BUY"][["symbol", "trade_date", "value_inr"]].rename(
        columns={"value_inr": "buy_v"}
    )
    sell = agg[agg.buy_sell == "SELL"][["symbol", "trade_date", "value_inr"]].rename(
        columns={"value_inr": "sell_v"}
    )

    # distinct buyer count per symbol+date (raw client_name distinct)
    buyer_distinct = (
        deals[deals.buy_sell == "BUY"]
        .groupby(["symbol", "trade_date"], as_index=False)["client_name"]
        .agg(lambda s: set(c for c in s if c))
        .rename(columns={"client_name": "buy_clients"})
    )

    # Restrict grid to symbols that ever appear, plus a 60-trading-day tail per symbol
    relevant_syms = set(deals.symbol.unique())
    grid = grid[grid.symbol.isin(relevant_syms)].copy()
    if grid.empty:
        return pd.DataFrame()

    # Keep only last 60 trading days per symbol to bound work
    grid = grid.groupby("symbol", group_keys=False).tail(120).reset_index(drop=True)

    df = grid.merge(buy, on=["symbol", "trade_date"], how="left") \
             .merge(sell, on=["symbol", "trade_date"], how="left") \
             .merge(buyer_distinct, on=["symbol", "trade_date"], how="left")
    df["buy_v"] = df["buy_v"].fillna(0.0)
    df["sell_v"] = df["sell_v"].fillna(0.0)
    df["buy_clients"] = df["buy_clients"].apply(lambda x: x if isinstance(x, set) else set())

    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["block_buy_5d_inr"] = g["buy_v"].rolling(5, min_periods=1).sum()
        g["block_sell_5d_inr"] = g["sell_v"].rolling(5, min_periods=1).sum()
        g["block_net_5d_inr"] = g["block_buy_5d_inr"] - g["block_sell_5d_inr"]
        g["block_buy_30d_inr"] = g["buy_v"].rolling(30, min_periods=1).sum()
        g["block_sell_30d_inr"] = g["sell_v"].rolling(30, min_periods=1).sum()
        g["block_net_30d_inr"] = g["block_buy_30d_inr"] - g["block_sell_30d_inr"]
        # distinct buyers in trailing 30 trading days (union of daily client sets)
        clients = g["buy_clients"].tolist()
        out = []
        for i in range(len(clients)):
            lo = max(0, i - 29)
            u: set = set()
            for s in clients[lo : i + 1]:
                u |= s
            out.append(len(u))
        g["distinct_buyers_30d"] = out
        return g

    # Apply rolling-window features per symbol. Use group_keys=True so symbol is preserved
    # in the index, then reset to a column. Avoid include_groups (newer pandas only).
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        gg = _per_symbol(g)
        gg["symbol"] = sym
        parts.append(gg)
    feats = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=df.columns)
    keep = [
        "symbol", "trade_date",
        "block_buy_5d_inr", "block_sell_5d_inr", "block_net_5d_inr",
        "block_buy_30d_inr", "block_sell_30d_inr", "block_net_30d_inr",
        "distinct_buyers_30d",
    ]
    feats = feats[keep].reset_index(drop=True)

    # Only keep rows where there's any activity in 30d window (compact output)
    nonzero = (feats["block_buy_30d_inr"] > 0) | (feats["block_sell_30d_inr"] > 0)
    feats = feats[nonzero].reset_index(drop=True)

    OUT_FEATS.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT_FEATS, index=False)
    return feats


def main() -> int:
    print("== fetch_block_deals ==")
    opener = _build_opener()
    warm(opener)

    block_rows = fetch_block_deals(opener)
    print(f"  block deals fetched: {len(block_rows)}")
    bulk_rows = fetch_bulk_deals(opener, days=14)
    print(f"  bulk  deals fetched: {len(bulk_rows)}")
    new_rows = block_rows + bulk_rows

    if not new_rows and not OUT_DEALS.exists():
        print("  [warn] endpoint blocked and no prior parquet — exiting cleanly")
        return 0

    deals = merge_deals(new_rows)
    print(f"  deals parquet rows: {len(deals)}  -> {OUT_DEALS}")

    if not PRICES_PARQUET.exists():
        print(f"  [warn] missing prices parquet at {PRICES_PARQUET}; skipping features")
        return 0
    prices = pd.read_parquet(PRICES_PARQUET, columns=["symbol", "trade_date"])
    feats = build_features(deals, prices)
    print(f"  feature rows: {len(feats)}  -> {OUT_FEATS}")

    # Sample preview
    if not deals.empty:
        print("\n  sample deals:")
        print(deals.tail(5).to_string(index=False))
    if not feats.empty:
        print("\n  sample features:")
        print(feats.tail(5).to_string(index=False))

    if len(deals) < 10:
        print(f"  [warn] fewer than 10 deals captured (have {len(deals)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
