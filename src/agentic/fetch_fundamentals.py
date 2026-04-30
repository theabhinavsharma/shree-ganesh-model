#!/usr/bin/env python3
"""
Enrich the universe with fundamentals ratios per stock from NSE.

Top-500 most-liquid symbols only (rank by avg_traded_value_20d in price parquet).
Per-symbol pulls quote-equity (PE, sector PE, face value, 52w hi/lo) and
corporates-financial-results quarterly (last 2 Q revenue / PAT) via stdlib urllib.

Append-only on (symbol, fetch_date) to data/derived/fundamentals_snapshot.parquet.
"""
from __future__ import annotations

import gzip
import http.cookiejar
import io
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "data" / "derived" / "stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data" / "derived" / "fundamentals_snapshot.parquet"

NSE_HOME = "https://www.nseindia.com"
QUOTE_REF = "https://www.nseindia.com/get-quotes/equity?symbol={sym}"
FIN_REF = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
QUOTE_URL = "https://www.nseindia.com/api/quote-equity?symbol={sym}"
FIN_URL = "https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={sym}&period=Quarterly"
# Returns the actual revenue/PAT figures (corporates-financial-results above only returns metadata).
FIN_FIGURES_URL = "https://www.nseindia.com/api/results-comparision?index=equities&symbol={sym}&period=Quarterly"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
COMMON_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}
DOC_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}
DELAY_SEC = 2.0
TOP_N = None  # None = full liquid EQ universe (no cap)
MIN_ADV_CR = 0.1  # ≥ ₹0.1cr/day average traded value (filters out ultra-illiquids)
TIMEOUT = 25
CHECKPOINT_EVERY = 25
LONG_BREAK_EVERY = 50  # every 50 calls, sleep 30s to evade NSE rate-limit
LONG_BREAK_SEC = 30
SESSION_REFRESH_EVERY = 100  # full session re-warm every 100 calls regardless
MAX_TIMEOUTS_BEFORE_BACKOFF = 3  # trigger backoff sooner
BACKOFF_SLEEP_SEC = 180  # longer backoff to clear throttle


def _build_opener() -> urllib.request.OpenerDirector:
    cj = http.cookiejar.CookieJar()
    ctx = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ctx),
    )


def _request(opener: urllib.request.OpenerDirector, url: str, *, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with opener.open(req, timeout=TIMEOUT) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw


def warm_session(opener: urllib.request.OpenerDirector, referer: str) -> None:
    """Hit a feature page so NSE mints the cookies needed for the API."""
    try:
        _request(opener, referer, headers=DOC_HEADERS)
    except urllib.error.HTTPError as exc:
        if exc.code != 403:
            raise
    except Exception:
        pass


def fetch_json(opener: urllib.request.OpenerDirector, url: str, referer: str, *, attempts: int = 2) -> Any:
    headers = dict(COMMON_HEADERS)
    headers["Referer"] = referer
    headers["Origin"] = NSE_HOME
    last: Exception | None = None
    for i in range(attempts):
        try:
            raw = _request(opener, url, headers=headers)
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                raise ValueError("empty response body")
            return json.loads(text)
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            # 401/403 → re-warm cookies and retry once
            warm_session(opener, referer)
            time.sleep(0.5)
    assert last is not None
    raise last


def get_top_symbols(n: int | None) -> list[str]:
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "avg_traded_value_20d", "series"])
    df = df[df["series"] == "EQ"]
    df = df.sort_values("trade_date").groupby("symbol", as_index=False).tail(1)
    df = df.dropna(subset=["avg_traded_value_20d"])
    df = df[df["avg_traded_value_20d"] >= MIN_ADV_CR * 1e7]
    df = df.sort_values("avg_traded_value_20d", ascending=False)
    if n is not None:
        df = df.head(n)
    return df["symbol"].astype(str).tolist()


def already_fetched_today(today: str) -> set[str]:
    """Resume-from-checkpoint: skip symbols already snapshot for today."""
    if not OUT.exists():
        return set()
    try:
        old = pd.read_parquet(OUT, columns=["symbol", "fetch_date"])
        old["fetch_date"] = pd.to_datetime(old["fetch_date"]).dt.date.astype(str)
        return set(old.loc[old["fetch_date"] == today, "symbol"].astype(str))
    except Exception:
        return set()


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            f = float(x)
            return f if f == f else None  # filter NaN
        except (TypeError, ValueError):
            return None
    s = str(x).strip()
    if not s or s.lower() in {"-", "na", "n/a", "nan", "none"}:
        return None
    s = s.replace(",", "").replace("\u20b9", "").strip()
    s = re.sub(r"[^0-9eE+\-.]", "", s)
    if not s or s in {"-", ".", "+", "-."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_quote(j: dict) -> dict:
    out: dict[str, Any] = {}
    metadata = j.get("metadata") or {}
    sec_info = j.get("securityInfo") or {}
    price_info = j.get("priceInfo") or {}
    out["pe"] = _to_float(metadata.get("pdSymbolPe"))
    out["sector_pe"] = _to_float(metadata.get("pdSectorPe"))
    out["face_value"] = _to_float(sec_info.get("faceValue"))
    week_hl = price_info.get("weekHighLow") or {}
    out["week52_high"] = _to_float(week_hl.get("max"))
    out["week52_low"] = _to_float(week_hl.get("min"))
    last_price = _to_float(price_info.get("lastPrice"))
    if last_price and out["week52_high"]:
        out["dist_from_52w_high_pct"] = round(
            (last_price - out["week52_high"]) / out["week52_high"] * 100.0, 4
        )
    else:
        out["dist_from_52w_high_pct"] = None
    if last_price and out["week52_low"]:
        out["dist_from_52w_low_pct"] = round(
            (last_price - out["week52_low"]) / out["week52_low"] * 100.0, 4
        )
    else:
        out["dist_from_52w_low_pct"] = None
    if out["pe"] and out["sector_pe"]:
        out["pe_vs_sector_ratio"] = round(out["pe"] / out["sector_pe"], 4)
    else:
        out["pe_vs_sector_ratio"] = None
    return out


# results-comparision schema: figures stored in paise/hundredths → divide by 100 for INR.
_REV_KEYS = ("re_net_sale", "re_income", "re_int_earned")
_PAT_KEYS = ("re_con_pro_loss", "re_proloss_ord_act", "re_net_pro_loss")


def _pick_first(d: dict, keys: tuple[str, ...]) -> Any:
    lower = {k.lower(): k for k in d.keys()}
    for key in keys:
        actual = lower.get(key.lower())
        if actual is not None:
            v = d.get(actual)
            if v not in (None, "", "-"):
                return v
    return None


def _parse_dt(s: str) -> str:
    # "31-DEC-2024" → "2024-12-31" for sortable comparisons
    if not s:
        return ""
    months = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
              "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}
    parts = s.upper().split("-")
    if len(parts) == 3 and parts[1] in months:
        return f"{parts[2]}-{months[parts[1]]}-{parts[0].zfill(2)}"
    return s


def parse_financials(j: Any) -> dict:
    """Parse results-comparision payload — top 2 quarters by re_to_dt."""
    out: dict[str, Any] = {
        "last_q_revenue": None,
        "last_q_pat": None,
        "prev_q_revenue": None,
        "prev_q_pat": None,
    }
    rows: list[dict] = []
    if isinstance(j, dict):
        cand = j.get("resCmpData") or j.get("data") or []
        if isinstance(cand, list):
            rows = [r for r in cand if isinstance(r, dict)]
    elif isinstance(j, list):
        rows = [r for r in j if isinstance(r, dict)]
    if not rows:
        return out

    rows = sorted(rows, key=lambda r: _parse_dt(str(r.get("re_to_dt", ""))), reverse=True)

    def _scaled(d: dict, keys: tuple[str, ...]) -> float | None:
        v = _to_float(_pick_first(d, keys))
        return v / 100.0 if v is not None else None  # NSE stores as INR * 100

    if len(rows) >= 1:
        out["last_q_revenue"] = _scaled(rows[0], _REV_KEYS)
        out["last_q_pat"] = _scaled(rows[0], _PAT_KEYS)
    if len(rows) >= 2:
        out["prev_q_revenue"] = _scaled(rows[1], _REV_KEYS)
        out["prev_q_pat"] = _scaled(rows[1], _PAT_KEYS)
    return out


def _growth(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None or prev == 0:
        return None
    try:
        return round((curr - prev) / abs(prev) * 100.0, 4)
    except (TypeError, ZeroDivisionError):
        return None


def fetch_one(opener: urllib.request.OpenerDirector, sym: str, today: str) -> dict | None:
    row: dict[str, Any] = {"symbol": sym, "fetch_date": today}
    encoded = urllib.parse.quote(sym, safe="")
    quote_ref = QUOTE_REF.format(sym=encoded)
    try:
        qj = fetch_json(opener, QUOTE_URL.format(sym=encoded), quote_ref)
        row.update(parse_quote(qj))
    except (TimeoutError, OSError) as exc:
        # propagate timeout so outer loop can trigger backoff + session re-warm
        print(f"  [{sym}] quote TIMEOUT: {str(exc)[:140]}")
        raise
    except Exception as exc:
        print(f"  [{sym}] quote ERR {type(exc).__name__}: {str(exc)[:140]}")
        return None

    try:
        fj = fetch_json(opener, FIN_FIGURES_URL.format(sym=encoded), FIN_REF)
        row.update(parse_financials(fj))
    except Exception as exc:
        print(f"  [{sym}] fin ERR {type(exc).__name__}: {str(exc)[:140]}")
        row.update({
            "last_q_revenue": None, "last_q_pat": None,
            "prev_q_revenue": None, "prev_q_pat": None,
        })

    row["qoq_revenue_growth"] = _growth(row.get("last_q_revenue"), row.get("prev_q_revenue"))
    row["qoq_pat_growth"] = _growth(row.get("last_q_pat"), row.get("prev_q_pat"))
    return row


COLUMNS = [
    "symbol", "fetch_date", "pe", "sector_pe", "pe_vs_sector_ratio",
    "face_value", "week52_high", "week52_low",
    "dist_from_52w_high_pct", "dist_from_52w_low_pct",
    "last_q_revenue", "last_q_pat", "prev_q_revenue", "prev_q_pat",
    "qoq_revenue_growth", "qoq_pat_growth",
]


def append_snapshot(rows: list[dict]) -> int:
    if not rows:
        return 0
    df_new = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df_new.columns:
            df_new[c] = None
    df_new = df_new[COLUMNS]
    df_new["fetch_date"] = pd.to_datetime(df_new["fetch_date"]).dt.date
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        old["fetch_date"] = pd.to_datetime(old["fetch_date"]).dt.date
        merged = pd.concat([old, df_new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["symbol", "fetch_date"], keep="last")
        merged.to_parquet(OUT, index=False)
        return len(merged) - len(old)
    df_new.to_parquet(OUT, index=False)
    return len(df_new)


def main() -> None:
    today = str(date.today())
    print(f"== fetch_fundamentals  fetch_date={today} ==")

    full_universe = get_top_symbols(TOP_N)
    done = already_fetched_today(today)
    symbols = [s for s in full_universe if s not in done]
    print(f"  universe: full liquid EQ (ADV >= ₹{MIN_ADV_CR}cr) → {len(full_universe)} symbols")
    print(f"  resume:   {len(done)} already fetched today → {len(symbols)} remaining")

    opener = _build_opener()
    warm_session(opener, "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE")
    warm_session(opener, FIN_REF)

    rows: list[dict] = []
    rate_limited = 0
    consecutive_timeouts = 0
    started = time.time()
    last_checkpoint_n = 0
    try:
        for i, sym in enumerate(symbols, 1):
            try:
                r = fetch_one(opener, sym, today)
                consecutive_timeouts = 0
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 429):
                    rate_limited += 1
                r = None
            except (TimeoutError, OSError) as exc:
                consecutive_timeouts += 1
                print(f"  [{sym}] timeout (#{consecutive_timeouts}): {str(exc)[:140]}", flush=True)
                r = None
            except Exception as exc:
                print(f"  [{sym}] FAIL {type(exc).__name__}: {str(exc)[:140]}", flush=True)
                r = None

            if r is not None and (r.get("pe") is not None or r.get("week52_high") is not None or r.get("face_value") is not None):
                rows.append(r)

            # adaptive backoff if NSE starts timing us out
            if consecutive_timeouts >= MAX_TIMEOUTS_BEFORE_BACKOFF:
                print(f"  [backoff] {consecutive_timeouts} consecutive timeouts → sleeping {BACKOFF_SLEEP_SEC}s + re-warming session", flush=True)
                time.sleep(BACKOFF_SLEEP_SEC)
                opener = _build_opener()
                warm_session(opener, "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE")
                warm_session(opener, FIN_REF)
                consecutive_timeouts = 0

            if i % CHECKPOINT_EVERY == 0:
                elapsed = time.time() - started
                eta_min = (len(symbols) - i) * (elapsed / i) / 60
                print(f"  progress {i}/{len(symbols)}  ok={len(rows)}  rate_limited={rate_limited}  elapsed={elapsed:.0f}s  eta={eta_min:.0f}min", flush=True)
                if len(rows) > last_checkpoint_n:
                    append_snapshot(rows)
                    last_checkpoint_n = len(rows)

            # periodic deeper sleep to dodge NSE rate-limit
            if i % SESSION_REFRESH_EVERY == 0:
                print(f"  [session-refresh] re-warming after {i} calls", flush=True)
                opener = _build_opener()
                warm_session(opener, "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE")
                warm_session(opener, FIN_REF)
                time.sleep(LONG_BREAK_SEC)
            elif i % LONG_BREAK_EVERY == 0:
                print(f"  [long-break] sleeping {LONG_BREAK_SEC}s after {i} calls", flush=True)
                time.sleep(LONG_BREAK_SEC)
            else:
                time.sleep(DELAY_SEC)
    except KeyboardInterrupt:
        print("  interrupted — flushing what we have", flush=True)

    delta = append_snapshot(rows)
    print(f"\n  fetched_ok={len(rows)}  appended_to_parquet={delta}  rate_limited_or_403={rate_limited}", flush=True)
    print(f"  output -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
