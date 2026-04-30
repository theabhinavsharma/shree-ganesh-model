#!/usr/bin/python3
"""
Fetch NSE F&O options chain data for each F&O underlying and write per-stock
IV / OI / PCR / max-pain features to data/derived/options_features.parquet.

Strategy:
  1. Warm up an NSE session by GETting the option-chain landing page (sets cookies).
  2. List F&O underlyings via /api/equity-stockIndices?index=SECURITIES%20IN%20F%26O.
  3. For each symbol (cap ~150), call /api/option-chain-equities?symbol=<SYM>,
     compute features off the nearest expiry, and append to the parquet.
  4. NSE is fragile from scripts: if cookies expire or the API blocks, log and
     continue — daily_pipeline.sh must NOT abort because of this fetcher.

stdlib + pandas only (urllib, NOT requests).
"""
from __future__ import annotations

import gzip
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path("data/derived/options_features.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

NSE_HOME = "https://www.nseindia.com"
WARMUP_URL = "https://www.nseindia.com/option-chain"
FNO_LIST_URL = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-equities?symbol={sym}"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
DOC_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
API_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Referer": WARMUP_URL,
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-Requested-With": "XMLHttpRequest",
}

REQUEST_DELAY_SEC = 1.5
MAX_SYMBOLS = 150
TIMEOUT = 25


def build_opener() -> urllib.request.OpenerDirector:
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def _read(opener: urllib.request.OpenerDirector, url: str, headers: dict) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with opener.open(req, timeout=TIMEOUT) as resp:
        raw = resp.read()
        enc = resp.headers.get("Content-Encoding", "").lower()
        if enc == "gzip" or (raw[:2] == b"\x1f\x8b"):
            raw = gzip.decompress(raw)
    return raw


WARMUP_SEQUENCE = [
    NSE_HOME,
    "https://www.nseindia.com/market-data/equity-derivatives-watch",
    WARMUP_URL,
]


def warmup(opener: urllib.request.OpenerDirector) -> bool:
    """Mint cookies by walking the same sequence a browser would: home → derivs → option-chain."""
    ok_any = False
    for url in WARMUP_SEQUENCE:
        try:
            _read(opener, url, DOC_HEADERS)
            ok_any = True
            time.sleep(0.4)
        except Exception as e:
            # NSE often 403s the bare home for scripted clients but still mints cookies.
            print(f"  warmup soft-fail {url[:60]}: {str(e)[:120]}")
    return ok_any


def fetch_json(opener: urllib.request.OpenerDirector, url: str, retries: int = 1) -> dict | None:
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = _read(opener, url, API_HEADERS)
            return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (401, 403):
                # cookies probably stale — re-warm and retry once
                warmup(opener)
        except Exception as e:
            last_err = str(e)[:120]
        if attempt < retries:
            time.sleep(REQUEST_DELAY_SEC)
    print(f"    fetch ERR ({last_err}) {url[:90]}")
    return None


def list_fno_underlyings(opener: urllib.request.OpenerDirector) -> list[str]:
    j = fetch_json(opener, FNO_LIST_URL)
    if not j:
        return []
    rows = j.get("data", []) or []
    syms = []
    for r in rows:
        s = r.get("symbol")
        if not s:
            continue
        # Skip the synthetic index row (first element is usually the index summary)
        if s.upper() in {"NIFTY 50", "NIFTY", "BANKNIFTY", "FINNIFTY"}:
            continue
        syms.append(s)
    # de-dupe preserve order
    seen, ordered = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _f(x) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def compute_features(symbol: str, payload: dict, trade_dt: date) -> dict | None:
    records = (payload.get("records") or {})
    filtered = (payload.get("filtered") or {})
    underlying = _f(records.get("underlyingValue"))
    if underlying is None or underlying <= 0:
        return None

    # Pick nearest expiry
    expiry_dates = records.get("expiryDates") or []
    if not expiry_dates:
        return None
    target_expiry = expiry_dates[0]

    chain = records.get("data") or []
    rows = [r for r in chain if r.get("expiryDate") == target_expiry]
    if not rows:
        return None

    strikes = []
    total_call_oi = total_put_oi = 0.0
    total_call_vol = total_put_vol = 0.0
    oi_by_strike: dict[float, float] = {}

    for r in rows:
        strike = _f(r.get("strikePrice"))
        if strike is None:
            continue
        ce = r.get("CE") or {}
        pe = r.get("PE") or {}
        c_oi = _f(ce.get("openInterest")) or 0.0
        p_oi = _f(pe.get("openInterest")) or 0.0
        c_vol = _f(ce.get("totalTradedVolume")) or 0.0
        p_vol = _f(pe.get("totalTradedVolume")) or 0.0
        c_iv = _f(ce.get("impliedVolatility"))
        p_iv = _f(pe.get("impliedVolatility"))

        total_call_oi += c_oi
        total_put_oi += p_oi
        total_call_vol += c_vol
        total_put_vol += p_vol
        oi_by_strike[strike] = oi_by_strike.get(strike, 0.0) + c_oi + p_oi

        strikes.append({
            "strike": strike, "c_iv": c_iv, "p_iv": p_iv,
            "c_oi": c_oi, "p_oi": p_oi,
        })

    if not strikes:
        return None

    # ATM strike = closest to underlying
    atm = min(strikes, key=lambda x: abs(x["strike"] - underlying))
    atm_iv_vals = [v for v in (atm["c_iv"], atm["p_iv"]) if v and v > 0]
    atm_iv = sum(atm_iv_vals) / len(atm_iv_vals) if atm_iv_vals else None

    # Approximate 25-delta wings: put ~10% OTM (strike below underlying),
    # call ~10% OTM (strike above). Find nearest available.
    def nearest_with_iv(target: float, side: str) -> float | None:
        cand = [s for s in strikes if (s["p_iv"] if side == "put" else s["c_iv"])]
        if not cand:
            return None
        best = min(cand, key=lambda s: abs(s["strike"] - target))
        return best["p_iv"] if side == "put" else best["c_iv"]

    put_wing = nearest_with_iv(underlying * 0.90, "put")
    call_wing = nearest_with_iv(underlying * 1.10, "call")
    iv_skew = (put_wing - call_wing) if (put_wing and call_wing) else None

    pcr_oi = (total_put_oi / total_call_oi) if total_call_oi > 0 else None
    pcr_vol = (total_put_vol / total_call_vol) if total_call_vol > 0 else None

    # Max pain: strike where total OI is max (proxy: pin where most OI sits;
    # a fully proper max-pain weights by intrinsic loss but this OI-peak proxy
    # is the common quick indicator)
    max_pain_strike = max(oi_by_strike.items(), key=lambda kv: kv[1])[0] if oi_by_strike else None
    mp_dist_pct = (
        100.0 * (max_pain_strike - underlying) / underlying
        if max_pain_strike is not None else None
    )

    return {
        "symbol": symbol,
        "trade_date": pd.Timestamp(trade_dt),
        "underlying_price": underlying,
        "atm_iv": atm_iv,
        "iv_skew": iv_skew,
        "pcr_oi": pcr_oi,
        "pcr_volume": pcr_vol,
        "max_pain": max_pain_strike,
        "max_pain_distance_pct": mp_dist_pct,
        "total_oi": total_call_oi + total_put_oi,
        "total_volume": total_call_vol + total_put_vol,
        "expiry_used": target_expiry,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    today = date.today()
    print(f"== fetch options chain {today} ==")
    opener = build_opener()
    if not warmup(opener):
        print("NSE warmup failed — skipping run gracefully")
        return

    syms = list_fno_underlyings(opener)
    if not syms:
        print("no F&O underlyings returned (NSE likely blocking) — skipping run")
        return
    syms = syms[:MAX_SYMBOLS]
    print(f"fetching options chain for {len(syms)} underlyings")

    rows: list[dict] = []
    fail_404 = []
    consecutive_empty = 0

    for i, sym in enumerate(syms, 1):
        url = OPTION_CHAIN_URL.format(sym=urllib.parse.quote(sym, safe=""))
        time.sleep(REQUEST_DELAY_SEC)
        payload = fetch_json(opener, url, retries=1)
        is_empty = (payload is None) or (not payload) or not (payload.get("records"))

        if is_empty:
            # Try one re-warm + retry — NSE cookies expire quickly.
            consecutive_empty += 1
            if consecutive_empty in (3, 8):
                print(f"  {consecutive_empty} consecutive empty payloads — re-warming session")
                warmup(opener)
                time.sleep(1.0)
                payload = fetch_json(opener, url, retries=0)
                is_empty = (payload is None) or (not payload) or not (payload.get("records"))

        if is_empty:
            fail_404.append(sym)
            if consecutive_empty >= 12:
                print(f"  {consecutive_empty} consecutive empty — NSE blocking. Halting (gracefully).")
                break
            continue
        consecutive_empty = 0

        try:
            feat = compute_features(sym, payload, today)
        except Exception as e:
            print(f"  {sym} parse ERR: {str(e)[:120]}")
            continue
        if feat is None:
            continue
        rows.append(feat)
        if i % 25 == 0:
            print(f"  progress: {i}/{len(syms)}  rows={len(rows)}  failed={len(fail_404)}")

    if not rows:
        print("no rows parsed — skipping write")
        return

    df_new = pd.DataFrame(rows)
    print(f"parsed {len(df_new)} rows; non-zero atm_iv: {(df_new['atm_iv'] > 0).sum()}")

    if OUT.exists():
        old = pd.read_parquet(OUT)
        before = len(old)
        # Align dtypes for trade_date
        old["trade_date"] = pd.to_datetime(old["trade_date"])
        df_new["trade_date"] = pd.to_datetime(df_new["trade_date"])
        merged = pd.concat([old, df_new], ignore_index=True)
        merged = merged.drop_duplicates(["symbol", "trade_date"], keep="last")
        merged = merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        merged.to_parquet(OUT, index=False)
        print(f"  appended: {before} → {len(merged)} (delta {len(merged)-before})")
    else:
        df_new = df_new.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        df_new.to_parquet(OUT, index=False)
        print(f"  fresh write: {len(df_new)} rows")

    if fail_404:
        print(f"  {len(fail_404)} symbols failed (sample): {fail_404[:10]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Never abort the daily pipeline because of this fetcher.
        print(f"FATAL (non-fatal to pipeline): {str(e)[:200]}")
        sys.exit(0)
