"""Cross-country macro fetcher — global signals affecting India.

Sources (free, no API key):
  • Frankfurter (ECB) — USDINR, EURINR, GBPINR, JPYINR
  • Yahoo Finance public quote API (without auth) — limited but works for indices
  • Wikipedia (last-resort) for headline events

Series we want:
  - SPX (US S&P 500) — global risk-on/off
  - US10Y yield — risk-free rate driver
  - DXY (US dollar index) — EM currency pressure
  - Brent crude — India CPI driver
  - Gold (XAU/USD) — safe-haven
  - VIX (US) — global vol
  - SHCOMP (Shanghai) — China leading indicator
  - NIFTY 50 close (own market)

Output: appended to data/derived/macro_timeseries.parquet (existing FX file)
"""
from __future__ import annotations
import gzip
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import json

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/macro_timeseries.parquet"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123.0.0.0"


def yahoo_chart(ticker: str, lookback_days: int = 1500) -> pd.DataFrame:
    """Pull daily history from Yahoo via public chart API. Note: rate-limited."""
    import time as _t
    end = int(_t.time())
    start = end - lookback_days * 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={start}&period2={end}&interval=1d")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": UA,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        })
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
            if r.headers.get("Content-Encoding", "").lower() == "gzip":
                raw = gzip.decompress(raw)
        j = json.loads(raw.decode("utf-8"))
        chart = j.get("chart", {}).get("result", [{}])[0]
        ts = chart.get("timestamp", [])
        quote = chart.get("indicators", {}).get("quote", [{}])[0]
        closes = quote.get("close", [])
        if not ts or not closes:
            return pd.DataFrame()
        return pd.DataFrame({
            "trade_date": pd.to_datetime([datetime.utcfromtimestamp(t) for t in ts]).normalize(),
            ticker: closes,
        }).dropna()
    except Exception as e:
        print(f"  yahoo {ticker} FAIL: {type(e).__name__}: {str(e)[:120]}")
        return pd.DataFrame()


# Global macro tickers (Yahoo)
TICKERS = {
    "^GSPC":        "spx",            # S&P 500
    "^TNX":         "us10y",          # US 10Y treasury yield
    "DX-Y.NYB":     "dxy",            # US Dollar Index
    "BZ=F":         "brent",          # Brent crude
    "GC=F":         "gold",           # Gold
    "^VIX":         "us_vix",         # US VIX
    "000001.SS":    "shcomp",         # Shanghai Composite
    "^NSEI":        "nifty_50",       # NIFTY 50
    "^NSEBANK":     "bank_nifty",     # Bank NIFTY
}


def main() -> None:
    print(f"== fetch_global_macro  {len(TICKERS)} tickers ==")
    import time as _t
    merged: pd.DataFrame | None = None
    for ticker, name in TICKERS.items():
        df = yahoo_chart(ticker, lookback_days=1500)
        if df.empty:
            print(f"  {ticker:<14} ({name}): empty")
            continue
        df = df.rename(columns={ticker: name})
        latest = df.iloc[-1][name]
        print(f"  {ticker:<14} ({name}): {len(df)} days, latest={latest:.2f}")
        merged = df if merged is None else merged.merge(df, on="trade_date", how="outer")
        _t.sleep(0.8)

    if merged is None or merged.empty:
        print("\nno data fetched (Yahoo throttled or down)")
        return

    # compute derived: 5d/20d changes
    derived_cols = []
    for col in TICKERS.values():
        if col in merged.columns:
            merged[f"{col}_5d_chg"] = merged[col].pct_change(5)
            merged[f"{col}_20d_chg"] = merged[col].pct_change(20)
            derived_cols.extend([f"{col}_5d_chg", f"{col}_20d_chg"])

    merged = merged.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        old["trade_date"] = pd.to_datetime(old["trade_date"])
        # outer merge new columns onto existing
        common = list(set(old.columns) & set(merged.columns))
        new_only_cols = [c for c in merged.columns if c not in old.columns]
        for c in new_only_cols:
            old[c] = pd.NA
        old_set = old.set_index("trade_date")
        new_set = merged.set_index("trade_date")
        out = old_set.combine_first(new_set).reset_index()
    else:
        out = merged
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} days × {len(out.columns)} cols")


if __name__ == "__main__":
    main()


# ── FRED FALLBACK (re-applied 2026-08-18; original Jul-7 rewrite was reverted) ──
# Yahoo 429s constantly. After the Yahoo attempt, ALWAYS top-up us_10y/dxy/us_vix/spx
# from FRED into global_rates.parquet (what build_macro_panel actually reads).
def fred_fallback() -> None:
    import io, time as _t
    import urllib.request as _rq
    import pandas as _pd
    # DO NOT use the spoofed-Chrome UA here: FRED's WAF tarpits it (connection
    # accepted, never answered → timeout). The honest bot UA gets HTTP 200 in 0.3s.
    # Verified empirically 2026-08-18.
    _UA = "Mozilla/5.0 (compatible; ZoomFetcher/1.0)"
    _SERIES = {"DGS10": "us_10y", "DGS2": "us_2y", "DGS3MO": "us_3m",
               "DTWEXBGS": "dxy", "VIXCLS": "us_vix", "SP500": "spx"}
    merged = None
    for sid, col in _SERIES.items():
        for attempt in range(3):
            try:
                req = _rq.Request(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                                  headers={"User-Agent": _UA})
                with _rq.urlopen(req, timeout=30) as r:
                    df = _pd.read_csv(io.StringIO(r.read().decode("utf-8", errors="replace")))
                dc = "observation_date" if "observation_date" in df.columns else "DATE"
                df = df.rename(columns={dc: "trade_date"})
                df["trade_date"] = _pd.to_datetime(df["trade_date"])
                vc = [c for c in df.columns if c != "trade_date"][0]
                df[vc] = _pd.to_numeric(df[vc], errors="coerce")
                df = df.dropna(subset=[vc]).rename(columns={vc: col})
                print(f"  FRED {col}: latest {df.iloc[-1][col]:.2f} on {df['trade_date'].max().date()}")
                merged = df if merged is None else merged.merge(df, on="trade_date", how="outer")
                break
            except Exception as e:
                print(f"  FRED {sid} attempt {attempt+1}: {e}")
                _t.sleep(2)
    if merged is None:
        print("  ❌ FRED fallback failed entirely — gate will catch stale columns")
        raise SystemExit(1)
    gr_path = ROOT / "data/derived/global_rates.parquet"
    import pandas as pd2
    gr = pd2.read_parquet(gr_path)
    gr["trade_date"] = pd2.to_datetime(gr["trade_date"])
    for c in _SERIES.values():
        if c in gr.columns:
            gr = gr.drop(columns=[c])
    gr = gr.merge(merged.sort_values("trade_date"), on="trade_date", how="outer")
    gr.sort_values("trade_date").reset_index(drop=True).to_parquet(gr_path, index=False)
    print(f"  global_rates.parquet topped up from FRED")


if __name__ == "__main__":
    fred_fallback()
