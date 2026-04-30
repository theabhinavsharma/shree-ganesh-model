"""Fetch comprehensive fundamentals from Screener.in per-company pages.

Source: https://www.screener.in/company/<SYMBOL>/consolidated/
Each page contains a <ul id="top-ratios"> with key ratios:
  Market Cap, Current Price, P/E, Book Value, Dividend Yield, ROCE, ROE,
  Face Value, High/Low, plus other sector/business-specific ratios

Plus the page contains:
  - Quarterly results table (sales, expenses, OPM%, PAT) — last 12-13 quarters
  - Annual P&L
  - Historical CAGR sections (sales growth 3y/5y/10y, profit growth 3y/5y/10y, ROE 3y/5y/10y)
  - Compounded sales growth, compounded profit growth
  - Stock price CAGR 1y/3y/5y/10y

Output: data/derived/screener_fundamentals.parquet
  cols: symbol, fetch_date, market_cap_cr, current_price, pe, book_value,
        dividend_yield, roce, roe, eps, debt_to_equity, promoter_holding,
        sales_growth_3y, sales_growth_5y, profit_growth_3y, profit_growth_5y,
        return_1y, return_3y, return_5y, ... (40+ columns)
"""
from __future__ import annotations
import gzip
import http.cookiejar
import re
import ssl
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/screener_fundamentals.parquet"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

DELAY = 2.0
LONG_BREAK_EVERY = 50
LONG_BREAK_SEC = 25


def _opener():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )


def _get(opener, url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with opener.open(req, timeout=20) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors="replace")


# Parse the <ul id="top-ratios"> block
TOP_RATIOS_PATTERN = re.compile(
    r'<li[^>]*data-source="default"[^>]*>\s*<span class="name">\s*([^<]+?)\s*</span>'
    r'.*?<span class="nowrap value">(.*?)</span>\s*</li>',
    re.DOTALL | re.IGNORECASE,
)
NUMBER_IN_VALUE = re.compile(r'<span class="number">([\d.,\-]+)</span>')


def _parse_value(value_html: str) -> float | None:
    """Extract first <span class='number'> as a float."""
    m = NUMBER_IN_VALUE.search(value_html)
    if not m:
        return None
    s = m.group(1).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


# Map Screener ratio names → snake_case column names we want to keep
NAME_MAP: dict[str, str] = {
    "market cap": "market_cap_cr",
    "current price": "current_price",
    "stock p/e": "pe",
    "p/e": "pe",
    "book value": "book_value",
    "dividend yield": "dividend_yield",
    "roce": "roce",
    "roe": "roe",
    "face value": "face_value",
    "promoter holding": "promoter_holding",
    "eps": "eps",
    "debt to equity": "debt_to_equity",
    "industry pe": "industry_pe",
    "earnings yield": "earnings_yield",
    "pledged percentage": "pledged_pct",
    "price to book value": "price_to_book",
    "price to sales": "price_to_sales",
    "ev/ebitda": "ev_ebitda",
    "enterprise value": "enterprise_value",
    "current ratio": "current_ratio",
    "interest coverage ratio": "interest_coverage",
    "peg ratio": "peg_ratio",
    "return on assets": "roa",
    "opm": "opm_pct",
    "debt": "debt_cr",
    "change in promoter holding": "promoter_holding_change",
}


def parse_top_ratios(html: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    # Restrict to the <ul id="top-ratios">...</ul> block
    m = re.search(r'<ul id="top-ratios">(.*?)</ul>', html, re.DOTALL)
    if not m:
        return out
    block = m.group(1)
    for nm, val_html in TOP_RATIOS_PATTERN.findall(block):
        key = NAME_MAP.get(nm.strip().lower())
        if key is None:
            # keep unknown keys as snake_case fallback
            key = re.sub(r"[^a-z0-9]+", "_", nm.strip().lower()).strip("_")
        v = _parse_value(val_html)
        if v is not None:
            out[key] = v
    return out


# Parse "Compounded Sales Growth", "Compounded Profit Growth", "Stock Price CAGR", "Return on Equity"
# from the historical sections (each a <table> with row labels like "10 Years:", "5 Years:", etc.)
def parse_historical_section(html: str, section_title: str) -> dict[str, float | None]:
    """Find a section with a header containing section_title and parse its
    rows of '<period>: <value>%' pairs."""
    out: dict[str, float | None] = {}
    # Look for <table class="ranges-table"> with the matching <th>
    m = re.search(rf'<th[^>]*>\s*{re.escape(section_title)}\s*</th>(.*?)</table>',
                   html, re.DOTALL | re.IGNORECASE)
    if not m:
        return out
    block = m.group(1)
    # rows like <tr><td>10 Years:</td><td>20%</td></tr>
    for rm in re.finditer(r'<td[^>]*>([^<]+?)</td>\s*<td[^>]*>([\d.,\-]+)\s*%?\s*</td>',
                          block, re.IGNORECASE):
        period = rm.group(1).strip().lower().replace(":", "").replace(" ", "_")
        try:
            v = float(rm.group(2).replace(",", ""))
            slug = re.sub(r"[^a-z0-9]+", "_",
                           f"{section_title}_{period}".lower()).strip("_")
            out[slug] = v
        except ValueError:
            pass
    return out


def fetch_one(opener, sym: str) -> dict | None:
    url = f"https://www.screener.in/company/{sym}/consolidated/"
    try:
        html = _get(opener, url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # try without /consolidated/
            try:
                html = _get(opener, f"https://www.screener.in/company/{sym}/")
            except Exception:
                return None
        else:
            raise
    if "Page Not Found" in html or len(html) < 5000:
        return None

    row: dict = {"symbol": sym, "fetch_date": pd.Timestamp(date.today())}
    row.update(parse_top_ratios(html))

    # historical sections (Compounded Sales Growth, Compounded Profit Growth, Stock Price CAGR, ROE)
    for section in ["Compounded Sales Growth", "Compounded Profit Growth",
                     "Stock Price CAGR", "Return on Equity"]:
        row.update(parse_historical_section(html, section))

    return row


def get_top_symbols(n: int) -> list[str]:
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "series", "avg_traded_value_20d"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    latest = df["trade_date"].max()
    snap = df[(df["trade_date"] == latest) & (df["series"] == "EQ")]
    snap = snap.sort_values("avg_traded_value_20d", ascending=False).head(n)
    return snap["symbol"].astype(str).tolist()


def main(top_n: int = 200) -> None:
    print(f"== fetch_screener_fundamentals  top-{top_n} ==")
    syms = get_top_symbols(top_n)
    print(f"  fetching {len(syms)} symbols")
    opener = _opener()
    rows = []
    started = time.time()
    for i, sym in enumerate(syms, 1):
        try:
            row = fetch_one(opener, sym)
            if row:
                rows.append(row)
                if i <= 3 or i % 25 == 0:
                    elapsed = time.time() - started
                    eta = (len(syms) - i) * (elapsed / i) / 60
                    keys = [k for k in row if k not in ("symbol", "fetch_date")]
                    print(f"  [{i}/{len(syms)}] {sym:<12} → {len(keys)} fields  elapsed={elapsed:.0f}s  eta={eta:.0f}min")
        except Exception as e:
            print(f"  [{i}/{len(syms)}] {sym} FAIL {type(e).__name__}: {str(e)[:120]}")
        if i % LONG_BREAK_EVERY == 0:
            print(f"    [break] {LONG_BREAK_SEC}s …")
            time.sleep(LONG_BREAK_SEC)
        else:
            time.sleep(DELAY)

    if not rows:
        print("no rows fetched")
        return

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        merged = pd.concat([old, df], ignore_index=True).drop_duplicates(["symbol", "fetch_date"], keep="last")
    else:
        merged = df
    merged.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(merged):,} rows × {len(merged.columns)} cols")
    print(f"\nField coverage:")
    cov = merged.notna().mean().sort_values(ascending=False)
    for k, v in cov.head(25).items():
        print(f"  {k:<30}  {v*100:.0f}%")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()
    main(top_n=args.top_n)
