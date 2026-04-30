"""Pull curated Screener.in stock screens — FII/DII buying, promoter buying, etc.

Source URL pattern: https://www.screener.in/screens/<id>/<slug>/

These screens return a static HTML table of stocks (NO login, fast, ~40 KB).
Every stock has <a href="/company/SYMBOL/"> — easy to parse.

Output: data/derived/screener_screens.parquet
  cols: screen_tag, screen_name, symbol, fetch_date, source_url

The hypothesis we test: stocks appearing on FII_BUYING + DII_BUYING screens
historically outperform random selections at 60-180 day horizons.
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
OUT = ROOT / "data/derived/screener_screens.parquet"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

# verified screens (URLs work as of 2026-04-29)
SCREENS = [
    {"tag": "FII_DII_BUYING", "name": "FII & DII Buying",
     "url": "https://www.screener.in/screens/675072/fii-dii-buying/"},
    # Common public screens — these IDs are widely shared on Screener community
    # Best to discover more via /screens/?category=Investors+%26+Insiders
]

DELAY = 2.0


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


def parse_screen(html: str) -> list[str]:
    """Extract NSE-style symbols from a Screener screen page.

    Pattern: <a href="/company/SYMBOL/" — but Screener also uses BSE numeric
    codes for stocks not on NSE. We filter to alphanumeric symbols only."""
    pat = re.compile(r'<a href="/company/([A-Z0-9&\-]+)/?', re.I)
    raw = pat.findall(html)
    seen = set()
    out = []
    for s in raw:
        s = s.upper()
        # filter out BSE numeric codes (e.g. 544291)
        if s.isdigit():
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def main() -> None:
    print(f"== fetch_screener_screens  {len(SCREENS)} screens ==")
    opener = _opener()
    rows = []
    for s in SCREENS:
        try:
            html = _get(opener, s["url"])
            symbols = parse_screen(html)
            print(f"  [{s['tag']:<18}] {s['name']:<28} → {len(symbols)} stocks")
            for sym in symbols:
                rows.append({
                    "screen_tag": s["tag"],
                    "screen_name": s["name"],
                    "symbol": sym,
                    "fetch_date": pd.Timestamp(date.today()),
                    "source_url": s["url"],
                })
        except Exception as e:
            print(f"  [{s['tag']}] FAIL {type(e).__name__}: {str(e)[:120]}")
        time.sleep(DELAY)

    if not rows:
        print("nothing fetched")
        return

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(["screen_tag", "symbol", "fetch_date"], keep="last")
    else:
        merged = df
    merged.to_parquet(OUT, index=False)

    print(f"\nwrote {OUT}: {len(merged):,} total rows")
    print(f"\nToday's stocks on the FII_DII_BUYING screen:")
    today_rows = df[df["screen_tag"] == "FII_DII_BUYING"]
    print(f"  {', '.join(today_rows['symbol'].tolist())}")


if __name__ == "__main__":
    main()
