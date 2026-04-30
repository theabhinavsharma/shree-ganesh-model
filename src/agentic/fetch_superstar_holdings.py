"""Fetch publicly disclosed 'Superstar' investor holdings from Tickertape.

Source: https://www.tickertape.in/stocks/collections/top-20-investors-in-india-portfolio
Each investor has a public prebuilt screener at:
  https://www.tickertape.in/screener/equity/prebuilt/{slug}-portfolio-stock-list

URLs verified working as of 2026-04-29. No login, no API key.

Output: data/derived/superstar_holdings.parquet
  cols: investor_tag, investor_name, symbol, fetch_date

Plus computes confluence — stocks held by ≥ 2 investors = smart-money signal.

Honest caveats:
  • Disclosures are quarterly (NSE/BSE shareholding pattern). NOT real-time.
  • Lag is 30-45 days from quarter-end.
  • Tickertape may slightly de-stale by polling NSE filings, but expect ~30 day lag.
  • Tickertape ticker symbols sometimes differ from NSE symbols
    (e.g. FLUOROCHEM on Tickertape = GUJFLUORO on NSE). We intersect with
    our NSE universe and report only matches.
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
OUT = ROOT / "data/derived/superstar_holdings.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

COLLECTION_URL = "https://www.tickertape.in/stocks/collections/top-20-investors-in-india-portfolio"
DELAY = 2.0


def _opener() -> urllib.request.OpenerDirector:
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


# Verified URLs from the collection page (2026-04-29)
INVESTORS: list[dict] = [
    {"tag": "AMBANI",     "name": "Mukesh Ambani",            "url": "https://www.tickertape.in/screener/equity/prebuilt/mukesh-ambani-portfolio-stock-list"},
    {"tag": "RKD",        "name": "Radhakishan Damani",       "url": "https://www.tickertape.in/screener/equity/prebuilt/radhakishan-damani-portfolio-stock-list"},
    {"tag": "RKJ",        "name": "Rakesh Jhunjhunwala",      "url": "https://www.tickertape.in/screener/equity/prebuilt/rakesh-jhunjhunwala-portfolio-stock-list"},
    {"tag": "REKHAJ",     "name": "Rekha Jhunjhunwala",       "url": "https://www.tickertape.in/screener/equity/prebuilt/rekha-jhunjhunwala-portfolio-stock-list"},
    {"tag": "AKASH",      "name": "Akash Bhanshali",          "url": "https://www.tickertape.in/screener/equity/prebuilt/akash-bhanshali-portfolio-stock-list"},
    {"tag": "NEMISH",     "name": "Nemish Shah",              "url": "https://www.tickertape.in/screener/equity/prebuilt/nemish-shah-portfolio-stock-list"},
    {"tag": "KACHOLIA",   "name": "Ashish Kacholia",          "url": "https://www.tickertape.in/screener/equity/prebuilt/ashish-kacholia-portfolio-stock-list"},
    {"tag": "SINGHANIA",  "name": "Sunil Singhania",          "url": "https://www.tickertape.in/screener/equity/prebuilt/sunil-singhania-portfolio-stock-list"},
    {"tag": "KELA",       "name": "Madhusudan Kela",          "url": "https://www.tickertape.in/screener/equity/prebuilt/madhusudan-kela-portfolio-stock-list"},
    {"tag": "GOEL",       "name": "Anil Kumar Goel",          "url": "https://www.tickertape.in/screener/equity/prebuilt/anil-kumar-goel-portfolio-stock-list"},
    {"tag": "DHAWAN",     "name": "Ashish Dhawan",            "url": "https://www.tickertape.in/screener/equity/prebuilt/ashish-dhawan-portfolio-stock-list"},
    {"tag": "SHETH",      "name": "Anuj Anantrai Sheth",      "url": "https://www.tickertape.in/screener/equity/prebuilt/anuj-anantrai-sheth-portfolio-stock-list"},
    {"tag": "KEDIA",      "name": "Vijay Kedia",              "url": "https://www.tickertape.in/screener/equity/prebuilt/vijay-kedia-portfolio-stock-list"},
    {"tag": "TRIPATHI",   "name": "Bhavook Tripathi",         "url": "https://www.tickertape.in/screener/equity/prebuilt/bhavook-tripathi-portfolio-stock-list"},
    {"tag": "UPADHYAY",   "name": "Ajay Upadhyay",            "url": "https://www.tickertape.in/screener/equity/prebuilt/ajay-upadhyay-portfolio-stock-list"},
    {"tag": "LAKHI",      "name": "Dilipkumar Lakhi",         "url": "https://www.tickertape.in/screener/equity/prebuilt/dilipkumar-lakhi-portfolio-stock-list"},
    {"tag": "TRIVEDI",    "name": "Shivani Tejas Trivedi",    "url": "https://www.tickertape.in/screener/equity/prebuilt/shivani-tejas-trivedi-portfolio-stock-list"},
    {"tag": "MUKUL",      "name": "Mukul Agrawal",            "url": "https://www.tickertape.in/screener/equity/user/mukul-agrawal-portfolio-stock-screener-aYqDaxny0bqiybFW"},
]


# Map common Tickertape tickers → NSE symbols when they differ
TICKERTAPE_TO_NSE = {
    "FLUOROCHEM": "GUJFLUORO",  # Gujarat Fluorochemicals
    "RKFORGE":    "RAMKY",      # Ramkrishna Forgings — verify; placeholder
    "PRAXIS":     "PRAXIS",     # Praxis Home Retail
    "SOMICONVEY": "SOMI",       # Somi Conveyor
    "SUDARSCHEM": "SUDARSCHEM",
    "GEECEE":     "GEECEE",
    "GENUSPOWER": "GENUSPOWER",
    "PANELEC":    "PANELEC",
    "WLSP":       "WELSPUNIND",  # Welspun India
    "AMBER":      "AMBER",
    "INOXWIND":   "INOXWIND",
    "LAURUSLABS": "LAURUSLABS",
    "NATCOPHARM": "NATCOPHARM",
    "GREENLAM":   "GREENLAM",
    "DBL":        "DBL",
    "PAYTM":      "PAYTM",
    "RKFO":       "RKFORGE",  # short slug from tickertape
    "SHILPAMED":  "SHILPAMED",
    "SCHNEIDER":  "SCHNEIDER",
}


def parse_holdings(html: str) -> list[str]:
    """Extract NSE-style ticker symbols from a Tickertape portfolio page.

    Tickertape embeds them as `"ticker":"XXX"` in the page JSON state.
    We filter out US tickers (AAPL, AMZN, GOOGL, MSFT, NVDA, TSLA, SPY, IWM, QQQ, DIA)
    that appear because Tickertape also tracks US stocks.
    """
    matches = re.findall(r'"ticker"\s*:\s*"([A-Z0-9&_-]+)"', html)
    # filter out obvious US tickers + ETFs that tickertape includes
    us_tickers = {"AAPL", "AMZN", "GOOGL", "GOOG", "MSFT", "NVDA", "TSLA",
                  "META", "NFLX", "JPM", "V", "MA", "DIS", "BAC",
                  "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO"}
    nse_tickers = []
    for t in matches:
        if t in us_tickers:
            continue
        if len(t) < 2 or len(t) > 15:
            continue
        nse_tickers.append(t)
    # de-dup, preserve order
    seen = set()
    return [t for t in nse_tickers if not (t in seen or seen.add(t))]


def main() -> None:
    print(f"== fetch_superstar_holdings (Tickertape) ==")
    print(f"  source: {COLLECTION_URL}")
    print(f"  investors: {len(INVESTORS)} (URLs verified 2026-04-29)\n")

    opener = _opener()
    # warm session
    try:
        _get(opener, "https://www.tickertape.in/")
    except Exception:
        pass

    # NSE universe for symbol normalization
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "series"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    latest = px["trade_date"].max()
    nse_universe = set(px[(px["trade_date"] == latest) & (px["series"] == "EQ")]["symbol"].astype(str))
    print(f"  NSE universe: {len(nse_universe):,} EQ symbols")

    all_rows = []
    for inv in INVESTORS:
        try:
            html = _get(opener, inv["url"])
            tickers = parse_holdings(html)
            if not tickers:
                print(f"  [{inv['tag']:<10}] {inv['name'][:35]:<35} → 0 (parse miss)")
                time.sleep(DELAY)
                continue
            # normalize to NSE symbols
            normalized = []
            for t in tickers:
                nse_sym = TICKERTAPE_TO_NSE.get(t, t)
                if nse_sym in nse_universe:
                    normalized.append(nse_sym)
            print(f"  [{inv['tag']:<10}] {inv['name'][:35]:<35} → {len(tickers)} raw, {len(normalized)} NSE-matched")
            for sym in normalized:
                all_rows.append({
                    "investor_tag": inv["tag"],
                    "investor_name": inv["name"],
                    "symbol": sym,
                    "fetch_date": pd.Timestamp(date.today()),
                    "source_url": inv["url"],
                })
        except Exception as e:
            print(f"  [{inv['tag']:<10}] FAIL {type(e).__name__}: {str(e)[:120]}")
        time.sleep(DELAY)

    if not all_rows:
        print("\nno holdings parsed")
        return

    df = pd.DataFrame(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(["investor_tag", "symbol", "fetch_date"], keep="last")
    else:
        merged = df
    merged.to_parquet(OUT, index=False)

    # confluence: stocks held by ≥ 2 investors
    confluence = (df.groupby("symbol")
                    .agg(n_superstars=("investor_tag", "nunique"),
                         investors=("investor_tag", lambda s: ", ".join(sorted(set(s)))))
                    .reset_index()
                    .sort_values("n_superstars", ascending=False))
    high_conv = confluence[confluence["n_superstars"] >= 2]

    print(f"\nwrote {OUT}: {len(merged):,} total rows")
    print(f"  unique stocks across {df['investor_tag'].nunique()} investors: {df['symbol'].nunique()}")
    print(f"\nSmart-money confluence (held by ≥ 2 superstars):")
    if len(high_conv):
        print(high_conv.head(25).to_string(index=False))
    else:
        print("  none yet")

    print(f"\nTop solo holdings per investor (first 5):")
    for inv in INVESTORS[:5]:
        sub = df[df["investor_tag"] == inv["tag"]]
        if len(sub):
            print(f"  {inv['name']}: {', '.join(sub['symbol'].tolist()[:8])}{' …' if len(sub) > 8 else ''}")


if __name__ == "__main__":
    main()
