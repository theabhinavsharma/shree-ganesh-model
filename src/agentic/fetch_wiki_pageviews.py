"""Fetch Wikipedia page-view counts as a retail-attention proxy.

Wikimedia REST API endpoint (free, no auth):
  https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/
       en.wikipedia/all-access/all-agents/<TITLE>/daily/<YYYYMMDD>/<YYYYMMDD>

For each symbol we pre-map a Wikipedia title (best-effort), then fetch
last 30 days of daily views. Output gives:
  symbol, trade_date, wiki_views, wiki_views_7d_z

The agent-loop hypothesis "wiki_pageviews" reads this.

Throttle: 1 req/s (Wikimedia is generous but be polite).
"""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/wiki_pageviews.parquet"

# Manual map for the highest-priority 50 symbols → English Wikipedia titles
# (could be enriched programmatically via Wikipedia search API, but exact
#  title matching matters and the search API rate-limits, so a hand list
#  for the names that matter most.)
SYMBOL_TO_WIKI: dict[str, str] = {
    "RELIANCE": "Reliance_Industries",
    "TCS": "Tata_Consultancy_Services",
    "HDFCBANK": "HDFC_Bank",
    "INFY": "Infosys",
    "ICICIBANK": "ICICI_Bank",
    "HINDUNILVR": "Hindustan_Unilever",
    "SBIN": "State_Bank_of_India",
    "BHARTIARTL": "Bharti_Airtel",
    "KOTAKBANK": "Kotak_Mahindra_Bank",
    "LT": "Larsen_%26_Toubro",
    "HCLTECH": "HCLTech",
    "AXISBANK": "Axis_Bank",
    "MARUTI": "Maruti_Suzuki",
    "ASIANPAINT": "Asian_Paints",
    "WIPRO": "Wipro",
    "OFSS": "Oracle_Financial_Services_Software",
    "BSE": "Bombay_Stock_Exchange",
    "BAJFINANCE": "Bajaj_Finance",
    "ONGC": "Oil_and_Natural_Gas_Corporation",
    "NTPC": "NTPC_Limited",
    "ITC": "ITC_Limited",
    "TECHM": "Tech_Mahindra",
    "TITAN": "Titan_Company",
    "ULTRACEMCO": "UltraTech_Cement",
    "POWERGRID": "Power_Grid_Corporation_of_India",
    "M&M": "Mahindra_%26_Mahindra",
    "TATAMOTORS": "Tata_Motors",
    "TATASTEEL": "Tata_Steel",
    "JSWSTEEL": "JSW_Steel",
    "INDUSINDBK": "IndusInd_Bank",
    "ADANIENT": "Adani_Enterprises",
    "ADANIPORTS": "Adani_Ports_%26_SEZ",
    "RPOWER": "Reliance_Power",
    "ZOMATO": "Zomato",
    "PAYTM": "Paytm",
    "OLAELEC": "Ola_Electric",
    "PNB": "Punjab_National_Bank",
    "BPCL": "Bharat_Petroleum",
    "IOC": "Indian_Oil_Corporation",
    "SUNPHARMA": "Sun_Pharmaceutical",
    "DRREDDY": "Dr._Reddy%27s_Laboratories",
    "CIPLA": "Cipla",
    "DIVISLAB": "Divi%27s_Laboratories",
    "EICHERMOT": "Eicher_Motors",
    "BAJAJFINSV": "Bajaj_Finserv",
    "JIOFIN": "Jio_Financial_Services",
    "DMART": "Avenue_Supermarts",
    "DLF": "DLF_(company)",
    "ADANIGREEN": "Adani_Green_Energy",
    "BHEL": "Bharat_Heavy_Electricals_Limited",
    "VEDL": "Vedanta_Limited",
    "COALINDIA": "Coal_India",
    "GAIL": "GAIL",
}

UA = ("StockAnalysis/1.0 (https://github.com/example/stock-analysis; "
      "fetch_wiki_pageviews.py)")
DELAY = 1.0


def fetch_views(title: str, days: int = 30) -> list[dict]:
    end = date.today()
    start = end - timedelta(days=days)
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           f"en.wikipedia/all-access/all-agents/{title}/daily/"
           f"{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    return data.get("items", [])


def main() -> None:
    print(f"== fetch_wiki_pageviews  symbols={len(SYMBOL_TO_WIKI)} ==")
    rows: list[dict] = []
    for sym, title in SYMBOL_TO_WIKI.items():
        try:
            items = fetch_views(title, days=45)
        except Exception as e:
            print(f"  [{sym}] {type(e).__name__}: {str(e)[:120]}")
            time.sleep(DELAY)
            continue
        if not items:
            print(f"  [{sym}] no data for '{title}'")
            time.sleep(DELAY)
            continue
        for it in items:
            d = it.get("timestamp", "")[:8]
            try:
                trade_date = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}")
            except Exception:
                continue
            rows.append({"symbol": sym, "trade_date": trade_date, "wiki_views": int(it.get("views", 0))})
        print(f"  [{sym}] {title}: {len(items)} days, latest={items[-1].get('views',0)}", flush=True)
        time.sleep(DELAY)

    if not rows:
        print("nothing fetched")
        return
    df = pd.DataFrame(rows).drop_duplicates(["symbol", "trade_date"], keep="last")
    # 7d z-score
    df = df.sort_values(["symbol", "trade_date"])
    df["wiki_views_7d_mean"] = df.groupby("symbol")["wiki_views"].transform(lambda s: s.rolling(7).mean())
    df["wiki_views_30d_mean"] = df.groupby("symbol")["wiki_views"].transform(lambda s: s.rolling(30).mean())
    df["wiki_views_z"] = (df["wiki_views"] - df["wiki_views_30d_mean"]) / (df["wiki_views_30d_mean"] + 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        old["trade_date"] = pd.to_datetime(old["trade_date"])
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(["symbol", "trade_date"], keep="last")
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df):,} (symbol, day) rows for {df['symbol'].nunique()} symbols")


if __name__ == "__main__":
    main()
