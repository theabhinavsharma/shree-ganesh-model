"""Market cap + price from screener.in for equities build_mcap_pit.py cannot size.

build_mcap_pit.py derives shares from quarterly P&L (PAT/EPS); pnl_quarterly covers ~1,750
companies, leaving ~1,540 symbols (855 live, mostly ADV < 0.5cr) with no market cap — exactly
the small names a 'market cap >= Rs 50cr' universe must include. screener.in's top-ratios block
gives Market Cap and Current Price; shares_now = market_cap / price (present-share basis; for a
delisted company, the basis at its last trade). Page URL: the one screener_industry resolved
(follows renames), else /company/<SYMBOL>/.
Output: data/derived/screener_mcap_backfill.parquet (+manifest), checkpoint .jsonl, resume-safe.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
from fetch_screener_fundamentals import parse_top_ratios  # noqa: E402

CKPT = ROOT / "data/derived/screener_mcap_backfill.jsonl"
OUT = ROOT / "data/derived/screener_mcap_backfill.parquet"
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
DELAY = 1.6


def main() -> None:
    sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
    eq = set(sm.loc[~sm["is_fund_unit"], "symbol"])
    have = set(pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet", columns=["symbol", "mcap_cr"]).dropna()["symbol"])
    si = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    urls = si[si["status"].str.startswith("OK")].set_index("symbol")["url"].to_dict()
    todo = sorted(eq - have)
    done = {json.loads(l)["symbol"] for l in CKPT.read_text().splitlines() if l.strip()} if CKPT.exists() else set()
    todo = [s for s in todo if s not in done]
    print(f"{len(todo)} symbols to size ({len(done)} checkpointed)", flush=True)
    s = requests.Session()
    with CKPT.open("a") as fh:
        for i, sym in enumerate(todo):
            url = urls.get(sym) or f"https://www.screener.in/company/{requests.utils.quote(sym, safe='')}/"
            rec = dict(symbol=sym, url=url, fetch_date=datetime.now().strftime("%Y-%m-%d"))
            r = None
            for _ in range(3):
                try:
                    r = s.get(url, headers=H, timeout=30)
                except requests.RequestException:
                    time.sleep(5); continue
                if r.status_code == 429:
                    time.sleep(60); continue
                break
            if r is None or r.status_code != 200:
                rec["status"] = f"HTTP_{getattr(r, 'status_code', 'ERR')}"
            else:
                tr = parse_top_ratios(r.text)
                rec.update(market_cap_cr=tr.get("market_cap_cr"), current_price=tr.get("current_price"))
                rec["status"] = "OK" if rec["market_cap_cr"] and rec["current_price"] else "NO_MCAP"
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} {sym} {rec['status']}", flush=True)
            time.sleep(DELAY)
    d = pd.DataFrame([json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]).drop_duplicates("symbol", keep="last")
    d.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="screener_mcap_backfill", path=str(OUT.relative_to(ROOT)), rows=len(d), key=["symbol"],
        producer="src/agentic/fetch_screener_mcap_backfill.py", source="screener.in top-ratios (Market Cap, Current Price); NOT an NSE feed",
        columns=dict(market_cap_cr="Rs crore as shown on the page at fetch_date (last trade for delisted)", current_price="Rs",
                     status="OK | NO_MCAP | HTTP_<code>", url="page fetched"),
        status_counts=d["status"].value_counts().to_dict(), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print("MCAP BACKFILL COMPLETE", d["status"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
