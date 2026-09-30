"""Market cap + price from screener.in for equities that screener_fundamentals does not size.

build_mcap_pit.py sizes a row from quarterly P&L shares (PAT/EPS, 2018-04+) and falls back, row by row, to a
screener share count (market_cap / current_price, present-share basis). screener_fundamentals covers ~1,000 live
names; this backfill fetches the same top-ratios block for every other equity so that (a) symbols with no P&L
history and (b) rows of P&L symbols before their first filing (2015 to mid-2018, first weeks after listing) can be
sized. screener.in is NOT an NSE feed; its share counts are present-day (last trade for a delisted company), so every
row sized from here is pit_ok=False in mcap_pit.

Page rule (v2, 2026-09-27): fetch /company/<SYMBOL>/ (or the screener_industry URL only when its slug IS the symbol),
record the final URL after redirects, and mark status SLUG_MISMATCH when the page that answered is not the symbol's
own. NSE old symbols (research_panel.rename_map keys) are never fetched: screener answers /company/<OLD>/ with the
successor's page (237 of the 255 old-symbol records in the checkpoint redirected, 17 were HTTP 404, 1 had its own
page). build_mcap_pit sizes an old symbol through its successor's page only when it chains the link (contiguous, or
bridged with the same ISIN) and otherwise leaves the row NULL with null_reason 'screener_rename_unbridged_<reason>'.
Output: data/derived/screener_mcap_backfill.parquet (+manifest), checkpoint .jsonl, resume-safe.
  --plan   print the fetch plan (counts + first symbols) and exit: no network, no writes.

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json):
  FIXED  fetch_screener_mcap_backfill.py 1-8,35-45 (high, via build_mcap_pit.py 65-73): URLs resolved by
         screener_industry that followed renames (slug != symbol) are no longer used; the final URL is recorded
         (final_url) and a redirected page is SLUG_MISMATCH. Legacy checkpoint records whose slug is not the symbol
         are re-fetched under the v2 rule. The docstring no longer claims a delisted company is sized "at its last
         trade" when the page belonged to a successor.
  FIXED  fetch_screener_mcap_backfill.py 35-37 (medium, todo = eq - have): the target set is every equity without a
         usable screener_fundamentals share count (not only symbols with no sized mcap_pit row), so P&L symbols get
         a screener count for rows before their first filing. No longer reads mcap_pit.parquet (removes the
         builder <-> fetcher cycle). REQUIRES a re-run of this fetcher, then build_mcap_pit.py, to reach the data.
  CHANGED  the equity set is research_panel.equity_filter (fund units AND non_equity removed), matching the
         builder's panel; previously only ISIN-master fund units were removed.
2026-09-27 review round 2:
  FIXED  fetch_screener_mcap_backfill.py 63-71 (medium): only contiguous pre-rename symbols were skipped, so an old
         symbol behind a >10-day panel gap (TRIL, GET&D, MAGMA, ...) was re-fetched, redirected to its successor,
         stored as SLUG_MISMATCH and rejected again on every run. Every NSE old symbol is now skipped; whether it is
         sized through the successor is decided once, in build_mcap_pit (rename links), not by this fetch plan.
         --plan on 2026-09-27: 763 symbols to fetch (round 1: 795), 255 old symbols skipped, 34 legacy
         slug-mismatch records re-fetched (round 1: 66).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
from build_mcap_pit import slug_of  # noqa: E402
from fetch_screener_fundamentals import parse_top_ratios  # noqa: E402

CKPT = ROOT / "data/derived/screener_mcap_backfill.jsonl"
OUT = ROOT / "data/derived/screener_mcap_backfill.parquet"
SCR = ROOT / "data/derived/screener_fundamentals.parquet"
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
DELAY = 1.6
URL_RULE = "symbol_slug_v2"


def plan() -> tuple[list[str], dict]:
    """Symbols to fetch and a count summary. Reads only small derived files."""
    sm = pd.read_parquet(rp.MASTER)
    eq = set(sm.loc[rp.equity_filter(sm["symbol"]).to_numpy(), "symbol"])
    scr = pd.read_parquet(SCR, columns=["symbol", "market_cap_cr", "current_price"]).dropna()
    scr_ok = set(scr.loc[(scr["current_price"] > 0) & (scr["market_cap_cr"] > 0), "symbol"])
    rmap = rp.rename_map()
    recs = [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()] if CKPT.exists() else []
    latest = {r["symbol"]: r for r in recs}                     # last record per symbol wins, as in the output
    done = {s for s, r in latest.items() if r.get("status") not in ("HTTP_ERR", "HTTP_429")   # transient: retry
            and (r.get("url_rule") == URL_RULE or slug_of(r.get("final_url") or r.get("url")) == s.upper())}
    legacy_mismatch = {s for s, r in latest.items() if s not in done}
    pre_rename = eq & set(rmap)                                 # screener redirects old slugs to the successor
    todo = sorted(s for s in eq if s not in scr_ok and s not in pre_rename and s not in done)
    succ_eq = {rmap[s] for s in pre_rename} & eq
    summary = dict(equities=len(eq), screener_fundamentals_sized=len(eq & scr_ok),
                   nse_old_symbols_not_fetched=len(pre_rename),
                   their_successors_in_scr_fundamentals_checkpoint_or_todo=len(succ_eq & (scr_ok | done | set(todo))),
                   their_successors_not_equities=len({rmap[s] for s in pre_rename} - eq),
                   checkpoint_done=len(done & eq),
                   legacy_slug_mismatch_refetch=len(legacy_mismatch & set(todo)), todo=len(todo))
    return todo, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the fetch plan and exit (no network, no writes)")
    args = ap.parse_args()
    todo, summary = plan()
    print(json.dumps(summary), flush=True)
    if args.plan:
        print("first:", todo[:30])
        return
    si = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    si = si[si["status"].str.startswith("OK")]
    own_url = {s: u for s, u in zip(si["symbol"], si["url"]) if slug_of(u) == str(s).upper()}
    s = requests.Session()
    with CKPT.open("a") as fh:
        for i, sym in enumerate(todo):
            url = own_url.get(sym) or f"https://www.screener.in/company/{requests.utils.quote(sym, safe='')}/"
            rec = dict(symbol=sym, url=url, url_rule=URL_RULE, fetch_date=datetime.now().strftime("%Y-%m-%d"))
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
                rec["final_url"] = r.url
                tr = parse_top_ratios(r.text)
                rec.update(market_cap_cr=tr.get("market_cap_cr"), current_price=tr.get("current_price"))
                if slug_of(r.url) != sym.upper():
                    rec["status"] = "SLUG_MISMATCH"                      # redirected to another company's page
                else:
                    rec["status"] = "OK" if rec["market_cap_cr"] and rec["current_price"] else "NO_MCAP"
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} {sym} {rec['status']}", flush=True)
            time.sleep(DELAY)
    d = pd.DataFrame([json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]).drop_duplicates("symbol", keep="last")
    for c in ("final_url", "url_rule"):
        if c not in d:
            d[c] = None
    d["slug_ok"] = [slug_of(f if isinstance(f, str) else u) == str(sy).upper() for sy, u, f in zip(d["symbol"], d["url"], d["final_url"])]
    d.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="screener_mcap_backfill", path=str(OUT.relative_to(ROOT)), rows=len(d), key=["symbol"],
        producer="src/agentic/fetch_screener_mcap_backfill.py", source="screener.in top-ratios (Market Cap, Current Price); NOT an NSE feed",
        columns=dict(market_cap_cr="Rs crore as shown on the page at fetch_date (present-day; last trade for a delisted company)",
                     current_price="Rs, then-traded at fetch_date",
                     status="OK | NO_MCAP | SLUG_MISMATCH (page answered for another slug) | HTTP_<code>",
                     url="page requested", final_url="page that answered after redirects (v2 records only)",
                     url_rule=f"{URL_RULE} for records fetched under the symbol-slug rule; NULL = legacy record",
                     slug_ok="slug of final_url (else url) == symbol; build_mcap_pit also accepts a CHAINED "
                             "rename-chain member's page (contiguous, or bridged with the same ISIN) and rebases it"),
        caveat="legacy records (url_rule NULL) stored only the requested URL, so a server-side redirect cannot be "
               "detected for them; builder accepts them only when that URL's slug is the symbol",
        status_counts=d["status"].value_counts().to_dict(), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print("MCAP BACKFILL COMPLETE", d["status"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
