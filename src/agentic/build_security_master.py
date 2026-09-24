"""SECURITY MASTER — one row per NSE symbol ever seen: ISIN, fund-unit flag, rename chain,
industry with provenance. data/derived/security_master.parquet (+ .manifest.json).

Why (2026-09-23): three silent universe defects found in one day —
  * the ETF filter was a symbol regex: it dropped real companies (SKYGOLD, SHANTIGOLD,
    GOLDTECH) and let dozens of ETFs through (LIQUID, SILVER, SETFNIF50, EBBETF0433, ...);
  * the industry map (modal smIndustry of announcements_historical) covered 832 of 2,245
    core-band symbols, so the leader cell never saw ~60% of the universe;
  * the fallback (stock_announcements.industry_hint) only knows names alive in 2026 —
    survivor-only, so using it alone biases any backtest.
Sources, all official NSE and already on disk or on nsearchives:
  ISIN        raw old-format bhavcopies data/raw/nse_full_history_official/*/cm*bhav.csv.zip
              (SYMBOL, SERIES, ISIN; 2016-2024) + EQUITY_L.csv + eq_etfseclist.csv (current)
  fund unit   ISIN prefix 'INF' (MF/ETF units) — companies are 'INE'; OR in NSE's ETF list
  renames     nsearchives symbolchange.csv (old -> new, date)
  industry    1. announcements_historical.smIndustry (mode)    source 'nse_smIndustry'
              2. stock_announcements.industry_hint (mode)       source 'nse_industry_hint'
              3. same ISIN or rename chain carries 1/2          source '<src>_via_isin|rename'
              4. otherwise NULL — never guessed                 source 'unmapped'
Literal 'None' strings in the NSE feeds are treated as missing.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
from src.ingest.nse.api import _request_headers, _request_with_retries  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

RAW = ROOT / "data/raw/nse_full_history_official"
OUT = ROOT / "data/derived/security_master.parquet"
CACHE = ROOT / "data/raw/nse_reference"
URLS = {"symbolchange": "https://nsearchives.nseindia.com/content/equities/symbolchange.csv",
        "equity_l": "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "etf_list": "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"}
BAD = {"none", "nan", "", "-", "null"}


def _fetch_refs() -> dict[str, pd.DataFrame]:
    CACHE.mkdir(parents=True, exist_ok=True)
    s = build_session(warm=True, referer="https://www.nseindia.com/")
    out = {}
    for k, u in URLS.items():
        p = CACHE / f"{k}.csv"
        try:
            r = _request_with_retries(s, u, request_headers=_request_headers(s, referer=None), referer=None, timeout=60)
            p.write_text(r.text)
        except Exception as e:                       # keep last good copy; say so loudly
            print(f"  ⚠ {k} fetch failed ({str(e)[:80]}) — using cached {p.name}" if p.exists() else f"  ❌ {k}: no copy")
        if p.exists():
            out[k] = pd.read_csv(p, header=None if k == "symbolchange" else "infer", skipinitialspace=True)
    sc = out["symbolchange"]; sc.columns = ["name", "old", "new", "date"]
    out["symbolchange"] = sc.apply(lambda c: c.str.strip() if c.dtype == object else c)
    out["equity_l"].columns = [c.strip() for c in out["equity_l"].columns]
    return out


def _isins_from_bhavcopies() -> pd.DataFrame:
    rows = []
    files = sorted(RAW.glob("trade_date=*/cm*bhav.csv.zip"))
    for i, f in enumerate(files):
        try:
            with zipfile.ZipFile(f) as z:
                d = pd.read_csv(z.open(z.namelist()[0]), usecols=["SYMBOL", "SERIES", "ISIN"])
            d["dt"] = f.parent.name.split("=")[1]
            rows.append(d.drop_duplicates(["SYMBOL", "ISIN"]))
        except Exception:
            continue
        if i % 500 == 0:
            print(f"  bhavcopy {i}/{len(files)}", flush=True)
    b = pd.concat(rows)
    b["dt"] = pd.to_datetime(b["dt"])
    return (b.sort_values("dt").groupby(["SYMBOL", "ISIN"])
             .agg(first=("dt", "min"), last=("dt", "max"), series=("SERIES", lambda s: ",".join(sorted(set(s.dropna().astype(str))))))
             .reset_index().rename(columns={"SYMBOL": "symbol", "ISIN": "isin"}))


def _mode(df: pd.DataFrame, col: str) -> pd.Series:
    d = df[~df[col].astype(str).str.strip().str.lower().isin(BAD)].dropna(subset=[col])
    return d.groupby("symbol")[col].agg(lambda s: s.mode().iloc[0])


def main() -> None:
    refs = _fetch_refs()
    print("bhavcopy ISINs…", flush=True)
    bi = _isins_from_bhavcopies()
    # latest ISIN per symbol (a symbol can change ISIN on a face-value split)
    isin = bi.sort_values("last").groupby("symbol").tail(1).set_index("symbol")["isin"]
    eq = refs["equity_l"].rename(columns={"SYMBOL": "symbol", "ISIN NUMBER": "isin"})
    isin = pd.concat([isin, eq.set_index("symbol")["isin"]]); isin = isin[~isin.index.duplicated(keep="last")]
    etf = refs["etf_list"].rename(columns={"Symbol": "symbol", "ISINNumber": "isin"})
    isin = pd.concat([isin, etf.set_index("symbol")["isin"]]); isin = isin[~isin.index.duplicated(keep="last")]

    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["symbol", "trade_date"])
    span = px.groupby("symbol")["trade_date"].agg(["min", "max"]).rename(columns={"min": "first_trade", "max": "last_trade"})
    m = span.copy()
    m["isin"] = m.index.map(isin)
    m["is_fund_unit"] = m["isin"].astype(str).str.startswith("INF") | m.index.isin(set(etf["symbol"]))

    sm = _mode(pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "smIndustry"]), "smIndustry")
    hint = _mode(pd.read_parquet(ROOT / "data/events_full_history/normalized/stock_announcements.parquet",
                                 columns=["symbol", "industry_hint"]), "industry_hint")
    m["industry"] = m.index.map(sm); m["industry_source"] = m["industry"].notna().map({True: "nse_smIndustry", False: None})
    h = m["industry"].isna() & m.index.isin(hint.index)
    m.loc[h, "industry"] = m.index[h].map(hint); m.loc[h, "industry_source"] = "nse_industry_hint"

    sc = refs["symbolchange"]
    nb: dict[str, set] = {}
    for o, n in zip(sc["old"], sc["new"]):
        nb.setdefault(o, set()).add(n); nb.setdefault(n, set()).add(o)
    by_isin = m.dropna(subset=["industry", "isin"]).groupby("isin")[["industry", "industry_source"]].first()
    for sym in m.index[m["industry"].isna()]:
        i = m.at[sym, "isin"]
        if isinstance(i, str) and i in by_isin.index:
            m.at[sym, "industry"] = by_isin.at[i, "industry"]; m.at[sym, "industry_source"] = by_isin.at[i, "industry_source"] + "_via_isin"
            continue
        seen, frontier = {sym}, set(nb.get(sym, ()))
        while frontier:
            x = frontier.pop(); seen.add(x)
            if x in m.index and pd.notna(m.at[x, "industry"]) and "_via_" not in str(m.at[x, "industry_source"]):
                m.at[sym, "industry"] = m.at[x, "industry"]; m.at[sym, "industry_source"] = f"{m.at[x, 'industry_source']}_via_rename:{x}"
                break
            frontier |= nb.get(x, set()) - seen
    m["renamed_to"] = m.index.map(lambda s: ",".join(sorted(sc.loc[sc["old"] == s, "new"])) or None)
    m["industry_source"] = m["industry_source"].fillna("unmapped")
    m = m.reset_index().rename(columns={"index": "symbol"})
    m.to_parquet(OUT, index=False)

    eqm = m[~m["is_fund_unit"]]
    stats = dict(symbols=len(m), fund_units=int(m["is_fund_unit"].sum()), equities=len(eqm),
                 equities_with_isin=int(eqm["isin"].notna().sum()),
                 industry_by_source=eqm["industry_source"].str.replace(r":.*", "", regex=True).value_counts().to_dict())
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="security_master", path=str(OUT.relative_to(ROOT)), key=["symbol"], producer="src/agentic/build_security_master.py",
        sources=dict(isin="raw cm*bhav.csv.zip 2016-2024 + EQUITY_L.csv + eq_etfseclist.csv (nsearchives)",
                     renames="symbolchange.csv (nsearchives)", industry="announcements_historical.smIndustry > stock_announcements.industry_hint > same-ISIN > rename chain"),
        columns=dict(symbol="NSE symbol as in the price panel", first_trade="first panel session", last_trade="last panel session",
                     isin="latest ISIN seen (INE = company, INF = MF/ETF unit)", is_fund_unit="ETF/MF unit: ISIN INF* or in NSE ETF list",
                     industry="NSE industry label (smIndustry taxonomy); NULL = unknown, never guessed",
                     industry_source="provenance of industry", renamed_to="new symbol(s) per NSE symbolchange.csv"),
        survivorship_note="industry_hint exists only for names filing in 2026 (survivors); backtests must report results with and without hint-sourced labels",
        stats=stats, updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
