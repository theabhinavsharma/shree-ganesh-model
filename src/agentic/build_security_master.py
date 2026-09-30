"""SECURITY MASTER — one row per NSE symbol ever seen: ISIN, instrument type (company / fund unit /
rights entitlement), rename chain, industry with provenance.
data/derived/security_master.parquet (+ .manifest.json).

Why (2026-09-23): three silent universe defects found in one day —
  * the ETF filter was a symbol regex: it dropped real companies (SKYGOLD, SHANTIGOLD,
    GOLDTECH) and let dozens of ETFs through (LIQUID, SILVER, SETFNIF50, EBBETF0433, ...);
  * the industry map (modal smIndustry of announcements_historical) covered 832 of 2,245
    core-band symbols, so the leader cell never saw ~60% of the universe;
  * the fallback (stock_announcements.industry_hint) only knows names alive in 2026 —
    survivor-only, so using it alone biases any backtest.
Sources, all official NSE and already on disk or on nsearchives:
  ISIN        raw bhavcopies under data/raw/nse_full_history_official that carry an ISIN column,
              then EQUITY_L.csv + eq_etfseclist.csv (current lists, nsearchives).
              On disk today only the old-format cm*bhav.csv.zip (SYMBOL, SERIES, ISIN) carry an ISIN,
              and they cover 2015-01-01..2019-12-31. From 2020 the NSE downloader stores
              sec_bhavdata_full_*.csv, which has NO ISIN column. The builder reads the header of every
              2020+ file (and of any UDiFF BhavCopy_NSE_CM_* file) and uses an ISIN column if one is
              present; the per-format coverage actually read goes to the manifest (isin_coverage).
              LIMITATION: a symbol that first traded in 2020+ and is not in today's EQUITY_L / ETF list
              (dead, renamed or relisted) has no ISIN (137 symbols in the 2026-09-23 build).
              TODO: extend the authoritative NSE downloader (src/ingest/nse) to also store an ISIN-bearing
              CM bhavcopy for 2020+ (the legacy cm*bhav.csv.zip for as long as NSE published it, the UDiFF
              BhavCopy_NSE_CM_* after). Availability on nsearchives is NOT verified here. Once such files
              are on disk under trade_date=*/ this builder picks them up with no code change.
  fund unit   fund_unit_source = the first rule that fires:
              'isin_INF'               ISIN prefix INF (MF/ETF units); companies are INE
              'nse_etf_list'           symbol is in NSE's current eq_etfseclist.csv
              'registered_name'        no INE ISIN and the registered name says Mutual Fund / ETF
              'rename:<SYM>'           no ISIN and a direct NSE symbolchange neighbour is a fund unit
                                       (NETFIT -> ITBEES; the NETFIT record carries the AMC's name)
              'symbol_pattern_no_isin' no ISIN and the symbol ends in ETF<digits> / BEES<digits>
              'manual'                 MANUAL_FUND_UNITS below; each entry carries its evidence
  rights ent. symbol <SYM>-RE / -RE<n> (NSE rights-entitlement naming; they trade in series BE/ST for a
              few sessions of a rights issue) or panel series 'RE'. Tradeable entitlements, not company
              shares: is_rights_entitlement, is_non_equity, excluded from the manifest's equities count.
  renames     nsearchives symbolchange.csv (old -> new, date)
  industry    1. announcements_historical.smIndustry (mode)    source 'nse_smIndustry'
              2. stock_announcements.industry_hint (mode)       source 'nse_industry_hint'
              3. same ISIN or rename chain carries 1/2          source '<src>_via_isin|rename'
              4. otherwise NULL — never guessed                 source 'unmapped'
Literal 'None' strings in the NSE feeds are treated as missing.
Consumers: is_fund_unit keeps its meaning (ETF/MF unit). To drop everything that is not a company share,
use is_non_equity (fund units + rights entitlements) or generate_hybrid_basket.non_equity().

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json -> confirmed):
  FIXED  build_security_master.py 12-13, 68, 153 (UNITS / data contract): the docstring and manifest
         said ISINs come from 'cm*bhav.csv.zip 2016-2024'; the files on disk cover 2015-2019 and the 2020+
         sec_bhavdata_full files carry no ISIN. The builder now scans every raw format (old cm*bhav zip,
         sec_bhavdata_full, UDiFF BhavCopy_NSE_CM), uses an ISIN column wherever one exists, and writes
         the coverage it actually read (files, files_with_isin, first/last date, unreadable files) to the
         manifest's isin_coverage and sources.isin. The no-ISIN gap is stated under limitations and the
         equities without an ISIN are listed in stats. Unreadable files are counted, not silently skipped.
  FIXED  build_security_master.py 106, 115-116, 126-128 (+ generate_hybrid_basket.py 105, 140)
         (PARSING / SURVIVORSHIP): NETFIT, NAVINIFTY and the rights entitlements DUCON-RE1, JAYKAY-RE1,
         KSHITIJ-RE, RATNA-RE, VHLTD-RE1 (and every future -RE) were counted as operating companies.
         The fund flag now crosses direct symbolchange renames into no-ISIN symbols (NETFIT <- ITBEES;
         also SDL24BEES / SDL26BEES <- NCPSESDL24 / NETFSDL26). NAVINIFTY is a documented manual fund
         unit. No-ISIN ETF<n>/BEES<n> symbols are fund units (AXISBPSETF, EBBETF0423, EBBETF0425).
         -RE<n> symbols and series RE are flagged as rights entitlements. New columns: fund_unit_source,
         is_rights_entitlement, is_non_equity, instrument_type, series_seen. The manifest's 'equities'
         now counts ~is_non_equity (it was ~is_fund_unit).
  NOT FIXED (other files, outside this change): NETFIT's made-up screener_backfill mcap (ETF price x
         NAM-INDIA share count) in build_mcap_pit / fetch_screener_mcap_backfill, and the 'Asset
         Management Company' label from fetch_screener_industry. Those two fetchers filter on
         is_fund_unit only. NETFIT becomes a fund unit on the next master build, but they need to switch
         to is_non_equity to drop -RE symbols. The research scripts that also call
         generate_hybrid_basket.non_equity() (sim_leader_cell_v2, sim_leader_portfolio_7x, anatomy_1p5x,
         build_mcap_pit, pocket_search_leader, event_materiality_study, research_panel.load_panel)
         drop all seven today through the exclusions file and the -RE rule. Reported results were NOT
         re-run.
  NOT FIXED (no data): no post-2019 ISIN source is on disk; see the TODO above.
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data/raw/nse_full_history_official"
PANEL = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/security_master.parquet"
CACHE = ROOT / "data/raw/nse_reference"
URLS = {"symbolchange": "https://nsearchives.nseindia.com/content/equities/symbolchange.csv",
        "equity_l": "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "etf_list": "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv"}
BAD = {"none", "nan", "", "-", "null"}

# Raw bhavcopy formats the NSE downloader can leave under trade_date=*/ (glob -> kind).
BHAV_KINDS = {"cm*bhav.csv.zip": "old_format_cm_bhav_zip",           # on disk 2015-2019; has ISIN
              "sec_bhavdata_full_*.csv": "sec_bhavdata_full_csv",     # on disk 2020+; no ISIN column
              "BhavCopy_NSE_CM_*": "udiff_bhavcopy_nse_cm"}           # UDiFF CM bhavcopy; has ISIN; none on disk
SYMBOL_COLS = ("SYMBOL", "TCKRSYMB")
SERIES_COLS = ("SERIES", "SCTYSRS")

FUND_NAME_RE = r"Mutual Fund|\bETF\b|Exchange Traded|Liquid BeES|Index Fund"
# No-ISIN symbols ending ETF<digits> / BEES<digits> are exchange-traded fund units (Bharat Bond EBBETF0423,
# AXISBPSETF, SDL24BEES). Applied ONLY where the master has no ISIN, so a company with an INE ISIN is safe.
NO_ISIN_FUND_SYMBOL_RE = re.compile(r"(?:ETF|BEES)\d*$")
# NSE rights-entitlement symbols: <SYM>-RE, <SYM>-RE1, <SYM>-RE2 ... (series BE/ST in sec_bhavdata_full).
# 2026-09-27 check: no equity symbol in the panel ends in -RE<digits> (the hyphenated equities are
# BAJAJ-AUTO, NAM-INDIA, MCDOWELL-N, MRO-TEK, ...).
RIGHTS_ENT_RE = re.compile(r"-RE\d*$", re.I)
RIGHTS_ENT_SERIES = {"RE"}
# Fund units with no ISIN on disk, no registered name and no symbolchange link. Evidence, not guesses.
MANUAL_FUND_UNITS = {
    "NAVINIFTY": ("audit 2026-09-27: traded 2023-09-25..2024-11-28 (after the ISIN-bearing bhavcopies end); no ISIN "
                  "in EQUITY_L or the ETF list; no registered name; close/NIFTYBEES mean 0.932, median 0.909, "
                  "std 0.057 over 307 sessions, i.e. a Nifty-tracking fund unit"),
}


def _fetch_refs() -> dict[str, pd.DataFrame]:
    # imported here so the classification helpers can be imported (and unit-tested) without the network stack
    from src.ingest.nse.api import _request_headers, _request_with_retries
    from src.ingest.nse.session import build_session
    CACHE.mkdir(parents=True, exist_ok=True)
    s = build_session(warm=True, referer="https://www.nseindia.com/")
    for k, u in URLS.items():
        p = CACHE / f"{k}.csv"
        try:
            r = _request_with_retries(s, u, request_headers=_request_headers(s, referer=None), referer=None, timeout=60)
            p.write_text(r.text)
        except Exception as e:                       # keep last good copy; say so loudly
            print(f"  ⚠ {k} fetch failed ({str(e)[:80]}) — using cached {p.name}" if p.exists() else f"  ❌ {k}: no copy")
    return _read_refs()


def _read_refs() -> dict[str, pd.DataFrame]:
    """The cached NSE reference CSVs (symbolchange, EQUITY_L, ETF list) as written by _fetch_refs."""
    out = {}
    for k in URLS:
        p = CACHE / f"{k}.csv"
        if p.exists():
            out[k] = pd.read_csv(p, header=None if k == "symbolchange" else "infer", skipinitialspace=True)
    sc = out["symbolchange"]; sc.columns = ["name", "old", "new", "date"]
    out["symbolchange"] = sc.apply(lambda c: c.str.strip() if c.dtype == object else c)
    out["equity_l"].columns = [c.strip() for c in out["equity_l"].columns]
    return out


# ---------------------------------------------------------------- ISIN from raw bhavcopies
def _open_text(f: Path):
    """Text handle on a raw bhavcopy (.csv, or the first member of a .zip)."""
    if f.suffix.lower() == ".zip":
        z = zipfile.ZipFile(f)
        return io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8", errors="replace")
    return open(f, encoding="utf-8", errors="replace")


def _header(f: Path) -> list[str]:
    with _open_text(f) as h:
        return [c.strip() for c in h.readline().strip().split(",")]


def _read_isin_rows(f: Path) -> pd.DataFrame | None:
    """SYMBOL/SERIES/ISIN rows of one raw file, or None when the file has no ISIN column."""
    cols = _header(f)
    up = {c.upper(): c for c in cols if c}
    if not up:
        raise ValueError(f"empty or headerless file: {f}")
    if "ISIN" not in up:
        return None
    sym = next((up[c] for c in SYMBOL_COLS if c in up), None)
    ser = next((up[c] for c in SERIES_COLS if c in up), None)
    if sym is None:
        return None
    use = {sym, up["ISIN"]} | ({ser} if ser else set())
    with _open_text(f) as h:
        d = pd.read_csv(h, usecols=lambda c: c.strip() in use, skipinitialspace=True, dtype=str)
    d.columns = [c.strip() for c in d.columns]
    ren = {sym: "SYMBOL", up["ISIN"]: "ISIN"}
    if ser:
        ren[ser] = "SERIES"
    d = d.rename(columns=ren)
    if "SERIES" not in d:
        d["SERIES"] = None
    for c in ("SYMBOL", "SERIES", "ISIN"):
        d[c] = d[c].astype("string").str.strip()
    d = d[d["ISIN"].notna() & ~d["ISIN"].str.lower().isin(BAD) & d["SYMBOL"].notna()]
    return d[["SYMBOL", "SERIES", "ISIN"]]


def _bhav_files(raw: Path = RAW) -> list[tuple[Path, str, str]]:
    """(path, kind, trade_date) for every raw bhavcopy under raw/trade_date=*/, sorted by date."""
    out = []
    for pat, kind in BHAV_KINDS.items():
        for f in raw.glob(f"trade_date=*/{pat}"):
            out.append((f, kind, f.parent.name.split("=")[1]))
    return sorted(out, key=lambda t: (t[2], t[1]))


def _isins_from_bhavcopies(files: list[tuple[Path, str, str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """(symbol, isin, first, last, series) from every raw file that has an ISIN column, plus the coverage
    actually read per format: files, files_with_isin / _without_isin / _failed, file and ISIN date spans."""
    files = _bhav_files() if files is None else files
    cov = {k: dict(files=0, files_with_isin=0, files_without_isin=0, files_failed=0,
                   first_file_date=None, last_file_date=None, isin_first_date=None, isin_last_date=None)
           for k in BHAV_KINDS.values()}
    rows = []
    for i, (f, kind, dt) in enumerate(files):
        c = cov[kind]; c["files"] += 1
        c["first_file_date"] = min(x for x in (c["first_file_date"], dt) if x)
        c["last_file_date"] = max(x for x in (c["last_file_date"], dt) if x)
        if i % 500 == 0:
            print(f"  bhavcopy {i}/{len(files)}", flush=True)
        try:
            d = _read_isin_rows(f)
        except Exception:
            c["files_failed"] += 1
            continue
        if d is None:
            c["files_without_isin"] += 1
            continue
        c["files_with_isin"] += 1
        c["isin_first_date"] = min(x for x in (c["isin_first_date"], dt) if x)
        c["isin_last_date"] = max(x for x in (c["isin_last_date"], dt) if x)
        d["dt"] = dt
        rows.append(d.drop_duplicates(["SYMBOL", "ISIN"]))
    if not rows:
        return pd.DataFrame(columns=["symbol", "isin", "first", "last", "series"]), cov
    b = pd.concat(rows)
    b["dt"] = pd.to_datetime(b["dt"])
    b = (b.sort_values("dt").groupby(["SYMBOL", "ISIN"])
          .agg(first=("dt", "min"), last=("dt", "max"), series=("SERIES", lambda s: ",".join(sorted(set(s.dropna().astype(str))))))
          .reset_index().rename(columns={"SYMBOL": "symbol", "ISIN": "isin"}))
    b["symbol"] = b["symbol"].astype(str); b["isin"] = b["isin"].astype(str)
    return b, cov


def isin_source_text(cov: dict) -> str:
    """ISIN provenance built from the coverage actually read (never a hard-coded date range)."""
    parts = []
    for kind, c in cov.items():
        if not c["files"]:
            parts.append(f"{kind}: 0 files on disk")
        elif c["files_with_isin"]:
            parts.append(f"{kind}: ISIN {c['isin_first_date']}..{c['isin_last_date']} "
                         f"({c['files_with_isin']}/{c['files']} files, {c['files_failed']} unreadable)")
        else:
            parts.append(f"{kind}: {c['first_file_date']}..{c['last_file_date']}, {c['files']} files, NO ISIN column")
    return "raw bhavcopies [" + "; ".join(parts) + "] + EQUITY_L.csv + eq_etfseclist.csv (current lists, nsearchives)"


# ---------------------------------------------------------------- instrument classification
def rename_graph(sc: pd.DataFrame) -> dict[str, set]:
    """Undirected symbolchange graph: symbol -> set of direct old/new neighbours."""
    nb: dict[str, set] = {}
    for o, n in zip(sc["old"], sc["new"]):
        if isinstance(o, str) and isinstance(n, str):
            nb.setdefault(o, set()).add(n); nb.setdefault(n, set()).add(o)
    return nb


def classify_instruments(m: pd.DataFrame, isin_map: pd.Series, etf_symbols: set, sc: pd.DataFrame) -> pd.DataFrame:
    """Add is_fund_unit, fund_unit_source, is_rights_entitlement, is_non_equity, instrument_type.

    m: indexed by symbol with columns isin, company_name and (optional) series_seen (comma-joined series).
    isin_map: symbol -> latest ISIN for every symbol known (in the panel or not); used for rename neighbours.
    sc: symbolchange frame with columns old, new.
    Rule order is in the module docstring. A symbol with an INE ISIN is never made a fund unit by the
    name / rename / pattern / manual rules."""
    m = m.copy()
    isin_s = m["isin"].astype(str)
    no_isin = m["isin"].isna()
    has_ine = isin_s.str.startswith("INE")
    etf_symbols = set(etf_symbols)
    src = pd.Series(None, index=m.index, dtype=object)
    src[isin_s.str.startswith("INF")] = "isin_INF"
    src[src.isna() & m.index.isin(etf_symbols)] = "nse_etf_list"
    by_name = m["company_name"].astype(str).str.contains(FUND_NAME_RE, case=False, regex=True)
    src[src.isna() & by_name & ~has_ine] = "registered_name"

    nb = rename_graph(sc)

    def _is_fund(x: str) -> bool:
        if x in src.index:
            return pd.notna(src.at[x])
        return str(isin_map.get(x, "")).startswith("INF") or x in etf_symbols

    changed = True
    while changed:                               # fixed point: a chain of no-ISIN renames inherits too
        changed = False
        for sym in m.index[src.isna() & no_isin]:
            for x in sorted(nb.get(sym, ())):
                if _is_fund(x):
                    src.at[sym] = f"rename:{x}"; changed = True
                    break
    sym_s = m.index.to_series()
    src[src.isna() & no_isin & sym_s.str.contains(NO_ISIN_FUND_SYMBOL_RE)] = "symbol_pattern_no_isin"
    src[src.isna() & no_isin & sym_s.isin(list(MANUAL_FUND_UNITS))] = "manual"

    m["is_fund_unit"] = src.notna()
    m["fund_unit_source"] = src
    series = m["series_seen"] if "series_seen" in m else pd.Series("", index=m.index)
    by_series = series.fillna("").astype(str).str.split(",").map(lambda xs: bool(RIGHTS_ENT_SERIES & {x.strip() for x in xs}))
    by_sym = sym_s.str.contains(RIGHTS_ENT_RE)
    m["is_rights_entitlement"] = (by_sym | by_series.astype(bool)) & ~m["is_fund_unit"]
    m["is_non_equity"] = m["is_fund_unit"] | m["is_rights_entitlement"]
    m["instrument_type"] = np.select(
        [m["is_fund_unit"].values, m["is_rights_entitlement"].values, isin_s.str.match(r"IN[E9]").values],
        ["fund_unit", "rights_entitlement", "equity"], "equity_no_isin")
    return m


def _mode(df: pd.DataFrame, col: str) -> pd.Series:
    d = df[~df[col].astype(str).str.strip().str.lower().isin(BAD)].dropna(subset=[col])
    return d.groupby("symbol")[col].agg(lambda s: s.mode().iloc[0])


def build_master(refs: dict[str, pd.DataFrame], bi: pd.DataFrame, panel: Path = PANEL) -> pd.DataFrame:
    """The master frame (no I/O besides reading panel / names / industry sources)."""
    # latest ISIN per symbol (a symbol can change ISIN on a face-value split)
    isin = bi.sort_values("last").groupby("symbol").tail(1).set_index("symbol")["isin"]
    eq = refs["equity_l"].rename(columns={"SYMBOL": "symbol", "ISIN NUMBER": "isin"})
    isin = pd.concat([isin, eq.set_index("symbol")["isin"]]); isin = isin[~isin.index.duplicated(keep="last")]
    etf = refs["etf_list"].rename(columns={"Symbol": "symbol", "ISINNumber": "isin"})
    isin = pd.concat([isin, etf.set_index("symbol")["isin"]]); isin = isin[~isin.index.duplicated(keep="last")]

    px = pd.read_parquet(panel, columns=["symbol", "trade_date", "series"])
    span = px.groupby("symbol")["trade_date"].agg(["min", "max"]).rename(columns={"min": "first_trade", "max": "last_trade"})
    ser = px[["symbol", "series"]].dropna().drop_duplicates()
    series_seen = ser.assign(series=ser["series"].astype(str).str.strip()).groupby("symbol")["series"].agg(lambda s: ",".join(sorted(set(s))))
    del px, ser
    m = span.copy()
    m["isin"] = m.index.map(isin)
    # dead ETFs have no ISIN on record: flag by registered name (announcements / CA / symbolchange)
    names = pd.concat([refs["symbolchange"].set_index("old")["name"],
                       pd.read_parquet(ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet",
                                       columns=["symbol", "company_name"]).dropna().drop_duplicates("symbol").set_index("symbol")["company_name"],
                       pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "sm_name"])
                         .dropna().drop_duplicates("symbol").set_index("symbol")["sm_name"]])
    names = names[~names.index.duplicated(keep="last")]
    m["company_name"] = m.index.map(names)
    m["series_seen"] = m.index.map(series_seen)
    m = classify_instruments(m, isin, set(etf["symbol"]), refs["symbolchange"])

    sm = _mode(pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "smIndustry"]), "smIndustry")
    hint = _mode(pd.read_parquet(ROOT / "data/events_full_history/normalized/stock_announcements.parquet",
                                 columns=["symbol", "industry_hint"]), "industry_hint")
    m["industry"] = m.index.map(sm); m["industry_source"] = m["industry"].notna().map({True: "nse_smIndustry", False: None})
    h = m["industry"].isna() & m.index.isin(hint.index)
    m.loc[h, "industry"] = m.index[h].map(hint); m.loc[h, "industry_source"] = "nse_industry_hint"

    sc = refs["symbolchange"]
    nb = rename_graph(sc)
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
    m = m[["symbol", "first_trade", "last_trade", "isin", "is_fund_unit", "company_name", "industry", "industry_source",
           "renamed_to", "fund_unit_source", "is_rights_entitlement", "is_non_equity", "instrument_type", "series_seen"]]
    return m


def master_stats(m: pd.DataFrame) -> dict:
    eqm = m[~m["is_non_equity"]]
    no_isin_eq = eqm.loc[eqm["isin"].isna(), "symbol"]
    stats = dict(symbols=len(m), fund_units=int(m["is_fund_unit"].sum()),
                 fund_units_by_source=m["fund_unit_source"].dropna().str.replace(r":.*", "", regex=True).value_counts().to_dict(),
                 rights_entitlements=int(m["is_rights_entitlement"].sum()),
                 non_equity=int(m["is_non_equity"].sum()), equities=len(eqm),
                 equities_with_isin=int(eqm["isin"].notna().sum()),
                 equities_no_isin=int(len(no_isin_eq)), equities_no_isin_symbols=sorted(no_isin_eq.tolist()),
                 industry_by_source=eqm["industry_source"].str.replace(r":.*", "", regex=True).value_counts().to_dict())
    return stats


def main() -> None:
    refs = _fetch_refs()
    print("bhavcopy ISINs…", flush=True)
    bi, cov = _isins_from_bhavcopies()
    print(json.dumps(cov, indent=1), flush=True)
    m = build_master(refs, bi)
    m.to_parquet(OUT, index=False)
    stats = master_stats(m)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="security_master", path=str(OUT.relative_to(ROOT)), key=["symbol"], producer="src/agentic/build_security_master.py",
        sources=dict(isin=isin_source_text(cov),
                     renames="symbolchange.csv (nsearchives)", industry="announcements_historical.smIndustry > stock_announcements.industry_hint > same-ISIN > rename chain",
                     manual_fund_units=MANUAL_FUND_UNITS),
        isin_coverage=cov,
        limitations=["ISIN-bearing raw bhavcopies on disk end where isin_coverage says (2019-12-31 as of 2026-09-27); the 2020+ "
                     "sec_bhavdata_full files have no ISIN column. A symbol that first traded later and is not in today's "
                     "EQUITY_L / ETF list has isin=NULL. Its type then rests on the ETF list, registered name, rename links, "
                     "the ETF/BEES symbol pattern and MANUAL_FUND_UNITS. instrument_type='equity_no_isin' means none of "
                     "those fired and the equity status is unverified (see stats.equities_no_isin_symbols).",
                     "company_name for renamed-away symbols comes from symbolchange.csv and can be the AMC rather than the "
                     "fund (NETFIT -> 'Nippon Life India Asset Management Limited'), so the name alone is not a type signal."],
        columns=dict(symbol="NSE symbol as in the price panel", first_trade="first panel session", last_trade="last panel session",
                     isin="latest ISIN seen (INE = company, IN9 = partly paid equity, INF = MF/ETF unit); NULL = none on disk",
                     is_fund_unit="ETF/MF unit (the rule that fired is in fund_unit_source)",
                     company_name="registered name (NSE symbolchange / CA store / announcements)",
                     industry="NSE industry label (smIndustry taxonomy); NULL = unknown, never guessed",
                     industry_source="provenance of industry", renamed_to="new symbol(s) per NSE symbolchange.csv",
                     fund_unit_source="first rule that made it a fund unit: isin_INF | nse_etf_list | registered_name | rename:<SYM> | symbol_pattern_no_isin | manual; NULL = not a fund unit",
                     is_rights_entitlement="rights entitlement: symbol ends -RE / -RE<n>, or panel series RE",
                     is_non_equity="is_fund_unit OR is_rights_entitlement; drop these for company-share universes",
                     instrument_type="fund_unit | rights_entitlement | equity (INE/IN9 ISIN) | equity_no_isin (no ISIN and no non-equity rule fired; unverified)",
                     series_seen="comma-joined panel series the symbol traded in"),
        survivorship_note="industry_hint exists only for names filing in 2026 (survivors); backtests must report results with and without hint-sourced labels",
        stats=stats, updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
