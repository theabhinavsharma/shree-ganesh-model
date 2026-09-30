"""Refresh the corporate-actions store (splits/bonuses) from NSE.

Extracted 2026-08-27 from the ad-hoc heredocs in monday_run_20260824.sh /
friday_run_20260828.sh so the weekly pipeline can refresh CAs as a proper step —
and BEFORE refresh_prices.py, so price adjustment always sees today's actions
(the late-arriving-CA bug class: CORDELIA/TDPOWERSYS/GOODLUCK/KIRLPNU 2026-08-27,
TRENT/LICI "113-day rot" 2026-08-18).

Fetch window: trailing 30 days (idempotent merge, dedup on symbol/ex_date/subject/series).
Exit 1 on fetch failure so run scripts can decide fatality.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT))
from src.ingest.corporate_actions.nse import (  # noqa: E402
    NseCorporateActionsFetchConfig,
    _parse_bonus_factor,
    _parse_split_factor,
    load_corporate_actions_from_nse,
)

STORE = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"
SYMCHG = ROOT / "data/raw/nse_symbol_change/symbolchange.csv"
SYMCHG_URL = "https://nsearchives.nseindia.com/content/equities/symbolchange.csv"


def fetch_symbol_changes() -> None:
    """NSE's official symbol-change list (old symbol, new symbol, date). Kept if a refresh fails."""
    import subprocess
    SYMCHG.parent.mkdir(parents=True, exist_ok=True)
    tmp = SYMCHG.with_suffix(".tmp")
    r = subprocess.run(["/usr/bin/curl", "-s", "-m", "60", "-A", "Mozilla/5.0", "-H", "Referer: https://www.nseindia.com/",
                        "-o", str(tmp), SYMCHG_URL], capture_output=True)
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 10_000:
        tmp.replace(SYMCHG)
    elif tmp.exists():
        tmp.unlink()


def alias_renamed(store: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """2026-09-30 fix: NSE lists a company's corporate actions under its CURRENT symbol, but the price panel keeps each
    day under the symbol traded THAT day. So a split before a rename (CADILAHC 2015 split, listed under ZYDUSLIFE) was
    never applied to the CADILAHC rows: 398 one-day 'crashes' in the adjusted panel. For every NSE record, add a copy
    keyed to the symbol the company traded under on its ex-date (NSE symbolchange.csv chain). Copies carry
    mapped_from_symbol = the NSE symbol; they are rebuilt on every run."""
    if not SYMCHG.exists():
        return store, 0
    sc = pd.read_csv(SYMCHG, header=None, names=["company", "old", "new", "date"], dtype=str)
    sc["old"], sc["new"] = sc["old"].str.strip(), sc["new"].str.strip()
    sc["date"] = pd.to_datetime(sc["date"].str.strip(), format="%d-%b-%Y", errors="coerce")
    sc = sc.dropna(subset=["old", "new", "date"])
    back = {}                                          # new symbol -> [(change date, old symbol)]
    for r in sc.itertuples():
        back.setdefault(r.new, []).append((r.date, r.old))
    base = store[store.get("mapped_from_symbol", pd.Series(index=store.index, dtype=object)).isna()].copy() \
        if "mapped_from_symbol" in store.columns else store.copy()
    base["mapped_from_symbol"] = None
    add = []
    for r in base.itertuples(index=False):
        if pd.isna(r.ex_date):
            continue
        sym, seen = r.symbol, set()
        while sym in back and sym not in seen:           # walk back while the rename happened after the ex-date
            seen.add(sym)
            prev = [o for d, o in sorted(back[sym], reverse=True) if d > r.ex_date]
            if not prev:
                break
            sym = prev[-1]                                 # the earliest rename after the ex-date gives the symbol then
        if sym != r.symbol:
            row = r._asdict(); row.update(symbol=sym, mapped_from_symbol=r.symbol); add.append(row)
    # second link (2026-09-30): the issuer part of the ISIN, for renames NSE's list misses (PHILIPCARB -> PCBL).
    # data/derived/symbol_isin_history.parquet (NSE bhavcopies 2015-2019) says which symbol traded that issuer's ISIN when.
    hist_f = ROOT / "data/derived/symbol_isin_history.parquet"
    if hist_f.exists() and "isin" in base.columns:
        H = pd.read_parquet(hist_f)
        by_issuer = {k: g for k, g in H.groupby("issuer")}
        have = {(a["symbol"], a["ex_date"], a["subject"]) for a in add}
        for r in base.itertuples(index=False):
            isin = str(getattr(r, "isin", "") or "").strip()
            if pd.isna(r.ex_date) or len(isin) < 9 or isin[:9] not in by_issuer:
                continue
            g = by_issuer[isin[:9]]
            live = g[(g["first"] <= r.ex_date + pd.Timedelta(days=10)) & (g["last"] >= r.ex_date - pd.Timedelta(days=10))]
            for sym in sorted(set(live["symbol"]) - {r.symbol}):
                if (sym, r.ex_date, r.subject) not in have:
                    row = r._asdict(); row.update(symbol=sym, mapped_from_symbol=r.symbol); add.append(row); have.add((sym, r.ex_date, r.subject))
    out = pd.concat([base, pd.DataFrame(add)], ignore_index=True) if add else base
    return out, len(add)
WINDOW_DAYS = 30


def main() -> None:
    start = date.today() - timedelta(days=WINDOW_DAYS)
    new = load_corporate_actions_from_nse(NseCorporateActionsFetchConfig(
        output_dir=ROOT / "data/corporate_actions_full_history/_incremental",
        start_date=start,
        end_date=date.today(),
    ))
    new["ex_date"] = pd.to_datetime(new["ex_date"], errors="coerce")
    full = pd.read_parquet(STORE)
    full["ex_date"] = pd.to_datetime(full["ex_date"], errors="coerce")
    key = ["symbol", "ex_date", "subject", "series"]
    merged = (
        pd.concat([full, new], ignore_index=True)
        .drop_duplicates(subset=key, keep="last")
        .sort_values(["ex_date", "symbol"])
        .reset_index(drop=True)
    )
    healed = _reparse_factorless(merged)
    fetch_symbol_changes()
    merged, n_alias = alias_renamed(merged)
    print(f"RENAME ALIASES: {n_alias} NSE corporate actions copied to the symbol traded on their ex-date")
    merged.to_parquet(STORE, index=False)
    if healed:
        print(f"STORE SELF-HEAL: {healed} historical rows gained a split/bonus factor "
              f"from the current parser (refresh_prices.py re-adjusts their symbols next)")
    fresh = new[new["adjustment_factor"].notna()]
    print(f"store: {len(merged):,} rows · max ex_date {merged['ex_date'].max().date()} · "
          f"fetched {len(new)} rows ({len(fresh)} with factors) since {start}")
    if len(fresh):
        print(fresh[["symbol", "ex_date", "subject", "adjustment_factor"]].to_string(index=False))


def _reparse_factorless(store: pd.DataFrame) -> int:
    """Re-run the CURRENT subject parser over the whole store; return rows whose
    factor changed.

    2026-09-19: the 2026-08-27 regex fix only ever touched newly fetched rows, so
    59 historical splits (JSWSTEEL/KARURVYSYA/TTL/DBEIL class) stayed factor-less
    and 13 same-day bonus+split pairs carried a half factor (CUPID 2 vs raw 12.6x,
    SBC 2 vs 29.6x) for three weeks after the parser could read them. Verified
    against raw ex-date close ratios: 23/24 changed rows move CLOSER to the tape.
    The store is a pure function of (NSE subject text, current parser) — nothing
    is hand-edited (source_note is uniform) — so re-parsing everything is safe and
    idempotent. refresh_prices.py's LATE-CA SELF-HEAL re-adjusts any symbol whose
    per-ex_date product changed.
    """
    bonus = store["subject"].map(_parse_bonus_factor)
    split = store["subject"].map(_parse_split_factor)
    factor = (bonus.fillna(1.0) * split.fillna(1.0)).where(bonus.notna() | split.notna())
    changed = factor.fillna(-1.0) != store["adjustment_factor"].astype(float).fillna(-1.0)
    if not changed.any():
        return 0
    store["bonus_factor"] = bonus
    store["split_factor"] = split
    store["is_bonus"] = bonus.notna()
    store["is_split"] = split.notna()
    store["adjustment_factor"] = factor
    return int(changed.sum())

def alias_only() -> None:
    """Re-key the stored actions to historical symbols without fetching new actions (--alias-only)."""
    fetch_symbol_changes()
    full = pd.read_parquet(STORE); full["ex_date"] = pd.to_datetime(full["ex_date"], errors="coerce")
    out, n = alias_renamed(full)
    out.to_parquet(STORE, index=False)
    print(f"RENAME ALIASES: {n} NSE corporate actions copied to the symbol traded on their ex-date · store rows {len(out)}")


if __name__ == "__main__":
    alias_only() if "--alias-only" in sys.argv else main()
