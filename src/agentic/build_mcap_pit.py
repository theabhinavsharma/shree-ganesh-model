"""MARKET CAP PANEL — data/derived/mcap_pit.parquet (+ .manifest.json).

One row per panel session per equity (research_panel.load_panel: ISIN-master fund units and non_equity removed).

mcap_cr(t) = raw_price(t) x shares_present / share_factor(t) / 1e7
  raw_price(t)    research_panel.raw_price: adjusted close / price_adjustment_factor_to_present = the then-traded
                  price (the price factor may include price-only corporate actions such as demergers).
  share_factor(t) the rename chain's share factor at t ("chain factor") =
                    share_adjustment_factor_to_present of the row's own symbol (splits/bonuses after t; NaN/0 -> 1)
                  x each CHAINED successor's factor at its first session (one present basis for the whole chain)
                  x every split/bonus the CA store files under a LATER symbol of the rename family with ex_date on or
                    before that symbol's first session and after t ("orphan" events). The CA store files history under
                    the current symbol and the panel adjuster matches events by symbol, so e.g. TRIL's 2017-09-28 10:1
                    split (filed under TARIL) and MOTHERSUMI's 2015/2017/2018 bonuses (filed under MOTHERSON) are not
                    in the old symbols' panel factors; 29 rename pairs have such events on the 2026-09-27 backup panel.
  Rename links, NSE symbolchange via research_panel.rename_map (old -> new, both in the panel):
    contiguous         successor's first session 1..10 days after the old symbol's last (the stitch_renames rule): CHAINED.
    bridged_same_isin  longer gap and the same security_master ISIN on both sides: CHAINED. An ISIN changes on a
                       face-value split, a consolidation or a capital reduction, so equal ISINs leave only a bonus as a
                       share event the link itself cannot rule out; bonuses the CA store files under either symbol are
                       applied through the chain factor (orphan events). Most such gaps were panel holes (missing BE/BZ
                       sessions): on the repaired panel 17 of the 19 same-ISIN links seen here are contiguous.
    unbridged_*        isin_differs / isin_unknown / overlap: NOT chained. The successor's share count and screener page
                       are never applied to the old symbol. The company's own P&L filings, which pnl_quarterly stores under
                       the successor symbol, ARE pooled for the old symbol's rows (filings dated up to its last session,
                       rebased with its own factor): then-shares from PAT/EPS are point-in-time whatever the symbol.
                       The successor's rows use only filings dated from its chain's first session, because the share
                       basis across the unbridged gap is unknown.
  shares_present  share count on the chain's present basis, from the first source that has it:
   1 'pnl_implied'  (pit_ok=True) PAT / basic EPS per quarterly filing (units per the pnl_quarterly manifest:
                    detail_api Rs lakh, xbrl Rs; |EPS| >= 0.05, PAT != 0, shares > 0), rebased to the present basis
                    with the chain factor at the filing date, then the median over the latest 4 usable filings known
                    at t, ordered by filing_dt (a re-filing replaces the earlier version of its quarter), computed
                    separately per basis. The chain's primary basis (standalone unless it has under half as many
                    usable filings as consolidated) sizes the row; the other basis's own median is used only where
                    the primary has no filing known within 400 days (before its first filing, or a gap). A filing is known from its filing date if stamped before 15:30 IST, otherwise from the
                    next day (date-only stamps are treated as after-hours). Stale after 400 days.
   2 'screener_backcast' (pit_ok=False) screener_fundamentals latest fetch: market_cap / current_price at fetch_date,
                    rebased with the chain factor at fetch_date. Present-day share count: ignores later
                    QIP/preferential/merger/buyback changes. Fills ANY row still NULL, including rows of symbols
                    that have pnl filings (before the first filing, or after the 400-day staleness limit).
   3 'screener_backfill' (pit_ok=False) screener_mcap_backfill.parquet (fetch_screener_mcap_backfill.py), same
                    arithmetic. A page is accepted only if its URL slug is the symbol itself or a member of the
                    symbol's CHAINED rename chain (then rebased through the chain factor). A page of an unbridged
                    NSE successor is rejected (rejected_rename_unbridged); any other slug (re-listings under a new ISIN
                    after a scheme or capital reduction such as HEXAWARE->HEXT or TUBEINVEST->TIINDIA, other companies
                    such as BSLIMITED->BSE or NETFIT->NAM-INDIA, BSE-code pages such as CMC->538319) is rejected
                    (rejected_slug_mismatch).
Rows before 2018-04 are NEVER point-in-time: pnl_quarterly filing_dt starts 2018-04-02.
NULL = unknown, never guessed; null_reason says why.

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json, confirmed findings for this file):
  FIXED  build_mcap_pit.py 40,46-53 (high, split/bonus share basis): filing shares are rebased to the present basis
         with the share factor at the filing date before the median and converted back with the factor at t, so a
         split no longer understates mcap for months (NYKAA 5:1 bonus 2022-11-10, VERTOZ 1:10 2024-07-08). NaN latest
         filings no longer leave a stale pre-split count: the carried median is already on the present basis.
  FIXED  build_mcap_pit.py 49-54 (low, same split lag seen via sim_leader_portfolio_7x H lever): same fix; the
         'PIT' wording is replaced by the pit_ok column (True only for pnl_implied rows).
  FIXED  build_mcap_pit.py 65-73 + fetch_screener_mcap_backfill.py 1-8,35-45 (high, wrong entity's share count):
         backfill pages are accepted only when the slug is the symbol or a CHAINED rename-chain member (rebased
         through the chain factor); NIITTECH/SRTRANSFIN/FAGBEARING/ANGELBRKG are rebased via their successors,
         CMC (BSE 538319 page), ALOKTEXT, HEXAWARE, DIAPOWER, 3IINFOTECH are rejected (NULL).
  FIXED  build_mcap_pit.py 1-13,56-63,65-73,77-92 (high, not PIT before 2018-04 / era asymmetry): the docstring and
         manifest no longer call the dataset point-in-time; new column pit_ok; the manifest reports source mix and
         pit_ok share per era (disc 2016-2022 / conf 2023+) and per year, and validates backcast shares against
         pnl-implied shares by year (history, not just the latest row). The one-way future-share bias itself is
         NOT removable without historical share counts (see NOT FIXED).
  FIXED  build_mcap_pit.py 56-73 + fetch_screener_mcap_backfill.py 35-37 (medium, row-level cascade vs
         symbol-level backfill): the cascade stays row-level, and the fetcher now targets every equity without a
         screener_fundamentals share count (not only symbols with no sized row), so symbols with pnl filings get a
         screener share count for rows before their first filing. The DATA gain needs a fetcher re-run; until
         then those rows stay NULL with null_reason 'before_first_filing|screener_not_fetched'.
  FIXED  build_mcap_pit.py 42-43 (low, basis-blind dedupe): one primary basis per chain (standalone preferred); the
         per-quarter dedupe and the median run inside a basis; ties are ordered by (filing_dt, quarter_end).
  FIXED  build_mcap_pit.py 57-62 (low, backcast shares vs adjustment date): screener shares are rebased with the
         share factor at fetch_date (TCC 5:1 2026-09-04, PGIL 2:1 2026-09-11 fetched 2026-08-30).
  FIXED  build_mcap_pit.py 43,48-52 (low, rolling median in quarter_end order / after-hours filings): median over
         the latest 4 filings known at t ordered by filing_dt; filings stamped at/after 15:30 are known next day.
  NOT FIXED  pre-2018-04 rows and all screener rows use present-day (or last-trade) share counts: no historical
         share-count source is on disk (TODO: NSE shareholding-pattern total shares, if an official archive exists).
  NOT FIXED  consolidations/reverse splits and capital reductions without an adjustment_factor in the CA store
         (critic item 1) are not in share_adjustment_factor_to_present, so pnl_implied mcap is off for up to 4
         filings after such an event. Upstream fix: src/transform/corporate_actions.py.
  NOT FIXED  mergers completed under a chained rename (e.g. SRTRANSFIN -> SHRIRAMFIN) backcast the post-merger
         share count onto pre-2018 rows of the old symbol (pit_ok=False; 2018+ rows use the entity's own filings).
  NOT FIXED  the basis choice (standalone vs consolidated) counts filings over the whole history: a method choice,
         not a value, but technically not PIT.
  NOT FIXED  screener_fundamentals.parquet has no URL column, so its slug cannot be checked (it was fetched from
         /company/<SYMBOL>/ by symbol; a server-side redirect to a successor would go unnoticed). Records of
         unbridged old symbols are dropped (none today).

2026-09-27 review round 2 (two independent reviews of the fixes above). Measured with an in-memory full-universe
build on the 2026-09-27 backup panel (4.78M rows; nothing written), "orig" = mcap_pit.parquet as built before the audit:
  FIXED  lines 107-117, 165-169, 244-261 (medium): the contiguity-only chain rule turned same-company renames with a
         >10-day panel gap into NULL rows (GET&D: all 1,728) and dropped the company's own P&L filings stored under
         the successor. Now (a) same-ISIN renames are bridged (21 links on the backup panel: GET&D, TRIL, MAGMA,
         JUBILANT, LSIL, ORIENTABRA, ATLANTA, ...), (b) P&L filings are pooled through the NSE rename family whether
         or not the link is chained (WEIZFOREX, INDOSOLAR), and (c) CA events filed under a later symbol are applied
         to earlier symbols' rows (orphan events). The 68 pages rejected in round 1 were 19 same-ISIN NSE renames,
         12 NSE renames with a different or unknown ISIN, and 37 non-rename pages (36 panel equities; NETFIT is a
         fund unit); the round-1 report called all 68 'another company or a re-listing', which was wrong for the 31
         NSE renames.
         Rows sized in orig but NULL: round 1 63,728 (51,079 at >= Rs 50 cr), now 38,741 (29,115):
           30,740  no_usable_pnl_filings|screener_rejected_slug_mismatch: 36 non-rename symbols whose only page is a
                   different listing (new ISIN after a scheme/capital reduction, another company, a BSE code); orig
                   sized them with that listing's share count. Kept NULL on purpose. None of them has P&L filings on
                   disk (pnl_quarterly holds nothing for the old listing), so 8,450 of these rows from 2018-04 on
                   stay NULL too. TODO: fetch quarterly results for the old symbols themselves.
            6,457  unbridged NSE renames (12 links: isin_differs 6, isin_unknown 6); on the repaired panel 7 of them
                   become contiguous (GISOLUTION, REVATHIEQU, SONAMCLOCK, SPYL, 4THDIM, GAMMNINFRA, HOTELRUGBY).
            1,544  P&L policy: after-hours filings known next day (312 symbols, 1 row each), staleness counted from the
                   last usable filing (VAKRANGEE, RAMASTEEL, NILAINFRA ...: EPS 0.00-0.04 filings no longer reset the
                   400-day clock as orig's NaN-skipping rolling median did), successor guard (WAAREEINDO, BALAXI).
         Median names >= Rs 50 cr per session, orig / round 1 / now: 2016Q1 980 / 949 / 961, 2018Q1 1026 / 1007 /
         1017, 2020Q1 1249 / 1234 / 1243, 2023Q1 1631 / 1623 / 1631. Sized rows by era, orig -> now: disc
         2,374,786 -> 2,355,569, conf 1,834,692 -> 1,834,258, pre2016 268,281 -> 261,457.
  FIXED  lines 107-117, 246-256 + fetch_screener_mcap_backfill.py 63-71 (medium): same root cause; the fetcher no
         longer fetches ANY NSE old symbol (screener answers /company/<OLD>/ with the successor's page: 237 of 255
         checkpoint records redirected, 17 were HTTP 404, 1 (DTIL) had its own page), so an old symbol is sized
         through its successor's page only when the builder chains the link, and otherwise stays NULL with
         null_reason '...|screener_rename_unbridged_<reason>' instead of looping through SLUG_MISMATCH.
  FIXED  (found in the round-2 full-universe check) the round-1 one-basis rule had dropped the other basis entirely,
         so rows before the primary basis's first filing or across a gap in it became NULL (STANLEY's first
         consolidated filing 2024-07-19 precedes its first standalone one 2024-08-14; FLAIR has no standalone filing
         2025-01-31..2026-05-21). The other basis's own median now fills only those rows (manifest
         pnl.rows_sized_by_fallback_basis); the two bases are never mixed in one median.
  FIXED  (found in the round-2 check) orphan events: the CA store files splits/bonuses under the CURRENT symbol, so a
         pre-rename symbol's own factor missed them (29 rename pairs; e.g. MINDAIND 5:1 2016-09-12 filed under
         UNOMINDA: orig and round 1 sized MINDAIND at Rs 87,871 cr on 2016-09-09 and 16,787 cr on 2016-09-12, now
         2,929 -> 2,798 cr; TRIL 10:1 2017-09-28: orig 9,218 -> 955 cr, now 461 -> 478 cr; MOTHERSUMI 1:2 bonus
         2018-10-30 (pnl_implied): round 1 48,051 -> 34,003 cr, now 48,051 -> 51,005 cr).
  NOT FIXED  the reviewer's raw-price continuity gate (successor first / old last within [0.67, 1.5]) is reported per
         bridged link in the manifest (raw_price_ratio, present_basis_price_ratio) but not used as a gate: it
         rejects genuine same-security links whose price moved for known reasons (FRL->FEL 0.18 across the 2016
         demerger, GET&D->GVT&D 4.63 over a 457-day panel hole during the 2024 rally, TRIL->TARIL 1.97), 12 of the
         21 bridged links fail it, and on the repaired panel most of these links are contiguous and accepted with no
         price test. The share-basis check reported next to it (pnl_share_ratio: median P&L-implied present-basis
         shares in the 400 days after the gap / the 400 days before; ATLANTA 1.00, JUBILANT 1.00, TRIL 1.13, LSIL
         1.16) is not a gate either, because real issuance moves it (MAGMA->POONAWALLA 2.83, the 2021 preferential
         allotment). Residual risk: a bonus in a bridged gap that the CA store does not hold under either symbol.
  NOT FIXED  (upstream, src/transform/corporate_actions.py) the panel adjuster matches CA events by symbol, so the
         orphan events above are also missing from the OLD symbols' adjusted prices: TRIL's close falls 306.95 ->
         31.80 on 2017-09-28 and MOTHERSUMI's 458.40 -> 305.45 on 2017-07-05 with price factor 1.0, i.e. fake
         -90%/-33% daily returns in every return-based backtest of those symbols. This builder corrects only the
         share basis it uses for mcap.
  NOT FIXED  P&L-implied counts lag real issuance by up to 4 quarters (basic EPS uses weighted-average shares):
         MAGMA 2021-06-07 is sized at Rs 4,370 cr on ~270M shares, weeks after an allotment that roughly tripled the
         count. Inherent to PAT/EPS; the pnl_share_ratio diagnostic exposes it at rename links only.
  NOT FIXED  a single bad filing can dominate the median while a chain has fewer than 3 filings (SPLPETRO
         2022-07-23: EPS 0.07 on PAT Rs 22.96 cr -> 3.28B shares, 35x; mcap ~18x too high until 2023-01-25). The
         manifest's drop-and-reversal counter flags such cases; no outlier filter was added in this round.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

OUT = ROOT / "data/derived/mcap_pit.parquet"
PNL = ROOT / "data/derived/pnl_quarterly.parquet"
SCR = ROOT / "data/derived/screener_fundamentals.parquet"
BF = ROOT / "data/derived/screener_mcap_backfill.parquet"
CA = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"
STALE_DAYS = 400
RENAME_GAP_DAYS = 10            # research_panel.stitch_renames contiguity rule
AFTER_HOURS_MIN = 15 * 60 + 30  # filings stamped at/after 15:30 IST are known from the next day
CHAINED = ("contiguous", "bridged_same_isin")
PANEL_COLS = ["close", "price_adjustment_factor_to_present", "share_adjustment_factor_to_present"]
EMPTY_SH = pd.DataFrame({"shares_present": pd.Series(dtype=float), "fetch_date": pd.Series(dtype="datetime64[ns]"),
                         "rec_symbol": pd.Series(dtype=object)})


# ---------------------------------------------------------------- helpers
def slug_of(url) -> str | None:
    """Upper-case screener company slug from a /company/<slug>/... URL (None if absent)."""
    if not isinstance(url, str):
        return None
    m = re.search(r"/company/([^/?#]+)", url)
    return unquote(m.group(1)).upper() if m else None


def rename_closure(s: str, rmap: dict[str, str]) -> list[str]:
    """s followed by its NSE successors: [s, rmap[s], rmap[rmap[s]], ...]."""
    out = [s]
    while out[-1] in rmap and rmap[out[-1]] not in out and len(out) < 50:
        out.append(rmap[out[-1]])
    return out


def rename_links(first: pd.Series, last: pd.Series, rmap: dict[str, str], isin: dict | None = None) -> pd.DataFrame:
    """One row per NSE rename old -> new with both symbols in the panel: gap_days = successor's first session - old
    symbol's last session, and verdict contiguous | bridged_same_isin (chained) or unbridged_isin_differs |
    unbridged_isin_unknown | unbridged_overlap | unbridged_gap_no_isin_check (isin not supplied). first/last:
    symbol -> first/last trade date; isin: symbol -> ISIN (security_master, latest seen)."""
    rows = []
    for o, n in rmap.items():
        if o == n or o not in last.index or n not in first.index or pd.isna(last[o]) or pd.isna(first[n]):
            continue
        gap = int((pd.Timestamp(first[n]) - pd.Timestamp(last[o])).days)
        if 0 < gap <= RENAME_GAP_DAYS:
            v = "contiguous"
        elif gap <= 0:
            v = "unbridged_overlap"
        elif isin is None:
            v = "unbridged_gap_no_isin_check"
        else:
            io, i_n = isin.get(o), isin.get(n)
            if not (isinstance(io, str) and io.strip() and isinstance(i_n, str) and i_n.strip()):
                v = "unbridged_isin_unknown"
            elif io.strip() != i_n.strip():
                v = "unbridged_isin_differs"
            else:
                v = "bridged_same_isin"
        rows.append((o, n, gap, v))
    return pd.DataFrame(rows, columns=["old", "new", "gap_days", "verdict"])


def rename_successors(first: pd.Series, last: pd.Series, rmap: dict[str, str], isin: dict | None = None) -> dict[str, str]:
    """old -> new for CHAINED renames (see rename_links)."""
    lk = rename_links(first, last, rmap, isin)
    ok = lk[lk["verdict"].isin(CHAINED)]
    return dict(zip(ok["old"], ok["new"]))


def chain_roots(succ: dict[str, str]) -> dict[str, str]:
    """symbol -> final successor (chain root). Dates strictly increase along a chain, so there are no cycles;
    the depth guard is only a safety net."""
    root = {}
    for s in succ:
        r, k = s, 0
        while r in succ and k < 50:
            r, k = succ[r], k + 1
        root[s] = r
    return root


def era_of(d: pd.Series) -> np.ndarray:
    y = d.dt.year
    return np.where(y < 2016, "pre2016", np.where(y <= 2022, "disc", "conf"))


def share_events(ca: pd.DataFrame | None) -> pd.DataFrame:
    """Split/bonus/consolidation events as the panel adjuster reads them: one row per (symbol, ex_date), factors
    multiplied, factor 1 dropped."""
    cols = ["symbol", "ex_date", "adjustment_factor"]
    if ca is None or ca.empty:
        return pd.DataFrame({"symbol": pd.Series(dtype=object), "ex_date": pd.Series(dtype="datetime64[ns]"),
                             "adjustment_factor": pd.Series(dtype=float)})
    a = ca[cols].copy()
    a["symbol"] = a["symbol"].astype(str).str.strip().str.upper()
    a["ex_date"] = pd.to_datetime(a["ex_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    a["adjustment_factor"] = pd.to_numeric(a["adjustment_factor"], errors="coerce")
    a = a[a["ex_date"].notna() & a["adjustment_factor"].gt(0)]
    a = a.groupby(["symbol", "ex_date"], as_index=False)["adjustment_factor"].prod()
    return a[a["adjustment_factor"] != 1.0].sort_values(["symbol", "ex_date"]).reset_index(drop=True)


def orphan_events(members: dict[str, list[str]], rmap: dict[str, str], first: pd.Series, ev: pd.DataFrame) -> dict:
    """Per chain root: CA share events filed under a symbol of the root's rename family (chain members and their NSE
    successors) with ex_date on or before that symbol's first panel session, so no panel factor applies them to the
    earlier symbols' rows. An ex_date that some family symbol files AFTER its own first session is already in a panel
    factor and is skipped. Returns root -> (ex_dates sorted, suffix products, [(ex_date, factor, filed_under)])."""
    by = {s: (g["ex_date"].to_numpy(dtype="datetime64[ns]"), g["adjustment_factor"].to_numpy(dtype=float))
          for s, g in ev.groupby("symbol")}
    out: dict = {}
    for r, mem in members.items():
        fam = list(dict.fromkeys(x for m in mem for x in rename_closure(m, rmap)))
        fam = [s for s in fam if s in by and s in first.index]
        if not fam:
            continue
        handled, cand = set(), {}
        for s in fam:
            ex, fx = by[s]
            f0 = np.datetime64(pd.Timestamp(first[s]), "ns")
            for e, x in zip(ex, fx):
                if e > f0:
                    handled.add(e)
                else:
                    cand.setdefault(e, (float(x), s))
        orph = sorted((e, x, s) for e, (x, s) in cand.items() if e not in handled)
        if orph:
            ex = np.array([o[0] for o in orph], dtype="datetime64[ns]")
            fx = np.array([o[1] for o in orph], dtype=float)
            out[r] = (ex, np.cumprod(fx[::-1])[::-1], [(str(pd.Timestamp(e).date()), x, s) for e, x, s in orph])
    return out


def _orphan_mult(roots: pd.Series, dates: pd.Series, orph: dict) -> np.ndarray:
    """Product of the root's orphan events with ex_date strictly after each date (1 when none)."""
    out = np.ones(len(roots))
    if not orph:
        return out
    rv = pd.Series(np.asarray(roots, dtype=object))
    dv = pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy(dtype="datetime64[ns]")
    sub = rv[rv.isin(list(orph))]
    for key, pos in sub.groupby(sub).groups.items():
        pos = np.asarray(pos)
        ex, suf, _ = orph[key]
        i = np.searchsorted(ex, dv[pos], side="right")
        has = i < len(ex)
        m = np.ones(len(pos))
        m[has] = suf[i[has]]
        out[pos] = m
    return out


def _asof_factor(keys: pd.DataFrame, date_col: str, chain_rows: pd.DataFrame, orph: dict) -> pd.Series:
    """Chain factor at keys[date_col] for keys['root']: the panel part from the last chain session on or before the
    date (a date before the chain's first session takes the first session's), times the orphan events after the
    date. Returned in keys' index order."""
    k = keys[["root", date_col]].copy()
    k["_i"] = np.arange(len(k))
    k[date_col] = pd.to_datetime(k[date_col]).dt.normalize().astype("datetime64[ns]")
    m = pd.merge_asof(k.sort_values(date_col), chain_rows.rename(columns={"trade_date": date_col}),
                      on=date_col, by="root", direction="backward")
    first_f = chain_rows.groupby("root")["f_panel"].first()
    m["f_panel"] = m["f_panel"].fillna(m["root"].map(first_f))
    m = m.sort_values("_i")
    return pd.Series(m["f_panel"].to_numpy() * _orphan_mult(m["root"], m[date_col], orph), index=keys.index)


def screener_present_shares(rec: pd.DataFrame, root: dict[str, str], chain_rows: pd.DataFrame, orph: dict) -> pd.DataFrame:
    """Per chain root: screener then-shares at fetch_date (market_cap / price) rebased to the chain's present basis
    with the chain factor at fetch_date. Latest fetch wins; the root's own page wins a tie."""
    rec = rec[(rec["current_price"] > 0) & (rec["market_cap_cr"] > 0)].copy()
    rec["root"] = rec["symbol"].map(root).fillna(rec["symbol"])
    rec = rec[rec["root"].isin(set(chain_rows["root"]))]
    if rec.empty:
        return EMPTY_SH.copy()
    rec["fetch_date"] = pd.to_datetime(rec["fetch_date"])
    rec["shares_present"] = rec["market_cap_cr"] * 1e7 / rec["current_price"] * _asof_factor(rec, "fetch_date", chain_rows, orph)
    rec["own"] = rec["symbol"] == rec["root"]
    rec = rec.sort_values(["fetch_date", "own"]).groupby("root").tail(1).set_index("root")
    return rec[["shares_present", "fetch_date"]].assign(rec_symbol=rec["symbol"])


def pnl_share_events(q: pd.DataFrame, root: dict[str, str], chain_rows: pd.DataFrame, orph: dict,
                     pool: pd.DataFrame | None = None, guard: dict | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """PIT share-count events: (root, primary, trade_date = first day the filing is known, shares_present), one median
    series per root and basis; primary=True for the root's chosen basis, False for the other basis (a fallback the
    builder uses only where the primary series has no filing within 400 days, so the bases never mix in a median).
    pool: extra (symbol, root, max_fdate) assignments: filings stored under `symbol` dated up to max_fdate also size
    `root` (unbridged NSE renames). guard: root -> first date; filings dated before it are not used for that root.
    Also returns the primary-basis filings used (root, fdate, sh_present, pooled)."""
    q = q.dropna(subset=["filing_dt", "pat", "eps_basic"]).copy()
    to_rs = np.where(q["source"] == "xbrl", 1.0, 1e5)                         # manifest: detail_api = Rs lakh
    ok = (q["eps_basic"].abs() >= 0.05) & (q["pat"] != 0)
    q["sh"] = np.where(ok, q["pat"] * to_rs / q["eps_basic"], np.nan)
    q = q[q["sh"] > 0].copy()
    fdt = pd.to_datetime(q["filing_dt"])
    q["fdate"] = fdt.dt.normalize()
    mins = fdt.dt.hour * 60 + fdt.dt.minute
    q["late"] = (mins >= AFTER_HOURS_MIN) | (fdt == q["fdate"])               # date-only stamps: after hours
    q["known"] = q["fdate"] + pd.to_timedelta(q["late"].astype(int), unit="D")
    base = q.assign(root=q["symbol"].map(root).fillna(q["symbol"]), pooled=False)
    parts = [base]
    if pool is not None and len(pool):
        ext = q.merge(pool, on="symbol")
        parts.append(ext[ext["fdate"] <= ext["max_fdate"]].drop(columns="max_fdate").assign(pooled=True))
    q = pd.concat(parts, ignore_index=True)
    q = q[q["root"].isin(set(chain_rows["root"]))]
    n_guard = 0
    if guard:
        g = q["root"].map(guard)
        drop = g.notna() & (q["fdate"] < g)
        n_guard = int(drop.sum())
        q = q[~drop]
    cnt = q.groupby(["root", "basis"]).size().unstack(fill_value=0)
    sa = cnt["sa"] if "sa" in cnt else pd.Series(0, index=cnt.index)
    con = cnt["con"] if "con" in cnt else pd.Series(0, index=cnt.index)
    basis = pd.Series(np.where(sa >= 0.5 * con, "sa", "con"), index=cnt.index)
    q = q.copy()
    q["primary"] = q["basis"] == q["root"].map(basis)
    q["sh_present"] = q["sh"] * _asof_factor(q, "fdate", chain_rows, orph)
    q = q.dropna(subset=["sh_present"]).sort_values(["root", "filing_dt", "quarter_end"], kind="mergesort")
    recs = []
    for (r, prim), g in q.groupby(["root", "primary"], sort=False):
        cur: dict = {}
        for qe, fd, kn, shp in zip(g["quarter_end"], g["filing_dt"], g["known"], g["sh_present"]):
            cur[qe] = (fd, shp)                                                # a re-filing replaces its quarter
            last4 = sorted(cur.values(), key=lambda v: v[0])[-4:]
            recs.append((r, prim, kn, float(np.median([v[1] for v in last4]))))
    ev = pd.DataFrame({"root": pd.Series([x[0] for x in recs], dtype=object),
                       "primary": pd.Series([x[1] for x in recs], dtype=bool),
                       "trade_date": pd.Series([x[2] for x in recs], dtype="datetime64[ns]"),
                       "shares_present": pd.Series([x[3] for x in recs], dtype=float)})
    ev = ev.drop_duplicates(["root", "primary", "trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
    qp = q[q["primary"]]
    info = dict(roots_with_usable_filings=int(q["root"].nunique()), basis_choice=basis.value_counts().to_dict(),
                usable_filings_primary_basis=int(len(qp)), usable_filings_fallback_basis=int(len(q) - len(qp)),
                after_hours_or_dateonly_share=round(float(q["late"].mean()), 3) if len(q) else None,
                pooled_filings_used=int(q["pooled"].sum()), pooled_roots=int(q.loc[q["pooled"], "root"].nunique()),
                guard_filings_dropped=n_guard)
    return ev, info, qp[["root", "fdate", "sh_present", "pooled"]]


# ---------------------------------------------------------------- build
def build(px: pd.DataFrame, q: pd.DataFrame, scr: pd.DataFrame, bf: pd.DataFrame | None,
          rmap: dict[str, str], isin: dict | None = None, ca: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """px: panel rows (symbol, trade_date + PANEL_COLS); isin: symbol -> ISIN (security_master); ca: CA store rows
    (symbol, ex_date, adjustment_factor). Returns (rows, build_info). Pure: reads/writes no files."""
    w = px[["symbol", "trade_date"] + PANEL_COLS].copy()
    w["trade_date"] = pd.to_datetime(w["trade_date"]).astype("datetime64[ns]")
    w = w.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    w["raw_px"] = rp.raw_price(w, "close")
    fs = pd.to_numeric(w["share_adjustment_factor_to_present"], errors="coerce").replace(0, np.nan).fillna(1.0)
    first = w.groupby("symbol")["trade_date"].min()
    last = w.groupby("symbol")["trade_date"].max()
    links = rename_links(first, last, rmap, isin)
    ch = links[links["verdict"].isin(CHAINED)]
    succ = dict(zip(ch["old"], ch["new"]))
    root = chain_roots(succ)
    f_first = fs.groupby(w["symbol"]).first()
    k_mult: dict[str, float] = {}

    def k_of(s: str) -> float:                     # product of successors' factors at their first sessions
        if s not in succ:
            return 1.0
        if s not in k_mult:
            n = succ[s]
            k_mult[s] = float(f_first[n]) * k_of(n)
        return k_mult[s]

    w["root"] = w["symbol"].map(root).fillna(w["symbol"])
    w["f_panel"] = fs * w["symbol"].map({s: k_of(s) for s in succ}).fillna(1.0)
    chain_rows = w[["root", "trade_date", "f_panel"]].sort_values("trade_date", kind="mergesort")
    members = w.drop_duplicates("symbol").groupby("root")["symbol"].apply(list).to_dict()
    orph = orphan_events(members, rmap, first, share_events(ca))
    w["f_chain"] = w["f_panel"] * _orphan_mult(w["root"], w["trade_date"], orph)

    # P&L pooling through the NSE rename family for old symbols whose rename is not chained, and the successor guard
    chain_first = chain_rows.groupby("root")["trade_date"].min()
    chain_last = chain_rows.groupby("root")["trade_date"].max()
    unb = links[~links["verdict"].isin(CHAINED)]
    pool = pd.DataFrame([(s, o, chain_last[o]) for o in sorted(set(rmap) & set(first.index) - set(succ))
                         for s in rename_closure(o, rmap)[1:]], columns=["symbol", "root", "max_fdate"])
    pool = pool.drop_duplicates(["symbol", "root"])
    guard = {root.get(n, n): chain_first[root.get(n, n)] for n in unb["new"] if root.get(n, n) in chain_first.index}

    # 1 pnl_implied: primary-basis median; the other basis's own median only where the primary has none within 400 days
    ev, pnl_info, filings = pnl_share_events(q, root, chain_rows, orph, pool, guard)
    w = w.sort_values("trade_date", kind="mergesort")
    for prim, col in ((True, "shares_present"), (False, "_sh_fallback")):
        e = ev.loc[ev["primary"] == prim, ["root", "trade_date", "shares_present"]].rename(columns={"shares_present": col})
        w = pd.merge_asof(w, e, on="trade_date", by="root", direction="backward", tolerance=pd.Timedelta(days=STALE_DAYS))
    fb = w["shares_present"].isna() & w["_sh_fallback"].notna()
    pnl_info["rows_sized_by_fallback_basis"] = int(fb.sum())
    w["shares_present"] = w["shares_present"].fillna(w["_sh_fallback"])
    w = w.drop(columns="_sh_fallback").sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    w["mcap_cr"] = w["raw_px"] * w["shares_present"] / w["f_chain"] / 1e7
    w["mcap_source"] = pd.Series(np.where(w["mcap_cr"].notna(), "pnl_implied", None), index=w.index, dtype=object)

    # 2 screener_backcast (screener_fundamentals latest fetch per symbol); an unbridged old symbol's page would be
    #   its successor's (screener redirects old slugs), so its records are dropped
    unb_old = set(unb["old"])
    scr = scr[~scr["symbol"].isin(unb_old)]
    scr = scr.dropna(subset=["fetch_date", "market_cap_cr", "current_price"]).sort_values("fetch_date").groupby("symbol").tail(1)
    scr_sh = screener_present_shares(scr, root, chain_rows, orph)
    # 3 screener_backfill: page slug must be the symbol or a member of its CHAINED rename chain
    bf_info: dict = {}
    bf_sh = EMPTY_SH.copy()
    bf_status: dict[str, str] = {}
    if bf is not None and len(bf):
        b = bf.copy()
        page = b["final_url"].fillna(b["url"]) if "final_url" in b else b["url"]
        b["slug"] = page.map(slug_of)
        b["root"] = b["symbol"].map(root).fillna(b["symbol"])
        slug_root = b["slug"].map(lambda s: root.get(s, s) if s is not None else None)
        own = b["slug"] == b["symbol"].str.upper()
        via_chain = ~own & b["slug"].isin(set(w["symbol"])) & (slug_root == b["root"])
        has_data = (b["status"].isin(["OK", "SLUG_MISMATCH"]) & (pd.to_numeric(b["current_price"], errors="coerce") > 0)
                    & (pd.to_numeric(b["market_cap_cr"], errors="coerce") > 0))
        acc = has_data & (own | via_chain)
        is_succ_page = pd.Series([sl in rename_closure(sy, rmap)[1:] for sy, sl in zip(b["symbol"], b["slug"])], index=b.index)
        rej = (has_data | (b["status"] == "SLUG_MISMATCH")) & ~acc
        b["verdict"] = np.where(acc & own, "accepted_own_slug", np.where(acc, "accepted_via_rename_chain",
                                np.where(rej & is_succ_page, "rejected_rename_unbridged",
                                np.where(rej, "rejected_slug_mismatch", "status_" + b["status"].astype(str)))))
        bf_status = b.set_index("symbol")["verdict"].to_dict()
        bf_info = dict(verdicts=b["verdict"].value_counts().to_dict(),
                       rejected_examples=b.loc[b["verdict"] == "rejected_slug_mismatch", ["symbol", "slug"]].head(40).values.tolist(),
                       rejected_rename_unbridged=b.loc[b["verdict"] == "rejected_rename_unbridged", ["symbol", "slug"]].values.tolist(),
                       via_chain_examples=b.loc[b["verdict"] == "accepted_via_rename_chain", ["symbol", "slug"]].head(25).values.tolist())
        bf_sh = screener_present_shares(b[acc][["symbol", "fetch_date", "market_cap_cr", "current_price"]], root, chain_rows, orph)

    for label, sh in (("screener_backcast", scr_sh), ("screener_backfill", bf_sh)):
        miss = w["mcap_cr"].isna() & w["root"].isin(sh.index) & w["raw_px"].notna()
        w.loc[miss, "mcap_cr"] = w.loc[miss, "raw_px"] * w.loc[miss, "root"].map(sh["shares_present"]) / w.loc[miss, "f_chain"] / 1e7
        w.loc[miss, "mcap_source"] = label
    w["pit_ok"] = w["mcap_source"].eq("pnl_implied")

    # why a row is NULL: '<share-count side>|<screener side>'
    null = w["mcap_cr"].isna()
    first_ev = ev.groupby("root")["trade_date"].min()
    fe = w.loc[null, "root"].map(first_ev)
    pnl_side = np.where(w.loc[null, "raw_px"].isna(), "no_price",
               np.where(fe.isna(), "no_usable_pnl_filings",
               np.where(w.loc[null, "trade_date"] < fe, "before_first_filing", "pnl_stale_gt400d")))
    roots_status: dict[str, str] = {}
    for s, v in bf_status.items():
        roots_status.setdefault(root.get(s, s), v)
    for o, v in zip(unb["old"], unb["verdict"]):                   # the only page an old symbol has is its successor's
        roots_status[o] = "rename_" + v
    for o in set(rmap) & set(first.index) - set(succ) - unb_old:
        roots_status.setdefault(o, "rename_successor_not_in_panel")
    scr_side = w.loc[null, "root"].map(lambda r: "screener_" + roots_status.get(r, "not_fetched"))
    w["null_reason"] = pd.Series(None, index=w.index, dtype=object)
    w.loc[null, "null_reason"] = pd.Series(pnl_side, index=scr_side.index) + "|" + scr_side
    w["sizing_symbol"] = w["root"]

    # rename-link diagnostics: price and P&L share-count continuity across each link (reported, not gates)
    fr = w.groupby("symbol").head(1).set_index("symbol")
    lr = w.groupby("symbol").tail(1).set_index("symbol")
    lk = links.copy()
    lk["raw_price_ratio"] = (lk["new"].map(fr["raw_px"]) / lk["old"].map(lr["raw_px"])).round(3)
    pp_first = fr["raw_px"] / fr["f_chain"]
    pp_last = lr["raw_px"] / lr["f_chain"]
    chained = lk["verdict"].isin(CHAINED)
    lk["present_basis_price_ratio"] = np.where(chained, lk["new"].map(pp_first) / lk["old"].map(pp_last), np.nan).round(3)
    fil = {r: g for r, g in filings.groupby("root")}
    win = pd.Timedelta(days=STALE_DAYS)

    def share_ratio(o: str, n: str) -> float:
        g = fil.get(root.get(o, o))
        if g is None:
            return np.nan
        a = g.loc[(g["fdate"] <= last[o]) & (g["fdate"] > last[o] - win), "sh_present"].median()
        z = g.loc[(g["fdate"] >= first[n]) & (g["fdate"] < first[n] + win), "sh_present"].median()
        return round(float(z / a), 3) if a > 0 and z > 0 else np.nan

    lk["pnl_share_ratio"] = [share_ratio(o, n) if c else np.nan for o, n, c in zip(lk["old"], lk["new"], chained)]
    lk_out = lk.astype(object).where(lk.notna(), None)                  # strict JSON: NaN -> null
    orph_list = sorted((r, e) for r, v in orph.items() for e in v[2])
    n_orph_rows = int((w["f_chain"] != w["f_panel"]).sum())
    link_info = dict(
        verdicts=lk["verdict"].value_counts().to_dict(),
        bridged=lk_out.loc[lk["verdict"] == "bridged_same_isin", ["old", "new", "gap_days", "raw_price_ratio",
                                                               "present_basis_price_ratio", "pnl_share_ratio"]].values.tolist(),
        unbridged=lk_out.loc[~chained, ["old", "new", "gap_days", "verdict"]].values.tolist(),
        contiguous_price_ratio_outside_0p67_1p5=lk_out.loc[(lk["verdict"] == "contiguous") & ~lk["present_basis_price_ratio"].between(0.67, 1.5),
                                                       ["old", "new", "present_basis_price_ratio"]].values.tolist()[:40],
        orphan_events=dict(roots=len(orph), events=len(orph_list), rows_rebased=n_orph_rows, examples=orph_list[:40]))
    info = dict(rename_chains=len(succ), rename_links=link_info, pnl=pnl_info,
                screener_backcast_roots=int(len(scr_sh)), screener_backfill=bf_info, _scr_sh=scr_sh, _bf_sh=bf_sh, _links=lk)
    return w, info


# ---------------------------------------------------------------- diagnostics (manifest)
def diagnostics(w: pd.DataFrame, info: dict, scr: pd.DataFrame) -> dict:
    d: dict = {}
    sized = w[w["mcap_cr"].notna()]
    d["rows"] = int(len(w)); d["rows_with_mcap"] = int(len(sized))
    d["symbols"] = int(w["symbol"].nunique()); d["symbols_with_mcap"] = int(sized["symbol"].nunique())
    d["by_source"] = sized["mcap_source"].value_counts().to_dict()
    era = pd.Series(era_of(w["trade_date"]), index=w.index)
    d["source_share_by_era_all_sized"] = (sized.groupby(era[sized.index])["mcap_source"].value_counts(normalize=True)
                                          .round(3).unstack(fill_value=0).to_dict(orient="index"))
    u = sized[sized["mcap_cr"] >= 50]
    d["source_share_by_era_mcap50"] = (u.groupby(era[u.index])["mcap_source"].value_counts(normalize=True)
                                       .round(3).unstack(fill_value=0).to_dict(orient="index"))
    d["pit_ok_share_of_sized_by_year"] = {int(k): round(float(v), 3) for k, v in
                                          sized.groupby(sized["trade_date"].dt.year)["pit_ok"].mean().items()}
    d["null_reason_rows"] = w["null_reason"].value_counts().to_dict()
    d["null_reason_symbols"] = w.dropna(subset=["null_reason"]).groupby("null_reason")["symbol"].nunique().to_dict()
    d["date_range"] = [str(w["trade_date"].min().date()), str(w["trade_date"].max().date())]
    nr = w.dropna(subset=["null_reason"])
    d["null_rows_by_era"] = nr.groupby(era[nr.index])["null_reason"].value_counts().unstack(fill_value=0).to_dict(orient="index")
    # size of the Rs 50 cr universe (names per session): the number a NULL hole shrinks
    n50 = u.groupby("trade_date")["symbol"].nunique()
    d["mcap50_names_per_session_median"] = {f"{y}Q{qq}": int(v) for (y, qq), v in
                                            n50.groupby([n50.index.year, n50.index.quarter]).median().items()}

    # latest pnl_implied row vs screener's own current market cap (within 45 days of the fetch)
    lastrow = w.sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")
    s1 = scr.dropna(subset=["market_cap_cr"]).sort_values("fetch_date").groupby("symbol").tail(1).set_index("symbol")
    v = lastrow[lastrow["mcap_source"] == "pnl_implied"].join(s1[["market_cap_cr", "fetch_date"]], how="inner")
    v = v[(pd.to_datetime(v["fetch_date"]) - v["trade_date"]).abs() < pd.Timedelta(days=45)]
    r = v["mcap_cr"] / v["market_cap_cr"]
    d["validation_pnl_latest_vs_screener"] = dict(n=int(len(r)), median_ratio=round(float(r.median()), 3) if len(r) else None,
                                                  within_20pct=round(float(((r > .8) & (r < 1.2)).mean()), 3) if len(r) else None,
                                                  within_50pct=round(float(((r > .5) & (r < 1.5)).mean()), 3) if len(r) else None)

    # history: present-share screener count vs pnl-implied count on the same rows (the error a backcast carries)
    both = w[w["mcap_source"].eq("pnl_implied")].copy()
    both["scr_present"] = both["root"].map(info["_scr_sh"]["shares_present"]).fillna(
        both["root"].map(info["_bf_sh"]["shares_present"]))
    both = both.dropna(subset=["scr_present"])
    if len(both):
        both["ratio"] = both["scr_present"] / both["shares_present"]
        per = both.groupby([both["trade_date"].dt.year, "symbol"])["ratio"].median()
        d["validation_backcast_vs_pnl_by_year"] = {
            int(y): dict(symbols=int(len(s)), p10=round(float(s.quantile(.1)), 3), p50=round(float(s.median()), 3),
                         p90=round(float(s.quantile(.9)), 3), over_1p2=round(float((s > 1.2).mean()), 3),
                         under_0p8=round(float((s < 0.8).mean()), 3))
            for y, s in per.groupby(level=0)}

    # split-lag signature: pnl_implied day-over-day mcap < 0.4x that reverses > 2.5x within 400 days
    p = w[w["mcap_source"].eq("pnl_implied")][["symbol", "trade_date", "mcap_cr"]]
    ratio = p["mcap_cr"] / p.groupby("symbol")["mcap_cr"].shift()
    drops = p[ratio < 0.4]
    rev = 0
    if len(drops):
        by = {s: g.set_index("trade_date")["mcap_cr"] for s, g in p[p["symbol"].isin(set(drops["symbol"]))].groupby("symbol")}
        for s, t, m in drops[["symbol", "trade_date", "mcap_cr"]].itertuples(index=False):
            fut = by[s].loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=STALE_DAYS)]
            rev += int(len(fut) > 0 and fut.max() / m > 2.5)
    d["pnl_dod_drops_below_0p4x"] = int(len(drops)); d["pnl_drops_reversed_gt_2p5x_within_400d"] = rev
    d["build"] = {k: v for k, v in info.items() if not k.startswith("_")}
    return d


def main() -> None:
    px = rp.load_panel(PANEL_COLS)
    q = pd.read_parquet(PNL)
    scr = pd.read_parquet(SCR, columns=["symbol", "fetch_date", "market_cap_cr", "current_price"])
    bf = pd.read_parquet(BF) if BF.exists() else None
    sm = pd.read_parquet(rp.MASTER, columns=["symbol", "isin"])
    isin = dict(zip(sm["symbol"], sm["isin"]))
    ca = pd.read_parquet(CA, columns=["symbol", "ex_date", "adjustment_factor"]) if CA.exists() else None
    w, info = build(px, q, scr, bf, rp.rename_map(), isin=isin, ca=ca)
    out = w[["symbol", "trade_date", "mcap_cr", "mcap_source", "pit_ok", "sizing_symbol", "null_reason"]]
    out.to_parquet(OUT, index=False)
    stats = diagnostics(w, info, scr)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="mcap_pit", path=str(OUT.relative_to(ROOT)), key=["symbol", "trade_date"], producer="src/agentic/build_mcap_pit.py",
        point_in_time=("NOT point-in-time as a whole. Only pit_ok=True rows (mcap_source=='pnl_implied') use share counts "
                       "known at t. Rows before 2018-04 are NEVER PIT (pnl_quarterly filing_dt starts 2018-04-02); "
                       "screener rows use present-day or last-trade share counts and ignore later issuance/buybacks, "
                       "biasing size upward for names that later raised equity. Era comparisons must report the pit_ok mix."),
        columns=dict(
            mcap_cr="market capitalisation, Rs CRORE = then-traded price x then-outstanding shares / 1e7 (NULL = unknown)",
            mcap_source="pnl_implied (PAT/basic EPS shares, median of latest 4 filings known at t, one basis per chain; "
                        "filings pooled through the NSE rename family) | screener_backcast (screener_fundamentals present "
                        "shares) | screener_backfill (screener_mcap_backfill present/last-trade shares; slug == symbol or "
                        "CHAINED rename-chain member)",
            pit_ok="True only for pnl_implied rows",
            sizing_symbol="rename-chain root whose filings/screener page size the row (== symbol unless the row is a "
                          "pre-rename symbol whose NSE rename is chained: contiguous within 10 days, or bridged with the "
                          "same ISIN)",
            null_reason="for NULL rows: '<share-count side>|<screener side>' (no_price, no_usable_pnl_filings, "
                        "before_first_filing, pnl_stale_gt400d | screener_not_fetched, screener_rejected_slug_mismatch, "
                        "screener_rename_unbridged_<isin_differs|isin_unknown|overlap>, "
                        "screener_rename_successor_not_in_panel, screener_status_<HTTP_404|NO_MCAP|...>)"),
        conventions=dict(
            price="research_panel.raw_price(close) = adjusted close / price_adjustment_factor_to_present",
            share_basis="present basis = then-shares x chain factor at the filing/fetch date; back to then-shares with "
                        "the chain factor at t. Chain factor = own share_adjustment_factor_to_present x chained successors' "
                        "factors at their first sessions x CA events filed under a later family symbol on/before its "
                        "first session and after t (orphan events)",
            rename_links="NSE symbolchange (research_panel.rename_map): contiguous (1-10 day gap) or bridged_same_isin "
                         "(longer gap, same security_master ISIN) are chained; unbridged links share P&L filings only",
            ca_store="data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet (orphan events)",
            filing_known="filing date if stamped before 15:30 IST, else next day; stale after 400 days",
            pnl_units="per pnl_quarterly manifest: detail_api Rs lakh, xbrl Rs",
            eras="pre2016 = 2015, disc = 2016-2022, conf = 2023+"),
        audit="logs/audits/audit_20260927_research_code.json (2026-09-27 fixes listed in the producer docstring)",
        stats=stats, updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(json.dumps({k: stats[k] for k in ("rows", "rows_with_mcap", "by_source", "source_share_by_era_mcap50")}, indent=1, default=str))


if __name__ == "__main__":
    main()
