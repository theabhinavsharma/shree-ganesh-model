"""HOT INDUSTRY x MATERIAL ORDER x ABOVE 200DMA (EXP-2026-09-27-hot-x-material-order).

Part A (registered): condition C = industry broad-heat pct >= 0.90 AND an order/L1 filing in the prior 60 days
with full-text amount >= 15% of PIT TTM revenue AND close > SMA200; control = hot & > SMA200 & no such order.
Outcomes on the anatomy stock-weeks (mcap >= 50cr): P(touch +50% within 95), P(sustained), mean 95-session
return, by era. Pass: C >= 1.5x base AND >= 1.2x control in BOTH eras, n >= 50 per era.
Part B (drawdown question): B+H rule (broad heat, top-10, EXTENDED), all-in, 90-session hold, 18 phases:
(1) as is, (2) HARD filter = only material-order names, (3) TIE-BREAK = material-order names first within each
hot industry, then own ret60. Windows 2016+ and 2019+.
Inputs: data/derived/order_fulltext.parquet (+ order_amounts.parquet), announcements_historical (filing time),
logs/leader_sleeve/anatomy_1p5x/rows.parquet, pnl_quarterly (units per manifest), the price panel (research_panel).

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json: the 17 confirmed findings that name this
file, listed by the file + lines the audit gives; the test had not been run before them)
FIXED
  * test_hot_order_combo.py 25-31, 35, 45, 49-65, 112-117 (high; PIT coverage / era split): pnl_quarterly filings
    start 2018-04, so the old disc C cell could only draw 2018-11+ rows while base and control spanned 2016-22.
    The disc era of every materiality comparison now starts at COV_START = the first week on which >= COV_MIN
    (50%) of the anatomy rows have a PIT TTM revenue. Base, C and both controls use that SAME window. Part B adds
    a COV_START window; the registered 2016+ window is kept but its HARD/TIE rows are labelled pre-coverage.
  * 79, 84-89 (medium; logic): AS-IS reproduces sim_allin_matrix.rule(broad=True, top=10, gate=False) step for
    step (rank by ret60 within the WHOLE hot industry, then core & ret252 > 0.5). HARD = those AS-IS picks that
    carry a material order (top-10 cap kept). TIE = the same pool ranked material-first, then ret60, top 10, same
    filters. `core` is the fixed sim_allin_matrix one (its lines 133, 142-143): panel 20d ADV >= 5cr AND
    research_panel.raw_price(close) > 50, not rows.parquet `core` (back-adjusted close). ADV for open-entry
    costs comes from the same panel column.
  * 41-46, 57-65 (medium; statistics): outcomes are also clustered by order episode (one row per material order:
    its first week inside the cell), with a symbol-cluster bootstrap CI for P(+50%), lift vs base and ratio vs
    both controls, and a clustered verdict next to the registered one.
  * 25-29 (low; TTM denominator) and event_materiality_study.py:127 as replicated here (low): one basis per symbol
    (more filed quarters, tie -> consolidated), 4 CONSECUTIVE calendar quarters, as-filed (first) values,
    available once the LAST of the four was filed; a late filing of an older quarter never replaces a newer TTM.
  * 24, 41, 45 and fetch_order_fulltext.py 64-66 -> ~39-43 (low; same-day lookahead, two findings): the filing
    time comes from announcements_historical.sort_date (IST). Filings at or after 15:30, or on a non-session
    day, count from the next session; filings with no timestamp are treated as after the close.
  * 71, 103 (low; returns across gaps zeroed): research_panel.gap_aware_returns + stitch_renames inside a hold,
    AND at every exit: a position whose name has no print at the exit session (close mode: the next signal
    close; open mode: the next open) is sold at the first session it prints again (its close / its open,
    within CATCHUP_MAX sessions, value booked at the rotation), as sim_allin_matrix._catchup does; no print
    within CATCHUP_MAX = frozen at the last close. Late exits are counted per arm and the largest are printed.
    The last rotation's exit (late sale, open leg) is booked on the final NAV mark.
  * event_materiality_study.py:33 (medium; bare 'awarded' pulls in honours / arbitration / ratings): every
    crawled filing is re-classified here with the CURRENT CATS (parse_order_amounts.classify on
    announcements_historical 'desc || attchmntText', the function order_amounts.cat_current comes from), in
    every --amounts mode. Filings no longer order / tender_L1 are dropped from both the material set and the
    any-order set. In revised mode a disagreement with order_amounts.cat_current makes the run PROVISIONAL.
  * Execution realism (audit critic #4): --entry open buys at the next session's open, skips entries that are
    upper-circuit locked or not trading at that open (weight spread over the rest; all skipped = cash), exits
    at the open after the next signal date and charges research_panel.cost_rt by 20d ADV. The registered close
    entry with a flat 0.5% round trip is kept (--entry close|open|both, default both).
  * Outcome labels: the anatomy y95/s95/fc95 count 95 ROWS per symbol (anatomy_1p5x audit, high). --labels panel
    (default) recomputes them on the session calendar with research_panel.forward_window; --labels anatomy keeps
    the file's labels. A row is labelled only when its whole 95-session window lies inside the panel.
PARTIAL
  * 23-24, 32-35 (high) and event_materiality_study.py 47-59 / fetch_order_fulltext.py 76, 91 (high; the
    largest-amount parser picks revenue / order-book boilerplate; two findings): order_fulltext.parquet is the
    filing population; amounts come from order_amounts.parquet:order_amount_cr (the revised order-attached
    parser, parse_order_amounts.py), LEFT-joined on (symbol, seq_id), only when that file covers EVERY crawled
    filing (--amounts auto). Otherwise the audited order_fulltext.amount_cr is used for every filing (one parser
    for both eras) and every verdict is PROVISIONAL. The parser fix itself is upstream; its output is still
    being written (2026-09-27 17:40: 1,499 of 6,340 filings, 2016-01..2021-04), so the default run is PROVISIONAL.
  * fetch_order_fulltext.py 52-53, 65-67, 76 (medium) and fetch 53/65/67/76 + ocr_order_filings.py 57/62/70 +
    event_materiality_study.py 110-113 (medium; USD at a hard-coded 83.0 before 2024-02-19; two findings):
    fixed upstream in the revised amounts (historical FRED DEXINUS rate, NaN when none). Here a revised file
    with a USD amount and no rate is refused. The legacy fallback still carries the 83.0 conversion; the
    PROVISIONAL reason says so. Legacy rows carry no currency, so they cannot be re-converted here.
  * event_materiality_study.py:33 (medium; order regex recall ~68% disc vs ~87% conf): the CATS fix is
    upstream, but order_fulltext.parquet is still the pre-fix crawl population. Every order / tender_L1 filing
    under the CURRENT CATS that is not in the crawl is counted by era, joins the any-order set (control 2 =
    no order filing under the current CATS) and the "unsized order" diagnostic of control 1, and makes the run
    PROVISIONAL. With no amount they can never be material until the v2 crawl adds them.
  * 30-35, 46, 52-53 (medium; survivorship / control population): C, base and both controls are the same
    population (rows whose symbol has a PIT TTM revenue at that date); control 2 drops rows with ANY order/L1
    filing in the window. Control 1 keeps the registered "no material order" definition; the rows in it that
    carry an order of unknown size (no amount, no TTM, or not crawled) are counted and their P(+50%) printed.
    Not fixable here: pnl_quarterly covers about 66% of surviving symbols and about 3% of symbols that left
    the panel, so results are conditional on a mostly surviving population.
NOT FIXED
  * anatomy_1p5x.py 89-94 (low; industry labels are one Sep-2026 screener label per symbol, not point-in-time):
    ind / ind_heat_pct are read from rows.parquet as they are. The repo has no dated industry history to
    rebuild them from; the fix is an anatomy rebuild. It affects the hot filter of every cell and every Part B
    arm alike.
  * ocr_order_filings.py 52-55, 75-78 (low; a transient OCR download error is never retried): producer-side
    bug, not in this file. Effect here is coverage only (those filings have no full-text amount, so they cannot
    be material); the crawl's status counts (HTTP_* / ERR_* / NEEDS_OCR) are printed and stored in meta.orders.
DATA GAP (review 2026-09-27, not an audit finding): pnl_quarterly has a hole at the detail_api -> xbrl switch.
  308 symbols that have both 2024-12-31 and 2025-06-30 have no usable 2025-03-31 quarter (165 missing, 143 with
  filing_dt NaT), and 1,679 xbrl sales rows (2025: 1,066, 2026: 613) have filing_dt NaT, so their
  4-consecutive-quarter TTM stops from about 2025-09 until 2026-05. Full-universe PIT coverage of anatomy rows:
  70% in 2024, 61% in 2025, 50% in 2026 (monthly means 46-50% from 2025-09 to 2026-05). Orders filed while a
  symbol has no TTM cannot be material. The script prints the sales rows with no filing_dt, the quarter breaks
  and the amount-but-no-TTM orders by year. It flags every measured week below COV_MIN as a PROVISIONAL reason
  and reports a COVERED-WEEKS CHECK (the stock-week verdict on weeks with >= COV_MIN coverage only). It also
  counts control-1 rows whose order had an amount but no TTM; those rows are usually uncovered themselves, so
  they drop out of every cell. The registered conf window is not cut: inside the hole C, base and both controls
  are still drawn from the same covered rows, and the check shows what dropping those weeks does.
  TODO(data): backfill the missing 2025-03-31 quarter and the NaT filing_dt of xbrl rows in pnl_quarterly
  (fetch_pnl_history.py), then rerun.
Inherited from rows.parquet as they are (anatomy_1p5x audit findings, not this file's): the anatomy universe
(mcap >= 50cr, partly a screener backcast) and the ret60 / ret252 / px_sma200 / ind_heat_pct features.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

MAT, WIN, HOLD = 0.15, 60, 90
ERA_SPLIT = pd.Timestamp("2023-01-01")
COV_MIN = 0.50                      # materiality window starts when >= 50% of anatomy rows have a PIT TTM revenue
TTM_TOL = pd.Timedelta(days=200)    # registered: the TTM used must have become available within 200 days
CUTOFF_MIN = 15 * 60 + 30           # filings at/after 15:30 IST count from the next session
REG_COST = 0.005                    # registered flat round trip (close-entry mode)
CATCHUP_MAX = 250                   # sessions a name with no print at its exit may take to print again (sim_allin_matrix)
WINDOWS = ("2016-06-01", "2019-01-01")
PNL_UNIT_TO_CR = {"detail_api": 1e-2, "xbrl": 1e-7}   # pnl_quarterly manifest: detail_api Rs lakh, xbrl Rs

ORDER_CATS = ("order", "tender_L1")
ORDERS = ROOT / "data/derived/order_fulltext.parquet"
AMOUNTS = ROOT / "data/derived/order_amounts.parquet"
ANN = ROOT / "data/derived/announcements_historical.parquet"
PNL = ROOT / "data/derived/pnl_quarterly.parquet"
ROWS = ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet"
OUT = ROOT / "logs/leader_sleeve/hot_order_combo.json"
C_KEY = "C = hot & material order & >200DMA"
K1_KEY = "control = hot & >200DMA & no material order"
K2_KEY = "control2 = hot & >200DMA & no order filing"


# ---------------------------------------------------------------- inputs
def _dups(df: pd.DataFrame, name: str) -> None:
    d = df.duplicated(["symbol", "seq_id"])
    if d.any():
        raise SystemExit(f"{name} has {int(d.sum())} duplicate (symbol, seq_id) keys, e.g. "
                         f"{df.loc[d, ['symbol', 'seq_id']].head(3).to_dict('records')}; fix the producer first")


def current_categories(symbols: list[str] | None, keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Category of every announcement under the CURRENT event_materiality_study.CATS, via parse_order_amounts.classify
    on 'desc || attchmntText' (the function and text order_amounts.cat_current comes from).
    Returns (crawled keys found in announcements_historical: symbol, seq_id, cat_now, sort_date;
             order / tender_L1 filings under the current CATS that are NOT crawled: same columns;
             last sort_date scanned)."""
    import pyarrow.dataset as ds
    from parse_order_amounts import classify
    want = pd.MultiIndex.from_frame(keys[["symbol", "seq_id"]])
    flt = ds.field("symbol").isin(symbols) if symbols else None
    cols = ["symbol", "seq_id", "cat_now", "sort_date"]
    hit, extra, last = [pd.DataFrame(columns=cols)], [pd.DataFrame(columns=cols)], None
    for b in ds.dataset(ANN).to_batches(columns=["symbol", "seq_id", "desc", "attchmntText", "sort_date"], filter=flt,
                                        batch_size=200_000):
        t = b.to_pandas()
        if t.empty:
            continue
        t["seq_id"] = t["seq_id"].astype(str)
        t["cat_now"] = classify(t["desc"].fillna("") + " || " + t["attchmntText"].fillna(""))
        t = t[cols]
        crawled = pd.MultiIndex.from_frame(t[["symbol", "seq_id"]]).isin(want)
        hit.append(t[crawled])
        extra.append(t[~crawled & t["cat_now"].isin(ORDER_CATS)])
        mx = t["sort_date"].dropna().max()
        last = mx if last is None or (isinstance(mx, str) and mx > last) else last
    H = pd.concat(hit, ignore_index=True).drop_duplicates(["symbol", "seq_id"], keep="last")
    U = pd.concat(extra, ignore_index=True).drop_duplicates(["symbol", "seq_id"], keep="last")
    return H.reset_index(drop=True), U.reset_index(drop=True), last


def load_orders(symbols: list[str] | None, mode: str, panel_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Every crawled order/L1 filing (order_fulltext.parquet = the population) still classed order / tender_L1
    under the current CATS, with its filing timestamp and amount (Rs crore, NaN if unparsed), plus the order
    filings under the current CATS that the crawl does not have (no amount). Returns (orders, uncrawled, info);
    info['provisional_reasons'] lists every reason the verdict is not final (empty = final)."""
    ocols = set(pq.read_schema(ORDERS).names)
    O = pd.read_parquet(ORDERS, columns=[c for c in ("symbol", "seq_id", "cat", "d", "status", "amount_cr",
                                                     "order_amount_cr") if c in ocols])
    O["seq_id"] = O["seq_id"].astype(str)
    if symbols:
        O = O[O["symbol"].isin(symbols)]
    _dups(O, ORDERS.name)
    reasons: list[str] = []
    info: dict = dict(n_crawled=len(O), crawl_first_filing=str(pd.to_datetime(O["d"]).min().date()),
                      crawl_last_filing=str(pd.to_datetime(O["d"]).max().date()),
                      crawl_status=O["status"].value_counts().to_dict() if "status" in O.columns else None)
    if info["crawl_status"]:
        bad = {k: v for k, v in info["crawl_status"].items() if k.startswith(("HTTP_", "ERR_", "NEEDS_OCR"))}
        print(f"crawl status {info['crawl_status']} · no full text (never retried upstream: ocr_order_filings audit): {bad}")
    t_cls = time.time()
    H, U, ann_last = current_categories(symbols, O)
    O = O.merge(H.assign(_in_ann=True), on=["symbol", "seq_id"], how="left", validate="one_to_one")
    O["_in_ann"] = O["_in_ann"].eq(True)
    info.update(ann_last_filing=ann_last, crawled_not_in_announcements=int((~O["_in_ann"]).sum()))
    print(f"current CATS re-classified announcements in {time.time() - t_cls:.0f}s (last filing scanned {ann_last}); "
          f"{info['crawled_not_in_announcements']} crawled filings are not in {ANN.name} (crawl-time category kept, "
          f"no timestamp)")
    A, cov = None, 0.0
    if AMOUNTS.exists():
        acols = set(pq.read_schema(AMOUNTS).names)
        need = {"symbol", "seq_id", "order_amount_cr", "cat_current", "d"}
        if not need <= acols:
            raise SystemExit(f"{AMOUNTS} lacks {sorted(need - acols)}; refusing to guess the join key / category")
        fxc = [c for c in ("currency", "usdinr") if c in acols]
        A = pd.read_parquet(AMOUNTS, columns=sorted(need) + fxc)
        A["seq_id"] = A["seq_id"].astype(str)
        if symbols:
            A = A[A["symbol"].isin(symbols)]
        _dups(A, AMOUNTS.name)
        if len(fxc) == 2:   # the audit's 83.0 fallback must not come back: a USD amount needs a historical rate
            nofx = A["currency"].eq("USD") & A["order_amount_cr"].notna() & A["usdinr"].isna()
            if nofx.any():
                raise SystemExit(f"{AMOUNTS.name}: {int(nofx.sum())} USD order amounts have no USDINR (fixed-rate "
                                 f"conversion?), e.g. {A.loc[nofx, ['symbol', 'seq_id', 'd']].head(3).to_dict('records')}")
            info["amounts_usd_rows"] = int(A["currency"].eq("USD").sum())
        elif mode != "legacy":
            print(f"WARNING: {AMOUNTS.name} has no currency/usdinr columns; USD conversion unverified")
        man = AMOUNTS.with_name(AMOUNTS.name + ".manifest.json")
        desc = json.loads(man.read_text()).get("columns", {}).get("order_amount_cr") if man.exists() else None
        print(f"amounts file: {AMOUNTS.name}:order_amount_cr — manifest says: {desc or 'NO MANIFEST ENTRY (unit unverified)'}")
        hit = O[["symbol", "seq_id"]].merge(A[["symbol", "seq_id"]], on=["symbol", "seq_id"], how="left", indicator=True)
        n_hit = int((hit["_merge"] == "both").sum())
        cov = n_hit / len(O) if len(O) else 0.0
        info.update(amounts_rows=len(A), amounts_matched=n_hit, amounts_coverage_pct=cov * 100,
                    amounts_rows_not_in_fulltext=len(A) - n_hit,
                    amounts_date_range=[str(pd.to_datetime(A["d"]).min().date()), str(pd.to_datetime(A["d"]).max().date())])
        print(f"amounts file covers {n_hit}/{len(O)} crawled filings ({cov:.1%}); filings "
              f"{info['amounts_date_range'][0]}..{info['amounts_date_range'][1]}; {len(A) - n_hit} amount rows not in "
              f"{ORDERS.name} are ignored (left join)")
    elif mode == "revised":
        raise SystemExit(f"--amounts revised but {AMOUNTS} does not exist")
    revised = A is not None and (mode == "revised" or (mode == "auto" and cov >= 1.0))
    if revised:
        O = O.drop(columns=[c for c in ("amount_cr", "order_amount_cr") if c in O.columns]).merge(
            A[["symbol", "seq_id", "cat_current", "order_amount_cr"]].assign(_in_amounts=True),
            on=["symbol", "seq_id"], how="left", validate="one_to_one").rename(
            columns={"order_amount_cr": "amount_cr", "cat_current": "cat_amounts"})
        known = O["_in_amounts"].eq(True)
        src = "data/derived/order_amounts.parquet:order_amount_cr (order-attached parser)"
        if cov < 1.0:
            reasons.append(f"{AMOUNTS.name} covers {cov:.1%} of crawled filings; the other {int((~known).sum())} have "
                           f"amount NaN (never material)")
        mism = known & O["_in_ann"] & (O["cat_amounts"].fillna("none") != O["cat_now"].fillna("none"))
        info["cat_mismatch_amounts_vs_current_cats"] = int(mism.sum())
        if mism.any():
            reasons.append(f"{AMOUNTS.name} cat_current disagrees with the current CATS on {int(mism.sum())} filings "
                           f"(built with another CATS version; rebuild it)")
    else:
        legacy = "order_amount_cr" not in ocols
        src = ("data/derived/order_fulltext.parquet:amount_cr" +
               (" (legacy largest-amount parser, audit-flagged)" if legacy else " (consolidated order-attached amount)"))
        why = ("not found" if A is None else f"covers only {cov:.1%} of crawled filings" if mode == "auto"
               else "not used (--amounts legacy)")
        print(f"WARNING: {AMOUNTS.name} {why} -> {src} for EVERY filing (one parser for both eras)")
        reasons.append(f"amounts from {src}; {AMOUNTS.name} {why}" +
                       ("; USD amounts before 2024-02-19 converted at a fixed 83.0 by the legacy crawl" if legacy else ""))
    # current-category filter (every mode): a filing no longer classed order / tender_L1 under the CURRENT CATS
    # (honours, arbitration awards, ratings, results releases ...) is not an order filing (MO and AO alike)
    drop = O["_in_ann"] & ~O["cat_now"].isin(ORDER_CATS)
    info["dropped_not_order_under_current_cats"] = O.loc[drop, "cat_now"].fillna("none").value_counts().to_dict()
    print(f"dropped {int(drop.sum())} crawled filings whose current category is not order/tender_L1: "
          f"{info['dropped_not_order_under_current_cats']}")
    O = O[~drop]
    # recall: order filings under the current CATS that the crawl does not have (no amount -> never material)
    U["ts"] = pd.to_datetime(U["sort_date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    n_nots = int(U["ts"].isna().sum())
    U = U.dropna(subset=["ts"])
    inside = (U["ts"] >= pd.Timestamp(info["crawl_first_filing"])) & \
             (U["ts"] < pd.Timestamp(info["crawl_last_filing"]) + pd.Timedelta(days=1))
    ue = np.where(U["ts"] >= ERA_SPLIT, "conf", "disc")
    info["uncrawled_current_cat_orders"] = dict(
        inside_crawl_dates={e: int((inside & (ue == e)).sum()) for e in ("disc", "conf")},
        outside_crawl_dates=int((~inside).sum()), no_timestamp_dropped=n_nots)
    print(f"order/L1 filings under the current CATS that are NOT crawled: {info['uncrawled_current_cat_orders']} "
          f"(no amount: never material; they count as order filings for control 2)")
    if inside.any():
        u = info["uncrawled_current_cat_orders"]["inside_crawl_dates"]
        reasons.append(f"{int(inside.sum())} order/L1 filings under the current CATS inside the crawl's dates are not "
                       f"crawled (disc {u['disc']}, conf {u['conf']}): they cannot be material")
    if pd.Timestamp(info["crawl_last_filing"]) < panel_end - pd.Timedelta(days=60):
        reasons.append(f"order crawl ends {info['crawl_last_filing']}, > 60 days before the panel end {panel_end.date()}")
    info.update(amount_source=src, provisional_reasons=reasons)
    O = O.drop(columns=[c for c in ("_in_amounts", "_in_ann", "cat_amounts") if c in O.columns])
    O["ts"] = pd.to_datetime(O["sort_date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    miss = O["ts"].isna()
    O.loc[miss, "ts"] = pd.to_datetime(O.loc[miss, "d"]) + pd.Timedelta(hours=23, minutes=59)  # unknown time = after close
    print(f"orders: {len(O)} filings · no timestamp (treated as after close): {int(miss.sum())} · "
          f"filed at/after 15:30: {(O['ts'].dt.hour * 60 + O['ts'].dt.minute >= CUTOFF_MIN).mean() * 100:.1f}%")
    O = O.dropna(subset=["ts"])
    O["ts"] = O["ts"].astype("datetime64[ns]")
    U["ts"] = U["ts"].astype("datetime64[ns]")
    return O.reset_index(drop=True), U[["symbol", "seq_id", "cat_now", "ts"]].reset_index(drop=True), info


def build_ttm(symbols: list[str] | None) -> tuple[pd.DataFrame, dict]:
    """PIT TTM revenue (Rs crore): one basis per symbol, 4 consecutive calendar quarters, as-filed values,
    available when the last of the four quarters was filed; stale late filings never replace a newer TTM.
    Also returns the pnl_quarterly gaps that stop a TTM: rows with sales but no filing date, and quarter breaks
    (a calendar quarter missing between two filed quarters of the chosen basis), by quarter_end year."""
    q = pd.read_parquet(PNL, columns=["symbol", "quarter_end", "filing_dt", "basis", "source", "net_sales"])
    if symbols:
        q = q[q["symbol"].isin(symbols)]
    nat = q["filing_dt"].isna() & q["net_sales"].notna()
    gaps: dict = dict(sales_without_filing_dt=nat.groupby(q["quarter_end"].dt.year).sum().loc[lambda s: s > 0]
                      .astype(int).to_dict(),
                      sales_without_filing_dt_by_source=q.loc[nat, "source"].value_counts().to_dict())
    q = q.dropna(subset=["filing_dt", "net_sales"])
    bad = set(q["source"].unique()) - set(PNL_UNIT_TO_CR)
    if bad:
        raise SystemExit(f"pnl_quarterly source(s) {sorted(bad)} have no unit in the manifest")
    q["sales_cr"] = q["net_sales"] * q["source"].map(PNL_UNIT_TO_CR)
    q = q.sort_values("filing_dt", kind="mergesort").drop_duplicates(["symbol", "quarter_end", "basis"], keep="first")
    n = q.groupby(["symbol", "basis"]).size().unstack(fill_value=0).reindex(columns=["con", "sa"], fill_value=0)
    keep = pd.DataFrame({"symbol": n.index, "basis": np.where(n["con"] >= n["sa"], "con", "sa")})
    q = q.merge(keep, on=["symbol", "basis"]).sort_values(["symbol", "quarter_end"]).reset_index(drop=True)
    qn = q["quarter_end"].dt.year * 4 + q["quarter_end"].dt.quarter
    brk = (qn - qn.groupby(q["symbol"]).shift(1)) > 1
    gaps["quarter_breaks"] = brk.groupby(q["quarter_end"].dt.year).sum().loc[lambda s: s > 0].astype(int).to_dict()
    consec = (qn - qn.groupby(q["symbol"]).shift(3)) == 3
    q["rev_ttm_cr"] = q.groupby("symbol")["sales_cr"].transform(lambda s: s.rolling(4).sum()).where(consec)
    fsec = (q["filing_dt"].astype("datetime64[ns]").astype("int64") // 10**9).astype(float)
    q["avail"] = pd.to_datetime(fsec.groupby(q["symbol"]).transform(lambda s: s.rolling(4).max()), unit="s")
    T = q.loc[q["rev_ttm_cr"] > 0, ["symbol", "quarter_end", "avail", "rev_ttm_cr"]].dropna()
    T = T.sort_values(["symbol", "avail", "quarter_end"], kind="mergesort")
    T = T[T["quarter_end"] >= T.groupby("symbol")["quarter_end"].cummax()]
    T["avail"] = T["avail"].astype("datetime64[ns]")
    print(f"pnl_quarterly gaps that stop a TTM (by quarter_end year): sales rows with no filing_dt "
          f"{gaps['sales_without_filing_dt']} {gaps['sales_without_filing_dt_by_source']} · quarter breaks "
          f"{gaps['quarter_breaks']}  (TODO(data): backfill in fetch_pnl_history.py)")
    return T.sort_values(["avail", "quarter_end"], kind="mergesort").reset_index(drop=True), gaps


def asof_ttm(left: pd.DataFrame, when: str, T: pd.DataFrame) -> pd.Series:
    """TTM revenue known at left[when] (NaN if none became available within TTM_TOL)."""
    L = left[["symbol", when]].reset_index().rename(columns={"index": "_i"})
    L[when] = L[when].astype("datetime64[ns]")
    L = L.sort_values(when, kind="mergesort")
    m = pd.merge_asof(L, T[["symbol", "avail", "rev_ttm_cr"]], left_on=when, right_on="avail", by="symbol",
                      direction="backward", tolerance=TTM_TOL)
    return m.set_index("_i")["rev_ttm_cr"].reindex(left.index)


def effective_session(ts: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    """First session whose CLOSE may use the filing: same day if filed before 15:30 on a session, else the next one."""
    day = ts.dt.normalize().values
    mins = (ts.dt.hour * 60 + ts.dt.minute).values
    same = cal.searchsorted(day, side="left")
    nxt = cal.searchsorted(day, side="right")
    is_sess = (same < len(cal)) & (cal.values[np.minimum(same, len(cal) - 1)] == day)
    pos = np.where(is_sess & (mins < CUTOFF_MIN), same, nxt)
    out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = pos < len(cal)
    out[ok] = cal.values[pos[ok]]
    return pd.Series(out, index=ts.index)


def window_flags(S: pd.DataFrame, ev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per weekly row: any event with effective session in (t - WIN days, t], and the id of the latest one."""
    flag = np.zeros(len(S), bool)
    key = np.full(len(S), -1, np.int64)
    ev = ev.sort_values(["symbol", "eff"])
    arr = {s: (v["eff"].values.astype("datetime64[ns]"), v["eid"].values) for s, v in ev.groupby("symbol")}
    t = S["trade_date"].values.astype("datetime64[ns]")
    for s, idx in S.groupby("symbol").indices.items():
        a = arr.get(s)
        if a is None:
            continue
        ti = t[idx]
        pos = np.searchsorted(a[0], ti, side="right") - 1
        ok = pos >= 0
        ok[ok] = a[0][pos[ok]] > ti[ok] - np.timedelta64(WIN, "D")
        flag[idx] = ok
        key[idx[ok]] = a[1][pos[ok]]
    return flag, key


def panel_labels(S: pd.DataFrame, close_w: pd.DataFrame, high_w: pd.DataFrame, low_w: pd.DataFrame,
                 h: int = 95) -> tuple[pd.DataFrame, pd.Timestamp]:
    """y95 / s95 / fc95 on the session calendar (anatomy definitions: max high over the next 95 sessions >= 1.5x
    the entry close; close at +95 sessions >= 1.5x; close +95 / close). One definedness rule for all three and
    for hits and non-hits alike: a row is labelled only when its whole window lies inside the panel (entry on or
    before cal[-1-h]) and the exit is known (forward_window freezes a delisted name at its final close). A name
    with no print inside the window keeps y95 = 0 (it could not touch +50%). Returns (labels, last labelled date)."""
    fw = rp.forward_window(close_w, high_w, low_w, h)
    last = len(close_w) - 1 - h
    r = close_w.index.get_indexer(S["trade_date"])
    c = close_w.columns.get_indexer(S["symbol"])
    ok = (r >= 0) & (c >= 0)

    def pick(W: pd.DataFrame) -> np.ndarray:
        out = np.full(len(S), np.nan)
        out[ok] = W.to_numpy(dtype=float)[r[ok], c[ok]]
        return out

    c0, hi, ex = pick(close_w), pick(fw["hi"]), pick(fw["exit"])
    known = ok & (r <= last) & np.isfinite(c0) & (c0 > 0) & np.isfinite(ex)
    y, s, fc = (np.full(len(S), np.nan) for _ in range(3))
    with np.errstate(invalid="ignore"):
        y[known] = (np.nan_to_num(hi[known], nan=-np.inf) >= 1.5 * c0[known]).astype(float)
        s[known] = (ex[known] >= 1.5 * c0[known]).astype(float)
    fc[known] = ex[known] / c0[known]
    return pd.DataFrame({"y95": y, "s95": s, "fc95": fc}, index=S.index), close_w.index[max(last, 0)]


# ---------------------------------------------------------------- Part A
def boot_means(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    """Cluster bootstrap of a mean: resample whole clusters (symbols) with replacement."""
    if len(y) == 0:
        return np.full(B, np.nan)
    codes, _ = pd.factorize(pd.Series(groups))
    G = int(codes.max()) + 1
    s = np.bincount(codes, weights=y, minlength=G)
    c = np.bincount(codes, minlength=G).astype(float)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    with np.errstate(all="ignore"):   # macOS Accelerate matmul raises spurious overflow warnings
        return (W @ s) / (W @ c)


def ci(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]


def part_a(S: pd.DataFrame, cov_start: pd.Timestamp, cov_wk: pd.Series, rng: np.random.Generator, B: int,
           reasons: list[str]) -> dict:
    res: dict = {}
    L = S.dropna(subset=["y95"])
    era = np.where(L["trade_date"] >= ERA_SPLIT, "conf", np.where(L["trade_date"] >= cov_start, "disc", ""))
    L = L.assign(era_w=era)[era != ""]
    hot, up, cov = L["ind_heat_pct"] >= 0.9, L["px_sma200"] > 0, L["covered"]
    cells = {C_KEY: cov & hot & L["mat_order"] & up, K1_KEY: cov & hot & up & ~L["mat_order"],
             K2_KEY: cov & hot & up & ~L["any_order"], "material order alone": cov & L["mat_order"],
             "material order & >200DMA": cov & L["mat_order"] & up, "hot & material order": cov & hot & L["mat_order"]}
    windows = {"disc": f"{cov_start.date()} .. 2022-12-31", "conf": f"2023-01-01 .. {L['trade_date'].max().date()}"}
    print(f"\n=== PART A === disc window {windows['disc']} (COV_START: first week with >= {COV_MIN:.0%} PIT TTM coverage) · "
          f"conf window {windows['conf']} · population = rows with a PIT TTM revenue at t")
    base, base_all, uncovered = {}, {}, {}
    for e in ("disc", "conf"):
        E = L[L["era_w"] == e]
        base[e], base_all[e] = E.loc[E["covered"], "y95"].mean(), E["y95"].mean()
        U = E[~E["covered"] & (E["ind_heat_pct"] >= 0.9) & (E["px_sma200"] > 0)]
        uncovered[e] = dict(n=len(U), p=U["y95"].mean() * 100)
        res[f"base|{e}"] = dict(window=windows[e], p_covered=base[e] * 100, p_all_rows=base_all[e] * 100,
                                n_covered=int(E["covered"].sum()), n_all=len(E), covered_share=E["covered"].mean() * 100,
                                hot_up_uncovered_excluded=uncovered[e])
        print(f"  base {e}: covered P(+50%) {base[e] * 100:5.1f}% (n={int(E['covered'].sum())}) · all rows {base_all[e] * 100:5.1f}% "
              f"(n={len(E)}) · excluded hot&>200DMA rows with no TTM: n={uncovered[e]['n']} P(+50%) {uncovered[e]['p']:.1f}%")
    boots: dict = {}
    for k, m in cells.items():
        out = []
        for e in ("disc", "conf"):
            E = L[m & (L["era_w"] == e)]
            r = dict(n=len(E), n_sym=int(E["symbol"].nunique()), p=E["y95"].mean() * 100, sus=E["s95"].mean() * 100,
                     ret=(E["fc95"].mean() - 1) * 100, lift=E["y95"].mean() / base[e], lift_all_rows=E["y95"].mean() / base_all[e])
            if "material" in k and "no material" not in k:
                ep = E.sort_values("trade_date", kind="mergesort").drop_duplicates("mat_key")
                r.update(n_ep=len(ep), n_ep_sym=int(ep["symbol"].nunique()), p_ep=ep["y95"].mean() * 100,
                         sus_ep=ep["s95"].mean() * 100, ret_ep=(ep["fc95"].mean() - 1) * 100, lift_ep=ep["y95"].mean() / base[e])
                if k == C_KEY:
                    boots[e] = boot_means(ep["y95"].to_numpy(float), ep["symbol"].to_numpy(), rng, B)
            if k in (K1_KEY, K2_KEY):
                boots[(k, e)] = boot_means(E["y95"].to_numpy(float), E["symbol"].to_numpy(), rng, B)
            res[f"{k}|{e}"] = r
            s = f"{e} n={r['n']:>5} ({r['n_sym']:>3} sym) P(+50%) {r['p']:5.1f}% ({r['lift']:.2f}x) sus {r['sus']:4.1f}% mean95 {r['ret']:+5.1f}%"
            if "n_ep" in r:
                s += f" | episodes {r['n_ep']:>4} ({r['n_ep_sym']} sym) P {r['p_ep']:5.1f}% ({r['lift_ep']:.2f}x)"
            out.append(s)
        print(f"  {k:<44}| " + " | ".join(out))
    # control 1 = "no material order": how many of its rows carry an order whose size is unknown (no amount, no PIT
    # TTM at filing, or not crawled) and could be material; order_no_ttm = amount known, TTM missing (pnl gaps)
    for e in ("disc", "conf"):
        E = L[cells[K1_KEY] & (L["era_w"] == e)]
        us, nt = E["order_unsized"], E["order_no_ttm"]
        d_ = dict(n_unsized_order=int(us.sum()), p_unsized_order=E.loc[us, "y95"].mean() * 100,
                  p_excl_unsized=E.loc[~us, "y95"].mean() * 100, n_order_no_ttm=int(nt.sum()),
                  p_order_no_ttm=E.loc[nt, "y95"].mean() * 100)
        res[f"{K1_KEY}|{e}"].update(d_)
        print(f"  control {e}: {d_['n_unsized_order']} of {len(E)} rows carry an order of unknown size "
              f"(P(+50%) {d_['p_unsized_order']:.1f}%; control without them {d_['p_excl_unsized']:.1f}%) · "
              f"{d_['n_order_no_ttm']} of those had an amount but no PIT TTM (P {d_['p_order_no_ttm']:.1f}%)")
    # bootstrap CIs for the C cell (episode level; symbol-cluster resampling; base and controls resampled too)
    for e in ("disc", "conf"):
        E = L[(L["era_w"] == e) & L["covered"]]
        bb = boot_means(E["y95"].to_numpy(float), E["symbol"].to_numpy(), rng, B)
        c_ = boots[e]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = dict(p_ep_ci=[x * 100 for x in ci(c_)], lift_ep_ci=ci(c_ / bb),
                     vs_control_ci=ci(c_ / boots[(K1_KEY, e)]), vs_control2_ci=ci(c_ / boots[(K2_KEY, e)]))
        res[f"{C_KEY}|{e}"].update(r)
        print(f"  C {e} episode bootstrap (B={B}, symbol clusters): P(+50%) 95% CI {r['p_ep_ci'][0]:.1f}..{r['p_ep_ci'][1]:.1f}% · "
              f"lift vs base {r['lift_ep_ci'][0]:.2f}..{r['lift_ep_ci'][1]:.2f}x · vs control {r['vs_control_ci'][0]:.2f}.."
              f"{r['vs_control_ci'][1]:.2f}x · vs control2 {r['vs_control2_ci'][0]:.2f}..{r['vs_control2_ci'][1]:.2f}x")
    C = {e: res[f"{C_KEY}|{e}"] for e in ("disc", "conf")}
    K = {e: res[f"{K1_KEY}|{e}"] for e in ("disc", "conf")}
    ok = all(C[e]["n"] >= 50 and C[e]["lift"] >= 1.5 and C[e]["p"] >= 1.2 * K[e]["p"] for e in C)
    ok_ep = all(C[e]["n_ep"] >= 50 and C[e]["lift_ep"] >= 1.5 and C[e]["p_ep"] >= 1.2 * K[e]["p"] for e in C)
    # covered-weeks check: the same stock-week test on weeks where >= COV_MIN of anatomy rows have a PIT TTM
    # (the COV_START rule applied to every week, so pnl_quarterly holes such as 2025-09..2026-05 drop out)
    good = L["trade_date"].map(cov_wk).ge(COV_MIN)
    chk: dict = {}
    for e in ("disc", "conf"):
        Ea, E = L[L["era_w"] == e], L[good & (L["era_w"] == e)]
        hot_, up_ = E["ind_heat_pct"] >= 0.9, E["px_sma200"] > 0
        b = E.loc[E["covered"], "y95"].mean()
        c_ = E.loc[E["covered"] & hot_ & up_ & E["mat_order"], "y95"]
        k_ = E.loc[E["covered"] & hot_ & up_ & ~E["mat_order"], "y95"]
        chk[e] = dict(weeks=int(E["trade_date"].nunique()), weeks_dropped=int(Ea["trade_date"].nunique() - E["trade_date"].nunique()),
                      n=len(c_), p=c_.mean() * 100, lift=c_.mean() / b, vs_control=c_.mean() / k_.mean())
    ok_cw = all(chk[e]["n"] >= 50 and chk[e]["lift"] >= 1.5 and chk[e]["vs_control"] >= 1.2 for e in chk)
    tag = f" — PROVISIONAL ({'; '.join(reasons)})" if reasons else ""
    print(f"\nREGISTERED VERDICT (stock-weeks, same window + population): {'PASS' if ok else 'FAIL'} "
          f"(C >= 1.5x base and >= 1.2x control, both eras, n >= 50){tag}")
    print(f"CLUSTERED CHECK (one row per order episode, n_ep >= 50): {'PASS' if ok_ep else 'FAIL'}{tag}")
    print(f"COVERED-WEEKS CHECK (weeks with >= {COV_MIN:.0%} PIT TTM coverage only): {'PASS' if ok_cw else 'FAIL'} · " +
          " · ".join(f"{e}: {chk[e]['weeks']} weeks ({chk[e]['weeks_dropped']} dropped) n={chk[e]['n']} P {chk[e]['p']:.1f}% "
                     f"({chk[e]['lift']:.2f}x base, {chk[e]['vs_control']:.2f}x control)" for e in chk) + tag)
    res["covered_weeks_check"] = dict(chk, verdict="PASS" if ok_cw else "FAIL")
    res["verdict"] = dict(registered="PASS" if ok else "FAIL", clustered="PASS" if ok_ep else "FAIL",
                          covered_weeks="PASS" if ok_cw else "FAIL", provisional=bool(reasons), provisional_reasons=reasons)
    return res


# ---------------------------------------------------------------- Part B
def bh_pool(S: pd.DataFrame) -> pd.DataFrame:
    """sim_allin_matrix.rule(broad=True, top=10, gate=False), step for step, keeping the ranked hot-industry pool.
    rk = the registered ret60 rank inside the WHOLE hot industry; rk_tie = material-order names first, then ret60."""
    W = S.dropna(subset=["ind", "ret60"]).copy()
    W["n"] = W.groupby(["trade_date", "ind"])["ret60"].transform("size"); W = W[W["n"] >= 5]
    h = W.groupby(["trade_date", "ind"])["ret60"].mean().rename("h").reset_index()
    h["hot"] = h.groupby("trade_date")["h"].rank(pct=True) >= 0.9
    W = W.merge(h[["trade_date", "ind", "hot"]], on=["trade_date", "ind"])
    W["rk"] = W.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
    W = W[W["hot"]].sort_values(["trade_date", "ind", "mat_order", "ret60"], ascending=[True, True, False, False], kind="mergesort")
    W["rk_tie"] = W.groupby(["trade_date", "ind"]).cumcount() + 1
    return W


def picks(P: pd.DataFrame, mode: str) -> dict:
    base = (P["ret252"] > 0.5) & P["core"]
    X = P[base & (P["rk_tie"] <= 10)] if mode == "tie" else P[base & (P["rk"] <= 10)]
    if mode == "hard":
        X = X[X["mat_order"]]
    return {d: (g["symbol"].tolist(), g["adv_cr"].to_numpy(float)) for d, g in X.groupby("trade_date")}


def era_dd(s: pd.Series) -> tuple[float, float]:
    pre = s[s.index < ERA_SPLIT]
    post = s[s.index >= (pre.index[-1] if len(pre) else ERA_SPLIT)]
    f = lambda x: float((x / x.cummax() - 1).min() * 100) if len(x) > 1 else np.nan  # noqa: E731
    return f(pre), f(post)


def exit_catchup(M: dict, j: np.ndarray, t0: int, open_exit: bool) -> tuple[np.ndarray, np.ndarray]:
    """sim_allin_matrix._catchup: exit factor for positions that must be sold at session t0 but have no print
    there. Sold at the first session they print again within CATCHUP_MAX (its close; its open when open_exit),
    else frozen at the last close (factor 1). Returns (factor, session index of that print or -1)."""
    f, at = np.ones(len(j)), np.full(len(j), -1)
    hi = min(t0 + CATCHUP_MAX, len(M["cal"]) - 1)
    if hi <= t0 or not len(j):
        return f, at
    tr = M["TR"][t0 + 1:hi + 1][:, j]
    has, first = tr.any(axis=0), tr.argmax(axis=0)
    for m in np.flatnonzero(has):
        t, c = t0 + 1 + first[m], j[m]
        f[m] = (np.prod(1 + np.nan_to_num(M["R"][t0:t, c])) * (1 + np.nan_to_num(M["ON"][t, c])) if open_exit
                else np.prod(1 + np.nan_to_num(M["R"][t0 + 1:t + 1, c])))
        at[m] = t
    return f, at


def simulate(pk: dict, start: str, entry: str, M: dict) -> dict:
    """All-in, equal weight, rotate every HOLD/5 weekly rows (HOLD sessions), HOLD/5 phase offsets.
    close: buy at the signal close, hold to the next signal close, flat REG_COST per rotation (registered).
    open : buy at the next open, skip locked / non-trading entries, sell at the open after the next signal,
           ADV-scaled round trip (research_panel.cost_rt).
    A held name with no print at its exit session is sold late (exit_catchup); the value is booked at the
    rotation. The last rotation's exit is booked on the final NAV mark."""
    cal, dpos, N = M["cal"], M["dpos"], len(M["cal"])
    Rn, CL, EP, LK, TR, ON, col = M["R"], M["CL"], M["EP"], M["LK"], M["TR"], M["ON"], M["col"]
    dates = [d for d in M["wk"] if d >= pd.Timestamp(start) and d in dpos]
    step = HOLD // 5
    rows, n_names, n_skip, n_try, n_pos, n_late = [], [], 0, 0, 0, 0
    late: dict = {}     # (symbol, exit session) -> (first print back or None, exit factor)

    def book(jj: np.ndarray, t0: int, f: np.ndarray, at: np.ndarray) -> None:
        for c, t, x in zip(jj, at, f):
            late[(str(M["syms"][c]), str(cal[t0].date()))] = (str(cal[t].date()) if t >= 0 else None, float(x))

    for ph in range(step):
        seq = dates[ph::step]
        if not seq:
            continue
        v, idx, vals = 1.0, [cal[dpos[seq[0]]]], [1.0]
        for k, d in enumerate(seq):
            i0 = dpos[d]
            i1 = dpos[seq[k + 1]] if k + 1 < len(seq) else min(i0 + HOLD, N - 1)
            if i1 <= i0:
                continue
            names, adv = pk.get(d, ([], np.array([])))
            j = np.array([col.get(n, -1) for n in names], dtype=int)
            adv = adv[j >= 0]; j = j[j >= 0]
            path, vend = np.ones(i1 - i0), 1.0
            if entry == "close" and len(j):
                g = np.cumprod(1 + np.nan_to_num(Rn[i0 + 1:i1 + 1][:, j]), axis=0)
                path = g.mean(axis=1) * (1 - REG_COST)
                fin, gone = g[-1].copy(), ~TR[i1, j]          # sold at the close of i1; no print there -> sold late
                if gone.any():
                    f, at = exit_catchup(M, j[gone], i1, False)
                    fin[gone] *= f; n_late += int(gone.sum()); book(j[gone], i1, f, at)
                vend = float(fin.mean() * (1 - REG_COST)); n_pos += len(j)
            elif entry == "open" and len(j):
                ok = ~LK[i0, j] & ~np.isnan(EP[i0, j])
                n_try += len(j); n_skip += int((~ok).sum())
                j, adv = j[ok], adv[ok]
                if len(j):
                    first = CL[i0 + 1, j] / EP[i0, j]
                    first = np.where(np.isfinite(first), first, 1.0)
                    g = np.vstack([first[None, :], first * np.cumprod(1 + np.nan_to_num(Rn[i0 + 2:i1 + 1][:, j]), axis=0)])
                    val = g * (1 - rp.cost_rt(pd.Series(adv)).to_numpy())
                    leg = np.ones(len(j))                          # no session after the hold: exit at the last close
                    if i1 + 1 < N:                                 # sold at the open of i1+1 (open / last close)
                        pr = TR[i1 + 1, j]
                        leg = np.where(pr, 1 + np.nan_to_num(ON[i1 + 1, j]), 1.0)
                        if (~pr).any():
                            f, at = exit_catchup(M, j[~pr], i1 + 1, True)
                            leg[~pr] = f; n_late += int((~pr).sum()); book(j[~pr], i1 + 1, f, at)
                    path = val.mean(axis=1); vend = float((val[-1] * leg).mean()); n_pos += len(j)
            n_names.append(len(j))
            vals.extend(v * path); idx.extend(cal[i0 + 1:i1 + 1]); v *= vend
        vals[-1] = v
        s = pd.Series(vals, index=idx)
        m = rp.nav_metrics(s)
        dd_d, dd_c = era_dd(s)
        rows.append(dict(cagr=m["cagr"], dd=m["maxdd"], sharpe=m["sharpe"], cagr_disc=m["cagr_disc"],
                         cagr_conf=m["cagr_conf"], dd_disc=dd_d, dd_conf=dd_c))
    r = pd.DataFrame(rows)
    out = dict(cagr_med=r["cagr"].median(), cagr_worst=r["cagr"].min(), dd_med=r["dd"].median(), dd_worst=r["dd"].min(),
               sharpe_med=r["sharpe"].median(), cagr_disc_med=r["cagr_disc"].median(), cagr_conf_med=r["cagr_conf"].median(),
               dd_disc_worst=r["dd_disc"].min(), dd_conf_worst=r["dd_conf"].min(), names_per_rotation=float(np.mean(n_names)))
    big = sorted(late.items(), key=lambda kv: -abs(kv[1][1] - 1))[:5]
    out.update(positions=n_pos, late_exits=n_late, late_exits_unique=len(late),
               late_never_back=sum(1 for x in late.values() if x[0] is None),
               late_largest=[dict(symbol=k[0], exit=k[1], sold=x[0], move_pct=(x[1] - 1) * 100) for k, x in big])
    if entry == "open":
        out["entries_skipped_pct"] = n_skip / n_try * 100 if n_try else np.nan
    return out


def part_b(S: pd.DataFrame, cov_start: pd.Timestamp, M: dict, entries: list[str], reasons: list[str]) -> dict:
    res: dict = {}
    P = bh_pool(S)
    starts = list(WINDOWS) + [str(cov_start.date())]
    print(f"\n=== PART B: B+H all-in, {HOLD}-session hold, {HOLD // 5} phases · windows {', '.join(starts)} "
          f"(last = COV_START) · entry {'/'.join(entries)} ===")
    if reasons:
        print(f"  HARD / TIE are PROVISIONAL ({'; '.join(reasons)})")
    for mode in ("asis", "hard", "tie"):
        pk = picks(P, mode)
        weeks = pd.DatetimeIndex(M["wk"])
        for start in starts:
            inv = np.mean([d in pk for d in weeks[weeks >= pd.Timestamp(start)]]) * 100
            pre_cov = mode != "asis" and pd.Timestamp(start) < cov_start
            for entry in entries:
                r = simulate(pk, start, entry, M)
                r.update(weeks_with_picks=inv, pre_coverage_not_a_test=pre_cov)
                res[f"B|{mode}|{start}|{entry}"] = r
                sk = f" · skipped {r['entries_skipped_pct']:.1f}%" if entry == "open" else ""
                print(f"  {mode:<4} {start}+ {entry:<5} CAGR med {r['cagr_med']:+6.1f}% (worst {r['cagr_worst']:+6.1f}) · maxDD med "
                      f"{r['dd_med']:6.1f}% (worst {r['dd_worst']:6.1f}) · Sharpe {r['sharpe_med']:.2f} · disc CAGR {r['cagr_disc_med']:+.1f}% "
                      f"DD {r['dd_disc_worst']:.1f} · conf CAGR {r['cagr_conf_med']:+.1f}% DD {r['dd_conf_worst']:.1f} · "
                      f"names {r['names_per_rotation']:.1f} · weeks w/ picks {inv:.0f}%{sk} · late exits "
                      f"{r['late_exits_unique']} ({r['late_never_back']} never back)"
                      f"{'  [PRE-COVERAGE: not a test]' if pre_cov else ''}", flush=True)
                if start == starts[0] and r["late_largest"]:
                    print("        largest late exits (no print at the exit session; sold at the first print back): " +
                          ", ".join(f"{x['symbol']} {x['exit']}->{x['sold'] or 'never'} {x['move_pct']:+.1f}%"
                                    for x in r["late_largest"]))
    return res


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entry", choices=["close", "open", "both"], default="both",
                    help="close = registered signal-close entry, flat 0.5%%; open = next open, locked skipped, ADV costs")
    ap.add_argument("--labels", choices=["panel", "anatomy"], default="panel",
                    help="panel = recompute y95/s95/fc95 on the session calendar (default); anatomy = rows.parquet labels")
    ap.add_argument("--amounts", choices=["auto", "revised", "legacy"], default="auto",
                    help="auto = order_amounts.parquet only when it covers every crawled filing, else legacy (PROVISIONAL); "
                         "revised = order_amounts.parquet even if partial (unmatched = NaN, PROVISIONAL); legacy = "
                         "order_fulltext.amount_cr (PROVISIONAL)")
    ap.add_argument("--panel", default=None, help="price panel parquet (default research_panel.PANEL)")
    ap.add_argument("--symbols", default=None, help="comma list: restrict every input (smoke tests)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap draws")
    ap.add_argument("--out", default=None, help=f"results JSON (default {OUT.relative_to(ROOT)}; not written with --symbols)")
    args = ap.parse_args()
    t0 = time.time()
    if args.panel:
        rp.PANEL = Path(args.panel)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    flt = [("symbol", "in", syms)] if syms else None
    entries = ["close", "open"] if args.entry == "both" else [args.entry]
    rng = np.random.default_rng(20260927)

    S = pd.read_parquet(ROWS, columns=["symbol", "trade_date", "era", "core", "ind", "ret60", "ret252", "ind_heat_pct",
                                       "px_sma200", "y95", "s95", "fc95"], filters=flt).reset_index(drop=True)
    S["trade_date"] = pd.to_datetime(S["trade_date"])
    S = S.rename(columns={"core": "core_anatomy"})
    px = rp.load_panel(["open", "high", "low", "close", "avg_traded_value_20d", "price_adjustment_factor_to_present"],
                       filters=flt)
    dup = px.duplicated(["symbol", "trade_date"])
    if dup.any():
        raise SystemExit(f"panel has {int(dup.sum())} duplicate (symbol, trade_date) rows, e.g. "
                         f"{px.loc[dup, ['symbol', 'trade_date']].head(3).to_dict('records')}; fix the panel first")
    cal = rp.session_calendar(px)
    # core band exactly as the fixed sim_allin_matrix.load: 20d ADV (manifest: Rs) >= 5 cr and then-traded close > 50
    px["adv_cr"] = px["avg_traded_value_20d"] / 1e7
    px["core"] = (px["adv_cr"] >= 5) & (rp.raw_price(px, "close") > 50)
    S = S.merge(px[["symbol", "trade_date", "core", "adv_cr"]], on=["symbol", "trade_date"], how="left", validate="many_to_one")
    S["core"] = S["core"].eq(True)
    core_diff = dict(rows=int((S["core"] != S["core_anatomy"].eq(True)).sum()), n=len(S),
                     raw_core_only=int((S["core"] & ~S["core_anatomy"].eq(True)).sum()),
                     anatomy_core_only=int((~S["core"] & S["core_anatomy"].eq(True)).sum()))
    print(f"core (raw-price, panel ADV) vs rows.parquet core (adjusted close): {core_diff['rows']} of {len(S)} rows differ "
          f"(raw only {core_diff['raw_core_only']}, anatomy only {core_diff['anatomy_core_only']})")
    close_w, open_w = rp.wide(px, "close", cal), rp.wide(px, "open", cal)
    high_w, low_w = rp.wide(px, "high", cal), rp.wide(px, "low", cal)
    del px
    print(f"panel {rp.PANEL.name}: {len(cal)} sessions {cal[0].date()}..{cal[-1].date()}, {close_w.shape[1]} symbols · "
          f"anatomy rows {len(S)} · {time.time() - t0:.0f}s", flush=True)

    # orders -> effective session, PIT TTM ratio, material flag
    O, UO, oinfo = load_orders(syms, args.amounts, cal[-1])
    T, ttm_gaps = build_ttm(syms)
    O["eff"] = effective_session(O["ts"], cal)
    UO["eff"] = effective_session(UO["ts"], cal)
    O["rev_ttm_cr"] = asof_ttm(O, "ts", T)
    O["ratio"] = O["amount_cr"] / O["rev_ttm_cr"]
    O, UO = O.dropna(subset=["eff"]), UO.dropna(subset=["eff"])
    yr = O["eff"].dt.year
    print("orders by year (crawled filings / with amount / with PIT TTM / with ratio / material / amount but no TTM; "
          "order filings under the current CATS not crawled):")
    print(pd.DataFrame({"filings": O.groupby(yr).size(), "amount": O["amount_cr"].notna().groupby(yr).sum(),
                        "ttm": O["rev_ttm_cr"].notna().groupby(yr).sum(), "ratio": O["ratio"].notna().groupby(yr).sum(),
                        "material": (O["ratio"] >= MAT).groupby(yr).sum(),
                        "amount_no_ttm": (O["amount_cr"].notna() & O["rev_ttm_cr"].isna()).groupby(yr).sum(),
                        "uncrawled": UO.groupby(UO["eff"].dt.year).size()}).fillna(0).astype(int).T.to_string())

    def events(*frames: pd.DataFrame) -> pd.DataFrame:
        E = pd.concat([f[["symbol", "eff"]] for f in frames]).drop_duplicates(["symbol", "eff"]).reset_index(drop=True)
        return E.assign(eid=np.arange(len(E)))

    MO = O[O["ratio"] >= MAT].drop_duplicates(["symbol", "eff"]).reset_index(drop=True)
    MO["eid"] = np.arange(len(MO))
    AO = events(O, UO)                                              # any order filing under the current CATS
    UN = events(O[O["ratio"].isna()], UO)                           # order of unknown size (could be material)
    NT = events(O[O["amount_cr"].notna() & O["rev_ttm_cr"].isna()])  # amount known, no PIT TTM (pnl_quarterly gaps)
    top = MO.nlargest(8, "ratio")[["symbol", "eff", "amount_cr", "rev_ttm_cr", "ratio"]]
    print("largest material ratios (eyeball for boilerplate amounts):\n" + top.to_string(index=False))

    S["mat_order"], S["mat_key"] = window_flags(S, MO)
    S["any_order"], _ = window_flags(S, AO)
    S["order_unsized"], _ = window_flags(S, UN)
    S["order_no_ttm"], _ = window_flags(S, NT)
    S["asof"] = S["trade_date"] + pd.Timedelta(minutes=CUTOFF_MIN - 1)
    S["covered"] = asof_ttm(S, "asof", T).notna()
    cov_wk = S.groupby("trade_date")["covered"].mean()
    hit = cov_wk[cov_wk >= COV_MIN]
    if hit.empty:
        raise SystemExit(f"PIT TTM coverage never reaches {COV_MIN:.0%} of anatomy rows; no materiality window")
    cov_start = pd.Timestamp(hit.index[0])
    cov_yr = (cov_wk.groupby(cov_wk.index.year).mean() * 100).round(1)
    print(f"PIT TTM coverage of anatomy rows by year (%): {cov_yr.to_dict()} · COV_START {cov_start.date()}")

    label_end = None
    if args.labels == "panel":
        lab, label_end = panel_labels(S, close_w, high_w, low_w)
        S[["y95", "s95", "fc95"]] = lab
        print(f"panel labels: entries through {label_end.date()} (last date whose 95-session window is inside the panel); "
              f"labelled rows {int(lab['y95'].notna().sum())} of {len(S)}")
    reasons = oinfo["provisional_reasons"]
    # the COV_MIN rule on every measured week, not only the first: pnl_quarterly holes (e.g. the missing 2025-03-31
    # quarter / NaT filing dates at the detail_api -> xbrl switch) leave weeks where most rows cannot be sized
    meas_end = label_end if label_end is not None else S.loc[S["y95"].notna(), "trade_date"].max()
    inwin = cov_wk[(cov_wk.index >= cov_start) & (cov_wk.index <= meas_end)]
    low = inwin[inwin < COV_MIN]
    cov_check = dict(measured_weeks=len(inwin), weeks_below=len(low),
                     weeks_below_by_year=low.groupby(low.index.year).size().to_dict(),
                     first_below=str(low.index.min().date()) if len(low) else None,
                     last_below=str(low.index.max().date()) if len(low) else None,
                     era_mean_pct={e: round(float(inwin[m].mean() * 100), 1) for e, m in
                                   (("disc", inwin.index < ERA_SPLIT), ("conf", inwin.index >= ERA_SPLIT)) if m.any()})
    print(f"PIT TTM coverage inside the measured window {cov_start.date()}..{pd.Timestamp(meas_end).date()}: mean by era "
          f"{cov_check['era_mean_pct']} · weeks below {COV_MIN:.0%}: {len(low)} of {len(inwin)} "
          f"{cov_check['weeks_below_by_year']}" + (f" ({cov_check['first_below']}..{cov_check['last_below']})" if len(low) else ""))
    if len(low):
        reasons.append(f"PIT TTM coverage < {COV_MIN:.0%} on {len(low)} of {len(inwin)} measured weeks "
                       f"{cov_check['weeks_below_by_year']} ({cov_check['first_below']}..{cov_check['last_below']}): "
                       f"pnl_quarterly quarters / filing dates missing, orders there cannot be sized")
    res = {"meta": dict(run_at_ist=datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
                        amount_source=oinfo["amount_source"], orders=oinfo, cov_start=str(cov_start.date()),
                        cov_min=COV_MIN, labels=args.labels, label_end=str(label_end.date()) if label_end is not None else None,
                        entries=entries, panel=str(rp.PANEL), symbols_filter=syms, coverage_by_year_pct=cov_yr.to_dict(),
                        coverage_in_measured_window=cov_check, pnl_quarterly_gaps=ttm_gaps,
                        core_vs_anatomy_core=core_diff, n_orders=len(O), n_uncrawled_orders=len(UO),
                        n_material_orders=len(MO), era_split=str(ERA_SPLIT.date()), provisional_reasons=reasons)}
    res.update(part_a(S, cov_start, cov_wk, rng, args.boot, reasons))

    R = rp.stitch_renames(rp.gap_aware_returns(close_w), close_w)
    ne = rp.next_open_entry(open_w, high_w, low_w, close_w)
    op = open_w.where(open_w > 0)
    M = dict(cal=cal, dpos={d: i for i, d in enumerate(cal)}, wk=sorted(S["trade_date"].unique()),
             col={s: i for i, s in enumerate(close_w.columns)}, syms=np.asarray(close_w.columns),
             R=R.to_numpy(float), CL=close_w.to_numpy(float),
             TR=rp.stitch_renames(close_w.notna().astype(float), close_w).to_numpy(float) > 0.5,   # printed a close
             ON=rp.stitch_renames(op / close_w.ffill().shift(1) - 1, close_w).to_numpy(float),   # open / last close - 1
             EP=ne["entry_px"].to_numpy(float), LK=ne["locked"].to_numpy(bool))
    del R, ne, op
    M["wk"] = [pd.Timestamp(d) for d in M["wk"]]
    res.update(part_b(S, cov_start, M, entries, reasons))

    out = Path(args.out) if args.out else (None if syms else OUT)
    if out is not None:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
        print(f"wrote {out}")
    else:
        print("--symbols without --out: results not written")
    print(f"HOT ORDER COMBO COMPLETE · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
