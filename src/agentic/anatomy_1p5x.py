"""ANATOMY OF A 1.5x — what precedes a +50% move within ~95 sessions (EXP-2026-09-27-1p5x-anatomy).

Universe : ISIN-master equities with market cap >= Rs50cr (data/derived/mcap_pit.parquet), weekly samples
           (every 5th session of the global calendar). "Session" everywhere in this file = a panel date that is
           not a non-session date (true_sessions: NSE holidays on which the panel copies every row). mcap is point-in-time only where
           mcap_source == 'pnl_implied'; the rest is a screener backcast/backfill (present or last-trade
           shares x price, ignores later dilution). mcap_source is kept per row (rows.parquet) and
           Parts 1-2 are re-reported on pnl_implied rows as a sensitivity.
Targets  : y95 = max high over the next 95 SESSIONS (global calendar) >= 1.5 x entry close   (primary)
           y63 = same within 63 sessions;  s95 = close at +95 sessions >= 1.5 x (sustained).
           Built with research_panel.forward_window; windows that run past the panel end are unlabelled.
Features : ~60, every one known at the entry close (see FEATURES below). Price LEVELS (log_px, PE, the core
           filter close > 50) use the then-traded price (research_panel.raw_price).
Coverage : per-feature per-year coverage report on labelled rows. A feature enters the MODEL only if it is
           covered (>= --min-coverage of labelled rows) in every year of BOTH eras; the rest are still
           reported in Part 1. "No data" is NaN, never "no event".
Part 1   : univariate lift per feature (deciles / flags), both eras (disc 2016-2022, conf >= 2023). Lift is
           against the base rate of rows where that feature is known in that era.
Part 2   : LightGBM walk-forward: for test year Y, train on labelled rows whose 95-session label window ends
           before Y-01-01; predict Y (Y = 2019..last). AUC / precision@top-k / lift by year and era;
           importance = gain + out-of-sample permutation AUC drop.
Part 3   : ALL-IN portfolio (user spec 2026-09-27): 100% capital enters at once, equal weight, held 95 sessions,
           then fully rotated; 19 phase offsets. Arms: model top-10 / top-20 (out-of-sample predictions,
           tradable names ADV>=5cr & then-traded close>50), BASELINE rule, G+H rule, EW market.
           --entry close     : registered mode, enter at the signal-day close, flat 0.5% round trip.
           --entry next_open : live contract, enter at the t+1 open, upper-circuit-locked entries skipped,
                               exit at the open after the hold, research_panel.cost_rt (ADV-scaled).
           Returns are gap-aware (research_panel.gap_aware_returns + stitch_renames). CAGR per era.
Output   : logs/leader_sleeve/anatomy_1p5x/ — rows.parquet, rows_audit.parquet, feature_coverage_by_year.csv,
           univariate_lift.csv, univariate_lift_pnl_implied.csv (each with a .manifest.json), results.json,
           README.md; stdout -> log. --no-write skips all writes (smoke tests).

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json; line numbers refer to the pre-fix file)
  FIXED
  - L65-67, 210-212, 261-262 (high): labels counted panel ROWS, so gappy names had '95-session' windows over
    100-300+ sessions; fold cut Y-01-01 minus 150 days let stretched windows reach the test year. Now
    research_panel.forward_window on the global session calendar, labels NaN when the window runs past the
    panel end, and the last training row for year Y is the last session whose +95-session window ends
    before Y-01-01. Labels on windows with missing sessions are lower bounds (only observed highs); a
    window with NO observed session is unlabelled. The observed-session count is kept in
    rows_audit.parquet and the gappy share is reported by year.
  - L68, 76-77, 86, 123-124, 301 (high): the back-adjusted close was used as a LEVEL. core, log_px and pe
    now use research_panel.raw_price. pe = then-traded price / TTM EPS on the share basis in force at t
    (each quarter's EPS rescaled by share_adjustment_factor_to_present at its filing), which also fixes
    the TTM sum that mixed pre- and post-split EPS.
  - L108-118 (medium): fundamentals kept whichever basis was filed last, so TTM/YoY mixed con and sa
    quarters. Now fundamentals_pit(): TTM EPS, EPS/sales YoY and loss_to_profit are computed per basis on
    calendar quarters, every quarter in a window on the same basis (NaN when a quarter is missing), EPS on
    the present share basis; loss_to_profit is NaN when the year-ago quarter is unknown (was scored 0).
    For each quarter the value is con when the con window is known by t, else sa when the sa window is
    known by t, else NaN from the session the quarter is first reported (pit_latest: only data known by t
    picks the basis). days_since_results counts from the latest quarter's first filing on either basis.
    Review follow-up (same day): the first version of this fix chose ONE basis per symbol from its whole
    history (con if it EVER filed con) and dropped every other quarter. That dropped 5,247 of 43,199
    symbol-quarters, mostly in disc (2019: 1,966 symbol-quarters with a complete q..q-4 window vs 3,907 now;
    2020 3,299 vs 4,042; 2024 4,887 vs 5,388), so the fundamentals population was picked by a FUTURE
    event (e.g. AGI lost all 2016-2024 fundamentals because it first filed con in 2024-09). A truncation
    test (rebuild with only filings known by T, compare the state at T) shows 444 of 1,022 symbols had
    their 2019-05-15 basis set by later filings under that rule; the per-quarter rule has 0 differences at
    7 dates 2019-2025. rows_audit.parquet records the basis behind each value (basis_<metric>) and
    results.json the per-quarter-year basis mix.
    NOT fixed (source limit): pnl_quarterly has 296 / 404 reported symbol-quarters for 2016 / 2017 against
    ~4,000+ from 2018, so fundamentals are NaN for most 2016-2018 rows; the coverage rule keeps them out
    of the model (disc-year coverage 0%) and Part 1 lifts use only rows where they are known.
  - L117-120, 139, 144, 154-157, 162, 176, 42 (low): same-day after-close information. Results and
    announcements count at t only when stamped before 15:30 IST, else from the next session; order events
    use event_rows.entry (first session after the filing); promoter PIT uses the exchange broadcast `date`
    (not intimDt); shareholding uses the submission date (next session); delivery features are lagged one
    session (NSE publishes delivery after the close).
  - L297, 335 (low): pct_change + fillna(0) deleted moves across gaps. Part 3 uses gap_aware_returns +
    stitch_renames.
  - L79 (low): mkt_ew_ret60 is now the cross-sectional MEDIAN of ret60 (name kept for downstream readers).
  - L164-166 (low): prom_buys90 / prom_sells90 were dead (buyValue/sellValue are '0'). Now acqMode
    'Market Purchase' / 'Market Sale'.
  - L225 (low): prom_delta was binned as a flag via the 'prom_' prefix. Binary features are now an explicit
    list; prom_delta gets quantile buckets. prom_delta is now a calendar-quarter difference. Related (L228,
    not in the audit): deciles came from rank(method="first"), which split TIED values (prom_delta's zero
    mass, up20, ind_n, days_since_results) by row order = date, so tied buckets became era markers. Buckets
    are now value quantiles with ties kept together (named b<i>of<k> when fewer than 10 remain).
  - L64 (low): age_yrs was calendar time since 2015 for 60% of rows. Now NaN for names already in the panel
    at its start (listing age unknown), and age_yrs is never a model feature.
  - L304-315 (low): rule arms now match sim_leader_portfolio_7x: BASELINE heat/rank on the full weekly core
    band (no mcap filter); the G gate uses PRIOR-session breadth. (Core uses the then-traded price here, per
    the L68 fix, so it differs from 7x's adjusted-close core on the ~1.6% of rows with a later split.)
  - L156-157, 201 (low): last_order_to_rev_90d description said 'largest'; merge_asof takes the most recent.
    Description corrected; behaviour unchanged.
  - critic item 2: count features are NaN outside their source's coverage (pit_history broadcast dates start
    2019-01, block deals end 2026-04-29, announcements end 2026-08-30); per-feature per-year coverage report;
    model drops features not covered in both eras.
  - critic item 4: --entry next_open (t+1 open, locked entries skipped, ADV-scaled cost) next to the
    registered close entry.
  - Review round 2 (2 reviewers, medium; research_panel.py 53-57 session_calendar, anatomy L378, 428-430,
    437-448, 677-741 of the round-1 file): the calendar counted 75 NSE-holiday dates as sessions (2020: 10,
    2021: 13, 2022: 11, 2023: 15, 2024: 13, 2025: 2, 2026: 11; none 2015-2019). On them the panel copies every
    symbol's previous row (OHLC, volume, value, delivery; return_1d 0), e.g. INDHOTEL 2024-12-25 = 2024-12-24.
    Effects: (a) a '95-session' label, the Part 3 hold and the walk-forward cut spanned ~95 real sessions in
    2016-2019 but ~90-91 in 2020-2024 and ~89 in 2026, an era-dependent horizon; (b) 12 weekly sample dates
    were holidays, with a stale close while known_session mapped the previous evening's after-close filings
    onto them (their reaction then fell inside the label window: lookahead); (c) in --entry next_open, 13
    cohorts had a holiday as t+1, so they "entered" at the signal day's own open (pre-signal price, handing
    the arm day t's move) and exited old baskets at a stale open; (d) the forward-high window for t
    contained a copy of high(t). The row-based pre-audit file had the same defect. Now true_sessions()
    (same rule as sim_leader_portfolio_7x) drops those dates from the calendar and their rows from the panel
    frame before any feature is built, so row-rolled features (ret_k, hi/lo252, dvol, uc20, up20, delivery
    lag), labels, the weekly grid, known_session, the walk-forward cut, t+1 entries/exits and Part 3 returns
    all run on real sessions. The dropped dates and sessions per year are printed and saved in results.json.
  PARTIAL / NOT FIXED
  - L3, 69-71, 78-79, 83, 86 (medium, PARTIAL): universe membership and log_mcap still use screener-sourced
    mcap for rows without PIT shares (no PIT share count exists for them). mcap_source is recorded
    per row and Parts 1-2 are re-reported on pnl_implied rows (Part 2 evaluated, not retrained).
  - L89-94 (x2, low, PARTIAL): industry is one 2026 screener label applied back to 2016; no point-in-time
    industry source exists. --no-analyst-labels runs without the hand-made analog labels (the labels
    README asks for results with and without); ind_src is recorded per row.
  - L65-67 related artefacts (508 one-row jumps > 50%, e.g. KAUSHALYA 2024-03-04) and critic item 1
    (demergers/rights/special dividends unadjusted): panel data defects, repaired by the panel rebuild
    (build_price_only_ca_factors / repair_be_series_gaps), not in this file. Re-run after the rebuild.
  - critic item 3 (PARTIAL): delivery features are NaN on BE/BZ rows (imputed 1.0 there) and 20d delivery
    features are NaN when any of the last 20 rows was BE/BZ. Tape features (ret_k, hi252, dvol, uc20,
    up20, panel SMAs and ADV) still roll over rows as registered; rows == sessions once the BE/BZ backfill
    lands (the copied holiday rows are already dropped here, see review round 2).
  - Review round 2, NOT FIXED here: research_panel.session_calendar itself still returns the holiday dates
    (not this file; true_sessions is a local stop-gap), and the daily writer still emits them (2026-09-14).
    Panel-computed rolling columns used as features or filters (sma_50, sma_200, avg_traded_value_20d,
    volume_vs_20d/60d, traded_value_vs_60d, rsi_14_daily/weekly/monthly, avg_delivery_pct_20d,
    delivery_pct_vs_20d) were built by the panel writer over the copied rows, so from 2020 a 20-row window
    holds about one duplicate day (~12 a year of ~250). Recomputing them here would change the registered
    feature definitions; the fix belongs in the panel writer.
  - event_materiality_study's mixed-basis rev_ttm_cr (feeds last_order_to_rev_90d) is another file. The
    ev60_* categories come from its CATS regexes, so they change when that file changes.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
from event_materiality_study import CATS  # noqa: E402

H, H2 = 95, 63
PHASES = H // 5                                  # 19 weekly phase offsets, one per week of the hold
ERA_SPLIT_YEAR = 2023                            # disc 2016-2022, conf 2023+
COST_CLOSE = 0.005                               # registered flat round trip (close-entry mode)
DD = ROOT / "data/derived"
EVENT_ROWS = ROOT / "logs/leader_sleeve/event_rows_mcap50_20260924.parquet"
EXPERIMENT = "EXP-2026-09-27-1p5x-anatomy"
T0 = datetime.now()

PANEL_COLS = ["series", "open", "high", "low", "close", "return_1d", "volume_vs_20d", "volume_vs_60d",
              "traded_value_vs_60d", "delivery_pct", "avg_delivery_pct_20d", "delivery_pct_vs_20d",
              "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly", "sma_50", "sma_200", "avg_traded_value_20d",
              "price_adjustment_factor_to_present", "share_adjustment_factor_to_present"]
DELIV = ["delivery_pct", "avg_delivery_pct_20d", "delivery_pct_vs_20d"]

FEATURES = {
    # tape
    "ret5": "1-week return", "ret20": "1-month return", "ret60": "3-month return", "ret126": "6-month return", "ret252": "12-month return",
    "off_high": "distance below 52-week high (0 = at high)", "off_low": "distance above 52-week low",
    "dvol20": "20d daily volatility", "dvol60": "60d daily volatility",
    "volume_vs_20d": "today's volume / 20d avg", "volume_vs_60d": "today's volume / 60d avg", "traded_value_vs_60d": "today's traded value / 60d avg",
    "delivery_pct": "delivery % (previous session; NaN if that session was BE/BZ)",
    "avg_delivery_pct_20d": "delivery % 20d avg (previous session; NaN if any BE/BZ row in the window)",
    "delivery_pct_vs_20d": "delivery % vs its 20d avg (previous session; NaN if any BE/BZ row in the window)",
    "rsi_14_daily": "RSI daily", "rsi_14_weekly": "RSI weekly", "rsi_14_monthly": "RSI monthly",
    "px_sma50": "price vs 50DMA", "px_sma200": "price vs 200DMA", "sma50_200": "50DMA vs 200DMA",
    "uc20": "upper-circuit closes in last 20d", "up20": "up-days in last 20d",
    # size / liquidity
    "log_mcap": "log10 market cap (Rs cr; PIT only where mcap_source == pnl_implied)", "log_adv": "log10 20d avg traded value (Rs cr)",
    "log_px": "log10 then-traded (raw) price", "age_yrs": "years since first panel row, only for names first seen after the panel start (NaN = listed before 2015); never a model feature",
    # industry
    "ind_heat_pct": "industry 60d-return heat percentile", "ind_heat": "industry mean 60d return", "ind_ret20": "industry mean 20d return",
    "ind_breadth": "share of industry above 50DMA", "ind_n": "industry size (names)", "rank_in_ind_pct": "own 60d-return rank within industry (0 = leader)",
    # fundamentals
    "profitable": "TTM EPS > 0 (four quarters on one basis, con preferred, else sa)",
    "pe": "PE: then-traded price / TTM EPS on the share basis in force at t (NaN if TTM EPS <= 0)",
    "pe_ind": "PE / industry median PE",
    "eps_yoy": "latest reported quarter EPS YoY vs the same calendar quarter a year earlier, both on one basis (con preferred, else sa), present share basis",
    "sales_yoy": "latest reported quarter sales YoY vs the same calendar quarter a year earlier, both on one basis (con preferred, else sa)",
    "loss_to_profit": "latest reported quarter profit after a year-ago loss, one basis (NaN if the year-ago quarter is unknown on that basis)",
    "days_since_results": "days since the session the latest quarter's results first became known (either basis)",
    # filings / events
    **{f"ev60_{c}": f"'{c}' filings in last 60d (NaN outside announcement coverage)" for c in CATS},
    "filings30_all": "all filings last 30d (NaN outside announcement coverage)",
    "last_order_to_rev_90d": "most recent order/L1 filing within 90d: order value / TTM revenue (NaN = none)",
    # insiders / flows / ownership
    "prom_buys90": "promoter PIT 'Market Purchase' disclosures, 90d (NaN outside PIT coverage)",
    "prom_sells90": "promoter PIT 'Market Sale' disclosures, 90d (NaN outside PIT coverage)",
    "block_buys60": "block-deal buys 60d (NaN outside block-deal coverage)", "block_sells60": "block-deal sells 60d (NaN outside block-deal coverage)",
    "promoter_pct": "promoter holding % (latest submitted)", "prom_delta": "promoter holding change vs the previous calendar quarter (pp)",
    # market
    "mkt_breadth": "share of core-band stocks above 50DMA (same-day close)", "mkt_med_ret20": "median stock 20d return (mcap>=50cr)",
    "mkt_ew_ret60": "MEDIAN stock 60d return, mcap>=50cr (EW mean before the 2026-09-27 audit fix; name kept)",
}
COUNT_FEATS = [f"ev60_{c}" for c in CATS] + ["filings30_all", "prom_buys90", "prom_sells90", "block_buys60", "block_sells60"]
BINARY = set(COUNT_FEATS) | {"uc20", "loss_to_profit", "profitable"}
META = ["symbol", "trade_date", "era", "close", "mcap_cr", "adv", "ind", "core", "y95", "y63", "s95", "fh95", "fc95", "pred"]


REPEAT_SHARE = 0.90          # a date where >= 90% of rows repeat the symbol's prior OHLC is not a session
MIN_REPEAT_ROWS = 100        # ... and it must have at least this many rows with a prior row


def say(*a):
    print(f"[{(datetime.now() - T0).seconds:>4}s]", *a, flush=True)


def true_sessions() -> tuple[pd.DatetimeIndex, pd.Series]:
    """Global session calendar (FULL panel, every symbol, whatever --symbols says) without non-session dates.

    From 2020 the panel carries NSE-holiday dates (2020-12-25, 2021-01-26, 2024-08-15, 2026-09-14, ...) on which
    every row copies the symbol's previous row: open/high/low/close, volume, traded value and delivery, with
    return_1d = 0. research_panel.session_calendar() counts them as sessions (that primitive is wrong; central
    fix pending there and in the daily writer). A date is dropped when >= REPEAT_SHARE of its rows that have a
    previous row repeat that row's OHLC exactly and it has >= MIN_REPEAT_ROWS such rows. On the 2026-09-27 backup
    the 75 dropped dates sit at 100% and real sessions at <= 1.1%. Same rule as
    sim_leader_portfolio_7x.true_sessions (local copy: that file is being edited separately).
    Returns (calendar, repeat share of each dropped date)."""
    t = pq.read_table(rp.PANEL, columns=["symbol", "trade_date"],
                      read_dictionary=["symbol"]).unify_dictionaries()   # symbol as int codes: low memory
    s = t.column("symbol").combine_chunks().indices.to_numpy(zero_copy_only=False)
    d = pd.to_datetime(t.column("trade_date").to_numpy()).to_numpy()
    del t
    o = np.lexsort((d, s))
    s, d = s[o], d[o]
    prior = np.r_[False, s[1:] == s[:-1]]
    rep = prior.copy()
    del s
    for c in ("open", "high", "low", "close"):                   # one column at a time (memory)
        v = pq.read_table(rp.PANEL, columns=[c]).column(c).to_numpy().astype(float)[o]
        rep[1:] &= v[1:] == v[:-1]
        del v
    del o
    st = pd.DataFrame({"d": d[prior], "rep": rep[prior]}).groupby("d")["rep"].agg(["mean", "size"])
    fake = st.loc[(st["mean"] >= REPEAT_SHARE) & (st["size"] >= MIN_REPEAT_ROWS), "mean"].rename("repeat_share")
    return pd.DatetimeIndex(np.unique(d)).difference(fake.index), fake


def era_of(d: pd.Series) -> np.ndarray:
    return np.where(pd.to_datetime(d).dt.year >= ERA_SPLIT_YEAR, "conf", "disc")


def known_session(ts: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    """First session at whose close (15:30 IST) a timestamped item is public: the same day if stamped
    before 15:30 on a session, else the next session. Date-only values (00:00) count as after the close."""
    ts = pd.to_datetime(pd.Series(ts), errors="coerce")
    day = ts.dt.normalize()
    late = ((ts - day) >= pd.Timedelta(hours=15, minutes=30)) | (ts == day)
    key = (day + pd.to_timedelta(late.astype(int), unit="D")).to_numpy()
    pos = cal.values.searchsorted(key, side="left")
    out = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = (pos < len(cal)) & ts.notna().to_numpy()
    out[ok] = cal.values[pos[ok]]
    return pd.Series(out, index=ts.index)


def window_count(S: pd.DataFrame, ev: pd.DataFrame, days_back: int, span) -> np.ndarray:
    """Events per symbol with known date in (t - days_back, t] for every row of S (positional).
    NaN where the source does not cover the whole window (span = first/last known date of the source)."""
    ev = ev.dropna(subset=["d"]).sort_values(["symbol", "d"])
    arr = {s: v["d"].values.astype("datetime64[ns]") for s, v in ev.groupby("symbol")}
    out = np.zeros(len(S))
    t = S["trade_date"].values.astype("datetime64[ns]"); lo = t - np.timedelta64(days_back, "D")
    for s, idx in S.groupby("symbol").indices.items():
        a = arr.get(s)
        if a is not None:
            out[idx] = np.searchsorted(a, t[idx], side="right") - np.searchsorted(a, lo[idx], side="right")
    if span is not None and pd.notna(span[0]):
        s0, s1 = np.datetime64(pd.Timestamp(span[0]), "ns"), np.datetime64(pd.Timestamp(span[1]), "ns")
        out[(lo + np.timedelta64(1, "D") < s0) | (t > s1)] = np.nan
    return out


def span_mask(t: pd.Series, days_back: int, span) -> pd.Series:
    """True where (t - days_back, t] lies inside the source span."""
    if span is None or pd.isna(span[0]):
        return pd.Series(False, index=t.index)
    return (t - pd.Timedelta(days=days_back - 1) >= span[0]) & (t <= span[1])


FUND_TOL = pd.Timedelta(days=200)                 # a fundamentals state older than this is not carried forward
BASIS_PREF = {"con": 2, "sa": 1}                  # 0 = quarter reported but metric not computable on either basis


def pit_latest(c: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time state per symbol. Rows are candidates (symbol, known, key); a candidate becomes the state at
    its known session only if its key beats every candidate known on or before that session. key = quarter * 3
    + basis preference, so a newer quarter always wins, and within a quarter consolidated beats standalone
    beats 'reported but not computable'. A candidate that arrives later with a lower key (an older quarter,
    or the other basis of the same quarter) is dropped. Returns one row per (symbol, known)."""
    c = c.sort_values(["symbol", "known", "key"], kind="mergesort").reset_index(drop=True)
    prev = c.groupby("symbol")["key"].cummax().groupby(c["symbol"]).shift(1)
    return c[prev.isna() | (c["key"] > prev)].drop_duplicates(["symbol", "known"], keep="last")


def fundamentals_pit(q: pd.DataFrame, cal: pd.DatetimeIndex, sf_w: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Quarterly fundamentals as point-in-time state tables (one per metric) for merge_asof on trade_date.

    Each basis (con, sa) is computed on its own: TTM EPS needs quarters q..q-3 and the YoY metrics need q and
    q-4, all on the SAME basis and on calendar quarters (a missing quarter -> NaN). A metric is known at the
    latest known session of the quarters in its window. For quarter q the state is con when the con window
    is known by t, else sa when the sa window is known by t, else NaN from the session quarter q is first
    reported on either basis. The choice uses only what is known at t (pit_latest), so a later consolidated
    filing never changes what an earlier date sees. EPS is on the present share basis
    (eps_basic / share_adjustment_factor_to_present at the filing session).
    Returns ({metric: table[symbol, trade_date, metric, basis_<metric>], "res_dt": table[symbol, trade_date,
    res_dt]}, per-year audit of symbol-quarters with a computable metric by basis)."""
    q = q.dropna(subset=["filing_dt"]).reset_index(drop=True)
    q["sales_cr"] = q["net_sales"] * np.where(q["source"] == "xbrl", 1e-7, 1e-2)     # pnl_quarterly manifest units
    fpos = cal.searchsorted(q["filing_dt"].dt.normalize(), side="right") - 1
    fci = sf_w.columns.get_indexer(q["symbol"])
    okf = (fpos >= 0) & (fci >= 0)
    sfq = np.full(len(q), np.nan); sfq[okf] = sf_w.to_numpy()[fpos[okf], fci[okf]]
    q["eps_p"] = q["eps_basic"] / pd.Series(sfq).fillna(1.0).replace(0, 1.0).to_numpy()   # present share basis
    q["known"] = known_session(q["filing_dt"], cal)
    q = q.dropna(subset=["known"]).reset_index(drop=True)      # filed after the calendar end: not usable, not a lag either
    q["qn"] = q["quarter_end"].dt.year * 4 + q["quarter_end"].dt.quarter - 1
    keys = ["symbol", "basis", "qn"]
    base = q[keys + ["eps_p", "sales_cr", "known"]]
    lag = {n: q[keys].merge(base.assign(qn=base["qn"] + n), on=keys, how="left") for n in (1, 2, 3, 4)}
    ttm = q["eps_p"] + lag[1]["eps_p"] + lag[2]["eps_p"] + lag[3]["eps_p"]
    k_ttm = pd.concat([q["known"]] + [lag[n]["known"] for n in (1, 2, 3)], axis=1).max(axis=1, skipna=False)
    y4 = lag[4]["eps_p"]
    k_yoy = pd.concat([q["known"], lag[4]["known"]], axis=1).max(axis=1, skipna=False)
    metrics = {"eps_ttm_p": (ttm, k_ttm),
               "eps_yoy": ((q["eps_p"] - y4) / y4.abs().clip(lower=0.1), k_yoy),
               "sales_yoy": (q["sales_cr"] / lag[4]["sales_cr"] - 1, k_yoy),
               "loss_to_profit": (((q["eps_p"] > 0) & (y4 <= 0)).astype(float).where(q["eps_p"].notna() & y4.notna()), k_yoy)}
    first = q.groupby(["symbol", "qn"], as_index=False)["known"].min()          # quarter first reported (either basis)
    pref = q["basis"].map(BASIS_PREF).fillna(0).astype(int)
    out = {}
    audit = {"reported": first.groupby(first["qn"] // 4).size()}
    for m, (v, k) in metrics.items():
        ok = v.notna() & k.notna()
        c = pd.concat([pd.DataFrame({"symbol": q.loc[ok, "symbol"], "known": k[ok], "key": q.loc[ok, "qn"] * 3 + pref[ok],
                                     m: v[ok], f"basis_{m}": q.loc[ok, "basis"]}),
                       first.assign(key=first["qn"] * 3, **{m: np.nan, f"basis_{m}": None}).drop(columns="qn")],
                      ignore_index=True)
        out[m] = pit_latest(c).drop(columns="key").rename(columns={"known": "trade_date"}).sort_values("trade_date")
        best = (pd.DataFrame({"symbol": q.loc[ok, "symbol"], "qn": q.loc[ok, "qn"], "p": pref[ok]})
                .sort_values("p").drop_duplicates(["symbol", "qn"], keep="last"))
        for b, p in BASIS_PREF.items():                                  # final (hindsight) basis per quarter, audit only
            audit[f"{m}:{b}"] = best[best["p"] == p].groupby(best.loc[best["p"] == p, "qn"] // 4).size()
    r = first.assign(key=first["qn"]); r["res_dt"] = r["known"]
    out["res_dt"] = pit_latest(r).drop(columns=["key", "qn"]).rename(columns={"known": "trade_date"}).sort_values("trade_date")
    audit = pd.DataFrame(audit).fillna(0).astype(int).rename_axis("quarter_year")
    return out, audit


def lift_table(L: pd.DataFrame, feats: list[str], mask: pd.Series | None = None) -> pd.DataFrame:
    """P(y95) by feature bucket and era. Buckets are cut on all of L; stats on L[mask] when given.
    Lift = bucket rate / base rate of rows where the feature is known, in that era."""
    uni = []
    for f in feats:
        x = L[f]
        if x.notna().mean() < 0.05:
            continue
        if f in BINARY or x.dropna().nunique() <= 3:
            b = pd.Series(np.where(x.isna(), "n/a", np.where(x > 0, "yes/>0", "no/0")), index=L.index)
        else:                                              # value quantiles: ties share a bucket (rank-first split
            xx = x.dropna()                                # tied values by row order, i.e. by date)
            cats = pd.qcut(xx, 10, duplicates="drop")
            k = len(cats.cat.categories)
            names = [f"d{i}" for i in range(1, 11)] if k == 10 else [f"b{i}of{k}" for i in range(1, k + 1)]
            b = pd.Series("n/a", index=L.index, dtype=object)
            b.loc[xx.index] = cats.cat.rename_categories(names).astype(str)
        sub_all = L if mask is None else L[mask]
        bb = b.loc[sub_all.index]
        known = sub_all[f].notna()
        base = {e: sub_all.loc[known & (sub_all["era"] == e), "y95"].mean() * 100 for e in ("disc", "conf")}
        covp = {e: known[sub_all["era"] == e].mean() * 100 if (sub_all["era"] == e).any() else np.nan for e in ("disc", "conf")}
        for bk, sub in sub_all.groupby(bb):
            if bk == "n/a":
                continue
            r = {"feature": f, "bucket": bk, "lo": float(sub[f].min()), "hi": float(sub[f].max())}
            for e in ("disc", "conf"):
                E = sub[sub["era"] == e]
                r[f"{e}_n"] = len(E); r[f"{e}_p"] = E["y95"].mean() * 100 if len(E) else np.nan
                r[f"{e}_base"] = base[e]; r[f"{e}_cov"] = covp[e]
                r[f"{e}_lift"] = r[f"{e}_p"] / base[e] if len(E) and base[e] > 0 else np.nan
            uni.append(r)
    U = pd.DataFrame(uni)
    if len(U):
        U["min_lift"] = U[["disc_lift", "conf_lift"]].min(axis=1)
        U["both_n_ok"] = (U["disc_n"] >= 300) & (U["conf_n"] >= 300)
    return U


def write_manifest(path: Path, **kw) -> None:
    kw.setdefault("experiment", EXPERIMENT); kw.setdefault("producer", "src/agentic/anatomy_1p5x.py")
    kw.setdefault("updated", datetime.now().isoformat(timespec="seconds"))
    Path(str(path) + ".manifest.json").write_text(json.dumps(kw, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", default=None, help="override research_panel.PANEL (smoke tests: the .bak copy)")
    ap.add_argument("--symbols", default=None, help="comma-separated symbol subset (smoke tests)")
    ap.add_argument("--entry", choices=["close", "next_open", "both"], default="both",
                    help="Part 3 entry: registered signal-day close, live next open, or both")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="model feature needs this share of labelled rows covered in EVERY year of both eras")
    ap.add_argument("--model-features", choices=["covered", "all"], default="covered",
                    help="'all' = legacy: every feature except age_yrs, coverage ignored")
    ap.add_argument("--no-analyst-labels", action="store_true", help="industry from screener only (no analyst analog labels)")
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--no-write", action="store_true", help="compute and print only; write nothing")
    ap.add_argument("--outdir", default=str(ROOT / "logs/leader_sleeve/anatomy_1p5x"))
    ap.add_argument("--pnl", default=None, help="P&L table to read instead of data/derived/pnl_quarterly.parquet "
                    "(registered A/B tests only, e.g. the old-format backfill in pnl_quarterly_enriched.parquet)")
    args = ap.parse_args()
    if args.panel:
        rp.PANEL = Path(args.panel)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    sfilt = [("symbol", "in", syms)] if syms else None
    say(f"panel {rp.PANEL.name}" + (f" · {len(syms)} symbols (subset)" if syms else ""))

    # ============ panel (row-based tape features, as registered) ============
    # global calendar (whole panel) minus non-session dates; their copied rows go before any feature is built,
    # so row-rolled features, labels, the weekly grid, known sessions, t+1 entries and Part 3 all use real sessions
    cal, fake = true_sessions()
    have = set(pq.read_schema(rp.PANEL).names)
    px = rp.load_panel([c for c in PANEL_COLS if c in have], filters=sfilt)
    if "share_adjustment_factor_to_present" not in px:
        px["share_adjustment_factor_to_present"] = 1.0 / pd.to_numeric(px["price_adjustment_factor_to_present"], errors="coerce").replace(0, np.nan).fillna(1.0)
    if "series" not in px:
        px["series"] = "EQ"
    dup = px.duplicated(["symbol", "trade_date"], keep=False)
    if dup.any():
        say(f"WARNING {int(dup.sum())} duplicate symbol-date rows; keeping the EQ-series row")
        px = (px.assign(_eq=px["series"].eq("EQ")).sort_values(["symbol", "trade_date", "_eq"])
                .drop_duplicates(["symbol", "trade_date"], keep="last").drop(columns="_eq").reset_index(drop=True))
    on_fake = px["trade_date"].isin(fake.index)
    px = px[~on_fake].reset_index(drop=True)
    say(f"non-session dates dropped: {len(fake)} (" + " · ".join(f"{y} {n}" for y, n in fake.index.year.value_counts().sort_index().items())
        + f") · {int(on_fake.sum()):,} copied panel rows removed")
    spy = pd.Series(1, index=cal).groupby(cal.year).size()
    print("sessions per year after the drop: " + " · ".join(f"{y} {n}" for y, n in spy.items()))
    g = px.groupby("symbol", sort=False)
    for k in (5, 20, 60, 126, 252):
        px[f"ret{k}"] = g["close"].pct_change(k, fill_method=None)
    px["hi252"] = g["high"].transform(lambda s: s.rolling(252, min_periods=60).max())
    px["lo252"] = g["low"].transform(lambda s: s.rolling(252, min_periods=60).min())
    px["off_high"] = px["close"] / px["hi252"] - 1
    px["off_low"] = px["close"] / px["lo252"] - 1
    px["dvol20"] = g["return_1d"].transform(lambda s: s.rolling(20, min_periods=15).std())
    px["dvol60"] = g["return_1d"].transform(lambda s: s.rolling(60, min_periods=40).std())
    uc = ((px["return_1d"] >= 0.0495) & (px["close"] >= px["high"] * 0.999)).astype(float)
    px["uc20"] = uc.groupby(px["symbol"]).transform(lambda s: s.rolling(20, min_periods=1).sum())
    px["up20"] = (px["return_1d"] > 0).astype(float).groupby(px["symbol"]).transform(lambda s: s.rolling(20, min_periods=1).sum())
    px["px_sma50"] = px["close"] / px["sma_50"] - 1
    px["px_sma200"] = px["close"] / px["sma_200"] - 1
    px["sma50_200"] = px["sma_50"] / px["sma_200"] - 1
    px["adv"] = px["avg_traded_value_20d"] / 1e7
    px["close_raw"] = rp.raw_price(px, "close")                   # then-traded price for LEVELS
    px["core"] = (px["adv"] >= 5) & (px["close_raw"] > 50)
    first = g["trade_date"].transform("min")
    px["age_yrs"] = ((px["trade_date"] - first).dt.days / 365.25).where(first > cal[0])
    # delivery: NaN on BE/BZ rows (imputed 1.0 there), then lag one session (published after the close)
    t2t = px["series"].isin(["BE", "BZ"])
    t2t20 = t2t.astype(float).groupby(px["symbol"]).transform(lambda s: s.rolling(20, min_periods=1).max()) > 0
    px["delivery_pct"] = px["delivery_pct"].where(~t2t)
    for c in ("avg_delivery_pct_20d", "delivery_pct_vs_20d"):
        px[c] = px[c].where(~t2t20)
    for c in DELIV:
        px[c] = px.groupby("symbol", sort=False)[c].shift(1)
    mc = pd.read_parquet(DD / "mcap_pit.parquet", columns=["symbol", "trade_date", "mcap_cr", "mcap_source"], filters=sfilt)
    mc["trade_date"] = pd.to_datetime(mc["trade_date"])
    px = px.merge(mc, on=["symbol", "trade_date"], how="left")
    say(f"panel {len(px):,} rows · {px['symbol'].nunique()} symbols · {len(cal)} sessions")

    # market regime (daily); the G gate uses PRIOR-session breadth (registered 7x definition)
    core_d = px[px["core"] & px["sma_50"].notna()]
    breadth = (core_d["close"] > core_d["sma_50"]).groupby(core_d["trade_date"]).mean()
    big = px[px["mcap_cr"] >= 50]
    mkt = pd.DataFrame({"mkt_breadth": breadth,
                        "mkt_med_ret20": big.groupby("trade_date")["ret20"].median(),
                        "mkt_ew_ret60": big.groupby("trade_date")["ret60"].median()})
    breadth_prev = breadth.reindex(cal).shift(1)
    del core_d, big

    # wide frames on the global calendar (labels, Part 3, EPS share basis)
    close_w = rp.wide(px, "close", cal); high_w = rp.wide(px, "high", cal); low_w = rp.wide(px, "low", cal)
    open_w = rp.wide(px, "open", cal); adv_w = rp.wide(px, "adv", cal)
    sf_w = rp.wide(px, "share_adjustment_factor_to_present", cal).ffill()

    # ============ weekly sample ============
    wkdays = cal[::5]
    on_wk = px["trade_date"].isin(wkdays) & (px["trade_date"] >= "2016-01-01")
    S = px[on_wk & (px["mcap_cr"] >= 50)].copy().reset_index(drop=True)
    Wc = px.loc[on_wk & px["core"], ["symbol", "trade_date", "ret60", "ret252", "core"]].copy()   # full core band (BASELINE)
    del px
    S = S.join(mkt, on="trade_date")
    S["log_mcap"] = np.log10(S["mcap_cr"]); S["log_adv"] = np.log10(S["adv"].clip(lower=1e-3))
    S["log_px"] = np.log10(S["close_raw"].clip(lower=0.01))

    # labels: next H sessions on the global calendar
    ri = cal.get_indexer(S["trade_date"]); ci = close_w.columns.get_indexer(S["symbol"])
    c0 = S["close"].to_numpy()
    for h, tag in ((H, "95"), (H2, "63")):
        fw = rp.forward_window(close_w, high_w, low_w, h)
        ex = fw["exit"].to_numpy()[ri, ci]; hi = fw["hi"].to_numpy()[ri, ci]
        S[f"fh{tag}"] = np.where(np.isfinite(ex), hi / c0, np.nan)
        if tag == "95":
            S["fc95"] = ex / c0
        obs = high_w.notna().astype(float)[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)
        S[f"label_obs{tag}"] = obs.to_numpy()[ri, ci]            # observed sessions in the window (audit only)
        del fw, obs
    say(f"weekly sample {len(S):,} rows · labels on the session calendar")

    # industry (nse4 map)
    sc = pd.read_parquet(DD / "screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry"])
    imap = sc.set_index("symbol")["industry"].map(html.unescape)
    isrc = pd.Series("screener", index=imap.index)
    if not args.no_analyst_labels:
        al = pd.read_csv(DD / "industry_analyst_labels.csv")
        extra = al[~al["symbol"].isin(imap.index)].set_index("symbol")["analog_symbol"].map(imap).dropna()
        imap = pd.concat([imap, extra]); isrc = pd.concat([isrc, pd.Series("analyst_inference", index=extra.index)])
    S["ind"] = S["symbol"].map(imap); S["ind_src"] = S["symbol"].map(isrc)
    Wc["ind"] = Wc["symbol"].map(imap)
    gi = S.groupby(["trade_date", "ind"])
    S["ind_n"] = gi["ret60"].transform("count")
    S["ind_heat"] = gi["ret60"].transform("mean")
    S["ind_ret20"] = gi["ret20"].transform("mean")
    S["ind_breadth"] = gi["px_sma50"].transform(lambda s: (s > 0).mean())
    S.loc[S["ind_n"] < 5, ["ind_heat", "ind_ret20", "ind_breadth"]] = np.nan
    hp = S.dropna(subset=["ind_heat"]).drop_duplicates(["trade_date", "ind"])[["trade_date", "ind", "ind_heat"]]
    hp["ind_heat_pct"] = hp.groupby("trade_date")["ind_heat"].rank(pct=True)
    S = S.merge(hp[["trade_date", "ind", "ind_heat_pct"]], on=["trade_date", "ind"], how="left")
    S["rank_in_ind_pct"] = S.groupby(["trade_date", "ind"])["ret60"].rank(pct=True, ascending=False)

    # ============ fundamentals (PIT, per basis, con preferred per quarter, calendar quarters) ============
    q = pd.read_parquet(Path(args.pnl) if args.pnl else DD / "pnl_quarterly.parquet", filters=sfilt)
    fund, fund_audit = fundamentals_pit(q, cal, sf_w)
    print("fundamentals: symbol-quarters with a computable metric, by quarter year and the basis used "
          "(con preferred; windows on one basis only):\n" + fund_audit.to_string())
    S = S.sort_values("trade_date")
    for m, tab in fund.items():
        S = pd.merge_asof(S, tab, on="trade_date", by="symbol", direction="backward", tolerance=FUND_TOL)
    S = S.reset_index(drop=True)
    S["days_since_results"] = (S["trade_date"] - S["res_dt"]).dt.days
    S["profitable"] = (S["eps_ttm_p"] > 0).astype(float).where(S["eps_ttm_p"].notna())
    eps_then = S["eps_ttm_p"] * S["share_adjustment_factor_to_present"]          # TTM EPS on the share basis at t
    S["pe"] = np.where(eps_then > 0, S["close_raw"] / eps_then, np.nan)
    S["pe_ind"] = S["pe"] / S.groupby(["trade_date", "ind"])["pe"].transform("median")
    S[["eps_yoy", "sales_yoy"]] = S[["eps_yoy", "sales_yoy"]].clip(-5, 20)
    del q, fund, sf_w
    say("fundamentals merged")

    # ============ events / insiders / block deals / promoter ============
    src_rows = {}
    ah = pd.read_parquet(DD / "announcements_historical.parquet", columns=["symbol", "desc", "attchmntText", "sort_date"], filters=sfilt)
    ah["txt"] = ah["desc"].fillna("") + " " + ah["attchmntText"].fillna("")
    ah["d"] = known_session(ah["sort_date"], cal)
    span_ann = (ah["d"].min(), ah["d"].max())
    src_rows["announcements"] = ah["d"].dt.year.value_counts().sort_index().to_dict()
    ah["cat"] = None
    for c, p in CATS.items():
        m = ah["cat"].isna() & ah["txt"].str.contains(p, case=False, regex=True)
        ah.loc[m, "cat"] = c
    for c in CATS:
        S[f"ev60_{c}"] = window_count(S, ah.loc[ah["cat"] == c, ["symbol", "d"]], 60, span_ann)
    S["filings30_all"] = window_count(S, ah[["symbol", "d"]], 30, span_ann)
    del ah
    er_all = pd.read_parquet(EVENT_ROWS, columns=["symbol", "entry", "cat", "order_to_rev"], filters=sfilt)
    er = er_all[er_all["cat"].isin(["order", "tender_L1"]) & er_all["order_to_rev"].notna()].dropna(subset=["entry"])
    er = er.rename(columns={"entry": "trade_date"})[["symbol", "trade_date", "order_to_rev"]]
    er["trade_date"] = pd.to_datetime(er["trade_date"]); er = er.sort_values("trade_date")
    span_ord = (er["trade_date"].min(), pd.to_datetime(er_all["entry"]).max())
    S = pd.merge_asof(S.sort_values("trade_date"), er, on="trade_date", by="symbol", direction="backward",
                      tolerance=pd.Timedelta(days=90)).rename(columns={"order_to_rev": "last_order_to_rev_90d"}).reset_index(drop=True)
    del er_all, er

    pit = pd.read_parquet(DD / "pit_history.parquet", columns=["symbol", "personCategory", "acqMode", "date"], filters=sfilt)
    prom = pit[pit["personCategory"].fillna("").str.contains("Promoter", case=False)].copy()
    ts = pd.to_datetime(prom["date"], format="%d-%b-%Y %H:%M", errors="coerce")
    ts = ts.fillna(pd.to_datetime(prom["date"], dayfirst=True, errors="coerce"))
    prom["d"] = known_session(ts, cal)                       # exchange broadcast time, not intimDt
    span_pit = (prom["d"].min(), prom["d"].max())
    src_rows["pit_promoter"] = prom["d"].dt.year.value_counts().sort_index().to_dict()
    mode = prom["acqMode"].fillna("").str.strip()
    S["prom_buys90"] = window_count(S, prom.loc[mode == "Market Purchase", ["symbol", "d"]], 90, span_pit)
    S["prom_sells90"] = window_count(S, prom.loc[mode == "Market Sale", ["symbol", "d"]], 90, span_pit)
    del pit, prom
    blk = pd.read_parquet(DD / "block_deals_history.parquet", columns=["BD_SYMBOL", "BD_DT_DATE", "BD_BUY_SELL"])
    blk = blk.rename(columns={"BD_SYMBOL": "symbol"})
    blk["d"] = pd.to_datetime(blk["BD_DT_DATE"], format="mixed", errors="coerce").dt.normalize()   # block window is intraday
    span_blk = (blk["d"].min(), blk["d"].max())
    src_rows["block_deals"] = blk["d"].dt.year.value_counts().sort_index().to_dict()
    if syms:
        blk = blk[blk["symbol"].isin(syms)]
    side = blk["BD_BUY_SELL"].astype(str).str.upper()
    S["block_buys60"] = window_count(S, blk.loc[side.str.startswith("B"), ["symbol", "d"]], 60, span_blk)
    S["block_sells60"] = window_count(S, blk.loc[side.str.startswith("S"), ["symbol", "d"]], 60, span_blk)
    shp = pd.read_parquet(DD / "stock_shareholding.parquet", columns=["symbol", "quarter_end", "promoter_pct", "submission"], filters=sfilt)
    shp["qe"] = pd.to_datetime(shp["quarter_end"], format="%d-%b-%Y", errors="coerce")
    shp["sub"] = pd.to_datetime(shp["submission"], format="%d-%b-%Y", errors="coerce")
    shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
    shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe", "sub"]).drop_duplicates(["symbol", "qe"], keep="last")
    shp["qp"] = shp["qe"].dt.to_period("Q")
    prev = shp[["symbol", "qp", "promoter_pct"]].assign(qp=lambda x: x["qp"] + 1).rename(columns={"promoter_pct": "pp_prev"})
    shp = shp.merge(prev, on=["symbol", "qp"], how="left")
    shp["prom_delta"] = shp["promoter_pct"] - shp["pp_prev"]
    shp["trade_date"] = known_session(shp["sub"].fillna(shp["qe"] + pd.Timedelta(days=21)), cal)
    src_rows["shareholding"] = shp["qe"].dt.year.value_counts().sort_index().to_dict()
    S = pd.merge_asof(S.sort_values("trade_date"), shp.dropna(subset=["trade_date"])[["symbol", "trade_date", "promoter_pct", "prom_delta"]].sort_values("trade_date"),
                      on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200)).reset_index(drop=True)
    del shp, blk
    say("events/insiders/blocks/promoter merged")

    # ============ labels, eras, coverage ============
    FEATS = [f for f in FEATURES if f in S.columns]
    S["y95"] = (S["fh95"] >= 1.5).astype(float).where(S["fh95"].notna())
    S["y63"] = (S["fh63"] >= 1.5).astype(float).where(S["fh63"].notna())
    S["s95"] = (S["fc95"] >= 1.5).astype(float).where(S["fc95"].notna())
    S["era"] = era_of(S["trade_date"])
    lab = S["y95"].notna()
    L = S[lab]
    base = L.groupby("era")[["y95", "y63", "s95"]].mean() * 100
    say(f"labelled rows {len(L):,} · features {len(FEATS)}")
    print("\nBASE RATES (%):\n" + base.round(2).to_string())
    gap_share = (L["label_obs95"] < H).groupby(L["trade_date"].dt.year).mean() * 100
    print("labelled rows whose 95-session window has missing sessions (labels = lower bound), % by year:\n  "
          + " · ".join(f"{y} {v:.1f}" for y, v in gap_share.items()))

    cov = pd.DataFrame({f: S[f].notna() for f in FEATS})
    cov["pe"] = cov["pe_ind"] = S["eps_ttm_p"].notna()                       # pe is NaN by design for loss-makers
    cov["last_order_to_rev_90d"] = span_mask(S["trade_date"], 90, span_ord)   # NaN by design when no order
    yr = S["trade_date"].dt.year
    cov_y = (cov[lab].groupby(yr[lab]).mean() * 100).T
    cnts = [f for f in COUNT_FEATS if f in S.columns]
    nz_y = ((S.loc[lab, cnts] > 0).groupby(yr[lab]).mean() * 100).T
    disc_y = [y for y in cov_y.columns if y < ERA_SPLIT_YEAR]; conf_y = [y for y in cov_y.columns if y >= ERA_SPLIT_YEAR]
    min_disc = cov_y[disc_y].min(axis=1) if disc_y else pd.Series(np.nan, index=cov_y.index)
    min_conf = cov_y[conf_y].min(axis=1) if conf_y else pd.Series(np.nan, index=cov_y.index)
    thr = args.min_coverage * 100
    eligible = (min_disc >= thr) & (min_conf >= thr)
    if args.model_features == "all":
        MF = [f for f in FEATS if f != "age_yrs"]
    else:
        MF = [f for f in FEATS if f != "age_yrs" and bool(eligible.get(f, False))]
    dropped = {f: ("never a model feature (time index)" if f == "age_yrs" else
                   f"coverage: min disc-year {min_disc[f]:.0f}%, min conf-year {min_conf[f]:.0f}% (< {thr:.0f}%)")
               for f in FEATS if f not in MF}
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30)
    print(f"\n=== FEATURE COVERAGE: % of labelled rows where the feature is known, by year (model needs >= {thr:.0f}% every year, both eras) ===")
    print(cov_y.round(0).astype("Int64").assign(model=["yes" if f in MF else "NO" for f in cov_y.index]).to_string())
    print("\ncount features: % of labelled rows with a non-zero count, by year")
    print(nz_y.round(1).to_string())
    print("\nsource rows per year (known date):")
    for k, v in src_rows.items():
        print(f"  {k:<14} " + " · ".join(f"{int(y)} {n:,}" for y, n in v.items() if pd.notna(y)))
    print(f"\nMODEL FEATURES ({len(MF)}; --model-features {args.model_features}). Kept out of the model:")
    for f, why in dropped.items():
        print(f"  {f:<24} {why}")

    # ============ Part 1: univariate lift ============
    U = lift_table(L, FEATS)
    pnl_mask = L["mcap_source"].eq("pnl_implied")
    Up = lift_table(L, FEATS, mask=pnl_mask)
    print(f"\nmcap_source of labelled rows: " + " · ".join(f"{k} {v*100:.1f}%" for k, v in L["mcap_source"].value_counts(normalize=True, dropna=False).items()))
    top = pd.DataFrame()
    if len(U):
        print("\n=== PART 1: buckets with lift >= 1.5x in BOTH eras (n >= 300 each), strongest first ===")
        top = U[U["both_n_ok"] & (U["min_lift"] >= 1.5)].sort_values("min_lift", ascending=False)
        for _, r in top.head(45).iterrows():
            print(f"{r['feature']:<24} {r['bucket']:<7} [{r['lo']:>9.3g} .. {r['hi']:>9.3g}]  disc {r['disc_p']:>5.1f}% ({r['disc_lift']:.2f}x, n{r['disc_n']:,})  "
                  f"conf {r['conf_p']:>5.1f}% ({r['conf_lift']:.2f}x, n{r['conf_n']:,})  — {FEATURES[r['feature']]}")
        print("\n=== buckets that SUPPRESS 1.5x (<= 0.5x in both eras) ===")
        for _, r in U[U["both_n_ok"] & (U[["disc_lift", "conf_lift"]].max(axis=1) <= 0.5)].sort_values("min_lift").head(20).iterrows():
            print(f"{r['feature']:<24} {r['bucket']:<7} [{r['lo']:>9.3g} .. {r['hi']:>9.3g}]  disc {r['disc_lift']:.2f}x  conf {r['conf_lift']:.2f}x  — {FEATURES[r['feature']]}")
        print("\n=== ERA-FLIPPERS (>=1.5x in one era, <=1.0x in the other) — not trustworthy ===")
        for _, r in U[U["both_n_ok"] & (((U["disc_lift"] >= 1.5) & (U["conf_lift"] <= 1)) | ((U["conf_lift"] >= 1.5) & (U["disc_lift"] <= 1)))].head(15).iterrows():
            print(f"{r['feature']:<24} {r['bucket']:<7} disc {r['disc_lift']:.2f}x  conf {r['conf_lift']:.2f}x")
        print("\n=== SENSITIVITY: the both-era buckets above, on pnl_implied (PIT) mcap rows only ===")
        if len(Up) and len(top):
            j = top.merge(Up, on=["feature", "bucket"], suffixes=("", "_pit"))
            for _, r in j.head(45).iterrows():
                print(f"{r['feature']:<24} {r['bucket']:<7} all: disc {r['disc_lift']:.2f}x conf {r['conf_lift']:.2f}x  |  "
                      f"PIT-mcap rows: disc {r['disc_lift_pit']:.2f}x (n{r['disc_n_pit']:,}) conf {r['conf_lift_pit']:.2f}x (n{r['conf_n_pit']:,})")

    # ============ Part 2: walk-forward LightGBM ============
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    S["pred"] = np.nan
    years = sorted(S["trade_date"].dt.year.unique())
    imp_gain = pd.Series(0.0, index=MF); perm = []; wf = []
    print(f"\n=== PART 2: walk-forward LightGBM on {len(MF)} features (train: label window ends before Y-01-01; predict Y) ===")
    for Y in [y for y in years if y >= args.first_test_year]:
        iY = int(cal.searchsorted(pd.Timestamp(f"{Y}-01-01")))
        if iY - H - 1 < 0:
            continue
        last_train = cal[iY - H - 1]                           # its window t+1..t+95 ends on the last session before Y
        tr = L[L["trade_date"] <= last_train]; te_mask = S["trade_date"].dt.year == Y
        if len(tr) < 1000 or tr["y95"].nunique() < 2 or not te_mask.any():
            print(f"  {Y}: skipped (train {len(tr):,} rows)")
            continue
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=63, min_child_samples=300, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        m.fit(tr[MF], tr["y95"])
        S.loc[te_mask, "pred"] = m.predict_proba(S.loc[te_mask, MF])[:, 1]
        imp_gain += pd.Series(m.booster_.feature_importance("gain"), index=MF)
        te = S[te_mask & S["y95"].notna()]
        if len(te) and te["y95"].nunique() == 2:
            auc = roc_auc_score(te["y95"], te["pred"]); b0 = te["y95"].mean() * 100
            k1 = te.nlargest(max(1, len(te) // 100), "pred")["y95"].mean() * 100
            k5 = te.nlargest(max(1, len(te) // 20), "pred")["y95"].mean() * 100
            wf.append(dict(year=int(Y), train_rows=len(tr), train_last=str(last_train.date()), test_rows=len(te), base=b0, auc=auc, top1=k1, top5=k5))
            print(f"  {Y}: train {len(tr):,} (<= {last_train.date()}) · test {len(te):,} · base {b0:4.1f}% · AUC {auc:.3f} · "
                  f"top1% {k1:4.1f}% ({k1/b0:.1f}x) · top5% {k5:4.1f}% ({k5/b0:.1f}x)", flush=True)
            if Y >= ERA_SPLIT_YEAR:                           # out-of-sample permutation importance
                smp = te.sample(min(len(te), 60000), random_state=Y)
                a0 = roc_auc_score(smp["y95"], m.predict_proba(smp[MF])[:, 1])
                rng = np.random.default_rng(Y)
                for f in MF:
                    X = smp[MF].copy(); X[f] = rng.permutation(X[f].values)
                    perm.append((Y, f, a0 - roc_auc_score(smp["y95"], m.predict_proba(X)[:, 1])))
    O = S[S["pred"].notna() & S["y95"].notna()]
    part2_era = {}
    for label, sub in (("all", O), ("pnl_implied", O[O["mcap_source"].eq("pnl_implied")])):
        for e in ("disc", "conf"):
            E = sub[sub["era"] == e]
            if len(E) >= 100 and E["y95"].nunique() == 2:
                k1 = E.nlargest(max(1, len(E) // 100), "pred")["y95"].mean() * 100; b0 = E["y95"].mean() * 100
                auc = roc_auc_score(E["y95"], E["pred"])
                part2_era[f"{label}_{e}"] = dict(rows=len(E), auc=auc, base=b0, top1=k1, top1_lift=k1 / b0)
                print(f"  ERA {e} [{label} rows]: AUC {auc:.3f} · base {b0:.1f}% · top1% {k1:.1f}% ({k1/b0:.1f}x) · n {len(E):,}")
    P = (pd.DataFrame(perm, columns=["year", "feature", "auc_drop"]).groupby("feature")["auc_drop"].mean().sort_values(ascending=False)
         if perm else pd.Series(dtype=float))
    G = (imp_gain / imp_gain.sum()).sort_values(ascending=False) if imp_gain.sum() > 0 else imp_gain
    print("\nTOP FEATURES — out-of-sample permutation AUC drop (2023+) and share of model gain:")
    for f in P.index[:25]:
        print(f"  {f:<24} AUC drop {P[f]:+.4f} · gain {G.get(f, 0)*100:4.1f}% — {FEATURES[f]}")

    # ============ Part 3: ALL-IN portfolio, hold 95, 19 phases ============
    Rv = rp.stitch_renames(rp.gap_aware_returns(close_w), close_w).to_numpy()
    cff = close_w.ffill().to_numpy()
    ent = rp.next_open_entry(open_w, high_w, low_w, close_w)
    epx = ent["entry_px"].to_numpy(); lck = ent["locked"].to_numpy(dtype=bool)
    opn = open_w.to_numpy(); advv = adv_w.to_numpy()
    col = {s: i for i, s in enumerate(close_w.columns)}
    dpos = {d: i for i, d in enumerate(cal)}; n = len(cal)
    wk = [d for d in wkdays if d >= pd.Timestamp("2016-01-01")]
    del high_w, low_w, open_w, adv_w, ent

    def rule_picks(U_: pd.DataFrame, gate: bool) -> dict:
        Wd = U_.dropna(subset=["ind", "ret60"]).copy()
        Wd["n"] = Wd.groupby(["trade_date", "ind"])["ret60"].transform("size"); Wd = Wd[Wd["n"] >= 5]
        h = Wd.groupby(["trade_date", "ind"])["ret60"].mean().rename("h").reset_index()
        h["hot"] = h.groupby("trade_date")["h"].rank(pct=True) >= 0.9
        Wd = Wd.merge(h[["trade_date", "ind", "hot"]], on=["trade_date", "ind"])
        Wd["rk"] = Wd.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
        Pk = Wd[Wd["hot"] & (Wd["rk"] <= 3) & (Wd["ret252"] > 0.5) & Wd["core"]]
        if gate:
            Pk = Pk[Pk["trade_date"].map(breadth_prev) >= 0.5]          # prior-session breadth (registered)
        return Pk.groupby("trade_date")["symbol"].apply(list).to_dict()

    T = S[S["core"]]                                                     # tradable names only
    def model_picks(k: int) -> dict:
        M = T.dropna(subset=["pred"])
        return M.sort_values("pred", ascending=False).groupby("trade_date").head(k).groupby("trade_date")["symbol"].apply(list).to_dict()

    def allin(picks: dict, start: pd.Timestamp, mode: str) -> tuple[list, dict]:
        wk_s = [d for d in wk if d >= start]
        res, st = [], dict(periods=0, invested=0, names=0, skipped_locked=0, skipped_no_open=0)
        for ph in range(PHASES):
            d0 = wk_s[ph::PHASES]
            if not d0:
                continue
            v = 1.0; dates = [d0[0]]; vals = [1.0]; prev = None
            for d in d0:
                i0 = dpos[d]; end = min(i0 + H, n - 1)
                if end <= i0:
                    break
                if prev is not None:                                     # next_open: exit the held basket at this open
                    pc, pw = prev
                    o, c = opn[i0 + 1, pc], cff[i0, pc]
                    f = np.where(np.isfinite(o) & np.isfinite(c) & (c > 0), o / np.where(c > 0, c, 1.0), 1.0)
                    v *= float((pw * f).sum() / pw.sum()); prev = None
                ci = np.array([col[s] for s in picks.get(d, []) if s in col], dtype=int)
                if mode == "next_open" and len(ci):
                    locked = lck[i0, ci]; no_open = ~np.isfinite(epx[i0, ci]) | ~(epx[i0, ci] > 0)
                    st["skipped_locked"] += int(locked.sum()); st["skipped_no_open"] += int((no_open & ~locked).sum())
                    ci = ci[~locked & ~no_open]
                st["periods"] += 1
                if len(ci):
                    seg = np.nan_to_num(Rv[i0 + 1:end + 1][:, ci], nan=0.0)
                    if mode == "next_open":
                        seg[0] = cff[i0 + 1, ci] / epx[i0, ci] - 1              # bought at the t+1 open
                        cost = np.asarray(rp.cost_rt(pd.Series(advv[i0, ci])), dtype=float)
                    else:
                        cost = np.full(len(ci), COST_CLOSE)
                    grow = np.cumprod(1 + seg, axis=0) * (1 - cost)
                    path = grow.mean(axis=1)
                    if mode == "next_open" and end < n - 1:
                        prev = (ci, grow[-1])
                    st["invested"] += 1; st["names"] += len(ci)
                else:
                    path = np.ones(end - i0)
                vals.extend(list(v * path)); dates.extend(list(cal[i0 + 1:end + 1])); v = float(v * path[-1])
            nav = pd.Series(vals, index=pd.DatetimeIndex(dates))
            nav = nav[~nav.index.duplicated(keep="last")]
            if len(nav) > 2:
                res.append(rp.nav_metrics(nav))
        return res, st

    bench = T.groupby("trade_date")["symbol"].apply(list).to_dict()
    arms = {"MODEL top-10": model_picks(10), "MODEL top-20": model_picks(20), "BASELINE rule": rule_picks(Wc, False),
            "G+H rule": rule_picks(S, True), "EW liquid market": bench}
    modes = ["close", "next_open"] if args.entry == "both" else [args.entry]
    summary = {}
    start3 = pd.Timestamp(f"{args.first_test_year}-01-01")
    for mode in modes:
        lbl = "signal-day CLOSE, 0.5% round trip (registered)" if mode == "close" else "NEXT OPEN, locked entries skipped, ADV-scaled cost"
        print(f"\n=== PART 3 [{mode}]: ALL-IN, 100% capital, hold 95 sessions, full rotation, {PHASES} phase offsets ({start3.date()} .. end); {lbl} ===")
        summary[mode] = {}
        for name, pk in arms.items():
            res, st = allin(pk, start3, mode)
            if not res:
                print(f"  {name:<18} no NAV"); continue
            r = pd.DataFrame([{k: v for k, v in x.items() if k != "years"} for x in res])
            summary[mode][name] = dict(cagr_med=r["cagr"].median(), cagr_min=r["cagr"].min(), cagr_max=r["cagr"].max(),
                                       dd_med=r["maxdd"].median(), dd_worst=r["maxdd"].min(), sharpe_med=r["sharpe"].median(),
                                       cagr_disc_med=r["cagr_disc"].median(), cagr_conf_med=r["cagr_conf"].median(),
                                       avg_names=st["names"] / max(st["invested"], 1), pct_invested=st["invested"] / max(st["periods"], 1) * 100,
                                       skipped_locked=st["skipped_locked"], skipped_no_open=st["skipped_no_open"])
            s_ = summary[mode][name]
            print(f"  {name:<18} CAGR median {s_['cagr_med']:>+6.1f}% (worst phase {s_['cagr_min']:+.1f}, best {s_['cagr_max']:+.1f}) · "
                  f"disc {s_['cagr_disc_med']:+.1f}% · conf {s_['cagr_conf_med']:+.1f}% · maxDD median {s_['dd_med']:.1f}% (worst {s_['dd_worst']:.1f}%) · "
                  f"Sharpe {s_['sharpe_med']:.2f}" + (f" · locked skipped {st['skipped_locked']}" if mode == "next_open" else ""), flush=True)

    # ============ save ============
    if args.no_write:
        say("--no-write: nothing written"); say("ANATOMY COMPLETE"); return
    outd = Path(args.outdir); outd.mkdir(parents=True, exist_ok=True)
    keep = META + ["mcap_source", "ind_src"] + FEATS
    S[keep].to_parquet(outd / "rows.parquet", index=False)
    write_manifest(outd / "rows.parquet", dataset="1.5x anatomy rows", key=["symbol", "trade_date"], rows=len(S),
                   columns={**{k: v for k, v in FEATURES.items() if k in FEATS},
                            "y95": "1 if max high over the next 95 sessions (global calendar) >= 1.5x entry close; NaN if the window passes the panel end",
                            "y63": "same within 63 sessions", "s95": "1 if close at +95 sessions >= 1.5x",
                            "fh95": "max high next 95 sessions / close", "fc95": "close at +95 sessions / close (frozen at the last close if delisted)",
                            "pred": "walk-forward LightGBM out-of-sample P(y95) (first test year onward)",
                            "core": "ADV>=5cr & then-traded close>50 (tradable)", "close": "back-adjusted close (returns only; use rows_audit.close_raw for levels)",
                            "mcap_source": "pnl_implied = PIT shares; screener_backcast / screener_backfill = present shares x price (NOT point-in-time)",
                            "ind_src": "industry label source: screener | analyst_inference"},
                   features=MF, not_model_features=dropped, era="disc 2016-2022, conf 2023+",
                   units="returns/ratios are fractions; mcap_cr, adv in Rs crore; delivery_pct as in the panel",
                   caveats=["mcap is PIT only where rows_audit.mcap_source == pnl_implied",
                            "industry is one 2026 screener label applied to all years (not PIT)",
                            "labels on windows with missing sessions are lower bounds (rows_audit.label_obs95 < 95)",
                            "sessions = panel dates minus non-session dates (NSE holidays whose rows copy the previous session; "
                            "results.json non_session_dates_dropped); panel-computed rolling columns (sma_50/200, avg_traded_value_20d, "
                            "volume_vs_*, traded_value_vs_60d, rsi_14_*, avg_delivery_pct_20d, delivery_pct_vs_20d) still include those copied rows",
                            "'features' = the columns the anatomy model used (coverage rule); other feature columns are Part 1 only"],
                   args=vars(args))
    bcols = [f"basis_{m}" for m in ("eps_ttm_p", "eps_yoy", "sales_yoy", "loss_to_profit")]
    aud = S[["symbol", "trade_date", "mcap_source", "ind_src", "close_raw", "share_adjustment_factor_to_present", "eps_ttm_p"]
            + bcols + ["label_obs95", "label_obs63"]]
    aud.to_parquet(outd / "rows_audit.parquet", index=False)
    write_manifest(outd / "rows_audit.parquet", dataset="1.5x anatomy row provenance (not features)", key=["symbol", "trade_date"], rows=len(aud),
                   columns={"mcap_source": "mcap_pit source: pnl_implied (PIT shares) | screener_backcast / screener_backfill (present shares, not PIT)",
                            "ind_src": "screener | analyst_inference (industry_analyst_labels.csv analog)",
                            "close_raw": "then-traded close = adjusted close / price_adjustment_factor_to_present (Rs)",
                            "share_adjustment_factor_to_present": "panel share factor at t (EPS share-basis conversion)",
                            "eps_ttm_p": "TTM EPS on the PRESENT share basis (Rs/share); pe uses eps_ttm_p x share factor at t",
                            **{c: f"pnl_quarterly basis behind {c[6:]} at t: con | sa | null (latest reported quarter not computable "
                                  "on one basis, or no state within 200 days)" for c in bcols},
                            "label_obs95": "FORWARD-LOOKING audit count: sessions with a high in t+1..t+95 (never use as a feature)",
                            "label_obs63": "same over t+1..t+63"})
    cov_long = cov_y.stack().rename("covered_pct").to_frame().join(nz_y.stack().rename("nonzero_pct"), how="left").reset_index()
    cov_long.columns = ["feature", "year", "covered_pct", "nonzero_pct"]
    cov_long["model_feature"] = cov_long["feature"].isin(MF)
    cov_long.to_csv(outd / "feature_coverage_by_year.csv", index=False)
    write_manifest(outd / "feature_coverage_by_year.csv", dataset="feature coverage by year (labelled rows)", key=["feature", "year"],
                   columns={"covered_pct": "% of labelled rows where the feature is known (pe/pe_ind: TTM EPS known; last_order_to_rev_90d: order source covers the window)",
                            "nonzero_pct": "count features only: % of labelled rows with a non-zero count",
                            "model_feature": f"feature used by the model (covered >= {thr:.0f}% in every year of both eras)"},
                   units="percent 0-100")
    for nm, tab in (("univariate_lift.csv", U), ("univariate_lift_pnl_implied.csv", Up)):
        tab.to_csv(outd / nm, index=False)
        write_manifest(outd / nm, dataset="P(y95) by feature bucket and era" + (" — pnl_implied mcap rows only" if "pnl" in nm else ""),
                       key=["feature", "bucket"],
                       columns={"lo/hi": "feature range in the bucket (feature units, see rows manifest)", "<era>_n": "rows",
                                "<era>_p": "P(y95) % in the bucket", "<era>_base": "P(y95) % over rows where the feature is known",
                                "<era>_cov": "% of era rows where the feature is known", "<era>_lift": "<era>_p / <era>_base",
                                "min_lift": "min over eras", "both_n_ok": "n >= 300 in both eras"},
                       units="percent 0-100 except lift (ratio)")
    json.dump(dict(base=base.round(3).to_dict(), label_gap_share_by_year={int(k): round(float(v), 2) for k, v in gap_share.items()},
                   model_features=MF, not_model_features=dropped, walk_forward=wf, part2_era=part2_era,
                   non_session_dates_dropped=[str(x.date()) for x in fake.index], sessions_per_year={int(y): int(n) for y, n in spy.items()},
                   fundamentals_by_quarter_year={int(y): {k: int(v) for k, v in r.items()} for y, r in fund_audit.iterrows()},
                   perm_auc_drop=P.round(5).to_dict(), gain_share=G.round(5).to_dict(), allin=summary,
                   source_rows_by_year={k: {int(y): int(c) for y, c in v.items()} for k, v in src_rows.items()}, args=vars(args)),
              open(outd / "results.json", "w"), indent=1, default=float)
    (outd / "README.md").write_text(
        f"# 1.5x anatomy ({EXPERIMENT})\n\n"
        "What precedes a +50% move within 95 sessions, for every stock-week with market cap >= Rs50cr since 2016.\n\n"
        "- `rows.parquet` — one row per stock-week: features known at the entry close, targets, out-of-sample model score (see manifest; "
        "`model_features` lists what the model used).\n"
        "- `rows_audit.parquet` — per-row provenance: mcap_source (PIT or not), industry label source, then-traded close, "
        "fundamentals basis (con/sa) behind each fundamentals value, observed sessions in the label window "
        "(forward-looking, audit only).\n"
        "- `feature_coverage_by_year.csv` — % of labelled rows where each feature is known, by year; count features also get the non-zero rate.\n"
        "- `univariate_lift.csv` / `univariate_lift_pnl_implied.csv` — P(+50% within 95 sessions) by feature bucket, per era, "
        "lift vs the base of rows where the feature is known; second file = PIT-mcap rows only.\n"
        "- `results.json` — base rates, walk-forward by year, era metrics (all rows and PIT-mcap rows), permutation importance, "
        "gain share, all-in portfolio per entry mode with per-era CAGR.\n\n"
        "Eras: disc = 2016-2022, conf = 2023+. A finding is only trusted if it holds in both.\n\n"
        "Sessions are the panel's dates minus the NSE-holiday dates on which the panel copies every symbol's previous row "
        "(75 dates 2020-2026 on the 2026-09-27 backup; list in results.json), so 95 sessions means 95 real sessions in every era.\n\n"
        "Limitations: mcap is PIT only for pnl_implied rows; industry is a 2026 label applied to all years; "
        "labels on windows with missing sessions are lower bounds; panel-computed rolling columns (SMAs, ADV, volume ratios, RSI, "
        "20d delivery) still include the copied holiday rows; see the module docstring for the 2026-09-27 audit fixes.\n")
    say(f"written to {outd}")
    say("ANATOMY COMPLETE")


if __name__ == "__main__":
    main()
