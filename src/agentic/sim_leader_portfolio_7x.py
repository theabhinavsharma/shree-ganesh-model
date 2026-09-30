"""LEADER SLEEVE — 10-year PORTFOLIO backtest, full factorial of 3 levers (EXP-2026-09-27-leader-7x-factorial).

BASELINE = production cell: nse4 industry map + analyst-analog labels (screen_theme_leaders' INDUSTRY_MAP,
called "nse4_full" in sim_leader_cell_v2), heat + rank on the core band (ADV>=5cr & then-traded close>50),
MEAN heat top decile (>=5 names), top-3 by own ret60, EXTENDED (ret252 > 50%).
Levers (every non-empty combination = 7 experiments):
  G  market gate  — enter a weekly cohort only if prior-session breadth (share of core-band stocks
                    above their SMA50, from the panel) >= 0.50, else that slot holds cash
  B  basket       — top-10 instead of top-3 per hot industry
  H  broad heat   — heat + rank on the mcap_pit.parquet mcap_cr >= Rs50cr universe; BUY only core-band names.
                    NOT point-in-time before 2018-07: mcap_pit has no pnl_implied rows before 2018-04, so the
                    earlier universe uses present-day share counts x adjusted close, and 23% of rows are NULL
                    (excluded). mcap_pit also predates the 2026-09-27 BE/BZ panel repair, so every pre-2025
                    BE/BZ row the repair added is NULL mcap and H can never buy it. See NOT FIXED, DATA GAPS and
                    the MCAP-SOURCE SENSITIVITY section.
Sessions: the panel's dates minus non-session dates (NSE holidays on which the panel repeats every symbol's
prior OHLC; see true_sessions). Holds, the weekly grid, t+1 and every return use this calendar.
Portfolio (registered): SLOTS = HOLD//5 + 1 overlapping weekly slots (26 at the default 126-session hold,
13 at 63, 17 at 84), 1/SLOTS each, equal-weight buy-and-hold per cohort, entry at close, exit at the close
of session t+HOLD (a name with no row that session exits at the close of its first row after it; see
--gap-exit), 0.5% round trip charged at entry, delisted names frozen at last close, empty cohort = cash.
A slot's next cohort enters 5*SLOTS sessions later, always strictly after its previous cohort exits
(asserted). Daily mark-to-market NAV. Benchmark: equal-weight core band, daily rebalanced, membership fixed
at the PRIOR close.
Sensitivity (not the registered result): same NAV with single-day stock returns clipped to +-40%.

Options (defaults = registered design):
  --entry next_open  live contract (score_leader_sleeve.py): buy at the OPEN of the session after the cohort
                     date; names upper-circuit locked at that open, or with no row that session, are skipped
                     and their 1/n share stays cash for the hold; ADV-scaled round trip (research_panel.cost_rt,
                     = 0.5% for every core-band pick); exit at the close of session t+HOLD.
  --rebalance N      reset every slot to NAV/SLOTS every N sessions (0 = never = registered). Moved capital
                     (sum|dw|/2) pays the 0.5% round trip.
  --labels nse4      screener labels only (no analyst analogs).
  --gap-exit stale   book a pick that has no row on its exit session at its stale carried close and drop the
                     rest of the gap's move (pre-review behaviour). Default next_row: the position stays held
                     until the name's first row after the scheduled exit, is sold at that close, and the cash
                     rejoins the slot at its next cohort.
Always reported unless --no-sensitivity: slot-phase spread (weekly grid shifted by 0..4 sessions, same slot
count, same options), the with/without analyst-label factorial, and the mcap-source sensitivity (H arms with
the heat universe restricted to pnl_implied mcap rows; every compared arm rerun fresh from the first cohort
dated >= 2018-07-01, where that source exists, and measured from one slot cycle later). DATA GAPS (always printed) lists the dropped non-session dates,
the core-band rows with NULL mcap by period and series, and example exits that fell in a gap.
Every result is reported per era: disc = first cohort date .. 2022-12-30 close, conf = 2022-12-30 close ..
end. Each era is based at the last session before it starts, so disc x conf = full-period NAV multiple and
CAGR, maxDD and Sharpe share that base.
Output: stdout (tee to logs/leader_sleeve/portfolio_7x_v2<suffix>_<date>.log) + portfolio_7x_nav_v2<suffix>.parquet
(+ .manifest.json). The pre-audit portfolio_7x_nav{,_h63,_h84}.parquet files are left untouched.
Smoke tests: --panel <copy> --symbols A,B,... (a symbol subset forces --dry-run: nothing is written).

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json, findings for this file; line numbers
refer to the pre-fix file, commit a8ad8d9)
FIXED
  [high]   48-50, 65, 107 (also 159) — a missing session inside a hold (pre-2025 BE/BZ sessions absent from the
           panel) was zeroed by pct_change(fill_method=None).fillna(0), deleting the move across the gap (TANLA
           Dec-2020 -30%). Returns now come from research_panel.gap_aware_returns on the session calendar
           (last close carried, full move on the first session back, frozen only after the final row) and
           stitch_renames (a renamed pick continues as its successor instead of freezing). Same returns feed
           the benchmark. This covers gaps strictly inside a hold. A gap that spans the exit session is
           handled by the review round 3 fix below (exit at the first row after it).
  [high]   61 (also 70-71, 159) — core filter `close > 50` used the back-adjusted close (future splits/bonuses
           pushed earlier history under Rs50). Core now uses research_panel.raw_price (then-traded close) > 50;
           this feeds the picks, the G breadth universe and the benchmark. Returns stay on adjusted prices.
  [medium] 35 (with 98-112) — SLOTS = round(HOLD/5) gave 25 at the default hold (registered design: 26) and a
           next cohort one session before the previous exit (one day compounded twice). SLOTS = HOLD//5 + 1
           (26/13/17 for 126/63/84) and run() asserts next-entry > exit for every hold.
  [medium] 159-160 — benchmark counted a day-t return only if the stock was core on t-1 AND t (dropped names
           on the day they fell). Membership is now the prior-close core flag, carried through gaps within a
           listing life; the benchmark NAV starts at 1.0 on the first cohort date like the strategy NAVs.
  [medium] 94-113 — slots never rebalanced, so results are one path-dependent draw. Added --rebalance N and a
           slot-phase spread (weekly grid offsets 0..4, min/mean/max per era, margins vs BASELINE at the same
           offset, number of offsets passing the registered rule) plus slot end-value dispersion. The
           registered non-rebalanced ladder at offset 0 stays the headline; margins inside the spread are not
           effects.
  [low]    130 (also 125-128) — calendar-year returns dropped each year's first session, and the era split
           dropped the first 2023 session. Calendar years now come from research_panel.nav_metrics (prior
           year-end to year-end). Era CAGR / maxDD / Sharpe come from nav_metrics applied to segment(), which
           bases each era at the last session before it (conf = 2022-12-30 close), so no session is dropped or
           counted twice. research_panel.nav_metrics' own cagr_conf is NOT used, and the choice is explicit:
           its conf segment starts at the first session >= 2022-12-25 (the 2022-12-26 close), so
           2022-12-27..30 count in both eras. Reviewers measured +1.5pp (1,254-symbol subset) and +2.2pp
           (600-symbol subset) on BASELINE conf CAGR. main() prints the gap whenever the primitive disagrees.
           Central fix proposed for research_panel.py:163:
           conf seg = nav[nav.index >= nav.index[nav.index < cut][-1]].
  [low]    42-46, 57 — analyst-analog labels had no with/without report (industry_analyst_labels.README.md
           requires one). Added --labels and an always-on label-sensitivity factorial; docstring now says the
           baseline map includes the analogs.
  critic 4 — execution realism: --entry next_open (above). Close entry stays the registered default.
FIXED (review round 3, 2026-09-27; line numbers refer to the round-2 file)
  [medium] 161, 181-184, 262-266 — non-session dates counted as sessions. From 2020 the panel carries NSE-holiday
           dates on which every row repeats the symbol's prior OHLC: 75 in the 2026-09-27 backup (2020..2026:
           10/13/11/15/13/2/11; e.g. 2020-12-25, 2021-01-26, 2024-12-25, 2026-09-14), none before 2020, and the
           repaired panel still has them. research_panel.session_calendar counts them (primitive defect). With
           --entry next_open, a cohort followed by one had "t+1 open" = the signal day's own open, so the model
           bought before the signal and skipped the real overnight gap. In close mode a 126-session hold covered
           ~5% fewer real sessions in 2020+ than in 2016-19, and the weekly grid stepped over fake dates.
           true_sessions() drops every date on which >= 90% of the rows that have a prior row repeat it exactly
           (the fake dates are at 100%; real sessions are at most 1.1%). Their rows are removed before
           ret60/ret252, the wide frames, breadth and the benchmark. Central fixes still needed in
           research_panel.session_calendar and in the daily writer (2026 has 11 such dates).
  [medium] 242, 272-273 — exit inside a gap. A pick with no row on its exit session (a pre-2025 BE/BZ gap in the
           pre-repair panel, or a suspension) was booked at its stale carried close, and the rest of the gap's
           move was dropped. Examples: MAJESCO exits dated 2021-02-04..04-01 missed +69.5% (next row 2021-04-27);
           HOVS exits dated 2017-07-31..08-22 missed -8.1% (next row 2017-08-23). The default (--gap-exit
           next_row) now holds such a position past the scheduled exit, marks it on its own gap-aware returns,
           and sells it at the close of its first row after the exit. The position leaves the slot's liquid
           capital, so the slot's next cohort invests only the rest, and the proceeds rejoin at the slot's
           first cohort on or after that row. A name with no later row is delisted and stays frozen at its last
           close (registered). The diagnostics count "exit in gap" and "exit after final row", and DATA GAPS
           prints the largest cases. --gap-exit stale reproduces the old booking.
  [medium] 356-405 — the MCAP-SOURCE window comparison reused the headline NAVs. Their un-rebalanced slots
           entered the window with weights set by 2016-18 (non-PIT) picks (slot end max/min 800-2,400x), while
           the [pnl] arms entered near equal. A reviewer measured that the conclusion flipped (H vs BASELINE
           disc-pit +2.7 -> -7.2pp). Every compared arm (BASELINE, and H / G+H / B+H / G+B+H both all-source
           and [pnl]) is now a fresh run of the registered ladder on the weekly cohorts dated >= 2018-07-01
           (all slots start equal and hold only post-2018-07 picks), measured over the round-2 window: from
           one slot cycle after the fresh start (every slot has held a cohort) to the 2022-12-30 close, then conf.
NOT FIXED
  [medium] review round 3: 167-176 (mcap merge), 77-79, 572-580 — mcap_pit.parquet was built on 2026-09-24 from
           the pre-repair panel, so it has no rows on the pre-2025 BE/BZ sessions that the 2026-09-27 repair
           added. The exact (symbol, trade_date) merge gives every such row NULL mcap. Those rows are outside the
           H heat universe and H arms can never buy them. BASELINE can, and 2025+ BE rows are sized, so inside
           the H-vs-BASELINE comparison the exclusion differs by era and concentrates in high-momentum
           surveillance names (e.g. TANLA BE 2020-12-11..18). Fix upstream: rebuild mcap_pit after the BE/BZ
           backfill. DATA GAPS prints core-band rows with NULL mcap by period and series (EQ vs BE/BZ), and
           the BASELINE picks that H could not have bought.
  [high]   48-50 (data half) — the pre-repair panel (the 2026-09-27 backup) lacks pre-2025 BE/BZ sessions, so a
           run on it has stale marks inside those gaps. The move across a gap is no longer lost, and an exit
           inside one now waits for the next row. The repaired panel fills these sessions. Suspensions stay
           real gaps with the same handling.
  [low]    42-46, 57 (PIT half) — screener labels are a 2026-09 crawl applied back to 2016; no point-in-time
           industry source exists. Known limitation, unquantified.
  critic 4 (exit half) — exits locked at the lower circuit are still filled at the close of t+HOLD. Counted
           in the diagnostics ("exit LC-locked"). TODO: per-name exit queue that carries a locked position
           past the slot's next entry.
  critic 1 — unadjusted demergers / rights / reverse splits are a data-layer defect (price-only CA factors
           are being rebuilt). With gap-aware returns an unadjusted action inside a gap now lands as one large
           move instead of being dropped; the +-40% clip line bounds it.
  critic 3 — ret60/ret252, avg_traded_value_20d and sma_50 roll over a symbol's rows, not sessions. These are
           the registered signal definitions and panel features; rows ~= sessions once BE/BZ is backfilled.
           ret60/ret252 are computed here after the non-session rows are dropped. The panel's own
           avg_traded_value_20d and sma_50 still include those duplicated rows (about 1 row in 20 or 50 near
           each such date). That is an upstream fix.
  critic 5 — live screen universe (screen_theme_leaders.py) differs from every arm here; out of scope for
           this file.
  build_mcap_pit.py findings that reach this file through mcap_cr (the H lever's heat universe). The fixes
  belong upstream in build_mcap_pit.py plus a rebuild of mcap_pit.parquet, which this task may not write.
  This file now reads mcap_source, no longer calls the universe PIT, and reports the MCAP-SOURCE SENSITIVITY:
  the heat-universe source mix per period, and the H / G+H / B+H / G+B+H arms rerun with the universe
  restricted to pnl_implied rows. Every compared arm (BASELINE, all-source and [pnl]) is a fresh run from the
  first cohort dated >= 2018-07-01, measured from one slot cycle later as disc-pit (to the 2022-12-30 close)
  and conf.
  [high]   build_mcap_pit.py 1-13, 56-63, 65-73, 77-92 — not point-in-time before 2018-04 (0 pnl_implied
           rows; present-day shares x adjusted close) and less PIT in disc than in conf. This file ignored
           mcap_source, so the H-arm both-era verdicts partly compare construction methods. Upstream fix.
  [low]    build_mcap_pit.py 49-54 (used via this file 58-62, H lever) — 4-quarter median share count lags
           splits/bonuses (mcap understated for months). About 46% of disc-era broad rows use screener
           backcast/backfill. Upstream fix.
  [high]   build_mcap_pit.py 40, 46-53 (split lag in pnl_implied), [high] 65-73 (screener_backfill sized
           with a renamed successor's shares), [medium] 56-73 (316 large live names NULL before mid-2018, so
           they are missing from the broad heat universe), [low] 42-43, 57-62, 43/48-52 (basis dedupe,
           backcast fetch-date basis, filing-order median). All upstream. The pnl_implied-only sensitivity
           still carries the split lag and cannot test the pre-2018-07 stretch at all.
"""
from __future__ import annotations

import argparse
import html
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

START = "2016-06-01"
COST = 0.005                      # registered flat round trip (close entry), charged at entry
REBAL_COST = 0.005                # round trip on capital moved by --rebalance
CLIP = 0.40                       # sensitivity only
ERA_SPLIT = pd.Timestamp("2023-01-01")
MCAP = ROOT / "data/derived/mcap_pit.parquet"
# mcap_pit pnl_implied rows start 2018-04 (filing_dt floor) and reach steady monthly coverage in 2018-07
# (2,384 / 4,731 / 10,442 / 17,710 rows in 2018-04..07, ~18k/month after); the sensitivity window starts here
PNL_STEADY = pd.Timestamp("2018-07-01")
# mcap_pit (built 2026-09-24) covers the BE/BZ rows of the 2026-08-28 backfill (2025+) but not the 2026-09-27 one
BE_SIZED = pd.Timestamp("2025-01-01")
REPEAT_SHARE = 0.90               # a date where >= 90% of rows repeat the symbol's prior OHLC is not a session
EXPS = {"BASELINE": ""}
for _k in range(1, 4):
    for _c in itertools.combinations("GBH", _k):
        EXPS["+".join(_c)] = "".join(_c)


def slots_for(hold: int) -> int:
    """Weekly ladder with 5*SLOTS > HOLD, so each slot's next cohort enters after the previous one exits."""
    return hold // 5 + 1


# ---------------- data ----------------
def industry_maps() -> dict[str, pd.Series]:
    sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry"])
    nse4 = sc.set_index("symbol")["industry"].map(html.unescape)
    al = pd.read_csv(ROOT / "data/derived/industry_analyst_labels.csv")
    full = pd.concat([nse4, al[~al["symbol"].isin(nse4.index)].set_index("symbol")["analog_symbol"].map(nse4).dropna()])
    return {"analogs": full, "nse4": nse4}


def true_sessions() -> tuple[pd.DatetimeIndex, pd.Series]:
    """Global session calendar (full panel, every symbol) without non-session dates.

    From 2020 the panel carries NSE-holiday dates (2020-12-25, 2021-01-26, 2024-12-25, 2026-09-14, ...) on which
    every row repeats the symbol's previous row's open/high/low/close exactly; research_panel.session_calendar
    counts them as sessions. A date is dropped when >= REPEAT_SHARE of its rows that have a previous row repeat
    it (these dates sit at 100%, real sessions at <= 1.1% in the 2026-09-27 backup) and it has >= 100 such rows.
    Returns (calendar, repeat share of each dropped date)."""
    import pyarrow.parquet as pq
    t = pq.read_table(rp.PANEL, columns=["symbol", "trade_date", "open", "high", "low", "close"],
                      read_dictionary=["symbol"]).unify_dictionaries()   # symbol as int codes: low memory
    s = t.column("symbol").combine_chunks().indices.to_numpy(zero_copy_only=False)
    d = pd.to_datetime(t.column("trade_date").to_numpy()).to_numpy()
    o = np.lexsort((d, s))
    s, d = s[o], d[o]
    prior = np.r_[False, s[1:] == s[:-1]]
    rep = prior.copy()
    for c in ("open", "high", "low", "close"):
        v = t.column(c).to_numpy().astype(float)[o]
        rep[1:] &= v[1:] == v[:-1]
    del t, v, o
    st = pd.DataFrame({"d": d[prior], "rep": rep[prior]}).groupby("d")["rep"].agg(["mean", "size"])
    fake = st.loc[(st["mean"] >= REPEAT_SHARE) & (st["size"] >= 100), "mean"].rename("repeat_share")
    return pd.DatetimeIndex(np.unique(d)).difference(fake.index), fake


def load(symbols: list[str] | None) -> dict:
    filt = [("symbol", "in", symbols)] if symbols else None
    cal, fake = true_sessions()                       # global calendar minus non-session dates
    px = rp.load_panel(["series", "open", "high", "low", "close", "sma_50", "avg_traded_value_20d",
                        "price_adjustment_factor_to_present"], filters=filt)
    on_fake = px["trade_date"].isin(fake.index)
    n_fake_rows = int(on_fake.sum())
    px = px[~on_fake].reset_index(drop=True)          # before ret60/ret252, wide frames, breadth, benchmark
    px["series"] = px["series"].astype(str).astype("category")
    g = px.groupby("symbol")["close"]
    px["ret60"] = g.pct_change(60, fill_method=None)  # registered signal: the symbol's own rows (critic 3)
    px["ret252"] = g.pct_change(252, fill_method=None)
    px["adv"] = px["avg_traded_value_20d"] / 1e7
    px["core"] = (px["adv"] >= 5) & (rp.raw_price(px, "close") > 50)
    mc = pd.read_parquet(MCAP, columns=["symbol", "trade_date", "mcap_cr", "mcap_source"], filters=filt)
    mc["trade_date"] = pd.to_datetime(mc["trade_date"])
    n_dup = int(mc.duplicated(["symbol", "trade_date"]).sum())
    if n_dup:
        print(f"WARNING mcap_pit has {n_dup} duplicate (symbol, trade_date) rows; keeping the last", flush=True)
        mc = mc.drop_duplicates(["symbol", "trade_date"], keep="last")
    px = px.merge(mc, on=["symbol", "trade_date"], how="left")
    px["broad"] = px["mcap_cr"] >= 50                  # registered H universe (all mcap sources)
    px["msrc"] = px["mcap_source"].where(px["mcap_cr"].notna()).fillna("NULL").astype("category")
    px["broad_pnl"] = px["broad"] & (px["msrc"] == "pnl_implied")   # sensitivity: shares known at t only

    C = rp.wide(px, "close", cal)
    R = rp.stitch_renames(rp.gap_aware_returns(C), C)
    O, H, L = (rp.wide(px, c, cal) for c in ("open", "high", "low"))
    ne = rp.next_open_entry(O, H, L, C)
    prev_c = C.ffill().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        first_fac = (C.shift(-1) / ne["entry_px"]).to_numpy()              # close/open of session t+1, row t
        uc_close = ((C >= prev_c * 1.049) & np.isclose(C, H)).to_numpy()   # diagnostic: close entry at UC
        lc_lock = ((O <= prev_c * 0.951) & np.isclose(O, H) & np.isclose(H, L)).to_numpy()
    adv_w = rp.wide(px, "adv", cal).to_numpy()
    # nextrow[t, j] = first session >= t on which symbol j (or its renamed successor, as in R) has a row; T if none
    pres = rp.stitch_renames(C.notna().astype(float).where(C.notna()), C).fillna(0).to_numpy() > 0
    T = len(cal)
    nextrow = np.where(pres, np.arange(T, dtype=np.int32)[:, None], np.int32(T))
    nextrow = np.minimum.accumulate(nextrow[::-1], axis=0)[::-1]
    del pres

    # benchmark: EW core band, membership at the prior close, carried through gaps inside a listing life
    corew = rp.wide(px.assign(core_f=px["core"].astype(float)), "core_f", cal)
    alive = C.ffill().notna() & C.bfill().notna()
    member = corew.ffill().where(alive).eq(1.0).shift(1, fill_value=False)
    bench_ret = R.where(member).mean(axis=1)

    core_rows = px[px["core"] & px["sma_50"].notna()]
    breadth = core_rows.assign(up=core_rows["close"] > core_rows["sma_50"]).groupby("trade_date")["up"].mean()
    return dict(px=px[["symbol", "trade_date", "series", "ret60", "ret252", "core", "broad", "broad_pnl", "msrc"]],
                cal=cal, fake=fake, n_fake_rows=n_fake_rows, R=R, syms=list(R.columns),
                Rv=R.to_numpy(), Rclip=R.clip(-CLIP, CLIP).to_numpy(), cidx={s: i for i, s in enumerate(R.columns)},
                dpos={d: i for i, d in enumerate(cal)}, first_fac=first_fac, nextrow=nextrow,
                locked=ne["locked"].to_numpy(dtype=bool), uc_close=uc_close, lc_lock=lc_lock, adv=adv_w,
                bench_ret=bench_ret, breadth_prev=breadth.shift(1))


def weekly_grid(cal: pd.DatetimeIndex, offset: int = 0) -> list:
    return [d for d in cal[offset::5] if d >= pd.Timestamp(START)]


def select(px: pd.DataFrame, weekly: list, heat_col: str, top_n: int, imap: pd.Series) -> dict:
    """cohort date -> list of symbols to buy (core band only)."""
    W = px[px["trade_date"].isin(set(weekly)) & px[heat_col]].copy()
    W["ind"] = W["symbol"].map(imap)
    W = W[W["ind"].notna() & W["ret60"].notna()]
    W["nm"] = W.groupby(["trade_date", "ind"])["ret60"].transform("size")
    W = W[W["nm"] >= 5]
    h = W.groupby(["trade_date", "ind"])["ret60"].mean().rename("heat").reset_index()
    h["hot"] = h.groupby("trade_date")["heat"].rank(pct=True) >= 0.9
    W = W.merge(h[["trade_date", "ind", "hot"]], on=["trade_date", "ind"])
    W["rk"] = W.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
    P = W[W["hot"] & (W["rk"] <= top_n) & (W["ret252"] > 0.50) & W["core"]]
    return P.groupby("trade_date")["symbol"].apply(list).to_dict()


# ---------------- portfolio ----------------
def run(D: dict, picks: dict, gate: bool, weekly: list, hold: int, entry: str = "close",
        rebalance: int = 0, clip: bool = False, gap_exit: str = "next_row") -> tuple[pd.Series, dict]:
    """Weekly slot ladder. Each slot compounds unit capital through its chain of cohorts (cash when a cohort
    is empty/gated/unbuyable); NAV = sum of slots at 1/SLOTS, optionally reset to equal slots every
    `rebalance` sessions. gap_exit="next_row": a name with no row on its exit session (still listed) is held
    on its own returns until its first row after it, sold at that close, and the cash rejoins the slot at the
    slot's first cohort on or after that row; "stale" books it at the carried close (pre-review)."""
    Rv = D["Rclip"] if clip else D["Rv"]
    dpos, cidx, cal, nextrow = D["dpos"], D["cidx"], D["cal"], D["nextrow"]
    T, slots = len(cal), slots_for(hold)
    start_i = dpos[weekly[0]]
    V = np.full((slots, T), np.nan)
    names_per, invested, events = [], 0, []
    dg = dict(picked=0, bought=0, skip_locked=0, skip_norow=0, exit_lc_locked=0, entry_uc_close=0, nan_ret=0,
              exit_in_gap=0, exit_in_gap_sessions=0, exit_after_final_row=0)
    for s in range(slots):
        cap = 1.0
        v = V[s]
        v[start_i:] = cap                                             # idle cash until its first cohort
        late = np.zeros(T)                                            # positions held past their scheduled exit
        pend = []                                                     # (exit session, proceeds) not yet reinvested
        for k in range(s, len(weekly), slots):
            d = weekly[k]; i0 = dpos[d]
            end = min(i0 + hold, T - 1)
            if k + slots < len(weekly):
                nxt = dpos[weekly[k + slots]]
                assert nxt > end, f"slot {s}: next cohort {weekly[k + slots].date()} enters before exit {cal[end].date()}"
            else:
                nxt = T - 1
            keep = []
            for j, pj in pend:                                        # late-exit cash rejoins at this cohort
                if j <= i0:
                    cap += pj; late[j + 1:i0 + 1] += pj
                else:
                    keep.append((j, pj))
            pend = keep
            names = picks.get(d, [])
            if gate and not (D["breadth_prev"].get(d, np.nan) >= 0.50):
                names = []
            if names and end > i0:
                cols = np.array([cidx[n] for n in names])
                seg = Rv[i0 + 1:end + 1, cols]
                dg["nan_ret"] += int(np.isnan(seg).sum())
                seg = np.nan_to_num(seg)                                 # copy; NaN only before a listing
                dg["picked"] += len(cols)
                if entry == "close":
                    ok = np.ones(len(cols), bool)
                    dg["entry_uc_close"] += int(D["uc_close"][i0, cols].sum())
                    wts = np.full(len(cols), (1 - COST) / len(cols))
                else:
                    ff = D["first_fac"][i0, cols]
                    lk = D["locked"][i0, cols]
                    ok = ~lk & np.isfinite(ff)
                    dg["skip_locked"] += int(lk.sum()); dg["skip_norow"] += int((~lk & ~np.isfinite(ff)).sum())
                    seg[0] = np.where(ok, np.nan_to_num(ff) - 1, 0.0)     # open(t+1) -> close(t+1)
                    c = rp.cost_rt(pd.Series(D["adv"][i0, cols])).to_numpy()
                    wts = np.where(ok, 1 - c, 1.0) / len(cols)             # skipped name: its share stays cash
                growth = np.cumprod(1 + seg, axis=0)
                growth[:, ~ok] = 1.0
                path = (growth * wts).sum(axis=1)                          # (BLAS matmul warns spuriously on macOS)
                v[i0 + 1:end + 1] = cap * path
                val = cap * wts * growth[-1]                               # each name's value at the scheduled exit
                cap = cap * float(path[-1])
                if ok.any():
                    invested += 1; names_per.append(int(ok.sum())); dg["bought"] += int(ok.sum())
                    if i0 + hold <= T - 1:                                 # scheduled exit inside the panel
                        for c, vc in zip(cols[ok], val[ok]):
                            j = int(nextrow[end, c])
                            if j == end:
                                dg["exit_lc_locked"] += int(D["lc_lock"][end, c])
                            elif j >= T:
                                dg["exit_after_final_row"] += 1          # delisted: frozen at last close (registered)
                            else:                                        # no row at exit, trades again at j
                                gj = np.cumprod(1 + np.nan_to_num(Rv[end + 1:j + 1, c]))
                                dg["exit_in_gap"] += 1; dg["exit_in_gap_sessions"] += j - end
                                events.append((int(c), i0, end, j, float(gj[-1] - 1)))
                                if gap_exit == "next_row":
                                    late[end + 1:j + 1] += vc * gj
                                    cap -= vc
                                    pend.append((j, vc * float(gj[-1])))
                                    dg["exit_lc_locked"] += int(D["lc_lock"][j, c])
            else:
                v[i0 + 1:end + 1] = cap                                  # empty / gated cohort: CASH
            v[end + 1:nxt + 1] = cap                                     # cash between exit and next cohort
        for j, pj in pend:                                               # never reinvested: cash to the end
            late[j + 1:] += pj
        v[start_i:] += late[start_i:]
    Vs = V[:, start_i:]
    fin = Vs[:, -1]
    if rebalance <= 0:
        nav = Vs.sum(axis=0) / slots
    else:
        gr = Vs[:, 1:] / Vs[:, :-1]
        w = np.full(slots, 1.0 / slots); out = [1.0]
        for t in range(gr.shape[1]):
            w = w * gr[:, t]
            if (t + 1) % rebalance == 0:
                tot = w.sum()
                tot -= REBAL_COST * np.abs(w - tot / slots).sum() / 2
                w = np.full(slots, tot / slots)
            out.append(w.sum())
        nav = np.array(out)
    top3 = np.sort(fin)[-3:].sum() / fin.sum() if fin.sum() > 0 else np.nan
    info = dict(avg_names=float(np.mean(names_per)) if names_per else 0.0, pct_invested=invested / len(weekly) * 100,
                slot_max_min=float(fin.max() / fin.min()) if fin.min() > 0 else np.nan, slot_top3_share=float(top3), **dg,
                _gap_events=events)                                       # "_" keys are not written to the manifest
    return pd.Series(nav, index=cal[start_i:]), info


def segment(nav: pd.Series, lo: pd.Timestamp | None = None, hi: pd.Timestamp | None = None) -> pd.Series:
    """NAV for the window [lo, hi), based at the close of the last session BEFORE lo (or the first NAV if the
    window starts at or before it), ending at the last session before hi. Adjacent windows chain exactly:
    no session is dropped or counted twice."""
    s = nav if hi is None else nav[nav.index < hi]
    if lo is not None:
        before = s.index[s.index < lo]
        s = s[s.index >= before[-1]] if len(before) else s[s.index >= lo]
    return s


def seg_metrics(nav: pd.Series, lo=None, hi=None) -> dict:
    s = segment(nav, lo, hi)
    return rp.nav_metrics(s) if len(s) > 1 else dict(cagr=np.nan, maxdd=np.nan, sharpe=np.nan)


def metrics(nav: pd.Series) -> dict:
    """Full-period CAGR/maxDD/Sharpe and calendar years from research_panel.nav_metrics. Per-era CAGR, maxDD and
    Sharpe come from nav_metrics on segment(), so every era metric shares one base. nav_metrics' own
    cagr_conf is overwritten on purpose because its conf base is the 2022-12-26 close (see docstring)."""
    m = rp.nav_metrics(nav, era_split=str(ERA_SPLIT.date()))
    m["final"] = nav.iloc[-1] / nav.iloc[0]
    for name, (lo, hi) in (("disc", (None, ERA_SPLIT)), ("conf", (ERA_SPLIT, None))):
        sm = seg_metrics(nav, lo, hi)
        m[f"cagr_{name}"], m[f"maxdd_{name}"], m[f"sharpe_{name}"] = sm["cagr"], sm["maxdd"], sm["sharpe"]
    return m


def beats(r: dict, base: dict) -> bool:
    """Registered rule: CAGR higher in BOTH eras AND full-period maxDD not worse by >2pp."""
    return bool(r["cagr_disc"] > base["cagr_disc"] and r["cagr_conf"] > base["cagr_conf"] and r["maxdd"] >= base["maxdd"] - 2)


def factorial(D: dict, imap: pd.Series, weekly: list, a, with_clip: bool) -> tuple[list, dict]:
    cache, rows, navs = {}, [], {}
    for name, lev in EXPS.items():
        key = ("broad" if "H" in lev else "core", 10 if "B" in lev else 3)
        if key not in cache:
            cache[key] = select(D["px"], weekly, *key, imap)
        nav, info = run(D, cache[key], "G" in lev, weekly, a.hold, a.entry, a.rebalance, gap_exit=a.gap_exit)
        m = metrics(nav)
        row = dict(exp=name, **{k: v for k, v in m.items() if k != "years"}, **info, years=m["years"])
        if with_clip:
            mc_ = metrics(run(D, cache[key], "G" in lev, weekly, a.hold, a.entry, a.rebalance, clip=True,
                              gap_exit=a.gap_exit)[0])
            row.update(clip_cagr=mc_["cagr"], clip_dd=mc_["maxdd"])
        rows.append(row); navs[name] = nav
    return rows, navs


def _period(d: pd.Series) -> np.ndarray:
    return np.where(d < PNL_STEADY, f"..{(PNL_STEADY - pd.Timedelta(days=1)).strftime('%Y-%m')}",
                    np.where(d < ERA_SPLIT, f"{PNL_STEADY.strftime('%Y-%m')}..2022-12", "2023+"))


def mcap_source_sensitivity(D: dict, imap: pd.Series, weekly: list, a) -> dict:
    """H arms rerun with the heat universe restricted to pnl_implied mcap rows (shares from filings known at t;
    still subject to the upstream split lag). pnl_implied does not exist before 2018-04, so every compared arm
    (BASELINE, all-source H and [pnl] H) is a FRESH run of the registered ladder on the weekly cohorts dated
    >= PNL_STEADY: all slots start equal at 1.0 and hold only post-PNL_STEADY picks, so no arm carries slot
    weights set by pre-window (non-PIT) picks. The slot chains are the headline chains from that date on.
    Measured from w0 = the date slot 0 enters its second cohort (one full slot cycle after the fresh start,
    every slot has held a cohort; same window as round 2): disc-pit = [w0, 2022-12-30 close], conf = 2022-12-30
    close .. end, maxDD/Sharpe from w0."""
    px, slots = D["px"], slots_for(a.hold)
    ww = [d for d in weekly if d >= PNL_STEADY]
    w0 = ww[min(slots, len(ww) - 1)]
    W = px[px["trade_date"].isin(set(weekly))]
    per = pd.Series(_period(W["trade_date"]), index=W.index, name="period")
    mix = (pd.crosstab(per[W["broad"]], W.loc[W["broad"], "msrc"].astype(str), normalize="index") * 100).round(1)
    sel = {}

    def picks(col: str, n: int) -> dict:
        if (col, n) not in sel:
            sel[(col, n)] = select(px, weekly, col, n, imap)
        return sel[(col, n)]

    def pairs(p: dict) -> set:
        return {(d, s) for d, ss in p.items() if d >= w0 for s in ss}

    p3 = picks("broad", 3)
    pk = pd.DataFrame([(d, s) for d, ss in p3.items() for s in ss], columns=["trade_date", "symbol"])
    pk = pk.merge(W[["trade_date", "symbol", "msrc"]], on=["trade_date", "symbol"], how="left") if len(pk) else pk
    pick_mix = ((pd.crosstab(_period(pk["trade_date"]), pk["msrc"].astype(str), normalize="index") * 100).round(1)
                if len(pk) else pd.DataFrame())

    def arm(col: str, n: int, gate: bool) -> dict:
        nav = run(D, picks(col, n), gate, ww, a.hold, a.entry, a.rebalance, gap_exit=a.gap_exit)[0]
        d, c, f = seg_metrics(nav, w0, ERA_SPLIT), seg_metrics(nav, ERA_SPLIT), seg_metrics(nav, w0)
        return dict(cagr_disc=d["cagr"], cagr_conf=c["cagr"], maxdd=f["maxdd"], sharpe=f["sharpe"])

    base = arm("core", 3, False)
    rows = [dict(exp="BASELINE", **base)]
    for name, lev in EXPS.items():
        if "H" not in lev:
            continue
        n = 10 if "B" in lev else 3
        all_p, pnl_p = pairs(picks("broad", n)), pairs(picks("broad_pnl", n))
        for tag, col, pp in ((name, "broad", all_p), (f"{name}[pnl]", "broad_pnl", pnl_p)):
            r = dict(exp=tag, **arm(col, n, "G" in lev))
            r["beats"] = beats(r, base)
            r["ungated_picks_from_w0"] = len(pp)
            if col == "broad_pnl":
                r["pct_also_in_all_source"] = len(pnl_p & all_p) / len(pnl_p) * 100 if pnl_p else np.nan
            rows.append(r)
    return dict(fresh_start=str(ww[0].date()), window_start=str(w0.date()), pnl_steady=str(PNL_STEADY.date()),
                window="every arm is a fresh run on cohorts >= fresh_start (equal slots, post-pnl_steady picks only), "
                       "measured from window_start (one slot cycle later): disc-pit to the 2022-12-30 close, conf "
                       "after; maxdd/sharpe from window_start",
                heat_universe_source_mix_pct=mix.to_dict(orient="index"),
                h_top3_pick_source_mix_pct=pick_mix.to_dict(orient="index"), arms=rows)


def data_gaps(D: dict, weekly: list, base_picks: dict, rows: list) -> dict:
    """Missing-data inventory for the run: dropped non-session dates, core-band rows (cohort dates) with NULL
    mcap by period x series (BE/BZ rows added by the panel repair have no mcap_pit row), the BASELINE picks H
    could not have bought for that reason, and exits that fell in a gap."""
    fake = D["fake"]
    px = D["px"]
    W = px[px["trade_date"].isin(set(weekly)) & px["core"]]
    grp = np.where(W["series"].astype(str).isin(["BE", "BZ"]), "BE/BZ", "EQ")
    null = W["msrc"].astype(str).eq("NULL").to_numpy()

    def gp(d: pd.Series) -> np.ndarray:                 # _period with 2023+ split at 2025 (first BE/BZ backfill)
        p = _period(d)
        return np.where(d >= BE_SIZED, "2025+", np.where(p == "2023+", "2023-2024", p))

    t = pd.DataFrame({"period": gp(W["trade_date"]), "series": grp, "null": null})
    core = t.groupby(["period", "series"])["null"].agg(core_rows="size", null_mcap="sum").reset_index()
    core["pct_null"] = (core["null_mcap"] / core["core_rows"] * 100).round(2)
    bp = pd.DataFrame([(d, s) for d, ss in base_picks.items() for s in ss], columns=["trade_date", "symbol"])
    bp = bp.merge(W[["trade_date", "symbol", "series", "msrc", "ret60"]], on=["trade_date", "symbol"], how="left")
    bp["series_grp"] = np.where(bp["series"].astype(str).isin(["BE", "BZ"]), "BE/BZ", "EQ")
    bp["null"] = bp["msrc"].astype(str).eq("NULL")
    bp["period"] = gp(bp["trade_date"])
    picks_null = (bp.groupby(["period", "series_grp"])["null"].agg(picks="size", null_mcap="sum").reset_index()
                  .rename(columns={"series_grp": "series"}))
    bn = bp[bp["null"]].sort_values("trade_date")
    ex_null = pd.concat([bn[bn["series_grp"] == "BE/BZ"].groupby("period").head(2),
                         bn[bn["series_grp"] == "EQ"].head(2)])[["symbol", "trade_date", "series", "ret60"]]
    ev, cal, syms = {}, D["cal"], D["syms"]
    for r in rows:
        for c, i0, end, j, mv in r["_gap_events"]:
            ev.setdefault((syms[c], cal[i0], cal[end]), (cal[j], j - end, mv, set()))[3].add(r["exp"])
    gap_ex = sorted(ev.items(), key=lambda kv: -abs(kv[1][2]))
    return dict(
        non_session_dates=[str(d.date()) for d in fake.index],
        non_session_per_year={int(y): int(n) for y, n in pd.Series(fake.index.year).value_counts().sort_index().items()},
        non_session_min_repeat_share=float(fake.min()) if len(fake) else np.nan,
        non_session_rows_dropped=D["n_fake_rows"],
        core_rows_null_mcap=core.to_dict(orient="records"),
        baseline_picks_null_mcap=picks_null.to_dict(orient="records"),
        baseline_null_mcap_examples=[dict(symbol=r.symbol, trade_date=str(r.trade_date.date()), series=str(r.series),
                                          ret60=float(r.ret60)) for r in ex_null.itertuples()],
        gap_exit_events=len(ev),
        gap_exit_examples=[dict(symbol=k[0], cohort=str(k[1].date()), scheduled_exit=str(k[2].date()),
                                next_row=str(v[0].date()), extra_sessions=int(v[1]), move_pct=round(v[2] * 100, 1),
                                arms=sorted(v[3])) for k, v in gap_ex[:8]])


def _plain(r: dict) -> dict:
    """Row for the manifest: no calendar years, no internal ("_") keys."""
    return {k: v for k, v in r.items() if k != "years" and not k.startswith("_")}


def fmt(m: dict) -> str:
    return (f"CAGR {m['cagr']:>+6.1f}% (disc {m['cagr_disc']:>+6.1f} / conf {m['cagr_conf']:>+6.1f}) · maxDD {m['maxdd']:>6.1f}% "
            f"(disc {m['maxdd_disc']:>6.1f} / conf {m['maxdd_conf']:>6.1f}) · Sharpe {m['sharpe']:.2f} "
            f"(disc {m['sharpe_disc']:.2f} / conf {m['sharpe_conf']:.2f})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hold", type=int, default=126)
    ap.add_argument("--entry", choices=["close", "next_open"], default="close")
    ap.add_argument("--rebalance", type=int, default=0, help="sessions between slot rebalances (0 = never)")
    ap.add_argument("--labels", choices=["analogs", "nse4"], default="analogs")
    ap.add_argument("--gap-exit", choices=["next_row", "stale"], default="next_row",
                    help="exit of a pick with no row on its exit session: its first row after it (default) or the "
                         "stale carried close (pre-review)")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip slot-phase spread, label sensitivity and mcap-source sensitivity")
    ap.add_argument("--panel", help="panel parquet override (smoke tests on a backup copy)")
    ap.add_argument("--symbols", help="comma list or @file; smoke tests only, forces --dry-run")
    ap.add_argument("--out-dir", default=str(ROOT / "logs/leader_sleeve"))
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    a = ap.parse_args(argv)
    if a.panel:
        rp.PANEL = Path(a.panel)
    syms = None
    if a.symbols:
        syms = (Path(a.symbols[1:]).read_text().split() if a.symbols.startswith("@") else a.symbols.split(","))
        syms = sorted({s.strip() for s in syms if s.strip()})
        a.dry_run = True
    slots = slots_for(a.hold)

    D = load(syms)
    maps = industry_maps()
    cal = D["cal"]
    weekly = weekly_grid(cal)
    print(f"HOLD {a.hold} sessions · {slots} weekly slots · entry {a.entry} · rebalance {a.rebalance or 'never'} · labels {a.labels}"
          f" · gap exit {a.gap_exit}{' · SYMBOL SUBSET ' + str(len(syms)) if syms else ''}\npanel {rp.PANEL.name} "
          f"{cal[0].date()}..{cal[-1].date()} · {len(cal)} sessions after dropping {len(D['fake'])} non-session dates "
          f"({D['n_fake_rows']} loaded rows; see DATA GAPS) · weekly cohorts {len(weekly)} from {weekly[0].date()} · "
          f"breadth>=0.50 on {(D['breadth_prev'].reindex(weekly) >= 0.5).mean()*100:.0f}% of cohort dates\n", flush=True)

    out_rows, navs = factorial(D, maps[a.labels], weekly, a, with_clip=True)
    for r in out_rows:
        print(f"{r['exp']:<9} {fmt(r)} · x{r['final']:.1f} · names/cohort {r['avg_names']:.1f} · invested {r['pct_invested']:.0f}% "
              f"| clip±40%: CAGR {r['clip_cagr']:+.1f} DD {r['clip_dd']:.1f}", flush=True)

    b = D["bench_ret"]
    b = b[b.index >= weekly[0]].copy()
    b.iloc[0] = 0.0                                                    # NAV = 1.0 on the first cohort date
    bnav = (1 + b.fillna(0)).cumprod()
    bm = metrics(bnav)
    print(f"{'BENCH EW':<9} {fmt(bm)}")
    conf_base = segment(navs["BASELINE"], ERA_SPLIT).index[0]
    rp_conf = rp.nav_metrics(navs["BASELINE"], era_split=str(ERA_SPLIT.date()))["cagr_conf"]
    print(f"eras: disc {weekly[0].date()}..{conf_base.date()} close · conf based at the {conf_base.date()} close"
          + (f" · research_panel.nav_metrics would give BASELINE conf {rp_conf:+.2f} (vs {out_rows[0]['cagr_conf']:+.2f}):"
             " its conf base is the first session >= 2022-12-25 (primitive defect, not used)"
             if abs(rp_conf - out_rows[0]["cagr_conf"]) > 0.005 else ""), flush=True)

    print("\nCALENDAR-YEAR RETURNS (%, prior year-end to year-end)")
    yrs = sorted(out_rows[0]["years"])
    print(f"{'':<9}" + "".join(f"{y:>7}" for y in yrs))
    for r in out_rows:
        print(f"{r['exp']:<9}" + "".join(f"{r['years'].get(y, np.nan):>7.1f}" for y in yrs))
    print(f"{'BENCH EW':<9}" + "".join(f"{bm['years'].get(y, np.nan):>7.1f}" for y in yrs))

    base = out_rows[0]
    print("\nVERDICT vs BASELINE (registered: CAGR higher in BOTH eras AND maxDD not worse by >2pp)")
    for r in out_rows[1:]:
        print(f"  {r['exp']:<7} {'BEATS' if beats(r, base) else 'no   '} · disc {r['cagr_disc']-base['cagr_disc']:+.1f}pp · "
              f"conf {r['cagr_conf']-base['cagr_conf']:+.1f}pp · maxDD {r['maxdd']-base['maxdd']:+.1f}pp", flush=True)

    print("\nEXECUTION / LADDER DIAGNOSTICS")
    for r in out_rows:
        ex = (f"skipped UC-locked {r['skip_locked']} · skipped no-row {r['skip_norow']} · " if a.entry == "next_open"
              else f"close-entry at probable UC lock {r['entry_uc_close']} · ")
        gx = (f"exit in gap {r['exit_in_gap']} (+{r['exit_in_gap_sessions'] / r['exit_in_gap']:.1f} sessions avg, "
              f"{'sold at next row' if a.gap_exit == 'next_row' else 'booked STALE'})" if r["exit_in_gap"] else "exit in gap 0")
        print(f"  {r['exp']:<7} picks {r['picked']} bought {r['bought']} · {ex}exit LC-locked {r['exit_lc_locked']} · "
              f"{gx} · exit after final row {r['exit_after_final_row']} · "
              f"NaN returns in holds {r['nan_ret']} · slot end max/min x{r['slot_max_min']:.1f} · top-3 slots "
              f"{r['slot_top3_share']*100:.0f}% of NAV")

    gaps = data_gaps(D, weekly, select(D["px"], weekly, "core", 3, maps[a.labels]), out_rows)
    print("\nDATA GAPS (what is missing from the inputs, with example rows)")
    print(f"  1) non-session dates dropped: {len(gaps['non_session_dates'])} (per year "
          + ", ".join(f"{y}:{n}" for y, n in gaps["non_session_per_year"].items())
          + f"; min repeat share {gaps['non_session_min_repeat_share']:.3f}; {gaps['non_session_rows_dropped']} loaded rows)"
          + (f"\n     e.g. {', '.join(gaps['non_session_dates'][:4])} ... {', '.join(gaps['non_session_dates'][-3:])}"
             if gaps["non_session_dates"] else ""))
    print("  2) core-band rows on cohort dates with NULL mcap (outside the H universe; H cannot buy them), and the"
          " BASELINE top-3 picks among them:")
    pn = {(x["period"], x["series"]): x for x in gaps["baseline_picks_null_mcap"]}
    for x in gaps["core_rows_null_mcap"]:
        q = pn.get((x["period"], x["series"]), dict(picks=0, null_mcap=0))
        print(f"     {x['period']:<17} {x['series']:<6} core rows {x['core_rows']:>7} · NULL mcap {x['null_mcap']:>6}"
              f" ({x['pct_null']:5.1f}%) · BASELINE picks {q['picks']:>5}, NULL mcap {q['null_mcap']:>4}")
    for e in gaps["baseline_null_mcap_examples"]:
        print(f"     e.g. BASELINE pick {e['symbol']:<12} {e['trade_date']} series {e['series']} ret60 {e['ret60']*100:+.0f}% · mcap NULL")
    print(f"  3) exits whose scheduled session has no row for the name (distinct symbol/cohort, headline arms): "
          f"{gaps['gap_exit_events']}; largest moves after the scheduled exit"
          f"{'' if a.gap_exit == 'next_row' else ' (NOT booked: --gap-exit stale)'}:")
    for e in gaps["gap_exit_examples"]:
        print(f"     {e['symbol']:<12} cohort {e['cohort']} · scheduled exit {e['scheduled_exit']} · next row {e['next_row']} "
              f"(+{e['extra_sessions']} sessions) · move {e['move_pct']:+.1f}% · arms {','.join(e['arms'])}")

    phase, label_rows = [], []
    if not a.no_sensitivity:
        print("\nSLOT-PHASE SPREAD (weekly grid offset 0..4 sessions, same slots/options; offset 0 = registered)")
        per = {o: (out_rows if o == 0 else factorial(D, maps[a.labels], weekly_grid(cal, o), a, with_clip=False)[0])
               for o in range(5)}
        for name in EXPS:
            rs = [next(r for r in per[o] if r["exp"] == name) for o in range(5)]
            bs = [per[o][0] for o in range(5)]
            dd_ = [r["cagr_disc"] for r in rs]; cc_ = [r["cagr_conf"] for r in rs]; md_ = [r["maxdd"] for r in rs]
            row = dict(exp=name, disc=[min(dd_), float(np.mean(dd_)), max(dd_)], conf=[min(cc_), float(np.mean(cc_)), max(cc_)],
                       maxdd=[min(md_), max(md_)], per_offset=[dict(offset=o, cagr=r["cagr"], cagr_disc=r["cagr_disc"],
                       cagr_conf=r["cagr_conf"], maxdd=r["maxdd"]) for o, r in enumerate(rs)])
            line = (f"  {name:<8} disc {min(dd_):+6.1f}/{np.mean(dd_):+6.1f}/{max(dd_):+6.1f} · conf {min(cc_):+6.1f}/"
                    f"{np.mean(cc_):+6.1f}/{max(cc_):+6.1f} · maxDD {min(md_):6.1f}..{max(md_):6.1f}")
            if name != "BASELINE":
                md = [r["cagr_disc"] - bb["cagr_disc"] for r, bb in zip(rs, bs)]
                mcf = [r["cagr_conf"] - bb["cagr_conf"] for r, bb in zip(rs, bs)]
                nb = sum(beats(r, bb) for r, bb in zip(rs, bs))
                row.update(margin_disc=[min(md), max(md)], margin_conf=[min(mcf), max(mcf)], offsets_beating=nb)
                line += f" · vs BASE disc {min(md):+.1f}..{max(md):+.1f} conf {min(mcf):+.1f}..{max(mcf):+.1f} · BEATS at {nb}/5"
            phase.append(row); print(line)
        print("  (min/mean/max CAGR %; margins vs BASELINE at the same offset)")

        other = "nse4" if a.labels == "analogs" else "analogs"
        print(f"\nLABEL SENSITIVITY — same factorial with labels={other} (README: report with and without analyst analogs)")
        lr, _ = factorial(D, maps[other], weekly, a, with_clip=False)
        for r, r0 in zip(lr, out_rows):
            v = "" if r["exp"] == "BASELINE" else (" · BEATS" if beats(r, lr[0]) else " · no")
            print(f"  {r['exp']:<8} {fmt(r)} · Δ vs {a.labels}: disc {r['cagr_disc']-r0['cagr_disc']:+.1f} conf "
                  f"{r['cagr_conf']-r0['cagr_conf']:+.1f}{v}")
            label_rows.append(_plain(r))

    msens = {}
    if not a.no_sensitivity:
        msens = mcap_source_sensitivity(D, maps[a.labels], weekly, a)
        print("\nMCAP-SOURCE SENSITIVITY (H lever). mcap_pit has 0 pnl_implied rows before 2018-04; earlier sized rows"
              " use present-day shares x adjusted close (not PIT)")
        print("  broad (mcap>=50cr) heat-universe rows on cohort dates, % by mcap_source (NULL-mcap core rows: DATA GAPS 2):")
        for p, d in msens["heat_universe_source_mix_pct"].items():
            print(f"    {p:<17} " + " · ".join(f"{k} {v:.1f}" for k, v in d.items()))
        print("  H top-3 picks, % by mcap_source:")
        for p, d in msens["h_top3_pick_source_mix_pct"].items():
            print(f"    {p:<17} " + " · ".join(f"{k} {v:.1f}" for k, v in d.items()))
        print(f"  every arm below is a FRESH run on cohorts from {msens['fresh_start']} (equal slots, no pre-window picks; "
              f"not the headline NAVs), measured from {msens['window_start']} (one slot cycle later) · disc-pit "
              f"{msens['window_start']}..2022-12-30 close · conf 2023+ · maxDD/Sharpe from {msens['window_start']} · "
              f"[pnl] = heat universe restricted to pnl_implied rows · rule as registered, on this window")
        bw = msens["arms"][0]
        for r in msens["arms"]:
            line = (f"  {r['exp']:<13} disc-pit {r['cagr_disc']:>+6.1f} · conf {r['cagr_conf']:>+6.1f} · maxDD {r['maxdd']:>6.1f}"
                    f" · Sharpe {r['sharpe']:.2f}")
            if r["exp"] != "BASELINE":
                line += (f" · vs BASE disc-pit {r['cagr_disc']-bw['cagr_disc']:+.1f} conf {r['cagr_conf']-bw['cagr_conf']:+.1f}"
                         f" maxDD {r['maxdd']-bw['maxdd']:+.1f} · {'BEATS' if r['beats'] else 'no'}")
            if "ungated_picks_from_w0" in r:
                line += f" · ungated picks {r['ungated_picks_from_w0']}"
            if "pct_also_in_all_source" in r:
                line += f" ({r['pct_also_in_all_source']:.0f}% also all-source picks)"
            print(line)
        print("  (all-source H on this window = same construction mix in both eras; [pnl] also drops every symbol that\n"
              "   has no quarterly filings (screener_backfill-only), and still carries the upstream split lag. The\n"
              "   pre-2018-07 H stretch cannot be tested point-in-time.)", flush=True)

    if a.dry_run:
        print("\nDRY RUN — nothing written")
        print("\nPORTFOLIO 7X COMPLETE")
        return
    suffix = (("" if a.hold == 126 else f"_h{a.hold}") + ("_nextopen" if a.entry == "next_open" else "")
              + (f"_rebal{a.rebalance}" if a.rebalance else "") + ("_nse4" if a.labels == "nse4" else "")
              + ("_stalegap" if a.gap_exit == "stale" else ""))
    out = Path(a.out_dir) / f"portfolio_7x_nav_v2{suffix}.parquet"
    pd.DataFrame(navs).assign(BENCH_EW=bnav.reindex(next(iter(navs.values())).index)).to_parquet(out)
    out.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="leader sleeve 7x factorial NAVs (post 2026-09-27 audit fixes)", experiment="EXP-2026-09-27-leader-7x-factorial",
        producer="src/agentic/sim_leader_portfolio_7x.py", panel=str(rp.PANEL),
        params=dict(hold=a.hold, slots=slots, entry=a.entry, rebalance=a.rebalance, labels=a.labels, start=START,
                    gap_exit=a.gap_exit, era_split=str(ERA_SPLIT.date()),
                    sessions=f"panel dates minus {len(D['fake'])} non-session dates (>= {REPEAT_SHARE:.0%} of rows repeat "
                             "the prior OHLC; listed in data_gaps.non_session_dates)",
                    era_base=f"each era based at the last session before it (conf base = {conf_base.date()} close); "
                             "disc x conf = full-period multiple",
                    cost="flat 0.5% round trip at entry" if a.entry == "close"
                    else "research_panel.cost_rt(ADV at signal date) round trip at entry", rebalance_cost=REBAL_COST),
        units={"cagr*, maxdd*, clip_*, pct_invested, margins, calendar_years": "percent",
               "sharpe*": "annualised from daily returns, rf=0", "final": "NAV multiple",
               "<exp> / BENCH_EW columns": "NAV, 1.0 on the first cohort date"},
        columns={"<exp>": "daily NAV (G=market gate, B=top-10 basket, H=broad heat on mcap_pit mcap_cr>=50, all "
                          "sources; not point-in-time before 2018-07; rows without an mcap_pit row, incl. pre-2025 "
                          "BE/BZ rows added by the 2026-09-27 repair, are outside H)",
                 "BENCH_EW": "equal-weight core band (ADV>=5cr & then-traded close>50), daily rebalanced, membership at the prior close"},
        summary=[_plain(r) for r in out_rows],
        calendar_years={**{r["exp"]: r["years"] for r in out_rows}, "BENCH_EW": bm["years"]},
        bench={k: v for k, v in bm.items() if k != "years"}, slot_phase_spread=phase, label_sensitivity=label_rows,
        mcap_source_sensitivity=msens, data_gaps=gaps,
        known_limitations=["on the pre-repair panel pre-2025 BE/BZ sessions are missing (stale marks inside gaps; an exit "
                           "inside one waits for the next row); suspensions stay real gaps on any panel",
                           "mcap_pit (built 2026-09-24) predates the 2026-09-27 BE/BZ backfill: pre-2025 BE/BZ rows have "
                           "NULL mcap, so they are outside the H universe and H arms cannot buy them while BASELINE can "
                           "(era-asymmetric); fix = rebuild mcap_pit after the backfill; counts in data_gaps",
                           "non-session (holiday) dates are dropped here; research_panel.session_calendar and the daily "
                           "writer still produce them, and the panel's sma_50/avg_traded_value_20d include those rows",
                           "H universe (mcap_pit) is not point-in-time before 2018-04/07: present-day share counts x "
                           "adjusted close, 23% NULL rows; see mcap_source_sensitivity",
                           "mcap_pit upstream defects (build_mcap_pit.py audit findings): split lag in pnl_implied, "
                           "screener_backfill sized with renamed successors' shares, 316 large names NULL before mid-2018",
                           "industry labels are a 2026-09 crawl applied back to 2016 (not point-in-time)",
                           "exits locked at the lower circuit are filled at the close (counted, not modelled)",
                           "unadjusted demergers/rights/reverse splits in the panel (clip line bounds them)",
                           "ret60/ret252/ADV/SMA50 roll over rows, not sessions"],
        updated=datetime.now().astimezone().isoformat(timespec="seconds")), indent=1, default=float))
    print(f"\nwrote {out} (+manifest)")
    print("\nPORTFOLIO 7X COMPLETE")


if __name__ == "__main__":
    main()
