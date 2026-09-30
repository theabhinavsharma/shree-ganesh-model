"""LEADER CELL v2 — the industry-leader philosophy rebuilt from scratch (EXP-2026-09-23b).

What changed vs sim_leader_sleeve.py (2026-09-08), each found on 2026-09-23:
  1. SURVIVORSHIP: v1 required a full 126td future (hi126/cl126 notna) BEFORE ranking, so
     names that later delisted/suspended were never selectable. v2 ranks every core-band
     name at t; a name that stops trading inside 126td exits at its last close.
  2. UNIVERSE: fund units removed by ISIN (security_master INF*), not by symbol regex.
  3. INDUSTRY MAP: v1 used modal smIndustry (covers ~40% of core names). v2 runs on
     --map nse (same labels), --map nse4 (NSE 4-level 'Industry' via screener.in for
     every equity — one taxonomy, ~all core names) and --map nse4_full (+ analyst analog
     labels for the ~14 liquid names no feed labels).
  4. HEAT: v1 = MEAN own ret60 of the group (one takeover stock made jewellery 'hot').
     v2 reports MEAN and MEDIAN heat.
  5. TAKEOVERS: names with an open-offer / detailed-public-statement filing in the prior
     180 days (announcements_historical, 2016+) can be excluded.
Arms (registered): A0 mean-heat top-3 · A1 = A0 + EXTENDED (ret252>0.5, production) ·
B = median heat + EXTENDED + no takeover targets (primary) · plus single-change arms and
A1 as originally computed (v1 survivorship) for the size of the bias.
Hold 126 sessions close-to-close, 0.5% round trip, weekly cohorts 2016-06+, core band.

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json; every 'confirmed' finding on this file, plus the
critic items that name it). Line numbers refer to the pre-fix file. Results from before this block (sim_v2_<map>.json,
sim_v2_<map>_<universe>.json and the ledger rows built on them) are superseded; this version writes
sim_v2_<map>_<universe>[_pit]_<entry>_<cost>.json so the old files are not overwritten.
  FIXED
  * [HIGH] lines 72-76, 80, 83-86: the 126-day hold, the forward high/low window and the ret60/ret252 lookbacks counted
    a symbol's own ROWS, so data gaps stretched them to 150-1,700 market days. Now all are on the global session
    calendar: exit/hi/lo come from research_panel.forward_window (next 126 SESSIONS), and ret60/ret252 compare the close
    with the last close on or before t-60 / t-252 sessions (price carried across gaps). When session t+126 falls inside
    a data hole the exit is the last close before it (the low end of the audit's range); 'stale_exits' counts picks
    whose exit close is more than 10 sessions old.
  * [MEDIUM] lines 151-154, 178, 182: '~maxDD' spread cohort endpoint returns over 26 steps, never marked positions to
    market and skipped weeks with no picks (about 3x too shallow). It is replaced by a daily mark-to-market NAV: 26
    overlapping slots (one per weekly cohort, reused every 26 weeks), each cohort equal-weighted inside its slot,
    positions marked daily with research_panel.gap_aware_returns, weeks with no picks held as cash, CAGR/maxDD/Sharpe
    from research_panel.nav_metrics, per era (disc NAV < 2023-01-01, conf from the last 2022 session). The registered
    'maxDD (sim proxy) not worse by >2pp' bars have to be re-read on NAV maxDD.
  * [MEDIUM] line 96: the core price floor used the back-adjusted close, which depends on later splits and bonuses.
    Now research_panel.raw_price(close) > 50 (the then-traded price).
  * [LOW] lines 72-73, 78-84: renames were booked as delistings and successors started with no history. Renames from
    research_panel.rename_map() whose successor starts within 10 days of the old symbol's last row are chained: the old
    symbol's hold continues on the successor (exit/hi/lo/NAV), and the successor inherits the old history for
    ret60/ret252. Same validity rule and the same zero-return junction as research_panel.stitch_renames (that helper
    only continues returns forward, and this script also needs price levels and the backward history).
  * [LOW] lines 32, 101-109: the takeover flag matched acquirer-side and non-offer text, and a filing made after the
    15:30 close counted at that day's close. Now only target-side filings count (SAST-category filings, which NSE posts
    on the target's page, a third party such as the manager to the offer filing on the symbol's page, or text that
    makes the filer the subject of the offer); filings that say the offer is made by the filer's subsidiary, or that
    the filer is buying another company's shares, are dropped, and self-filed updates with no target evidence are left
    out. Filings timestamped at or after 15:30 IST count from the next session.
  * [review follow-up, 2026-09-27] right-censoring in the survivorship reference. No signal in the last 126 sessions
    has a full future, so the 'A1 as v1' arm had almost no picks there and its NAV sat in cash while the other arms
    were invested; its conf NAV mixed that cash drag into the survivorship gap it exists to measure (backup panel,
    nse4/core from 2021-06: conf NAV CAGR A1 +24.4% vs v1 +18.1%, but +13.3% vs +12.4% on a common window). The v1
    NAV now ends at session n-1-126 (NaN after it in the NAV parquet), and every arm also reports NAV metrics on that
    same window (nav_*_cmp in the era rows, '<arm>|nav_cmp' blocks, a like-for-like table). Per-trade stats had the
    matching problem: a signal whose 126-session window runs past the panel end was kept only if the name had stopped
    trading (its exit is known), which selects on the outcome. Those signals are now left out of per-trade stats in
    every arm; they still enter the NAV, marked to market.
  PARTIAL
  * [MEDIUM] lines 92-94: mcap50 treated every mcap_pit row as point-in-time. --pit_only keeps only rows whose
    mcap_source == 'pnl_implied'. It is a sensitivity run, not a replacement: pnl_implied has no rows before 2018-04,
    so under --pit_only the disc era is 2018-2022 only, and names without P&L history (many delisted) drop out, which
    tilts toward survivors. The default still uses every source, as registered, and the help text no longer calls it PIT.
  * [LOW] lines 47-59, 88, 98: rows with no industry label are dropped before heat and rank, and delisted names are
    unlabeled more often. Renamed symbols with no label now take their rename chain's label (e.g. an old symbol whose
    successor is labeled). The label coverage of universe rows (names still trading vs names that stopped) is printed
    and saved so the remaining tilt is visible for each map, and --map nse prints a warning. NOT FIXED: every map
    applies today's taxonomy to 2016+ dates; the repo has no point-in-time industry source.
  Critic items
  * #4 execution: --entry next_open buys at the t+1 open (research_panel.next_open_entry), skips entries that are
    upper-circuit locked or have no t+1 row, and uses ADV-scaled costs (research_panel.cost_rt; --cost auto).
    --entry close keeps the registered close entry with a flat 0.5%. TODO (not handled): exits locked at the lower
    circuit, and locks in the 2% price band, which sit below the primitive's 4.9% trigger.
  * NOT FIXED here: demergers, rights and capital reductions are still unadjusted in the panel (critic #1; the panel is
    being repaired upstream); avg_traded_value_20d, which sets ADV and the core band, rolls over rows not sessions
    (critic #3; src/features/indicators.py); the live screen uses a different universe from these arms (critic #5;
    screen_theme_leaders.py); screener_backfill mcap rows can carry a successor's share count (build_mcap_pit.py
    finding; --pit_only drops those rows).
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

COST, HOLD = 0.5, 126          # registered: flat 0.5% round trip (percent) in close mode; hold in SESSIONS
SLOTS, STEP = 26, 5            # weekly cohorts = every 5th session; 26 overlapping NAV slots (26 x 5 >= 126)
START = pd.Timestamp("2016-06-01")
ERA_CUT = pd.Timestamp("2023-01-01")   # disc = cohorts in 2016-2022, conf = 2023+
CLOSE_MIN = 15 * 60 + 30               # NSE close, IST; filings at or after it are known from the next session
OO_RE = r"open offer|detailed public statement"
V1_ARM = "A1 as v1 (survivorship ref)"
assert HOLD <= SLOTS * STEP, "a slot must be free again before its next cohort"

# ---------- takeover filing side (target vs acquirer) ----------
SAST_DESC = {"open offer", "public announcement-open offer", "post offer public announcement", "corrigendum",
             "disclosure under sebi takeover regulations"}
_STOP = re.compile(r"\b(limited|ltd|the|india|company|co|corporation|corp|inc|pvt|private|and|of)\b")
_ACQ = re.compile(r"subsidiary of (the )?(company|bank)\b|company s participation|further acquisition of|"
                  r"open offer (made |launched )?by (the )?(company|bank)\b")
_TGT = re.compile(r"\b(shareholders?|shares|share capital|offer|stake) (of|for|to|in) (the )?(company|bank|tc)\b|"
                  r"\beach of the (company|bank)\b|\bcompany s (equity )?shares\b|independent directors|\bidc\b|"
                  r"\b26 7\b|\bour (public )?shareholders\b|\bfrom the acquirer\b|"
                  r"\b(receipt|received)\b.{0,120}\b(public announcement|detailed public statement|"
                  r"letter of (open )?offer|dps|advertisement)\b|"
                  r"\b(public announcement|detailed public statement|letter of offer|dps|advertisement)\b.{0,80}"
                  r"\breceived (by the (company|bank) )?from\b")
_SUBMIT = re.compile(r"^(.{3,140}?) (has )?(submitted|submited|informed)\b")


def _norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s or "").lower())).strip()


def filing_side(desc, text, sm_name) -> str:
    """'target' when the filer is the company whose shares the offer seeks, 'acquirer' when the text says the filer
    (or its subsidiary) is the buyer, 'unclear' otherwise (not counted as a takeover target)."""
    d = str(desc or "").strip().lower()
    body = _norm(text)
    t = _norm(desc) + " " + body
    core = " ".join(_STOP.sub(" ", _norm(sm_name)).split()[:2])
    ce = re.escape(core) if len(core) >= 4 else None
    if _ACQ.search(t) or (ce and re.search(r"subsidiary of " + ce, t)):
        return "acquirer"
    if d in SAST_DESC or _TGT.search(t):
        return "target"
    m = _SUBMIT.match(body)
    if m and ce and not re.search(ce, m.group(1)) and "exchange" not in m.group(1):
        return "target"            # a third party (the manager to the offer) filed on this symbol's page
    if ce and (re.search(ce + r".{0,80}target company", t)
               or re.search(r"\b(shareholders|shares|offer|stake) (of|for|in) .{0,20}" + ce, t)):
        return "target"
    return "unclear"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", choices=["nse", "nse4", "nse4_full"], default="nse")
    ap.add_argument("--universe", choices=["core", "mcap50"], default="core",
                    help="core = ADV>=5cr & then-traded close>50 (validated); mcap50 = mcap_pit >= Rs 50cr, no liquidity "
                         "floor (mcap_pit mixes point-in-time pnl_implied rows with screener backcast/backfill rows that "
                         "use present or last-trade share counts; see --pit_only)")
    ap.add_argument("--pit_only", action="store_true",
                    help="mcap50 only: keep rows with mcap_source == 'pnl_implied' (none before 2018-04; drops names "
                         "without P&L history, so it tilts toward survivors). Sensitivity run next to the default.")
    ap.add_argument("--entry", choices=["close", "next_open"], default="close",
                    help="close = registered signal-day close entry; next_open = buy at the t+1 open, skip "
                         "upper-circuit-locked entries and entries with no t+1 row (held as cash in the NAV). Both "
                         "modes exit at the close of session t+126")
    ap.add_argument("--cost", choices=["auto", "flat", "adv"], default="auto",
                    help="flat = 0.5%% round trip; adv = research_panel.cost_rt (0.5/1/2%% by ADV); auto = flat for "
                         "close entry, adv for next_open")
    ap.add_argument("--panel", help="panel parquet override (smoke tests: the .bak copy)")
    ap.add_argument("--date_from", help="load panel rows from this date (smoke tests)")
    ap.add_argument("--date_to", help="load panel rows up to this date (smoke tests)")
    ap.add_argument("--dry_run", action="store_true", help="print results only; write no files")
    a = ap.parse_args(argv)
    if a.pit_only and a.universe != "mcap50":
        ap.error("--pit_only needs --universe mcap50")
    a.cost_mode = a.cost if a.cost != "auto" else ("flat" if a.entry == "close" else "adv")
    return a


# ---------- industry map ----------
# nse       : NSE smIndustry (old 74-industry list; ~1,213 equities — what v1 saw)
# nse4      : NSE 4-level 'Industry' via screener.in for EVERY equity it covers (one taxonomy)
# nse4_full : nse4 + analyst analog labels (industry_analyst_labels.csv) for the residual
def industry_map(map_name: str) -> pd.Series:
    sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
    fund = set(sm.loc[sm["is_fund_unit"], "symbol"])
    if map_name == "nse":
        imap = sm.dropna(subset=["industry"]).set_index("symbol")["industry"]
    else:
        sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
        sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry"])
        imap = sc.set_index("symbol")["industry"].map(_html.unescape)
        if map_name == "nse4_full":
            al = pd.read_csv(ROOT / "data/derived/industry_analyst_labels.csv")
            al = al[~al["symbol"].isin(imap.index)]
            add = al.set_index("symbol")["analog_symbol"].map(imap).dropna()
            imap = pd.concat([imap, add])
            print(f"analyst analog labels: +{len(add)} (of {len(al)})", flush=True)
    imap = imap[~imap.index.isin(fund)]
    print(f"industry labels: {len(imap):,} equities, {imap.nunique()} industries ({map_name})", flush=True)
    if map_name == "nse":
        print("WARNING --map nse: delisted names are labeled far less often than survivors (audit 2026-09-27); "
              "compare with nse4/nse4_full", flush=True)
    return imap


# ---------- renames: chain old -> successor on the session grid ----------
def first_last(nn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    has = nn.any(0)
    first = np.where(has, nn.argmax(0), -1)
    last = np.where(has, len(nn) - 1 - nn[::-1].argmax(0), -1)
    return first, last


def rename_chains(C: pd.DataFrame) -> tuple[list[list[str]], dict]:
    """Chains [oldest, ..., newest] from rename_map(): successor present, starting on or after the old symbol's last
    row and within 10 days of it (research_panel.stitch_renames' rule). A successor claimed by two old symbols is
    ambiguous (merger-like) and skipped."""
    cal = C.index
    first, last = first_last(C.notna().to_numpy())
    pos = {s: i for i, s in enumerate(C.columns)}
    rmap = rp.rename_map()
    st = dict(rename_pairs=len(rmap), not_in_panel=0, successor_before_old_end=0, gap_over_10d=0, ambiguous=0)
    nxt = {}
    for old, new in rmap.items():
        if old == new or old not in pos or new not in pos:
            st["not_in_panel"] += 1
            continue
        lo, fn = last[pos[old]], first[pos[new]]
        if fn < lo:
            st["successor_before_old_end"] += 1
            continue
        if (cal[fn] - cal[lo]).days > 10:
            st["gap_over_10d"] += 1
            continue
        nxt[old] = new
    tgt = pd.Series(list(nxt.values()), dtype=object).value_counts()
    amb = set(tgt[tgt > 1].index)
    st["ambiguous"] = int(sum(v in amb for v in nxt.values()))
    nxt = {o: n for o, n in nxt.items() if n not in amb}
    prev = {n: o for o, n in nxt.items()}
    chains = []
    for head in nxt:
        if head in prev:
            continue
        ch, seen = [head], {head}
        while ch[-1] in nxt and nxt[ch[-1]] not in seen:
            ch.append(nxt[ch[-1]]); seen.add(ch[-1])
        chains.append(ch)
    st["chained_pairs"] = len(nxt)
    st["chains"] = len(chains)
    return chains, st


def stitch_chains(frames: dict[str, pd.DataFrame], chains: list[list[str]]) -> dict[str, pd.DataFrame]:
    """Every member column of a chain gets the whole chain's rows (predecessors before, successors after), rescaled to
    the member's price level. The junction session (old last close -> successor first close) carries zero return, as
    in research_panel.stitch_renames. Member rows keep their own values (on a same-day overlap the old row wins)."""
    close = frames["close"]
    pos = {s: i for i, s in enumerate(close.columns)}
    cl = close.to_numpy()
    first, last = first_last(~np.isnan(cl))
    orig = {k: v.to_numpy() for k, v in frames.items()}
    out = {k: v.to_numpy(copy=True) for k, v in frames.items()}
    n = len(close)
    for ch in chains:
        cols = [pos[s] for s in ch]
        scale = [1.0]
        for a, b in zip(cols[:-1], cols[1:]):
            scale.append(scale[-1] * cl[last[a], a] / cl[first[b], b])
        if not all(np.isfinite(scale)) or min(scale) <= 0:
            continue
        for k, arr in orig.items():
            chain = np.full(n, np.nan)
            lo_prev = -1
            for c, s in zip(cols, scale):
                seg = slice(lo_prev + 1, last[c] + 1)
                chain[seg] = arr[seg, c] * s
                lo_prev = last[c]
            for c, s in zip(cols, scale):
                out[k][:, c] = chain / s
    return {k: pd.DataFrame(out[k], index=close.index, columns=close.columns) for k in out}


# ---------- takeover targets (point-in-time, 180d) ----------
def takeover_filings(cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    pf = pq.ParquetFile(ROOT / "data/derived/announcements_historical.parquet")
    keep = []
    for b in pf.iter_batches(batch_size=100_000, columns=["symbol", "desc", "attchmntText", "sort_date", "sm_name"]):
        d = b.to_pandas()
        t = d["desc"].fillna("") + " " + d["attchmntText"].fillna("")
        m = t.str.contains(OO_RE, case=False, regex=True)
        if m.any():
            keep.append(d[m])
    oo = pd.concat(keep, ignore_index=True)
    oo["side"] = [filing_side(a, b, c) for a, b, c in zip(oo["desc"], oo["attchmntText"], oo["sm_name"])]
    ts = pd.to_datetime(oo["sort_date"], errors="coerce")          # IST wall-clock
    oo = oo[ts.notna()].copy(); ts = ts[ts.notna()]
    day = ts.dt.normalize()
    after = (ts.dt.hour * 60 + ts.dt.minute) >= CLOSE_MIN
    # known at the close of the filing day if filed before 15:30, else from the next session's close
    # (outside the loaded calendar the next calendar day stands in; merge_asof backward maps it to the next session)
    nxt_i = np.searchsorted(cal.values, day.values, side="right")
    ok = (day.values >= cal.values[0]) & (nxt_i < len(cal))
    nxt = np.where(ok, cal.values[np.minimum(nxt_i, len(cal) - 1)], (day + pd.Timedelta(days=1)).values)
    oo["same_day"] = day
    oo["trade_date"] = pd.to_datetime(np.where(after.values, nxt, day.values))
    st = dict(filings=len(oo), symbols=int(oo["symbol"].nunique()),
              by_side={k: int(v) for k, v in oo["side"].value_counts().items()},
              after_close=int(after.sum()))
    return oo, st


def flag_takeover(wk: pd.DataFrame, oo: pd.DataFrame, date_col: str) -> pd.Series:
    f = oo.dropna(subset=[date_col])[["symbol", date_col]].drop_duplicates().rename(columns={date_col: "trade_date"})
    f = f.sort_values("trade_date"); f["oo_dt"] = f["trade_date"]
    m = pd.merge_asof(wk[["trade_date", "symbol"]].reset_index().sort_values("trade_date"), f, on="trade_date",
                      by="symbol", direction="backward", tolerance=pd.Timedelta(days=180))
    return m.set_index("index")["oo_dt"].reindex(wk.index)


# ---------- heat (mean + median), leaders ----------
def add_heat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nm"] = df.groupby(["trade_date", "ind"])["ret60"].transform("size")
    df = df[df["nm"] >= 5].copy()
    for how in ("mean", "median"):
        h = df.groupby(["trade_date", "ind"])["ret60"].agg(how).rename("h").reset_index()
        h[f"hot_{how}"] = h.groupby("trade_date")["h"].rank(pct=True) >= 0.9
        df = df.merge(h[["trade_date", "ind", f"hot_{how}"]], on=["trade_date", "ind"])
    df["rk"] = df.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
    df["ext"] = df["ret252"] > 0.50
    return df


def outcomes(df: pd.DataFrame) -> pd.DataFrame:
    e = df["entry_px"]
    df["net"] = (df["exit_close"] / e - 1) * 100 - df["cost_pct"]
    df["x2"] = df["hi_f"] / e - 1 >= 1.0
    df["x50"] = df["hi_f"] / e - 1 >= 0.5
    df["trough"] = (df["lo_f"] / e - 1) * 100
    df["year"] = df["trade_date"].dt.year
    return df


# ---------- daily mark-to-market NAV: 26 overlapping weekly slots ----------
def cohort_nav(sel: pd.DataFrame, G: np.ndarray, wk_idx: np.ndarray, cal: pd.DatetimeIndex, entry: str) -> pd.Series:
    """sel = the arm's picks (all signals, including right-censored ones) with columns k (weekly index), ci (symbol
    column), cost (fraction), tradable, e_mult (close(t+1)/open(t+1) for next-open entry, else 1). Slot k % 26 invests
    its whole cash in cohort k, equal weight, half the round-trip cost on entry and half on exit; a position is marked
    with the gap-aware growth index G until session t+126 (frozen after a delisting), then returns to cash. A pick that
    cannot be entered stays in cash for that cycle. Positions still open at the panel end are marked, not sold."""
    n = len(cal); i_start = int(wk_idx[0]); lag = 1 if entry == "next_open" else 0
    groups = {k: g for k, g in sel.groupby("k")}
    nav = np.zeros(n - i_start)
    for s in range(SLOTS):
        val = np.empty(n - i_start)
        cash, cur = 1.0 / SLOTS, i_start
        for k in range(s, len(wk_idx), SLOTS):
            i0 = int(wk_idx[k])
            val[cur - i_start:i0 - i_start + 1] = cash
            cur = i0 + 1
            g = groups.get(k)
            if g is None or len(g) == 0:
                continue
            ix = min(i0 + HOLD, n - 1)
            alloc = cash / len(g)
            path = np.full(ix - i0 + 1, alloc * (~g["tradable"]).sum())       # untradable picks sit in cash
            e = i0 + lag
            gt = g[g["tradable"]]
            exit_cash = path[-1]
            if len(gt) and e <= ix:
                c = gt["ci"].to_numpy(); cost = gt["cost"].to_numpy()
                w = alloc * (1 - cost / 2) * gt["e_mult"].to_numpy()
                pv = w * G[e:ix + 1, c] / G[e, c]                               # (sessions, picks)
                path[:e - i0] += alloc * len(gt)
                path[e - i0:] += pv.sum(1)
                exit_cash += (pv[-1] * (1 - cost / 2)).sum() if i0 + HOLD <= n - 1 else pv[-1].sum()
            else:
                path += alloc * len(gt)
                exit_cash += alloc * len(gt)
            if i0 + HOLD <= n - 1:
                path[-1] = exit_cash
            val[i0 - i_start:ix - i_start + 1] = path
            cash, cur = path[-1], ix + 1
        val[cur - i_start:] = cash
        nav += val
    return pd.Series(nav, index=cal[i_start:])


def era_nav(nav: pd.Series) -> tuple[dict, dict]:
    nav = nav.dropna()
    if len(nav) < 2:                                   # empty window (e.g. a smoke-test panel shorter than the hold)
        nanm = dict(cagr=np.nan, maxdd=np.nan, sharpe=np.nan)
        return dict(nanm, cagr_disc=np.nan, cagr_conf=np.nan, years={}), {"disc": dict(nanm), "conf": dict(nanm)}
    full = rp.nav_metrics(nav)
    disc = nav[nav.index < ERA_CUT]
    conf = nav[nav.index >= (disc.index[-1] if len(disc) else nav.index[0])]
    eras = {}
    for name, seg in (("disc", disc), ("conf", conf)):
        if len(seg) > 20:
            m = rp.nav_metrics(seg)
            eras[name] = dict(cagr=m["cagr"], maxdd=m["maxdd"], sharpe=m["sharpe"])
        else:
            eras[name] = dict(cagr=np.nan, maxdd=np.nan, sharpe=np.nan)
    return full, eras


def main(argv=None):
    args = parse_args(argv)
    imap = industry_map(args.map)

    # ---------- panel on the session calendar (delisting- and rename-aware) ----------
    if args.panel:
        rp.PANEL = Path(args.panel)
    filters = []
    if args.date_from:
        filters.append(("trade_date", ">=", pd.Timestamp(args.date_from)))
    if args.date_to:
        filters.append(("trade_date", "<=", pd.Timestamp(args.date_to)))
    px = rp.load_panel(["open", "high", "low", "close", "avg_traded_value_20d", "price_adjustment_factor_to_present"],
                       filters=filters or None)
    cal = rp.session_calendar(px)
    n = len(cal)
    print(f"panel {rp.PANEL.name}: {len(px):,} equity rows, {px['symbol'].nunique():,} symbols, "
          f"{cal[0].date()}..{cal[-1].date()} ({n:,} sessions)", flush=True)
    frames = {c: rp.wide(px, c, cal) for c in ("close", "high", "low", "open")}
    cols = frames["close"].columns
    assert all(f.columns.equals(cols) for f in frames.values())
    chains, rstat = rename_chains(frames["close"])
    frames = stitch_chains(frames, chains)
    C = frames["close"]
    # a renamed symbol with no label takes its chain's latest label (same company; the map is a snapshot anyway)
    filled = 0
    for ch in chains:
        known = [imap[s] for s in ch if s in imap.index]
        for s in ch:
            if known and s not in imap.index:
                imap.loc[s] = known[-1]; filled += 1
    rstat["labels_filled_from_chain"] = filled
    print(f"renames: {rstat}", flush=True)

    wk_dates = cal[::STEP]
    wk_dates = wk_dates[wk_dates >= START]
    wk_idx = cal.get_indexer(wk_dates)
    cand = px[px["trade_date"].isin(wk_dates)].copy()
    del px
    ri = cal.get_indexer(cand["trade_date"]); ci = cols.get_indexer(cand["symbol"])
    cand["ri"], cand["ci"] = ri, ci
    cand["k"] = pd.Index(wk_dates).get_indexer(cand["trade_date"])

    Cff = C.ffill().to_numpy()
    for L, name in ((60, "ret60"), (252, "ret252")):          # lookbacks in SESSIONS, last close carried across gaps
        v = np.full(len(cand), np.nan); ok = ri >= L
        v[ok] = Cff[ri[ok], ci[ok]] / Cff[ri[ok] - L, ci[ok]] - 1
        cand[name] = v
    nn = C.notna().to_numpy()
    _, last_st = first_last(nn)                                 # last row per (stitched) symbol
    rowpos = np.where(nn, np.arange(n)[:, None], -1)
    np.maximum.accumulate(rowpos, axis=0, out=rowpos)           # last row on or before each session
    del Cff, nn

    fw = rp.forward_window(C, frames["high"], frames["low"], HOLD)
    cand["exit_close"] = fw["exit"].to_numpy()[ri, ci]
    cand["hi_f"] = fw["hi"].to_numpy()[ri, ci]
    cand["lo_f"] = fw["lo"].to_numpy()[ri, ci]
    del fw
    cand["held"] = np.minimum(HOLD, last_st[ci] - ri)            # sessions until the exit (or the final row)
    cand["full_future"] = last_st[ci] >= ri + HOLD
    tgt = np.minimum(ri + HOLD, n - 1)
    cand["stale_exit"] = (ri + HOLD <= n - 1) & (last_st[ci] >= ri + HOLD) & (tgt - rowpos[tgt, ci] > 10)
    del rowpos

    cl_np = C.to_numpy()
    if args.entry == "next_open":
        ne = rp.next_open_entry(frames["open"], frames["high"], frames["low"], C)
        cand["entry_px"] = ne["entry_px"].to_numpy()[ri, ci]
        cand["locked"] = ne["locked"].to_numpy()[ri, ci].astype(bool)
        del ne
        c1 = np.full(len(cand), np.nan); ok = ri + 1 <= n - 1
        c1[ok] = cl_np[ri[ok] + 1, ci[ok]]
        cand["no_open"] = ~(np.isfinite(cand["entry_px"].to_numpy()) & np.isfinite(c1))
        cand["e_mult"] = c1 / cand["entry_px"].to_numpy()
    else:
        cand["entry_px"] = cand["close"]
        cand["locked"] = False
        cand["no_open"] = False
        cand["e_mult"] = 1.0
    cand["tradable"] = ~cand["locked"] & ~cand["no_open"]
    G = (1.0 + rp.gap_aware_returns(C).fillna(0.0)).cumprod().to_numpy()
    del frames, C, cl_np

    cand["adv"] = cand["avg_traded_value_20d"] / 1e7            # rupees -> Rs crore (docs/index.html data contract)
    cand["raw_close"] = rp.raw_price(cand, "close")
    cand["ind"] = cand["symbol"].map(imap)
    if args.cost_mode == "flat":
        cand["cost"] = COST / 100
    else:
        cand["cost"] = rp.cost_rt(cand["adv"]).to_numpy()
    cand["cost_pct"] = cand["cost"] * 100
    # per-trade outcome known and not outcome-selected: the whole 126-session window must lie inside the panel (in the
    # tail only names that stopped trading have an exit, so keeping them would select on the outcome)
    cand["valid"] = cand["exit_close"].notna() & cand["tradable"] & (cand["ri"] + HOLD <= n - 1)

    if args.universe == "mcap50":
        mc = pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet", columns=["symbol", "trade_date", "mcap_cr", "mcap_source"],
                             filters=[("trade_date", ">=", wk_dates[0]), ("trade_date", "<=", wk_dates[-1])])
        mc["trade_date"] = pd.to_datetime(mc["trade_date"])
        cand = cand.merge(mc, on=["symbol", "trade_date"], how="left")
        in_uni = cand["mcap_cr"] >= 50                            # mcap_cr: Rs crore (mcap_pit manifest)
        if args.pit_only:
            in_uni &= cand["mcap_source"] == "pnl_implied"
    else:
        in_uni = (cand["adv"] >= 5) & (cand["raw_close"] > 50)
    base = cand[in_uni & cand["ret60"].notna()]
    stopped = last_st[base["ci"].to_numpy()] < n - 1 - STEP * 2  # name stops trading before the panel end
    cov = dict(rows=len(base), labeled=float(base["ind"].notna().mean()),
               labeled_still_trading=float(base.loc[~stopped, "ind"].notna().mean()) if (~stopped).any() else None,
               labeled_stopped=float(base.loc[stopped, "ind"].notna().mean()) if stopped.any() else None,
               unlabeled_rows_stopped=int(base.loc[stopped, "ind"].isna().sum()))
    print(f"label coverage of universe rows ({args.map}): {cov}", flush=True)
    wk = base[base["ind"].notna()].copy()
    del cand, base

    oo, ostat = takeover_filings(cal)
    oo_t = oo[oo["side"] == "target"]
    wk["oo_dt"] = flag_takeover(wk, oo_t, "trade_date")
    wk["takeover"] = wk["oo_dt"].notna()
    old_flag = flag_takeover(wk, oo, "same_day").notna()          # pre-fix rule, for the record
    ostat.update(weekly_rows=len(wk), flagged_rows=int(wk["takeover"].sum()), flagged_rows_prefix_rule=int(old_flag.sum()))
    print(f"open-offer filings: {ostat}", flush=True)

    # v1 method: drop every name without a full 126-session future BEFORE heat/rank (survivorship reference)
    v1 = outcomes(add_heat(wk[wk["full_future"]]))
    wk = outcomes(add_heat(wk))
    top3 = wk["rk"] <= 3
    ARMS = {
        "A0 mean-heat top3": wk["hot_mean"] & top3,
        "A1 PRODUCTION (A0+EXT)": wk["hot_mean"] & top3 & wk["ext"],
        "  A1 + median heat only": wk["hot_median"] & top3 & wk["ext"],
        "  A1 + no-takeover only": wk["hot_mean"] & top3 & wk["ext"] & ~wk["takeover"],
        "B PRIMARY (median+EXT+no-TO)": wk["hot_median"] & top3 & wk["ext"] & ~wk["takeover"],
        "  B without EXT": wk["hot_median"] & top3 & ~wk["takeover"],
        "  A1 & ADV<1cr": wk["hot_mean"] & top3 & wk["ext"] & (wk["adv"] < 1),
        "  A1 & ADV 1-5cr": wk["hot_mean"] & top3 & wk["ext"] & (wk["adv"] >= 1) & (wk["adv"] < 5),
        "  A1 & ADV>=5cr": wk["hot_mean"] & top3 & wk["ext"] & (wk["adv"] >= 5),
    }
    ARMS_DF = {k: (wk, m) for k, m in ARMS.items()}
    ARMS_DF[V1_ARM] = (v1, v1["hot_mean"] & (v1["rk"] <= 3) & v1["ext"])

    # like-for-like NAV window: signals up to session n-1-HOLD have their whole hold inside the panel, so the v1 arm
    # (full future required) is not right-censored before it. After it v1 has almost no picks and would sit in cash.
    i_cmp = n - 1 - HOLD
    cmp_end = cal[i_cmp] if i_cmp >= int(wk_idx[0]) else None
    cmp_lbl = str(cmp_end.date()) if cmp_end is not None else "n/a (panel shorter than one hold)"
    out, navs, cmp_rows = {}, {}, {}
    print(f"\nentry={args.entry} cost={args.cost_mode} hold={HOLD} sessions · NAV = {SLOTS} overlapping weekly slots, "
          f"daily mark-to-market", flush=True)
    print(f"\n{'ARM':<32s}| era      |     n | tr/yr | mean/tr | med/tr | win% | P2x  | P50  | medTrough | worst coh "
          f"| navDD  | navCAGR | yrs+ ", flush=True)
    for name, (D, mask) in ARMS_DF.items():
        P = D[mask]
        nav_all = cohort_nav(P, G, wk_idx, cal, args.entry)
        nav_cmp = nav_all[nav_all.index <= cmp_end] if cmp_end is not None else nav_all.iloc[:0]
        nav = nav_cmp if name == V1_ARM else nav_all           # v1 is reported only where it is not right-censored
        full, eras = era_nav(nav)
        full_c, eras_c = era_nav(nav_cmp)
        navs[name.strip()] = nav
        end_lbl = cmp_lbl if name == V1_ARM else str(cal[-1].date())
        out[f"{name.strip()}|nav"] = dict(full, window_end=end_lbl)
        out[f"{name.strip()}|nav_cmp"] = dict(full_c, window_end=cmp_lbl, eras=eras_c)
        cmp_rows[name] = eras_c
        V = P[P["valid"]]
        for era, em, pm in (("disc<=22", V["year"] <= 2022, P["year"] <= 2022),
                            ("conf>=23", V["year"] >= 2023, P["year"] >= 2023)):
            ek = "disc" if era.startswith("disc") else "conf"
            S = V[em]; E = eras[ek]; Ec = eras_c[ek]
            skip = dict(signals=int(pm.sum()), skipped_locked=int((P.loc[pm, "locked"]).sum()),
                        skipped_no_open=int((P.loc[pm, "no_open"] & ~P.loc[pm, "locked"]).sum()),
                        censored_tail=int((P.loc[pm, "ri"] + HOLD > n - 1).sum()))
            navk = dict(nav_maxdd=E["maxdd"], nav_cagr=E["cagr"], nav_sharpe=E["sharpe"], nav_window_end=end_lbl,
                        nav_maxdd_cmp=Ec["maxdd"], nav_cagr_cmp=Ec["cagr"], nav_sharpe_cmp=Ec["sharpe"])
            if len(S) < 30:
                print(f"{name:<32s}| {era} | {len(S):>5,} | too few", flush=True)
                out[f"{name.strip()}|{era}"] = dict(n=len(S), **navk, **skip)
                continue
            coh = S.groupby("trade_date")["net"].mean()
            yrs = max((S["trade_date"].max() - S["trade_date"].min()).days / 365.25, 0.1)
            yp = S.groupby("year")["net"].mean()
            r = dict(n=len(S), tr_yr=len(S) / yrs, mean=S["net"].mean(), med=S["net"].median(),
                     win=(S["net"] > 0).mean() * 100, p2x=S["x2"].mean() * 100, p50=S["x50"].mean() * 100,
                     trough=S["trough"].median(), worst=coh.min(), yrs=f"{(yp > 0).sum()}/{len(yp)}", **navk,
                     delisted_exits=int((S["held"] < HOLD).sum()), stale_exits=int(S["stale_exit"].sum()),
                     mean_cost_pct=S["cost_pct"].mean(), **skip)
            out[f"{name.strip()}|{era}"] = r
            print(f"{name:<32s}| {era} | {r['n']:>5,} | {r['tr_yr']:>5.0f} | {r['mean']:>+6.2f}% | {r['med']:>+6.2f}% | "
                  f"{r['win']:>4.1f} | {r['p2x']:>4.1f} | {r['p50']:>4.1f} | {r['trough']:>+8.1f}% | {r['worst']:>+8.1f}% | "
                  f"{r['nav_maxdd']:>+6.1f}% | {r['nav_cagr']:>+6.1f}% | {r['yrs']:>5s}"
                  f"{'  *' if name == V1_ARM else ''}", flush=True)
    print(f"* {V1_ARM}: NAV columns end at {cmp_lbl} (session n-1-{HOLD}); after it no signal has a full future, so "
          f"v1 would sit in cash. Per-trade columns in every arm exclude signals whose hold runs past the panel end.",
          flush=True)
    print(f"\nLIKE-FOR-LIKE NAV, every arm through {cmp_lbl}  (disc / conf)", flush=True)
    print(f"{'ARM':<32s}| {'CAGR disc':>9s} | {'maxDD disc':>10s} | {'Sharpe d':>8s} | {'CAGR conf':>9s} | "
          f"{'maxDD conf':>10s} | {'Sharpe c':>8s}", flush=True)
    for name, ec in cmp_rows.items():
        d, c = ec["disc"], ec["conf"]
        print(f"{name:<32s}| {d['cagr']:>+8.1f}% | {d['maxdd']:>+9.1f}% | {d['sharpe']:>8.2f} | {c['cagr']:>+8.1f}% | "
              f"{c['maxdd']:>+9.1f}% | {c['sharpe']:>8.2f}", flush=True)

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")
    out["_meta"] = dict(created_ist=now_ist, args={k: v for k, v in vars(args).items()}, panel=str(rp.PANEL),
                        sessions=[str(cal[0].date()), str(cal[-1].date())], hold_sessions=HOLD, slots=SLOTS,
                        era_cut=str(ERA_CUT.date()), renames=rstat, takeover=ostat, label_coverage=cov,
                        nav_cmp_end=cmp_lbl,
                        note="per-trade eras by cohort date; NAV eras by calendar date (disc < 2023-01-01, conf from "
                             "the last 2022 session); exits inside data holes use the last close before t+126. "
                             "Per-trade stats use only signals whose 126-session hold ends inside the panel "
                             "(censored_tail counts the rest; they still enter the NAV). The '" + V1_ARM + "' NAV "
                             "ends at nav_cmp_end, because after it no signal has a full future and v1 would sit in "
                             "cash; compare it with the other arms on nav_*_cmp / '<arm>|nav_cmp' (same window), not "
                             "on their full-window NAV.")
    if args.dry_run:
        print("dry run: no files written", flush=True)
    else:
        tag = f"{args.map}_{args.universe}{'_pit' if args.pit_only else ''}_{args.entry}_{args.cost_mode}"
        dest = ROOT / "logs/leader_sleeve"
        (dest / f"sim_v2_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
        nav_path = dest / f"sim_v2_{tag}_nav.parquet"
        pd.DataFrame(navs).rename_axis("trade_date").to_parquet(nav_path)
        manifest = dict(
            dataset=f"leader cell v2 daily NAVs ({tag})", experiment="EXP-2026-09-23b-leader-cell-v2 (2026-09-27 audit fixes)",
            producer="src/agentic/sim_leader_cell_v2.py", path=str(nav_path.relative_to(ROOT)), created_ist=now_ist,
            args=out["_meta"]["args"], panel=str(rp.PANEL),
            columns={"trade_date": "session (global NSE equity calendar)",
                     "<arm>": "daily NAV, 1.0 = starting capital on the first weekly cohort date (2016-06+), "
                              f"{SLOTS} overlapping weekly slots, marked to market daily with gap-aware returns; "
                              f"cost {args.cost_mode} ({'0.5% round trip' if args.cost_mode == 'flat' else 'cost_rt by ADV'}), "
                              f"entry at the {'signal-day close' if args.entry == 'close' else 't+1 open, locked entries held as cash'}. "
                              f"The '{V1_ARM}' column ends at {cmp_lbl} (NaN after it)"},
            nav_cmp_end=cmp_lbl,
            summary={k: {m: out[k][m] for m in ("cagr", "maxdd", "sharpe", "cagr_disc", "cagr_conf", "window_end")}
                     for k in out if k.endswith("|nav") or k.endswith("|nav_cmp")},
            caveats=["exits inside data holes use the last close before t+126 (low end of the audit range)",
                     f"'{V1_ARM}' is right-censored after {cmp_lbl}: its NAV stops there; compare arms on the "
                     "'|nav_cmp' summaries (all arms through the same date), not v1 against full-window NAVs",
                     "panel still has unadjusted demergers/rights (critic #1)",
                     "industry labels are today's taxonomy applied to all dates"])
        Path(str(nav_path) + ".manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
        print(f"wrote {dest / f'sim_v2_{tag}.json'} and {nav_path.name} (+ manifest)", flush=True)
    print("LEADER CELL V2 COMPLETE", flush=True)
    return out


if __name__ == "__main__":
    main()
