"""ALL-IN x HOLD MATRIX (EXP-2026-09-27-allin-hold-matrix).

Every approach x every hold, with 100% of capital entering at once, equal weight, held h sessions,
sold, and rotated into that date's picks. Phase offsets = h/5 weekly starts; median and worst phase
reported. 0.5% round trip per rotation. Empty pick set = cash. Buys restricted to tradable names
(ADV >= 5cr & close > 50).
Approaches: 8 rule variants of the leader cell (BASELINE + G gate / B top-10 / H broad heat),
MODEL top-10 / top-20 (walk-forward LightGBM out-of-sample score from anatomy_1p5x, 2019+),
equal-weight tradable market.
Inputs: logs/leader_sleeve/anatomy_1p5x/rows.parquet (weekly mcap>=50cr rows with features + pred),
        adjusted price panel for daily returns.
Output: logs/leader_sleeve/allin_matrix.csv (+ manifest) + stdout.
        (since the 2026-09-27 audit fixes: logs/leader_sleeve/allin_matrix/, see below)

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json, file src/agentic/sim_allin_matrix.py;
line numbers refer to the pre-fix file, commit 77bf4c8). The old logs/leader_sleeve/allin_matrix.csv is the
pre-audit output and is superseded by logs/leader_sleeve/allin_matrix/matrix_<entry>_<cost>.csv.
  FIXED
  - [high] L28-30, L71  gap moves deleted. pct_change(fill_method=None).fillna(0) booked every missing
    session AND the first session back as 0%, so pre-2025 BE/T2T lower-circuit runs vanished (TANLA
    2020-12-10 -> 12-21). Returns now come from research_panel.gap_aware_returns (price carried across
    the gap, the whole move lands on the first session back). A position that must be SOLD while its
    name has no print is sold at the first session it prints again (<= CATCHUP_MAX sessions; value booked
    at the rotation, proceeds treated as reinvested then), not at the stale pre-gap close.
  - [medium] L28-30, L69-71  renames treated as delistings. research_panel.stitch_renames continues an
    old symbol with its successor's returns (splice-day return 0, so the post-rename split factors that
    exist only on the new symbol cannot inject fake -50..-95% days).
  - [medium] L36, L42, L48, L54  core band tested on the back-adjusted close (future splits/bonuses).
    core is now ADV >= 5cr & research_panel.raw_price(close) > 50, computed from the panel, and drives
    BASELINE heat, every rule pick, the MODEL candidate pool, the EW benchmark and the G breadth.
  - [low] L43-44  G gate read same-day breadth. It now uses PRIOR-session breadth (breadth.shift(1) on
    the global session calendar; share of raw-price core names with close > SMA50), the registered G
    lever of EXP-2026-09-27-leader-7x-factorial.
  - [low] L58-61, L82, L88-89  worst phase = min over h//5 phases (6 at h30 .. 50 at h250) and no era
    split. Every hold now also reports the worst of a FIXED 6 evenly spaced phase offsets
    (cagr_worst6 / dd_worst6, comparable across holds) next to the full distribution (median, p10, min
    over all phases, and the per-phase CSV). Every metric is reported for era = all / disc (< 2023) /
    conf (>= 2023), cut from the same daily NAV path (research_panel.nav_metrics).
  - [low] L71  0.5% charged on 100% of capital every rotation, EW benchmark included. Default
    --cost turnover charges every arm only on the weight that changes: sum |w_target - w_drifted| x RT/2
    (a full rotation = the registered 0.5%; final liquidation charged). --cost full keeps the registered
    convention (RT on all invested capital at every rotation).
  - (critic item 4, execution realism) --entry next_open: signal at the close of d, buy at the open of
    d+1 (research_panel.next_open_entry); names upper-circuit locked at that open are not bought (their
    weight stays cash, or stays at the already-held amount); the old basket is sold at the same open;
    costs scale with ADV (research_panel.cost_rt). --entry close (default) is the registered mode.
  PARTIAL
  - [medium] L51-53  MODEL `pred` leaks future splits through adjusted-close log_px / pe (anatomy_1p5x
    L86, L123). Not fixable here: it needs an anatomy refit on raw-basis prices. The script tests
    rows.parquet log_px against log10(adjusted close) and log10(raw price) on scored rows that have a
    later split/bonus. If pred is not point-in-time the MODEL arms are SKIPPED, because the raw-price
    core re-admits exactly the future splitters the leak ranks highest (audit: +5..+11pp spurious
    CAGR). --allow-leaky-model runs them anyway, labelled pred_pit=False.
  NOT FIXED
  - [low] L51-53  after-close (>= 15:30) filings keyed to the same day in the pred features
    (anatomy_1p5x L117-139). Lives in anatomy; MODEL arms inherit it until anatomy is refit.
  - [low] L25-26, L48, L54  universe = rows.parquet (PIT mcap >= Rs50cr). Securities with no mcap_pit
    value (mostly delisted or renamed: LTI, PHILIPCARB, DHFL, WABCOINDIA, ...; ~0.33% of core rows plus
    ~1.8% of core rows with a stale share count, concentrated in 2016-2018) never enter heat, picks or
    EW. Needs a mcap_pit / anatomy rebuild, not a change here.
  - [low] L36-42  industry labels are one static 2026 screener map applied back to 2016 (not point-in-
    time). No historical label source exists in the repo; rule and MODEL arms carry an unquantified
    hindsight-grouping bias, largest in disc.
  - Names that never print again (genuine delistings) are frozen at their last close, not marked down:
    the panel carries no delisting value. Unadjusted corporate actions in the panel (e.g. TIDEWATER's
    2021 split, factor 1.0 across a 67-session gap; demergers with no factor) now show up as real moves
    across gaps; the run prints the largest single-session moves on picked names so they can be checked.
  - Lower-circuit-locked exits are not modelled (sold at the open print even if locked).
  - ret60 / ret252 are taken from rows.parquet as built by anatomy_1p5x (pct_change over each symbol's
    own ROWS, not sessions); fixing them belongs to anatomy_1p5x.

CLI
  --entry close|next_open   (default close = registered)   --cost turnover|full (default turnover)
  --holds 30,60,...  --windows 2019+,2016+  --allow-leaky-model
  --symbols A,B,...  --panel PATH  --out-dir DIR   (smoke tests: subset universe, never the default dir)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

ROWS = ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet"
OUT_DIR = ROOT / "logs/leader_sleeve/allin_matrix"
HOLDS = [30, 60, 90, 100, 180, 250]
WINDOWS = {"2019+": "2019-01-01", "2016+": "2016-06-01"}
ERA_SPLIT = pd.Timestamp("2023-01-01")
FLAT_RT = 0.005        # registered round trip (close-entry mode)
N_FIXED = 6            # phase offsets used for cross-hold worst-phase comparison
CATCHUP_MAX = 250      # sessions a gapped name may take to print again before it is frozen at its last close
RULES = [(), ("G",), ("B",), ("H",), ("G", "B"), ("G", "H"), ("B", "H"), ("G", "B", "H")]


# ----------------------------------------------------------------------------------------- data
@dataclass
class Data:
    cal: pd.DatetimeIndex
    syms: np.ndarray
    col: dict                   # symbol -> column index
    dpos: dict                  # session date -> row index
    R: np.ndarray               # gap-aware, rename-stitched close-to-close returns (NaN before first print)
    TR: np.ndarray              # stitched "printed a close this session" mask
    ADV: np.ndarray | None      # 20d ADV Rs cr, forward-filled (next_open costs)
    OPEN_OK: np.ndarray | None  # own open and close printed this session (buyable)
    ID: np.ndarray | None       # close/open - 1 (entry session)
    ON: np.ndarray | None       # open / previous close - 1, rename-stitched (exit at the open)
    LOCK: np.ndarray | None     # row t: session t+1 upper-circuit locked at the open


def load(entry: str, syms: list[str] | None):
    filt = [("symbol", "in", syms)] if syms else None
    S = pd.read_parquet(ROWS, columns=["symbol", "trade_date", "ind", "ret60", "ret252", "pred", "log_px"], filters=filt)
    S["trade_date"] = pd.to_datetime(S["trade_date"])
    cols = ["close", "sma_50", "avg_traded_value_20d", "price_adjustment_factor_to_present"]
    if entry == "next_open":
        cols += ["open", "high", "low"]
    px = rp.load_panel(cols, filters=filt)
    dup = px.duplicated(["symbol", "trade_date"])
    if dup.any():
        raise SystemExit(f"panel has {int(dup.sum())} duplicate (symbol, trade_date) rows, e.g. "
                         f"{px.loc[dup, ['symbol', 'trade_date']].head(3).to_dict('records')}; fix the panel first")
    px["adv"] = px["avg_traded_value_20d"] / 1e7
    px["core"] = (px["adv"] >= 5) & (rp.raw_price(px, "close") > 50)
    cal = rp.session_calendar(px)

    # G gate input: prior-session breadth of the raw-price core band
    cd = px[px["core"] & px["sma_50"].notna()]
    breadth = (cd["close"] > cd["sma_50"]).groupby(cd["trade_date"]).mean().reindex(cal)
    breadth_prev = breadth.shift(1)

    pit, pit_note = pred_pit_check(S, px)
    S = S.merge(px[["symbol", "trade_date", "core"]], on=["symbol", "trade_date"], how="left")
    S["core"] = S["core"].eq(True)

    C = rp.wide(px, "close", cal)
    R = rp.stitch_renames(rp.gap_aware_returns(C), C)
    TR = rp.stitch_renames(C.notna().astype(float), C)
    syms_ = np.asarray(C.columns)
    D = Data(cal=cal, syms=syms_, col={s: j for j, s in enumerate(syms_)}, dpos={d: i for i, d in enumerate(cal)},
             R=R.to_numpy(float), TR=TR.to_numpy(float) > 0.5, ADV=None, OPEN_OK=None, ID=None, ON=None, LOCK=None)
    del R, TR
    if entry == "next_open":
        O = rp.wide(px, "open", cal).reindex(columns=C.columns)
        O = O.where(O > 0)
        Hh = rp.wide(px, "high", cal).reindex(columns=C.columns)
        L = rp.wide(px, "low", cal).reindex(columns=C.columns)
        D.LOCK = rp.next_open_entry(O, Hh, L, C)["locked"].to_numpy(bool)
        del Hh, L
        D.OPEN_OK = (O.notna() & C.notna()).to_numpy()
        D.ID = (C / O - 1).to_numpy(float)
        D.ON = rp.stitch_renames(O / C.ffill().shift(1) - 1, C).to_numpy(float)
        D.ADV = rp.wide(px, "adv", cal).reindex(columns=C.columns).ffill().to_numpy(float)
        del O
    del px, C
    return S, D, breadth_prev, pit, pit_note


def pred_pit_check(S: pd.DataFrame, px: pd.DataFrame) -> tuple[bool | None, str]:
    """Is rows.parquet log_px (a pred feature) on the then-traded price? Compared on scored rows that have a
    later split/bonus (factor < 0.95), where adjusted and raw prices differ."""
    m = S.loc[S["pred"].notna() & S["log_px"].notna(), ["symbol", "trade_date", "log_px"]].merge(
        px[["symbol", "trade_date", "close", "price_adjustment_factor_to_present"]], on=["symbol", "trade_date"])
    f = pd.to_numeric(m["price_adjustment_factor_to_present"], errors="coerce")
    m = m[(f > 0) & (f < 0.95) & (m["close"] > 0)]
    if len(m) < 20:
        return None, f"undetermined: only {len(m)} scored rows with a later split/bonus"
    adj = (m["log_px"] - np.log10(m["close"])).abs().median()
    raw = (m["log_px"] - np.log10(rp.raw_price(m, "close"))).abs().median()
    return bool(raw < adj), (f"log_px median |diff| vs log10(adjusted close) {adj:.4f}, vs log10(raw price) {raw:.4f} "
                             f"on {len(m)} scored rows with a later split/bonus")


# ------------------------------------------------------------------------------------- approaches
def rule(S: pd.DataFrame, broad: bool, top: int, gate: bool, breadth_prev: pd.Series) -> dict:
    W = (S if broad else S[S["core"]]).dropna(subset=["ind", "ret60"]).copy()
    W["n"] = W.groupby(["trade_date", "ind"])["ret60"].transform("size"); W = W[W["n"] >= 5]
    h = W.groupby(["trade_date", "ind"])["ret60"].mean().rename("h").reset_index()
    h["hot"] = h.groupby("trade_date")["h"].rank(pct=True) >= 0.9
    W = W.merge(h[["trade_date", "ind", "hot"]], on=["trade_date", "ind"])
    W["rk"] = W.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
    P = W[W["hot"] & (W["rk"] <= top) & (W["ret252"] > 0.5) & W["core"]]
    if gate:  # prior-session breadth; NaN (no prior session) = gate closed
        P = P[P["trade_date"].map(breadth_prev).ge(0.5)]
    return P.groupby("trade_date")["symbol"].apply(list).to_dict()


def build_approaches(S, breadth_prev, pit, allow_leaky):
    A = {f"{'+'.join(c) or 'BASELINE'}": rule(S, "H" in c, 10 if "B" in c else 3, "G" in c, breadth_prev) for c in RULES}
    T = S[S["core"]]
    if pit or allow_leaky:
        M = T.dropna(subset=["pred"]).sort_values("pred", ascending=False)
        A["MODEL top-10"] = M.groupby("trade_date").head(10).groupby("trade_date")["symbol"].apply(list).to_dict()
        A["MODEL top-20"] = M.groupby("trade_date").head(20).groupby("trade_date")["symbol"].apply(list).to_dict()
    A["EW market"] = T.groupby("trade_date")["symbol"].apply(list).to_dict()
    return A


# ------------------------------------------------------------------------------------- simulation
def _lookup(pos: np.ndarray, amt: np.ndarray, keys: np.ndarray) -> np.ndarray:
    out = np.zeros(len(keys))
    if pos.size and keys.size:
        i = np.clip(np.searchsorted(pos, keys), 0, len(pos) - 1)
        hit = pos[i] == keys
        out[hit] = amt[i[hit]]
    return out


def _catchup(D: Data, cols: np.ndarray, t0: int, open_exit: bool) -> np.ndarray:
    """Exit factor for positions that must be sold at session t0 but have no print there: sold at the first
    session they print again (its close, or its open in next_open mode) within CATCHUP_MAX sessions, else
    frozen at the last close (factor 1)."""
    f = np.ones(len(cols))
    n = len(D.cal)
    hi = min(t0 + CATCHUP_MAX, n - 1)
    if hi <= t0 or not len(cols):
        return f
    tr = D.TR[t0 + 1:hi + 1][:, cols]
    has, first = tr.any(axis=0), tr.argmax(axis=0)
    for m in np.flatnonzero(has):
        t, j = t0 + 1 + first[m], cols[m]
        if open_exit:
            f[m] = np.prod(1 + np.nan_to_num(D.R[t0:t, j])) * (1 + np.nan_to_num(D.ON[t, j]))
        else:
            f[m] = np.prod(1 + np.nan_to_num(D.R[t0 + 1:t + 1, j]))
    return f


def run_phase(picks: dict, seq: list, D: Data, h: int, entry: str, cost_mode: str):
    """One phase = one weekly start schedule. picks: session index -> sorted unique column indices.
    Returns (daily NAV at closes, base 1.0 at the first signal close; stats) or (None, None)."""
    n = len(D.cal)
    rot = [D.dpos[d] for d in seq if d in D.dpos]
    rot = [i for i in rot if i + 1 < n]
    if not rot:
        return None, None
    nxo = entry == "next_open"
    marks_i, marks_v = [rot[0]], [1.0]
    cash, pos, amt = 1.0, np.array([], int), np.array([], float)
    st = dict(n_rot=0, trade_frac=[], invested=[], uc_blocked=0, no_open=0, catchup=0, irregular=0)
    for k, i0 in enumerate(rot):
        nxt = rot[k + 1] if k + 1 < len(rot) else None
        end = min(i0 + h, n - 1) if nxt is None else min(nxt, n - 1)
        if nxt is not None and nxt != i0 + h:
            st["irregular"] += 1
        e = i0 + 1
        new = picks.get(i0, np.array([], int))
        # A. bring the held basket to the trade point (close of i0, or open of e)
        if pos.size:
            sold = ~np.isin(pos, new)
            if nxo:
                printed = D.TR[e, pos]
                fac = np.where(printed, 1 + np.nan_to_num(D.ON[e, pos]), 1.0)
                gone = sold & ~printed
                fac[gone] = _catchup(D, pos[gone], e, True)
            else:
                fac = np.ones(len(pos))
                gone = sold & ~D.TR[i0, pos]
                fac[gone] = _catchup(D, pos[gone], i0, False)
            st["catchup"] += int(gone.sum())
            amt = amt * fac
        nav = cash + amt.sum()
        # B. target amounts
        tgt = np.full(len(new), nav / len(new)) if len(new) else np.array([], float)
        if nxo and len(new):
            cur = _lookup(pos, amt, new)
            ok, lk = D.OPEN_OK[e, new], D.LOCK[i0, new]
            st["no_open"] += int((~ok & (cur < tgt)).sum())
            st["uc_blocked"] += int((ok & lk & (cur < tgt)).sum())
            tgt = np.where(~ok, cur, np.where(lk, np.minimum(tgt, cur), tgt))
        # C. costs
        u = np.union1d(pos, new)
        cur_u, tgt_u = _lookup(pos, amt, u), _lookup(new, tgt, u)
        rt = rp.cost_rt(pd.Series(D.ADV[i0, u])).to_numpy() if nxo else np.full(len(u), FLAT_RT)
        traded = np.abs(tgt_u - cur_u)
        cost = float((traded * rt / 2).sum()) if cost_mode == "turnover" else float((tgt_u * rt).sum())
        if nav > 0:
            st["trade_frac"].append(traded.sum() / nav)
            scale = (nav - cost) / nav
            tgt_u = tgt_u * scale
            cash = nav - cost - tgt_u.sum()
            st["invested"].append(tgt_u.sum() / max(nav - cost, 1e-12))
        keep = tgt_u > 0
        pos, amt = u[keep], tgt_u[keep]
        st["n_rot"] += 1
        # D. hold to `end`, marked at every close
        if pos.size:
            if nxo:
                first = 1 + np.where(D.OPEN_OK[e, pos], np.nan_to_num(D.ID[e, pos]), 0.0)
                G = first * np.vstack([np.ones(len(pos)), np.cumprod(1 + np.nan_to_num(D.R[e + 1:end + 1][:, pos]), axis=0)])
            else:
                G = np.cumprod(1 + np.nan_to_num(D.R[i0 + 1:end + 1][:, pos]), axis=0)
            vals = cash + (G * amt).sum(axis=1)   # not G @ amt: numpy 2 + Accelerate emits spurious matmul warnings
            amt = amt * G[-1]
        else:
            vals = np.full(end - i0, cash)
        marks_i.extend(range(i0 + 1, end + 1)); marks_v.extend(vals.tolist())
    # final liquidation at the last mark
    if pos.size:
        last = marks_i[-1]
        gone = ~D.TR[last, pos]
        fac = np.ones(len(pos)); fac[gone] = _catchup(D, pos[gone], last, False)
        amt = amt * fac
        if cost_mode == "turnover":
            rt = rp.cost_rt(pd.Series(D.ADV[last, pos])).to_numpy() if nxo else np.full(len(pos), FLAT_RT)
            cash -= float((amt * rt / 2).sum())
        marks_v[-1] = cash + amt.sum()
    nav = pd.Series(marks_v, index=D.cal[marks_i])
    nav = nav[~nav.index.duplicated(keep="last")]
    stats = dict(n_rot=st["n_rot"], trade_frac=float(np.median(st["trade_frac"])) if st["trade_frac"] else np.nan,
                 invested=float(np.mean(st["invested"])) if st["invested"] else 0.0, uc_blocked=st["uc_blocked"],
                 no_open=st["no_open"], catchup=st["catchup"], irregular=st["irregular"])
    return nav, stats


def _dd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min() * 100) if len(s) > 1 else np.nan


def phase_metrics(nav: pd.Series) -> dict:
    m = rp.nav_metrics(nav, era_split=str(ERA_SPLIT.date()))
    disc = nav[nav.index < ERA_SPLIT]
    conf = nav[nav.index >= ERA_SPLIT - pd.Timedelta(days=7)]      # same segment as rp.nav_metrics
    return dict(cagr=m["cagr"], maxdd=m["maxdd"], sharpe=m["sharpe"], cagr_disc=m["cagr_disc"], dd_disc=_dd(disc),
                cagr_conf=m["cagr_conf"], dd_conf=_dd(conf))


def fixed_offsets(n_ph: int) -> set[int]:
    return set(np.unique(np.round(np.linspace(0, n_ph - 1, min(N_FIXED, n_ph))).astype(int)).tolist())


def summarise(P: pd.DataFrame) -> list[dict]:
    out = []
    f6 = P[P["fixed6"]]
    for era, c, d in (("all", "cagr", "maxdd"), ("disc", "cagr_disc", "dd_disc"), ("conf", "cagr_conf", "dd_conf")):
        out.append(dict(era=era, cagr_med=P[c].median(), cagr_p10=P[c].quantile(0.1), cagr_worst_all=P[c].min(),
                        cagr_worst6=f6[c].min(), dd_med=P[d].median(), dd_worst_all=P[d].min(), dd_worst6=f6[d].min(),
                        sharpe_med=P["sharpe"].median() if era == "all" else np.nan,
                        trade_frac_med=P["trade_frac"].median(), invested_mean=P["invested"].mean(),
                        uc_blocked=int(P["uc_blocked"].sum()), no_open=int(P["no_open"].sum()),
                        catchup_exits=int(P["catchup"].sum()), phases_all=len(P), phases_fixed=len(f6)))
    return out


def extreme_moves(D: Data, cols: np.ndarray, start: str, top: int = 15) -> tuple[int, list]:
    """Single-session moves <= -45% or >= +100% on picked names since `start` (candidate unadjusted CAs)."""
    i_s = int(np.searchsorted(D.cal, pd.Timestamp(start)))
    Rp = D.R[i_s:, cols]
    hit = np.argwhere((Rp <= -0.45) | (Rp >= 1.0))
    rows = []
    for i, j in hit:
        t, c = i_s + i, cols[j]
        prior = np.flatnonzero(D.TR[:t, c])
        rows.append(dict(symbol=str(D.syms[c]), date=str(D.cal[t].date()), ret=round(float(Rp[i, j]), 3),
                         gap_sessions=int(t - prior[-1]) if prior.size else None))
    rows.sort(key=lambda r: -abs(np.log1p(r["ret"])))
    return len(rows), rows[:top]


# ------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="All-in x hold matrix (EXP-2026-09-27-allin-hold-matrix)")
    ap.add_argument("--entry", choices=["close", "next_open"], default="close",
                    help="close = registered (trade at the signal-day close, flat 0.5%% RT); "
                         "next_open = live contract (t+1 open, UC-locked buys skipped, ADV-scaled cost)")
    ap.add_argument("--cost", choices=["turnover", "full"], default="turnover",
                    help="turnover = charge only the weight that changes; full = registered RT on all capital per rotation")
    ap.add_argument("--holds", default=",".join(map(str, HOLDS)))
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--allow-leaky-model", action="store_true",
                    help="run MODEL arms even when pred is not point-in-time (labelled pred_pit=False)")
    ap.add_argument("--symbols", default="", help="comma list restricting rows + panel (smoke tests; needs --out-dir)")
    ap.add_argument("--panel", default="", help="panel parquet path (default research_panel.PANEL)")
    ap.add_argument("--out-dir", default="", help=f"output folder (default {OUT_DIR})")
    return ap.parse_args(argv)


def main(argv=None) -> pd.DataFrame:
    a = parse_args(argv)
    holds = [int(x) for x in a.holds.split(",") if x.strip()]
    windows = {w: WINDOWS[w] for w in a.windows.split(",") if w.strip()}
    syms = [s.strip() for s in a.symbols.split(",") if s.strip()] or None
    if syms and not a.out_dir:
        raise SystemExit("--symbols is a smoke-test subset: pass --out-dir so the registered outputs are not overwritten")
    if a.panel:
        rp.PANEL = Path(a.panel)
    out_dir = Path(a.out_dir) if a.out_dir else OUT_DIR
    tag = f"{a.entry}_{a.cost}"
    t0 = datetime.now()

    S, D, breadth_prev, pit, pit_note = load(a.entry, syms)
    APPROACHES = build_approaches(S, breadth_prev, pit, a.allow_leaky_model)
    model_status = ("included (pred point-in-time)" if pit else
                    "included WITH LEAK (--allow-leaky-model)" if a.allow_leaky_model else
                    "SKIPPED: pred not point-in-time; refit anatomy_1p5x on raw-basis log_px/pe or pass --allow-leaky-model")
    print(f"panel {rp.PANEL.name} | {len(D.cal)} sessions x {len(D.syms)} symbols | entry={a.entry} cost={a.cost}")
    print(f"MODEL arms: {model_status} [{pit_note}]")
    wk = sorted(S["trade_date"].unique())
    picked = sorted({D.col[s] for nm, pk in APPROACHES.items() if nm != "EW market" for v in pk.values() for s in v if s in D.col})
    n_ext, ext = extreme_moves(D, np.array(picked, int), min(windows.values())) if picked else (0, [])
    print(f"single-session moves <= -45% or >= +100% on rule/MODEL-picked names: {n_ext} (largest {len(ext)}; "
          f"check for unadjusted corporate actions)")
    for r in ext:
        print(f"   {r['symbol']:<14} {r['date']}  {r['ret']:+.1%}  gap {r['gap_sessions']} sessions")

    prow = []
    for window, start in windows.items():
        dates = [d for d in wk if d >= pd.Timestamp(start)]
        for name, pk in APPROACHES.items():
            if window == "2016+" and name.startswith("MODEL"):
                continue
            pki = {D.dpos[d]: np.unique(np.array([D.col[s] for s in v if s in D.col], int))
                   for d, v in pk.items() if d in D.dpos}
            for h in holds:
                step = max(1, h // 5)
                fixed = fixed_offsets(step)
                for ph in range(step):
                    seq = dates[ph::step]
                    nav, st = run_phase(pki, seq, D, h, a.entry, a.cost)
                    if nav is None or len(nav) < 3:
                        continue
                    prow.append(dict(window=window, approach=name, hold=h, phase=ph, start=str(seq[0].date()),
                                     fixed6=ph in fixed, **phase_metrics(nav), **st))
            print(f"done {window} {name}  [{(datetime.now() - t0).seconds}s]", flush=True)
    P = pd.DataFrame(prow)
    irregular = int(P["irregular"].sum()) if len(P) else 0
    if irregular:
        print(f"WARNING: {irregular} rotations where the next weekly date was not exactly h sessions later "
              "(weekly grid of rows.parquet vs panel calendar); held until the next rotation date")
    X = pd.DataFrame([dict(window=w, approach=ap_, hold=h, **r)
                      for (w, ap_, h), g in P.groupby(["window", "approach", "hold"], sort=False) for r in summarise(g)])
    X["pred_pit"] = np.where(X["approach"].str.startswith("MODEL"), str(pit), "")

    out_dir.mkdir(parents=True, exist_ok=True)
    fm, fp = out_dir / f"matrix_{tag}.csv", out_dir / f"phases_{tag}.csv"
    X.round(3).to_csv(fm, index=False)
    P.round(4).to_csv(fp, index=False)
    common = dict(experiment="EXP-2026-09-27-allin-hold-matrix", producer="src/agentic/sim_allin_matrix.py",
                  entry=a.entry, cost=a.cost, panel=str(rp.PANEL), rows_input=str(ROWS), symbols_subset=syms,
                  holds=holds, windows=windows, era_split=str(ERA_SPLIT.date()),
                  model_arms=model_status, pred_pit_check=pit_note,
                  audit="logs/audits/audit_20260927_research_code.json (fixes listed in the producer docstring)",
                  caveats=["industry labels are a static 2026 map (not point-in-time)",
                           "universe excludes securities with no PIT mcap (mostly delisted/renamed, 2016-2018 heavy)",
                           "names that never print again are frozen at their last close (no delisting mark-down)",
                           "unadjusted corporate actions in the panel appear as real moves across gaps",
                           "ret60/ret252 from rows.parquet are row-based (anatomy_1p5x)",
                           "lower-circuit-locked exits not modelled"],
                  extreme_moves_on_picks=dict(count=n_ext, largest=ext),
                  updated=datetime.now().isoformat(timespec="seconds"))
    units = "CAGR, drawdown in percent (15.0 = 15%); trade_frac, invested are fractions of NAV"
    Path(str(fm) + ".manifest.json").write_text(json.dumps(dict(
        dataset="all-in hold matrix summary", file=fm.name, rows=len(X), units=units, **common,
        columns=dict(window="evaluation window (2019+ from 2019-01-01, 2016+ from 2016-06-01)", approach="pick rule",
                     hold="sessions held per rotation", era="all = whole window; disc = sessions < 2023-01-01; "
                     "conf = sessions >= 2023-01-01 (cut from the same NAV path)",
                     cagr_med="median CAGR % over all phase offsets", cagr_p10="10th percentile CAGR % over all phases",
                     cagr_worst_all="min CAGR % over all h//5 phases (count differs by hold)",
                     cagr_worst6="min CAGR % over the fixed 6 evenly spaced phases (comparable across holds)",
                     dd_med="median max drawdown % (daily NAV)", dd_worst_all="min max-drawdown % over all phases",
                     dd_worst6="min max-drawdown % over the fixed 6 phases", sharpe_med="median daily Sharpe (era=all only)",
                     trade_frac_med="median over phases of the median per-rotation traded weight sum|dw| (2.0 = full rotation)",
                     invested_mean="mean invested fraction after each rotation's trades",
                     uc_blocked="picks not bought because upper-circuit locked at the entry open (next_open)",
                     no_open="picks with no print at the entry open (next_open)",
                     catchup_exits="sales delayed to the first session the name printed again",
                     phases_all="number of phase offsets (h//5)", phases_fixed="phases in the fixed-6 set",
                     pred_pit="MODEL rows: pred passed the point-in-time log_px check")), indent=1, default=str))
    Path(str(fp) + ".manifest.json").write_text(json.dumps(dict(
        dataset="all-in hold matrix per-phase results", file=fp.name, rows=len(P), units=units, **common,
        columns=dict(window="evaluation window", approach="pick rule", hold="sessions per rotation", phase="weekly offset",
                     start="first rotation date", fixed6="in the fixed 6-phase set", cagr="CAGR % whole window",
                     maxdd="max drawdown %", sharpe="daily Sharpe", cagr_disc="CAGR % sessions < 2023",
                     dd_disc="max drawdown % < 2023", cagr_conf="CAGR % >= 2023", dd_conf="max drawdown % >= 2023",
                     n_rot="rotations", trade_frac="median per-rotation traded weight (2.0 = full rotation)",
                     invested="mean invested fraction", uc_blocked="UC-locked picks not bought", no_open="picks with no open print",
                     catchup="delayed (gap) exits", irregular="rotations not exactly h sessions apart")), indent=1, default=str))
    if not syms:
        write_readme(out_dir)

    order = list(APPROACHES)
    for window in windows:
        for era in ("all", "disc", "conf"):
            W = X[(X["window"] == window) & (X["era"] == era)]
            if W.empty:
                continue
            for metric, lab in (("cagr_med", "MEDIAN CAGR %"), ("cagr_worst6", "WORST-OF-6-PHASES CAGR %"),
                                ("dd_med", "MEDIAN MAX DRAWDOWN %")):
                print(f"\n=== {window} · era {era} · {lab} · entry={a.entry} cost={a.cost} (rows = approach, cols = hold) ===")
                print(W.pivot(index="approach", columns="hold", values=metric)
                      .reindex([o for o in order if o in set(W["approach"])]).round(1).to_string())
    print(f"\nwrote {fm} and {fp} (+ manifests)")
    print("ALLIN MATRIX COMPLETE")
    return X


def write_readme(out_dir: Path) -> None:
    files = sorted(p.name for p in out_dir.glob("*.csv"))
    (out_dir / "README.md").write_text(
        "# All-in x hold matrix (EXP-2026-09-27-allin-hold-matrix)\n\n"
        "Producer: `src/agentic/sim_allin_matrix.py` (2026-09-27 audit fixes are listed in its docstring).\n"
        "100% of capital rotates every h sessions into that week's picks, equal weight; empty picks = cash.\n\n"
        "Files: `matrix_<entry>_<cost>.csv` (summary, one row per window x approach x hold x era) and\n"
        "`phases_<entry>_<cost>.csv` (every phase offset). Each has a `.manifest.json` with columns and units.\n\n"
        "- entry `close`: registered mode, trade at the signal-day close, flat 0.5% round trip.\n"
        "- entry `next_open`: buy/sell at the next session's open, upper-circuit-locked buys skipped, cost by ADV.\n"
        "- cost `turnover`: only the weight that changes pays; `full`: registered 0.5% on all capital per rotation.\n"
        "- era: `all`, `disc` (< 2023) and `conf` (2023+), cut from the same daily NAV. Use `cagr_worst6` /\n"
        "  `dd_worst6` to compare worst phases across holds (6 phases each); `*_worst_all` is over h//5 phases.\n"
        "- MODEL arms run only when rows.parquet `pred` passes the point-in-time price check (see manifest).\n\n"
        "Known limits: static 2026 industry labels; no-mcap securities absent; delistings frozen at last close;\n"
        "unadjusted corporate actions in the panel show up as real moves.\n\n"
        f"Present: {', '.join(files)}\n\n"
        "`../allin_matrix.csv` is the pre-audit output (gap moves zeroed, adjusted-price core, same-day G);\n"
        "do not use it.\n")


if __name__ == "__main__":
    main()
