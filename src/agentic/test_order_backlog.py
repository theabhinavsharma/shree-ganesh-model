"""ORDER BACKLOG / PIT TTM REVENUE (EXP-2026-09-28-order-backlog; registered in logs/experiments.jsonl before any run).

Question: does a large stated order book (backlog, in years of revenue) raise the chance that a stock touches +50%
within 95 sessions? The earlier order tests sized individual order filings (amount / revenue). Decks, transcripts and
results press releases state the whole book, which is the better measure of revenue visibility.

Inputs
  data/derived/order_book_filings.parquet       fetch_order_book.py: backlog_cr per filing, up to 5 documents per company-quarter
  data/derived/pnl_quarterly.parquet            PIT TTM revenue via test_hot_order_combo.build_ttm (manifest units)
  logs/leader_sleeve/anatomy_1p5x/rows.parquet  weekly rows (ind_heat_pct, px_sma200); labels recomputed on the price panel

Registered definitions
  statement  a filing with backlog_cr >= 1 cr, usable from its effective session (filed before 15:30 IST on a session ->
             that session, else the next one).
  book_pit   median of the symbol's statements with effective session in (eff - 45 days, eff]: a deck, its transcript and
             the press release restate one book; the median of what was public by then, never a later filing.
  ratio      book_pit / PIT TTM revenue known at the filing time = years of revenue in hand. ratio > 20 is treated as an
             extraction error (group or segment book, unit mix-up): excluded and counted.
  row value  the latest statement with effective session within the 120 days up to weekly row t; else unknown.
  growth     book_pit / the symbol's latest book_pit with effective session 300-430 days earlier.
  outcome    y95 = max high over the next 95 sessions >= 1.5x the close at t; also s95 (close at +95 >= 1.5x) and fc95.
  eras       disc = first week with >= 50% PIT TTM coverage .. 2022-12-31; conf = 2023-01-01 .. last labelled week.
  base       P(y95) of rows with a PIT TTM revenue in that era.
H1 (primary) HIGH = ratio >= 2 & close > SMA200 vs LOW = ratio < 1 & close > SMA200.
             PASS iff in BOTH eras: P(HIGH) / base >= 1.3, the 95% symbol-clustered bootstrap CI of P(HIGH) - P(LOW) has a
             lower bound > 0, and HIGH spans >= 30 symbols. Otherwise FAIL. Reported as is, no retuning.
Secondary (descriptive, no pass rule): ratio buckets; hot (ind_heat_pct >= 0.9) & >200DMA split by ratio; book growth
             >= 1.25 vs < 1; episode view (first weekly row after each statement, one per statement).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
from test_hot_order_combo import (COV_MIN, CUTOFF_MIN, ERA_SPLIT, PNL, PNL_UNIT_TO_CR, ROWS, asof_ttm, boot_means,  # noqa: E402
                                  build_ttm, ci, effective_session, panel_labels)

EXP_ID = "EXP-2026-09-28-order-backlog"
BOOK = ROOT / "data/derived/order_book_filings.parquet"
OUT = ROOT / "logs/leader_sleeve/order_backlog_test.json"
MIN_CR, LOOK_D, STALE_D, RATIO_MAX = 1.0, 45, 120, 20.0
GROW_LO_D, GROW_HI_D = 300, 430
HI, LO, LIFT_MIN, NSYM_MIN = 2.0, 1.0, 1.3, 30
BUCKETS = [0, 0.5, 1, 2, 3, np.inf]
BOOK_IN = BOOK
H_KEY, L_KEY = "HIGH = ratio >= 2 & >200DMA", "LOW = ratio < 1 & >200DMA"


def boot_diff(yA: np.ndarray, gA: np.ndarray, yB: np.ndarray, gB: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    """Symbol-clustered bootstrap of mean(A) - mean(B), both cells resampled with the same draw of symbols."""
    if len(yA) == 0 or len(yB) == 0:
        return np.full(B, np.nan)
    codes, uniq = pd.factorize(pd.Series(np.concatenate([gA, gB])))
    G = len(uniq)
    a, b = codes[:len(gA)], codes[len(gA):]
    sA, nA = np.bincount(a, weights=yA, minlength=G), np.bincount(a, minlength=G).astype(float)
    sB, nB = np.bincount(b, weights=yB, minlength=G), np.bincount(b, minlength=G).astype(float)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    with np.errstate(all="ignore"):
        return (W @ sA) / (W @ nA) - (W @ sB) / (W @ nB)


def build_ttm_by_basis() -> pd.DataFrame:
    """NOT REGISTERED - sensitivity added 2026-09-28 after the pre-run review: build_ttm keeps one basis per symbol by quarter
    count, so ~1/3 of ratios used standalone revenue against decks that state a consolidated book. Same rules as build_ttm
    (as-filed values, 4 consecutive quarters, available at the last filing, stale late filings never replace a newer TTM),
    kept per (symbol, basis); ttm_con_first() then takes the consolidated TTM when one is known, else standalone."""
    q = pd.read_parquet(PNL, columns=["symbol", "quarter_end", "filing_dt", "basis", "source", "net_sales"])
    q = q.dropna(subset=["filing_dt", "net_sales"])
    q["sales_cr"] = q["net_sales"] * q["source"].map(PNL_UNIT_TO_CR)
    q = q.sort_values("filing_dt", kind="mergesort").drop_duplicates(["symbol", "quarter_end", "basis"], keep="first")
    q = q.sort_values(["symbol", "basis", "quarter_end"]).reset_index(drop=True)
    key = [q["symbol"], q["basis"]]
    qn = q["quarter_end"].dt.year * 4 + q["quarter_end"].dt.quarter
    consec = (qn - qn.groupby(key).shift(3)) == 3
    q["rev_ttm_cr"] = q.groupby(key)["sales_cr"].transform(lambda x: x.rolling(4).sum()).where(consec)
    fsec = (q["filing_dt"].astype("datetime64[ns]").astype("int64") // 10**9).astype(float)
    with np.errstate(all="ignore"):   # seconds * 1e9 cannot overflow; macOS Accelerate leaves spurious FP flags (see boot_means)
        q["avail"] = pd.to_datetime(fsec.groupby(key).transform(lambda x: x.rolling(4).max()), unit="s").astype("datetime64[ns]")
    T = q.loc[q["rev_ttm_cr"] > 0, ["symbol", "basis", "quarter_end", "avail", "rev_ttm_cr"]].dropna()
    T = T.sort_values(["symbol", "basis", "avail", "quarter_end"], kind="mergesort")
    T = T[T["quarter_end"] >= T.groupby(["symbol", "basis"])["quarter_end"].cummax()]
    return T.sort_values(["avail", "quarter_end"], kind="mergesort").reset_index(drop=True)


def ttm_con_first(TB: pd.DataFrame):
    con = TB[TB["basis"] == "con"].reset_index(drop=True)
    sa = TB[TB["basis"] == "sa"].reset_index(drop=True)
    return lambda left, when: asof_ttm(left, when, con).fillna(asof_ttm(left, when, sa))


def statements(cal: pd.DatetimeIndex, ttm_at) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backlog statements -> PIT consensus book, ratio to PIT TTM revenue, growth. Returns (events, coverage table)."""
    O = pd.read_parquet(BOOK_IN)
    O["filed"] = pd.to_datetime(O["filed"], errors="coerce")
    fy = O["filed"].dt.year
    cov = pd.DataFrame({"filings_read": O.groupby(fy).size(), "backlog_stated": O["backlog_cr"].notna().groupby(fy).sum()})
    O = O[O["backlog_cr"] >= MIN_CR].dropna(subset=["filed"]).copy()
    O["eff"] = effective_session(O["filed"], cal).astype("datetime64[ns]")
    O = O.dropna(subset=["eff"]).sort_values(["symbol", "eff", "filed"], kind="mergesort").reset_index(drop=True)
    bp = np.full(len(O), np.nan)
    eff, val = O["eff"].values, O["backlog_cr"].values.astype(float)
    for _, idx in O.groupby("symbol").indices.items():
        e, v = eff[idx], val[idx]
        lo = np.searchsorted(e, e - np.timedelta64(LOOK_D, "D"), side="right")
        hi = np.searchsorted(e, e, side="right")                   # every statement usable at that same close
        bp[idx] = [np.median(v[l:h]) for l, h in zip(lo, hi)]
    O["book_pit"] = bp
    O["rev_ttm_cr"] = ttm_at(O, "filed")
    O["ratio"] = O["book_pit"] / O["rev_ttm_cr"]
    y = O["eff"].dt.year
    cov["with_ttm"] = O["ratio"].notna().groupby(y).sum()
    cov["ratio_gt_20_excluded"] = (O["ratio"] > RATIO_MAX).groupby(y).sum()
    E = O[(O["ratio"] > 0) & (O["ratio"] <= RATIO_MAX)].drop_duplicates(["symbol", "eff"], keep="last").reset_index(drop=True)
    cov["events_used"] = E.groupby(E["eff"].dt.year).size()
    cov["companies"] = E.groupby(E["eff"].dt.year)["symbol"].nunique()
    g = np.full(len(E), np.nan)
    eff, book = E["eff"].values, E["book_pit"].values
    for _, idx in E.groupby("symbol").indices.items():
        e, v = eff[idx], book[idx]
        lo = np.searchsorted(e, e - np.timedelta64(GROW_HI_D, "D"), side="left")
        hi = np.searchsorted(e, e - np.timedelta64(GROW_LO_D, "D"), side="right")
        ok = hi > lo
        g[idx[ok]] = v[ok] / v[hi[ok] - 1]
    E["growth"] = g
    return E, cov.fillna(0).astype(int)


def stats(X: pd.DataFrame, base: float) -> dict:
    p = X["y95"].mean()
    return dict(n=len(X), n_sym=int(X["symbol"].nunique()), p=p * 100, sus=X["s95"].mean() * 100,
                ret=(X["fc95"].mean() - 1) * 100, lift=p / base if base else np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry", action="store_true", help="smoke run: do not append the result to logs/experiments.jsonl")
    ap.add_argument("--ttm", choices=["registered", "con-first"], default="registered",
                    help="con-first = NOT REGISTERED sensitivity: consolidated TTM when known, else standalone")
    ap.add_argument("--book", default=str(BOOK), help="order_book_filings parquet (default: the current producer output)")
    ap.add_argument("--tag", default="", help="label for a non-registered rerun: RESULT id suffix and default output name")
    args = ap.parse_args()
    global BOOK_IN
    BOOK_IN = Path(args.book) if Path(args.book).is_absolute() else ROOT / args.book
    t0 = time.time()
    rng = np.random.default_rng(20260928)

    px = rp.load_panel(["high", "low", "close"])
    cal = rp.session_calendar(px)
    close_w, high_w, low_w = (rp.wide(px, c, cal) for c in ("close", "high", "low"))
    del px
    if args.ttm == "registered":
        T, _ = build_ttm(None)
        ttm_at = lambda left, when: asof_ttm(left, when, T)  # noqa: E731
    else:
        ttm_at = ttm_con_first(build_ttm_by_basis())
    E, cov = statements(cal, ttm_at)
    print(f"panel {len(cal)} sessions {cal[0].date()}..{cal[-1].date()} · {time.time() - t0:.0f}s\n"
          f"backlog statements by year (effective session):\n{cov.T.to_string()}")
    top = E.nlargest(8, "ratio")[["symbol", "eff", "book_pit", "rev_ttm_cr", "ratio"]]
    print("largest ratios kept (eyeball for group/segment books):\n" + top.to_string(index=False))

    S = pd.read_parquet(ROWS, columns=["symbol", "trade_date", "ind_heat_pct", "px_sma200"]).reset_index(drop=True)
    S["trade_date"] = pd.to_datetime(S["trade_date"]).astype("datetime64[ns]")
    lab, label_end = panel_labels(S, close_w, high_w, low_w)
    S[["y95", "s95", "fc95"]] = lab
    del close_w, high_w, low_w
    S["asof"] = S["trade_date"] + pd.Timedelta(minutes=CUTOFF_MIN - 1)
    S["covered"] = ttm_at(S, "asof").notna()
    Lft = S[["symbol", "trade_date"]].reset_index().sort_values("trade_date", kind="mergesort")
    R = E[["symbol", "eff", "ratio", "growth"]].rename(columns={"eff": "bl_eff"}).sort_values("bl_eff", kind="mergesort")
    m = pd.merge_asof(Lft, R, left_on="trade_date", right_on="bl_eff", by="symbol", direction="backward",
                      tolerance=pd.Timedelta(days=STALE_D)).set_index("index").reindex(S.index)
    S[["bl_eff", "ratio", "growth"]] = m[["bl_eff", "ratio", "growth"]]

    cov_wk = S.groupby("trade_date")["covered"].mean()
    hit = cov_wk[cov_wk >= COV_MIN]
    if hit.empty:
        raise SystemExit(f"PIT TTM coverage never reaches {COV_MIN:.0%} of anatomy rows")
    cov_start = pd.Timestamp(hit.index[0])
    L = S.dropna(subset=["y95"])
    era = np.where(L["trade_date"] >= ERA_SPLIT, "conf", np.where(L["trade_date"] >= cov_start, "disc", ""))
    L = L.assign(era_w=era)[era != ""]
    up, hot, c_ = L["px_sma200"] > 0, L["ind_heat_pct"] >= 0.9, L["covered"]
    r, g = L["ratio"], L["growth"]
    cells = {H_KEY: c_ & up & (r >= HI), L_KEY: c_ & up & (r < LO),
             "stated book, any ratio": c_ & r.notna(), ">200DMA, no stated book": c_ & up & r.isna(),
             "hot & >200DMA & ratio >= 2": c_ & hot & up & (r >= HI), "hot & >200DMA & ratio < 1": c_ & hot & up & (r < LO),
             "hot & >200DMA & no stated book": c_ & hot & up & r.isna(),
             "growth >= 1.25 & >200DMA": c_ & up & (g >= 1.25), "growth < 1 & >200DMA": c_ & up & (g < 1)}
    for lo_, hi_ in zip(BUCKETS[:-1], BUCKETS[1:]):
        cells[f"ratio [{lo_}, {hi_})"] = c_ & (r >= lo_) & (r < hi_)
    windows = {"disc": f"{cov_start.date()} .. 2022-12-31", "conf": f"2023-01-01 .. {L['trade_date'].max().date()}"}
    res: dict = dict(id=EXP_ID + (f"-{args.tag}" if args.tag else ""), ttm=args.ttm, book=str(BOOK_IN.relative_to(ROOT)), windows=windows, label_end=str(label_end.date()), coverage_by_year=cov.to_dict(orient="index"),
                     cells={}, h1={})
    print(f"\neras: disc {windows['disc']} · conf {windows['conf']} (labels through {label_end.date()})")
    base = {}
    for e in ("disc", "conf"):
        X = L[(L["era_w"] == e) & c_]
        base[e] = X["y95"].mean()
        res["cells"][f"base|{e}"] = dict(n=len(X), n_sym=int(X["symbol"].nunique()), p=base[e] * 100,
                                         stated_share=float(X["ratio"].notna().mean() * 100))
        print(f"  base {e}: P(+50% in 95) {base[e] * 100:.1f}% n={len(X):,} · rows with a stated book "
              f"{X['ratio'].notna().mean() * 100:.1f}%")
    print(f"\n{'cell':34s} {'era':4s} {'n':>7s} {'syms':>5s} {'P+50':>6s} {'lift':>5s} {'sus':>6s} {'ret95':>6s}")
    for k, msk in cells.items():
        for e in ("disc", "conf"):
            st = stats(L[msk & (L["era_w"] == e)], base[e])
            res["cells"][f"{k}|{e}"] = st
            print(f"{k:34s} {e:4s} {st['n']:7,d} {st['n_sym']:5d} {st['p']:6.1f} {st['lift']:5.2f} {st['sus']:6.1f} {st['ret']:6.1f}")
    # episode view: the first weekly row after each statement, one per statement
    ep = L[L["bl_eff"].notna()].sort_values("trade_date", kind="mergesort").drop_duplicates(["symbol", "bl_eff"])
    for k, cond in ((H_KEY, (ep["px_sma200"] > 0) & (ep["ratio"] >= HI)), (L_KEY, (ep["px_sma200"] > 0) & (ep["ratio"] < LO))):
        for e in ("disc", "conf"):
            st = stats(ep[cond & ep["covered"] & (ep["era_w"] == e)], base[e])
            res["cells"][f"episodes: {k}|{e}"] = st
            print(f"{'episodes: ' + k[:24]:34s} {e:4s} {st['n']:7,d} {st['n_sym']:5d} {st['p']:6.1f} {st['lift']:5.2f} {st['sus']:6.1f} {st['ret']:6.1f}")

    ok_all = True
    print("\nH1 (registered): P(HIGH)/base >= 1.3, clustered CI of P(HIGH) - P(LOW) above 0, HIGH >= 30 symbols, both eras")
    for e in ("disc", "conf"):
        A, Bc = L[cells[H_KEY] & (L["era_w"] == e)], L[cells[L_KEY] & (L["era_w"] == e)]
        d = boot_diff(A["y95"].to_numpy(float), A["symbol"].to_numpy(), Bc["y95"].to_numpy(float), Bc["symbol"].to_numpy(), rng, args.boot)
        pa = boot_means(A["y95"].to_numpy(float), A["symbol"].to_numpy(), rng, args.boot)
        lift = A["y95"].mean() / base[e] if len(A) else np.nan
        dci = ci(d * 100)
        ok = bool(np.isfinite(lift) and lift >= LIFT_MIN and np.isfinite(dci[0]) and dci[0] > 0 and A["symbol"].nunique() >= NSYM_MIN)
        ok_all &= ok
        res["h1"][e] = dict(p_high=A["y95"].mean() * 100, p_low=Bc["y95"].mean() * 100, lift=lift,
                            diff=(A["y95"].mean() - Bc["y95"].mean()) * 100, diff_ci=dci, p_high_ci=ci(pa * 100),
                            n_sym_high=int(A["symbol"].nunique()), n_sym_low=int(Bc["symbol"].nunique()), pass_=ok)
        h = res["h1"][e]
        print(f"  {e}: HIGH {h['p_high']:.1f}% (CI {h['p_high_ci'][0]:.1f}-{h['p_high_ci'][1]:.1f}, {h['n_sym_high']} syms) vs "
              f"LOW {h['p_low']:.1f}% ({h['n_sym_low']} syms) · diff {h['diff']:+.1f} pts CI {dci[0]:+.1f}..{dci[1]:+.1f} · "
              f"lift {lift:.2f} -> {'pass' if ok else 'fail'}")
    res["verdict"] = "PASS" if ok_all else "FAIL"
    print(f"\nVERDICT H1: {res['verdict']}  ({time.time() - t0:.0f}s)")

    out = Path(args.out) if args.out != str(OUT) or not args.tag else OUT.with_name(f"{OUT.stem}_{args.tag}.json")
    out.write_text(json.dumps(res, indent=1, default=str))
    out.with_name(out.name + ".manifest.json").write_text(json.dumps(dict(
        dataset="order_backlog_test", path=str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
        producer="src/agentic/test_order_backlog.py", experiment=EXP_ID,
        inputs=[str(BOOK_IN.relative_to(ROOT)), "data/derived/pnl_quarterly.parquet", str(ROWS.relative_to(ROOT)), str(rp.PANEL.relative_to(ROOT))],
        units=dict(p="percent of weekly rows whose max high over the next 95 sessions reached 1.5x the close",
                   sus="percent whose close at +95 sessions was >= 1.5x", ret="mean close-to-close return over 95 sessions, percent",
                   lift="P(cell) / P(base) in the same era", ratio="stated order book / PIT TTM revenue, years",
                   diff_ci="95% symbol-clustered bootstrap CI, percentage points"),
        definitions=__doc__, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    if not args.dry:
        with (ROOT / "logs/experiments.jsonl").open("a") as fh:
            fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + "-RESULT" + (f"-{args.tag}" if args.tag else ""), registered=not args.tag,
                                     ttm=args.ttm, verdict=res["verdict"],
                                     h1=res["h1"], windows=windows, out=str(out)), default=str) + "\n")


if __name__ == "__main__":
    main()
