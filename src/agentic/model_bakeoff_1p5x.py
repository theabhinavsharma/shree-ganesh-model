"""MODEL BAKE-OFF for the 1.5x-in-95-sessions target (companion to EXP-2026-09-27-1p5x-anatomy).

Question (user 2026-09-27): is LightGBM the right tool, or is something more powerful needed?
Same rows (logs/leader_sleeve/anatomy_1p5x/rows.parquet) and the same model features (the rows manifest's
`model_features`, else `features`), taken as-is from the anatomy rows (nothing is re-derived here).
Walk-forward: predict test year Y from labelled rows whose LABEL WINDOW ends before the first session of Y.
The window end is measured from the panel, not assumed. Session-counted labels (the fixed anatomy,
research_panel.forward_window) end at session t+95 of the GLOBAL calendar. Row-counted labels (the pre-fix
anatomy: the symbol's next 95 panel ROWS, which stretch over gaps) end at the symbol's 95th next panel row.
--label-basis auto finds out which definition the rows carry by recomputing fh95 from the panel both ways on
rows whose two windows differ. Session labels are purged at t+95; row-counted or unverifiable labels are purged
on max(session end, row end) and the run refuses unless --allow-stale-labels (result flagged stale).
Contenders:
  logit      standardised logistic regression (median-imputed)            — linear baseline
  lgbm       LightGBM binary (same params as the anatomy run)
  xgb        XGBoost hist binary
  mlp        sklearn MLP 128-64, quantile-transformed inputs              — neural-net proxy. Epoch count chosen
             on a TIME-ordered validation block (last 15% of training dates; fit rows' measured label windows
             end before it; criterion = mean within-week AUC), then refit on the full training subsample for
             that many epochs
  lgbm_rank  LightGBM LambdaRank, one query per week                      — optimises within-week order
  ensemble   mean of within-week (per trade_date) percentile ranks of lgbm, xgb, mlp
Training subsample for logit/mlp: 300k rows (speed); tree models use all rows.
Metrics, per era (disc = 2019-2022 test years, conf = 2023+) and per test year:
  wk_auc, wk_top1   AUC and top-1% precision computed WITHIN each week (trade_date) over all labelled rows,
                    averaged over weeks; wk_top1_lift = mean weekly top-1% precision / mean weekly base rate
  wk10              weekly top-10 TRADABLE hit rate — the number a portfolio actually trades. Tradable =
                    ADV >= Rs5cr and THEN-TRADED close > Rs50 (research_panel.raw_price). Picks are chosen
                    point-in-time among ALL tradable rows of the week (labelled or not); picks without a label
                    are counted and reported with lower/upper bounds (unknown = miss / unknown = hit).
                    hit_ex_long_span = the same picks' hit rate leaving out picks whose symbol's 95-row window
                    spans >= 105 sessions (~150 calendar days): inflated labels when the rows are row-counted,
                    lower-bound labels (missing sessions) when they are session-counted.
  --entry close      registered mode: entry at the signal-day close; hit = y95.
  --entry next_open  live-contract mode (research_panel.next_open_entry): entry at the open of the next REAL
                     session (copied holiday sessions skipped, see copied_sessions); names upper-circuit locked at
                     that open, or not trading that session, are skipped and the next-ranked name takes the slot;
                     hit = fh95 x close_t / open_t+1 >= 1.5 (same label window as y95, which starts with the entry
                     session). Default --entry both reports the two side by side.
                    hit_ex_copied_session = the same picks leaving out weeks dated on a copied session (their
                    features are the prior session's copy, with that session counted twice in rolling windows).
  Costs (research_panel.cost_rt) do not enter a hit-rate metric; this script produces no NAV.
Outputs (--out-dir, default logs/leader_sleeve/anatomy_1p5x/):
  bakeoff.json                         metrics per model x era and model x year, fold details, checks, coverage
  bakeoff_preds.parquet (+manifest)    per-model out-of-sample scores for every test row (2019+)
  stdout log

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json, file model_bakeoff_1p5x.py)
  PARTIAL lines 5, 34, 48-49 (LOOKAHEAD, high).
          FIXED here, train/test leakage: the fixed `Y-01-01 minus 150 days` cut is replaced by a purge on each
          training row's MEASURED label-window end (label_window_ends + detect_label_basis, see above), which is
          exact for row-counted labels built on the given panel and for session-counted labels. On the current
          (row-counted) rows with the .bak-2026-09-27 panel, training rows whose 95-row window reaches Y, folds
          2019..2026: old 150-day cut 850 / 1,113 / 938 / 2,204 / 1,647 / 4,441 / 5,035 / 314 (the audit's
          counts); a session-only purge (first review round) 921 / 1,227 / 1,188 / 2,421 / 1,871 / 4,716 /
          5,427 / 330, i.e. MORE; the applied purge 0 in every fold. The MLP's inner validation purge uses the
          same measured end. Metrics score only weeks whose full 95-session window lies inside the rows' span.
          NOT fixed here: the label ITSELF. Row-counted y95 stretched over the pre-2025 BE/BZ gaps is built in
          anatomy_1p5x.py (fixed there; rerun it). Until then the TEST side is inflated as well (long-span labels,
          worst in 2017/2021/2023/2024). The run therefore refuses without --allow-stale-labels, prints and saves
          the long-span share and y95 of labelled rows by year, and reports every weekly top-10 hit rate also
          without the long-span picks (hit_ex_long_span).
  FIXED   lines 82-83, 86-87 (STATISTICAL VALIDITY, medium): pooled per-era AUC / top-1% replaced by within-week
          AUC and top-1% precision averaged over weeks (valid for lgbm_rank, whose scores have no cross-week level).
          Per-year breakdowns are reported too.
  FIXED   lines 61-64 (STATISTICAL VALIDITY, medium): MLP early stopping used a RANDOM 10% of rows scored on
          accuracy, so label-overlapping neighbours leaked and it never stopped. Now: time-ordered validation block
          (last 15% of training dates), fit rows' measured label windows end before it, mean within-week AUC,
          patience 5, refit at best epoch.
  FIXED   line 75 (LOOKAHEAD, low): the ensemble ranked scores over all 2019-2026 test rows pooled; it now ranks
          within each trade_date (point-in-time).
  FIXED   lines 84-86 (SURVIVORSHIP, medium): TRADABLE used the back-adjusted close (`core`, close > 50 on a price
          that encodes later splits/bonuses). Tradable is now ADV >= 5cr & raw_price(close) > 50, joined from the
          panel; the rows' own `core` flag is ignored (its disagreement rate is reported in checks).
  PARTIAL lines 34, 49, 84-85 (SURVIVORSHIP, low): the weekly top-10 is now chosen among ALL tradable rows of the
          week, so an unlabelled name (rename, merger, suspension) is no longer silently replaced by the next
          name; unlabelled picks are counted and bounded. NOT fixed here: giving renamed symbols the successor's
          outcome (label stitching) and the NaN `ind` of dead symbols — both belong to anatomy_1p5x.py.
  PARTIAL lines 31-32 (LOOKAHEAD/UNITS, medium): log_px, pe, pe_ind come from the anatomy rows and are not
          re-derived here (the fix is in anatomy_1p5x.py). GUARD: whenever the rows HAVE a log_px column (model
          feature or not, so --drop-features log_px cannot skip it) 10**log_px is compared with the then-traded
          close. The three share one price basis, so if the rows are on the back-adjusted basis and ANY of
          log_px / pe / pe_ind is a model feature the run refuses unless --allow-stale-features (then flagged
          stale in bakeoff.json and the preds manifest). pe/pe_ind cannot be checked directly without EPS.
          FEATS come from the rows manifest (`model_features`, then `features`, then `columns` minus
          non-features minus `not_model_features`; not_model_features is saved in bakeoff.json) with a
          forward-column name guard, instead of "every non-META column".
  Critic item 2 (feature coverage by era, "no data" scored as "no event"): NOT fixed here (feature construction
          is anatomy's). The fixed anatomy's manifest `features` list applies its coverage rule, and this script
          now reads that list, so era-marker features stay out once anatomy is rerun. On the current rows (no
          list) the non-null and non-zero rate of every feature per year is reported so era-marker features are
          visible, and --drop-features lets a run exclude them.
  Reviewer round 3 (defect in the PANEL and in research_panel.session_calendar / next_open_entry, not in this
          file's logic; fixed here for this file's use, the source fix belongs to the panel / research_panel):
          the panel carries copied non-trading sessions, rows whose trade_date_source is an EARLIER genuine
          session (the prior bhavcopy re-filed under a holiday). On the .bak-2026-09-27 panel: 75 dates, 140,428
          rows, 127,610 of them equities (dates by year 2020: 10, 2021: 13, 2022: 11, 2023: 15, 2024: 13,
          2025: 2, 2026: 11), e.g. 2021-03-11 = the 2021-03-10 file. The reviewer's 79 dates include 4 that are
          REAL special sessions filed under the next weekday (2020-11-16 <- Muhurat 2020-11-14, 2021-11-05 <-
          Muhurat 2021-11-04, 2024-01-22 <- 2024-01-20, 2024-05-20 <- 2024-05-18: own prices, low volume); kept.
          In-sample test years only ("test weeks" = rows-file weeks >= 2019).
  FIXED   next-open entry (panel_joins): research_panel.next_open_entry took the open of calendar session t+1,
          which for the 13 test weeks before a copied session (2021-03-10, 2021-05-12, 2021-08-18, 2021-11-03,
          2022-04-13, 2023-04-06, 2023-04-13, 2024-01-25, 2024-03-07, 2024-03-28, 2026-03-02, 2026-03-30,
          2026-04-13) was the signal day's OWN open (lookahead), and its upper-circuit check compared day t with
          itself. Copied rows are now dropped before the wide frames, the entry is the next REAL session's open,
          and keys dated on a copied session are taken as-of the last real session. INFY 2021-03-10: entry was
          1356.90 (the 03-10 open), now 1373.00 (2021-03-12 open). Over the rows file's 21,621 rows on those 13
          weeks the old entry equalled the signal day's own open in 100% of cases; mean y95_next_open moves
          27.4% -> 26.6% (844 labels flip) and 43 upper-circuit-locked entries are now caught (0 before).
  FIXED   next-open label (y95_next_open): when t+1 is copied, the label window's first entry repeats the high of t,
          a pre-entry price; if that high alone reaches 1.5x the entry open, y95_next_open is set unknown (NaN).
          Count in checks.next_open_label_unknown_pre_entry_high. Panel-wide 3 equity rows qualify (AXITA 2023-04-28,
          KSERASERA 2020-04-01, RELINFRA 2024-04-10), none on a rows-file week, so 0 today; the guard is for reruns.
  FLAGGED the 12 test weeks dated ON a copied session (2020-03-10, 2021-04-14, 2021-04-21, 2021-07-21,
          2022-01-26, 2023-01-26, 2023-03-30, 2023-06-29, 2024-04-11, 2024-08-15, 2026-01-26, 2026-09-14): their
          features are the prior session's copy. Kept (no lookahead: close entry = the prior real close, next-open
          entry = the next real open) and reported as hit_ex_copied_session.
  NOT FIXED here (belongs to panel / research_panel / anatomy): session_calendar() counts the copies (261
          sessions in 2020, 2021 and 2024; real 251 / 248 / 248), so the anatomy's and this script's '95 sessions'
          contain up to ~5 copies; the rows' rolling features count the copied
          session twice; the close-entry y95 window also repeats the high of t when t+1 is copied (it cannot set
          y95 unless high/close >= 1.5 in one session: 3 equity rows panel-wide, all Rs 0.05 shells, none on a
          rows-file week; counted in checks.close_y95_pre_entry_high_could_set, not masked). The purge
          and label-basis detection deliberately keep the copies, because the rows' labels were built on them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

D = ROOT / "logs/leader_sleeve/anatomy_1p5x"
H = 95                                   # label horizon in sessions (anatomy y95)
FIRST_TEST_YEAR = 2019
CONF_FROM = 2023
LONG_SPAN = 105                          # sessions: a 95-row window this long is 'long-span' (~150 calendar days, audit)
BASE_MODELS = ["logit", "lgbm", "xgb", "mlp", "lgbm_rank"]
MODELS = BASE_MODELS + ["ensemble"]
PRICE_LEVEL_FEATS = ("log_px", "pe", "pe_ind")   # one price basis: all three are stale if log_px is adjusted
# columns of rows.parquet that are never features (ids, universe, targets, forward outcomes, stored score)
NON_FEATURES = {"symbol", "trade_date", "era", "close", "raw_close", "mcap_cr", "adv", "ind", "core", "y95", "y63", "s95",
                "fh95", "fh63", "fc95", "pred", "price_adjustment_factor_to_present", "share_adjustment_factor_to_present",
                "year", "sidx", "era2", "tradable", "row_end_sidx", "rows_ahead", "truncated", "label_span_sess",
                "long_span", "win_end", "copied_session", "next_session_copied", "high_over_close_t", "entry_session",
                "entry_open_next", "locked_next_open", "close_adj_t"}
# any name that looks like a forward outcome or a to-present adjustment factor is refused as a feature
LEAK_PAT = re.compile(r"^(y\d|s\d|fh\d|fc\d|pred)|fwd|future|win_end|span|factor", re.I)
IST = ZoneInfo("Asia/Kolkata")


def say(*a):
    print(f"[{datetime.now(IST):%H:%M:%S}]", *a, flush=True)


# ---------------------------------------------------------------- metrics
def auc_rank(y, s) -> float:
    """Mann-Whitney AUC with average ranks for ties; NaN if one class is absent."""
    y = np.asarray(y, dtype=bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(np.asarray(s, dtype=float)).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mean_weekly_auc(dates, y, s) -> float:
    """Mean over trade_dates of the within-date AUC (dates with one class only are skipped)."""
    df = pd.DataFrame({"d": dates, "y": y, "s": s}).dropna()
    v = [auc_rank(g["y"].to_numpy(), g["s"].to_numpy()) for _, g in df.groupby("d", sort=False)]
    v = [x for x in v if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


def weekly_metrics(df: pd.DataFrame, score: str, label: str = "y95") -> pd.DataFrame:
    """Per trade_date over labelled rows: within-week AUC, top-1% precision (top max(1, n//100) by score), base."""
    out = []
    for d, g in df[["trade_date", score, label]].dropna().groupby("trade_date", sort=True):
        y = g[label].to_numpy(dtype=float); s = g[score].to_numpy(dtype=float)
        k = max(1, len(y) // 100)
        top = np.argsort(-s, kind="stable")[:k]
        out.append((d, auc_rank(y, s), y[top].mean(), y.mean(), len(y)))
    return pd.DataFrame(out, columns=["trade_date", "auc", "top1", "base", "n"])


def summarise_weekly(W: pd.DataFrame) -> dict:
    if W.empty:
        return dict(weeks=0)
    return dict(weeks=int(len(W)), wk_auc=W["auc"].mean(), wk_auc_weeks=int(W["auc"].notna().sum()),
                wk_top1=W["top1"].mean() * 100, wk_base=W["base"].mean() * 100,
                wk_top1_lift=W["top1"].mean() / W["base"].mean() if W["base"].mean() > 0 else np.nan)


def weekly_topk(df: pd.DataFrame, score: str, eligible: str, label: str, k: int = 10,
                flags: tuple[str, ...] = ()) -> dict:
    """Point-in-time weekly top-k among eligible rows (labelled or not). Hit rate over labelled picks, plus
    bounds treating unlabelled picks as misses (lo) or hits (hi). Base = mean label over eligible labelled rows.
    For each flag in `flags` (e.g. long_span): the same picks' hit rate leaving out flagged picks (the picks are
    not re-chosen)."""
    C = df[df[eligible].astype(bool) & df[score].notna()]
    P = C.sort_values(["trade_date", score], ascending=[True, False], kind="stable").groupby("trade_date").head(k)
    n = len(P)
    if n == 0:
        return dict(picks=0)
    lab = P[label]; n_unl = int(lab.isna().sum()); hits = float(lab.sum())
    base = C[label].mean(); hit = lab.mean()
    out = dict(picks=n, weeks=int(P["trade_date"].nunique()), unlabelled_picks=n_unl,
               hit=hit * 100, hit_lo=hits / n * 100, hit_hi=(hits + n_unl) / n * 100,
               base=base * 100, lift=hit / base if base and base > 0 else np.nan)
    for flag in flags:
        f = P[flag].fillna(False).astype(bool)
        out[f"{flag}_picks"] = int(f.sum())
        out[f"hit_{flag}"] = lab[f].mean() * 100 if f.any() else np.nan
        out[f"hit_ex_{flag}"] = lab[~f].mean() * 100
    return out


# ---------------------------------------------------------------- MLP with time-ordered early stopping
def fit_mlp(sub: pd.DataFrame, feats: list[str], seed: int, max_epochs: int = 40, patience: int = 5,
            val_frac: float = 0.15):
    """Choose the epoch count on a time-ordered validation block (last `val_frac` of training dates); fit rows
    must have their MEASURED label window (`win_end`, a global-calendar session index, see main) end before
    the block starts. Criterion: mean within-week AUC. Then refit a fresh network on all of `sub` for the
    chosen number of epochs. Returns (predict_fn, info)."""
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    def fresh():
        pre = make_pipeline(SimpleImputer(strategy="median"),
                            QuantileTransformer(output_distribution="normal", subsample=100_000, random_state=seed))
        net = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=1024, learning_rate_init=1e-3,
                            early_stopping=False, random_state=seed)
        return pre, net

    dates = np.sort(sub["trade_date"].unique())
    v0 = pd.Timestamp(dates[min(len(dates) - 1, int(np.floor(len(dates) * (1 - val_frac))))])
    iv0 = int(sub.loc[sub["trade_date"] == v0, "sidx"].iloc[0])
    fit = sub[sub["win_end"] < iv0]
    val = sub[sub["trade_date"] >= v0]
    if len(fit) < 1000 or val["y95"].nunique() < 2:
        raise RuntimeError(f"MLP validation split unusable (fit {len(fit)}, val {len(val)}, val classes {val['y95'].nunique()})")
    cls = np.array([0, 1])
    pre, net = fresh()
    Xf = pre.fit_transform(fit[feats]); yf = fit["y95"].to_numpy().astype(int)
    Xv = pre.transform(val[feats]); yv = val["y95"].to_numpy(); dv = val["trade_date"].to_numpy()
    curve, best, best_ep, bad = [], -np.inf, 1, 0
    for ep in range(1, max_epochs + 1):
        net.partial_fit(Xf, yf, classes=cls)
        sc = mean_weekly_auc(dv, yv, net.predict_proba(Xv)[:, 1]); curve.append(sc)
        if sc > best + 1e-4:
            best, best_ep, bad = sc, ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    pre, net = fresh()
    X = pre.fit_transform(sub[feats]); y = sub["y95"].to_numpy().astype(int)
    for _ in range(best_ep):
        net.partial_fit(X, y, classes=cls)
    info = dict(val_first_date=str(v0.date()), fit_last_date=str(fit["trade_date"].max().date()), n_fit=int(len(fit)),
                n_val=int(len(val)), best_epoch=int(best_ep), epochs_run=len(curve), val_wk_auc_best=float(best),
                val_wk_auc_curve=[round(float(c), 4) for c in curve])
    return (lambda Z: net.predict_proba(pre.transform(Z[feats]))[:, 1]), info


# ---------------------------------------------------------------- copied (non-trading) sessions in the panel
def _source_dates(src: pd.Series) -> pd.Series:
    """trade_date_source ('10-Mar-2021', sometimes with a leading space) -> Timestamp (NaT if unparseable)."""
    u = pd.Series(pd.unique(src.astype(str)))
    m = dict(zip(u, pd.to_datetime(u.str.strip(), format="%d-%b-%Y", errors="coerce")))
    return src.astype(str).map(m)


def _copy_rows(td: pd.Series, src_dt: pd.Series, genuine: pd.DatetimeIndex) -> pd.Series:
    """True for rows that are a COPY of another session's bhavcopy: the row's source date differs from its
    trade_date and that source date is a genuine panel session. Unparseable sources are kept (cannot tell)."""
    return src_dt.notna() & (src_dt != td) & src_dt.isin(genuine)


def copied_sessions() -> dict:
    """Panel dates whose rows are copies of an earlier session's bhavcopy, i.e. not trading sessions.
    A date is 'genuine' when some row on it has trade_date_source == trade_date; a row is a copy when its source
    date differs from its trade_date AND is a genuine date. On the .bak-2026-09-27 panel all 79 dates with
    trade_date_source != trade_date have every row mismatched: 75 are copies (NSE holidays 2020-2026, e.g.
    2021-03-11 = the 2021-03-10 bhavcopy), 4 are real special sessions filed under the next weekday and are KEPT
    (2020-11-16 <- Muhurat 2020-11-14, 2021-11-05 <- Muhurat 2021-11-04, 2024-01-22 <- 2024-01-20,
    2024-05-20 <- 2024-05-18: different prices, low volume). Returns dict(copy_dates, real_cal, genuine, info).
    TODO(panel / research_panel): drop these rows at the source; session_calendar(), wide() and
    next_open_entry() currently treat them as sessions."""
    try:
        d = pd.read_parquet(rp.PANEL, columns=["trade_date", "trade_date_source"]).drop_duplicates()
    except Exception as e:                                   # noqa: BLE001  (column missing on some panel)
        cal = rp.session_calendar()
        return dict(copy_dates=pd.DatetimeIndex([]), real_cal=cal, genuine=cal, has_source=False,
                    info=dict(status=f"unverified: panel has no readable trade_date_source ({type(e).__name__})"))
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d["src"] = _source_dates(d["trade_date_source"])
    genuine = pd.DatetimeIndex(sorted(d.loc[d["src"] == d["trade_date"], "trade_date"].unique()))
    d["copy"] = _copy_rows(d["trade_date"], d["src"], genuine)
    per = d.groupby("trade_date")["copy"].all()
    copy_dates = pd.DatetimeIndex(sorted(per.index[per]))
    cal = pd.DatetimeIndex(sorted(d["trade_date"].unique()))
    misdated = d[(d["src"] != d["trade_date"]) & ~d["copy"] & d["src"].notna()]
    info = dict(status="checked", panel=str(rp.PANEL), copy_dates=len(copy_dates),
                copy_dates_by_year={int(k): int(v) for k, v in pd.Series(copy_dates.year).value_counts().sort_index().items()},
                partially_copied_dates=int((d.groupby("trade_date")["copy"].any() & ~per).sum()),
                unparsed_source_dates=int(d.loc[d["src"].isna(), "trade_date"].nunique()),
                kept_misdated_special_sessions={str(r.trade_date.date()): str(r.src.date())
                                                for r in misdated.drop_duplicates("trade_date").itertuples()})
    return dict(copy_dates=copy_dates, real_cal=cal.difference(copy_dates), genuine=genuine, has_source=True, info=info)


# ---------------------------------------------------------------- panel joins (then-traded price, next open)
def panel_joins(keys: pd.DataFrame, sess: dict, need_open: bool, symbols=None) -> pd.DataFrame:
    """For each (symbol, trade_date) in keys, from the key's own panel row: then-traded close (raw_price) and
    high_over_close_t (adjusted high / close; a copied next session repeats this high inside the label window).
    If need_open: the next-open entry (research_panel.next_open_entry) built on REAL sessions only — copied
    rows (copied_sessions) are dropped first, so the entry is the open of the first real session after t, the
    upper-circuit check compares it with the last real close, and a key dated on a copied session is taken
    as-of the last real session on or before it (the copy holds exactly that session's prices)."""
    lo = keys["trade_date"].min() - pd.Timedelta(days=10); hi = keys["trade_date"].max() + pd.Timedelta(days=20)
    cols = ["close", "high", "price_adjustment_factor_to_present"] + (["open", "low"] if need_open else []) + \
           (["trade_date_source"] if sess["has_source"] else [])
    filt = [("trade_date", ">=", lo), ("trade_date", "<=", hi)]
    if symbols is not None:
        filt.append(("symbol", "in", list(symbols)))
    px = rp.load_panel(cols, filters=filt)
    px["raw_close"] = rp.raw_price(px, "close")
    px["high_over_close_t"] = px["high"] / px["close"]
    out = keys[["symbol", "trade_date"]].merge(px[["symbol", "trade_date", "raw_close", "high_over_close_t"]],
                                              on=["symbol", "trade_date"], how="left")
    out.index = keys.index
    if need_open:
        if sess["has_source"]:
            px = px[~_copy_rows(px["trade_date"], _source_dates(px["trade_date_source"]), sess["genuine"])]
        rc = sess["real_cal"]
        c = rc[(rc >= px["trade_date"].min()) & (rc <= px["trade_date"].max())]
        W = {k: rp.wide(px, k, c) for k in ("open", "high", "low", "close")}
        ne = rp.next_open_entry(W["open"], W["high"], W["low"], W["close"])
        ri = c.searchsorted(pd.DatetimeIndex(keys["trade_date"]), side="right") - 1    # as-of real session
        ci = W["close"].columns.get_indexer(keys["symbol"])
        ok = (ri >= 0) & (ci >= 0)
        e = np.full(len(keys), np.nan); lk = np.zeros(len(keys), dtype=bool); c0 = np.full(len(keys), np.nan)
        e[ok] = ne["entry_px"].to_numpy(dtype=float)[ri[ok], ci[ok]]
        lk[ok] = ne["locked"].to_numpy(dtype=bool)[ri[ok], ci[ok]]
        c0[ok] = W["close"].to_numpy(dtype=float)[ri[ok], ci[ok]]
        out["entry_open_next"] = e; out["locked_next_open"] = lk; out["close_adj_t"] = c0
        ed = np.full(len(keys), np.datetime64("NaT"), dtype="datetime64[ns]")
        nxt = ri + 1; okn = ok & (nxt < len(c))
        ed[okn] = c.values[nxt[okn]]
        out["entry_session"] = ed
    return out


# ---------------------------------------------------------------- label windows (FORWARD-LOOKING: purge + audit only)
def label_window_ends(keys: pd.DataFrame, cal: pd.DatetimeIndex, h: int = H) -> pd.DataFrame:
    """Per (symbol, trade_date) in keys, measured on the panel (never a feature):
      row_end_sidx  global-calendar index of the symbol's h-th next panel ROW = where a row-counted label window
                    (pre-fix anatomy fwd_max) ends; the symbol's last row when fewer than h rows remain (the
                    pre-fix label used what was left if >= 0.8h rows); -1 if the key is not in the panel
      rows_ahead    panel rows of the symbol after t
      truncated     fewer than h rows ahead and the symbol still trades at the panel end (window cut by the end)"""
    px = rp.load_panel([], filters=[("symbol", "in", list(pd.unique(keys["symbol"])))], equities_only=False)
    g = px.groupby("symbol", sort=False)["trade_date"]
    last = g.transform("max")
    px["row_end"] = g.shift(-h).fillna(last)
    px["rows_ahead"] = g.cumcount(ascending=False)
    px["truncated"] = (px["rows_ahead"] < h) & (last >= cal[-1] - pd.Timedelta(days=10))
    k = keys[["symbol", "trade_date"]].merge(px, on=["symbol", "trade_date"], how="left")
    if len(k) != len(keys):
        raise SystemExit("panel has duplicate (symbol, trade_date) rows: row-counted windows are ambiguous")
    return pd.DataFrame({"row_end_sidx": cal.get_indexer(pd.DatetimeIndex(k["row_end"])),
                         "rows_ahead": k["rows_ahead"].to_numpy(dtype=float),
                         "truncated": k["truncated"].fillna(False).to_numpy(dtype=bool)}, index=keys.index)


def detect_label_basis(S: pd.DataFrame, cal: pd.DatetimeIndex, h: int = H, n_sym: int = 150, seed: int = 0) -> tuple[str, dict]:
    """Which forward window produced the rows' fh95 (and so y95 = fh95 >= 1.5)? On labelled rows whose row-counted
    and session-counted windows differ (the symbol misses sessions inside the window), recompute fh95 from the
    panel both ways:
      rows    = max high over the symbol's next h panel rows / close (pre-fix anatomy, >= 0.8h rows)
      session = max high over the next h sessions of the global calendar / close (research_panel.forward_window)
    and count exact matches among rows where the two recomputations differ. Returns (basis, stats) with basis in
    'session' | 'rows' | 'indistinguishable' (< 30 such rows: the two windows barely differ on this panel) |
    'unverified' (neither reproduces fh95: the panel is not the one the rows were built on)."""
    cand = S[S["fh95"].notna() & S["y95"].notna() & (S["row_end_sidx"] - S["sidx"] > h)]
    syms = pd.Series(pd.unique(cand["symbol"]))
    if len(syms) > n_sym:
        syms = syms.sample(n_sym, random_state=seed)
    C = cand[cand["symbol"].isin(set(syms))]
    st = dict(candidate_rows=int(len(cand)), sampled_symbols=int(len(syms)), sampled_rows=int(len(C)), panel=str(rp.PANEL))
    if C.empty:
        return "indistinguishable", {**st, "discriminating_rows": 0}
    px = rp.load_panel(["high", "close"], filters=[("symbol", "in", syms.tolist())], equities_only=False)
    px["fh_rows"] = px.groupby("symbol", sort=False)["high"].transform(
        lambda s: s.shift(-1)[::-1].rolling(h, min_periods=int(h * .8)).max()[::-1]) / px["close"]
    hw = rp.wide(px, "high", cal); cw = rp.wide(px, "close", cal)
    hs = hw[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1)
    ri = cal.get_indexer(C["trade_date"]); ci = hw.columns.get_indexer(C["symbol"])
    fs = np.full(len(C), np.nan); at = (ri >= 0) & (ci >= 0)
    fs[at] = hs.to_numpy(dtype=float)[ri[at], ci[at]] / cw.to_numpy(dtype=float)[ri[at], ci[at]]
    fr = C[["symbol", "trade_date"]].merge(px[["symbol", "trade_date", "fh_rows"]], on=["symbol", "trade_date"],
                                           how="left")["fh_rows"].to_numpy(dtype=float)
    f0 = C["fh95"].to_numpy(dtype=float)
    ok = np.isfinite(fs) & np.isfinite(fr) & np.isfinite(f0) & (fs > 0)
    d = ok.copy(); d[ok] = np.abs(fr[ok] / fs[ok] - 1) > 1e-3
    n = int(d.sum())
    mr = float(np.mean(np.abs(fr[d] / f0[d] - 1) < 1e-6)) if n else np.nan
    ms = float(np.mean(np.abs(fs[d] / f0[d] - 1) < 1e-6)) if n else np.nan
    st.update(discriminating_rows=n, match_rows=mr, match_session=ms)
    if n < 30:
        return "indistinguishable", st
    if ms >= 0.9 and ms > mr:
        return "session", st
    if mr >= 0.9 and mr > ms:
        return "rows", st
    return "unverified", st


def select_features(man: dict, S: pd.DataFrame, drop: set[str]) -> tuple[list[str], str, dict]:
    """Model features from the rows manifest, never re-derived. Preference: `model_features`, then `features`
    (the fixed anatomy's rows manifest: the columns its model used), else the documented `columns` (pre-fix
    manifest) or the rows' own columns, minus non-features and minus `not_model_features`. Names that look
    forward-looking are refused. Returns (feats, source, info)."""
    nmf = man.get("not_model_features")
    nmf = {str(k): v for k, v in nmf.items()} if isinstance(nmf, dict) else \
        ({str(k): "" for k in nmf} if isinstance(nmf, list) else {})
    src = next((f"manifest.{key}" for key in ("model_features", "features") if isinstance(man.get(key), list)), None)
    if src is not None:
        cand = [str(c) for c in man[src.split(".", 1)[1]]]
        missing = [c for c in cand if c not in S.columns]
        if missing:
            raise SystemExit(f"rows {src} lists columns missing from the rows file: {missing} (stale rows/manifest pair)")
    else:
        has_cols = isinstance(man.get("columns"), dict)
        base = list(man["columns"]) if has_cols else list(S.columns)
        src = ("manifest.columns" if has_cols else "rows columns (no manifest)") + " minus non-features" + \
              (" minus not_model_features" if nmf else "")
        cand = [c for c in base if c not in nmf]
    usable = lambda c: pd.api.types.is_numeric_dtype(S[c]) and not pd.api.types.is_bool_dtype(S[c])
    present = [c for c in cand if c in S.columns]
    feats = [c for c in present if c not in NON_FEATURES and usable(c)]
    leak = [c for c in feats if LEAK_PAT.search(c)]
    info = dict(not_model_features=nmf,
                listed_but_non_feature=[c for c in present if c in NON_FEATURES] if src.startswith("manifest.") and
                "columns" not in src else [],
                non_numeric_skipped=[c for c in present if c not in NON_FEATURES and not usable(c)],
                leak_guard_refused=leak, drop_not_in_candidates=sorted(d for d in drop if d not in cand))
    return [c for c in feats if c not in leak and c not in drop], src, info


def feature_coverage(L: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """years x features: share non-null, and share non-zero (NaN counted as zero) among labelled rows."""
    yr = L["trade_date"].dt.year
    nn = pd.DataFrame({f: L[f].notna().groupby(yr).mean() for f in feats})          # column by column: no full copy
    nz = pd.DataFrame({f: (L[f].fillna(0) != 0).groupby(yr).mean() for f in feats})
    return nn, nz


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default=str(D / "rows.parquet"), help="anatomy rows (features + labels)")
    ap.add_argument("--out-dir", default=str(D))
    ap.add_argument("--panel", default=None, help="override research_panel.PANEL (smoke tests: the .bak copy)")
    ap.add_argument("--entry", choices=["close", "next_open", "both"], default="both",
                    help="close = registered signal-close entry; next_open = live contract (skip UC-locked); both")
    ap.add_argument("--years", default=None, help="comma list of test years (default: every year >= 2019)")
    ap.add_argument("--drop-features", default="", help="comma list of features to exclude (e.g. era-marker features)")
    ap.add_argument("--allow-stale-features", action="store_true",
                    help="run even if the rows' price levels (log_px, hence pe/pe_ind) are on the back-adjusted basis "
                         "and one of them is a model feature (result flagged stale)")
    ap.add_argument("--label-basis", choices=["auto", "session", "rows"], default="auto",
                    help="label window definition of the rows' y95: auto = detect from the panel (default); "
                         "session / rows = assert it (recorded as not verified)")
    ap.add_argument("--allow-stale-labels", action="store_true",
                    help="run even if y95 is row-counted or its window definition cannot be verified on this panel "
                         "(purge on the measured row window; result flagged stale)")
    ap.add_argument("--smoke", action="store_true",
                    help="code smoke test only: 300 symbols, small models, <=4 MLP epochs. Numbers are NOT results.")
    ap.add_argument("--no-save", action="store_true", help="print only; write nothing")
    args = ap.parse_args(argv)
    if args.panel:
        rp.PANEL = Path(args.panel)
    out_dir = Path(args.out_dir); rows_path = Path(args.rows)
    do_close = args.entry in ("close", "both"); do_open = args.entry in ("next_open", "both")

    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # ---- rows + feature list (from the anatomy manifest, never re-derived)
    man_path = rows_path.with_name(rows_path.name + ".manifest.json")
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    syms = None
    if args.smoke:
        allsym = pd.read_parquet(rows_path, columns=["symbol"])["symbol"].drop_duplicates().sort_values()
        syms = allsym.sample(min(300, len(allsym)), random_state=7).tolist()
        S = pd.read_parquet(rows_path, filters=[("symbol", "in", syms)])
    else:
        S = pd.read_parquet(rows_path)
    S["trade_date"] = pd.to_datetime(S["trade_date"])
    drop = {c.strip() for c in args.drop_features.split(",") if c.strip()}
    FEATS, feat_src, feat_info = select_features(man, S, drop)
    leak_refused = feat_info["leak_guard_refused"]
    if leak_refused:
        say(f"WARNING forward-looking-name guard refused features: {leak_refused}")
    if feat_info["drop_not_in_candidates"]:
        say(f"WARNING --drop-features names not among the candidate features: {feat_info['drop_not_in_candidates']}")
    S["year"] = S["trade_date"].dt.year
    cal = rp.session_calendar()
    S["sidx"] = cal.get_indexer(S["trade_date"])
    if (S["sidx"] < 0).any():
        raise SystemExit(f"{int((S['sidx'] < 0).sum())} rows have trade_dates missing from the session calendar")
    # copied (non-trading) sessions: `cal` keeps them, because the rows' labels and the purge were measured on
    # this calendar; the next-open entry skips them (panel_joins)
    sess = copied_sessions()
    S["copied_session"] = S["trade_date"].isin(sess["copy_dates"])                      # features = prior session copy
    S["next_session_copied"] = np.isin(S["sidx"].to_numpy() + 1, cal.get_indexer(sess["copy_dates"]))
    say(f"copied sessions: {sess['info']}")

    # ---- measured label windows (purge + audit; never features)
    S = S.join(label_window_ends(S[["symbol", "trade_date"]], cal))
    S["label_span_sess"] = (S["row_end_sidx"] - S["sidx"]).where(S["row_end_sidx"] >= 0)   # sessions to the 95th next row
    S["long_span"] = S["label_span_sess"] >= LONG_SPAN
    if args.label_basis == "auto":
        basis, basis_info = detect_label_basis(S, cal)
    else:
        basis, basis_info = args.label_basis, dict(source="--label-basis (asserted, not verified)")
    sess_end = S["sidx"] + H                                   # session-counted window t+1..t+95 on the global calendar
    row_end = S["row_end_sidx"].where(S["row_end_sidx"] >= 0, np.iinfo(np.int32).max)   # key not in panel: never trained on
    S["win_end"] = sess_end if basis == "session" else np.maximum(sess_end, row_end)
    stale_labels = basis in ("rows", "unverified")
    say(f"label basis: {basis} {basis_info}")
    L = S.dropna(subset=["y95"]).copy()
    # evaluation weeks: the full 95-session window lies inside the rows' span (the anatomy labels windows
    # truncated at the panel end once 76 rows exist; those weeks are scored in no metric here)
    full_ok = S["sidx"] <= S["sidx"].max() - H
    t_max = min(L["trade_date"].max(), S.loc[full_ok, "trade_date"].max())
    say(f"rows {len(S):,} (labelled {len(L):,}, last labelled week {L['trade_date'].max().date()}, last evaluated week "
        f"{t_max.date()}) · features {len(FEATS)} [{feat_src}]")

    years = sorted(y for y in S["year"].unique() if y >= FIRST_TEST_YEAR)
    if args.years:
        years = [y for y in years if y in {int(x) for x in args.years.split(",")}]
    if args.smoke and not args.years:
        years = years[-3:-2] or years[-1:]

    # ---- then-traded price for the tradable universe (+ next-open entry), before any training
    tmask = S["year"].isin(years)
    J = panel_joins(S.loc[tmask, ["symbol", "trade_date"]], sess, do_open, symbols=syms)
    for c in J.columns.drop(["symbol", "trade_date"]):
        S.loc[tmask, c] = J[c]
    S["tradable"] = (S["adv"] >= 5) & (S["raw_close"] > 50)   # adv in Rs crore (rows manifest)
    T = S[tmask]
    tw = pd.DatetimeIndex(sorted(T["trade_date"].unique()))
    checks = dict(raw_close_join_share=float(T["raw_close"].notna().mean()),
                  copied_sessions=dict(sess["info"],
                                       test_weeks=int(len(tw)),
                                       test_weeks_on_copied_session=[str(d.date()) for d in tw[tw.isin(sess["copy_dates"])]],
                                       test_weeks_before_copied_session=sorted(
                                           str(d.date()) for d in T.loc[T["next_session_copied"], "trade_date"].unique())))
    if "core" in S.columns:
        checks["tradable_vs_rows_core_disagree_share"] = float((T["tradable"] != T["core"].astype(bool)).mean())
    refuse = []
    # price basis of the ROWS, checked whether or not log_px is a model feature (log_px, pe, pe_ind share it)
    level_used = [f for f in PRICE_LEVEL_FEATS if f in FEATS]
    if "log_px" in S.columns:
        m = T["log_px"].notna() & T["raw_close"].notna()
        mism = float((np.abs(10 ** T.loc[m, "log_px"] / T.loc[m, "raw_close"] - 1) > 0.01).mean()) if m.any() else np.nan
        checks["log_px_not_raw_basis_share"] = mism
        rows_basis = "unverified (no test rows to compare)" if np.isnan(mism) else \
            ("back-adjusted" if mism > 0.005 else "then-traded")
    else:
        rows_basis = "unverified (rows have no log_px column)"
    stale = rows_basis != "then-traded" and bool(level_used)
    checks.update(rows_price_level_basis=rows_basis, price_level_features_used=level_used, stale_adjusted_price_features=stale)
    if stale:
        how = (f" (log_px vs then-traded close disagree on {checks['log_px_not_raw_basis_share']:.1%} of test rows)"
               if rows_basis == "back-adjusted" else "")
        msg = (f"the rows' price levels are {rows_basis}{how} and {level_used} "
               f"{'is' if len(level_used) == 1 else 'are'} model features: they may encode later splits/bonuses. Re-run "
               f"anatomy_1p5x.py after its audit fix, drop all of {list(PRICE_LEVEL_FEATS)}, or pass "
               f"--allow-stale-features to proceed with a flagged result.")
        if args.allow_stale_features:
            say("WARNING " + msg)
        else:
            refuse.append("STALE FEATURES: " + msg)
    elif rows_basis == "back-adjusted":
        say(f"note: rows carry back-adjusted price levels, but none of {list(PRICE_LEVEL_FEATS)} is a model feature")

    # label span by year (labelled rows): long-span windows are inflated labels when row-counted, lower bounds
    # (missing sessions) when session-counted
    span_by_year = {}
    for y, g in L.groupby("year"):
        ls = g["long_span"].astype(bool)
        span_by_year[int(y)] = dict(labelled=int(len(g)), gappy_window_share=float((g["label_span_sess"] > H).mean()),
                                    long_span_share=float(ls.mean()), y95_long_span=g.loc[ls, "y95"].mean() * 100,
                                    y95_other=g.loc[~ls, "y95"].mean() * 100)
    print(f"\nLABEL WINDOWS by year (labelled rows; label basis = {basis}; long-span = 95 rows over >= {LONG_SPAN} sessions):")
    print(pd.DataFrame(span_by_year).T.round(3).to_string())

    # label window definition (the training purge is measured either way; the TEST labels are what stays stale)
    checks.update(label_basis=basis, stale_row_counted_labels=stale_labels,
                  unknown_row_window_rows=int((S["row_end_sidx"] < 0).sum()))
    if stale_labels:
        why = ("y95 counts the symbol's next 95 panel ROWS" if basis == "rows" else
               "neither a row- nor a session-counted window reproduces the rows' fh95 on this panel (the panel is "
               "probably not the one the rows were built on, so the measured row window may be too short)")
        msg = (f"{why}. Training rows are purged on the measured row window, but test-year labels stay stretched "
               f"over gaps (long-span share by year in the LABEL WINDOWS table above). Re-run anatomy_1p5x.py after its audit fix, "
               f"or pass --allow-stale-labels to proceed with a flagged result.")
        if args.allow_stale_labels:
            say("WARNING " + msg)
        else:
            refuse.append("STALE LABELS: " + msg)
    if refuse:
        raise SystemExit("\n".join(refuse))
    say(f"checks: {checks}")

    # ---- feature coverage per year (critic item 2: era-marker features)
    nn, nz = feature_coverage(L, FEATS)
    flag_nn = [f for f in FEATS if nn[f].max() >= 0.02 and nn[f].min() < 0.5 * nn[f].max()]
    flag_nz = [f for f in FEATS if nz[f].max() >= 0.02 and nz[f].min() < 0.25 * nz[f].max() and f not in flag_nn]
    if flag_nn or flag_nz:
        print("\nFEATURE COVERAGE differs strongly by year (share of labelled rows; NaN/zero may mean 'no data', not 'no event'):")
        show = nn[flag_nn].round(2).T if flag_nn else None
        if show is not None:
            print("  non-null share:\n" + show.to_string())
        if flag_nz:
            print("  non-zero share:\n" + nz[flag_nz].round(2).T.to_string())

    # ---- walk-forward
    small = args.smoke
    n_trees, n_rank, sub_cap = (60, 40, 50_000) if small else (400, 300, 300_000)
    mlp_epochs, mlp_pat = (4, 2) if small else (40, 5)
    preds = {k: pd.Series(np.nan, index=S.index) for k in BASE_MODELS}
    fold_end = pd.Series(pd.NaT, index=S.index)
    folds = {}
    for Y in years:
        first = cal[cal >= pd.Timestamp(f"{Y}-01-01")]
        if first.empty:
            continue
        i0 = cal.get_loc(first[0])
        tr = L[L["win_end"] < i0]                              # measured label window ends before the first test session
        te = S[S["year"] == Y]                                 # ALL rows of Y: the point-in-time candidate set
        if te.empty or tr["y95"].nunique() < 2:
            continue
        sub = tr.sample(min(len(tr), sub_cap), random_state=Y)
        lo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, C=0.5))
        lo.fit(sub[FEATS], sub["y95"]); preds["logit"][te.index] = lo.predict_proba(te[FEATS])[:, 1]
        gb = lgb.LGBMClassifier(n_estimators=n_trees, learning_rate=0.03, num_leaves=63, min_child_samples=300, subsample=0.8,
                                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        gb.fit(tr[FEATS], tr["y95"]); preds["lgbm"][te.index] = gb.predict_proba(te[FEATS])[:, 1]
        xg = xgb.XGBClassifier(n_estimators=n_trees, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                               min_child_weight=50, tree_method="hist", eval_metric="auc", n_jobs=4)
        xg.fit(tr[FEATS], tr["y95"]); preds["xgb"][te.index] = xg.predict_proba(te[FEATS])[:, 1]
        mlp_predict, mlp_info = fit_mlp(sub, FEATS, seed=Y, max_epochs=mlp_epochs, patience=mlp_pat)
        preds["mlp"][te.index] = mlp_predict(te)
        trs = tr.sort_values("trade_date")
        rk = lgb.LGBMRanker(n_estimators=n_rank, learning_rate=0.05, num_leaves=63, min_child_samples=300, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        rk.fit(trs[FEATS], trs["y95"].astype(int), group=trs.groupby("trade_date", sort=True).size().values)
        preds["lgbm_rank"][te.index] = rk.predict(te[FEATS])
        fold_end[te.index] = tr["trade_date"].max()
        # audit: labelled rows whose ROW-counted 95-row window reaches Y (a leak only if labels are row-counted)
        reach = L["row_end_sidx"] >= i0
        old_cut = L["trade_date"] < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=150)
        folds[Y] = dict(train_rows=int(len(tr)), train_last_date=str(tr["trade_date"].max().date()),
                        first_test_session=str(first[0].date()), test_rows=int(len(te)),
                        test_labelled=int(te["y95"].notna().sum()), mlp=mlp_info,
                        row_window_reaches_test_year=dict(
                            old_150d_cut=int((old_cut & reach).sum()),
                            session_only_purge=int(((L["sidx"] <= i0 - H - 1) & reach).sum()),
                            applied_purge=int(((L["win_end"] < i0) & reach).sum())))
        rw = folds[Y]["row_window_reaches_test_year"]
        say(f"fold {Y}: train {len(tr):,} (<= {tr['trade_date'].max().date()}) test {len(te):,} · "
            f"MLP best epoch {mlp_info['best_epoch']}/{mlp_info['epochs_run']} (val wk-AUC {mlp_info['val_wk_auc_best']:.3f}) · "
            f"row windows reaching {Y}: old cut {rw['old_150d_cut']}, session purge {rw['session_only_purge']}, "
            f"applied {rw['applied_purge']}")

    O = S[preds["lgbm"].notna()].copy()
    for k, v in preds.items():
        O[k] = v[O.index]
    O["fold_train_end"] = fold_end[O.index]
    O["ensemble"] = O.groupby("trade_date")[["lgbm", "xgb", "mlp"]].rank(pct=True).mean(axis=1)   # within-week ranks
    O["era2"] = np.where(O["year"] >= CONF_FROM, "conf", "disc")
    if do_open:
        gap = O["entry_open_next"] / O["close_adj_t"]
        O["y95_next_open"] = (O["fh95"] / gap >= 1.5).astype(float).where(O["fh95"].notna() & gap.notna())
        # when session t+1 is a copied session the label window's first 'session' repeats the high of t, a price
        # from BEFORE the next-open entry: if that high alone reaches 1.5x the entry, the label is unknown
        pre = O["next_session_copied"] & (O["high_over_close_t"] / gap >= 1.5)
        O.loc[pre, "y95_next_open"] = np.nan
        checks["next_open_label_unknown_pre_entry_high"] = int(pre.sum())
        moved = O["next_session_copied"] & O["entry_open_next"].notna()
        checks["next_open_entries_after_copied_session"] = int(moved.sum())
        O["buyable_next_open"] = O["tradable"] & O["entry_open_next"].notna() & ~O["locked_next_open"].astype(bool)
        checks["next_open_skipped_share_of_tradable"] = float(1 - O.loc[O["tradable"], "buyable_next_open"].mean())
    E_all = O[O["trade_date"] <= t_max]                        # weeks whose label windows are complete
    checks["truncated_labelled_evaluated_rows"] = int((E_all["y95"].notna() & E_all["truncated"]).sum())
    # close entry: the high of t itself (intraday, before the close) sits in the window when t+1 is copied;
    # counted, not masked (anatomy's registered label; high/close >= 1.5 in one session is ~never)
    checks["close_y95_pre_entry_high_could_set"] = int((E_all["next_session_copied"] & (E_all["high_over_close_t"] >= 1.5)
                                                         & (E_all["y95"] == 1)).sum())

    # ---- metrics: per era and per year
    res, per_year = {}, {}
    hdr = "MODEL        era   wkAUC  wkTop1%  lift |"
    hdr += " close-entry wk10 hit% [lo..hi] lift (unlab) exLS hit% (long-span picks)" if do_close else ""
    hdr += ((" |" if do_close else "") + " next-open wk10 hit% lift exLS") if do_open else ""
    print("\n" + hdr)
    for k in MODELS:
        W = weekly_metrics(E_all, k)
        W["year"] = W["trade_date"].dt.year
        W["era2"] = np.where(W["year"] >= CONF_FROM, "conf", "disc")
        groups = [("era", e, W["era2"] == e, E_all["era2"] == e) for e in ("disc", "conf")] + \
                 [("year", y, W["year"] == y, E_all["year"] == y) for y in sorted(W["year"].unique())]
        for kind, g, wm, em in groups:
            r = summarise_weekly(W[wm])
            E = E_all[em]
            if do_close:
                r["close"] = weekly_topk(E, k, "tradable", "y95", flags=("long_span", "copied_session"))
            if do_open:
                r["next_open"] = weekly_topk(E, k, "buyable_next_open", "y95_next_open", flags=("long_span", "copied_session"))
            (res if kind == "era" else per_year)[f"{k}|{g}"] = r
            if kind == "era" and r.get("weeks"):
                line = f"{k:<12} {g}  {r['wk_auc']:.3f}  {r['wk_top1']:5.1f}%  {r['wk_top1_lift']:4.2f}x |"
                if do_close and r["close"].get("picks"):
                    c = r["close"]
                    line += (f"  {c['hit']:5.1f}% [{c['hit_lo']:4.1f}..{c['hit_hi']:4.1f}] {c['lift']:4.2f}x ({c['unlabelled_picks']})"
                             f" exLS {c['hit_ex_long_span']:5.1f}% ({c['long_span_picks']})")
                if do_open and r["next_open"].get("picks"):
                    c = r["next_open"]
                    line += (" |" if do_close else "") + f" {c['hit']:5.1f}% {c['lift']:4.2f}x exLS {c['hit_ex_long_span']:5.1f}%"
                print(line, flush=True)

    # ---- save
    run_ts = datetime.now(IST).isoformat(timespec="seconds")
    wf_text = ("test year Y scored by models trained on labelled rows whose label window ends before Y's first session; "
               + ("window = sessions t+1..t+95 of the global calendar (session-counted labels, verified on the panel)"
                  if basis == "session" else
                  f"window end = later of session t+95 and the symbol's 95th next panel row (label basis '{basis}': "
                  f"measured on {rp.PANEL.name})"))
    out = dict(experiment="model-bakeoff-1p5x", run_ist=run_ts, smoke=bool(args.smoke), args=vars(args),
               rows_file=str(rows_path), rows_manifest_updated=man.get("updated"),
               label_definition=(man.get("columns") or {}).get("y95"), panel=str(rp.PANEL),
               label_basis=basis, label_basis_detection=basis_info, stale_row_counted_labels=stale_labels,
               stale_adjusted_price_features=stale, walk_forward=wf_text, label_span_by_year=span_by_year,
               eras=dict(disc=f"{FIRST_TEST_YEAR}-{CONF_FROM - 1} test years", conf=f"{CONF_FROM}+"),
               metric_notes=dict(
                   wk_auc="mean over weeks of within-week AUC (labelled rows, full mcap>=50cr universe)",
                   wk_top1="mean over weeks of precision in the top max(1, n//100) rows by score; lift = mean precision / mean base",
                   close="weekly top-10 among tradable rows (ADV>=5cr, raw close>50), entry at signal close, hit = y95; "
                         "hit over labelled picks, hit_lo/hit_hi treat unlabelled picks as miss/hit",
                   next_open="same, entry at the next REAL session's open (copied sessions skipped), UC-locked/not-trading "
                             "names skipped, hit = fh95*close_t/open_t+1 >= 1.5",
                   hit_ex_copied_session="same picks, hit rate without picks dated on a copied (non-trading) session",
                   hit_ex_long_span=f"same picks, hit rate without picks whose 95-row window spans >= {LONG_SPAN} sessions "
                                    "(row-counted labels: inflated; session-counted: lower bounds); long_span_picks = their count"),
               features=FEATS, feature_source=feat_src, dropped_features=sorted(drop), leak_guard_refused=leak_refused,
               not_model_features=feat_info["not_model_features"], feature_selection=feat_info,
               checks=checks, folds=folds, metrics=res, per_year=per_year,
               feature_coverage=dict(non_null=nn.round(4).to_dict(), non_zero=nz.round(4).to_dict(),
                                     flagged_non_null=flag_nn, flagged_non_zero=flag_nz))
    if args.no_save:
        print("BAKEOFF COMPLETE (not saved)")
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bakeoff.json").write_text(json.dumps(_clean(out), indent=1))
    keep = ["symbol", "trade_date", "year", "era2", "y95", "fh95", "label_span_sess", "long_span", "adv", "raw_close", "tradable"] + \
           (["core"] if "core" in O.columns else []) + \
           ["copied_session", "next_session_copied", "high_over_close_t"] + \
           (["entry_session", "entry_open_next", "close_adj_t", "locked_next_open", "y95_next_open", "buyable_next_open"]
            if do_open else []) + \
           MODELS + ["fold_train_end"]
    P = O[keep].rename(columns={"era2": "era", "label_span_sess": "label_span_sessions"}).reset_index(drop=True)
    P.to_parquet(out_dir / "bakeoff_preds.parquet", index=False)
    cols = dict(symbol="NSE symbol", trade_date="signal date (weekly sample, session close)", year="test year (walk-forward fold)",
                era="disc = test years 2019-2022, conf = 2023+", y95="anatomy label (see rows.parquet.manifest.json)",
                fh95="anatomy: max high over the label window / entry close (ratio); window per label_basis in bakeoff.json",
                label_span_sessions="FORWARD-LOOKING audit only: global-calendar sessions from t to the symbol's 95th next "
                                    "panel row (> 95 = the symbol misses sessions in the window); never a feature",
                long_span=f"label_span_sessions >= {LONG_SPAN} (row-counted labels: inflated; session-counted: lower bound)",
                adv="20d average traded value, Rs crore (from rows)", raw_close="then-traded close, Rs per share (research_panel.raw_price)",
                tradable="adv >= 5 and raw_close > 50", core="anatomy's own flag (adjusted close > 50), kept for comparison only",
                copied_session="trade_date is a copied non-trading session (panel row = an earlier session's bhavcopy); "
                               "features are that earlier session's",
                next_session_copied="the next calendar date of the panel is a copied session (label window repeats the high of t)",
                high_over_close_t="adjusted high / close of the trade_date row (ratio)",
                entry_session="first REAL session after trade_date (copied sessions skipped): the next-open entry date",
                entry_open_next="open at entry_session, Rs per share on the panel's ADJUSTED basis (NaN = not trading)",
                close_adj_t="adjusted close at the last real session on or before trade_date (same basis as entry_open_next)",
                locked_next_open="next session upper-circuit locked (research_panel.next_open_entry): not buyable",
                y95_next_open="1 if fh95 * close_adj_t / entry_open_next >= 1.5; NaN if next_session_copied and the "
                              "pre-entry high of t alone reaches 1.5x the entry",
                buyable_next_open="tradable and entry_open_next present and not locked",
                logit="P(y95), standardised logistic regression", lgbm="P(y95), LightGBM", xgb="P(y95), XGBoost",
                mlp="P(y95), MLP (time-ordered early stopping)", lgbm_rank="LambdaRank score: only its within-week order is meaningful",
                ensemble="mean within-week percentile rank of lgbm, xgb, mlp (0-1)",
                fold_train_end="last training trade_date of the fold that scored this row")
    (out_dir / "bakeoff_preds.parquet.manifest.json").write_text(json.dumps(_clean(dict(
        dataset="1.5x model bake-off out-of-sample predictions", experiment="model-bakeoff-1p5x (EXP-2026-09-27-1p5x-anatomy companion)",
        producer="src/agentic/model_bakeoff_1p5x.py", rows=len(P), key=["symbol", "trade_date"],
        date_range=[str(P["trade_date"].min().date()), str(P["trade_date"].max().date())] if len(P) else None,
        inputs=[str(rows_path), str(rp.PANEL)], smoke=bool(args.smoke), stale_adjusted_price_features=stale,
        label_basis=basis, stale_row_counted_labels=stale_labels, walk_forward=wf_text,
        columns={c: cols.get(c, "") for c in P.columns},
        units="scores are probabilities except lgbm_rank (unitless) and ensemble (percentile 0-1); ratios are fractions",
        updated=run_ts)), indent=1))
    readme = out_dir / "README.md"
    txt = readme.read_text() if readme.exists() else "# 1.5x anatomy (EXP-2026-09-27-1p5x-anatomy)\n"
    if "bakeoff_preds.parquet" not in txt:
        readme.write_text(txt.rstrip("\n") + "\n\n## Model bake-off (src/agentic/model_bakeoff_1p5x.py)\n\n"
                          "- `bakeoff.json` — within-week AUC / top-1% and weekly top-10 tradable hit rate per model, per era and per "
                          "year, close-entry and next-open entry; fold details and feature coverage by year.\n"
                          "- `bakeoff_preds.parquet` — every model's out-of-sample score for each test row (see manifest).\n")
    say(f"saved {out_dir / 'bakeoff.json'} and bakeoff_preds.parquet ({len(P):,} rows)")
    print("BAKEOFF COMPLETE")
    return out


if __name__ == "__main__":
    main()
