"""LIVE SCREEN — winner theme > hot industry > right-to-win names (2026-09-08; universe aligned 2026-09-27).

Applies the registered PRODUCTION leader cell — EXP-2026-09-23b-leader-cell-v2 arm A1 on the nse4 map,
which is the BASELINE of EXP-2026-09-27-leader-7x-factorial (sim_leader_portfolio_7x.py):
  universe  core band on the as-of session: ADV20 >= Rs 5cr AND then-traded (raw) price > Rs 50
  industry  nse4 = NSE 4-level 'Industry' via screener.in, + analyst analog labels for the residual
            (the registered BASELINE code does the same; those names carry industry_source =
            analyst_inference per data/derived/industry_analyst_labels.README.md)
  heat      MEAN own ret60 of the industry's core-band names; only industries with >= 5 core names;
            HOT = heat percentile rank >= 0.9 (top decile)
  leaders   top-3 by own ret60 inside each hot industry, EXTENDED only (own ret252 > +50%,
            EXP-2026-09-23-extended-leader PASSED)
No market gate and no takeover exclusion: the registered BASELINE has neither (lever G did not beat
BASELINE under EXP-2026-09-27-leader-7x-factorial's rule; B / no-takeover failed or was a no-op in
EXP-2026-09-23b). Only these names go into `names`, the list score_leader_sleeve.py scores forward.

Overlays reported per name (evidence, not filters):
  pe_ind (cheap<1 preferred per conf era) · TTM sign · landmine filings 90d · promoter delta
  (latest known SHP) · 200DMA position. (A SUE value is computed in ttm_sue but has never been
  reported; unchanged.)
UNTESTED appendix: the pre-2026-09-27 universe (ADV >= 1.5cr & raw price > 25) run through the same
heat/rank rule. Names it adds are listed for information only — no registered backtest covers that
band, they are NOT scored by score_leader_sleeve.py and NOT for sizing.
DATA GAPS block: what the screen could not see on the as-of date (unlabelled core names that drop out
of heat, names too young for ret60/ret252, missing TTM / SHP / 200DMA for leaders, overlay source dates).

Output: ranked names + reports/theme_leaders_<asof>.md + logs/leader_sleeve/screen_<asof>.json
(<asof> = data date, YYYYMMDD). Each screen is an immutable artifact like a basket: an
existing screen is never rewritten. Paper/venture-sleeve candidates ONLY — historical
P(2x/6mo) ~8-12%, median troughs -14 to -19%. Scored forward by score_leader_sleeve.py.

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json). No entry in "confirmed" names
this file; the items come from the audit's critic text and the shared-primitive defects listed in
research_panel.py. Line numbers refer to the 2026-09-23 version (commit 1d00105).
  FIXED  critic 5 (lines 64-65, 83-90, 144): heat and the top-3 rank were computed on the expanded
         band ADV>=1.5cr & close>25 with core_band only a display flag, so the forward record scored a
         population no registered arm tested. Heat/rank now run on the core band exactly as the
         registered cell; the expanded band survives only as the UNTESTED appendix (separate JSON key
         `untested_expanded_appendix`, never in `names`).
  FIXED  hot-decile rule (line 82-83): was grp_ret60 >= quantile(0.9); now percentile rank >= 0.9 as in
         sim_leader_cell_v2.add_heat / sim_leader_portfolio_7x.select (they differ when 0.9 x n_industries
         is an integer). Leader rank now method='first' as registered (was 'average').
  FIXED  price LEVELS used the back-adjusted close (line 65 close>25, line 144 close>50, line 116 PE):
         now research_panel.raw_price (then-traded price). Identical for a live run on the panel's last
         date; differs for --asof replays across later splits/bonuses/demergers.
  FIXED  ret60/ret252 (lines 48-49) were pct_change over the symbol's own ROWS with the default pad fill;
         a suspension stretched the window. Now compounded research_panel.gap_aware_returns over 60/252
         SESSIONS of the global calendar (NaN if the name has no close before the window).
  FIXED  200DMA (line 50) over 200 rows -> over the last 200 sessions, >= 120 closes present.
  FIXED  universe (line 45): fund units now also removed by the ISIN master (research_panel.load_panel)
         as the registered sims do, not only by generate_hybrid_basket.non_equity.
  FIXED  pe_ind (lines 117-121): industry median PE was over the expanded band, with no minimum count,
         and its TTM summed the last 4 rows in FILING order even when fewer than 4 quarters existed. Now
         the sim_leader_sleeve.py definition behind the 12.4% cell: median over the industry's core-band
         names with >= 5 PEs; TTM = 4 latest quarters spanning <= 380 days whose latest filing is <= 200
         days old (applied to the leader's own TTM too, so stale EPS no longer yields a PE).
  FIXED  shareholding load errors were swallowed silently (lines 131-141); now recorded in data_gaps.
  FIXED  cost (line 187): per-name cost_rt_pct from research_panel.cost_rt(ADV). Every core-band name
         is ADV>=5cr, so this is the registered 0.5% round trip; the header field is unchanged.
  NOT FIXED critic 4 (line 186) execution: this screen is not a simulation; it already specifies
         next-open entry. Skipping upper-circuit-locked entries when scoring belongs to
         score_leader_sleeve.py (not this file); the JSON now states the rule in `entry_skip_rule`
         (research_panel.next_open_entry definition) so the scorer can apply it. No close/next-open CLI
         flag is added because there is no backtest here to run in two modes.
  NOT FIXED existing screens screen_20260908.json (2 expanded-band names: KALYANIFRG, ALPHAGEO) and
         screen_20260923.json (1: RSWM) were produced on the expanded band. Screens are immutable and are
         not rewritten; the forward record for those two cohorts includes untested-band names.
  NOT FIXED critic 3: avg_traded_value_20d rolls over ROWS, not sessions (panel feature,
         src/features/indicators.py). Used as-is so the band matches the registered cell; the fix
         belongs to the panel builder.
  NOT FIXED industry labels are a 2026-09 snapshot (not point-in-time). Irrelevant for a live run; an
         --asof replay applies today's taxonomy.
  NOT FIXED TODO: TTM EPS is as reported; a split/bonus inside the last 4 quarters mixes per-share bases
         (pnl_quarterly has no per-share adjustment factor).
  Per era: a screen is one as-of date; the JSON carries `era` (disc < 2023 <= conf) for --asof replays.
"""
import argparse
import html as _html
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402

PAPER_RULE = ("Sizing: PAPER until the forward record (logs/leader_sleeve/outcomes.jsonl) holds "
              ">=13 closed-or-scored weekly cohorts. No capital before that.")
# 6e (EXP-2026-09-23-extended-leader): set True only if the registered A/B passed.
EXTENDED_ONLY = True   # PASSED 2026-09-23 (logs/experiments.jsonl)

# registered production cell (EXP-2026-09-23b A1 on nse4 = EXP-2026-09-27-leader-7x-factorial BASELINE)
CORE_ADV_CR, CORE_PX = 5.0, 50.0           # core band: ADV20 >= Rs 5cr & raw price > Rs 50
EXP_ADV_CR, EXP_PX = 1.5, 25.0             # pre-2026-09-27 screen universe -> UNTESTED appendix only
MIN_NAMES, HOT_PCT, TOP_N, EXT_RET252 = 5, 0.9, 3, 0.50
H60, H252, DMA_N, DMA_MIN = 60, 252, 200, 120
TTM_SPAN_D, TTM_STALE_D, IND_PE_MIN = 380, 200, 5   # sim_leader_sleeve.py TTM / industry-PE definition
LOOKBACK_DAYS = 800                        # calendar days loaded: >= 252 sessions + suspension slack
ERA_SPLIT = pd.Timestamp("2023-01-01")
REGISTERED_CELL = ("EXP-2026-09-23b-leader-cell-v2 A1 on nse4 = EXP-2026-09-27-leader-7x-factorial BASELINE: "
                   "core band ADV>=5cr & raw price>50; MEAN heat of own ret60 over industries with >=5 core "
                   "names, top decile by percentile rank; top-3 by own ret60; EXTENDED ret252>0.5")
ENTRY_SKIP_RULE = ("buy at the first session OPEN after data_through; a name whose entry session is "
                   "upper-circuit locked (open==high==low and open >= 4.9% above the prior close; "
                   "research_panel.next_open_entry) is not buyable and is skipped")

ap = argparse.ArgumentParser()
ap.add_argument("--asof", default=None, help="data date YYYYMMDD or YYYY-MM-DD (default: panel max date)")
ap.add_argument("--force", action="store_true", help="rewrite an existing screen (repairs only; ledger it)")
ap.add_argument("--preview", action="store_true", help="print the screen, write nothing (ignores cadence/immutability)")
args = ap.parse_args()

print("panel…", flush=True)
hi_dt = (pd.Timestamp(args.asof) if args.asof
         else pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]).max())
px = rp.load_panel(["close", "avg_traded_value_20d", "price_adjustment_factor_to_present"],
                   filters=[("trade_date", ">=", hi_dt - pd.Timedelta(days=LOOKBACK_DAYS)),
                            ("trade_date", "<=", hi_dt)])
cal = rp.session_calendar(px)
last_dt = cal[-1]
TAG = last_dt.strftime("%Y%m%d")
OUT_JSON = ROOT / f"logs/leader_sleeve/screen_{TAG}.json"
if OUT_JSON.exists() and not (args.force or args.preview):
    print(f"screen {OUT_JSON.name} already exists — immutable, not rewritten", flush=True)
    sys.exit(0)
_wk = last_dt.isocalendar()[:2]
_same = [p.name for p in OUT_JSON.parent.glob("screen_*.json")
         if pd.Timestamp(json.loads(p.read_text())["data_through"]).isocalendar()[:2] == _wk]
if _same and not (args.force or args.preview):
    print(f"weekly cadence: {_same[0]} already covers ISO week {_wk[1]} — no second cohort", flush=True)
    sys.exit(0)

# ---- session-calendar returns (gap-aware) and 200-session DMA, as of last_dt ----
cw = rp.wide(px, "close", cal)
R = rp.gap_aware_returns(cw)


def window_ret(h: int) -> pd.Series:
    """Compounded return over the last h SESSIONS; NaN if any session in the window precedes the name's
    first close (too young) — equals close_t / last close on or before session t-h, minus 1."""
    if len(R) <= h:
        return pd.Series(np.nan, index=cw.columns)
    return (1 + R.iloc[-h:]).prod(min_count=h) - 1


_tail = cw.iloc[-DMA_N:]
dma200 = _tail.mean().where(_tail.count() >= DMA_MIN)

snap = px[px["trade_date"] == last_dt].copy()
snap["px_raw"] = rp.raw_price(snap, "close")                 # then-traded price for LEVEL filters / PE
snap["adv"] = snap["avg_traded_value_20d"] / 1e7             # panel manifest: Rs (20-row mean) -> Rs crore
snap["ret60"] = snap["symbol"].map(window_ret(H60))
snap["ret252"] = snap["symbol"].map(window_ret(H252))
snap["dma200"] = snap["symbol"].map(dma200)
snap["core_band"] = (snap["adv"] >= CORE_ADV_CR) & (snap["px_raw"] > CORE_PX)
snap["exp_band"] = (snap["adv"] >= EXP_ADV_CR) & (snap["px_raw"] > EXP_PX)

# Industry map = NSE 4-level 'Industry' (via screener.in) for every equity + analyst analog labels
# for the residual (EXP-2026-09-23b: the old smIndustry map covered ~40% of the universe; on the
# full map the cell keeps 86%/92% of its net/trade by era -> switched per the registered rule).
_sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
_sc = _sc[_sc["status"].str.startswith("OK")].dropna(subset=["industry"])
imap = _sc.set_index("symbol")["industry"].map(_html.unescape)
_al = pd.read_csv(ROOT / "data/derived/industry_analyst_labels.csv")
_add = _al[~_al["symbol"].isin(imap.index)].set_index("symbol")["analog_symbol"].map(imap).dropna()
isrc = pd.concat([pd.Series("nse4", index=imap.index), pd.Series("analyst_inference", index=_add.index)])
imap = pd.concat([imap, _add])
INDUSTRY_MAP = "nse4 (NSE 4-level via screener.in) + analyst analogs"
snap["ind"] = snap["symbol"].map(imap)
snap["ind_source"] = snap["symbol"].map(isrc)
print(f"as-of {last_dt.date()} · core band {int(snap['core_band'].sum()):,} · "
      f"expanded band (appendix only) {int(snap['exp_band'].sum()):,}", flush=True)


def leader_cell(uni: pd.DataFrame):
    """Registered heat + rank rule on one universe of as-of rows -> (groups, hot groups, leaders)."""
    u = uni.dropna(subset=["ind", "ret60"]).sort_values("symbol")
    grp = u.groupby("ind").agg(grp_ret60=("ret60", "mean"), n=("ret60", "size"))
    grp = grp[grp["n"] >= MIN_NAMES].copy()
    grp["heat_pct"] = grp["grp_ret60"].rank(pct=True)
    grp = grp.sort_values("grp_ret60", ascending=False)
    hot = grp[grp["heat_pct"] >= HOT_PCT]
    u = u[u["ind"].isin(grp.index)].copy()
    u["rk"] = u.groupby("ind")["ret60"].rank(ascending=False, method="first")
    lead = u[u["ind"].isin(hot.index) & (u["rk"] <= TOP_N)]
    if EXTENDED_ONLY:
        lead = lead[lead["ret252"] > EXT_RET252]
    lead = lead.sort_values(["ind", "rk"]).copy()
    lead["grp60"] = lead["ind"].map(grp["grp_ret60"])
    return grp, hot, lead


core = snap[snap["core_band"]]
grp, hot, lead = leader_cell(core)
print(f"industries {len(grp)} (core band, >= {MIN_NAMES} names) · HOT (top decile): {len(hot)}", flush=True)
for i, r in hot.iterrows():
    print(f"  🔥 {i:<38s} grp60 {r.grp_ret60*100:+6.1f}%  ({int(r.n)} names)", flush=True)

# UNTESTED appendix: same rule on the expanded band; only names the production cell does not list
grp_x, hot_x, lead_x = leader_cell(snap[snap["exp_band"]])
appx = lead_x[~lead_x["symbol"].isin(lead["symbol"])].copy()

# ---- overlays ----
print("overlays…", flush=True)
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
_pnl_through = q["filing_dt"].max()
q = q[pd.to_datetime(q["filing_dt"]) <= last_dt]                     # point-in-time for --asof replays
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
_o = pd.Timestamp("1900-01-01")
_c = lastq["con"] if "con" in lastq else pd.Series(_o, index=lastq.index)
_s = lastq["sa"] if "sa" in lastq else pd.Series(_o, index=lastq.index)
bmap = pd.Series(np.where(_c.fillna(_o) >= _s.fillna(_o), "con", "sa"), index=lastq.index)
q = q[q["basis"] == q["symbol"].map(bmap)].sort_values(["symbol", "quarter_end"])

# TTM EPS table (registered definition): last 4 quarters spanning <= 380d, latest filing <= 200d old
_l4 = q.groupby("symbol").tail(4).groupby("symbol").agg(
    nq=("eps_basic", "size"), ttm=("eps_basic", "sum"), known=("filing_dt", "max"),
    q0=("quarter_end", "min"), q1=("quarter_end", "max"))
_ok = ((_l4["nq"] == 4) & ((_l4["q1"] - _l4["q0"]).dt.days <= TTM_SPAN_D)
       & ((last_dt - pd.to_datetime(_l4["known"]).dt.normalize()).dt.days <= TTM_STALE_D))
ttm_map = _l4["ttm"].where(_ok)


def ttm_sue(sym):
    gg = q[q["symbol"] == sym]
    if len(gg) < 4:
        return np.nan, np.nan
    sue = (gg["eps_basic"].iloc[-1] - gg["eps_basic"].iloc[-5]) if len(gg) >= 5 else np.nan
    return ttm_map.get(sym, np.nan), sue


vals = {s: ttm_sue(s) for s in lead["symbol"]}
lead["ttm"] = lead["symbol"].map(lambda s: vals[s][0])
lead["pe"] = np.where(lead["ttm"] > 0, lead["px_raw"] / lead["ttm"], np.nan)
ipe = core.dropna(subset=["ind", "ret60"]).copy()                    # the heat universe
ipe["t4"] = ipe["symbol"].map(ttm_map)
ipe["pe"] = np.where(ipe["t4"] > 0, ipe["px_raw"] / ipe["t4"], np.nan)
_med = ipe.dropna(subset=["pe"]).groupby("ind")["pe"].agg(["median", "size"])
ind_med = _med.loc[_med["size"] >= IND_PE_MIN, "median"]
lead["pe_ind"] = lead["pe"] / lead["ind"].map(ind_med)

fs = pd.read_parquet(ROOT / "data/derived/filing_scores.parquet")
fs["event_dt"] = pd.to_datetime(fs["event_dt"])
_fs_through = fs["event_dt"].max()
recent = fs[(fs["event_dt"] >= last_dt - pd.Timedelta(days=90)) & (fs["event_dt"] <= last_dt)]
nneg = recent[recent["catdir"] == -1].groupby("symbol").size()
npos = recent[recent["catdir"] == 1].groupby("symbol").size()
lead["neg90"] = lead["symbol"].map(nneg).fillna(0).astype(int)
lead["pos90"] = lead["symbol"].map(npos).fillna(0).astype(int)

shp_error, _shp_through = None, None
try:
    shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
    shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
    shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
    shp = shp.dropna(subset=["qe"])
    _shp_through = shp["qe"].max()
    shp = shp[shp["qe"] <= last_dt].sort_values(["symbol", "qe"])
    shp["d"] = shp.groupby("symbol")["promoter_pct"].diff()
    pd_map = shp.groupby("symbol")["d"].last()
    lead["prom_d"] = lead["symbol"].map(pd_map)
except Exception as e:                                               # recorded in data_gaps, not hidden
    shp_error = f"{type(e).__name__}: {e}"
    lead["prom_d"] = np.nan

lead["above200"] = lead["close"] > lead["dma200"]
lead["cost_rt_pct"] = rp.cost_rt(lead["adv"]) * 100

# right-to-win score: leader rank + conf-era-validated cheap + clean overlays
lead["rtw"] = ((4 - lead["rk"])                       # leadership strength
               + (lead["pe_ind"] < 1).fillna(False) * 2   # conf-era 12.4% cell
               + (lead["ttm"] > 0).fillna(False) * 1
               + lead["above200"] * 1
               + (lead["prom_d"] > 0).fillna(False) * 1
               - (lead["neg90"] > 0) * 1)
lead = lead.sort_values(["rtw", "grp60"], ascending=False)

cols_hdr = f"{'SYM':<14}{'INDUSTRY':<30}{'grp60':>7} {'own60':>7} {'PE':>7} {'peInd':>6} {'TTM':>5} {'promΔ':>6} {'neg':>4}{'pos':>4} {'200d':>5} {'band':>5} {'RTW':>4}"
print("\n" + cols_hdr, flush=True)
out_lines = [cols_hdr]
for _, r in lead.iterrows():
    line = (f"{r['symbol']:<14}{r['ind'][:29]:<30}{r['grp60']*100:>+6.1f}% {r['ret60']*100:>+6.1f}% "
            f"{r['pe'] if pd.notna(r['pe']) else float('nan'):>7.1f} "
            f"{r['pe_ind'] if pd.notna(r['pe_ind']) else float('nan'):>6.2f} "
            f"{'PROF' if r['ttm'] and r['ttm'] > 0 else 'LOSS' if pd.notna(r['ttm']) else '?':>5} "
            f"{r['prom_d'] if pd.notna(r['prom_d']) else float('nan'):>+6.2f} "
            f"{r['neg90']:>4}{r['pos90']:>4} {'✓' if r['above200'] else '✗':>5} "
            f"{'core':>5} {r['rtw']:>4.0f}")
    print(line, flush=True)
    out_lines.append(line)
if lead.empty:
    print("(no name passes the registered cell this week)", flush=True)
    out_lines.append("(no name passes the registered cell this week)")

appx_hdr = (f"{'SYM':<14}{'INDUSTRY':<30}{'grp60x':>7} {'own60':>7} {'own252':>7} {'ADVcr':>7} {'price':>8}  why not in the production list")
appx_lines = [appx_hdr]


def _why(r) -> str:
    if not r["core_band"]:
        return f"outside core band (ADV {r['adv']:.1f}cr, price Rs {r['px_raw']:.0f})"
    return "core-band name; top-3 in a hot industry only when expanded-band names enter heat/rank"


for _, r in appx.iterrows():
    appx_lines.append(f"{r['symbol']:<14}{r['ind'][:29]:<30}{r['grp60']*100:>+6.1f}% {r['ret60']*100:>+6.1f}% "
                      f"{r['ret252']*100:>+6.1f}% {r['adv']:>7.1f} {r['px_raw']:>8.1f}  {_why(r)}")
print(f"\nUNTESTED APPENDIX — expanded band ADV>={EXP_ADV_CR}cr & raw price>{EXP_PX:.0f}; no registered "
      f"backtest covers it; NOT scored, NOT for sizing ({len(appx)} extra name(s))", flush=True)
print("\n".join(appx_lines), flush=True)

# ---- data gaps (what the screen could not see) ----
_cu = core.copy()


def _sample(df: pd.DataFrame, k: int = 8) -> list:
    d = df.sort_values("adv", ascending=False).head(k)
    return [dict(symbol=s, adv_cr=round(float(a), 2), price_raw=round(float(p), 2))
            for s, a, p in zip(d["symbol"], d["adv"], d["px_raw"])]


def _d(v):
    return None if v is None or pd.isna(v) else str(pd.Timestamp(v).date())


data_gaps = dict(
    core_band_names=int(len(_cu)),
    core_no_industry_label=dict(n=int(_cu["ind"].isna().sum()), effect="excluded from heat and rank",
                                sample=_sample(_cu[_cu["ind"].isna()])),
    core_no_ret60=dict(n=int(_cu["ret60"].isna().sum()), effect="< 60 sessions of closes; excluded",
                       sample=_sample(_cu[_cu["ret60"].isna()])),
    core_no_ret252=dict(n=int(_cu["ret252"].isna().sum()), effect="< 252 sessions of closes; can never be EXTENDED",
                        sample=_sample(_cu[_cu["ret252"].isna()])),
    core_in_small_industry=dict(n=int((_cu["ind"].notna() & ~_cu["ind"].isin(grp.index)).sum()),
                                effect=f"industry has < {MIN_NAMES} core names with ret60; no heat"),
    asof_rows_missing_adj_factor=int(snap["price_adjustment_factor_to_present"].isna().sum()),
    leaders_no_ttm=sorted(lead.loc[lead["ttm"].isna(), "symbol"]),
    leaders_no_industry_pe=sorted(lead.loc[lead["pe_ind"].isna() & lead["pe"].notna(), "symbol"]),
    leaders_no_promoter_delta=sorted(lead.loc[lead["prom_d"].isna(), "symbol"]),
    leaders_no_dma200=sorted(lead.loc[lead["dma200"].isna(), "symbol"]),
    leaders_analyst_inferred_industry=sorted(lead.loc[lead["ind_source"] == "analyst_inference", "symbol"]),
    shareholding_error=shp_error,
    source_through=dict(panel=_d(last_dt), pnl_quarterly_filing=_d(_pnl_through),
                        filing_scores_event=_d(_fs_through), shareholding_quarter=_d(_shp_through)),
)
gap_lines = [f"core band {data_gaps['core_band_names']:,} names on {last_dt.date()}"]
for k in ("core_no_industry_label", "core_no_ret60", "core_no_ret252", "core_in_small_industry"):
    v = data_gaps[k]
    gap_lines.append(f"  {k:<26s} {v['n']:>5,}  ({v['effect']})")
    for s in v.get("sample", [])[:5]:
        gap_lines.append(f"      {s['symbol']:<14} ADV {s['adv_cr']:>8.2f}cr  price Rs {s['price_raw']:>9.2f}")
gap_lines.append(f"  as-of rows with no adjustment factor (raw price assumed = adjusted): "
                 f"{data_gaps['asof_rows_missing_adj_factor']}")
for k in ("leaders_no_ttm", "leaders_no_industry_pe", "leaders_no_promoter_delta", "leaders_no_dma200",
          "leaders_analyst_inferred_industry"):
    gap_lines.append(f"  {k:<34s} {', '.join(data_gaps[k]) or '-'}")
if shp_error:
    gap_lines.append(f"  shareholding NOT LOADED: {shp_error}")
gap_lines.append("  sources through: " + " · ".join(f"{k} {v}" for k, v in data_gaps["source_through"].items()))
print("\nDATA GAPS\n" + "\n".join(gap_lines), flush=True)


def _f(v):
    return None if v is None or pd.isna(v) else round(float(v), 4)


names = [dict(rank=i + 1, symbol=r["symbol"], industry=r["ind"], industry_source=r["ind_source"],
              close=_f(r["close"]), price_raw=_f(r["px_raw"]), adv_cr=_f(r["adv"]),
              grp60=_f(r["grp60"]), own60=_f(r["ret60"]), own252=_f(r["ret252"]),
              run=("extended" if pd.notna(r["ret252"]) and r["ret252"] > 0.50 else "fresh"),
              pe=_f(r["pe"]), pe_ind=_f(r["pe_ind"]), ttm_positive=None if pd.isna(r["ttm"]) else bool(r["ttm"] > 0),
              prom_d=_f(r["prom_d"]), neg90=int(r["neg90"]), pos90=int(r["pos90"]),
              above200=bool(r["above200"]), band="core", cost_rt_pct=_f(r["cost_rt_pct"]), rtw=int(r["rtw"]))
         for i, (_, r) in enumerate(lead.iterrows())]
appendix = dict(
    status="UNTESTED — informational only; no registered backtest covers this band; NOT scored by "
           "score_leader_sleeve.py; NOT for sizing",
    universe=f"ADV20 >= Rs {EXP_ADV_CR}cr & raw price > Rs {EXP_PX:.0f} (the pre-2026-09-27 screen universe)",
    rule="same heat/rank/EXTENDED rule as the production cell, computed on this band",
    hot_industries={i: round(float(r.grp_ret60), 4) for i, r in hot_x.iterrows()},
    names=[dict(symbol=r["symbol"], industry=r["ind"], industry_source=r["ind_source"],
                grp60_expanded=_f(r["grp60"]), own60=_f(r["ret60"]), own252=_f(r["ret252"]),
                adv_cr=_f(r["adv"]), price_raw=_f(r["px_raw"]), core_band=bool(r["core_band"]),
                cost_rt_pct=_f(rp.cost_rt(float(r["adv"])) * 100), why_not_production=_why(r))
           for _, r in appx.iterrows()])
if args.preview:
    print("\nPREVIEW — nothing written", flush=True)
    sys.exit(0)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(dict(
    screen_id=TAG, data_through=str(last_dt.date()), entry="next session open after data_through",
    entry_skip_rule=ENTRY_SKIP_RULE,
    hold_td=126, cost_rt_pct=0.5, exit="TIME (126td close); no stops — trailing stops killed (sell the trough)",
    sizing="PAPER", paper_rule=PAPER_RULE, extended_only=EXTENDED_ONLY, industry_map=INDUSTRY_MAP,
    registered_cell=REGISTERED_CELL, era="conf" if last_dt >= ERA_SPLIT else "disc",
    universe=f"core band: ADV20 >= Rs {CORE_ADV_CR:.0f}cr & raw price > Rs {CORE_PX:.0f} (heat, rank and names)",
    screen_version="2026-09-27 (core-band heat/rank; earlier screens used the expanded band)",
    hot_industries={i: round(float(r.grp_ret60), 4) for i, r in hot.iterrows()},
    names=names, untested_expanded_appendix=appendix, data_gaps=data_gaps), indent=1))

rep = ROOT / f"reports/theme_leaders_{TAG}.md"
rep.write_text(f"# Theme > Industry > Right-to-Win leaders — {last_dt.date()}\n\n"
               f"**{PAPER_RULE}**\n\n"
               f"Registered cell: {REGISTERED_CELL}. Entry: {ENTRY_SKIP_RULE}.\n\n"
               "Cell evidence: hot-industry top-3 leaders = 2.1-2.4x P(2x/126td) both eras; "
               "conf-era LEADER & pe_ind<1 = 12.4% P(2x), +18.4% mean 6-mo. Venture/paper "
               "sizing only; median troughs -14 to -19%. Exit is TIME (126td), no stops. "
               f"Machine-readable: logs/leader_sleeve/{OUT_JSON.name}\n\n```\n" + "\n".join(out_lines) + "\n```\n\n"
               f"## Appendix — UNTESTED expanded band (ADV>={EXP_ADV_CR}cr & raw price>{EXP_PX:.0f})\n\n"
               "Informational only. No registered backtest covers this band; these names are not scored "
               "forward and are not for sizing.\n\n```\n" + "\n".join(appx_lines) + "\n```\n\n"
               "## Data gaps\n\n```\n" + "\n".join(gap_lines) + "\n```\n")
print(f"\nwrote {rep}\nwrote {OUT_JSON}", flush=True)
print("THEME LEADERS SCREEN COMPLETE", flush=True)
