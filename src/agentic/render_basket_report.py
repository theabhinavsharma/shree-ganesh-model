"""Render the two human-facing basket reports from the committed basket JSON.

  BASKET report (once per 15d window, at emit):   reports/basket_report_<DATE>.md
    Stock | Buy range | Target | Confidence | Rationale micro | Rationale macro | ETA
  DAILY report (every session while the basket is live): reports/daily_actions_<DATE>.md
    Stock | Action (BUY / HOLD / SELL PART / SELL / SKIP) | + the same columns, plus
    entry, last close, P&L, SL and days held so the action is auditable.

Actions are the C2 contract replayed day-by-day on the price panel (same rules as
score_basket_outcomes.py: entry next-open at or below buy_high, +5% touch -> sell half
and trail the rest at +2.5%, vol-scaled SL, day-15 timeout). A fresh distribution day
(<= -3% on > 1.5x 20d volume) is surfaced as SELL? (manual eviction rule, human confirms).

ETA is VOL-IMPLIED, not calibrated: median first-passage time of a driftless walk to
+5% in the stock's own 20d daily vol, ~0.45 * (0.05 / vol)^2 sessions, capped at 15.
Read it as "how fast this name normally moves 5%", not a forecast. Calibrating it on
the 4,222-trade backtest is a ledgered TODO.

Usage:
  python3 src/agentic/render_basket_report.py                 # latest basket, both reports
  python3 src/agentic/render_basket_report.py --basket live_predictions/2026-09-08_15d5pct.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SGM_ROOT", "/Users/abhinavs./Documents/Zoom"))  # env override for off-Mac runs
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
INDUSTRY = ROOT / "data/derived/industry_panel.parquet"
MACRO = ROOT / "data/derived/macro_panel.parquet"
NEWS = ROOT / "data/derived/news_event_features.parquet"
SECTOR_SRC = [ROOT / "data/derived/confluence_picks.parquet", ROOT / "data/derived/paper_trading_ledger.parquet"]
ANN = ROOT / "data/events_full_history/normalized/stock_announcements.parquet"
SCREENER = ROOT / "data/derived/screener_fundamentals.parquet"
FUND_SNAP = ROOT / "data/derived/fundamentals_snapshot.parquet"
PIT = ROOT / "data/derived/pit_history.parquet"
BLOCK = ROOT / "data/derived/block_deals_history.parquet"
SUPERSTAR = ROOT / "data/derived/superstar_holdings.parquet"
CA = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"
PROFILES = ROOT / "data/derived/company_profiles.json"   # analyst notes, labelled as such
QUAL_WINDOW_DAYS = 45
# Only these NSE wordings are a real order win; "Spurt in Volume" and "order position"
# updates are attention flags (2026-07-07 lesson).
REAL_ORDER_RE = r"bagging|receiving of order|receipt of order|awarded|letter of award|work order|purchase order"
RED_RE = r"resign|pledge|default|show cause|penalty|sebi order|search|raid|fraud|impairment|delay in|non-compliance"


def _latest_basket() -> Path:
    files = sorted((ROOT / "live_predictions").glob("*_15d5pct.json"))
    if not files:
        raise SystemExit("no *_15d5pct.json in live_predictions/")
    return files[-1]


def _sector_map() -> dict[str, str]:
    out: dict[str, str] = {}
    if NEWS.exists():
        h = pd.read_parquet(NEWS, columns=["symbol", "industry_hint"]).dropna().drop_duplicates("symbol")
        out.update({s: f"{v}" for s, v in zip(h["symbol"], h["industry_hint"]) if v})
    for src in SECTOR_SRC:
        if src.exists():
            m = pd.read_parquet(src, columns=["symbol", "sector"]).dropna().drop_duplicates("symbol")
            for s, sec in zip(m["symbol"], m["sector"]):
                out.setdefault(s, sec)
    return out


def _profiles() -> dict:
    try:
        return json.loads(PROFILES.read_text()) if PROFILES.exists() else {}
    except Exception:
        return {}


def _nz(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except Exception:
        return None


def _eta_days(dvol: float | None) -> str:
    if not dvol or dvol <= 0:
        return "n/a"
    d = 0.45 * (0.05 / dvol) ** 2
    if d > 15:
        return ">15d (needs a catalyst)"
    lo, hi = max(1, int(d * 0.6)), min(15, int(d * 1.6) + 1)
    return f"day {lo}-{hi}"


def _quant(p: dict, tape: dict, fund: dict) -> str:
    """Numbers only — what the engines, bands, tape and balance sheet say."""
    tier = ("3-engine consensus" if p["engines_count"] >= 3 else
            "2-engine consensus" if p["engines_count"] == 2 else "single-engine (ML-led)")
    bits = [f"{tier} · z-band fit {p['band_fit']:.1f}/3 (RSI {p['rsi']:.0f}, 20d {p['return_20d_pct']:+.1f}%, 5d {p['ret_5d_pct']:+.1f}%)",
            f"ML {p['ml_score']:.2f} ({'honest zone' if p['ml_score'] <= 0.75 else 'above honest zone — discounted'}) · cs {p['cs_score']:.2f}",
            f"ADV ₹{p['adv_cr']:.0f}cr · 20d vol {tape.get('dvol', 0)*100:.1f}%/d → SL {p['sl_pct']:+.1f}%"]
    st = []
    if tape.get("above_50") is not None:
        st.append(("above" if tape["above_50"] else "BELOW") + " 50DMA")
        st.append(("above" if tape["above_200"] else "BELOW") + " 200DMA")
    if tape.get("dist_days"):
        st.append(f"DISTRIBUTION x{tape['dist_days']} last 15d")
    if tape.get("deliv") is not None:
        st.append(f"delivery {tape['deliv']*100:.0f}% (20d avg {tape['deliv20']*100:.0f}%)")
    if st:
        bits.append("tape: " + ", ".join(st))
    f = []
    if fund.get("pe") is not None:
        f.append(f"PE {fund['pe']:.0f}" + (f" vs sector {fund['sector_pe']:.0f}" if fund.get("sector_pe") else ""))
    if fund.get("roce") is not None:
        f.append(f"ROCE {fund['roce']:.0f}%" + (f" / ROE {fund['roe']:.0f}%" if fund.get("roe") is not None else ""))
    if fund.get("sales3y") is not None:
        f.append(f"3y sales {fund['sales3y']:+.0f}% CAGR" + (f", profit {fund['profit3y']:+.0f}%" if fund.get("profit3y") is not None else ""))
    if fund.get("qoq_pat") is not None:
        f.append(f"last Q PAT {fund['qoq_pat']:+.0f}% QoQ")
    if fund.get("mcap"):
        f.append(f"mcap ₹{fund['mcap']/1000:.1f}k cr")
    if f:
        bits.append("fundamentals: " + ", ".join(f) + f" (as of {fund.get('asof', '?')})")
    return " · ".join(bits)


def _qual(sym: str, company: str, industry: str | None, ann: pd.DataFrame, pit: pd.DataFrame,
          block: pd.DataFrame, stars: list[str], thru: pd.Timestamp) -> str:
    """The business case, from actual filings — never from the score."""
    head = company or sym
    if industry:
        head += f" ({industry})"
    prof = _profiles().get(sym)
    lines = [f"{prof} [analyst note]" if prof else head]
    if ann is not None and not ann.empty:
        a = ann[ann["event_date"] >= thru - pd.Timedelta(days=QUAL_WINDOW_DAYS)].sort_values("event_date")
        txt = (a["description"].fillna("") + " " + a["attachment_text"].fillna("")).str.lower()
        real_orders = a[txt.str.contains(REAL_ORDER_RE, regex=True) & ~txt.str.contains("spurt|order position|monthly execution", regex=True)]
        pos_updates = a[txt.str.contains("order position|monthly execution|order book", regex=True)]
        results = a[a["event_category"].isin(["results"]) & txt.str.contains("financial result|quarter ended|un-audited|unaudited", regex=True)]
        concall = a[a["event_category"] == "investor_communication"]
        approvals = a[a["event_category"].isin(["approval", "mna", "fund_raise"]) & ~txt.str.contains("esop|esos", regex=True)]
        reds = a[txt.str.contains(RED_RE, regex=True) & ~txt.str.contains("revokation|revocation|release of pledge", regex=True)]
        bonus = a[txt.str.contains("bonus|split|record date|dividend", regex=True) & ~txt.str.contains("esop", regex=True)]
        agm = a[txt.str.contains("annual general meeting", regex=True)]

        def _d(df): return ", ".join(sorted({x.strftime("%d-%b") for x in df["event_date"]}))
        def _first(df, n=110):
            r = df.iloc[-1]; t = str(r["attachment_text"] or r["description"]).replace("\n", " ").strip()
            return t[:n] + ("…" if len(t) > n else "")
        cat = []
        if len(real_orders):
            cat.append(f"ORDER WIN(s) {_d(real_orders)}: {_first(real_orders)}")
        if len(pos_updates):
            cat.append(f"order-position update {_d(pos_updates)} (execution disclosure, not a new win)")
        if len(approvals):
            cat.append(f"board/M&A {_d(approvals)}: {_first(approvals)}")
        if len(results):
            cat.append(f"results filed {_d(results)}")
        if len(concall):
            cat.append(f"concall/investor meet {_d(concall)}")
        if len(bonus):
            cat.append(f"corporate action {_d(bonus)}: {_first(bonus, 80)}")
        if len(agm):
            cat.append(f"AGM {_d(agm)}")
        if len(reds):
            cat.append(f"⚠ RED {_d(reds)}: {_first(reds, 90)}")
        n_other = len(a) - len(set(real_orders.index) | set(pos_updates.index) | set(approvals.index) | set(results.index) | set(concall.index) | set(bonus.index) | set(agm.index) | set(reds.index))
        lines.append(f"filings {QUAL_WINDOW_DAYS}d ({len(a)}): " + ("; ".join(cat) if cat else "routine only") + (f"; {n_other} routine" if cat and n_other > 0 else ""))
    else:
        lines.append(f"no filings in {QUAL_WINDOW_DAYS}d")
    if pit is not None and not pit.empty:
        q = pit[pit["d"] >= thru - pd.Timedelta(days=90)]
        if not q.empty:
            prom = q[q["personCategory"].fillna("").str.contains("Promoter", case=False)]
            buys = prom[prom["tdpTransactionType"].fillna("").str.contains("Buy|Acquisition", case=False)]
            sells = prom[prom["tdpTransactionType"].fillna("").str.contains("Sell|Disposal", case=False)]
            pl = []
            if len(buys): pl.append(f"promoter BUYS x{len(buys)}")
            if len(sells): pl.append(f"promoter SELLS x{len(sells)} ⚠")
            if pl: lines.append("insider 90d: " + ", ".join(pl))
    if block is not None and not block.empty:
        b = block[block["dt"] >= thru - pd.Timedelta(days=60)]
        if not b.empty:
            lines.append("block deals 60d: " + "; ".join(f"{r['BD_BUY_SELL']} {int(r['BD_QTY_TRD']):,} @ {r['BD_TP_WATP']:.0f} ({str(r['BD_CLIENT_NAME'])[:28]})" for _, r in b.tail(3).iterrows()))
    if stars:
        lines.append("known holders: " + ", ".join(stars[:3]))
    return " · ".join(lines)


def _macro(sym: str, regime: str, sec_map: dict, ind: pd.DataFrame, mac: pd.Series) -> str:
    bits = [f"regime {regime}"]
    sec = sec_map.get(sym)
    if sec and not ind.empty and sec in set(ind["sector"]):
        row = ind[ind["sector"] == sec].sort_values("trade_date").tail(1)
        if not row.empty:
            r = row.iloc[0]
            bits.append(f"{sec}: 20d {r['sector_20d_ret']*100:+.1f}% (RS vs Nifty {r['rs_20d']*100:+.1f}%), "
                        f"breadth>50dma {r['sector_breadth_50']*100:.0f}%")
    elif sec:
        bits.append(f"industry: {sec} (no sector panel)")
    else:
        bits.append("industry: unmapped")
    if mac is not None:
        parts = []
        for col, lab, scale in (("breadth_50", "mkt breadth>50dma", 100), ("median_return_20d", "median stock 20d", 100),
                                ("usdinr", "USDINR", 1), ("us_vix", "VIX", 1)):
            v = mac.get(col)
            if v is not None and pd.notna(v):
                parts.append(f"{lab} {float(v)*scale:.1f}{'%' if scale == 100 else ''}")
        if parts:
            bits.append(", ".join(parts))
    return " · ".join(bits)


def _replay(p: dict, g: pd.DataFrame) -> dict:
    """Day-by-day C2 state for one pick. g = sessions AFTER data_through, ascending."""
    st = {"entry": None, "fill_day": None, "half": False, "trail": None, "status": "PENDING",
          "exit_px": None, "days": 0, "last": None, "hi": None, "action": "BUY", "note": ""}
    if g.empty:
        st["note"] = f"AMO limit {p['buy_high']:.2f} into pre-open auction"
        return st
    d0 = g.iloc[0]
    if pd.isna(d0["open"]) or d0["open"] > p["buy_high"]:
        st.update(status="SKIPPED", action="SKIP",
                  note=f"opened {d0['open']:.2f} above zone high {p['buy_high']:.2f} — never chase; weight → best reserve")
        return st
    ep = float(min(d0["open"], p["buy_high"]))
    sl = ep * (1 + p["sl_pct"] / 100)
    tgt = ep * 1.05
    st.update(entry=round(ep, 2), fill_day=str(d0["trade_date"].date()), status="OPEN")
    n = len(g)
    for i, (_, d) in enumerate(g.iterrows(), start=1):
        st["days"], st["last"] = i, float(d["close"])
        st["hi"] = max(st["hi"] or 0, float(d["high"]))
        if not st["half"]:
            if d["low"] <= sl:                                   # SL-first on both-touch
                st.update(status="STOPPED", action="SELL" if i == n else "CLOSED", exit_px=round(sl, 2),
                          note=f"SL {sl:.2f} hit day {i} ({p['sl_pct']:+.1f}%)"); return st
            if d["high"] >= tgt:
                st.update(half=True, trail=ep * 1.025,
                          note=f"+5% touched day {i}: half booked at {tgt:.2f}, rest trails {ep*1.025:.2f}")
                if i == n:
                    st["action"] = "SELL PART"; return st
                continue
        else:
            if d["low"] <= st["trail"]:
                st.update(status="TRAIL_OUT", action="SELL" if i == n else "CLOSED", exit_px=round(st["trail"], 2),
                          note=f"trail {st['trail']:.2f} hit day {i} after half booked"); return st
        if i >= 15:
            st.update(status="TIMEOUT", action="SELL" if i == n else "CLOSED", exit_px=round(float(d["close"]), 2),
                      note="day-15 timeout: exit remainder at market"); return st
        vol_x = d.get("volume_vs_20d")
        if d["close"] / d["prev_close"] - 1 <= -0.03 and vol_x is not None and pd.notna(vol_x) and vol_x > 1.5 and i == n:
            st.update(action="SELL?", note=f"fresh distribution day ({(d['close']/d['prev_close']-1)*100:+.1f}% on {vol_x:.1f}x vol) — manual eviction rule, confirm")
            return st
    st["action"] = "HOLD"
    if st["half"]:
        st["note"] = f"half booked; trailing at {st['trail']:.2f}"
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basket", type=Path, default=None)
    args = ap.parse_args()
    bp = args.basket or _latest_basket()
    b = json.loads(bp.read_text())
    date, thru = b["as_of_date"], pd.Timestamp(b["data_through"])
    names = [p["symbol"] for p in b["picks"]] + [r["symbol"] for r in b.get("reserves", [])]

    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "open", "high", "low", "close", "prev_close",
                                          "return_1d", "volume_vs_20d"], filters=[("symbol", "in", names)])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    dvol = {s: g[g["trade_date"] <= thru]["return_1d"].tail(20).std() for s, g in px.groupby("symbol")}
    sec_map = _sector_map()
    ind = pd.read_parquet(INDUSTRY) if INDUSTRY.exists() else pd.DataFrame()
    mac = pd.read_parquet(MACRO).sort_values("trade_date").iloc[-1] if MACRO.exists() else None
    last_session = px["trade_date"].max().date()

    # --- qualitative sources (all optional; each degrades to nothing, never to a guess)
    def _opt(path, **kw):
        try:
            return pd.read_parquet(path, **kw) if path.exists() else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    ann_all = _opt(ANN, filters=[("symbol", "in", names)])
    if not ann_all.empty:
        ann_all["event_date"] = pd.to_datetime(ann_all["event_date"], errors="coerce")
    pit_all = _opt(PIT, columns=["symbol", "intimDt", "personCategory", "tdpTransactionType"])
    if not pit_all.empty:
        pit_all = pit_all[pit_all["symbol"].isin(names)].copy()
        pit_all["d"] = pd.to_datetime(pit_all["intimDt"], errors="coerce")
    blk_all = _opt(BLOCK)
    if not blk_all.empty:
        blk_all = blk_all[blk_all["BD_SYMBOL"].isin(names)].copy()
        blk_all["dt"] = pd.to_datetime(blk_all["BD_DT_DATE"], errors="coerce")
    stars_all = _opt(SUPERSTAR, columns=["symbol", "investor_name"])
    stars = {s: sorted(set(g["investor_name"])) for s, g in stars_all.groupby("symbol")} if not stars_all.empty else {}
    company = {}
    ca_names = _opt(CA, columns=["symbol", "company_name"])
    if not ca_names.empty:
        company = dict(ca_names.drop_duplicates("symbol").values)
    ind_hint = {}
    if NEWS.exists():
        h = pd.read_parquet(NEWS, columns=["symbol", "industry_hint"]).dropna().drop_duplicates("symbol")
        ind_hint = dict(h.values)
    scr = _opt(SCREENER); snap = _opt(FUND_SNAP)
    def _fund(s):
        out = {}
        if not scr.empty:
            r = scr[scr["symbol"] == s].sort_values("fetch_date").tail(1)
            if len(r):
                r = r.iloc[0]; out.update(pe=_nz(r.get("pe")), roce=_nz(r.get("roce")), roe=_nz(r.get("roe")),
                                          sales3y=_nz(r.get("compounded_sales_growth_3_years")), profit3y=_nz(r.get("compounded_profit_growth_3_years")),
                                          mcap=_nz(r.get("market_cap_cr")), asof=str(r.get("fetch_date"))[:10])
        if not snap.empty:
            r = snap[snap["symbol"] == s].sort_values("fetch_date").tail(1)
            if len(r):
                r = r.iloc[0]; out.setdefault("pe", _nz(r.get("pe"))); out["sector_pe"] = _nz(r.get("sector_pe")); out["qoq_pat"] = _nz(r.get("qoq_pat_growth"))
        return out
    px_full = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "return_1d", "volume_vs_20d", "sma_50", "sma_200", "delivery_pct", "avg_delivery_pct_20d"], filters=[("symbol", "in", names)])
    px_full["trade_date"] = pd.to_datetime(px_full["trade_date"])
    def _tape(s):
        g = px_full[(px_full["symbol"] == s) & (px_full["trade_date"] <= thru)].sort_values("trade_date")
        if g.empty: return {}
        r = g.iloc[-1]; last15 = g.tail(15)
        dist = int(((last15["return_1d"] <= -0.03) & (last15["volume_vs_20d"] > 1.5)).sum())
        return dict(dvol=float(g["return_1d"].tail(20).std() or 0), above_50=bool(r["close"] > r["sma_50"]) if pd.notna(r["sma_50"]) else None,
                    above_200=bool(r["close"] > r["sma_200"]) if pd.notna(r["sma_200"]) else None, dist_days=dist,
                    deliv=_nz(r.get("delivery_pct")), deliv20=_nz(r.get("avg_delivery_pct_20d")))

    rows_b, rows_d, qual_short = [], [], {}
    replay = {}
    for p in b["picks"] + b.get("reserves", []):
        replay[p["symbol"]] = _replay(p, px[(px["symbol"] == p["symbol"]) & (px["trade_date"] > thru)].head(15))
    skipped = [p["symbol"] for p in b["picks"] if replay[p["symbol"]]["action"] == "SKIP"]
    deployed = {}   # reserve symbol -> pick it replaces
    for sk in skipped:
        for r in sorted(b.get("reserves", []), key=lambda r: r.get("rank", 99)):
            if r["symbol"] not in deployed and replay[r["symbol"]]["entry"] is not None:
                deployed[r["symbol"]] = sk; break
    for p in b["picks"] + [dict(r, role=r.get("role", "RESERVE")) for r in b.get("reserves", [])]:
        s = p["symbol"]
        role = p.get("role", f"#{p['rank']}")
        tape = _tape(s)
        quant = _quant(p, tape, _fund(s))
        qual = _qual(s, company.get(s), ind_hint.get(s) or sec_map.get(s), ann_all[ann_all["symbol"] == s] if not ann_all.empty else None,
                     pit_all[pit_all["symbol"] == s] if not pit_all.empty else None,
                     blk_all[blk_all["BD_SYMBOL"] == s] if not blk_all.empty else None, stars.get(s, []), thru)
        macro = _macro(s, b["regime_gate"], sec_map, ind, mac)
        eta = _eta_days(tape.get("dvol"))
        rows_b.append(f"| {s} ({role}) | {p['buy_low']:.2f}–{p['buy_high']:.2f} | {p['target_5pct']:.2f} (+5%) | "
                      f"{p['confidence']:.2f} | {quant} | {qual} | {macro} | {eta} |")
        qual_short[s] = qual.split(" · ")[1] if " · " in qual else qual
        st = replay[s]
        if p.get("role"):                                        # reserves only deploy when a pick gaps
            if s in deployed:
                st["note"] = f"DEPLOYED for {deployed[s]} (gapped) — " + st["note"]
            else:
                st["action"], st["note"] = "STANDBY", ("shadow: " + st["note"] if st["note"] else "")
        pnl = ""
        if st["entry"] is not None and st["last"] is not None:
            mark = st["exit_px"] if st["exit_px"] is not None else st["last"]
            rest = (mark / st["entry"] - 1) * 100
            pnl = f"{(5.0 + rest) / 2 if st['half'] else rest:+.1f}%"
        rows_d.append(f"| {s} ({role}) | **{st['action']}** | {st['entry'] or '—'} | {st['last'] or '—'} | {pnl} | "
                      f"{p['sl_3pct']:.2f} | {p['target_5pct']:.2f} | {st['days']} | {st['note']} | {qual_short.get(s, '')} |")

    hdr = (f"_basket {bp.name} · data through {b['data_through']} · regime {b['regime_gate']} · "
           f"exit contract {b['exit_contract'].split('(')[0].strip()}_\n\n")
    basket_md = (f"# Basket report — {date}\n\n{hdr}"
                 "| Stock | Buy range | Target | Confidence | Rationale — QUANT | Rationale — QUAL (filings, insiders, holders) | Rationale — MACRO | ETA (vol-implied) |\n"
                 "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows_b) +
                 "\n\n**Entry:** " + " ".join(b["entry_rules"][:2]) +
                 "\n\n**Exit (C2):** " + " · ".join(b["exit_rules"][:3]) +
                 "\n\n_ETA = median first-passage of a driftless walk to +5% in the stock's own 20d vol; "
                 "not a calibrated forecast. Confidence = min(engines,3)×1.5 + band_fit + honest-zone-ML×2 + cs×0.5._\n")
    daily_md = (f"# Daily actions — session {last_session}\n\n{hdr}"
                "| Stock | Action | Entry | Last close | P&L | SL | Target | Day | Why (contract) | Qual — latest filings |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows_d) +
                "\n\n_BUY = AMO limit at buy_high · HOLD = inside contract · SELL PART = +5% touched, book half · "
                "SELL = SL / trail / day-15 · SELL? = fresh distribution day, manual rule · SKIP = gapped above zone · "
                "STANDBY = reserve, deploys only if a pick gaps._\n")
    out_b = ROOT / f"reports/basket_report_{date}.md"
    out_d = ROOT / f"reports/daily_actions_{last_session}.md"
    out_b.write_text(basket_md); out_d.write_text(daily_md)
    print(f"wrote {out_b.relative_to(ROOT)}\nwrote {out_d.relative_to(ROOT)}")
    print(daily_md)


if __name__ == "__main__":
    main()
