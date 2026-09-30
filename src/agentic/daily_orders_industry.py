"""DAILY ORDERS + INDUSTRY HEAT digest (2026-09-28; user: "fetch orders data and industry evaluations daily — new order
info for anyone might give us a 2xer or a high uptick in the middle").

Monitoring, not a validated signal. Three order measures were tested (EXP-2026-09-27-hot-x-material-order,
EXP-2026-09-28-order-backlog, the order-book announcement event study) and none held in both eras; the average
announcement-day move is about zero. This digest makes every listed company's new orders visible the day they are filed,
sized against point-in-time TTM revenue, next to the stock's trend and its industry's heat, and it accumulates a forward
record (data/derived/order_daily.parquet) that can be re-tested later.

Inputs
  tmp/from_scratch_7d_run/alt/corp_announcements.parquet  NSE announcements refreshed daily by refresh_announcements.py
                                                           (keeps the attachment URL and dissemination time)
  order filings = event_materiality_study.CATS, first match in {order, tender_L1} on desc + attchmntText (as the crawls)
  attachment text through the NSE session (pdftotext; tesseract OCR for scanned PDFs when installed)
  amount = parse_order_amounts.extract_order_amount (order-attached amount; USD at the historical FRED rate)
  revenue = test_hot_order_combo.build_ttm / asof_ttm (point-in-time TTM, pnl_quarterly manifest units)
  trend + industry heat = price panel (core band: 20d ADV >= Rs5cr, then-traded close > Rs50; heat = mean own 60-session
  return of the industry's core names, >= 5 names; nse4 industry map + analyst analogs, as the leader screen)
Outputs
  data/derived/order_daily.parquet (+ .manifest.json): one row per order filing (symbol, seq_id), appended, never rewritten
  reports/daily_orders_industry_<date>.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
from event_materiality_study import CATS  # noqa: E402
from fetch_order_book import REF, blob_text  # noqa: E402
from parse_order_amounts import extract_order_amount, load_usdinr, FX_TOL  # noqa: E402
from test_hot_order_combo import asof_ttm, build_ttm  # noqa: E402
from src.ingest.nse.api import _request_headers, _request_with_retries  # noqa: E402
from src.ingest.nse.session import build_session  # noqa: E402

RAW = ROOT / "tmp/from_scratch_7d_run/alt/corp_announcements.parquet"
OUT = ROOT / "data/derived/order_daily.parquet"
LOOKBACK_D, MATERIAL, HEAT_MIN_N = 7, 0.15, 5


def order_filings(since: pd.Timestamp) -> pd.DataFrame:
    a = pd.read_parquet(RAW, columns=["symbol", "desc", "attchmntText", "attchmntFile", "an_dt", "exchdisstime", "seq_id", "sm_name"])
    a["ts"] = pd.to_datetime(a["exchdisstime"], errors="coerce").fillna(pd.to_datetime(a["an_dt"], errors="coerce"))
    a = a[a["ts"] >= since].copy()
    t = a["desc"].fillna("") + " " + a["attchmntText"].fillna("")
    first = pd.Series(None, index=a.index, dtype=object)
    for c, p in CATS.items():
        m = first.isna() & t.str.contains(p, case=False, regex=True)
        first[m] = c
    a["cat"] = first
    a = a[a["cat"].isin(["order", "tender_L1"])].copy()
    a["seq_id"] = a["seq_id"].astype(str)
    return a.drop_duplicates(["symbol", "seq_id"]).sort_values("ts")


def fetch_text(s, url: str) -> tuple[str, str]:
    if not isinstance(url, str) or not url.startswith("http"):
        return "", "NO_ATTACHMENT"
    try:
        resp = _request_with_retries(s, url, request_headers=_request_headers(s, referer=REF), referer=REF, timeout=60)
        txt = re.sub(r"\s+", " ", blob_text(url, resp.content))
        if not txt.strip() and url.lower().endswith(".pdf"):
            try:
                from ocr_order_filings import ocr_pdf
                txt = ocr_pdf(resp.content)
                return txt, "OCR" if txt else "NO_TEXT"
            except Exception:
                return "", "NO_TEXT"
        return txt, "OK" if txt else "NO_TEXT"
    except Exception as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        return "", f"HTTP_{code}" if code else f"ERR_{type(e).__name__}"


def market_state() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Latest-session trend per symbol and the industry heat table (today and 5 sessions ago)."""
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=560)
    px = rp.load_panel(["low", "close", "sma_50", "sma_200", "avg_traded_value_20d", "price_adjustment_factor_to_present"],
                       filters=[("trade_date", ">=", start)])
    g = px.groupby("symbol")
    px["ret60"] = g["close"].pct_change(60, fill_method=None)
    px["ret252"] = g["close"].pct_change(252, fill_method=None)
    px["lo252"] = g["low"].transform(lambda x: x.rolling(252, min_periods=60).min())
    px["core"] = (px["avg_traded_value_20d"] / 1e7 >= 5) & (rp.raw_price(px, "close") > 50)
    imap = sp.industry_maps()["analogs"]
    px["ind"] = px["symbol"].map(imap)
    cal = sorted(px["trade_date"].unique())
    d, d5 = cal[-1], cal[-6]

    def heat(day):
        c = px[(px["trade_date"] == day) & px["core"] & px["ind"].notna() & px["ret60"].notna()]
        h = c.groupby("ind").agg(heat=("ret60", "mean"), n=("ret60", "size"),
                                 above200=("close", lambda s: float((s > c.loc[s.index, "sma_200"]).mean())))
        h = h[h["n"] >= HEAT_MIN_N]
        h["pct"] = h["heat"].rank(pct=True)
        return h
    H, H5 = heat(d), heat(d5)
    H["pct_5d_ago"] = H5["pct"].reindex(H.index)
    H["heat_5d_ago"] = H5["heat"].reindex(H.index)
    last = px[px["trade_date"] == d].set_index("symbol")
    with np.errstate(invalid="ignore", divide="ignore"):
        last["off_low"] = last["close"] / last["lo252"] - 1
        last["trend"] = ((last["off_low"] >= 0.5) & (last["ret252"] >= 0.30) & (last["close"] > last["sma_200"])
                         & (last["sma_50"] > last["sma_200"]))
    last["heat_pct"] = last["ind"].map(H["pct"])
    return last, H.sort_values("pct", ascending=False), pd.Timestamp(d)


def held_names() -> dict:
    out = {}
    for folder, tag in (("logs/leader_sleeve", "sleeve"), ("logs/model_screen", "model")):
        for p in (ROOT / folder).glob("screen_*.json"):
            sc = json.loads(p.read_text())
            if pd.Timestamp(sc["data_through"]) >= pd.Timestamp.today() - pd.Timedelta(days=190):   # 126 sessions ~ 6 months
                for n in sc["names"]:
                    out.setdefault(n["symbol"], []).append(f"{tag} {sc['screen_id']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_D)
    a = ap.parse_args()
    t0 = time.time()
    since = pd.Timestamp.today().normalize() - pd.Timedelta(days=a.lookback_days)
    O = order_filings(since)
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["symbol", "seq_id"])
    done = set(zip(old["symbol"], old["seq_id"].astype(str)))
    todo = O[[(s, q) not in done for s, q in zip(O["symbol"], O["seq_id"])]]
    fx = load_usdinr().set_index("fx_date")["usdinr"]
    rows = []
    if len(todo):
        s = build_session(warm=True, referer=REF)
        for r in todo.itertuples():
            txt, status = fetch_text(s, r.attchmntFile)
            w = fx.loc[r.ts.normalize() - FX_TOL: r.ts.normalize()]
            amt = extract_order_amount(r.attchmntText, txt, float(w.iloc[-1]) if len(w) else np.nan)
            rows.append(dict(symbol=r.symbol, seq_id=r.seq_id, company=r.sm_name, ts=r.ts, cat=r.cat, desc=r.desc,
                             headline=str(r.attchmntText)[:500], url=r.attchmntFile, text_status=status, text_chars=len(txt),
                             amount_cr=amt["order_amount_cr"], amount_method=amt["amount_method"],
                             amount_confidence=amt["amount_confidence"], amount_snippet=amt["amount_snippet"],
                             fetched=pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).isoformat(timespec="seconds")))   # IST, like ts
            time.sleep(0.5)
    N = pd.DataFrame(rows)
    if len(N):
        T, _ = build_ttm(None)
        N["rev_ttm_cr"] = asof_ttm(N, "ts", T)
        N["pct_of_rev"] = N["amount_cr"] / N["rev_ttm_cr"]
        ALL = pd.concat([old, N], ignore_index=True) if len(old) else N
        ALL.to_parquet(OUT, index=False)
    else:
        ALL = old
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="order_daily", path=str(OUT.relative_to(ROOT)), rows=len(ALL), key=["symbol", "seq_id"],
        producer="src/agentic/daily_orders_industry.py", source="NSE corporate announcements (order / L1 filings) + attachments",
        columns=dict(ts="dissemination time (IST, naive)", fetched="time the attachment was fetched (IST, naive; before 2026-09-29 20:45 ET rows were stored in US Eastern and were converted)", amount_cr="order-attached amount, Rs crore (NaN = not stated / not parsed)",
                     rev_ttm_cr="point-in-time TTM revenue known at ts, Rs crore (build_ttm, pnl_quarterly manifest units)",
                     pct_of_rev="amount_cr / rev_ttm_cr as a fraction (0.15 = 15% of a year's revenue)",
                     text_status="OK | OCR | NO_TEXT | NO_ATTACHMENT | HTTP_<code> | ERR_<type>"),
        status="monitoring record, not a validated signal", updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))

    last, H, d = market_state()
    held = held_names()
    rec = ALL[pd.to_datetime(ALL["ts"]) >= since].copy() if len(ALL) else ALL
    lines = [f"# Daily orders + industry heat — {d.date()}", "",
             "**Monitoring, not a signal.** Order measures failed every registered test in both eras; the average "
             "announcement-day move is about zero. Use this to see news early, not as a buy list.", "",
             f"## New order filings, last {a.lookback_days} days ({len(rec)}; {len(N)} fetched this run)", ""]
    if len(rec):
        rec = rec.assign(pct=rec["pct_of_rev"] * 100).sort_values(["ts"])
        # one order is often filed twice (intimation + press release): show it once, earliest filing kept
        dup = rec["amount_cr"].notna() & rec.duplicated(["symbol", "amount_cr"], keep="first")
        rec = rec[~dup].sort_values("pct", ascending=False, na_position="last")
        lines += ["| Filed (IST) | Stock | Amount (Rs cr) | % of TTM revenue | Industry (heat pct) | Trend rule | Held | Headline |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rec.itertuples():
            L = last.loc[r.symbol] if r.symbol in last.index else None
            ind = f"{L['ind']} ({L['heat_pct']:.2f})" if L is not None and pd.notna(L.get("heat_pct")) else (L["ind"] if L is not None else "n/a")
            tr = "—" if L is None else ("✅" if bool(L["trend"]) else "no")
            amt = "n/a" if pd.isna(r.amount_cr) else f"{r.amount_cr:,.1f}"
            pc = "n/a" if pd.isna(r.pct) else (f"**{r.pct:.0f}%**" if r.pct >= MATERIAL * 100 else f"{r.pct:.1f}%")
            lines.append(f"| {pd.Timestamp(r.ts):%d-%b %H:%M} | {r.symbol} | {amt} | {pc} | {ind} | {tr} | "
                         f"{', '.join(held.get(r.symbol, [])) or ''} | {str(r.headline)[:90]} |")
    lines += ["", "## Industry heat (core-band names; heat = mean own 60-session return)", "",
              "| Rank | Industry | Heat | Pct today | Pct 5 sessions ago | Names | Share above 200DMA |", "|---|---|---|---|---|---|---|"]
    for i, (ind, h) in enumerate(H.head(15).iterrows(), 1):
        lines.append(f"| {i} | {ind} | {h['heat']:+.1%} | {h['pct']:.2f} | {h['pct_5d_ago']:.2f} | {int(h['n'])} | {h['above200']:.0%} |")
    warm = H[(H["pct"] >= 0.7) & (H["pct_5d_ago"] < 0.7)]
    cool = H[(H["pct"] < 0.7) & (H["pct_5d_ago"] >= 0.7)]
    lines += ["", f"Entered hot/warming (pct >= 0.70) in the last 5 sessions: {', '.join(warm.index) or 'none'}",
              f"Dropped out: {', '.join(cool.index) or 'none'}", ""]
    # 2026-09-30: the same rows (deduplicated, with industry / heat / trend / held) as a small file the phone message
    # reads (notify.py), so the message shows the digest instead of recomputing it. Overwritten each run.
    js = []
    for r in (rec.itertuples() if len(rec) else []):
        L = last.loc[r.symbol] if r.symbol in last.index else None
        ind = None if L is None or pd.isna(L["ind"]) else str(L["ind"])
        hp = None if L is None or pd.isna(L.get("heat_pct")) else round(float(L["heat_pct"]), 2)
        js.append(dict(symbol=r.symbol, filed_ist=str(pd.Timestamp(r.ts)), fetched_ist=str(r.fetched),
                       amount_cr=None if pd.isna(r.amount_cr) else round(float(r.amount_cr), 1),
                       pct_of_rev=None if pd.isna(r.pct_of_rev) else round(float(r.pct_of_rev), 4),
                       industry=ind, heat_pct=hp, trend=None if L is None else bool(L["trend"]),
                       held=held.get(r.symbol, []), headline=str(r.headline)[:160],
                       url=r.url if isinstance(r.url, str) and r.url.startswith("http") else None))
    (ROOT / "logs/daily_orders").mkdir(exist_ok=True)
    (ROOT / "logs/daily_orders/latest.json").write_text(json.dumps(dict(
        data_through=str(d.date()), written=datetime.now().isoformat(timespec="seconds"),
        note="pct_of_rev is a fraction of trailing-12-month revenue (0.15 = 15%); times are IST", filings=js,
        warming=list(warm.index), cooling=list(cool.index)), indent=1))
    rep = ROOT / f"reports/daily_orders_industry_{d.strftime('%Y%m%d')}.md"
    rep.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\nwrote {rep} · {len(N)} new filings fetched · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
