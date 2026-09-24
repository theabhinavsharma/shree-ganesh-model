"""EVENT x MATERIALITY x MARKET'S VOTE — what happens after each kind of filing (EXP-2026-09-24-event-materiality).

First principles: an event matters in proportion to what it does to THIS company's earnings.
  materiality  = order value / point-in-time TTM revenue (and / PIT market cap, shares = PAT/EPS)
  market's vote = day-0 reaction (entry-session close vs prior close) and its volume multiple
Events: announcements_historical (2016-2026) by regex taxonomy. Entry: first session OPEN strictly
after the filing date. Outcomes from the entry open: abnormal return (minus same-date universe
median) at 5/20/60/126 sessions; P(peak >= +50%) within 20 and 60 sessions; P(peak >= +100%)
within 126. Base rates from ALL universe stock-days. Every cell is split by era.
Universe: ISIN-master equities (no fund units), close > 25, ADV >= 1cr.
Output: stdout -> logs/leader_sleeve/event_study_20260924.log;
        logs/leader_sleeve/event_rows_20260924.parquet (+manifest) — one row per event.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
from generate_hybrid_basket import non_equity  # noqa: E402

H = (5, 20, 60, 126)
CATS = {   # first match wins, most specific first
    "open_offer_target": r"open offer|detailed public statement",
    "tender_L1": r"\bL-?1\b|lowest bidder",
    "order": r"bagging|receiv\w* of (?:an )?(?:new )?orders?|letter of (?:award|intent)|\bLoA\b|work order|purchase order|order (?:win|worth|valued)|secures? (?:an? )?(?:new )?(?:order|contract)|awarded",
    "client_partner": r"strategic partnership|partnership with|strategic alliance|\bMoU\b|memorandum of understanding|collaborat|tie-?up|new client|signs? (?:an? )?(?:agreement|contract)|definitive agreement",
    "acquisition": r"acquisition of|acquires? (?:a |an |the )?(?:stake|majority|\d)|completion of acquisition",
    "approval_launch": r"USFDA|US FDA|\bEIR\b|\bANDA\b|approval from|launch(?:es|ed)? |commercial production|commissioning|commissioned",
    "capex": r"capacity expansion|expansion of capacity|greenfield|brownfield|new plant|new facility",
    "fundraise_pref": r"preferential|warrants",
    "fundraise_qip": r"\bQIP\b|qualified institutions placement",
    "buyback": r"buy-?back",
    "rating_up": r"rating.{0,40}upgrade|upgrade.{0,40}rating",
    "bonus_split": r"\bbonus\b|sub-?division|stock split",
}
AMT = re.compile(r"(?:(Rs\.?|INR|₹)|(USD|US\$|\$))\s?([\d,]+(?:\.\d+)?)\s?(crores?|cr\b|lakhs?|lacs?|millions?|mn\b|billions?|bn\b)", re.I)


def parse_amount_cr(text: str, usdinr: float) -> float | None:
    """Largest amount in the text, in Rs crore. None if no amount."""
    best = None
    for inr, usd, num, unit in AMT.findall(text or ""):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        u = unit.lower()
        mult = {"c": 1.0, "l": 0.01, "m": 0.1, "b": 100.0}[u[0]]     # to crore (INR); million=0.1cr
        cr = v * mult * (usdinr if usd else 1.0)
        best = cr if best is None or cr > best else best
    return best


def main() -> None:
    sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
    fund = set(sm.loc[sm["is_fund_unit"], "symbol"])

    # ---- panel + forward windows from each session's OPEN ----
    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                         columns=["symbol", "trade_date", "open", "high", "low", "close", "prev_close",
                                  "volume_vs_20d", "avg_traded_value_20d"])
    px = px[~px["symbol"].isin(fund) & ~non_equity(px["symbol"])]
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = px.groupby("symbol")
    for k in H:
        px[f"r{k}"] = g["close"].shift(-(k - 1)) / px["open"] - 1                 # k sessions incl. entry
    for k in (20, 60, 126):
        px[f"pk{k}"] = g["high"].transform(lambda s: s[::-1].rolling(k, min_periods=1).max()[::-1]) / px["open"] - 1
    # from the entry-session CLOSE (for conditioning on that session's reaction — no same-day leak)
    for k in (20, 60):
        px[f"c{k}"] = g["close"].shift(-k) / px["close"] - 1
    px["pkc60"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(60, min_periods=1).max()[::-1]) / px["close"] - 1
    px["react0"] = px["close"] / px["prev_close"] - 1
    px["adv"] = px["avg_traded_value_20d"] / 1e7
    if "--mcap50" in sys.argv:          # PIT market cap >= Rs 50cr, no liquidity floor
        mc = pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet"); mc["trade_date"] = pd.to_datetime(mc["trade_date"])
        px = px.merge(mc[["symbol", "trade_date", "mcap_cr"]], on=["symbol", "trade_date"], how="left")
        uni = px[px["mcap_cr"] >= 50].drop(columns=["mcap_cr"]).copy()
    else:
        uni = px[(px["close"] > 25) & (px["adv"] >= 1)].copy()
    med = uni.groupby("trade_date")[[f"r{k}" for k in H] + ["c20", "c60"]].median().add_prefix("m_")
    uni = uni.join(med, on="trade_date")
    for k in H:
        uni[f"ab{k}"] = uni[f"r{k}"] - uni[f"m_r{k}"]
    for k in (20, 60):
        uni[f"abc{k}"] = uni[f"c{k}"] - uni[f"m_c{k}"]
    uni["era"] = np.where(uni["trade_date"].dt.year >= 2023, "conf", "disc")

    # ---- events ----
    ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "desc", "attchmntText", "sort_date"])
    ah["txt"] = ah["desc"].fillna("") + " || " + ah["attchmntText"].fillna("")
    ah["ts"] = pd.to_datetime(ah["sort_date"], errors="coerce")
    ah = ah.dropna(subset=["ts"])
    ah["cat"] = None
    for c, p in CATS.items():
        m = ah["cat"].isna() & ah["txt"].str.contains(p, case=False, regex=True)
        ah.loc[m, "cat"] = c
    ev = ah.dropna(subset=["cat"]).copy()
    ev["d"] = ev["ts"].dt.normalize()
    ev = ev.drop_duplicates(["symbol", "d", "cat"])                             # one per symbol/day/category
    mac = pd.read_parquet(ROOT / "data/derived/macro_panel.parquet", columns=["trade_date", "usdinr"])
    mac["trade_date"] = pd.to_datetime(mac["trade_date"])
    ev = pd.merge_asof(ev.sort_values("d"), mac.rename(columns={"trade_date": "d"}).sort_values("d"), on="d", direction="backward")
    ev["amount_cr"] = [parse_amount_cr(t, u if pd.notna(u) else 83.0) for t, u in zip(ev["txt"], ev["usdinr"])]
    # entry = first session strictly after the filing date
    ev["entry_key"] = ev["d"] + pd.Timedelta(days=1)
    U = uni.rename(columns={"trade_date": "entry"})[["symbol", "entry", "open", "react0", "volume_vs_20d", "adv", "close", "era"]
                  + [f"r{k}" for k in H] + [f"ab{k}" for k in H] + ["pk20", "pk60", "pk126", "abc20", "abc60", "pkc60"]]
    ev = pd.merge_asof(ev.sort_values("entry_key"), U.sort_values("entry"), left_on="entry_key", right_on="entry",
                       by="symbol", direction="forward", tolerance=pd.Timedelta(days=7)).dropna(subset=["entry"])

    # ---- PIT TTM revenue + market cap ----
    q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["filing_dt"])
    q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end"], keep="last").sort_values(["symbol", "quarter_end"])
    # units per pnl_quarterly manifest: detail_api = Rs lakh, xbrl = Rs  ->  Rs crore
    to_cr = np.where(q["source"] == "xbrl", 1e-7, 1e-2)
    q["sales_cr"] = q["net_sales"] * to_cr; q["pat_cr"] = q["pat"] * to_cr
    q["rev_ttm_cr"] = q.groupby("symbol")["sales_cr"].transform(lambda s: s.rolling(4).sum())
    q["shares"] = np.where((q["eps_basic"].abs() > 0.01) & q["pat_cr"].notna(), q["pat_cr"] * 1e7 / q["eps_basic"], np.nan)
    q["known"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
    qq = q.dropna(subset=["rev_ttm_cr"])[["symbol", "known", "rev_ttm_cr", "shares"]].sort_values("known")
    ev = pd.merge_asof(ev.sort_values("d"), qq.rename(columns={"known": "d"}), on="d", by="symbol",
                       direction="backward", tolerance=pd.Timedelta(days=200))
    ev["mcap_cr"] = ev["close"] * ev["shares"] / 1e7
    ev["order_to_rev"] = ev["amount_cr"] / ev["rev_ttm_cr"]
    ev["amt_to_mcap"] = ev["amount_cr"] / ev["mcap_cr"]

    # ---- reporting ----
    base = {era: dict(p50_20=(u["pk20"] >= .5).mean() * 100, p50_60=(u["pk60"] >= .5).mean() * 100,
                      p50c_60=(u["pkc60"] >= .5).mean() * 100,
                      p2x=(u["pk126"] >= 1).mean() * 100, n=len(u)) for era, u in uni.groupby("era")}
    print(f"panel through {uni['entry'].max().date() if 'entry' in uni else uni['trade_date'].max().date()} · universe stock-days {len(uni):,} · events {len(ev):,}")
    print("BASE RATES (all universe stock-days):", {e: {k: round(v, 2) for k, v in b.items()} for e, b in base.items()})

    def line(label, S):
        out = []
        for era in ("disc", "conf"):
            E = S[S["era"] == era]
            if len(E) < 30:
                out.append(f"{era} n={len(E):>5} {'—':>46}"); continue
            b = base[era]
            out.append(f"{era} n={len(E):>5} ab5 {E['ab5'].mean()*100:>+5.1f} ab20 {E['ab20'].mean()*100:>+5.1f} "
                       f"ab60 {E['ab60'].mean()*100:>+5.1f} ab126 {E['ab126'].mean()*100:>+6.1f} "
                       f"P50/60d {(E['pk60']>=.5).mean()*100:>4.1f}({(E['pk60']>=.5).mean()*100/b['p50_60']:>3.1f}x) "
                       f"P2x {(E['pk126']>=1).mean()*100:>4.1f}({(E['pk126']>=1).mean()*100/b['p2x']:>3.1f}x)")
        print(f"{label:<34}| " + " | ".join(out), flush=True)

    print("\n=== A. EVENT TYPE (abnormal %, entry = next open) ===")
    for c in CATS:
        line(c, ev[ev["cat"] == c])

    print("\n=== B. ORDER MATERIALITY: order value / PIT TTM revenue (orders + L1 with a parsed amount) ===")
    O = ev[ev["cat"].isin(["order", "tender_L1"]) & ev["order_to_rev"].notna()]
    print(f"orders with amount & revenue: {len(O):,}")
    for lo, hi, lab in [(0, .05, "<5% of revenue"), (.05, .15, "5-15%"), (.15, .40, "15-40%"), (.40, 1.0, "40-100%"), (1.0, 1e9, ">100% of revenue")]:
        line(f"order {lab}", O[(O["order_to_rev"] >= lo) & (O["order_to_rev"] < hi)])
    print("  ...by order / market cap:")
    for lo, hi, lab in [(0, .05, "<5% of mcap"), (.05, .20, "5-20%"), (.20, 1e9, ">20% of mcap")]:
        line(f"order {lab}", O[(O["amt_to_mcap"] >= lo) & (O["amt_to_mcap"] < hi)])

    def vline(label, S):   # returns measured from the entry-session CLOSE (after the vote is known)
        out = []
        for era in ("disc", "conf"):
            E = S[S["era"] == era]
            if len(E) < 30:
                out.append(f"{era} n={len(E):>5} {'—':>30}"); continue
            p = (E["pkc60"] >= .5).mean() * 100
            out.append(f"{era} n={len(E):>5} post-vote ab20 {E['abc20'].mean()*100:>+5.1f} ab60 {E['abc60'].mean()*100:>+5.1f} "
                       f"P50/60d {p:>4.1f}({p/base[era]['p50c_60']:>3.1f}x)")
        print(f"{label:<34}| " + " | ".join(out), flush=True)

    print("\n=== C. THE MARKET'S VOTE: entry-session reaction, returns measured AFTER it (from that close) ===")
    for lo, hi, lab in [(-9, 0, "react < 0"), (0, .05, "react 0-5%"), (.05, .10, "react 5-10%"), (.10, 9, "react >= 10%")]:
        vline(lab, ev[(ev["react0"] >= lo) & (ev["react0"] < hi)])
    vline("react>=5% & vol>=3x", ev[(ev["react0"] >= .05) & (ev["volume_vs_20d"] >= 3)])
    print("  ...material orders (>=15% of revenue) by the vote:")
    MO = O[O["order_to_rev"] >= .15]
    vline("material order & react<5%", MO[MO["react0"] < .05])
    vline("material order & react>=5%", MO[MO["react0"] >= .05])

    print("\n=== D. SIZE: every event by market cap ===")
    for lo, hi, lab in [(0, 500, "mcap <500cr"), (500, 2000, "500-2,000cr"), (2000, 10000, "2k-10k cr"), (10000, 1e9, ">10k cr")]:
        line(lab, ev[(ev["mcap_cr"] >= lo) & (ev["mcap_cr"] < hi)])
    print("  ...material orders by size:")
    for lo, hi, lab in [(0, 2000, "material order & mcap<2k"), (2000, 1e9, "material order & mcap>=2k")]:
        line(lab, MO[(MO["mcap_cr"] >= lo) & (MO["mcap_cr"] < hi)])

    keep = ["symbol", "d", "cat", "entry", "era", "amount_cr", "rev_ttm_cr", "mcap_cr", "order_to_rev", "amt_to_mcap",
            "react0", "volume_vs_20d", "adv"] + [f"ab{k}" for k in H] + ["pk20", "pk60", "pk126", "abc20", "abc60", "pkc60", "txt"]
    out = ROOT / ("logs/leader_sleeve/event_rows_mcap50_20260924.parquet" if "--mcap50" in sys.argv
                  else "logs/leader_sleeve/event_rows_20260924.parquet")
    ev[keep].to_parquet(out, index=False)
    out.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="event rows", experiment="EXP-2026-09-24-event-materiality", rows=len(ev), producer="src/agentic/event_materiality_study.py",
        columns=dict(d="filing date", cat="regex category (first match)", entry="first session after filing date (entry at its OPEN)",
                     amount_cr="largest Rs/USD amount in headline text, Rs crore (USD at that day's USDINR)",
                     rev_ttm_cr="PIT trailing-4-quarter net sales, Rs crore", mcap_cr="close x PIT shares (PAT/EPS), Rs crore",
                     order_to_rev="amount_cr / rev_ttm_cr (fraction)", amt_to_mcap="amount_cr / mcap_cr (fraction)",
                     react0="entry-session close / prior close - 1 (fraction)", volume_vs_20d="entry-session volume / 20d avg",
                     abK="return from entry open over K sessions minus same-date universe median (fraction)",
                     pkK="max high within K sessions / entry open - 1 (fraction)", txt="headline + attachment text"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print("\nEVENT STUDY COMPLETE")


if __name__ == "__main__":
    main()
