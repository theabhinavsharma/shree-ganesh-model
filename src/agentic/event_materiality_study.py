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

2026-09-27 audit fixes (logs/audits/audit_20260927_research_code.json). Only CATS, AMT and parse_amount_cr
changed, plus the FX input (historical series, no 83.0 fallback) at the amount call; the study logic is untouched.
Results logged before this date (event_study_20260924.log, event_rows_*_20260924.parquet) used the old rules.
  FIXED  event_materiality_study.py:33 'order' matched bare 'awarded' (honours, arbitration awards, credit
         ratings pulled in). Now 'awarded' needs order/contract/work/project/deal/... within 60 chars (40 in the
         brief; 60 keeps 'HCL Awarded Five Year ... Contract'), and the whole headline is rejected on
         regulatory/tax/court orders, arbitration, credit ratings/loans, honours/trophies/certifications and
         terminations, MCA/ROC/AGM and Competition Commission orders. On announcements_historical this
         drops 430 order/L1 filings (disc 313, conf 117); a spot check of 30 drops found ~4 real orders
         phrased 'X awarded <client> <project>' with no order noun within 60 chars.
  FIXED  :33 recall — added NSE subject 'Awarding of order(s)/contract(s)', plural 'orders worth/valued/
         aggregating', 'Receipt of order/contract/LoA', bags/wins/receives/secured ... order|contract,
         'bags/wins Rs X', '<project/work/order> awarded', 'award of new projects', 'work -order', 'got new
         orders'. Adds 3,201 filings (disc 1,685, conf 1,516); a spot check of 30 additions found 2 non-orders.
         Order+L1 population 6,340 -> 9,111. Classifying all 1.29M announcements now takes ~70 s (was ~22 s).
         Known residual: buyer-side filings ('Board approved award of EPC contract to M/s ...') still match.
  FIXED  :32-33 `\bLoA\b` no longer fires on NCD 'Letter of Allotment' (NCD/debenture guard); tender_L1 needs
         bid/tender context or 'declared/emerged/announced ... L1' (drops 'unit at L-1, Chikalthana').
  FIXED  :44 AMT — '&#8377;', 'Rs.  500' (several spaces), 'Crs', 'Rupees', 'lakh crore', full-rupee Indian
         grouping ('Rs. 15,38,34,500.00'), unit from words ('Rs. 15.56 (Fifteen Crores ...)'), word boundary
         before Rs ('FVRS10LAC' is not an amount), S$/SG$/A$ are not USD, 'dollars 5 million' is USD.
  FIXED  :47-59 parse_amount_cr returned the LARGEST amount; now the order-attached amount (order cue within
         150 chars before, no boilerplate cue within 80 chars) — see parse_order_amounts.py.
  FIXED  :110-113 USD at a fixed 83.0 when macro_panel had no rate (every date before 2024-02-19). The rate
         now comes from parse_order_amounts.load_usdinr (macro_panel.usdinr, else data/derived/
         usdinr_history.parquet = FRED DEXINUS, as-of with 7-day tolerance); with no rate the USD amount is
         None unless an INR equivalent is written beside it. The FX read is the only change inside main().
  NOT FIXED :127 rev_ttm_cr = rolling(4) without a 4-consecutive-quarter check, and PIT revenue starts
         2018-04 so 2016-17 orders get no ratio. This is study logic, outside this change; TODO.
  NOT FIXED  one amount per filing (no summing of several listed orders) — documented in parse_order_amounts.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
from generate_hybrid_basket import non_equity  # noqa: E402

H = (5, 20, 60, 126)
# ---- order / tender_L1 taxonomy (rewritten 2026-09-27 after the audit; other categories unchanged) ----
# Exclusions apply to the whole headline (desc || attchmntText): regulatory / tax / court orders, arbitration
# awards, credit ratings and loans, honours and trophies, terminations — none of them is a business order.
_ORDER_EXCL = (
    r"arbitra|tribunal|hon'?ble|(?:high|supreme|appellate|commercial|district|sessions) court\b|\bcourt (?:of|order|has|passed|directed)|"
    r"\bNCLT\b|\bNCLAT\b|income[- ]?tax|\bGST (?:authorit|department|demand|order)|state tax|central tax|\bCGST\b|\bSGST\b|"
    r"commissioner of (?:state |central )?(?:tax|GST|customs|income)|tax (?:demand|authorit)|show[- ]cause|penalty|"
    r"assessment order|demand order|adjudicat|order (?:passed|issued) by|orders? passed|passed an order|"
    r"action\(?s?\)? (?:taken|initiated)|clause 20\b|interim order|stay order|award against|"
    r"award in (?:its|our|the company'?s) favou?r|"
    r"credit rating|rating agenc|\bCRISIL\b|\bICRA\b|CARE Ratings|India Ratings|Brickwork|Acuit[eé]|term loan|"
    r"credit facilit|sanction of (?:loan|term|credit)|senior secured|secured (?:notes|bonds|debentures|loans?)|bond issue|"
    r"financial results|reports? Q[1-4]\b|regional director|ministry of corporate affairs|registrar of companies|"
    r"registered office|annual general meeting|competition commission|\bCCI\b|"
    r"trophy|excellence award|\bawards? (?:ceremony|function|night)|great place to work|recogni[sz](?:ed|ing|tion)|"
    r"certificate of (?:recognition|merit|appreciation|excellence)|\bhonou?r(?:ed)?\b|best (?:employer|supplier|vendor|company)|"
    r"awarded (?:with |as |the |a |an )?(?:[\w'\"&-]+ ){0,6}(?<!letter of )awards?\b|certification|export house|nomination|"
    r"\b(?:IGBC|LEED|GRIHA)\b|\bMoU (?:rating|performance)|"
    r"terminat|commencement of (?:the )?(?:contract|work|project|operation)|completion certificate")
_ORDER_NOUN = r"(?:orders?|contracts?|works?|projects?|packages?|tenders?|deals?|engagements?|mandates?|EPC|LoA|LoI|letters? of (?:award|intent))"
_ORDER_CCY = r"(?:\bRs\.?|\bINR|₹|\bUSD|US\$)\s*[\d,.]+"
_ORDER_POS = (
    r"bagging|awarding (?:of )?(?:orders?|contracts?)|"                                  # NSE subject 'Awarding of order(s)/contract(s)'
    r"receiv\w* (?:of )?(?:an? |the )?(?:new |fresh |repeat |large |major |prestigious )?(?:orders?|contracts?|work\s*-?\s*orders?|purchase orders?)\b|"
    r"receipt of (?:an? |the )?(?:new |fresh |repeat |large |major )?(?:orders?|contracts?|work\s*-?\s*orders?|purchase orders?|letters? of (?:award|intent)|LoA|LoI)\b|"
    r"letter of (?:award|intent)|\bLoI\b|(?<!NCD )\bLoA\b|"
    r"work\s*-?\s*orders?|purchase orders?|orders? (?:wins?|worth|valued|aggregating|amounting)|"
    r"secures? (?:an? )?(?:new )?(?:orders?|contracts?)|"
    r"\b(?:bags?|bagged|wins?|won|secured|receives?|received|got|obtained)\b[^.|]{0,60}?\b(?<!in )(?:orders?|contracts?)\b|"
    r"\b(?:bags?|bagged|wins?|won|secures?|secured|awarded)\b[^.|]{0,30}?" + _ORDER_CCY + r"|"
    r"awarded\b[^|]{0,60}?\b" + _ORDER_NOUN + r"\b|"                                     # no bare 'awarded'
    r"\baward of (?:an? )?(?:new )?" + _ORDER_NOUN + r"\b|"
    r"\b" + _ORDER_NOUN + r"\s*(?:has been |have been |was |were |is |being |now )?awarded\b")
CATS = {   # first match wins, most specific first
    "open_offer_target": r"open offer|detailed public statement",
    # L1 only with bid / tender context ('declared L1', 'L1 bidder', 'lowest bidder'); not plant addresses like 'L-1, Chikalthana'
    "tender_L1": (r"lowest (?:evaluated )?(?:bidder|bid)\b|\bL-?1 status\b|"
                  r"\b(?:declared|emerged|announced|stood|become|became|being|been|as)[\s,:]+(?:as\s+)?(?:the\s+)?(?:lowest\s+)?[^\w\s]{0,3}L-?1\b|"
                  r"\bL-?1\b[^.|]{0,80}?\b(?:bid|bidder|bidders|tender|tenders|bidding)\b|"
                  r"\b(?:bid|bidder|bidders|tender|tenders|bidding)\b[^.|]{0,80}?\bL-?1\b"),
    # positive pattern first (fast reject), then whole-headline exclusions, then NCD 'Letter of Allotment' guard for LoA
    "order": (r"^(?=[\s\S]*?(?:" + _ORDER_POS + r"))(?![\s\S]*?(?:" + _ORDER_EXCL + r"))"
              r"(?![\s\S]*?(?:\bNCDs?\b|debenture|letter of allotment)[\s\S]*?\bLoA\b)"),
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
# Amount parsing lives in parse_order_amounts (shared with the order crawler). AMT is the currency-amount regex
# (Rs/Rs./INR/Rupees/U+20B9/&#8377;, USD/US$/$ but not S$/A$/SG$, crore/Cr/Crs/lakh/lac/million/mn/billion/bn).
from parse_order_amounts import AMT, FX_TOL, extract_order_amount, load_usdinr  # noqa: E402,F401


def parse_amount_cr(text: str, usdinr: float | None) -> float | None:
    """Order-attached amount in the headline text, Rs crore (was: LARGEST amount anywhere — audit 2026-09-27).
    A candidate counts when an order cue precedes it within 150 chars and no boilerplate cue (revenue, order
    book, group size, ...) sits within 80 chars. USD needs a real historical rate: usdinr NaN/None -> None
    unless an INR equivalent is written next to it. None if no order-attached amount."""
    r = extract_order_amount(text, None, usdinr)
    v = r["order_amount_cr"]
    return float(v) if v is not None and v == v else None


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
    # historical USDINR from data/derived (macro_panel, else usdinr_history = FRED DEXINUS), as-of, 7-day tolerance
    mac = load_usdinr()[["fx_date", "usdinr"]].rename(columns={"fx_date": "d"})
    ev = pd.merge_asof(ev.sort_values("d"), mac.sort_values("d"), on="d", direction="backward", tolerance=FX_TOL)
    ev["amount_cr"] = [parse_amount_cr(t, u) for t, u in zip(ev["txt"], ev["usdinr"])]   # no fixed-rate fallback (audit 2026-09-27)
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
                     amount_cr="order-attached Rs/USD amount in headline text (parse_order_amounts rule), Rs crore; USD at the historical USDINR as-of the filing date (macro_panel, else usdinr_history = FRED DEXINUS), NaN when no rate",
                     rev_ttm_cr="PIT trailing-4-quarter net sales, Rs crore", mcap_cr="close x PIT shares (PAT/EPS), Rs crore",
                     order_to_rev="amount_cr / rev_ttm_cr (fraction)", amt_to_mcap="amount_cr / mcap_cr (fraction)",
                     react0="entry-session close / prior close - 1 (fraction)", volume_vs_20d="entry-session volume / 20d avg",
                     abK="return from entry open over K sessions minus same-date universe median (fraction)",
                     pkK="max high within K sessions / entry open - 1 (fraction)", txt="headline + attachment text"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print("\nEVENT STUDY COMPLETE")


if __name__ == "__main__":
    main()
