"""Event ledger from NSE's own filings, 2016+ (2026-10-02; news plan step 1). Official, time-stamped, covers small caps.

One row per filing that is a corporate EVENT (routine filings such as trading-window notices are dropped):
  symbol, isin, company, filed_at (NSE dissemination time, else filing time), session (first trading session at whose
  OPEN the news is known: filed before 09:15 IST -> that day, else the next session), bucket, direction (+1 good,
  -1 bad, 0 neutral/needs numbers), severity (1 normal, 2 severe), amount_cr (order amounts already parsed from the
  PDFs, data/derived/order_amounts.parquet; other buckets blank in v1), amt_to_mcap (amount / market cap the day before),
  credibility = "T1 exchange filing", category (NSE's own), rule (what matched), source_url (the NSE PDF).
Buckets match on NSE's category OR the filing's own text, because NSE only added specific categories (e.g. "Bagging/
Receiving of orders/contracts") around 2023; before that the same events were filed as "Updates" / "Press Release".
Text matching is case-insensitive at word boundaries with the company's own name removed (lesson: keyword substrings).
QC printed and saved to logs/news/event_ledger_qc.md: bucket counts by year (era jumps), a results-calendar completeness
check (official NSE count), and 5 random examples per bucket to eyeball.
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
OUT = ROOT / "data/derived/event_ledger.parquet"
QC = ROOT / "logs/news/event_ledger_qc.md"

# (bucket, direction, severity, NSE-category regex, text regex) — first match wins, so specific/bad rules come first
RULES = [
    ("default_insolvency", -1, 2, r"Insolvency|CIRP|Defaults on Payment", r"\b(insolvency|NCLT admit\w*|CIRP|resolution professional|default(ed)? (in|on) (payment|repayment))\b"),
    ("suspension", -1, 2, r"Suspension of Trading", None),
    ("auditor_exit", -1, 2, r"Change in Auditors", r"\bauditor\w*\b.{0,60}\bresign"),
    ("regulatory_action", -1, 1, r"Action\(s\) (taken|initiated)|Pendency of Litigation", r"\b(show cause notice|penalty (of|imposed)|search (and|&) seizure|SEBI (order|interim order)|enforcement directorate)\b"),
    ("order_cancel", -1, 1, None, r"\b(cancel\w*|terminat\w*|short[- ]?clos\w*|foreclos\w*)\b.{0,80}\b(order|contract|LoA|letter of award|project)\b"),
    ("strike_disruption", -1, 1, r"Strikes/Lockouts", r"\b(lock[- ]?out|strike at|fire (at|in) (the|our) (plant|factory|unit))\b"),
    ("delayed_results", -1, 1, r"Delayed/Non-submission", None),
    ("mgmt_exit", -1, 1, r"Resignation|Cessation|Change in Management", r"\b(managing director|MD and CEO|chief executive|CEO|chief financial|CFO|whole[- ]time director)\b"),
    ("independent_director_exit", -1, 1, r"Resignation of Independent", None),
    ("order_win", 1, 1, r"Bagging/Receiving of orders|Awarding of order", r"\b(bag(s|ged)?|receiv(ed|es)|secur(ed|es)|won|award(ed)?)\b.{0,60}\b(order|contract|work order|letter of (award|acceptance|intent)|LoA|L1)\b"),
    ("takeover_target", 1, 1, r"^Open Offer|Public Announcement-Open Offer|Delisting", r"\b(open offer|delisting offer)\b"),
    ("buyback", 1, 1, r"^Buyback|Public Announcement - Buyback", None),
    ("bonus_split", 1, 1, r"^Bonus|Stock split", r"\b(bonus issue|sub-?division of (equity )?shares|stock split)\b"),
    ("acquisition", 1, 1, r"^Acquisition", r"\bacqui(re[sd]?|sition of)\b"),
    ("capacity_start", 1, 1, r"Capacity addition|Commencement of commercial production", r"\b(commercial production|commissioning of|capacity expansion)\b"),
    ("approval_launch", 1, 1, r"Product launch", r"\b(USFDA|US FDA|EIR|establishment inspection report|approval (from|of) (the )?(DCGI|USFDA|CDSCO))\b"),
    ("agreement", 1, 1, r"^Agreements|Memorandum of Understanding", r"\b(memorandum of understanding|MoU|joint venture|strategic partnership)\b"),
    ("rating_change", 0, 1, r"Credit Rating", None),
    ("fund_raise", 0, 1, r"Qualified Institutional Placement|Preferential issue|Rights Issue|Raising of Funds|Issue of Securities", r"\b(qualified institutions? placement|QIP|preferential (issue|allotment)|rights issue)\b"),
    ("merger_scheme", 0, 1, r"Amalgamation/Merger|Scheme of Arrangement", r"\b(amalgamation|demerger|scheme of arrangement)\b"),
    ("clarification", 0, 1, r"Clarification|News Verification|Rumour Verification", None),
    ("price_movement_query", 0, 1, r"^Price movement", None),
    ("results", 0, 1, r"Financial Result|Integrated Filing- Financial|Publish Audited Results", None),
]
UP, DOWN = r"\bupgrad\w*|revised upward", r"\bdowngrad\w*|revised downward|negative outlook|credit watch with negative"


def main() -> None:
    A = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                        columns=["symbol", "sm_isin", "sm_name", "desc", "attchmntText", "attchmntFile", "an_dt", "exchdisstime", "seq_id"])
    A["category"] = A["desc"].fillna("").str.strip()
    name = A["sm_name"].fillna("").str.replace(r"\b(limited|ltd\.?)\b", "", case=False, regex=True).str.strip()
    text = [t.replace(n, " ") if n else t for t, n in zip(A["attchmntText"].fillna("").tolist(), name.tolist())]
    A["text"] = text
    A["bucket"], A["direction"], A["severity"], A["rule"] = None, 0, 1, None
    open_ = A["bucket"].isna()
    for b, dirn, sev, cat_re, txt_re in RULES:
        m = pd.Series(False, index=A.index)
        if cat_re:
            m |= A["category"].str.contains(cat_re, case=False, regex=True)
        if txt_re:
            m |= A["text"].str.contains(txt_re, case=False, regex=True)
        if b in ("mgmt_exit",):            # category alone (any resignation) is too broad: need a top-role mention too
            m = A["category"].str.contains(cat_re, case=False, regex=True) & A["text"].str.contains(txt_re, case=False, regex=True)
        if b == "auditor_exit":
            m = (A["category"].str.contains(cat_re, case=False, regex=True) & A["text"].str.contains(r"resign", case=False)) | A["text"].str.contains(txt_re, case=False, regex=True)
        hit = m & open_
        A.loc[hit, ["bucket", "direction", "severity", "rule"]] = [b, dirn, sev, f"cat:{cat_re}" if cat_re else f"text:{b}"]
        open_ &= ~hit
    A = A[A["bucket"].notna()].copy()
    r = A["bucket"] == "rating_change"                                   # direction for ratings only when the text says so
    A.loc[r & A["text"].str.contains(UP, case=False, regex=True), ["bucket", "direction"]] = ["rating_up", 1]
    A.loc[r & A["text"].str.contains(DOWN, case=False, regex=True), ["bucket", "direction"]] = ["rating_down", -1]
    t = pd.to_datetime(A["exchdisstime"].where(A["exchdisstime"].fillna("-") != "-"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
    A["filed_at"] = t.fillna(pd.to_datetime(A["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce"))
    A = A.dropna(subset=["filed_at"])
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["trade_date"])["trade_date"].unique())))
    day = A["filed_at"].dt.normalize()
    before_open = A["filed_at"].dt.hour * 60 + A["filed_at"].dt.minute < 9 * 60 + 15
    idx = np.where(before_open, cal.searchsorted(day, side="left"), cal.searchsorted(day, side="right"))
    A["session"] = [cal[i] if i < len(cal) else pd.NaT for i in idx]
    # materiality: order amounts already read from the PDFs (incl. OCR), by NSE sequence id
    O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet", columns=["seq_id", "order_amount_cr", "parsed_amount_cr"])
    O["amount_cr"] = O["order_amount_cr"].fillna(O["parsed_amount_cr"])
    A = A.merge(O[["seq_id", "amount_cr"]].dropna().drop_duplicates("seq_id").astype({"seq_id": str}), on="seq_id", how="left")
    M = pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet", columns=["symbol", "trade_date", "mcap_cr"]).sort_values("trade_date")
    A = A.sort_values("session")
    A = pd.merge_asof(A.dropna(subset=["session"]), M.rename(columns={"trade_date": "session"}), on="session", by="symbol",
                      direction="backward", allow_exact_matches=False)                         # market cap the session before
    A["amt_to_mcap"] = A["amount_cr"] / A["mcap_cr"]
    L = pd.DataFrame(dict(symbol=A["symbol"], isin=A["sm_isin"], company=A["sm_name"], filed_at=A["filed_at"], session=A["session"],
                          bucket=A["bucket"], direction=A["direction"].astype(int), severity=A["severity"].astype(int),
                          amount_cr=A["amount_cr"], amt_to_mcap=A["amt_to_mcap"], credibility="T1 exchange filing",
                          category=A["category"], rule=A["rule"], text=A["attchmntText"].str[:300], source_url=A["attchmntFile"], seq_id=A["seq_id"]))
    L = L.sort_values(["session", "symbol"]).reset_index(drop=True)
    L.to_parquet(OUT, index=False)
    by = L.groupby([L["session"].dt.year, "bucket"]).size().unstack(fill_value=0).T
    # completeness vs NSE's results calendar (official): share of quarterly results filings that appear as a filing
    C = pd.read_parquet(ROOT / "data/derived/results_calendar_full.parquet", columns=["symbol", "broadCastDate", "period"])
    C["d"] = pd.to_datetime(C["broadCastDate"], format="%d-%b-%Y %H:%M:%S", errors="coerce").dt.normalize(); C = C.dropna(subset=["d"])
    R = A[A["bucket"] == "results"][["symbol", "filed_at"]].assign(d=lambda x: x["filed_at"].dt.normalize())
    key = set(zip(R["symbol"], R["d"])) | set(zip(R["symbol"], R["d"] - pd.Timedelta(days=1))) | set(zip(R["symbol"], R["d"] + pd.Timedelta(days=1)))
    C["found"] = [(s, d) in key for s, d in zip(C["symbol"], C["d"])]
    comp = C.groupby(C["d"].dt.year)["found"].mean().round(3)
    lines = [f"# Event ledger QC · {datetime.now():%Y-%m-%d %H:%M}", "", f"rows {len(L):,} · symbols {L['symbol'].nunique():,} · sessions {L['session'].min().date()}..{L['session'].max().date()}",
             "", "## Events by bucket and year", "", "```\n" + by.to_string() + "\n```", "", "## Completeness: NSE results-calendar filings found in the archive (±1 day)", "",
             "```\n" + comp.loc[2016:].to_frame("share").to_string() + "\n```", "", "## Order amounts", "",
             f"order_win rows with an amount: {L.loc[L.bucket == 'order_win', 'amount_cr'].notna().mean():.1%}", "", "## Examples (5 per bucket, random)", ""]
    for b, g in L.groupby("bucket"):
        lines += [f"**{b}**"] + [f"- {r.session.date()} {r.symbol}: {str(r.text)[:160]}" for r in g.sample(min(5, len(g)), random_state=1).itertuples()] + [""]
    QC.parent.mkdir(parents=True, exist_ok=True); QC.write_text("\n".join(lines))
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="event_ledger.parquet", producer="src/agentic/build_event_ledger.py", rows=len(L), definitions=__doc__,
        source="data/derived/announcements_historical.parquet (NSE corporate announcements, fetch_announcements_historical.py) + order_amounts.parquet + mcap_pit.parquet",
        units=dict(amount_cr="Rs crore", amt_to_mcap="fraction of market cap the session before", direction="+1 good / -1 bad / 0 neutral"),
        buckets=[r[0] for r in RULES] + ["rating_up", "rating_down"], qc="logs/news/event_ledger_qc.md",
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"ledger {len(L):,} events · {L['symbol'].nunique():,} symbols · QC -> {QC.relative_to(ROOT)}")
    print(by.to_string())
    print("\nresults-calendar completeness by year:", comp.loc[2016:].to_dict())


if __name__ == "__main__":
    main()
