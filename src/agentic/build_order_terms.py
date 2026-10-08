"""Order terms beyond the amount, one row per order filing (2026-10-08, for EXP-2026-10-08-big-order-annual; Abhinav:
"clean the order table: drop non-orders, repeats, and not-yet-awarded orders until they're awarded; use the company's own
share; flag operating contracts and concessions; divide each order's size by its execution period").

Input:  data/derived/order_amounts.parquet (categories order | tender_L1) + data/derived/order_fulltext_text.parquet (text)
Output: data/derived/order_terms.parquet (+ manifest). Columns:
  exec_months       execution period in months: the duration (digits, "(48)", or number words) or end date nearest after an
                    execution cue ("time period by which ... executed", "execution / completion / delivery period", "to be
                    completed", "tenure", "duration", "contract period"); end dates (dd-mm-yyyy, dd.mm.yyyy, dd-mon-yy, "Month
                    dd, yyyy", "Month yyyy", FY27, CY2027) become months from the filing date (the LAST date in the window,
                    so "Mar 16 to Jun 30, 2024" ends Jun 30); else "within / over / spread over N months" next to an
                    execution word. Not stated -> NaN (never assumed).
  firm              awarded | l1 (lowest bidder, not yet awarded) | loi (letter of intent) | mou | framework (rate
                    contract / empanelment, no committed quantity) | unclear (no firmness words; kept as awarded).
                    Strong award words (LoA, NoA, work / purchase / supply order, contract signed) win over L1 / LoI words.
  non_order         the filing is not an order win: an order cancellation or a loan / financing (always), or a stake sale,
                    investment, fund raise, acquisition, scheme, sales bookings, development value / revenue potential with no
                    order wording
  operating         amount covers years of operation: concession, toll, BOT / DBFOT / DBFOOT, annuity / HAM, mine developer
                    & operator (MDO), gross-cost-contract buses, O&M for N years
  jv, own_share     joint venture / consortium; own_share = the company's share (stated share amount / total, or stated
                    share %); 1.0 when no JV; NaN when JV and no share is stated (cannot size)
  gst               incl | excl | both | unknown (amount includes / excludes GST or taxes)
  aggregate         the amount sums several orders (orders worth X during the quarter / month, multiple orders, order inflow,
                    "new orders worth", "has received two orders"); such roundups state no single execution period
  repeat_of         seq_id of an earlier awarded filing of the same company with the same amount (within 0.1%) in the prior
                    180 days: a clarification, revised intimation, or contract signing after the award (a 2% / 400-day rule
                    was tried and paired different orders of similar size, e.g. LT 2,357 vs 2,388 cr five months apart)
  amount_cr         the order amount as parsed (order_amount_cr, else parsed_amount_cr), Rs crore
  own_amount_ex_gst_cr  amount_cr x own_share, / 1.18 when gst = incl (GST on works contracts is mostly 18%; approximation)
Usage: /usr/bin/python3 src/agentic/build_order_terms.py   (rebuilds the whole table in about a minute; daily layer step)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/derived/order_terms.parquet"
MON = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "twenty four": 24, "twenty-four": 24, "thirty": 30, "thirty six": 36,
         "thirty-six": 36, "forty two": 42, "forty-two": 42, "forty eight": 48, "forty-eight": 48, "sixty": 60, "ninety": 90}
WRE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, WORDS), key=len, reverse=True)) + r")\b(?=\s*(?:\(\s*\d+\s*\)\s*)?-?\s*(?:calendar\s+)?(?:months?|years?|days?|weeks?)\b)")
DUR = r"\(?\s*(\d{1,3}(?:\.\d)?)\s*\)?\s*-?\s*(?:\(\s*[a-z\- ]+\s*\)\s*)?(?:calendar\s+)?(months?|mths?|years?|yrs?|days?|weeks?)\b"
CUE = re.compile(r"to be executed|time period by which|time period, if any|associated with the order|execution period|period of execution|completion period|period of completion|"
                 r"to be completed|completion time|completion schedule|delivery period|delivery schedule|contract period|period of contract|"
                 r"tenure|duration|time frame|timeline")
WITHIN = re.compile(r"(?:within|over a period of|over the next|over|in a period of|spread over|for a period of)\s+" + DUR)
MN = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATES = [(re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d\d|\d\d)\b"), "dmy"), (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s./-]*" + MN + r"[\s,./'-]*(20\d\d|\d\d)\b"), "dMy"),
         (re.compile(r"\b" + MN + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)\b"), "Mdy"), (re.compile(r"\b" + MN + r"[\s,'-]*(20\d\d)\b"), "My"),
         (re.compile(r"\b(?:fy|financial year)\s*'?\s*(?:20)?(\d\d)(?:\s*[-/]\s*(?:20)?(\d\d))?\b"), "fy"), (re.compile(r"\bcy\s*'?\s*(20\d\d)\b"), "cy")]


def months_of(n: str, unit: str) -> float:
    v = float(n)
    return v * 12 if unit.startswith(("year", "yr")) else v / 30.4 if unit.startswith("day") else v / 4.35 if unit.startswith("week") else v


def _date(m: re.Match, kind: str) -> pd.Timestamp | None:
    g = m.groups()
    try:
        if kind == "dmy":
            y = int(g[2]); y = y + 2000 if y < 100 else y
            return pd.Timestamp(y, int(g[1]), int(g[0]))
        if kind == "dMy":
            y = int(g[2]); y = y + 2000 if y < 100 else y
            return pd.Timestamp(y, MON[g[1][:3]], int(g[0]))
        if kind == "Mdy":
            return pd.Timestamp(int(g[2]), MON[g[0][:3]], int(g[1]))
        if kind == "My":
            return pd.Timestamp(int(g[1]), MON[g[0][:3]], 28)
        if kind == "fy":
            return pd.Timestamp(2000 + int(g[1] or g[0]), 3, 31)
        if kind == "cy":
            return pd.Timestamp(int(g[0]), 12, 31)
    except (ValueError, KeyError):
        return None
    return None


def end_date_months(w: str, d: pd.Timestamp) -> float:
    fy_end = pd.Timestamp(d.year + (1 if d.month >= 4 else 0), 3, 31)
    rel = re.search(r"\b(next|current|this|same)\s+(?:financial year|fy)\b", w)
    if rel:
        return ((fy_end + pd.DateOffset(years=1) if rel.group(1) == "next" else fy_end) - d).days / 30.4
    ends = []
    for rx, kind in DATES:
        for m in rx.finditer(w):
            if re.search(r"(?:dated|date of|dt\.?|letter no|ref)\W{0,3}$", w[max(0, m.start() - 15): m.start()]):
                continue
            x = _date(m, kind)
            if x is not None:
                ends.append((m.start(), x))
        if ends:
            break
    if not ends:
        return np.nan
    mo = (max(ends)[1] - d).days / 30.4
    return mo if 0 < mo < 240 else np.nan


def exec_months(t: str, d: pd.Timestamp) -> tuple[float, str]:
    best = None
    for m in CUE.finditer(t):
        w = t[m.end(): m.end() + 220]
        a = re.search(DUR, w)
        e = end_date_months(w, d)
        if a and (np.isnan(e) or a.start() <= 60):
            return months_of(*a.groups()), t[max(0, m.start() - 20): m.end() + a.end() + 10]
        if not np.isnan(e):
            return e, t[m.start(): m.end() + 120]
        b = list(re.finditer(DUR, t[max(0, m.start() - 120): m.start()]))
        if b and best is None:
            best = (months_of(*b[-1].groups()), t[max(0, m.start() - 120): m.end() + 10])
    if best:
        return best
    w = WITHIN.search(t)
    if w and re.search(r"execut|complet|suppl|deliver|commission|implement", t[max(0, w.start() - 150): w.end() + 60]):
        return months_of(*w.groups()), t[max(0, w.start() - 60): w.end() + 10]
    return np.nan, ""


STRONG = r"letter of (?:acceptance|award)|\bloa\b|notification of award|\bnoa\b|work order|purchase order|supply order|service order|contract agreement|signed (?:a |the )?(?:contract|agreement)|agreement (?:has been |was )?(?:signed|executed)"
WEAK = r"has been awarded|have been awarded|is awarded|awarded (?:a|the|an) |received (?:an? |the )?(?:new )?(?:order|contract)|bagged|secured (?:an? |the )?(?:new )?(?:order|contract)|won (?:an? |the )?(?:new )?(?:order|contract)|order (?:has been )?(?:received|placed)"
NONORDER = r"dilution of|stake sale|sale of (?:its |our )?stake|divest|investment of (?:rs|inr|₹)|fund ?rais|preferential (?:issue|allot)|acquisition of|amalgamation|scheme of arrangement|sales bookings?|pre-?sales|gross development value|\bgdv\b|revenue potential|top ?line potential"
HARD_NONORDER = (r"cancell?ation of (?:the )?(?:aforesaid |said |above )?(?:work |purchase |supply )?(?:orders?|contracts?)|(?:orders?|contracts?) (?:has been |have been |stands )?cancell?ed|"
                 r"external commercial borrowing|\becb\b|sole lender|term loan|loan agreement")   # 2026-10-08 audit: VIKRAN cancellation, CREDITACC IFC loan, ACMESOLAR REC financing
ORDERWORD = r"orders? (?:for|worth|valued|of rs|of inr|amounting)|contracts? (?:for|worth|valued)|" + STRONG + "|" + WEAK
OPERATING = (r"concession|\bdbfot\b|\bdbfoot\b|\bdbfoo\b|\bbot\b|\bboot\b|build[- ]own|build[- ]operate|\btoll\b|hybrid annuity|\bham\b|annuity|mine developer|\bmdo\b|"
             r"gross cost contract|\bgcc\b|opex model|o\s?&\s?m (?:for|period)|operation (?:and|&) maintenance (?:for|period)|operate and maintain")
GENERIC = {"the", "shree", "shri", "sri", "indian", "india", "national", "hindustan", "bharat", "power", "steel", "global", "new", "united", "om"}


def own_share(t: str, toks: list[str], amount: float) -> tuple[float, str]:
    if re.search(r"(?:consortium|joint venture|\bjv\b)[^.]{0,80}(?:its|our|the company'?s) (?:wholly[- ]owned )?(?:step[- ]down )?subsidiar", t) and \
            not re.search(r"(?:consortium|joint venture|\bjv\b) (?:partner|with) (?:m/s\.? )?(?!its |our )[a-z]", t):
        return 1.0, "consortium_with_own_subsidiary"
    who = r"(?:the company'?s?|company'?s|our|" + "|".join(map(re.escape, toks)) + r")"
    a = re.search(who + r"\s*(?:ltd\.?|limited)?\s*(?:'s)?\s*share\s*(?:of|is|:|-|–|\(|in)?\s*(?:the\s*)?(?:order\s*)?(?:value\s*)?(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(crores?|cr\b\.?|lakhs?|lacs?|million|mn)?", t)
    if a and amount and amount > 0:
        v = float(a.group(1).replace(",", "")); u = a.group(2) or ""
        cr = v if u.startswith("cr") else v / 100 if u.startswith(("lakh", "lac")) else v / 10 if u.startswith(("million", "mn")) else v / 1e7 if v > 1e5 else np.nan
        if np.isfinite(cr) and 0.01 <= cr / amount <= 1.0:
            return cr / amount, "stated_amount"
    for p in (who + r"\s*(?:ltd\.?|limited)?\s*(?:'s)?\s*(?:share|stake|participation)\s*(?:ratio)?\s*(?:of|is|:|-|–)?\s*\(?\s*(\d{1,3}(?:\.\d+)?)\s*%",
              who + r"\s*(?:ltd\.?|limited)?[^%\d]{0,25}?\(\s*(\d{1,3}(?:\.\d+)?)\s*%\s*\)",
              r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:share|stake|participation)\s*(?:of|for|by)\s*" + who,
              r"held\s*(\d{1,3}(?:\.\d+)?)\s*%\s*by\s*" + who):
        m = re.search(p, t)
        if m and 1 <= float(m.group(1)) <= 99:
            return float(m.group(1)) / 100, "stated_pct"
    return np.nan, "jv_share_not_stated"


def terms(t: str, d: pd.Timestamp, cat: str, toks: list[str], amount: float) -> dict:
    t = re.sub(r"\s+", " ", t.lower())
    t = WRE.sub(lambda m: str(WORDS[m.group(1)]), t)
    em, snip = exec_months(t, d)
    strong, weak = bool(re.search(STRONG, t)), bool(re.search(WEAK, t))
    l1 = bool(re.search(r"\bl-?1\b|lowest bid|lowest quoted|least quoted|emerged as (?:the )?lowest", t)) or cat == "tender_L1"
    firm = ("awarded" if strong else "l1" if l1 else "loi" if re.search(r"letter of intent|\bloi\b", t) else "mou" if re.search(r"\bmou\b|memorandum of understanding", t)
            else "framework" if re.search(r"rate contract|framework agreement|empanel", t) else "awarded" if weak else "unclear")
    non_order = bool(re.search(HARD_NONORDER, t)) or (bool(re.search(NONORDER, t)) and not re.search(ORDERWORD, t))
    op = re.search(OPERATING, t)
    jv = bool(re.search(r"\bj\.?v\.?\b|joint venture|consortium", t))
    share, src = own_share(t, toks, amount) if jv else (1.0, "no_jv")
    incl = re.search(r"incl(?:\.|uding|usive|usive of|uding of)?\s*(?:of\s*)?(?:all\s*)?(?:applicable\s*)?(?:gst|taxes|tax|duties)", t)
    excl = re.search(r"excl(?:\.|uding|usive|usive of|uding of)?\s*(?:of\s*)?(?:all\s*)?(?:applicable\s*)?(?:gst|taxes|tax|duties|goods)|plus (?:applicable )?(?:gst|taxes)|\+\s*gst|gst extra|taxes extra", t)
    agg = bool(re.search(r"(?:orders?|contracts?)\s+(?:worth|aggregating|totall?ing|amounting|valued)[^.]{0,100}(?:during|in|for)\s+(?:the\s+)?(?:quarter|month|period|q[1-4]|fy)"
                         r"|multiple orders|various orders|several orders|series of orders|order inflow|orders? received (?:during|in) the"
                         r"|orders?\(s\) received|orders? received during (?:the month|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|accumulated orders|ytd order|new orders (?:worth|of)"
                         r"|has received (?:two|three|four|five|six|\d+)\s*(?:\(\d+\)\s*)?(?:new\s+)?orders", t))
    return dict(exec_months=em, exec_snip=snip[:240], firm=firm, non_order=non_order, operating=bool(op), operating_snip=t[max(0, op.start() - 60): op.end() + 60] if op else "",
                jv=jv, own_share=share, own_share_source=src, gst="incl" if incl and not excl else "excl" if excl and not incl else "both" if incl and excl else "unknown",
                aggregate=agg, sebi_table=bool(re.search(r"broad consideration or size|time period by which", t)))


def build() -> pd.DataFrame:
    O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet")
    O = O[O["cat_current"].isin(["order", "tender_L1"])].copy()
    O["amount_cr"] = O["order_amount_cr"].fillna(O["parsed_amount_cr"])
    T = pd.read_parquet(ROOT / "data/derived/order_fulltext_text.parquet")
    M = O.merge(T, on=["symbol", "seq_id"], how="left")
    M["text"] = (M["headline"].fillna("") + " " + M["text"].fillna("")).str.strip()
    M["ts"] = pd.to_datetime(M["ts"]); M["d"] = pd.to_datetime(M["d"])
    L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "company"])
    name = L.dropna().groupby("symbol")["company"].agg(lambda x: x.mode().iat[0])
    toks = {s: [s.lower()] + [w for w in re.findall(r"[a-z]+", str(name.get(s, "")).lower())[:1] if len(w) >= 4 and w not in GENERIC] for s in M["symbol"].unique()}
    X = pd.DataFrame([terms(t, d, c, toks[s], a) for t, d, c, s, a in zip(M["text"], M["d"], M["cat_current"], M["symbol"], M["amount_cr"])], index=M.index)
    R = pd.concat([M[["symbol", "seq_id", "ts", "d", "d_actionable", "cat_current", "amount_cr", "amount_confidence", "text_source"]], X], axis=1)
    R["has_text"] = M["text_source"].fillna("none") != "none"
    R["own_amount_ex_gst_cr"] = R["amount_cr"] * R["own_share"] / np.where(R["gst"] == "incl", 1.18, 1.0)
    # repeats: same company, same amount (0.1%), an earlier awarded order filing in the prior 180 days
    R = R.sort_values("ts"); R["repeat_of"] = None
    ok = (R["firm"].isin(["awarded", "unclear"])) & ~R["non_order"] & (R["amount_cr"] > 0)
    for s, g in R[ok].groupby("symbol"):
        seen: list = []
        for i, r in g.iterrows():
            hit = next((q for q, t0, a0 in seen if (r.ts - t0).days <= 180 and abs(r.amount_cr / a0 - 1) <= 0.001), None)
            if hit is not None:
                R.at[i, "repeat_of"] = hit
            else:
                seen.append((r.seq_id, r.ts, r.amount_cr))
    return R


def main() -> None:
    R = build()
    R.to_parquet(OUT, index=False)
    y = R["d"].dt.year
    cov = pd.DataFrame(dict(filings=R.groupby(y).size(), with_text=R.groupby(y)["has_text"].mean().round(3),
                            period_stated=R.groupby(y)["exec_months"].apply(lambda x: round(float(x.notna().mean()), 3)))).to_dict(orient="index")
    (OUT.with_name(OUT.name + ".manifest.json")).write_text(json.dumps(dict(
        dataset="order_terms", path=str(OUT.relative_to(ROOT)), rows=len(R), key=["symbol", "seq_id"], producer="src/agentic/build_order_terms.py",
        inputs=["data/derived/order_amounts.parquet", "data/derived/order_fulltext_text.parquet", "data/derived/event_ledger.parquet (company names)"],
        definitions=__doc__, units=dict(exec_months="months", amount_cr="Rs crore", own_amount_ex_gst_cr="Rs crore", own_share="fraction 0-1"),
        counts=dict(firm=R["firm"].value_counts().to_dict(), non_order=int(R["non_order"].sum()), operating=int(R["operating"].sum()),
                    jv=int(R["jv"].sum()), jv_share_not_stated=int((R["own_share_source"] == "jv_share_not_stated").sum()),
                    repeats=int(R["repeat_of"].notna().sum()), gst=R["gst"].value_counts().to_dict()),
        coverage_by_year=cov, updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(f"order_terms: {len(R)} filings · period stated {R['exec_months'].notna().mean():.0%} · firm {R['firm'].value_counts().to_dict()} · "
          f"non-orders {int(R['non_order'].sum())} · operating {int(R['operating'].sum())} · repeats {int(R['repeat_of'].notna().sum())} · "
          f"JV {int(R['jv'].sum())} (share not stated {int((R['own_share_source'] == 'jv_share_not_stated').sum())})")


if __name__ == "__main__":
    sys.exit(main())
