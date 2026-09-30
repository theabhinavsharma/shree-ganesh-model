"""Order value ATTACHED TO THE ORDER for every crawled order / L1-tender filing (EXP-2026-09-27-hot-x-material-order).

Why (audit 2026-09-27, logs/audits/audit_20260927_research_code.json, finding on
event_materiality_study.py:47-59 + fetch_order_fulltext.py:76,91): the old parser returned the LARGEST
amount anywhere in the text. In full attachment text that is usually boilerplate ('Siemens Limited had
Revenue of INR 106,728 million', 'flagship of the USD 2.5 billion Kalyani Group', 'order book ... over
Rs. 4,100 crores'), and `fulltext.fillna(headline)` let that figure overwrite a correct headline amount.
USD was converted at a hard-coded 83.0 whenever macro_panel had no USDINR (every date before 2024-02-19).

Rule used here (deterministic, no model):
  1. Candidates = every currency amount in the text (AMT): Rs / Rs. / INR / Rupees / Rupee / U+20B9 /
     '&#8377;' (word boundary before Rs, so 'FVRS10LAC' is not an amount), USD / US$ / US Dollars / bare '$'
     (S$, SG$, A$, AU$, C$, HK$, NZ$ are OTHER currency, never USD), units crore / Cr / Cr. / Crs / lakh / lac /
     lakh crore / thousand crore / million / mn / mio / billion / bn. A unit-less figure counts only when it is
     a full-rupee (or full-dollar) amount >= 1,00,000, or when the words in the following bracket give the unit
     ('Rs. 15.56 (Fifteen Crores Fifty Six Lakhs)').  'X-Y crore' and 'Rs X crore to Rs Y crore' are ranges:
     the LOWER bound is used and flagged.
  2. A candidate is ORDER-ATTACHED when an order cue (order(s), contract(s), LoA, LoI, letter of award /
     intent / acceptance, work order, purchase order, worth, valued at, value of, amounting to, aggregating,
     bid value, contract price, bags/bagged/wins/won/secures/receives/awarded) ends within 150 characters
     before it, or an order cue follows it within 40 characters in the same sentence ('Rs 5,600 Cr order').
  3. A candidate is BOILERPLATE when a boilerplate cue sits within 80 characters before it or 40 after it,
     inside the same sentence: turnover, revenue, order book, order backlog, order inflow / intake, orders in
     hand, unexecuted, market cap, net worth, consolidated, since inception, cumulative, total orders, till
     date, so far, YTD, profit, EBITDA, PAT, net sales, total income, paid-up / share capital, AUM, market value,
     investment / capex / term loan / performance bond / bank guarantee / preferential / 'impact of', and the
     group-size pattern ('USD 2.5 billion Kalyani Group', 'a $4 billion conglomerate', "group's turnover").
     DEVIATION from the brief: a bare 'group' before the amount is NOT boilerplate, because 'order from the
     Tata Group worth Rs 500 crore' is a real order amount; only the group-SIZE patterns above are.
  4. Selection, headline first: (a) headline (NSE desc || attchmntText) order-attached, non-boilerplate;
     (b) full text order-attached, non-boilerplate. (An uncued-headline fallback was tried and dropped: on
     the 2016-18 checkpoint 10 of its 13 picks were investments, fraud impact, bonds or payments, not orders.) Within a stage the first candidate with a STRONG value cue
     (worth / valued / value of / amounting / aggregating / contract price / order value / bid value, within
     40 characters) wins, else the first order-attached one; an INR figure within 80 characters of the chosen
     USD figure is its INR equivalent and is preferred.
  5. USD: converted only with a HISTORICAL USDINR series from data/derived, as-of the filing date (7-day
     tolerance): macro_panel.usdinr where present (FX feed from 2024-02-19, FRED DEXINUS fill from 2018 —
     its usdinr_source column is carried into fx_source), else data/derived/usdinr_history.parquet (FRED
     DEXINUS, Federal Reserve H.10 noon buying rate, 1973+; built by fetch_usdinr_history.py on 2026-09-27).
     If neither file has a rate for the date, the order amount is NaN with usd_unconverted=True and the USD
     figure stays in amount_usd_mn. No fixed rate is ever used. Other currencies (EUR, GBP, AED, S$, ...) are
     never converted.

Output: data/derived/order_amounts.parquet (+ .manifest.json), one row per (symbol, seq_id) in the
order_fulltext_v2.jsonl checkpoint (latest record per key; the crawler may still be running, so the file
covers whatever exists). `python parse_order_amounts.py --validate N` prints N random rows with the
chosen snippet for hand-checking.

Limitations / TODO:
  - One order per filing: filings that list several orders get the first strong-cued amount, not the sum,
    unless a later figure equals the sum of the parts ('Rs 305 cr and Rs 278 cr aggregating to Rs 583 cr').
    Such rows carry amount_flags 'multiple_amounts' and at most medium confidence.
  - Rates are not order values: an amount next to per day / per month / per annum / daily / remittance is
    boilerplate, so fee-collection and O&M contracts stated only as a rate get NaN.
  - Headline text is re-read from announcements_historical.parquet (the checkpoint does not store it).
  - OCR text is noisy ('Rs.3,577.93' can come out as 'Rs.3.577.93'); such figures are parsed as written.
  - Category is re-derived with the CURRENT event_materiality_study.CATS (cat_current); rows whose filing
    no longer classifies as order / tender_L1 (honours, arbitration, ratings ...) get order_amount_cr NaN and
    keep the figure in parsed_amount_cr.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
CKPT = ROOT / "data/derived/order_fulltext_v2.jsonl"
OUT = ROOT / "data/derived/order_amounts.parquet"
ANN = ROOT / "data/derived/announcements_historical.parquet"
MACRO = ROOT / "data/derived/macro_panel.parquet"
USDINR_HIST = ROOT / "data/derived/usdinr_history.parquet"
FX_TOL = pd.Timedelta(days=7)

# ----------------------------------------------------------------------------------------------- regexes
_NUM = r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_NUM2 = r"(?P<num2>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_INR = r"(?P<inr>(?<![A-Za-z0-9])(?:Rs|INR|Rupees|Rupee)(?![A-Za-z])\.?|₹|&#8377;|&#x20[bB]9;)"
_USD = r"(?P<usd>(?<![A-Za-z0-9$])(?:US\s?\$|USD|US\s?Dollars?|Dollars?)(?![A-Za-z])|(?<![A-Za-z0-9$])\$)"
_OTH = r"(?P<oth>(?<![A-Za-z0-9])(?:EUR|Euros?|GBP|AED|SAR|JPY|SGD|AUD|CAD|(?:SG|S|AU|A|C|CA|HK|NZ)\$)(?![A-Za-z])|€|£)"
_UNIT = (r"(?P<unit>lakh\s*crores?|lac\s*crores?|thousand\s+crores?|crores?|crs?\b\.?|lakhs?|lacs?\b|"
         r"millions?|mn\b|mio\b|billions?|bn\b)")
AMT = re.compile(
    rf"(?:{_INR}|{_USD}|{_OTH})[\s.:,/-]*(?:in\s+)?{_NUM}"
    rf"(?:\s*(?:-|–|to)\s*(?:(?:Rs|INR)\.?\s*|₹\s*|US\$\s*|USD\s*)?{_NUM2}(?=\s*[A-Za-z]))?"
    rf"(?:\s*{_UNIT})?", re.I)
# a unit word without a currency ('worth 500 crore'): accepted only right after a strong value cue
BARE = re.compile(rf"(?<![\w.,$₹;]){_NUM}\s*(?P<unit>lakh\s*crores?|crores?|crs?\b\.?|lakhs?)", re.I)
_UNIT_WORDS = re.compile(r"^\s*[(\[]\s*(?:Rupees|Rs\.?|INR)?\s*([A-Za-z ,\-]{3,120})[)\]]", re.I)

_VERB = r"bag(?:s|ged|ging)?|wins?|won|secure[sd]?|secures|receiv\w*|receipt|awarded|bagged"
CUE = re.compile(
    r"(?<!\sin )(?<!^in )\b(?:orders?|contracts?|LoA|LoI|letters? of (?:award|intent|acceptance)|work orders?|"
    r"purchase orders?|worth|valued|value of|valuing|amounting|amounts? to|aggregat\w*|bid value|contract price|"
    r"order value|contract value|project cost|cost of|bid price|bid amount|quoted|consideration|tender|bid|bidder|L-?1|"
    r"value|" + _VERB + r")\b", re.I)
STRONG = re.compile(
    r"\b(?:worth|valued|value of|valuing|amounting|amounts? to|aggregat\w*|bid value|contract price|order value|"
    r"contract value|order of|orders of|contract of|wins? of|for a value|at a value|total value|value is|value:|"
    r"value -|value \(|value at|price of|cost of|bid price|bid amount|quoted)", re.I)
BOILER = re.compile(
    r"turnover|revenues?|order[- ]?book|backlog|order inflows?|order intake|orders? in hand|unexecuted|"
    r"market cap|market capitali[sz]ation|net[- ]?worth|consolidated|since inception|cumulative|total orders?|"
    r"till date|to date|so far|year[- ]to[- ]date|\bYTD\b|\bprofits?\b|EBITDA|\bPAT\b|net sales|total income|"
    r"paid[- ]up|share capital|authori[sz]ed capital|assets under|\bAUM\b|group(?:'s|’s)? (?:turnover|revenue|size)|"
    r"conglomerate|enterprise value|market value|invest(?:ment|ed|ing)|capital expenditure|capex|term loan|"
    r"performance (?:bond|guarantee)|bank guarantee|preferential|impact of|fraud|"
    # rates, not order values ('daily remittance Rs 10,80,000', 'Rs 87 lacs per day')
    r"remittance|per (?:day|month|annum|year|km|unit|MT|tonne|ton|kg|litre|hour|sq)|\bdaily\b|\bmonthly\b|\bp\.a\.", re.I)
# amount immediately followed by a (capitalised) name + Group/company = size of the group, not the order
ENTITY_AFTER = re.compile(r"^\s*(?:[A-Z][\w&.\-]*\s+){0,3}(?i:group|conglomerate|multinational|business house|enterprise)\b")
_SENT = re.compile(
    r"(?<![A-Z][a-z])(?<!\b[A-Z])(?<!Ltd)(?<!Pvt)(?<!Inc)(?<!Mrs)(?<!viz)(?<!Nos)(?<!Sr)(?<!Jr)(?<!Govt)(?<!Corp)"
    r"(?<!approx)(?<!Dept)\.\s+(?=[A-Z(\"'“])|[;•·]|\|\||\s\|\s")

_MULT = {"crore": 1.0, "lakhcrore": 1e5, "thousandcrore": 1e3, "lakh": 0.01, "million": 0.1, "billion": 100.0}


def _unit_key(u: str | None) -> str | None:
    if not u:
        return None
    u = re.sub(r"[\s.]", "", u.lower())
    if u.startswith(("lakhcr", "laccr")):
        return "lakhcrore"
    if u.startswith("thousand"):
        return "thousandcrore"
    if u.startswith("cr"):
        return "crore"
    if u.startswith(("lakh", "lac")):
        return "lakh"
    if u.startswith(("million", "mn", "mio")):
        return "million"
    if u.startswith(("billion", "bn")):
        return "billion"
    return None


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _clip_before(win: str) -> str:
    last = None
    for m in _SENT.finditer(win):
        last = m
    return win[last.end():] if last else win


def _clip_after(win: str) -> str:
    m = _SENT.search(win)
    return win[:m.start()] if m else win


def find_amounts(text: str) -> list[dict]:
    """Every currency amount in `text` with its context flags. value_native is in crore of its own currency."""
    text = text or ""
    out = []
    spans = []
    for m in AMT.finditer(text):
        v = _num(m.group("num"))
        if v is None:
            continue
        cur = "INR" if m.group("inr") else ("USD" if m.group("usd") else "OTHER")
        uk = _unit_key(m.group("unit"))
        flags = []
        if uk is None and v < 1e5:                          # 'Rs. 15.56 (Fifteen Crores ...)'; never for full-rupee figures
            w = _UNIT_WORDS.match(text[m.end():m.end() + 140])
            if w and not re.search(r"\d", w.group(1)):
                words = w.group(1).lower()
                uk = "crore" if ("crore" in words or "cr " in words) else ("lakh" if re.search(r"lakh|lac", words) else None)
                if uk:
                    flags.append("unit_from_words")
        if uk is None:
            if v >= 1e5:                                   # full rupees / full dollars
                native = v / 1e7
                flags.append("unit_inferred_full_amount")
            else:
                continue                                   # 'Rs 10 each', 'Rs 2 per share', years ...
        else:
            native = v * _MULT[uk]
        v2 = _num(m.group("num2")) if m.group("num2") else None
        if v2 is not None and v2 > v:
            flags.append("range_low")
        out.append(dict(start=m.start(), end=m.end(), cur=cur, native=native, raw=m.group(0), flags=flags, bare=False))
        spans.append((m.start(), m.end()))
    for m in BARE.finditer(text):
        if any(a <= m.start() < b or a < m.end() <= b for a, b in spans):
            continue
        if not STRONG.search(text[max(0, m.start() - 40):m.start()]):
            continue
        v = _num(m.group("num")); uk = _unit_key(m.group("unit"))
        if v is None or uk is None:
            continue
        out.append(dict(start=m.start(), end=m.end(), cur="INR", native=v * _MULT[uk], raw=m.group(0),
                        flags=["no_currency_symbol"], bare=True))
    out.sort(key=lambda c: c["start"])
    # 'Rs 1,000 crore to Rs 2,500 crore' -> range, keep the lower bound on the first, drop the second
    for a, b in zip(out, out[1:]):
        gap = text[a["end"]:b["start"]]
        if a["cur"] == b["cur"] and re.fullmatch(r"\s*(?:-|–|to|and)\s*", gap, re.I) and b["native"] > a["native"] \
                and re.search(r"between|range|in the range|band", text[max(0, a["start"] - 40):a["start"]], re.I):
            a["flags"] = a["flags"] + ["range_low"]; b["flags"] = b["flags"] + ["range_high"]
    out = [c for c in out if "range_high" not in c["flags"]]
    for c in out:
        s, e = c["start"], c["end"]
        before150 = text[max(0, s - 150):s]
        after40 = _clip_after(text[e:e + 40])
        c["cued"] = bool(CUE.search(before150) or CUE.search(after40))
        c["strong"] = bool(STRONG.search(text[max(0, s - 40):s]))
        b80 = _clip_before(text[max(0, s - 80):s])
        c["boiler"] = bool(BOILER.search(b80) or BOILER.search(after40) or ENTITY_AFTER.match(text[e:e + 60]))
    return out


def _pick(cands: list[dict], need_cue: bool) -> dict | None:
    ok = [c for c in cands if not c["boiler"] and (c["cued"] or not need_cue)]
    if not ok:
        return None
    strong = [c for c in ok if c["strong"]]
    best = (strong or ok)[0]
    # 'orders worth Rs 305 crore and Rs 278 crore aggregating to Rs 583 crore' -> the aggregate of the parts
    i = ok.index(best)
    for k in range(i + 2, len(ok)):
        if ok[k]["start"] - best["start"] > 250:
            break
        parts = [c["native"] for c in ok[i:k] if c["cur"] == ok[k]["cur"]]
        if len(parts) >= 2 and abs(sum(parts) - ok[k]["native"]) <= 0.02 * ok[k]["native"]:
            best = dict(ok[k], flags=ok[k]["flags"] + ["aggregate_of_parts"])
            break
    distinct = {round(c["native"], 2) for c in ok if c["cur"] == best["cur"] and c is not best
                and abs(c["native"] - best["native"]) > 0.02 * best["native"]}
    if distinct and "aggregate_of_parts" not in best["flags"]:
        best = dict(best, flags=best["flags"] + ["multiple_amounts"])
    if best["cur"] != "INR":                               # INR equivalent written next to a USD figure
        near = [c for c in cands if c["cur"] == "INR" and not c["boiler"] and abs(c["start"] - best["start"]) <= 80]
        if near:
            best = dict(near[0], flags=near[0]["flags"] + ["inr_equiv_of_" + best["cur"].lower()])
    return best


def _to_cr(c: dict, usdinr: float) -> float:
    if c["cur"] == "INR":
        return c["native"]
    if c["cur"] == "USD" and usdinr is not None and np.isfinite(usdinr):
        return c["native"] * usdinr
    return float("nan")


def extract_order_amount(headline: str | None, fulltext: str | None, usdinr: float | None) -> dict:
    """Order-attached amount (Rs crore) from the headline first, then the full text. See module docstring."""
    usdinr = float(usdinr) if usdinr is not None and pd.notna(usdinr) else float("nan")
    H = find_amounts(headline or "")
    F = find_amounts(fulltext or "")
    res = dict(order_amount_cr=np.nan, amount_method=None, amount_confidence=None, currency=None, amount_usd_mn=np.nan,
               usd_unconverted=False, amount_snippet=None, amount_flags="", n_amounts_headline=len(H), n_amounts_fulltext=len(F),
               n_attached_fulltext=sum(1 for c in F if c["cued"] and not c["boiler"]), headline_fulltext_agree=None)
    stages = [("headline_cue", H, True, headline), ("fulltext_cue", F, True, fulltext)]
    for name, cands, need_cue, txt in stages:
        c = _pick(cands, need_cue)
        if c is None:
            continue
        cr = _to_cr(c, usdinr)
        res.update(currency=c["cur"], amount_flags=";".join(c["flags"]),
                   amount_snippet=(txt or "")[max(0, c["start"] - 100):c["end"] + 60])
        if c["cur"] == "USD":
            res["amount_usd_mn"] = round(c["native"] * 10.0, 6)   # crore of USD -> USD million
        if c["cur"] == "USD" and not np.isfinite(cr):
            # look for an INR equivalent of the same USD figure in the full text before giving up
            eq = None
            for u in (x for x in F if x["cur"] == "USD" and abs(x["native"] - c["native"]) <= 0.01 * c["native"]):
                near = [x for x in F if x["cur"] == "INR" and not x["boiler"] and abs(x["start"] - u["start"]) <= 80]
                if near:
                    eq = near[0]; break
            if eq is not None:
                res.update(order_amount_cr=eq["native"], amount_method=name + "+fulltext_inr_equiv", amount_confidence="low",
                           currency="INR", amount_flags=";".join(eq["flags"] + ["inr_equiv_of_usd"]),
                           amount_snippet=(fulltext or "")[max(0, eq["start"] - 100):eq["end"] + 60])
            else:
                res.update(amount_method=name + "+usd_unconverted", usd_unconverted=True)
            break
        if c["cur"] == "OTHER":
            res.update(amount_method=name + "+foreign_ccy_unconverted")
            break
        conf = "high" if (c["strong"] and not c["bare"]) else "medium"
        if any(f in c["flags"] for f in ("range_low", "unit_inferred_full_amount", "unit_from_words", "multiple_amounts")) \
                or c["bare"] or any(f.startswith("inr_equiv_of") for f in c["flags"]):
            conf = "low" if conf != "high" else "medium"
        if name == "fulltext_cue":
            vals = {round(_to_cr(x, usdinr), 2) for x in F if x["cued"] and not x["boiler"] and np.isfinite(_to_cr(x, usdinr))}
            if len(vals) > 1 and conf == "high":
                conf = "medium"                           # several distinct order-attached figures in the text
        res.update(order_amount_cr=cr, amount_method=name + ("+range_low" if "range_low" in c["flags"] else ""),
                   amount_confidence=conf)
        break
    if res["amount_method"] is None:
        if not (headline or "").strip() and not (fulltext or "").strip():
            res["amount_method"] = "none_no_text"
        elif not H and not F:
            res["amount_method"] = "none_no_amount_in_text"
        elif all(c["boiler"] for c in H + F):
            res["amount_method"] = "none_boilerplate_only"
        else:
            res["amount_method"] = "none_amounts_not_order_attached"
    # agreement between an order-attached headline figure and an order-attached full-text figure
    h = _pick(H, True); f = _pick(F, True)
    if h is not None and f is not None:
        hv, fv = _to_cr(h, usdinr), _to_cr(f, usdinr)
        if np.isfinite(hv) and np.isfinite(fv) and hv > 0:
            res["headline_fulltext_agree"] = bool(abs(hv - fv) <= 0.02 * hv)
            if res["headline_fulltext_agree"] and res["amount_confidence"] == "medium":
                res["amount_confidence"] = "high"
    return res


def parse_amount_cr(text: str, usdinr: float | None) -> float | None:
    """Order-attached amount in ONE text (a headline), Rs crore; None when there is none or it is USD without a rate."""
    r = extract_order_amount(text, None, usdinr)
    v = r["order_amount_cr"]
    return float(v) if v is not None and np.isfinite(v) else None


def largest_amount_cr(text: str, usdinr: float | None) -> float | None:
    """The retired 'largest amount anywhere' rule (for comparison only), with the same FX policy (no fixed rate)."""
    usdinr = float(usdinr) if usdinr is not None and pd.notna(usdinr) else float("nan")
    vals = [_to_cr(c, usdinr) for c in find_amounts(text or "")]
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else None


# ----------------------------------------------------------------------------------------------- data
def load_usdinr(source: str = "derived") -> pd.DataFrame:
    """Daily historical USDINR from data/derived with its source (fx_date, usdinr, fx_source).
    macro_panel.usdinr first (fx_source 'macro_panel:<usdinr_source>'), then usdinr_history.parquet (FRED
    DEXINUS) for dates before macro_panel's first rate. Never a fixed rate; empty series -> every USD is NaN."""
    if source != "derived":
        raise ValueError(f"unknown FX source {source!r}; only data/derived series are used")
    import pyarrow.parquet as pq
    cols = [c for c in ("trade_date", "usdinr", "usdinr_source") if c in pq.read_schema(MACRO).names]
    mac = pd.read_parquet(MACRO, columns=cols).dropna(subset=["usdinr"])
    src = ("macro_panel:" + mac["usdinr_source"].fillna("unknown").astype(str)) if "usdinr_source" in mac else "macro_panel"
    out = pd.DataFrame({"fx_date": pd.to_datetime(mac["trade_date"]), "usdinr": mac["usdinr"].astype(float), "fx_source": src})
    if USDINR_HIST.exists():
        h = pd.read_parquet(USDINR_HIST, columns=["trade_date", "usdinr"]).dropna()
        h = pd.DataFrame({"fx_date": pd.to_datetime(h["trade_date"]), "usdinr": h["usdinr"].astype(float),
                          "fx_source": "usdinr_history:fred_dexinus"})
        first = out["fx_date"].min() if len(out) else pd.Timestamp.max
        out = pd.concat([h[h["fx_date"] < first], out], ignore_index=True)
    return out.sort_values("fx_date").reset_index(drop=True)


def attach_fx(df: pd.DataFrame, fx: pd.DataFrame, date_col: str = "d") -> pd.DataFrame:
    """As-of (backward, 7-day tolerance) USDINR for each row's filing date; NaN when the series has no rate."""
    x = df.copy()
    x["_d"] = pd.to_datetime(x[date_col], errors="coerce")
    has = x["_d"].notna()
    a = x[has].reset_index().sort_values("_d")
    a = pd.merge_asof(a, fx.rename(columns={"fx_date": "_d"}), on="_d", direction="backward", tolerance=FX_TOL)
    a = a.set_index("index")
    x["usdinr"] = a["usdinr"]; x["fx_source"] = a["fx_source"]
    return x.drop(columns=["_d"])


def load_checkpoint(path: Path = CKPT) -> pd.DataFrame:
    """Latest record per (symbol, seq_id) from the crawl checkpoint (OCR records are appended later and win)."""
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:                    # partial last line while the crawler is writing
                continue
    D = pd.DataFrame(rows)
    D["seq_id"] = D["seq_id"].astype(str)
    return D.drop_duplicates(["symbol", "seq_id"], keep="last").reset_index(drop=True)


def headlines_for(D: pd.DataFrame) -> pd.DataFrame:
    """NSE headline (desc || attchmntText) and filing timestamp for the checkpoint keys."""
    import pyarrow.dataset as ds
    keys = sorted(set(D["seq_id"].astype(str)))
    t = ds.dataset(ANN).to_table(columns=["symbol", "seq_id", "desc", "attchmntText", "sort_date"],
                                 filter=ds.field("seq_id").isin(keys)).to_pandas()
    t["seq_id"] = t["seq_id"].astype(str)
    t = t.drop_duplicates(["symbol", "seq_id"], keep="last")
    t["headline"] = t["desc"].fillna("") + " || " + t["attchmntText"].fillna("")
    return t[["symbol", "seq_id", "headline", "sort_date"]]


def classify(headline: pd.Series) -> pd.Series:
    """Current event-study category (first match wins) — same taxonomy the crawl used, as fixed on 2026-09-27."""
    sys.path.insert(0, str(ROOT / "src/agentic"))
    from event_materiality_study import CATS
    out = pd.Series(None, index=headline.index, dtype=object)
    for c, p in CATS.items():
        m = out.isna() & headline.str.contains(p, case=False, regex=True)
        out[m] = c
    return out


def build(D: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, seq_id): order-attached amount + provenance. D = latest checkpoint records."""
    H = headlines_for(D)
    X = D.merge(H, on=["symbol", "seq_id"], how="left")
    X = attach_fx(X, fx, "d")
    recs = [extract_order_amount(h, t, u) for h, t, u in zip(X["headline"].fillna(""), X.get("text", pd.Series(index=X.index)).fillna(""), X["usdinr"])]
    R = pd.DataFrame(recs, index=X.index)
    X = pd.concat([X, R], axis=1)
    X["legacy_largest_cr"] = [largest_amount_cr((h or "") + " " + (t or ""), u)
                              for h, t, u in zip(X["headline"].fillna(""), X.get("text", pd.Series(index=X.index)).fillna(""), X["usdinr"])]
    X["cat_current"] = classify(X["headline"].fillna(""))
    # an amount is an ORDER amount only while the filing is still an order / L1 filing under the current CATS
    X["parsed_amount_cr"] = X["order_amount_cr"]
    X["order_amount_cr"] = X["order_amount_cr"].where(X["cat_current"].isin(["order", "tender_L1"]))
    ts = pd.to_datetime(X["sort_date"], errors="coerce")
    X["ts"] = ts
    post = ts.notna() & ((ts.dt.hour * 60 + ts.dt.minute) >= 15 * 60 + 30)
    X["post_close"] = post.where(ts.notna())
    X["d_actionable"] = (ts.dt.normalize() + pd.to_timedelta(post.astype(int), unit="D")).dt.date.astype("string")
    X["text_source"] = np.where(X.get("ocr", pd.Series(False, index=X.index)).fillna(False).astype(bool)
                                & X["text"].fillna("").str.len().gt(0), "ocr",
                                np.where(X["text"].fillna("").str.len().gt(0), "pdf_text", "none"))
    X["era"] = np.where(pd.to_datetime(X["d"], errors="coerce").dt.year >= 2023, "conf", "disc")
    keep = ["symbol", "seq_id", "cat", "cat_current", "d", "ts", "post_close", "d_actionable", "era", "status", "text_source",
            "text_chars", "order_amount_cr", "parsed_amount_cr", "amount_method", "amount_confidence", "currency", "amount_usd_mn", "usdinr",
            "fx_source", "usd_unconverted", "amount_flags", "headline_fulltext_agree", "n_amounts_headline",
            "n_amounts_fulltext", "n_attached_fulltext", "legacy_largest_cr", "amount_snippet", "headline"]
    for k in keep:
        if k not in X:
            X[k] = np.nan
    return X[keep]


# Hand-check (2026-09-27) of the rule on order_fulltext_v2.jsonl, read against headline + attachment text.
# Set A = 30 random rows with at least one amount, seed 2027, held out from the 30 rows used to tune the rule,
# scored BEFORE the aggregate-of-parts / rate fixes it motivated. Set B = 30 fresh rows scored after them
# (15 crawled 2016-20 filings + 15 conf-era 2023-26 headlines; conf attachments were not crawled yet).
VALIDATION = dict(
    set_A=dict(rows=30, amount_given=25, amount_correct=22, precision=0.88, nan_given=5, nan_correct=5,
               legacy_largest_rule=dict(amount_given=29, amount_correct=22, precision=0.76),
               wrong=["HFCL 102780124: 305 picked, 583 = aggregate of 305+278 (fixed afterwards: aggregate_of_parts)",
                      "MEP 101080834: 0.108 = daily remittance per fee plaza, a rate (rate cues added; list form still slips)",
                      "SETUINFRA 101034407: 108 = first of two listed orders (108 + 50); flagged multiple_amounts"],
               note="GVPIL 101261537 scored correct: 270 cr is the contract; the listed company's scope is 180 cr"),
    set_B=dict(rows=30, amount_given=15, amount_correct=13, precision=0.87, nan_given=15, nan_correct=15,
               wrong=["KEC 103713505: 655 (T&D only) instead of 1,115 total new orders (fixed afterwards: 'wins of' strong cue); "
                      "the filing is a results release and is no longer an order under the new CATS",
                      "ITI 106112342: 35 instead of 64 total ('Rs. 29.14' has no unit, so the parts do not add up)"],
               note="GIPCL 105393132 scored correct for the amount (259 cr contract) but the company is the BUYER; "
                    "buyer-side 'award of contract to M/s ...' filings are a known category false positive",
               fx_note="scored before data/derived/usdinr_history.parquet existed: MOTHERSON 100902880 (USD 2.5 bn) and "
                       "A2ZINFRA 101578178 (USD 7.07 mn) were correct NaN + usd_unconverted then; they now convert at FRED DEXINUS"))


MANIFEST_COLUMNS = dict(
    symbol="NSE symbol", seq_id="NSE announcement seq_id (string)",
    cat="category assigned when the filing was crawled (event_materiality_study.CATS at crawl time)",
    cat_current="category under the CURRENT CATS (2026-09-27 fix); rows not in order|tender_L1 are no longer order filings",
    d="filing date (IST) as stored by the crawler", ts="filing timestamp (IST) from announcements_historical.sort_date",
    post_close="True when filed at or after 15:30 IST", d_actionable="first date whose CLOSE can use the filing: d, or d+1 calendar day when post_close",
    era="disc (<2023) / conf (2023+) by d", status="latest crawl status (OK | NO_AMOUNT | NO_ATTACHMENT | NEEDS_OCR | OCR_OK | OCR_NO_AMOUNT | OCR_EMPTY | OCR_SKIPPED_NON_PDF | HTTP_<code> | ERR_<type>)",
    text_source="pdf_text | ocr | none", text_chars="characters of attachment text",
    order_amount_cr="order-attached amount, Rs crore; NaN when none found, USD without a historical rate, other currency, or the filing is no longer order|tender_L1 under the current CATS",
    parsed_amount_cr="the parsed amount before the cat_current filter (non-orders keep their figure here), Rs crore",
    amount_method="headline_cue | fulltext_cue, optional +range_low | +usd_unconverted | +foreign_ccy_unconverted | +fulltext_inr_equiv; or none_no_text | none_no_amount_in_text | none_boilerplate_only | none_amounts_not_order_attached",
    amount_confidence="high (strong value cue, or headline and full text agree within 2%) | medium | low (range, inferred unit, no currency symbol, INR-equivalent)",
    currency="currency of the chosen figure: INR | USD | OTHER", amount_usd_mn="chosen USD figure in USD million (only when currency=USD)",
    usdinr="USDINR used, as-of filing date, 7-day tolerance (NaN when no data/derived series has a rate)",
    fx_source="macro_panel:fx_feed | macro_panel:fred_dexinus | usdinr_history:fred_dexinus | null",
    usd_unconverted="True when the order amount is in USD and no historical rate exists for that date (never a fixed rate)",
    amount_flags="unit_from_words | unit_inferred_full_amount | range_low | no_currency_symbol | inr_equiv_of_usd | aggregate_of_parts | multiple_amounts",
    headline_fulltext_agree="order-attached headline and full-text figures agree within 2% (null when either is missing)",
    n_amounts_headline="currency amounts in the headline", n_amounts_fulltext="currency amounts in the full text",
    n_attached_fulltext="full-text amounts that are order-attached and not boilerplate",
    legacy_largest_cr="RETIRED rule: largest amount in headline+text (same FX policy) — for comparison only, do not use",
    amount_snippet="text around the chosen figure (for audit)", headline="NSE desc || attchmntText")


def write(T: pd.DataFrame, out: Path = OUT, validation: dict | None = None) -> None:
    T.to_parquet(out, index=False)
    ok = T["order_amount_cr"].notna()
    by_era = {e: dict(rows=int((T["era"] == e).sum()), with_amount=int((ok & (T["era"] == e)).sum()))
              for e in ("disc", "conf")}
    out.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="order_amounts", path=str(out.relative_to(ROOT)), rows=len(T), key=["symbol", "seq_id"],
        producer="src/agentic/parse_order_amounts.py", experiment="EXP-2026-09-27-hot-x-material-order",
        inputs=[str(CKPT.relative_to(ROOT)), str(ANN.relative_to(ROOT)), str(MACRO.relative_to(ROOT)),
                str(USDINR_HIST.relative_to(ROOT))],
        units=dict(order_amount_cr="Rs crore", amount_usd_mn="USD million", usdinr="INR per USD"),
        fx_policy="historical USDINR as-of the filing date (7-day tolerance): macro_panel.usdinr, else usdinr_history "
                  "(FRED DEXINUS); no rate -> order amount NaN + usd_unconverted; never a fixed rate",
        fx_rows_by_source=T["fx_source"].value_counts(dropna=False).rename(index=str).to_dict(),
        rule="order-attached amount: headline first, then full text; candidate within 150 chars after an order cue, "
             "not within 80 chars (same sentence) of boilerplate cues; see module docstring",
        columns=MANIFEST_COLUMNS,
        coverage=dict(by_era=by_era, amount_method=T["amount_method"].value_counts().to_dict(),
                      amount_confidence=T["amount_confidence"].value_counts(dropna=False).rename(index=str).to_dict(),
                      usd_unconverted=int(T["usd_unconverted"].sum()), date_range=[str(T["d"].min()), str(T["d"].max())]),
        validation=validation or VALIDATION,
        caveat="checkpoint may be incomplete while the crawler runs; one amount per filing (not the sum of several orders)",
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0, help="print N random rows with snippets for hand-checking; write nothing")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    D = load_checkpoint()
    T = build(D, load_usdinr())
    if a.validate:
        S = T[T["text_source"] != "none"].sample(min(a.validate, int((T["text_source"] != "none").sum())), random_state=a.seed)
        for r in S.itertuples():
            print(f"==== {r.symbol} {r.seq_id} {r.d} {r.cat}->{r.cat_current} amt={r.order_amount_cr} {r.amount_method} {r.amount_confidence} legacy={r.legacy_largest_cr}")
            print("   HEAD:", (r.headline or "")[:300])
            print("   SNIP:", r.amount_snippet)
        return
    write(T, OUT)
    print(f"order_amounts: {len(T):,} rows · with amount {int(T['order_amount_cr'].notna().sum()):,} · "
          f"usd_unconverted {int(T['usd_unconverted'].sum())}", flush=True)
    print(T["amount_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
