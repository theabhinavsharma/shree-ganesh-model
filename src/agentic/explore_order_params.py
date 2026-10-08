"""What does an order filing say beyond its amount? (2026-10-08, EXPLORATION for Abhinav: "large order distributed over
time hai kya? kuch caveat hai kya usmein bhi? can you go across some orders and find params of the order itself? can we
consume it as one info or need to understand order deeply?"; no rule changed).

For every order filing with text (data/derived/order_fulltext_text.parquet joined to order_amounts.parquet, current
categories order | tender_L1), read from the filing text:
  exec_months   execution / completion period: the duration nearest after an execution cue ("time period by which ... is to
                be executed", "execution period", "completion period", "to be completed", "within N months", "tenure",
                "contract period", "duration"); also "by <Month YYYY>" / "by FY27" converted to months from the filing date
  om            operation & maintenance years mentioned (long tail after construction)
  jv            joint venture / consortium mentioned; jv_pct = a percentage stated next to share / stake / participation
  gst           incl (amount includes GST / taxes) | excl (excludes) | unknown
  kind          framework (rate contract / empanelment) > l1 (lowest bidder, not yet awarded) > loi > loa > po_wo > mou > other
  aggregate     the amount sums several orders (orders worth X received during the quarter / multiple orders / inflow)
  intl          international | domestic | unknown (SEBI's 2023 table states it)
  sebi_table    the filing uses SEBI's order-disclosure table (July 2023+ format: "broad consideration or size", "time
                period by which")
Annual-equivalent ratio = order ratio (order amount / trailing revenue, as in the big-order tests) / max(exec years, 1):
the share of a year's revenue the order can add per year. Unknown period -> not computed.
Outcomes (exploration): the big-order reference trades (logs/leader_sleeve/big_order_hold/reference_trades.parquet, cleaned
order wins >= 15% of trailing revenue) split by these parameters; 12-month return from the entry open (i0) to the close
251 sessions later, vs the equal-weight average of all priced stocks over the same dates. Before costs.
Output: logs/explorations/order_params/ (params.parquet, audit_sample.csv, manifests, README).
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
OUT = ROOT / "logs/explorations/order_params"
MON = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
DUR = r"(\d{1,3}(?:\.\d)?)\s*(?:\(\s*[a-z\- ]+\s*\)\s*)?(months?|mths?|years?|yrs?|days?|weeks?)\b"
CUE = re.compile(r"to be executed|time period by which|execution period|period of execution|completion period|period of completion|"
                 r"to be completed|completion time|completion schedule|delivery period|delivery schedule|contract period|period of contract|"
                 r"tenure|duration|time frame|timeline")
WITHIN = re.compile(r"(?:within|over a period of|over the next|over|in a period of|spread over|for a period of)\s+" + DUR)
BYDATE = re.compile(r"(?:by|till|until|upto|up to|before|ending|end of)\s+(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[,'\s\-]+(20\d\d)")
BYFY = re.compile(r"(?:by|in|during|within|till|until|in the)\s+(?:the\s+)?(?:fy|financial year)\s*'?\s*(?:20)?(\d\d)(?:\s*[-/]\s*(?:20)?(\d\d))?\b")


def months_of(n: str, unit: str) -> float:
    v = float(n)
    return v * 12 if unit.startswith(("year", "yr")) else v / 30.4 if unit.startswith("day") else v / 4.35 if unit.startswith("week") else v


def exec_months(t: str, d: pd.Timestamp) -> tuple[float, str]:
    best = None
    for m in CUE.finditer(t):
        after = re.search(DUR, t[m.end(): m.end() + 220])
        if after:
            return months_of(*after.groups()), t[max(0, m.start() - 20): m.end() + after.end() + 10]
        before = list(re.finditer(DUR, t[max(0, m.start() - 120): m.start()]))
        if before and best is None:
            b = before[-1]; best = (months_of(*b.groups()), t[max(0, m.start() - 120) + b.start() - 20: m.end() + 10])
        bd = BYDATE.search(t[m.end(): m.end() + 220])
        if bd:
            end = pd.Timestamp(int(bd.group(2)), MON[bd.group(1)], 28)
            mo = (end - d).days / 30.4
            if 0 < mo < 240:
                return mo, t[m.start(): m.end() + bd.end() + 10]
        fy = BYFY.search(t[m.end(): m.end() + 220])
        if fy:
            yy = int(fy.group(2) or fy.group(1)); end = pd.Timestamp(2000 + yy, 3, 31)
            mo = (end - d).days / 30.4
            if 0 < mo < 240:
                return mo, t[m.start(): m.end() + fy.end() + 10]
    if best:
        return best
    w = WITHIN.search(t)
    if w and re.search(r"execut|complet|suppl|deliver|commission|implement", t[max(0, w.start() - 150): w.end() + 60]):
        return months_of(*w.groups()), t[max(0, w.start() - 60): w.end() + 10]
    return np.nan, ""


def params(t: str, d: pd.Timestamp) -> dict:
    t = re.sub(r"\s+", " ", t.lower())
    em, snip = exec_months(t, d)
    jv = bool(re.search(r"\bj\.?v\.?\b|joint venture|consortium", t))
    pct = re.search(r"(?:share|stake|participation)(?: ratio)?(?: of)?[^.%]{0,30}?(\d{1,3}(?:\.\d+)?)\s*%|(\d{1,3}(?:\.\d+)?)\s*%\s*(?:share|stake|participation)", t)
    incl = re.search(r"incl(?:\.|uding|usive|usive of|uding of)?\s*(?:of\s*)?(?:all\s*)?(?:applicable\s*)?(?:gst|taxes|tax|duties)", t)
    excl = re.search(r"excl(?:\.|uding|usive|usive of|uding of)?\s*(?:of\s*)?(?:all\s*)?(?:applicable\s*)?(?:gst|taxes|tax|duties)|plus (?:applicable )?(?:gst|taxes)|\+\s*gst|gst extra|taxes extra", t)
    kinds = [("framework", r"rate contract|framework agreement|empanel|annual rate contract|\barc\b"),
             ("l1", r"\bl-?1\b|lowest bid|lowest quoted|lowest price bidder|emerged as the lowest"),
             ("loi", r"letter of intent|\bloi\b"), ("loa", r"letter of (?:acceptance|award)|\bloa\b|notification of award|\bnoa\b"),
             ("po_wo", r"work order|purchase order|supply order|\bpo\b|contract agreement|signed (?:a|the) contract"), ("mou", r"\bmou\b|memorandum of understanding")]
    kind = next((k for k, p in kinds if re.search(p, t)), "other")
    agg = bool(re.search(r"(?:orders?|contracts?)\s+(?:worth|aggregating|totall?ing|amounting|valued)[^.]{0,100}(?:during|in|for)\s+(?:the\s+)?(?:quarter|month|period|q[1-4]|fy)"
                         r"|multiple orders|various orders|several orders|series of orders|order inflow|orders? received (?:during|in) the", t))
    di = re.search(r"domestic\s*/\s*international", t)
    intl = "unknown"
    if di:
        rest = t[di.end(): di.end() + 250].replace("domestic/ international", "").replace("domestic/international", "")
        h = re.search(r"\b(domestic|international|overseas|export)\b", rest)
        intl = "unknown" if not h else ("domestic" if h.group(1) == "domestic" else "international")
    om = re.search(r"o\s?&\s?m|operation(?:s)? (?:and|&) maintenance", t)
    return dict(exec_months=em, exec_snip=snip[:240], om=bool(om), jv=jv, jv_pct=float(pct.group(1) or pct.group(2)) if pct else np.nan,
                gst="incl" if incl and not excl else "excl" if excl and not incl else "both" if incl and excl else "unknown",
                kind=kind, aggregate=agg, intl=intl, sebi_table=bool(re.search(r"broad consideration or size|time period by which", t)))


O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet")
O = O[O["cat_current"].isin(["order", "tender_L1"])]
T = pd.read_parquet(ROOT / "data/derived/order_fulltext_text.parquet")
M = O.merge(T, on=["symbol", "seq_id"], how="inner")
M = M[M["text"].fillna("").str.len() > 50].copy()
M["d"] = pd.to_datetime(M["d"])
X = pd.DataFrame([params(t, d) for t, d in zip(M["text"], M["d"])], index=M.index)
P = pd.concat([M[["symbol", "seq_id", "d", "ts", "cat_current", "order_amount_cr", "headline"]], X], axis=1)
P["year"] = P["d"].dt.year
OUT.mkdir(parents=True, exist_ok=True)

# ---- coverage by year
g = P.groupby("year")
cov = pd.DataFrame(dict(filings=g.size(), sebi_table=g["sebi_table"].mean() * 100, period_stated=g["exec_months"].apply(lambda x: x.notna().mean() * 100),
                        jv=g["jv"].mean() * 100, gst_incl=g["gst"].apply(lambda x: (x == "incl").mean() * 100),
                        gst_excl=g["gst"].apply(lambda x: (x == "excl").mean() * 100), aggregate=g["aggregate"].mean() * 100,
                        om=g["om"].mean() * 100, intl=g["intl"].apply(lambda x: (x == "international").mean() * 100))).round(0)
print("=== order filings with text, by year · % of filings ===\n" + cov.to_string())
print("\nkind of order (% of filings): " + (P["kind"].value_counts(normalize=True) * 100).round(1).to_string().replace("\n", " · "))
e = P["exec_months"].dropna()
print(f"\nexecution period stated: {len(e)} of {len(P)} · months: 25th {e.quantile(.25):.0f} · median {e.median():.0f} · 75th {e.quantile(.75):.0f} · "
      f"share <= 12m {(e <= 12).mean():.0%} · 12-24m {((e > 12) & (e <= 24)).mean():.0%} · 24-36m {((e > 24) & (e <= 36)).mean():.0%} · > 36m {(e > 36).mean():.0%}")

# ---- the big-order trades
R = pd.read_parquet(ROOT / "logs/leader_sleeve/big_order_hold/reference_trades.parquet")
R = R.merge(P.drop(columns=["headline"]), on=["symbol", "ts"], how="left")
print(f"\nbig-order reference trades: {len(R)} · matched to a filing text {R['kind'].notna().sum()} · period stated {R['exec_months'].notna().sum()}")
R["years"] = (R["exec_months"] / 12).clip(lower=1)
R["ratio_annual"] = R["ratio"] / R["years"]
k = R.dropna(subset=["exec_months"])
for cut in (0.15, 0.5):
    big = k[k["ratio"] >= cut]
    print(f"  ratio >= {cut:.0%} with a stated period: {len(big)} · still >= {cut:.0%} per year of execution: {(big['ratio_annual'] >= cut).sum()} "
          f"({(big['ratio_annual'] >= cut).mean():.0%}) · median period {big['exec_months'].median():.0f} months")
for col in ("jv", "aggregate", "om"):
    print(f"  {col}: {R[col].fillna(False).astype(bool).mean():.0%} of trades")
print("  kind: " + (R["kind"].value_counts(normalize=True) * 100).round(0).to_string().replace("\n", " · "))
print("  gst: " + (R["gst"].value_counts(normalize=True) * 100).round(0).to_string().replace("\n", " · "))

# ---- outcomes by parameter (exploration)
import research_panel as rp  # noqa: E402
PX = rp.load_panel(["open", "close"]); cal = rp.session_calendar(PX)
Ow = rp.wide(PX, "open", cal); Cw = rp.wide(PX, "close", cal).ffill(limit=300); del PX
On, Cn, col = Ow.to_numpy(), Cw.to_numpy(), {s: i for i, s in enumerate(Ow.columns)}
H = 251


def ret(r):
    c = col.get(r.symbol); a = int(r.i0)
    if c is None or a + H >= len(cal) or not np.isfinite(On[a, c]) or On[a, c] <= 0 or not np.isfinite(Cn[a + H, c]):
        return np.nan, np.nan
    e, x = On[a], Cn[a + H]; ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
    return Cn[a + H, c] / On[a, c] - 1, float(np.mean(np.clip(x[ok] / e[ok] - 1, -1, 10)))


R[["r12", "mkt12"]] = [ret(r) for r in R.itertuples()]
R["exc12"] = R["r12"] - R["mkt12"]
R["period"] = pd.cut(R["exec_months"], [0, 12, 24, 36, 1e9], labels=["<= 12m", "12-24m", "24-36m", "> 36m"]).astype(str).replace("nan", "not stated")
R["annual_band"] = np.where(R["ratio_annual"].isna(), "not stated", np.where(R["ratio_annual"] >= 0.5, ">= 50%/yr", np.where(R["ratio_annual"] >= 0.15, "15-50%/yr", "< 15%/yr")))
R["firm"] = R["kind"].map({"l1": "not yet awarded (L1)", "loi": "LoI", "framework": "framework / rate contract", "mou": "MoU"}).fillna("awarded (LoA / PO / WO / other)")
done = R.dropna(subset=["r12"])
print(f"\n=== big-order trades with a finished 12-month hold: {len(done)} · 12m return avg (median) · vs market · [n] — exploration, before costs ===")
for c in ("period", "annual_band", "firm", "jv", "aggregate", "gst", "sebi_table"):
    s = done.groupby(c).agg(n=("r12", "size"), avg=("r12", "mean"), med=("r12", "median"), exc=("exc12", "mean"))
    print(f" {c}: " + " | ".join(f"{i}: {v.avg:+.0%} ({v.med:+.0%}) vs mkt {v.exc:+.0%} [{int(v.n)}]" for i, v in s.iterrows()))

# ---- audit sample: 30 random filings with a period, 12 big-order trades
aud = pd.concat([P.dropna(subset=["exec_months"]).sample(30, random_state=7).assign(sample="random filing with a period"),
                 R.dropna(subset=["kind"]).sort_values("ratio", ascending=False).head(12).assign(sample="largest-ratio trade")])
aud[["sample", "symbol", "seq_id", "d", "order_amount_cr", "ratio", "exec_months", "exec_snip", "jv", "jv_pct", "gst", "kind", "aggregate"]].to_csv(OUT / "audit_sample.csv", index=False)
P.drop(columns=["headline"]).to_parquet(OUT / "params.parquet", index=False)
R.to_parquet(OUT / "big_order_trades_params.parquet", index=False)
man = dict(producer="src/agentic/explore_order_params.py", status="EXPLORATION (no rule changed)", definitions=__doc__,
           units=dict(exec_months="months", jv_pct="percent", order_amount_cr="Rs crore", ratio="fraction of trailing revenue",
                      ratio_annual="fraction of trailing revenue per year of execution", r12="fraction", mkt12="fraction", exc12="fraction"),
           coverage_by_year=cov.to_dict(orient="index"), updated=datetime.now().isoformat(timespec="seconds"))
for f in ("params.parquet", "big_order_trades_params.parquet", "audit_sample.csv"):
    (OUT / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, **man), indent=1, default=str))
(OUT / "README.md").write_text("# Order parameters (exploration)\n\nWhat order filings say beyond the amount: execution period, joint-venture share, GST, "
                               "L1 / LoI / LoA, aggregated orders. params.parquet = every order filing with text; big_order_trades_params.parquet = "
                               "the big-order reference trades with these fields and 12-month outcomes; audit_sample.csv = rows read by hand. "
                               "Definitions in the manifests.\n")
print("done")
