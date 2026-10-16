"""Company lists behind the SGM Data World bot cards ("Check oil & fuel companies in India >").

Writes shreeganeshmodel-deploy/world/companies.json from repo data only:
  industries  data/derived/screener_industry.parquet (the NSE four-level industry V3.2 uses)
  size        data/derived/mcap_pit.parquet (latest market cap; used only to order the list, biggest first)
  policy      data/derived/industry_scores_policy.parquet (latest P_pct; "industries the government is pushing")
Groups are fixed industry sets, except "govt", which takes the five industries with the highest policy score on the
latest date. Each list shows up to 15 companies. Run before deploying the site; the page shows the as-of dates.
"""
import html
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "shreeganeshmodel-deploy/world/companies.json"
GROUPS = {
    "oil": ("Oil and fuel companies in India", ["Oil Exploration & Production", "Refineries & Marketing", "Gas Transmission/Marketing",
                                                "LPG/CNG/PNG/LNG Supplier", "Oil Equipment & Services", "Oil Storage & Transportation"]),
    "exporters": ("Companies that earn in dollars (IT and pharma exporters)", ["Computers - Software & Consulting", "IT Enabled Services", "Pharmaceuticals"]),
    "metals": ("Metal companies in India", ["Iron & Steel", "Aluminium", "Copper", "Zinc", "Diversified Metals"]),
}


def main() -> int:
    sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry"]).copy()
    sc["industry"] = sc["industry"].map(html.unescape)
    mc = pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet", columns=["symbol", "trade_date", "mcap_cr"]).dropna(subset=["mcap_cr"])
    mc_date = mc["trade_date"].max()
    last = mc.sort_values("trade_date").groupby("symbol").tail(1)
    last = last[last["trade_date"] >= mc_date - pd.Timedelta(days=10)].set_index("symbol")["mcap_cr"]
    pol = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); pol["date"] = pd.to_datetime(pol["date"])
    pdate = pol["date"].max(); top = pol[pol["date"] == pdate].dropna(subset=["P_pct"]).sort_values("P_pct", ascending=False).head(5)
    sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet", columns=["symbol", "company_name"]).dropna()
    names = sm.drop_duplicates("symbol", keep="last").set_index("symbol")["company_name"]
    groups = dict(GROUPS)
    groups["govt"] = ("Industries in the government's news most right now", top["industry"].tolist())

    def listing(inds):
        g = sc[sc["industry"].isin(inds)][["symbol", "company_name", "industry"]].copy()
        g["mcap"] = g["symbol"].map(last)
        g["company_name"] = g["symbol"].map(names).fillna(g["company_name"])
        g = g.dropna(subset=["mcap"]).sort_values("mcap", ascending=False).head(15)
        return [dict(symbol=r.symbol, name=html.unescape(str(r.company_name or r.symbol)), industry=r.industry) for r in g.itertuples()]

    out = dict(built=str(date.today()), market_cap_as_of=str(mc_date.date()), policy_as_of=str(pdate.date()),
               source="screener.in industry labels (NSE four-level industry), market caps from the repo's mcap_pit table, government focus from PIB-based policy scores",
               groups={k: dict(title=t, industries=inds, companies=listing(inds)) for k, (t, inds) in groups.items()})
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: " + ", ".join(f"{k} {len(v['companies'])}" for k, v in out["groups"].items()) + f" · govt top: {top['industry'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
