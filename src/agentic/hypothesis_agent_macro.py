"""Macro / aggregate hypothesis agent — appends NON-CONVENTIONAL,
NON-PER-STOCK factors to the registry.

Categories:
  • commodity_macro       — crude, gold, copper, agri commodities
  • global_rates_macro    — US 10y, DXY, VIX, SPX
  • mf_flow_macro         — AMFI AUM, SIP, equity allocation
  • market_breadth_macro  — breadth, dispersion, new highs/lows
  • sector_rotation_macro — sector RS, breadth, dispersion
  • macro_sentiment       — country/topic-level news tone
  • cross_asset_macro     — gold/copper ratio, brent×inr (importer pain)
  • behavioral_macro      — fear/greed (VIX z), risk-on/off composite

These factors apply IDENTICALLY to all stocks on a given day; they are
*regime indicators*, not per-stock signals. The model can interact them
with stock-level features (e.g. high-beta names underperform when VIX z > 2).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
REGISTRY = ROOT / "data/derived/factor_registry.json"

MACRO_HYPOTHESES = [
    # ════════════ COMMODITY MACRO ════════════
    {"id": "macro_brent_5d", "name": "Brent crude 5d return",
     "category": "commodity_macro",
     "description": "Brent up 3%+ in 5d → OMC margin pain, aviation/paint cost shock, CPI risk",
     "formula": "brent.pct_change(5)",
     "data_needed": ["brent"], "has_data": True,
     "notes": "fetched via fetch_commodity_prices.py"},

    {"id": "macro_brent_60d_regime", "name": "Brent 60d regime",
     "category": "commodity_macro",
     "description": "Brent +20% over 60d = inflation + INR pressure (large-cap pharma/IT lift)",
     "formula": "brent.pct_change(60)",
     "data_needed": ["brent"], "has_data": True},

    {"id": "macro_gold_5d", "name": "Gold 5d return (safe-haven flow)",
     "category": "commodity_macro",
     "description": "Gold rising while equities flat = risk-off rotation; defensive lift",
     "formula": "gold.pct_change(5)",
     "data_needed": ["gold"], "has_data": True},

    {"id": "macro_copper_60d", "name": "Dr. Copper — 60d global growth signal",
     "category": "commodity_macro",
     "description": "Copper rising 60d = global capex cycle; metals + capital goods lift in India",
     "formula": "copper.pct_change(60)",
     "data_needed": ["copper"], "has_data": True},

    {"id": "macro_gold_brent_ratio", "name": "Gold/Brent ratio (risk regime)",
     "category": "commodity_macro",
     "description": "Gold/Brent rising = global risk-off (oil weak, gold strong)",
     "formula": "gold / brent",
     "data_needed": ["gold", "brent"], "has_data": True},

    {"id": "macro_natgas_5d", "name": "Natural gas 5d (city-gas margin signal)",
     "category": "commodity_macro",
     "description": "Henry Hub up 10%+ in 5d → IGL/MGL/GAIL margin compression",
     "formula": "natgas.pct_change(5)",
     "data_needed": ["natgas"], "has_data": True},

    {"id": "macro_zinc_aluminum", "name": "Zinc + Aluminum 60d (Hindalco/Vedanta direct)",
     "category": "commodity_macro",
     "description": "Base-metals 60d return drives metals-sector earnings",
     "formula": "(zinc.pct_change(60) + aluminum.pct_change(60)) / 2",
     "data_needed": ["zinc", "aluminum"], "has_data": True},

    {"id": "macro_agri_basket", "name": "Agri commodity basket (wheat+corn+sugar)",
     "category": "commodity_macro",
     "description": "Agri up = rural inflation; HUL/ITC/Marico margin pressure but rural-discretionary lifts",
     "formula": "(wheat.pct_change(60) + corn.pct_change(60) + sugar.pct_change(60)) / 3",
     "data_needed": ["wheat", "corn", "sugar"], "has_data": True},

    # ════════════ GLOBAL RATES MACRO ════════════
    {"id": "macro_us10y_5d_chg", "name": "US 10y yield 5d change",
     "category": "global_rates_macro",
     "description": "US 10y +20bp in 5d → FII outflow risk, EM equities sell off",
     "formula": "us_10y.diff(5)",
     "data_needed": ["us_10y"], "has_data": True},

    {"id": "macro_dxy_5d", "name": "DXY 5d return (INR pressure)",
     "category": "global_rates_macro",
     "description": "DXY up 2%+ in 5d historically → -3% NIFTY 7d ahead",
     "formula": "dxy.pct_change(5)",
     "data_needed": ["dxy"], "has_data": True},

    {"id": "macro_us_vix_z", "name": "US VIX z-score 60d",
     "category": "global_rates_macro",
     "description": "VIX z > 2 → high-beta India underperforms 5-10d ahead",
     "formula": "(us_vix - mean(us_vix, 60)) / std(us_vix, 60)",
     "data_needed": ["us_vix"], "has_data": True},

    {"id": "macro_spx_60d", "name": "S&P 500 60d return (global risk appetite)",
     "category": "global_rates_macro",
     "description": "SPX +5%+ over 60d = global risk-on; small/mid caps in India also rally",
     "formula": "spx.pct_change(60)",
     "data_needed": ["spx"], "has_data": True},

    {"id": "macro_eem_relative_spx", "name": "EEM vs SPX 5d relative",
     "category": "global_rates_macro",
     "description": "EEM > SPX = EM rotation in (positive for India), opposite = out",
     "formula": "eem.pct_change(5) - spx.pct_change(5)",
     "data_needed": ["eem", "spx"], "has_data": True},

    {"id": "macro_china_proxy", "name": "Hang Seng 60d (China proxy)",
     "category": "global_rates_macro",
     "description": "HSI up = China stimulus working; Indian metal/textile/chem importers feel pinch",
     "formula": "hsi.pct_change(60)",
     "data_needed": ["hsi"], "has_data": True},

    {"id": "macro_jpy_carry_unwind", "name": "JPY strength (carry-unwind risk)",
     "category": "global_rates_macro",
     "description": "USDJPY -2%+ in 5d = carry-trade unwind, global risk cascade",
     "formula": "-1 * usdjpy.pct_change(5)",
     "data_needed": ["usdjpy"], "has_data": True},

    {"id": "macro_btc_proxy", "name": "Bitcoin 30d return (liquidity proxy)",
     "category": "global_rates_macro",
     "description": "BTC +20% over 30d = liquidity flush, global risk-on",
     "formula": "btc.pct_change(30)",
     "data_needed": ["btc"], "has_data": True},

    # ════════════ INR + IMPORTER PAIN ════════════
    {"id": "macro_inr_5d", "name": "USDINR 5d move (INR weakness)",
     "category": "global_rates_macro",
     "description": "INR weakening 1%+ in 5d → IT/Pharma USD-revenue lift",
     "formula": "usdinr.pct_change(5)",
     "data_needed": ["usdinr"], "has_data": True},

    {"id": "macro_brent_inr_pain", "name": "Brent × INR (importer pain index)",
     "category": "commodity_macro",
     "description": "(brent × usdinr) up 5%+ in 20d = double-whammy on margins",
     "formula": "(brent * usdinr).pct_change(20)",
     "data_needed": ["brent", "usdinr"], "has_data": True},

    # ════════════ MF FLOW MACRO ════════════
    {"id": "macro_mf_equity_yoy", "name": "MF equity AUM YoY %",
     "category": "mf_flow_macro",
     "description": "Equity AUM growing 25%+ YoY = strong domestic flow (cushion in selloffs)",
     "formula": "equity_aum_cr / equity_aum_cr.shift(252) - 1",
     "data_needed": ["equity_aum_cr"], "has_data": True,
     "notes": "monthly data ffill'd to daily"},

    {"id": "macro_sip_yoy", "name": "SIP inflow YoY %",
     "category": "mf_flow_macro",
     "description": "SIP YoY > 20% = sticky retail money; market drawdowns absorbed",
     "formula": "sip_inflow_cr / sip_inflow_cr.shift(252) - 1",
     "data_needed": ["sip_inflow_cr"], "has_data": True},

    {"id": "macro_mf_equity_share", "name": "MF equity share of total AUM",
     "category": "mf_flow_macro",
     "description": "Equity share rising = re-allocation from debt; multibagger fuel",
     "formula": "equity_aum_cr / total_aum_cr",
     "data_needed": ["equity_aum_cr", "total_aum_cr"], "has_data": True},

    # ════════════ MARKET BREADTH MACRO ════════════
    {"id": "macro_breadth_50_5d_chg", "name": "NSE breadth_50 5d change",
     "category": "market_breadth_macro",
     "description": "Breadth widening 5%+ in 5d = follow-through; multibagger basket window",
     "formula": "breadth_50.diff(5)",
     "data_needed": ["breadth_50"], "has_data": True,
     "notes": "from market_breadth_panel"},

    {"id": "macro_breadth_dispersion_z", "name": "Cross-section dispersion z-score",
     "category": "market_breadth_macro",
     "description": "Dispersion z > 1.5 = stock-pickers' market; multibagger names emerge",
     "formula": "(dispersion_20d - mean(dispersion_20d, 60)) / std(...)",
     "data_needed": ["cross_section_dispersion_20d"], "has_data": True},

    {"id": "macro_new_high_low_diff", "name": "New 52w highs minus lows",
     "category": "market_breadth_macro",
     "description": "Net new highs ≥ +20 = momentum environment; breakouts work",
     "formula": "new_52w_highs - new_52w_lows",
     "data_needed": ["new_52w_highs", "new_52w_lows"], "has_data": True},

    {"id": "macro_smid_lcap_rotation", "name": "Small/mid-cap vs large-cap breadth diff",
     "category": "market_breadth_macro",
     "description": "Smid breadth > lcap breadth by 10pp = small-cap rally regime",
     "formula": "breadth_50_smid - breadth_50_lcap",
     "data_needed": ["breadth_50_smid", "breadth_50_lcap"], "has_data": True},

    # ════════════ SECTOR ROTATION MACRO ════════════
    {"id": "macro_sector_rs_top_q", "name": "Top-quartile sector RS_60d",
     "category": "sector_rotation_macro",
     "description": "Stocks belonging to top-RS sectors outperform; sector tailwind effect",
     "formula": "rank within sector_rs_60d, top quartile dummy",
     "data_needed": ["sector_rs_60d"], "has_data": True,
     "notes": "join to industry_panel"},

    {"id": "macro_sector_breadth_lift", "name": "Sector breadth 5d-rising",
     "category": "sector_rotation_macro",
     "description": "Sector breadth rising 10pp+ in 5d = sector momentum entering",
     "formula": "sector_breadth_50.diff(5) per sector",
     "data_needed": ["sector_breadth_50"], "has_data": True},

    {"id": "macro_sector_dispersion", "name": "Within-sector dispersion (low = leader-driven)",
     "category": "sector_rotation_macro",
     "description": "Low sector dispersion = leaders carry sector; high = stockpicker",
     "formula": "sector_dispersion_20d",
     "data_needed": ["sector_dispersion_20d"], "has_data": True},

    # ════════════ MACRO SENTIMENT ════════════
    {"id": "macro_sentiment_avg", "name": "Macro news sentiment composite",
     "category": "macro_sentiment",
     "description": "Avg sentiment across rbi/economy/fii/oil/fed/geopolitics topics; risk-on proxy",
     "formula": "mean(sentiment_7d) across all macro topics",
     "data_needed": ["macro_sent_avg"], "has_data": True},

    {"id": "macro_sent_recession_risk", "name": "Recession-risk topic sentiment",
     "category": "macro_sentiment",
     "description": "Negative recession-risk sentiment = upcoming sell pressure",
     "formula": "macro_sent__recession_risk",
     "data_needed": ["macro_sent__recession_risk"], "has_data": True},

    {"id": "macro_sent_rbi_dovish", "name": "RBI policy sentiment (dovish dummy)",
     "category": "macro_sentiment",
     "description": "Positive RBI sentiment = rate cut expected; banks/NBFCs lift",
     "formula": "macro_sent__rbi_policy",
     "data_needed": ["macro_sent__rbi_policy"], "has_data": True},

    # ════════════ COMPOSITE / INTERACTION ════════════
    {"id": "macro_risk_off_composite", "name": "Risk-off composite (DXY+VIX+gold up)",
     "category": "behavioral_macro",
     "description": "DXY 5d > 0 AND VIX z > 1 AND gold 5d > 0 = full risk-off triple",
     "formula": "(dxy_5d>0)*(us_vix_z_60d>1)*(gold_5d_pct>0)",
     "data_needed": ["dxy", "us_vix", "gold"], "has_data": True},

    {"id": "macro_risk_on_composite", "name": "Risk-on composite (SPX+EEM+breadth)",
     "category": "behavioral_macro",
     "description": "SPX 60d > 5% AND EEM > SPX AND NSE breadth_50 > 60% = full risk-on",
     "formula": "(spx_60d>0.05)*(eem_rs_to_spx>0)*(breadth_50>0.60)",
     "data_needed": ["spx", "eem", "breadth_50"], "has_data": True},

    {"id": "macro_regime_gate_v1_dummy", "name": "Regime Gate v1 dummy (deploy day flag)",
     "category": "behavioral_macro",
     "description": "Encodes the validated gate: market_20d ≤ -2% AND breadth_50 ∈ [50,75]",
     "formula": "(market_20d_sum <= -0.02) * (breadth_50 in [0.5, 0.75])",
     "data_needed": ["market_20d_sum", "breadth_50"], "has_data": True,
     "notes": "tests whether the gate itself works as a feature in the model"},

    # ════════════ CROSS-ASSET LEAD-LAG ════════════
    {"id": "macro_oil_to_omc_lead", "name": "Oil 5d → OMC lag (sector inverse)",
     "category": "cross_asset_macro",
     "description": "Brent up 5%+ → BPCL/HPCL/IOC underperform 3-5d later",
     "formula": "brent_5d_pct (interacted with is_omc dummy)",
     "data_needed": ["brent"], "has_data": True},

    {"id": "macro_yield_to_bank_lead", "name": "10y yield → bank lift lag",
     "category": "cross_asset_macro",
     "description": "US 10y +30bp → India banks NIM expansion thesis (5-10d lift)",
     "formula": "us_10y_5d_chg (interacted with is_bank dummy)",
     "data_needed": ["us_10y"], "has_data": True},

    {"id": "macro_inr_to_it_pharma", "name": "INR weakness → IT/Pharma lift",
     "category": "cross_asset_macro",
     "description": "INR -1%+ in 5d → IT/Pharma export-heavy names lift 3-7d",
     "formula": "usdinr.pct_change(5) (interacted with is_it_pharma dummy)",
     "data_needed": ["usdinr"], "has_data": True},

    {"id": "macro_copper_to_capital_goods", "name": "Copper 60d → capital goods lift",
     "category": "cross_asset_macro",
     "description": "Copper +10% in 60d → L&T/Siemens/ABB get re-rated upward 30d ahead",
     "formula": "copper.pct_change(60) (interacted with is_capital_goods)",
     "data_needed": ["copper"], "has_data": True},
]


def main() -> None:
    print(f"== hypothesis_agent_macro: adding {len(MACRO_HYPOTHESES)} macro/aggregate hypotheses ==")
    if not REGISTRY.exists():
        print(f"  registry not found at {REGISTRY} — initialize via factor_registry.py first")
        return
    reg = json.loads(REGISTRY.read_text())
    existing_ids = {h["id"] for h in reg}
    added = 0
    for h in MACRO_HYPOTHESES:
        if h["id"] in existing_ids:
            continue
        h.setdefault("state", "PROPOSED")
        h.setdefault("lift_ic", None)
        h.setdefault("lift_top5_precision", None)
        h.setdefault("notes", "")
        reg.append(h)
        added += 1
    REGISTRY.write_text(json.dumps(reg, indent=2))
    print(f"  added {added} new hypotheses (skipped {len(MACRO_HYPOTHESES)-added} dupes)")
    print(f"  total hypotheses now: {len(reg)}")
    by_cat = {}
    for h in reg:
        by_cat[h["category"]] = by_cat.get(h["category"], 0) + 1
    print("\nBy category:")
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {c:<24} {n:>3}")


if __name__ == "__main__":
    main()
