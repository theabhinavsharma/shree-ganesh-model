import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/agentic"))
import build_security_master as bsm  # noqa: E402
import generate_hybrid_basket as ghb  # noqa: E402
from generate_hybrid_basket import non_equity  # noqa: E402


def test_etfs_excluded():
    s = pd.Series(["GOLDBEES", "SETFGOLD", "GOLDIETF", "MASPTOP50", "NIFTYBEES", "HDFCGOLD",
                   "LIQUID", "SILVER", "SETFNIF50", "EBBETF0433", "N100", "TOP10ADD"])
    assert non_equity(s).all()


def test_gold_named_companies_kept():
    s = pd.Series(["SKYGOLD", "SHANTIGOLD", "GOLDTECH", "GOLDIAM", "TBZ", "TITAN"])
    assert not non_equity(s).any()


# ---- 2026-09-27 audit: rights entitlements and dead no-ISIN ETFs ------------------------------
def test_rights_entitlements_excluded():
    # the five found in the panel, plus -RE / -RE1 / -RE2 forms seen in raw 2020+ sec_bhavdata_full
    s = pd.Series(["DUCON-RE1", "JAYKAY-RE1", "KSHITIJ-RE", "RATNA-RE", "VHLTD-RE1",
                   "AIRTEL-RE", "BHANDA-RE2", "3IINFO-RE", "jaykay-re1"])
    assert non_equity(s).all()


def test_rights_rule_not_masked_by_company_isin(monkeypatch):
    # even if a future master gives the entitlement an INE ISIN (company set), it stays non-equity
    monkeypatch.setitem(ghb._MASTER_CACHE, "v", (set(), {"JAYKAY-RE1", "JAYKAY"}))
    out = non_equity(pd.Series(["JAYKAY-RE1", "JAYKAY"]))
    assert out.tolist() == [True, False]


def test_hyphenated_companies_kept():
    s = pd.Series(["BAJAJ-AUTO", "NAM-INDIA", "MCDOWELL-N", "MRO-TEK", "SHIV-VANI", "KLBRENG-B",
                   "BOSCH-HCIL", "HCL-INSYS", "UMIYA-MRO", "RELIANCE", "IRE", "PREMIERENE"])
    assert not non_equity(s).any()


def test_dead_no_isin_etfs_excluded():
    assert non_equity(pd.Series(["NETFIT", "NAVINIFTY", "AXISBPSETF", "SDL24BEES", "SDL26BEES",
                                 "EBBETF0423", "EBBETF0425"])).all()


def test_signature_backward_compatible():
    # same call shape as the 15D pipeline: Series in, boolean Series out, aligned to the input index
    s = pd.Series(["NIFTYBEES", "TITAN", "JAYKAY-RE1"], index=[10, 20, 30])
    out = non_equity(s)
    assert isinstance(out, pd.Series) and out.dtype == bool
    assert out.index.tolist() == [10, 20, 30]
    assert out.tolist() == [True, False, True]


def test_master_is_non_equity_column_used(monkeypatch, tmp_path):
    # a master with is_non_equity (2026-09-27+ builds) drives the non-equity set; an old master
    # without it falls back to is_fund_unit
    new = pd.DataFrame(dict(symbol=["XYZ-RE", "ABCETFX", "COMP"], isin=[None, "INF000", "INE000"],
                            is_fund_unit=[False, True, False], is_non_equity=[True, True, False]))
    old = new.drop(columns="is_non_equity")
    for frame, expect_fund in ((new, {"XYZ-RE", "ABCETFX"}), (old, {"ABCETFX"})):
        p = tmp_path / "m.parquet"
        frame.to_parquet(p, index=False)
        monkeypatch.setattr(ghb, "SECURITY_MASTER", p)
        ghb._MASTER_CACHE.clear()
        fund, company = ghb._master()
        assert fund == expect_fund and company == {"COMP"}
    ghb._MASTER_CACHE.clear()


def test_builder_and_basket_rights_regex_agree():
    syms = ["DUCON-RE1", "RATNA-RE", "BHANDA-RE2", "BAJAJ-AUTO", "NAM-INDIA", "PREMIERENE", "IRE", "X-REIT"]
    a = [bool(bsm.RIGHTS_ENT_RE.search(x)) for x in syms]
    b = [bool(ghb.RIGHTS_ENTITLEMENT_RE.search(x)) for x in syms]
    assert a == b == [True, True, True, False, False, False, False, False]


# ---- security master classification ------------------------------------------------------------
def _toy_master():
    m = pd.DataFrame({
        "isin": [None, "INF204KB15I9", None, "INE000A01010", None, None, None, None, None, None, "INE111A01011"],
        "company_name": ["Nippon Life India Asset Management Limited", None, None, "Some Gold Ltd", None, None,
                         None, None, "Nippon India Mutual Fund - ETF Nifty SDL - 2026", None, "Some ETF Company Ltd"],
        "series_seen": ["EQ", "EQ", "EQ", "EQ", "BE", "BE", "RE", "EQ", "EQ", "EQ", "EQ"],
    }, index=["NETFIT", "ITBEES", "NAVINIFTY", "SKYGOLD", "JAYKAY-RE1", "KSHITIJ-RE", "ODDSERIES",
              "EBBETF0423", "NETFSDL26", "SDL26BEES", "COMPANYX"])
    sc = pd.DataFrame({"old": ["NETFIT", "NETFSDL26", "COMPANYX"], "new": ["ITBEES", "SDL26BEES", "NETFIT_OLDNAME"]})
    isin_map = pd.Series({"ITBEES": "INF204KB15I9", "SKYGOLD": "INE000A01010", "COMPANYX": "INE111A01011"})
    return bsm.classify_instruments(m, isin_map, {"ITBEES"}, sc)


def test_master_flags_netfit_via_rename_and_navinifty_manual():
    m = _toy_master()
    assert m.at["NETFIT", "is_fund_unit"] and m.at["NETFIT", "fund_unit_source"] == "rename:ITBEES"
    assert m.at["ITBEES", "fund_unit_source"] == "isin_INF"
    assert m.at["NAVINIFTY", "is_fund_unit"] and m.at["NAVINIFTY", "fund_unit_source"] == "manual"
    assert m.at["EBBETF0423", "fund_unit_source"] == "symbol_pattern_no_isin"
    assert m.at["NETFSDL26", "fund_unit_source"] == "registered_name"
    assert m.at["SDL26BEES", "fund_unit_source"] == "rename:NETFSDL26"


def test_master_rights_entitlements_are_non_equity_not_funds():
    m = _toy_master()
    for s in ("JAYKAY-RE1", "KSHITIJ-RE", "ODDSERIES"):          # ODDSERIES: flagged by series RE
        assert m.at[s, "is_rights_entitlement"] and m.at[s, "is_non_equity"]
        assert not m.at[s, "is_fund_unit"] and m.at[s, "instrument_type"] == "rights_entitlement"


def test_master_ine_companies_never_flagged():
    m = _toy_master()
    for s in ("SKYGOLD", "COMPANYX"):                               # COMPANYX: 'ETF' in name but INE ISIN
        assert not m.at[s, "is_non_equity"] and m.at[s, "instrument_type"] == "equity"
    assert set(m["instrument_type"]) <= {"fund_unit", "rights_entitlement", "equity", "equity_no_isin"}


# ---- ISIN from raw bhavcopies: every format, ISIN used only where present ----------------------
def test_isin_scan_reads_every_format(tmp_path):
    def day(d):
        p = tmp_path / f"trade_date={d}"; p.mkdir()
        return p
    old = day("2019-12-31") / "cm31DEC2019bhav.csv.zip"
    with zipfile.ZipFile(old, "w") as z:
        z.writestr("cm31DEC2019bhav.csv", "SYMBOL,SERIES,OPEN,CLOSE,ISIN,\nTITAN,EQ,1,2,INE280A01028,\nNIFTYBEES,EQ,1,2,INF204KB14I2,\n")
    (day("2024-07-08") / "sec_bhavdata_full_08072024.csv").write_text(
        "SYMBOL, SERIES, DATE1, CLOSE_PRICE\nTITAN, EQ, 08-Jul-2024, 3000\n")
    (day("2024-07-09") / "sec_bhavdata_full_09072024.csv").write_text(          # hypothetical ISIN-bearing file
        "SYMBOL, SERIES, DATE1, CLOSE_PRICE, ISIN\nNEWCO, EQ, 09-Jul-2024, 10, INE999Z01011\n")
    (day("2024-07-10") / "BhavCopy_NSE_CM_0_0_0_20240710_F_0000.csv").write_text(
        "TradDt,TckrSymb,SctySrs,ISIN,ClsPric\n2024-07-10,NETFIT2,EQ,INF000X01011,31\n")
    (day("2024-07-11") / "sec_bhavdata_full_11072024.csv").write_bytes(b"")     # unreadable -> counted
    b, cov = bsm._isins_from_bhavcopies(bsm._bhav_files(tmp_path))
    got = dict(zip(b["symbol"], b["isin"]))
    assert got == {"TITAN": "INE280A01028", "NIFTYBEES": "INF204KB14I2",
                   "NEWCO": "INE999Z01011", "NETFIT2": "INF000X01011"}
    c = cov["sec_bhavdata_full_csv"]
    assert (c["files"], c["files_with_isin"], c["files_without_isin"]) == (3, 1, 1)
    assert c["files_failed"] == 1
    assert cov["old_format_cm_bhav_zip"]["isin_last_date"] == "2019-12-31"
    assert cov["udiff_bhavcopy_nse_cm"]["files_with_isin"] == 1
    txt = bsm.isin_source_text(cov)
    assert "2019-12-31" in txt and "2016-2024" not in txt
