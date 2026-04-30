import io
import zipfile

import pandas as pd

from src.ingest.derivatives.nse_oi import _parse_daily_zip
from src.ingest.events.nse_bulk_block import _normalize_rows as _normalize_bulk_block_rows
from src.ingest.events.nse_insider import _normalize_rows as _normalize_insider_rows


def test_normalize_insider_rows_maps_values_and_roles() -> None:
    rows = [
        {
            "symbol": "ABC",
            "company": "ABC Ltd",
            "date": "15-Mar-2026 13:19",
            "pid": "123",
            "acqName": "Jane Doe",
            "personCategory": "Promoter Group",
            "tdpTransactionType": "Buy",
            "buyValue": "397440",
            "sellValue": "0",
            "buyQuantity": "800",
            "sellquantity": "0",
            "secVal": "397440",
            "befAcqSharesPer": "0.52",
            "afterAcqSharesPer": "0.53",
            "acqfromDt": "10-Mar-2026",
            "acqtoDt": "11-Mar-2026",
            "intimDt": "13-Mar-2026",
            "acqMode": "Market Purchase",
            "exchange": "NSE",
            "remarks": "-",
        }
    ]
    result = _normalize_insider_rows(rows)
    row = result.iloc[0]
    assert row["symbol"] == "ABC"
    assert row["buy_value"] == 397440.0
    assert row["net_value"] == 397440.0
    assert bool(row["is_promoter_group_or_promoter"])
    assert bool(row["is_buy_transaction"])


def test_normalize_bulk_block_rows_keeps_deal_type_and_value() -> None:
    rows = [
        {
            "BD_DT_DATE": "02-MAR-2026",
            "BD_SYMBOL": "AGIIL",
            "BD_SCRIP_NAME": "Agi Infra Limited",
            "BD_CLIENT_NAME": "ABC CAPITAL",
            "BD_BUY_SELL": "BUY",
            "BD_QTY_TRD": 100,
            "BD_TP_WATP": 20.5,
            "BD_REMARKS": "-",
        }
    ]
    result = _normalize_bulk_block_rows(rows, option_type="bulk_deals")
    row = result.iloc[0]
    assert row["deal_type"] == "bulk_deals"
    assert row["symbol"] == "AGIIL"
    assert row["traded_value"] == 2050.0
    assert bool(row["is_buy"])


def test_parse_daily_oi_zip_maps_core_columns() -> None:
    csv = io.StringIO()
    csv.write("Date,ISIN,Scrip Name,NSE Symbol,MWPL,NCL Open Interest,NCL FutEq OI\n")
    csv.write("27-Mar-2026,INE123,ABC LTD,ABC,1000,250,200\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ncloi_27032026.csv", csv.getvalue())
    result = _parse_daily_zip(buffer.getvalue(), pd.Timestamp("2026-03-27").date())
    row = result.iloc[0]
    assert row["symbol"] == "ABC"
    assert row["mwpl"] == 1000.0
    assert row["ncl_open_interest"] == 250.0
    assert row["oi_share_of_mwpl"] == 0.25
