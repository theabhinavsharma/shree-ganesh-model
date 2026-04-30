from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.ingest.nse.normalize import normalize_bhavcopy_csv, normalize_delivery_file, read_bhavcopy_csv_text


def test_bhavcopy_text_parser_reads_csv() -> None:
    raw_text = (
        "SYMBOL,SERIES,DATE1,PREV_CLOSE,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,LAST_PRICE,CLOSE_PRICE,AVG_PRICE,"
        "TTL_TRD_QNTY,TURNOVER_LACS,NO_OF_TRADES,DELIV_QTY,DELIV_PER\n"
        "ABC,EQ,01-Jan-2024,1.1,1,2,0.5,1.7,1.5,1.4,100,1.5,10,75,75\n"
    )
    df = read_bhavcopy_csv_text(raw_text)
    assert list(df.columns) == [
        "SYMBOL",
        "SERIES",
        "DATE1",
        "PREV_CLOSE",
        "OPEN_PRICE",
        "HIGH_PRICE",
        "LOW_PRICE",
        "LAST_PRICE",
        "CLOSE_PRICE",
        "AVG_PRICE",
        "TTL_TRD_QNTY",
        "TURNOVER_LACS",
        "NO_OF_TRADES",
        "DELIV_QTY",
        "DELIV_PER",
    ]


def test_normalize_bhavcopy_csv_maps_fields(tmp_path: Path) -> None:
    path = tmp_path / "sec_bhavdata_full_01012024.csv"
    path.write_text(
        " SYMBOL , SERIES , DATE1 , PREV_CLOSE , OPEN_PRICE , HIGH_PRICE , LOW_PRICE , LAST_PRICE , CLOSE_PRICE , AVG_PRICE , TTL_TRD_QNTY , TURNOVER_LACS , NO_OF_TRADES , DELIV_QTY , DELIV_PER \n"
        " ABC , EQ , 01-Jan-2024 , 1.1 , 1 , 2 , 0.5 , 1.7 , 1.5 , 1.4 , 100 , 1.5 , 10 , 75 , 75 \n",
        encoding="utf-8",
    )
    result = normalize_bhavcopy_csv(path, date(2024, 1, 1), "https://example.com")
    row = result.iloc[0]
    assert row["symbol"] == "ABC"
    assert row["series"] == "EQ"
    assert row["close"] == 1.5
    assert row["deliverable_qty"] == 75
    assert row["delivery_pct"] == 0.75
    assert row["total_traded_value"] == 150000.0
    assert bool(row["verified_price_flag"]) is True


def test_normalize_legacy_bhavcopy_zip_with_delivery_file(tmp_path: Path) -> None:
    market_path = tmp_path / "cm02JAN2015bhav.csv.zip"
    with ZipFile(market_path, "w") as archive:
        archive.writestr(
            "cm02JAN2015bhav.csv",
            "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
            "ABC,EQ,10,12,9,11,11,9.5,1000,125000.5,02-JAN-2015,125,INE000A01000,\n",
        )
    delivery_path = tmp_path / "MTO_02012015.DAT"
    delivery_path.write_text(
        "Security Wise Delivery Position - Compulsory Rolling Settlement\n"
        "10,MTO,02012015,325859130,0001536\n"
        "20,SETTLEMENT,2015001,N,1\n"
        "Record Type,Sr No,Name of Security,Type,Quantity Traded,Deliverable Quantity,Percent of Deliverable Quantity to Traded Quantity\n"
        "20,1,ABC,EQ,1000,650,65.00\n",
        encoding="utf-8",
    )
    result = normalize_bhavcopy_csv(
        market_path,
        date(2015, 1, 2),
        "https://example.com/market",
        delivery_path=delivery_path,
        delivery_source_url="https://example.com/delivery",
    )
    row = result.iloc[0]
    assert row["symbol"] == "ABC"
    assert row["close"] == 11.0
    assert row["total_traded_value"] == 125000.5
    assert row["deliverable_qty"] == 650
    assert row["delivery_pct"] == 0.65
    assert row["delivery_raw_file_name"] == "MTO_02012015.DAT"
    assert row["delivery_source_url"] == "https://example.com/delivery"


def test_normalize_delivery_file_parses_mto_records(tmp_path: Path) -> None:
    delivery_path = tmp_path / "MTO_02012015.DAT"
    delivery_path.write_text(
        "Security Wise Delivery Position - Compulsory Rolling Settlement\n"
        "10,MTO,02012015,325859130,0001536\n"
        "20,SETTLEMENT,2015001,N,1\n"
        "Record Type,Sr No,Name of Security,Type,Quantity Traded,Deliverable Quantity,Percent of Deliverable Quantity to Traded Quantity\n"
        "20,1,ABC,EQ,1000,650,65.00\n"
        "20,2,XYZ,BE,500,125,25.00\n",
        encoding="utf-8",
    )
    result = normalize_delivery_file(delivery_path)
    assert list(result["symbol"]) == ["ABC", "XYZ"]
    assert list(result["deliverable_qty"]) == [650, 125]
    assert list(result["delivery_pct"]) == [0.65, 0.25]
