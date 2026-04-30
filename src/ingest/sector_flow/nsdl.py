from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ingest.nse.session import build_session
from src.utils.io import write_json, write_parquet

NSDL_SELECTION_URL = "https://pilot.fpi.nsdl.co.in/Reports/FPI_Fortnightly_Selection.aspx"
REPORT_URL_PATTERN = re.compile(
    r"StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/FIIInvestSector_[A-Za-z0-9]+\.html"
)


@dataclass(frozen=True)
class NsdlSectorFlowFetchConfig:
    output_dir: Path
    limit: int | None = None
    delay_seconds: float = 0.1


def load_sector_flow_from_nsdl(config: NsdlSectorFlowFetchConfig) -> pd.DataFrame:
    session = build_session()
    as_of_date = pd.Timestamp.utcnow().date().isoformat()
    selection_response = session.get(NSDL_SELECTION_URL, timeout=30)
    selection_response.raise_for_status()
    selection_html = selection_response.text
    report_urls = extract_report_urls(selection_html)
    if config.limit is not None:
        report_urls = report_urls[: config.limit]

    write_json(report_urls, config.output_dir / "raw" / f"as_of_date={as_of_date}" / "report_urls.json")

    rows: list[pd.DataFrame] = []
    for url in report_urls:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        html = response.text
        file_name = url.rsplit("/", 1)[-1]
        raw_path = config.output_dir / "raw" / f"as_of_date={as_of_date}" / "reports" / file_name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(html, encoding="utf-8")
        parsed = parse_sector_flow_report(html, url)
        if not parsed.empty:
            rows.append(parsed)
        time.sleep(config.delay_seconds)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True).sort_values(["sector_name", "fortnight_end_date"])
    write_parquet(combined, config.output_dir / "normalized" / "sector_flow_fortnightly.parquet")
    return combined


def extract_report_urls(selection_html: str) -> list[str]:
    matches = REPORT_URL_PATTERN.findall(selection_html)
    deduped: list[str] = []
    seen: set[str] = set()
    for match in matches:
        full_url = f"https://pilot.fpi.nsdl.co.in/{match}"
        if full_url not in seen:
            seen.add(full_url)
            deduped.append(full_url)
    return deduped


def parse_sector_flow_report(html: str, report_url: str) -> pd.DataFrame:
    table = pd.read_html(io.StringIO(html), header=None)[0]
    data = _flatten_sector_table(table)
    if data.empty:
        return data

    current_auc_col = _find_column(data.columns, prefix="auc", period_position="last")
    previous_auc_col = _find_column(data.columns, prefix="auc", period_position="first")
    current_flow_col = _find_column(data.columns, prefix="net_investment", period_position="last")
    if current_auc_col is None or previous_auc_col is None or current_flow_col is None:
        return pd.DataFrame()

    period_label = current_auc_col.split("|")[1]
    fortnight_end_date = pd.to_datetime(period_label, errors="coerce")
    result = pd.DataFrame(
        {
            "sector_name": data["sector_name"],
            "fortnight_end_date": fortnight_end_date,
            "published_date": pd.NaT,
            "effective_from_date": pd.NaT,
            "fpi_investment_value": pd.to_numeric(data[current_auc_col], errors="coerce"),
            "fpi_change_abs": pd.to_numeric(data[current_flow_col], errors="coerce"),
            "previous_auc_value": pd.to_numeric(data[previous_auc_col], errors="coerce"),
            "source_report_url": report_url,
        }
    )
    result["fpi_change_pct"] = result["fpi_change_abs"] / result["previous_auc_value"]
    result["fpi_positive_flag"] = result["fpi_change_abs"] > 0
    return result.drop(columns=["previous_auc_value"])


def _flatten_sector_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.shape[0] < 5:
        return pd.DataFrame()
    header_rows = table.iloc[:4].fillna("")
    data = table.iloc[4:].copy()

    columns: list[str] = []
    for idx in range(table.shape[1]):
        if idx == 0:
            columns.append("sr_no")
            continue
        if idx == 1:
            columns.append("sector_name")
            continue

        section = str(header_rows.iloc[0, idx]).strip()
        unit = str(header_rows.iloc[1, idx]).strip()
        asset_group = str(header_rows.iloc[2, idx]).strip()
        subtype = str(header_rows.iloc[3, idx]).strip()

        section_label = section.lower()
        if section_label.startswith("auc as on"):
            period = section.replace("AUC as on ", "").strip()
            prefix = "auc"
        elif section_label.startswith("net investment"):
            period = section.replace("Net Investment ", "").strip()
            prefix = "net_investment"
        else:
            period = section
            prefix = section_label.replace(" ", "_")

        columns.append(f"{prefix}|{period}|{unit}|{asset_group}|{subtype}")

    data.columns = columns
    data = data[data["sector_name"].notna()].copy()
    data["sector_name"] = data["sector_name"].astype(str).str.strip()
    data = data[data["sector_name"].ne("")].copy()
    return data.reset_index(drop=True)


def _find_column(columns: pd.Index, *, prefix: str, period_position: str) -> str | None:
    matching = [
        column
        for column in columns
        if column.startswith(f"{prefix}|")
        and column.endswith("|IN INR Cr.|Alternative Investment Funds (AIFs)|Total")
    ]
    if not matching:
        matching = [
            column
            for column in columns
            if column.startswith(f"{prefix}|") and column.endswith("|IN INR Cr.||Total")
        ]
    if not matching:
        return None
    return matching[-1] if period_position == "last" else matching[0]
