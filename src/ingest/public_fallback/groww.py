from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils.io import write_json
from src.utils.io import write_parquet

GROWW_SEARCH_URL = "https://groww.in/v1/api/search/v3/query/global/st_query"
GROWW_STOCK_URL_TEMPLATE = "https://groww.in/stocks/{search_id}"
GROWW_FINANCIAL_URL_TEMPLATE = "https://groww.in/stocks/{search_id}/company-financial"
GROWW_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/json",
}
NEXT_DATA_PATTERN = re.compile(r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>')


@dataclass(frozen=True)
class GrowwFallbackConfig:
    output_dir: Path
    delay_seconds: float = 0.15
    financial_detail_symbols: set[str] | None = None
    max_workers: int = 8


def build_groww_fallback_snapshot(
    symbol_frame: pd.DataFrame,
    *,
    config: GrowwFallbackConfig,
) -> pd.DataFrame:
    if symbol_frame.empty:
        return pd.DataFrame(columns=_fallback_columns())
    working = symbol_frame.copy()
    working["symbol"] = working["symbol"].astype(str).str.upper().str.strip()
    run_date = pd.Timestamp.utcnow().date().isoformat()
    rows: list[dict[str, object]] = []
    financial_detail_symbols = {value.upper() for value in (config.financial_detail_symbols or set())}
    records = []
    for record in working[["symbol", "company_name"]].drop_duplicates().itertuples(index=False):
        symbol = str(record.symbol).strip().upper()
        company_name = str(record.company_name).strip() if not pd.isna(record.company_name) else ""
        records.append(
            {
                "symbol": symbol,
                "company_name": company_name,
                "fetch_financial_detail": symbol in financial_detail_symbols,
            }
        )

    max_workers = max(1, int(config.max_workers))
    if max_workers == 1:
        for record in records:
            rows.append(
                _fetch_symbol_snapshot_with_session(
                    symbol=record["symbol"],
                    company_name=record["company_name"],
                    output_dir=config.output_dir,
                    run_date=run_date,
                    fetch_financial_detail=bool(record["fetch_financial_detail"]),
                )
            )
            time.sleep(config.delay_seconds)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _fetch_symbol_snapshot_with_session,
                    symbol=record["symbol"],
                    company_name=record["company_name"],
                    output_dir=config.output_dir,
                    run_date=run_date,
                    fetch_financial_detail=bool(record["fetch_financial_detail"]),
                    delay_seconds=config.delay_seconds,
                )
                for record in records
            ]
            for future in as_completed(futures):
                rows.append(future.result())

    result = pd.DataFrame(rows, columns=_fallback_columns())
    write_parquet(result, config.output_dir / "normalized" / "groww_fallback_snapshot.parquet")
    return result


def _fetch_symbol_snapshot_with_session(
    *,
    symbol: str,
    company_name: str,
    output_dir: Path,
    run_date: str,
    fetch_financial_detail: bool,
    delay_seconds: float = 0.0,
) -> dict[str, object]:
    session = requests.Session()
    session.headers.update(GROWW_HEADERS)
    try:
        row = _fetch_symbol_snapshot(
            session,
            symbol=symbol,
            company_name=company_name,
            output_dir=output_dir,
            run_date=run_date,
            fetch_financial_detail=fetch_financial_detail,
        )
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        return row
    finally:
        session.close()


def _fetch_symbol_snapshot(
    session: requests.Session,
    *,
    symbol: str,
    company_name: str,
    output_dir: Path,
    run_date: str,
    fetch_financial_detail: bool,
) -> dict[str, object]:
    base_row = {column: pd.NA for column in _fallback_columns()}
    base_row["symbol"] = symbol
    try:
        search_payload = _search_symbol(session, symbol=symbol, company_name=company_name)
        write_json(
            search_payload,
            output_dir / "raw" / f"as_of_date={run_date}" / "search" / f"{symbol}.json",
        )
        match = _pick_search_match(search_payload, symbol=symbol)
        if not match:
            base_row["groww_error"] = "search_match_not_found"
            return base_row

        search_id = str(match.get("search_id") or match.get("id") or "").strip()
        if not search_id:
            base_row["groww_error"] = "missing_search_id"
            return base_row

        stock_url = GROWW_STOCK_URL_TEMPLATE.format(search_id=search_id)
        stock_page = _fetch_text(session, stock_url)
        stock_data = _parse_stock_page(stock_page)
        stock_row = _extract_stock_row(symbol=symbol, search_id=search_id, stock_data=stock_data, stock_url=stock_url)
        base_row.update(stock_row)

        if fetch_financial_detail:
            financial_url = GROWW_FINANCIAL_URL_TEMPLATE.format(search_id=search_id)
            financial_page = _fetch_text(session, financial_url)
            financial_data = _parse_financial_page(financial_page)
            base_row.update(_extract_financial_row(financial_data, financial_url=financial_url))
        return base_row
    except Exception as exc:  # noqa: BLE001
        base_row["groww_error"] = f"{type(exc).__name__}: {exc}"
        return base_row


def _search_symbol(session: requests.Session, *, symbol: str, company_name: str) -> dict[str, Any]:
    queries = [symbol]
    if company_name:
        queries.append(company_name)
    last_payload: dict[str, Any] = {}
    for query in queries:
        response = session.get(
            GROWW_SEARCH_URL,
            params={"query": query, "from": 0, "size": 10, "web": "true", "entity_type": "stocks"},
            headers={"Referer": f"https://groww.in/search?q={query.replace(' ', '%20')}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        if _pick_search_match(payload, symbol=symbol):
            return payload
    return last_payload


def _pick_search_match(payload: dict[str, Any], *, symbol: str) -> dict[str, Any] | None:
    content = (((payload or {}).get("data") or {}).get("content") or [])
    wanted = symbol.strip().upper()
    exact = [row for row in content if str(row.get("nse_scrip_code", "")).strip().upper() == wanted]
    if exact:
        return exact[0]
    return content[0] if content else None


def _fetch_text(session: requests.Session, url: str) -> str:
    response = session.get(url, headers={"Referer": url}, timeout=30)
    response.raise_for_status()
    return response.text


def _parse_stock_page(html: str) -> dict[str, Any]:
    data = _extract_next_data_payload(html)
    return (((data.get("props") or {}).get("pageProps") or {}).get("stockData") or {})


def _parse_financial_page(html: str) -> dict[str, Any]:
    data = _extract_next_data_payload(html)
    return (((data.get("props") or {}).get("pageProps") or {}).get("stockFinancialData") or {})


def _extract_next_data_payload(html: str) -> dict[str, Any]:
    match = NEXT_DATA_PATTERN.search(html)
    if not match:
        raise ValueError("next_data_missing")
    return json.loads(match.group(1))


def _extract_stock_row(*, symbol: str, search_id: str, stock_data: dict[str, Any], stock_url: str) -> dict[str, object]:
    header = stock_data.get("header") or {}
    stats = stock_data.get("stats") or {}
    shareholding = stock_data.get("shareHoldingPattern") or {}
    latest_quarter_label, latest_quarter = _latest_shareholding_block(shareholding)
    promoter_pct = _sum_dict_percents((latest_quarter.get("promoters") or {}))
    mf_pct = _nested_percent(latest_quarter, ["mutualFunds"])
    odi = latest_quarter.get("otherDomesticInstitutions") or {}
    dii_pct = _sum_values(
        [
            mf_pct,
            _nested_percent(odi, ["insurance"]),
            _nested_percent(odi, ["banks"]),
            _nested_percent(odi, ["otherFirms"]),
            _nested_percent(odi, ["financialInstitutions"]),
        ]
    )
    groww_scope, annuals = _pick_financial_statement_scope(stock_data.get("financialStatementV2") or {})
    revenue_cagr = _cagr_from_year_dict(annuals.get("Revenue"))
    pat_cagr = _cagr_from_year_dict(annuals.get("Profit"))
    debt_to_equity = _to_float(stats.get("debtToEquity"))
    return {
        "symbol": symbol,
        "groww_search_id": search_id,
        "groww_company_name": header.get("displayName") or header.get("shortName"),
        "groww_source_url": stock_url,
        "groww_market_cap_cr": _to_float(stats.get("marketCap")),
        "groww_pe_ttm": _to_float(stats.get("peRatio")),
        "groww_debt_to_equity": debt_to_equity,
        "groww_debt_free_proxy_flag": _debt_free_from_ratio(debt_to_equity),
        "groww_promoter_pct": promoter_pct,
        "groww_fii_fpi_pct": _nested_percent(latest_quarter, ["foreignInstitutions"]),
        "groww_dii_pct": dii_pct,
        "groww_mf_pct": mf_pct,
        "groww_revenue_cagr_5y_proxy": revenue_cagr,
        "groww_pat_cagr_5y_proxy": pat_cagr,
        "groww_scope_used": groww_scope,
        "groww_shareholding_quarter": latest_quarter_label,
        "groww_source_compromise_notes": _source_note(revenue_cagr, pat_cagr),
    }


def _extract_financial_row(financial_data: dict[str, Any], *, financial_url: str) -> dict[str, object]:
    statements = financial_data.get("statements") or []
    income_statement = next((row for row in statements if row.get("title") == "Income Statement"), {})
    consolidated_q = ((income_statement.get("consolidatedQuarterly") or {}).get("financial") or [])
    standalone_q = ((income_statement.get("standaloneQuarterly") or {}).get("financial") or [])
    ebitda_values = _pick_line_values(consolidated_q, "EBITDA") or _pick_line_values(standalone_q, "EBITDA")
    return {
        "groww_financial_url": financial_url,
        "groww_ebitda_positive_last_5q_flag": _all_positive(ebitda_values, required=5),
    }


def _pick_financial_statement_scope(financial_statement_v2: dict[str, Any]) -> tuple[str | None, dict[str, dict[str, Any]]]:
    for scope in ("CONSOLIDATED", "STANDALONE"):
        rows = financial_statement_v2.get(scope) or []
        if rows:
            return scope.title(), {str(row.get("title")): row.get("yearly") or {} for row in rows}
    return None, {}


def _latest_shareholding_block(shareholding: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    if not shareholding:
        return None, {}
    dated = [(label, _parse_groww_quarter_label(label)) for label in shareholding]
    dated = [row for row in dated if not pd.isna(row[1])]
    if not dated:
        label = list(shareholding)[-1]
        return label, shareholding.get(label) or {}
    latest_label = max(dated, key=lambda row: row[1])[0]
    return latest_label, shareholding.get(latest_label) or {}


def _parse_groww_quarter_label(label: str) -> pd.Timestamp:
    return pd.to_datetime(label.replace("'", "20"), format="%b %Y", errors="coerce")


def _pick_line_values(rows: list[dict[str, Any]], title: str) -> list[float]:
    for row in rows:
        if str(row.get("title", "")).strip().lower() == title.strip().lower():
            return [_to_float(value) for value in row.get("value") or []]
    return []


def _cagr_from_year_dict(year_map: dict[str, Any] | None) -> float | pd.NA:
    if not year_map:
        return pd.NA
    cleaned: list[tuple[int, float]] = []
    for year_label, value in year_map.items():
        year = pd.to_numeric(year_label, errors="coerce")
        numeric = _to_float(value)
        if pd.isna(year) or pd.isna(numeric):
            continue
        cleaned.append((int(year), float(numeric)))
    cleaned = sorted(cleaned)
    if len(cleaned) < 5:
        return pd.NA
    start_year, start_value = cleaned[0]
    end_year, end_value = cleaned[-1]
    year_span = end_year - start_year
    if start_value <= 0 or end_value <= 0 or year_span <= 0:
        return pd.NA
    return (end_value / start_value) ** (1 / year_span) - 1


def _all_positive(values: list[float], *, required: int) -> bool | pd.NA:
    cleaned = [value for value in values if not pd.isna(value)]
    if len(cleaned) < required:
        return pd.NA
    return all(value > 0 for value in cleaned[-required:])


def _nested_percent(container: dict[str, Any], path: list[str]) -> float | pd.NA:
    current: Any = container
    for key in path:
        if not isinstance(current, dict):
            return pd.NA
        current = current.get(key)
    if isinstance(current, dict):
        return _to_float(current.get("percent"))
    return _to_float(current)


def _sum_dict_percents(container: dict[str, Any]) -> float | pd.NA:
    values = []
    for value in container.values():
        if isinstance(value, dict):
            numeric = _to_float(value.get("percent"))
        else:
            numeric = _to_float(value)
        if not pd.isna(numeric):
            values.append(float(numeric))
    return _sum_values(values)


def _sum_values(values: list[float | pd.NA]) -> float | pd.NA:
    cleaned = [float(value) for value in values if not pd.isna(value)]
    if not cleaned:
        return pd.NA
    return float(sum(cleaned))


def _debt_free_from_ratio(debt_to_equity: float | pd.NA) -> bool | pd.NA:
    if pd.isna(debt_to_equity):
        return pd.NA
    return bool(float(debt_to_equity) <= 0.01)


def _source_note(revenue_cagr: float | pd.NA, pat_cagr: float | pd.NA) -> str | pd.NA:
    notes: list[str] = []
    if not pd.isna(revenue_cagr) or not pd.isna(pat_cagr):
        notes.append("Groww annual-display CAGR proxy based on latest 5 reported fiscal years.")
    return " ".join(notes) if notes else pd.NA


def _to_float(value: Any) -> float | pd.NA:
    if value in (None, "", "-", "--"):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).replace(",", "").replace("%", "").replace("\u20b9", "").replace("Cr", "").strip()
        if not text:
            return pd.NA
        try:
            return float(text)
        except ValueError:
            return pd.NA


def _fallback_columns() -> list[str]:
    return [
        "symbol",
        "groww_search_id",
        "groww_company_name",
        "groww_source_url",
        "groww_financial_url",
        "groww_market_cap_cr",
        "groww_pe_ttm",
        "groww_debt_to_equity",
        "groww_debt_free_proxy_flag",
        "groww_promoter_pct",
        "groww_fii_fpi_pct",
        "groww_dii_pct",
        "groww_mf_pct",
        "groww_revenue_cagr_5y_proxy",
        "groww_pat_cagr_5y_proxy",
        "groww_ebitda_positive_last_5q_flag",
        "groww_scope_used",
        "groww_shareholding_quarter",
        "groww_source_compromise_notes",
        "groww_error",
    ]
