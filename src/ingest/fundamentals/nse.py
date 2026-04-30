from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json, write_parquet

FINANCIALS_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
FINANCIALS_LISTING_URL = "https://www.nseindia.com/api/corporates-financial-results?index=equities&period=Quarterly"
FINANCIALS_DETAIL_URL = "https://www.nseindia.com/api/corporates-financial-results-data?{query}"

REVENUE_KEYS = [
    "re_net_sale",
    "re_total_income",
    "re_tot_income",
    "re_income_from_operations",
]
PAT_KEYS = [
    "re_net_profit",
    "re_con_pro_loss",
    "re_net_pro_loss_for_period",
    "re_profit_loss_period",
]
EPS_KEYS = [
    "re_diluted_eps",
    "re_dilut_eps_for_cont_dic_opr",
    "re_basic_eps_for_cont_dic_opr",
]
PBT_KEYS = [
    "re_pro_loss_bef_tax",
    "re_profit_bef_tax",
]
TAX_KEYS = [
    "re_tax",
]
CURRENT_TAX_KEYS = [
    "re_curr_tax",
]
DEFERRED_TAX_KEYS = [
    "re_deff_tax",
]
INTEREST_KEYS = [
    "re_int_new",
    "re_int_expd",
]
DEPRECIATION_KEYS = [
    "re_depr_und_exp",
    "re_dep_and_amor_exp",
]
DEBT_EQ_RATIO_KEYS = [
    "re_debt_eqt_rat",
]
DEBT_SERVICE_COVERAGE_KEYS = [
    "re_debt_ser_cov",
]
FACE_VALUE_DEBT_KEYS = [
    "re_face_value_debt",
]
PAID_DEBT_KEYS = [
    "re_paid_debt",
]
DEBT_REDEMPTION_KEYS = [
    "re_debt_rdmption",
]


@dataclass(frozen=True)
class NseFundamentalsFetchConfig:
    output_dir: Path
    symbols: set[str] | None = None
    delay_seconds: float = 0.1
    statement_scope: str = "Non-Consolidated"
    from_date: date | None = None
    to_date: date | None = None


def load_fundamentals_from_nse(config: NseFundamentalsFetchConfig) -> pd.DataFrame:
    session = build_session(warm=True, referer=FINANCIALS_REFERER)
    as_of_date = pd.Timestamp.utcnow().date().isoformat()
    if config.symbols:
        listing_rows = _fetch_symbol_history(session, config, as_of_date)
    else:
        listing_rows = get_json(session, FINANCIALS_LISTING_URL, referer=FINANCIALS_REFERER)
        write_json(listing_rows, config.output_dir / "raw" / f"as_of_date={as_of_date}" / "financial_results_listing.json")

    listing_rows = [row for row in listing_rows if str(row.get("consolidated", "")).strip() == config.statement_scope]

    normalized_rows: list[dict[str, object]] = []
    for row in listing_rows:
        detail_url = _build_financial_detail_url(row)
        detail = get_json(session, detail_url, referer=FINANCIALS_REFERER)
        seq_number = str(row.get("seqNumber", "")).strip()
        symbol = str(row.get("symbol", "")).strip().upper()
        if detail:
            write_json(
                detail,
                config.output_dir / "raw" / f"as_of_date={as_of_date}" / "financial_results_detail" / f"{symbol}_{seq_number}.json",
            )
        normalized_rows.append(_normalize_financial_row(row, detail))
        time.sleep(config.delay_seconds)

    if not normalized_rows:
        return pd.DataFrame()

    df = pd.DataFrame(normalized_rows)
    df = df.sort_values(["symbol", "fiscal_period_end", "effective_from_date"]).reset_index(drop=True)
    df = _add_growth_fields(df)
    write_parquet(df, config.output_dir / "normalized" / "stock_quarterly_fundamentals.parquet")
    return df


def _fetch_symbol_history(session, config: NseFundamentalsFetchConfig, as_of_date: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in sorted({value.strip().upper() for value in config.symbols or set() if value}):
        url = _build_financial_listing_url(symbol, config.from_date, config.to_date)
        symbol_rows = get_json(session, url, referer=FINANCIALS_REFERER)
        write_json(
            symbol_rows,
            config.output_dir / "raw" / f"as_of_date={as_of_date}" / "financial_results_listing_by_symbol" / f"{symbol}.json",
        )
        rows.extend(symbol_rows)
        time.sleep(config.delay_seconds)
    return rows


def _build_financial_listing_url(symbol: str, from_date: date | None, to_date: date | None) -> str:
    query = {"index": "equities", "period": "Quarterly", "symbol": symbol}
    if from_date:
        query["from_date"] = from_date.strftime("%d-%m-%Y")
    if to_date:
        query["to_date"] = to_date.strftime("%d-%m-%Y")
    return f"https://www.nseindia.com/api/corporates-financial-results?{urlencode(query)}"


def _build_financial_detail_url(listing_row: dict[str, object]) -> str:
    query = urlencode(
        {
            "index": "equities",
            "params": listing_row.get("params"),
            "seq_id": listing_row.get("seqNumber"),
            "industry": "-",
            "frOldNewFlag": listing_row.get("oldNewFlag", "N"),
            "ind": listing_row.get("indAs"),
            "format": listing_row.get("format", "New"),
        }
    )
    return FINANCIALS_DETAIL_URL.format(query=query)


def _normalize_financial_row(listing_row: dict[str, object], detail: dict[str, object]) -> dict[str, object]:
    result_values = detail.get("resultsData2") or {}
    revenue = _pick_first_numeric(result_values, REVENUE_KEYS)
    pat = _pick_first_numeric(result_values, PAT_KEYS)
    eps = _pick_first_numeric(result_values, EPS_KEYS)
    pbt = _pick_first_numeric(result_values, PBT_KEYS)
    tax = _pick_first_numeric(result_values, TAX_KEYS)
    current_tax = _pick_first_numeric(result_values, CURRENT_TAX_KEYS)
    deferred_tax = _pick_first_numeric(result_values, DEFERRED_TAX_KEYS)
    if tax is None:
        tax_components = [value for value in [current_tax, deferred_tax] if value is not None]
        if tax_components:
            tax = float(sum(tax_components))
    if pbt is None and pat is not None and tax is not None:
        pbt = pat + tax
    interest = _pick_first_numeric(result_values, INTEREST_KEYS)
    depreciation = _pick_first_numeric(result_values, DEPRECIATION_KEYS)
    debt_equity_ratio = _pick_first_numeric(result_values, DEBT_EQ_RATIO_KEYS)
    debt_service_coverage_ratio = _pick_first_numeric(result_values, DEBT_SERVICE_COVERAGE_KEYS)
    face_value_debt = _pick_first_numeric(result_values, FACE_VALUE_DEBT_KEYS)
    paid_debt = _pick_first_numeric(result_values, PAID_DEBT_KEYS)
    debt_redemption = _pick_first_numeric(result_values, DEBT_REDEMPTION_KEYS)

    ebit = pbt + interest if pbt is not None and interest is not None else None
    ebitda = ebit + depreciation if ebit is not None and depreciation is not None else None
    interest_coverage = ebit / interest if ebit is not None and interest not in {None, 0} else None

    filing_date = _parse_any_timestamp(detail.get("filingDate") or listing_row.get("filingDate")).normalize()
    announced_date = _parse_any_timestamp(listing_row.get("broadCastDate")).normalize()
    effective_from_date = announced_date
    if pd.isna(effective_from_date):
        effective_from_date = filing_date

    return {
        "symbol": str(listing_row.get("symbol", "")).strip().upper(),
        "statement_scope": str(listing_row.get("consolidated", "")).strip() or None,
        "seq_number": str(listing_row.get("seqNumber", "")).strip() or None,
        "fiscal_period_end": _parse_any_timestamp(detail.get("periodEndDT") or listing_row.get("toDate")).normalize(),
        "announced_date": announced_date,
        "effective_from_date": effective_from_date,
        "revenue": revenue,
        "ebitda": ebitda,
        "ebit": ebit,
        "pat": pat,
        "eps": eps,
        "debt": None,
        "cash": None,
        "net_debt": None,
        "cfo": None,
        "fcf": None,
        "roe": None,
        "roce": None,
        "interest_coverage": interest_coverage,
        "debt_equity_ratio": debt_equity_ratio,
        "debt_service_coverage_ratio": debt_service_coverage_ratio,
        "face_value_debt": face_value_debt,
        "paid_debt": paid_debt,
        "debt_redemption": debt_redemption,
        "debt_data_unavailable_flag": True,
    }


def _add_growth_fields(df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["symbol"]
    if "statement_scope" in df.columns:
        group_keys.append("statement_scope")
    grouped = df.groupby(group_keys, group_keys=False)
    df["revenue_yoy"] = grouped["revenue"].transform(_yoy_growth_series)
    df["pat_yoy"] = grouped["pat"].transform(_yoy_growth_series)
    df["eps_yoy"] = grouped["eps"].transform(_yoy_growth_series)
    df["ebitda_yoy"] = grouped["ebitda"].transform(_yoy_growth_series)
    df["revenue_yoy_acceleration"] = grouped["revenue_yoy"].transform(lambda s: s - s.shift(1))
    df["pat_yoy_acceleration"] = grouped["pat_yoy"].transform(lambda s: s - s.shift(1))
    df["eps_yoy_acceleration"] = grouped["eps_yoy"].transform(lambda s: s - s.shift(1))
    df["ebitda_yoy_acceleration"] = grouped["ebitda_yoy"].transform(lambda s: s - s.shift(1))
    df["revenue_yoy_positive_flag"] = _positive_flag(df["revenue_yoy"])
    df["pat_yoy_positive_flag"] = _positive_flag(df["pat_yoy"])
    df["eps_yoy_positive_flag"] = _positive_flag(df["eps_yoy"])
    df["revenue_yoy_acceleration_positive_flag"] = _positive_flag(df["revenue_yoy_acceleration"])
    df["pat_yoy_acceleration_positive_flag"] = _positive_flag(df["pat_yoy_acceleration"])
    df["revenue_ttm"] = grouped["revenue"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    df["pat_ttm"] = grouped["pat"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    df["eps_ttm"] = grouped["eps"].transform(lambda s: s.rolling(4, min_periods=4).sum())
    df["revenue_cagr_3y"] = grouped["revenue_ttm"].transform(lambda s: _cagr_from_shifted_series(s, 12, 3))
    df["revenue_cagr_5y"] = grouped["revenue_ttm"].transform(lambda s: _cagr_from_shifted_series(s, 20, 5))
    df["pat_cagr_3y"] = grouped["pat_ttm"].transform(lambda s: _cagr_from_shifted_series(s, 12, 3))
    df["pat_cagr_5y"] = grouped["pat_ttm"].transform(lambda s: _cagr_from_shifted_series(s, 20, 5))
    df["ebitda_positive_last_5q_flag"] = grouped["ebitda"].transform(
        lambda s: s.rolling(5, min_periods=5).apply(_all_positive, raw=False)
    )
    df["ebitda_positive_last_5q_flag"] = df["ebitda_positive_last_5q_flag"].map({1.0: True, 0.0: False})
    return df.drop(columns=["revenue_ttm", "pat_ttm"])


def select_preferred_statement_scope(
    df: pd.DataFrame,
    *,
    scope_preference: tuple[str, ...] = ("Non-Consolidated", "Consolidated"),
) -> pd.DataFrame:
    if df.empty or "statement_scope" not in df.columns:
        return df.copy()

    working = df.copy()
    scope_rank = {scope: rank for rank, scope in enumerate(scope_preference)}
    working["_scope_rank"] = working["statement_scope"].map(scope_rank).fillna(len(scope_rank))

    coverage = (
        working.groupby(["symbol", "statement_scope", "_scope_rank"], as_index=False)
        .agg(
            row_count=("fiscal_period_end", "size"),
            revenue_rows=("revenue", lambda s: int(pd.Series(s).notna().sum())),
            pat_rows=("pat", lambda s: int(pd.Series(s).notna().sum())),
            latest_period=("fiscal_period_end", "max"),
        )
        .sort_values(
            ["symbol", "row_count", "revenue_rows", "pat_rows", "latest_period", "_scope_rank"],
            ascending=[True, False, False, False, False, True],
        )
        .drop_duplicates(subset=["symbol"], keep="first")
    )
    preferred = working.merge(
        coverage[["symbol", "statement_scope"]],
        on=["symbol", "statement_scope"],
        how="inner",
    )
    return preferred.drop(columns=["_scope_rank"]).sort_values(["symbol", "fiscal_period_end", "effective_from_date"]).reset_index(drop=True)


def _cagr_from_shifted_series(series: pd.Series, periods_back: int, years: int) -> pd.Series:
    base = series.shift(periods_back)
    valid = series.notna() & base.notna() & (series > 0) & (base > 0)
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result.loc[valid] = (series.loc[valid] / base.loc[valid]) ** (1 / years) - 1
    return result


def _all_positive(values: pd.Series) -> float:
    return float(bool(values.notna().all() and (values > 0).all()))


def _yoy_growth_series(series: pd.Series) -> pd.Series:
    base = series.shift(4)
    current = pd.to_numeric(series, errors="coerce")
    previous = pd.to_numeric(base, errors="coerce")
    valid = current.notna() & previous.notna() & previous.ne(0)
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result.loc[valid] = (current.loc[valid] - previous.loc[valid]) / previous.loc[valid].abs()
    return result


def _positive_flag(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    valid = values.notna()
    result.loc[valid] = values.loc[valid] > 0
    return result


def _pick_first_numeric(values: dict[str, object], keys: list[str]) -> float | None:
    for key in keys:
        parsed = _to_number(values.get(key))
        if parsed is not None:
            return parsed
    return None


def _to_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_numeric(str(value).replace(",", "").strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def _parse_any_timestamp(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    return pd.to_datetime(str(value).strip(), errors="coerce", dayfirst=True)
