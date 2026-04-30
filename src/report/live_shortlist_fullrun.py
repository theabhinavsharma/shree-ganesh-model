from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from src.features.indicators import add_daily_price_features
from src.ingest.nse.fetch_bhavcopy import build_nse_bhavcopy_url
from src.ingest.nse.fetch_bhavcopy import build_nse_delivery_url
from src.ingest.nse.normalize import normalize_trade_date_directory
from src.ingest.nse.quote_snapshot import build_quote_snapshot_from_symbols
from src.ingest.nse.quote_snapshot import _normalize_quote_snapshot_row
from src.ingest.public_fallback.groww import build_groww_fallback_snapshot
from src.ingest.public_fallback.groww import GrowwFallbackConfig
from src.screen.build_universe import apply_screen_filters
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme
from src.utils.io import write_parquet

RULES: list[tuple[str, str]] = [
    ("filter_market_cap", "Market cap >= {mcap} Cr"),
    ("filter_debt", "Debt-free"),
    ("filter_revenue_growth", "Revenue CAGR 5Y >= 10%"),
    ("filter_profit_cagr", "PAT CAGR 5Y >= 20%"),
    ("filter_ebitda_positive", "EBITDA positive last 5 quarters"),
    ("filter_volume_expansion", "Volume >= 1.5x 20-day average"),
    ("filter_volume_high_3m", "Volume at 3-month high"),
    ("filter_delivery_expansion", "Delivery % at 3-month high"),
    ("filter_rsi_daily", "Daily RSI > 60"),
    ("filter_rsi_weekly", "Weekly RSI > 60"),
    ("filter_rsi_monthly", "Monthly RSI > 60"),
    ("filter_pe", "PE < 30"),
    ("filter_promoter_holding", "Promoter holding >= 50%"),
    ("filter_above_50_dma", "Above 50 DMA"),
    ("filter_above_200_dma", "Above 200 DMA"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--raw-dir", default="data/raw/nse_full_history_official")
    parser.add_argument("--derived-daily-facts-path", default="data/derived/daily_facts_official_2015_2026.parquet")
    parser.add_argument("--fundamentals-path", default="data/fundamentals_full_history/normalized/stock_quarterly_fundamentals_all_scopes.parquet")
    parser.add_argument("--shareholding-path", default="data/shareholding_full_history/normalized/stock_shareholding_quarterly.parquet")
    parser.add_argument("--quote-snapshot-path", default="")
    parser.add_argument("--quote-delay-seconds", type=float, default=0.02)
    parser.add_argument("--groww-delay-seconds", type=float, default=0.05)
    parser.add_argument("--groww-max-workers", type=int, default=8)
    parser.add_argument("--groww-retry-delay-seconds", type=float, default=0.25)
    parser.add_argument("--groww-retry-workers", type=int, default=1)
    parser.add_argument("--groww-retry-sleep-seconds", type=float, default=8.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of_date = _resolve_as_of_date(raw_dir=Path(args.raw_dir), requested_date=args.as_of_date or None)

    daily_slice = _build_current_daily_slice(
        derived_daily_facts_path=Path(args.derived_daily_facts_path),
        raw_dir=Path(args.raw_dir),
        as_of_date=as_of_date,
    )
    quote_snapshot = _load_or_build_quote_snapshot(
        symbols=daily_slice["symbol"].astype(str).tolist(),
        output_dir=output_dir / "quote_snapshot",
        quote_snapshot_path=Path(args.quote_snapshot_path) if args.quote_snapshot_path else None,
        delay_seconds=args.quote_delay_seconds,
    )
    universe = daily_slice.merge(quote_snapshot, on="symbol", how="left")
    universe["market_cap_cr"] = (
        pd.to_numeric(universe["close"], errors="coerce") * pd.to_numeric(universe["issued_size"], errors="coerce") / 10_000_000
    ).round(2)
    if "quote_pe_ttm" in universe.columns:
        quote_pe = pd.to_numeric(universe["quote_pe_ttm"], errors="coerce")
        existing_pe = (
            pd.to_numeric(universe["pe_ttm"], errors="coerce")
            if "pe_ttm" in universe.columns
            else pd.Series(float("nan"), index=universe.index, dtype="float64")
        )
        universe["pe_ttm"] = existing_pe.where(existing_pe.notna(), quote_pe)

    universe = universe[universe["instrument_type"].fillna("Equity").astype(str).str.casefold().eq("equity")].copy()
    universe = universe.reset_index(drop=True)

    fundamentals = _build_official_fundamentals_snapshot(Path(args.fundamentals_path), as_of_date=as_of_date)
    shareholding = _build_official_shareholding_snapshot(Path(args.shareholding_path), as_of_date=as_of_date)
    universe = universe.merge(fundamentals, on="symbol", how="left")
    universe = universe.merge(shareholding, on="symbol", how="left", suffixes=("", "_share"))

    missing_mask = (
        universe["pe_ttm"].isna()
        | universe["promoter_pct"].isna()
        | universe["revenue_cagr_5y"].isna()
        | universe["pat_cagr_5y"].isna()
        | universe["ebitda_positive_last_5q_flag"].isna()
        | _compute_debt_flag(universe).isna()
    )
    groww_candidates = universe.loc[missing_mask, ["symbol", "company_name"]].drop_duplicates()
    ebitda_missing_symbols = set(
        universe.loc[universe["ebitda_positive_last_5q_flag"].isna(), "symbol"].dropna().astype(str).str.upper()
    )
    groww_fallback = build_groww_fallback_snapshot(
        groww_candidates,
        config=GrowwFallbackConfig(
            output_dir=output_dir / "groww_fallback",
            delay_seconds=args.groww_delay_seconds,
            financial_detail_symbols=ebitda_missing_symbols,
            max_workers=args.groww_max_workers,
        ),
    )
    universe = universe.merge(groww_fallback, on="symbol", how="left")
    universe = _apply_groww_fallback(universe)

    groww_retry_candidates = _groww_retry_candidates(universe)
    if not groww_retry_candidates.empty:
        time.sleep(max(0.0, float(args.groww_retry_sleep_seconds)))
        groww_retry = build_groww_fallback_snapshot(
            groww_retry_candidates,
            config=GrowwFallbackConfig(
                output_dir=output_dir / "groww_fallback_retry",
                delay_seconds=args.groww_retry_delay_seconds,
                financial_detail_symbols=set(
                    groww_retry_candidates.loc[
                        groww_retry_candidates["symbol"].isin(ebitda_missing_symbols),
                        "symbol",
                    ].astype(str).str.upper()
                ),
                max_workers=args.groww_retry_workers,
            ),
        )
        universe = universe.drop(columns=[column for column in groww_fallback.columns if column != "symbol"], errors="ignore")
        combined_groww = _combine_fallback_frames(groww_fallback, groww_retry)
        universe = universe.merge(combined_groww, on="symbol", how="left")
    else:
        universe = universe.drop(columns=[column for column in groww_fallback.columns if column != "symbol"], errors="ignore")
        universe = universe.merge(groww_fallback, on="symbol", how="left")
    universe = _apply_groww_fallback(universe)
    universe["debt_source_compromise_flag"] = universe["debt_equity_ratio_source"].eq("Groww")
    universe["revenue_growth_source_compromise_flag"] = universe["revenue_cagr_5y_source"].eq("Groww")
    universe["profit_growth_source_compromise_flag"] = universe["pat_cagr_5y_source"].eq("Groww")
    universe["pe_source_compromise_flag"] = universe["pe_ttm_source"].eq("Groww")
    universe["promoter_source_compromise_flag"] = universe["promoter_pct_source"].eq("Groww")

    write_parquet(universe, output_dir / "current_universe_enriched.parquet")
    written_files: list[Path] = [output_dir / "current_universe_enriched.parquet"]
    write_dataframe_manifest(
        output_dir / "current_universe_enriched.parquet",
        universe,
        generated_by="src.report.live_shortlist_fullrun",
        as_of_date=as_of_date.date().isoformat(),
        extra_notes=[
            "This file is the enriched live universe used by the screen run in this folder.",
            "Fields may come from official NSE sources first and Groww fallback second where explicitly noted by source columns.",
        ],
    )

    summary = {
        "as_of_trade_date": as_of_date.date().isoformat(),
        "base_equity_universe_count": int(len(universe)),
        "results": {},
        "notes": [
            "Market cap and PE prefer official NSE quote metadata, then Groww fallback.",
            "Revenue CAGR and PAT CAGR use official filing history first, then Groww annual-display CAGR proxy when official history is blank.",
            "Debt-free uses official debt/debt-equity first, then Groww debt-to-equity proxy when official debt fields are blank.",
            "Delivery 3-month-high remains unavailable for genuine insufficient lookback; it is not backfilled from unofficial sources.",
        ],
    }
    for mcap in (1000, 5000):
        result = _run_variant(universe, market_cap_threshold=float(mcap), output_dir=output_dir)
        summary["results"][str(mcap)] = result

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written_files.append(summary_path)
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src.report.live_shortlist_fullrun",
        as_of_date=as_of_date.date().isoformat(),
        extra_notes=[
            "Open this file first to understand the run outcome before opening the CSV details.",
        ],
    )
    written_files.extend(sorted(path for path in output_dir.glob("*.csv")))
    write_report_directory_readme(
        output_dir,
        title=f"Live Screen Output For {as_of_date.date().isoformat()}",
        intro_lines=[
            f"This folder contains the live screen output built as of trade date `{as_of_date.date().isoformat()}`.",
            "Every tabular artifact has a matching `.manifest.json` sidecar with plain-English explanations, null counts, and sample values.",
        ],
        files=written_files,
    )


def _resolve_as_of_date(*, raw_dir: Path, requested_date: str | None) -> pd.Timestamp:
    available = sorted(
        pd.Timestamp(path.name.split("=")[1]).normalize()
        for path in raw_dir.glob("trade_date=*")
        if path.is_dir()
    )
    if not available:
        raise FileNotFoundError(f"No trade directories found under {raw_dir}")
    if requested_date:
        wanted = pd.Timestamp(requested_date).normalize()
        eligible = [value for value in available if value <= wanted]
        if not eligible:
            raise ValueError(f"No raw trade data on or before {wanted.date().isoformat()}")
        return eligible[-1]
    return available[-1]


def _build_current_daily_slice(*, derived_daily_facts_path: Path, raw_dir: Path, as_of_date: pd.Timestamp) -> pd.DataFrame:
    history_start = as_of_date - pd.Timedelta(days=500)
    derived = pd.read_parquet(derived_daily_facts_path)
    derived["trade_date"] = pd.to_datetime(derived["trade_date"]).dt.normalize()
    latest_derived_date = derived["trade_date"].max()
    base = derived.loc[derived["trade_date"].between(history_start, min(latest_derived_date, as_of_date))].copy()
    raw_rows: list[pd.DataFrame] = []
    if as_of_date > latest_derived_date:
        for trade_dir in sorted(path for path in raw_dir.glob("trade_date=*") if path.is_dir()):
            trade_date = pd.Timestamp(trade_dir.name.split("=")[1]).normalize()
            if trade_date <= latest_derived_date or trade_date < history_start or trade_date > as_of_date:
                continue
            raw_rows.append(
                normalize_trade_date_directory(
                    trade_dir,
                    trade_date.date(),
                    market_source_url=build_nse_bhavcopy_url(trade_date.date()),
                    delivery_source_url=build_nse_delivery_url(trade_date.date()),
                )
            )
    base_columns = [
        "symbol",
        "series",
        "open",
        "high",
        "low",
        "close",
        "last_price",
        "prev_close",
        "total_traded_qty",
        "total_traded_value",
        "trade_date_source",
        "num_trades",
        "isin",
        "trade_date",
        "raw_file_name",
        "fetch_timestamp",
        "source_url",
        "delivery_raw_file_name",
        "delivery_source_url",
        "delivery_report_traded_qty",
        "deliverable_qty",
        "delivery_pct",
        "verified_price_flag",
        "avg_price",
        "turnover_lacs",
    ]
    base = base[[column for column in base_columns if column in base.columns]].copy()
    combined = pd.concat([base] + raw_rows, ignore_index=True)
    combined = combined.sort_values(["symbol", "trade_date"]).drop_duplicates(subset=["symbol", "trade_date"], keep="last")
    combined = combined[combined["series"].fillna("").str.upper().eq("EQ")].copy()
    featured = add_daily_price_features(combined)
    current_slice = featured.loc[featured["trade_date"].eq(as_of_date)].copy()
    if current_slice.empty:
        raise ValueError(f"No EQ rows found for {as_of_date.date().isoformat()}")
    return current_slice


def _load_or_build_quote_snapshot(
    *,
    symbols: list[str],
    output_dir: Path,
    quote_snapshot_path: Path | None,
    delay_seconds: float,
) -> pd.DataFrame:
    if quote_snapshot_path and quote_snapshot_path.exists():
        cached = pd.read_parquet(quote_snapshot_path)
        cached = _enrich_cached_quote_snapshot_from_raw(cached, quote_snapshot_path=quote_snapshot_path)
        for column in ["quote_pe_ttm", "company_name", "sector", "industry", "basic_industry", "instrument_type", "issued_size", "quote_last_price", "quote_last_update_time"]:
            if column not in cached.columns:
                cached[column] = pd.NA
        return cached
    return build_quote_snapshot_from_symbols(symbols, output_dir=output_dir, delay_seconds=delay_seconds)


def _enrich_cached_quote_snapshot_from_raw(cached: pd.DataFrame, *, quote_snapshot_path: Path) -> pd.DataFrame:
    raw_root = quote_snapshot_path.parent.parent / "raw"
    json_files = sorted(raw_root.rglob("quote/*.json"))
    if not json_files:
        return cached
    rows: list[dict[str, object]] = []
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_normalize_quote_snapshot_row(path.stem.upper(), payload))
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return cached
    merged = cached.merge(raw_df, on="symbol", how="outer", suffixes=("", "_raw"))
    for column in raw_df.columns:
        if column == "symbol":
            continue
        raw_column = f"{column}_raw"
        if raw_column not in merged.columns:
            continue
        base = merged[column] if column in merged.columns else pd.Series(pd.NA, index=merged.index, dtype="object")
        merged[column] = base.where(base.notna(), merged[raw_column])
        merged = merged.drop(columns=[raw_column])
    return merged


def _build_official_fundamentals_snapshot(path: Path, *, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    df = pd.read_parquet(path)
    df["effective_from_date"] = pd.to_datetime(df["effective_from_date"]).dt.normalize()
    df = df.loc[df["effective_from_date"].le(as_of_date)].copy()
    if df.empty:
        return pd.DataFrame(columns=["symbol"])
    df["_scope_rank"] = df["statement_scope"].map({"Non-Consolidated": 0, "Consolidated": 1}).fillna(9)
    latest_per_scope = (
        df.sort_values(["symbol", "statement_scope", "effective_from_date", "_scope_rank"], ascending=[True, True, False, True])
        .groupby(["symbol", "statement_scope"], as_index=False)
        .head(1)
        .sort_values(["symbol", "effective_from_date", "_scope_rank"], ascending=[True, False, True])
    )
    fields = [
        "revenue_cagr_5y",
        "pat_cagr_5y",
        "ebitda_positive_last_5q_flag",
        "eps_ttm",
        "debt_equity_ratio",
        "debt",
        "net_debt",
        "face_value_debt",
        "paid_debt",
        "debt_redemption",
        "interest_coverage",
    ]
    rows: list[dict[str, object]] = []
    for symbol, group in latest_per_scope.groupby("symbol", sort=False):
        row: dict[str, object] = {"symbol": symbol}
        for field in fields:
            nonnull = group.loc[group[field].notna(), field]
            row[field] = nonnull.iloc[0] if not nonnull.empty else pd.NA
            row[f"{field}_source"] = "Official" if not nonnull.empty else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def _build_official_shareholding_snapshot(path: Path, *, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    df = pd.read_parquet(path)
    df["effective_from_date"] = pd.to_datetime(df["effective_from_date"]).dt.normalize()
    df = df.loc[df["effective_from_date"].le(as_of_date)].copy()
    if df.empty:
        return pd.DataFrame(columns=["symbol"])
    latest = df.sort_values(["symbol", "effective_from_date"]).groupby("symbol", as_index=False).tail(1)
    keep = [column for column in ["symbol", "promoter_pct", "promoter_pledged_pct", "fii_fpi_pct", "dii_pct", "mf_pct"] if column in latest.columns]
    latest = latest[keep].copy()
    for column in ["promoter_pct", "promoter_pledged_pct", "fii_fpi_pct", "dii_pct", "mf_pct"]:
        if column in latest.columns:
            latest[f"{column}_source"] = "Official"
    return latest


def _apply_groww_fallback(universe: pd.DataFrame) -> pd.DataFrame:
    working = universe.copy()
    field_pairs = [
        ("market_cap_cr", "groww_market_cap_cr"),
        ("pe_ttm", "groww_pe_ttm"),
        ("debt_equity_ratio", "groww_debt_to_equity"),
        ("promoter_pct", "groww_promoter_pct"),
        ("fii_fpi_pct", "groww_fii_fpi_pct"),
        ("dii_pct", "groww_dii_pct"),
        ("mf_pct", "groww_mf_pct"),
        ("revenue_cagr_5y", "groww_revenue_cagr_5y_proxy"),
        ("pat_cagr_5y", "groww_pat_cagr_5y_proxy"),
        ("ebitda_positive_last_5q_flag", "groww_ebitda_positive_last_5q_flag"),
    ]
    for base_field, groww_field in field_pairs:
        if groww_field not in working.columns:
            continue
        base_series = working[base_field] if base_field in working.columns else pd.Series(pd.NA, index=working.index, dtype="object")
        groww_series = working[groww_field]
        working[base_field] = base_series.where(base_series.notna(), groww_series)
        source_column = f"{base_field}_source"
        source_series = working[source_column] if source_column in working.columns else pd.Series(pd.NA, index=working.index, dtype="object")
        working[source_column] = source_series.where(source_series.notna(), groww_series.notna().map({True: "Groww", False: pd.NA}))
    return working


def _groww_retry_candidates(universe: pd.DataFrame) -> pd.DataFrame:
    unresolved_mask = (
        universe["market_cap_cr"].isna()
        | universe["pe_ttm"].isna()
        | universe["promoter_pct"].isna()
        | universe["revenue_cagr_5y"].isna()
        | universe["pat_cagr_5y"].isna()
        | universe["ebitda_positive_last_5q_flag"].isna()
        | _compute_debt_flag(universe).isna()
    )
    error_mask = universe.get("groww_error", pd.Series(pd.NA, index=universe.index)).astype("string").str.contains("403", na=False)
    candidates = universe.loc[unresolved_mask & error_mask, ["symbol", "company_name"]].drop_duplicates()
    return candidates


def _combine_fallback_frames(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    if first.empty:
        return second.copy()
    if second.empty:
        return first.copy()
    combined = first.merge(second, on="symbol", how="outer", suffixes=("", "_retry"))
    for column in second.columns:
        if column == "symbol":
            continue
        retry_column = f"{column}_retry"
        if retry_column not in combined.columns:
            continue
        base = combined[column] if column in combined.columns else pd.Series(pd.NA, index=combined.index, dtype="object")
        retry = combined[retry_column]
        combined[column] = base.where(base.notna(), retry)
        if column == "groww_error":
            combined[column] = base.where(base.isna(), base)
            combined[column] = combined[column].where(~combined[column].astype("string").str.contains("403", na=False), retry)
        combined = combined.drop(columns=[retry_column])
    return combined


def _run_variant(universe: pd.DataFrame, *, market_cap_threshold: float, output_dir: Path) -> dict[str, object]:
    config = {
        "universe": {
            "min_market_cap": market_cap_threshold,
            "max_pe_ttm": 30.0,
            "min_promoter_pct": 50.0,
            "require_above_50_dma": True,
            "require_above_200_dma": True,
            "min_volume_vs_20d": 1.5,
            "require_volume_high_3m": True,
            "require_delivery_high_3m": True,
            "min_rsi_14_daily": 60.0,
            "min_rsi_14_weekly": 60.0,
            "min_rsi_14_monthly": 60.0,
            "min_revenue_cagr_5y": 0.10,
            "min_pat_cagr_5y": 0.20,
            "require_debt_free": True,
            "require_sector_fii_dii_buying_30d": False,
        }
    }
    screened = apply_screen_filters(universe, config=config, include_missing_inputs=True)
    final_mask = _combine_rule_columns(screened, [column for column, _ in RULES])
    screened["live_shortlist_pass"] = final_mask

    prefix = f"mcap_{int(market_cap_threshold)}"
    final_shortlist = screened.loc[screened["live_shortlist_pass"].eq(True)].copy()
    _write_stock_list(final_shortlist, output_dir / f"{prefix}_final_shortlist.csv")
    write_dataframe_manifest(
        output_dir / f"{prefix}_final_shortlist.csv",
        final_shortlist,
        generated_by="src.report.live_shortlist_fullrun",
    )

    individual_counts: list[dict[str, object]] = []
    sequential_counts: list[dict[str, object]] = []
    survivors = pd.Series(True, index=screened.index, dtype=bool)
    cutoff: dict[str, object] | None = None
    cutoff_before = pd.DataFrame()
    cutoff_after = pd.DataFrame()
    for step, (rule_column, rule_label_template) in enumerate(RULES, start=1):
        rule_label = rule_label_template.format(mcap=int(market_cap_threshold))
        rule = screened[rule_column]
        individual_counts.append(
            {
                "rule_column": rule_column,
                "rule_label": rule_label,
                "individual_pass_count": int(rule.eq(True).sum()),
                "missing_count": int(rule.isna().sum()),
            }
        )
        prior = survivors.copy()
        survivors = prior & rule.eq(True)
        sequential_counts.append(
            {
                "step": step,
                "rule_column": rule_column,
                "rule_label": rule_label,
                "survivors_before": int(prior.sum()),
                "survivors_after": int(survivors.sum()),
                "rule_false_in_prior_survivors": int((prior & rule.eq(False)).sum()),
                "rule_missing_in_prior_survivors": int((prior & rule.isna()).sum()),
            }
        )
        if cutoff is None and int(survivors.sum()) < 30:
            cutoff = {
                "rule_column": rule_column,
                "rule_label": rule_label,
                "before_count": int(prior.sum()),
                "after_count": int(survivors.sum()),
            }
            cutoff_before = screened.loc[prior].copy()
            cutoff_after = screened.loc[survivors].copy()
    individual_df = pd.DataFrame(individual_counts)
    sequential_df = pd.DataFrame(sequential_counts)
    individual_df.to_csv(output_dir / f"{prefix}_individual_counts.csv", index=False)
    sequential_df.to_csv(output_dir / f"{prefix}_sequential_counts.csv", index=False)
    write_dataframe_manifest(
        output_dir / f"{prefix}_individual_counts.csv",
        individual_df,
        generated_by="src.report.live_shortlist_fullrun",
    )
    write_dataframe_manifest(
        output_dir / f"{prefix}_sequential_counts.csv",
        sequential_df,
        generated_by="src.report.live_shortlist_fullrun",
    )
    if cutoff is not None:
        _write_stock_list(cutoff_before, output_dir / f"{prefix}_cutoff_before_{cutoff['rule_column']}.csv")
        _write_stock_list(cutoff_after, output_dir / f"{prefix}_cutoff_after_{cutoff['rule_column']}.csv")
        write_dataframe_manifest(
            output_dir / f"{prefix}_cutoff_before_{cutoff['rule_column']}.csv",
            cutoff_before,
            generated_by="src.report.live_shortlist_fullrun",
        )
        write_dataframe_manifest(
            output_dir / f"{prefix}_cutoff_after_{cutoff['rule_column']}.csv",
            cutoff_after,
            generated_by="src.report.live_shortlist_fullrun",
        )

    return {
        "final_count": int(final_shortlist["symbol"].nunique()),
        "cutoff": cutoff,
        "missing_diag": _missing_diag(screened),
        "groww_usage_diag": _groww_usage_diag(screened),
        "individual_counts": individual_counts,
        "sequential_counts": sequential_counts,
        "top_final": _stock_list(final_shortlist.head(30)),
    }


def _write_stock_list(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "symbol",
        "company_name",
        "trade_date",
        "close",
        "market_cap_cr",
        "promoter_pct",
        "pe_ttm",
        "revenue_cagr_5y",
        "pat_cagr_5y",
        "ebitda_positive_last_5q_flag",
        "volume_vs_20d",
        "delivery_pct_high_63d_flag",
        "rsi_14_daily",
        "rsi_14_weekly",
        "rsi_14_monthly",
        "industry",
        "sector",
        "missing_inputs",
    ]
    keep = [column for column in columns if column in df.columns]
    df[keep].sort_values(["symbol"]).to_csv(path, index=False)


def _stock_list(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for row in df.sort_values(["symbol"]).itertuples(index=False):
        rows.append(
            {
                "symbol": row.symbol,
                "company_name": getattr(row, "company_name", pd.NA),
                "close": getattr(row, "close", pd.NA),
                "market_cap_cr": getattr(row, "market_cap_cr", pd.NA),
                "pe_ttm": getattr(row, "pe_ttm", pd.NA),
                "promoter_pct": getattr(row, "promoter_pct", pd.NA),
            }
        )
    return rows


def _combine_rule_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = df[columns].copy()
    false_any = values.eq(False).fillna(False).any(axis=1)
    missing_any = values.isna().any(axis=1)
    result = pd.Series(True, index=df.index, dtype="boolean")
    result.loc[false_any] = False
    result.loc[~false_any & missing_any] = pd.NA
    return result


def _compute_debt_flag(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    any_signal = pd.Series(False, index=df.index, dtype=bool)
    any_positive = pd.Series(False, index=df.index, dtype=bool)
    any_zero = pd.Series(False, index=df.index, dtype=bool)
    for column in ["debt_equity_ratio", "debt", "net_debt", "face_value_debt", "paid_debt", "debt_redemption"]:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        valid = values.notna()
        any_signal = any_signal | valid
        any_positive = any_positive | (valid & values.gt(0.01))
        any_zero = any_zero | (valid & values.le(0.01))
    result.loc[any_positive] = False
    result.loc[~any_positive & any_signal & any_zero] = True
    return result


def _missing_diag(df: pd.DataFrame) -> dict[str, int]:
    listing_counts = df.groupby("symbol")["trade_date"].transform("size")
    return {
        "market_cap_cr_missing": int(df["market_cap_cr"].isna().sum()),
        "pe_ttm_missing": int(df["pe_ttm"].isna().sum()),
        "promoter_pct_missing": int(df["promoter_pct"].isna().sum()),
        "revenue_cagr_5y_missing": int(df["revenue_cagr_5y"].isna().sum()),
        "pat_cagr_5y_missing": int(df["pat_cagr_5y"].isna().sum()),
        "ebitda_positive_last_5q_flag_missing": int(df["ebitda_positive_last_5q_flag"].isna().sum()),
        "debt_flag_missing": int(_compute_debt_flag(df).isna().sum()),
        "delivery_pct_high_63d_flag_missing": int(df["delivery_pct_high_63d_flag"].isna().sum()),
        "delivery_pct_high_63d_insufficient_history": int(
            (df["delivery_pct_high_63d_flag"].isna() & listing_counts.lt(63)).sum()
        ),
    }


def _groww_usage_diag(df: pd.DataFrame) -> dict[str, int]:
    return {
        "market_cap_from_groww": int(df.get("market_cap_cr_source", pd.Series(pd.NA, index=df.index)).eq("Groww").sum()),
        "pe_from_groww": int(df.get("pe_ttm_source", pd.Series(pd.NA, index=df.index)).eq("Groww").sum()),
        "promoter_from_groww": int(df.get("promoter_pct_source", pd.Series(pd.NA, index=df.index)).eq("Groww").sum()),
        "revenue_cagr_from_groww": int(df.get("revenue_cagr_5y_source", pd.Series(pd.NA, index=df.index)).eq("Groww").sum()),
        "pat_cagr_from_groww": int(df.get("pat_cagr_5y_source", pd.Series(pd.NA, index=df.index)).eq("Groww").sum()),
        "ebitda_flag_from_groww": int(df.get("ebitda_positive_last_5q_flag_source", pd.Series(pd.NA, index=df.index)).eq("Groww").sum()),
        "debt_from_groww": int(df.get("debt_equity_ratio_source", pd.Series(pd.NA, index=df.index)).eq("Groww").sum()),
    }


if __name__ == "__main__":
    main()
