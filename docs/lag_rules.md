# Lag Rules

- `stock_daily_facts` is same-day market data for each `trade_date`.
- `stock_quarterly_fundamentals` is usable only when `effective_from_date <= trade_date`, where `effective_from_date` is the official NSE result broadcast date if present, otherwise the NSE filing date.
- `stock_shareholding_quarterly` is usable only when `effective_from_date <= trade_date`, where `effective_from_date` is the official NSE shareholding broadcast date if present, otherwise the NSE submission date, otherwise the exchange system timestamp.
- `sector_flow_fortnightly` is usable only when `published_date <= trade_date`; because a clean official `published_date` is not yet available from the NSDL pages, sector-flow rows should not enter the historical daily screen until that field is resolved.
- All periodic joins use backward-only as-of joins. Future-dated records are invalid and should fail validation.
