# Data Dictionary

Canonical schemas live in `data_contracts/*.yaml`.

- `stock_master`: identity and classification layer for the research universe, currently populated from official NSE quote/meta APIs plus an optional approved classification mapping file.
- `stock_daily_facts`: one row per symbol per trading date with raw market fields, delivery fields, and leak-safe technical features.
- `stock_quarterly_fundamentals`: quarterly financial-result layer keyed by symbol, fiscal period, and effective date using official NSE result filings.
- `stock_shareholding_quarterly`: quarterly shareholding pattern keyed by symbol, quarter, and effective date using official NSE corporate filing APIs.
- `sector_flow_fortnightly`: fortnightly sector-wise FPI state parsed from official NSDL report pages, with publication-date handling still pending.
- `daily_screen_universe`: daily lag-valid screening view.

Unavailable fields remain nullable and must not be imputed silently.
