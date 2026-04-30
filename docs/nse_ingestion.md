# NSE Ingestion

The v1 market-data ingestion path starts from the provided NSE UDiFF bhavcopy downloader code and extends it backward with older official NSE archives so daily market data can be pulled back to 2015.

- `src/ingest/nse/session.py`: HTTP session with retry policy.
- `src/ingest/nse/fetch_bhavcopy.py`: date loop, weekend skipping, date-aware NSE URL generation, idempotent raw-file writes, manifest logging, and error capture.
- `src/ingest/nse/normalize.py`: bhavcopy normalization into `stock_daily_facts` fields, including legacy bhavcopy zip parsing and delivery merge logic.
- `src/ingest/nse/io.py`: raw storage helpers.
- `src/ingest/nse/models.py`: fetch request/result models.
- `src/ingest/nse/cli.py`: CLI entry point.

Official source behavior by period:

- `2020-01-01` onward: use the provided UDiFF bhavcopy path `https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_ddmmyyyy.csv`.
- Before `2020-01-01`: use the older official NSE bhavcopy zip archive `https://nsearchives.nseindia.com/content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip`.
- Before `2020-01-01`: merge official delivery data from `https://nsearchives.nseindia.com/archives/equities/mto/MTO_ddmmyyyy.DAT`.

The module preserves the source behavior from the original script where applicable:

- Weekend dates are skipped.
- The user-agent header is retained.
- Empty-body and non-200 responses are recorded as fetch errors.
- Raw daily files are saved separately by `trade_date`.
- Older delivery files are not approximated; they are merged from the official NSE delivery archive.

Production gaps that still require approved inputs:

- Delivery-aware daily features are supported from official NSE files across the 2015+ history, but the older period requires the separate delivery archive to be present.
- Market-cap and valuation snapshot fields still need an approved point-in-time reference dataset.
- NSE stock classification by symbol still needs an approved NSE mapping source.
- Some quarterly fields such as debt, cash, CFO, FCF, ROE, and ROCE remain unavailable from the current mapped result payload.
- Sector-flow report publication timing remains unresolved, so the raw and normalized sector-flow table is available but should not yet be used for historical daily joins.
- Long-history return studies still need a separate corporate-action adjustment layer if they are meant to be split/bonus adjusted rather than raw-price based.
