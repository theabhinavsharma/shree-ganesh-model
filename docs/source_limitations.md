# Source Limitations

The v1 daily market-data source is the NSE UDiFF bhavcopy archive URL pattern supplied by the user:

- `https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_ddmmyyyy.csv`

Confirmed official v1 public sources now wired:

- NSE bhavcopy UDiFF archive for daily prices, volume, trades, deliverable quantity, and delivery percent.
- NSE quote/meta APIs for stock identity fields such as ISIN, company name, current industry string, and listing status.
- NSE corporate shareholding APIs for symbol-by-symbol historical shareholding pattern.
- NSE promoter pledged-data API for symbol-by-symbol historical promoter pledge percentages.
- NSE corporate financial-results APIs for symbol-by-symbol historical quarterly result records.
- NSDL fortnightly sector-wise FPI report pages for sector-level investment values and net investment changes.

Still limited or unresolved:

- Official NSE industry classification mapping by symbol is still not exposed here as a clean public symbol-to-sector/basic-industry file, so `sector` and `basic_industry` remain pending unless the user provides an approved mapping.
- NSE quarterly-results payloads do not cleanly expose all balance-sheet and cash-flow fields needed for debt, cash, CFO, FCF, ROE, and ROCE in the current mapped implementation.
- NSDL sector-flow report pages expose the period-end report values, but a clean official machine-readable `published_date` was not found. The raw/normalized table is loaded, but historical daily forward-fill should remain disabled until publication timing is resolved or an approved assumption is provided.
- Shareholding history and quarterly-results history require symbol-by-symbol pulls. The unfiltered list endpoints return only a latest-style snapshot and should not be mistaken for full history.

Explicitly unavailable as clean public v1 dataset:

- Daily stock-wise FII buy/sell values.
- Daily stock-wise DII buy/sell values.
- Clean merged daily stock-wise FII+DII flow feed across all NSE equities.

These fields are intentionally left unavailable rather than approximated.
