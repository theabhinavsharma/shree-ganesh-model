Durable repository rules:

- Use the provided NSE downloader as the authoritative v1 market-data ingestion path.
- Do not silently substitute unofficial sources for market data.
- Do not fabricate unavailable data.
- Prefer explicit TODOs and documented limitations over fake completeness.
- Preserve behavior when refactoring functional code.
- Do not write opaque report artifacts. Important parquet or csv outputs should have a matching `.manifest.json` sidecar and a folder `README.md` when the output is meant for human review.
- Do not assume unit meaning from suffixes like `_pct`; use the data contract or manifest as the source of truth.
