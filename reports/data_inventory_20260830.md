# Deep-History Data Inventory — fetched & QC'd 2026-08-30

All feeds below are on disk, checkpointed, and re-runnable. Coverage verified year-by-year.

| Feed | Rows | Symbols | Coverage | QC verdict |
|---|---|---|---|---|
| Corporate announcements | 1,290,621 | 2,932 | 2016-01 → 2026-08 | ✅ continuous, volume grows with NSE filing growth |
| Insider trades (PIT), sized | 174,341 | 2,151 | 2019-01 → 2026-02 | ✅ before/after holding %, buy qty present |
| Results calendar (legacy) | 111,439 | 1,550 | 2005-02 → 2024-12 | ✅ 2005→2024 via legacy endpoint |
| Results calendar (integrated era) | 32,273 | 1,782 | 2025-01 → 2026-08 | ✅ closes the 2025-26 gap natively |
| Shareholding (promoter/public %) | 31,767 | 1,753 | 2015-12 → 2026-08 | ✅ ~22 quarters/symbol; FII/DII% needs XBRL phase-2 |
| Block deals (with client names) | 4,367 | 533 | 2016-01 → 2026-04 | ✅ continuous |
| Prices + delivery (CA-repaired) | 5.0M | 2,900+ | 2015-01 → daily cron | ✅ 27-check gate |
| Corporate actions | 25.7k | — | 2015 → daily cron | ✅ repaired + verified whitelist |
| Quarterly P&L numbers (screener) | 2,094 | 1,206 | ~3yr rolling → weekly cron | ⚠️ 3yr only; 10yr = archive crawl (decision pending) |

## Cross-validation

- Announcements results-flags vs integrated calendar, Apr–Jul 2026: **75%** of flagged names present in both sources (1713 of 2291).

## Known gaps (stated, not hidden)

- **Analyst estimates**: no free source exists. Surprise must be proxied by time-series expectations.
- **FII/DII % per stock-quarter**: requires ~9k XBRL fetches on top of the SHP master (phase 2, ~overnight).
- **10-yr quarterly P&L numbers**: NSE archive HTML crawl, ~44k requests over several days (decision pending). Reaction-based studies do NOT need it — print dates + prices suffice.
- **News (media)**: RSS is live-capture only; no honest historical backfill exists without paid feeds.