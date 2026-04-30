# Proxy Signals

Unavailable institutional-flow fields should not be faked. If the research process still needs related signals, use explicit proxy fields in a separate layer with clear labeling.

Possible future proxies:

- Sector-level FPI momentum from fortnightly sector-wise FPI investment changes.
- Shareholding trend proxies from quarterly `fii_fpi_pct_qoq_change`, `dii_pct_qoq_change`, and `mf_pct`.
- Participation proxies from volume expansion and delivery expansion, using official NSE bhavcopy delivery fields rather than unofficial estimates.
- If point-in-time valuation data remains unavailable, price-only momentum and shareholding-trend signals can still be used, but they are not substitutes for true market-cap or PE filters.

These proxies are not substitutes for true daily stock-wise FII/DII flow.
