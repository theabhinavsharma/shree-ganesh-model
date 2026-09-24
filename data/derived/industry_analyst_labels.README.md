# industry_analyst_labels.csv

Hand-labelled industry pointers for liquid (core-band) NSE equities that NO feed labels:
NSE smIndustry absent, screener.in page missing / unclassified after the rename-chain and
name-search retry (fetch_screener_industry.py --retry). Written 2026-09-23.

- `analog_symbol`: the listed symbol whose industry label this name takes — the same company
  under its current symbol where NSE's symbolchange.csv missed the rename, else the closest
  listed peer by business. The label itself is resolved at use time in whichever taxonomy
  the consumer runs (NSE smIndustry or NSE 4-level via screener), so the file never hardcodes
  a label string.
- `basis`, `confidence`: why; `medium` = peer analog, not the company itself.
- Provenance tag downstream: `analyst_inference`. Backtests report results with and without.
- ETFs among the residual are NOT here: they are flagged as fund units by registered name in
  build_security_master.py.
