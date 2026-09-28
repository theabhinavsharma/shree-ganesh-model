# Order backlog test: where the effect comes from (descriptive)

Half-year breakdown behind the FAILED registered test EXP-2026-09-28-order-backlog (logs/leader_sleeve/order_backlog_test.log).
Not a test and not pre-registered; see the manifests for units and caveats.

- `h1_by_half.csv`: HIGH (book >= 2 years of revenue, above SMA200) vs LOW (< 1 year, above SMA200), P(+50% within 95 sessions) by entry half-year.
- `hot_by_half.csv`: the same split inside hot industries.

Reading:
- 2019-2022: HIGH trails LOW in every half-year from 2020H1 to 2022H2 (-4.7 to -22.1 pts).
- 2023+: the advantage sits in entries from 2023H1 to 2024H1 (+15.6, +14.6, +14.8 pts; rally window HIGH 46.4% vs LOW 31.2%,
  clustered CI +5.3..+24.5). From 2024H2 to 2026H1 it is HIGH 9.0% vs LOW 10.8% (CI -7.3..+3.9).
- Civil construction, defence, rail wagons and shipbuilding dominate HIGH in the rally window; the top 10 HIGH symbols hold 44% of hits.
  Reweighting LOW to HIGH's industry mix cuts the rally-window gap from +15.2 to +7.1 pts and the whole 2023+ gap to about zero.
- `ind` in rows.parquet is a current screener label, not point-in-time.
