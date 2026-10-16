# EXP-2026-10-16-industry-leadlag

Industry lead-lag map, walk-forward. Part A (does the map predict industries' next 4 weeks?): no pass, mean weekly IC +nan. Part B (add 'about to get hot' industries to V3.2's gate): no pass. Files: results.csv, links.csv, ic_weekly.csv, added_industries.csv.

Why zero links: zero industry pairs survived the false discovery rate step in any year (2019-2026). This was checked after the registered result, as exploration only. Small p-values showed up at chance rates: p < 0.05 for 5.1% / 5.6% / 5.9% of pairs in the 2019 / 2022 / 2026 maps, against 5% expected by chance. Their signs were close to a coin flip, 44-58% positive. The smallest p-value was 3e-4, while the correction needed about 2e-5. There is no hidden, spread-out signal that a looser correction would rescue. For comparison, an industry's own last 4 weeks does predict its next 4 weeks a little (weekly rank correlation +0.075), and V3.2's hot-industry gate already uses that. Run log: run.log.
