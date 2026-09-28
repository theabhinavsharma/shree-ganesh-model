# All-in x hold matrix (EXP-2026-09-27-allin-hold-matrix)

Producer: `src/agentic/sim_allin_matrix.py` (2026-09-27 audit fixes are listed in its docstring).
100% of capital rotates every h sessions into that week's picks, equal weight; empty picks = cash.

Files: `matrix_<entry>_<cost>.csv` (summary, one row per window x approach x hold x era) and
`phases_<entry>_<cost>.csv` (every phase offset). Each has a `.manifest.json` with columns and units.

- entry `close`: registered mode, trade at the signal-day close, flat 0.5% round trip.
- entry `next_open`: buy/sell at the next session's open, upper-circuit-locked buys skipped, cost by ADV.
- cost `turnover`: only the weight that changes pays; `full`: registered 0.5% on all capital per rotation.
- era: `all`, `disc` (< 2023) and `conf` (2023+), cut from the same daily NAV. Use `cagr_worst6` /
  `dd_worst6` to compare worst phases across holds (6 phases each); `*_worst_all` is over h//5 phases.
- MODEL arms run only when rows.parquet `pred` passes the point-in-time price check (see manifest).

Known limits: static 2026 industry labels; no-mcap securities absent; delistings frozen at last close;
unadjusted corporate actions in the panel show up as real moves.

Present: matrix_close_turnover.csv, matrix_next_open_turnover.csv, phases_close_turnover.csv, phases_next_open_turnover.csv

`../allin_matrix.csv` is the pre-audit output (gap moves zeroed, adjusted-price core, same-day G);
do not use it.
