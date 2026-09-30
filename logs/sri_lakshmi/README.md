# Sri Lakshmi paper batches

One file per week: `screen_<date>.json` = the 9 stocks the Sri Lakshmi rule picked from data through that date
(written by `src/agentic/screen_sri_lakshmi.py`, never edited afterwards). `outcomes.jsonl` = the daily score of every
batch (written by `src/agentic/score_sri_lakshmi.py`): buy at the next open, sell at the close of session 126,
0.5% round-trip cost, equal weight, no stop.

Rule: the model-ranked screen (trend stock, industry heat percentile >= 0.70, top 9 by the ensemble score) minus hot
industries whose budget-capex + activity percentile (`P_pct` in `data/derived/industry_scores_policy.parquet`) is
below 0.30. It passed EXP-2026-09-29-industry-policy (arm G1, 5/5 phases) in `logs/experiments.jsonl`.

Status: PAPER. A batch becomes real money only after `evals/human_review/sri_lakshmi_<date>.md` is filled in and
signed. `created_note` in each file says whether it was saved before or after the entry session opened.
