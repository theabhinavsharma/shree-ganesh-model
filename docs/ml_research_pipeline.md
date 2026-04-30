# ML Research Pipeline

This repo now includes a cache-first ML research layer built on top of the existing lag-safe feature history.

## What it does

- builds a reusable feature panel for each objective
- reuses cached panels when the objective, feature set, and source files have not changed
- trains a pure `numpy` logistic model so the stack works in the current environment without external ML packages
- evaluates multiple universes with yearly walk-forward folds
- selects the best universe per objective using cross-sectional ranking metrics instead of one-off hit rates
- scores the latest available cross-section and saves a current ranked universe

## Why this avoids redoing work

- feature panels are stored in `data/ml/panels/<objective>_<fingerprint>/`
- the fingerprint includes:
  - objective definition
  - feature list
  - model-independent research settings
  - source file paths and mtimes
- if none of those change, the panel is reused

## Core command

```bash
python3 -m src.ml.cli --config configs/ml_research.yaml
```

## Output layout

- `data/ml/runs/<timestamp>/fold_metrics.csv`
- `data/ml/runs/<timestamp>/universe_summary.csv`
- `data/ml/runs/<timestamp>/selected_models.csv`
- `data/ml/runs/<timestamp>/<objective>/current_scores.csv`
- `data/ml/runs/<timestamp>/<objective>/selected_model.json`

## Important design choices

- No unofficial proxy fields are required for the model to run.
- Filing and ownership features stay point-in-time because they come through the existing lag-safe joins.
- Macro is joined as-of, backward only.
- Events are joined from the precomputed daily event table when current enough, otherwise rebuilt from raw announcement history for the requested calendar.
- The model is deliberately simple. In this environment, repeatability and leak control matter more than pretending a heavier package exists.

## What to upgrade later

- add tree-based models when `scikit-learn` or `lightgbm` is available
- add nested hyperparameter tuning
- add transaction-cost-aware portfolio optimization on top of the raw scores
- add explicit sector-neutral and market-beta-neutral portfolio construction

