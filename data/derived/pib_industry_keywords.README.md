# pib_industry_keywords.csv

Reviewable map from policy terms in PIB (Press Information Bureau) releases to our industry names.
`src/agentic/fetch_pib_releases.py` applies it when it consolidates `data/derived/pib_releases.parquet`
and writes the result to `industries_tagged` (with `industry_tag_evidence` showing the matched text).

- One row per `pattern -> industry`. The pattern is a case-insensitive Python regex, word-bounded.
- `industry` must be an exact string from `data/derived/screener_industry.parquet` (`industry`, HTML-unescaped);
  the fetcher refuses to run if it is not.
- `ministry_regex` (optional) limits a row to releases from matching ministries (Defence procurement terms).
- Matched against the title and the first 600 characters of the body, after ministry names are removed.
- Only direct links. Second-order effects are left out on purpose.

Edit the CSV, then rerun `python3 src/agentic/fetch_pib_releases.py --consolidate-only` (no refetch needed).
Rules, exclusions and coverage are listed in `pib_industry_keywords.csv.manifest.json`.
