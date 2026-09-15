# Disease Outbreak & Peak Forecasting (India)

## Scope
- **Geography**: India only. No other countries.
- **Unit of analysis**: district (not state, not national).
- **Diseases**: dengue, malaria, influenza/ILI.
- **Generalization**: tested across disease-state pairs within India
  (e.g. train/validate on dengue in Karnataka, test on malaria in Goa).
- **Target frequency**: weekly. Fall back to monthly only if weekly data
  is unavailable for a given source/disease/region, and state explicitly
  in code comments, notebook markdown, and metrics output when a monthly
  fallback is used and why.

## Data integrity rules (non-negotiable)
- **Never shuffle time series data.** Only chronological splits or
  rolling-window (expanding/sliding) splits are allowed for train/val/test.
  Random shuffling of temporally ordered records is a bug.
- **`data/raw/` is read-only.** Never write, modify, or delete anything
  inside `data/raw/`. All derived/cleaned data goes into `data/processed/`
  or similar, never back into `raw/`.
- **Never invent or estimate a missing metric.** If a value cannot be
  computed from available data, write `TODO` (with a short note on what's
  missing) instead of guessing, interpolating silently, or hardcoding a
  placeholder number.
- **Never delete outlier rows.** An apparent outlier may be a real
  outbreak signal — the exact thing this project is trying to detect.
  Outliers may be flagged/annotated but must not be dropped from the
  dataset.
- **District name normalization**: all district names must be normalized
  against GADM India level 2 boundaries before any join across datasets.
  Never join datasets on raw/unnormalized district name strings — spelling
  and naming inconsistencies across Indian government data sources will
  silently corrupt joins.

## Build order
1. Ingestion
2. EDA (exploratory data analysis)
3. SARIMA (baseline time series model)
4. XGBoost
5. LSTM
6. Peak detection
7. SHAP (model interpretability)
8. Streamlit (dashboard/app)

Do not skip ahead in this order without explicit instruction.

## Experiment tracking
- Every experiment/model run writes its metrics to `results/metrics/` as
  a JSON file (one file per run, or clearly namespaced/timestamped).
