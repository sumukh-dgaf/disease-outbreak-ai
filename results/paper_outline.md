# Paper Outline

Maps what this project has actually produced onto a 12-section paper
structure. Built only from `results/findings.md` (F1-F9) plus the figure
files present in `results/figures/`. No new numeric claims are introduced
here beyond what F1-F9 already state; figures are referenced as
complementary visual assets, not as independent evidence.

Hypotheses under test:
- **H1** — multimodal features (climate + reporting history) improve forecasting.
- **H2** — a generalized framework works across disease-region combinations.
- **H3** — XAI identifies contributing factors.

---

## 1. Abstract

- One-line problem statement: India lacks openly available weekly,
  district-level outbreak surveillance data (F1); this project instead
  models the *reporting process itself* using the best available proxy
  data (F2).
- One-line method statement: state-week reporting-propensity models
  (persistence, seasonal, logistic regression, XGBoost) trained on
  reporting history, calendar seasonality, and climate (F8, F9).
- Headline result: learned models beat both naive baselines on test
  PR-AUC, but by a narrow margin, and climate contributes almost nothing
  on top of reporting history alone (F8, F9).
- Headline caveat: the framework only runs on 3 of 6 candidate
  disease-state combinations; the other 3 had to be dropped for data
  sparsity (F6) — this belongs in the abstract because it directly bears
  on the generalization claim (H2).
- State H1/H2/H3 verdicts in one line each (see Discussion, section 10, for
  the citations).

---

## 2. Introduction

- Motivate from data availability, not just disease burden: public
  channels publish only monthly, and only some diseases at district level
  (F1) — this is why a project studying "weekly" forecasting has to build
  its own weekly proxy target rather than use a ready-made feed.
- Frame the core object of study as *whether a district/state reports a
  case in a given week*, not case counts — a direct consequence of F2
  (EpiClim is sporadic outbreak-report data, not incidence surveillance)
  and F4 (even the best-covered combos report in well under half of all
  possible weeks).
- State the three hypotheses (H1, H2, H3) explicitly here so the
  Discussion section has something to resolve against.
- Flag early that COVID-era disruption (F5) forced a scope decision
  (excluding 2020 entirely) that shapes everything downstream.

---

## 3. Literature Review

- **GAP.** Nothing in `results/findings.md` speaks to prior work — this
  project's findings are entirely empirical/data-derived, not a synthesis
  of existing literature.
- To fill this: a separate reading pass over (a) outbreak/case-count
  forecasting literature for dengue/malaria in India, (b) surveillance
  "reporting-completeness" or "nowcasting" literature (the closest existing
  frame for what F2/F4 describe), and (c) prior applications of SHAP/XAI to
  epidemiological forecasting. None of this is producible from the current
  repo — it requires external reading and citation, not another script run.

---

## 4. Research Gap

- Partially supported by findings, but the literature-side half of the
  argument is missing (depends on section 3, which is a GAP).
- What's evidenced: existing public data (F1) does not support weekly,
  district-level, multi-disease modeling directly, forcing derived
  proxies — this is a legitimate, data-grounded gap statement on its own.
- What's evidenced: standard case-count modeling assumptions break down
  under F2/F4 — sporadic, duplicate-laden, low-coverage reporting means a
  "predict next week's case count" framing is not viable without first
  characterizing *whether a report happens at all* (motivates the
  reporting-propensity framing used in Methodology/Results).
- **GAP:** cannot state "this is a gap *relative to the literature*"
  without section 3 existing first.

---

## 5. Methodology

- Disease-label normalization and duplicate handling as data-cleaning
  steps precede modeling — not directly numbered in findings.md as an F#,
  but F2's duplicate-group evidence (1,098 rows / 497 groups) is the
  reason this step exists; treat as methodological justification, not a
  separate result.
- Climate source replacement: EpiClim's bundled Temp/preci/LAI columns
  were dropped and replaced with NASA POWER pulls (F3) — state this as a
  methodological decision with its exact justification (implausible
  Assam district-week temperatures, F3).
- Feature construction: reporting-history features (lags, rolling rates)
  plus calendar seasonality plus climate lags 0-4 for
  rainfall/temperature/humidity (F9 names these explicitly as the two
  feature groups compared).
- Scope-reduction decisions as methodology, not results: combos dropped
  for insufficient density (F6), and 2020 excluded entirely from all
  splits (F8's 159-row exclusion figure).
- State the chronological, non-random train/validation/test split scheme
  used in the final experimental setup (F8: train 2009-2017 / validation
  2018-2019 / test 2021-01-01 to 2022-06-29).

---

## 6. Dataset

- Primary source and its nature: EpiClim (`data/raw/Final_data.csv`),
  8,985 rows, 2009-05-04 to 2022-06-29, characterized as sporadic
  outbreak-report data rather than routine surveillance (F2).
- Concrete illustration of sparsity: top Dengue district (Nadia) has 36
  rows over only 29 unique (year, week) pairs; top Malaria district
  (Gadchiroli) has 39 rows over 35 unique pairs (F2).
- Secondary source: NASA POWER climate, pulled per state centroid to
  replace EpiClim's unreliable climate columns — 5 states × 731 ISO weeks
  = 3,655 rows (F3). Illustrate with `results/figures/climate_seasonality_
  rainfall.png`, `climate_seasonality_temperature.png`, and
  `climate_seasonality_humidity.png` (per-state seasonal climate curves).
- Case-magnitude characteristics among reported weeks only: illustrate
  with `results/figures/cases_distribution_histograms.png` (log-scale,
  right-skewed) and `results/figures/cases_median_p90_by_week.png`
  (median/p90 by week-of-year) — both are visual complements, not
  separately numbered findings.
- Final modeled scope: 3 of 6 state-disease combinations, after F6's
  density screening.

---

## 7. Experimental Setup

- Target definition: whether a state-disease combination reports in a
  given week (binary), not case counts — implicit in every F4-F9 finding,
  which all operate on reported/total week counts and reporting rates.
- Chronological split (F8): train 2009-2017, validation 2018-2019, test
  2021-01-01 to 2022-06-29, with all of 2020 excluded (159 rows, F8) —
  justified by F5's 2019→2020 collapse.
- Four models compared (F7, F8): persistence, seasonal climatology,
  logistic regression, XGBoost.
- Ablation design (F9): reporting-history-only feature set vs. +15
  climate lag features (rainfall/temperature/humidity, lags 0-4), run for
  both logistic regression and XGBoost.
- Class balance at each split, needed to interpret PR-AUC correctly:
  train 39.0% positive (n=1,386), validation 38.1% (n=312), test 25.6%
  (n=234) (F8).

---

## 8. Results

- Headline comparison table (F8): persistence/seasonal/logistic/XGBoost
  test PR-AUC = 0.344 / 0.273 / 0.473 / 0.478 — both learned models clear
  both baselines, XGBoost narrowly ahead.
- Seasonal baseline's out-of-sample instability (F7): PR-AUC train 0.591
  to test 0.273, the largest train→test drop of the four models;
  illustrate with `results/figures/reporting_rate_by_year.png` (shows the
  year-to-year pattern the seasonal model is implicitly assuming is
  stable, and F5 shows it isn't).
- Climate ablation result (F9): +climate improves test PR-AUC by only
  +0.014 (logistic) and +0.001 (XGBoost) over reporting-history-only —
  illustrate with `results/figures/lag_correlation_heatmaps.png`, which
  shows why: raw Spearman correlations between reporting and lagged
  climate are weak (mostly 0.0-0.35) even before any modeling.
- Per-combo failure mode (F8): Maharashtra/ADD test set has only 11
  positive weeks of 78; both logistic regression and XGBoost predict zero
  positives there at the default threshold (precision = recall = 0.0),
  despite non-trivial PR-AUC (0.157 / 0.130) — a threshold/calibration
  issue, not pure non-signal.
- Reporting-rate seasonality context: `results/figures/
  reporting_rate_by_week_of_year.png` shows the 6-series seasonal pattern
  underlying F4's density figures, useful as a Results-section visual even
  though F4 itself is stated as an aggregate density table.

---

## 9. Explainability

- **GAP.** No SHAP or other feature-importance analysis exists in
  `results/findings.md` or anywhere in `results/metrics/`. Per CLAUDE.md's
  build order, SHAP comes after LSTM and peak detection, neither of which
  has been run yet.
- To fill this: run SHAP against the trained XGBoost model(s) from F8/F9
  (with-climate and reporting-only variants both, to make the ablation
  interpretable feature-by-feature, not just as an aggregate PR-AUC
  delta), and write the resulting importances to `results/metrics/` as
  its own JSON per CLAUDE.md's experiment-tracking rule.
- This section is also where H3 would normally be resolved — currently
  impossible; see Discussion.

---

## 10. Discussion

- **H1 (multimodal features improve forecasting): partially supported.**
  Climate features move test PR-AUC in the right direction for both
  models, but the effect is small enough to call negligible in practice
  — logistic +0.014, XGBoost +0.001 (F9). The direction supports the
  hypothesis; the magnitude does not support a strong version of it.
- **H2 (a generalized framework works across disease-region
  combinations): not supported.** Half of the six candidate combinations
  (Karnataka/Dengue, West Bengal/ADD, West Bengal/Dengue) could not be
  modeled at all and were dropped for insufficient positive density (F6,
  with West Bengal/Dengue down to 3-4 positive weeks in some splits).
  Among the 3 combinations that remained, performance is still uneven
  enough that one of them (Maharashtra/ADD) produces zero correct
  positive predictions at the default threshold in test (F8). A framework
  that only runs on half its intended targets, and fails outright on one
  of the survivors, has not demonstrated generalization.
- **H3 (XAI identifies contributing factors): GAP — untested.** No SHAP
  analysis has been run (section 9); this hypothesis cannot yet be
  scored as supported, partial, or not supported. It should not be
  marked "not supported" — that would misrepresent absence of evidence as
  a negative result.
- Reconcile F7 and F9: the seasonal baseline's instability (F7) and
  climate's weak marginal contribution (F9) point the same direction —
  the signal in this data is dominated by reporting history/administrative
  factors (F5, F6), not by seasonal or climate regularities.
- The COVID-era collapse (F5, marked INTERPRETATION there) should be
  discussed as a likely confound behind both the seasonal baseline's
  instability and the need to exclude 2020 from all splits (F8) — flagged
  explicitly as interpretation, not fact, consistent with F5.

---

## 11. Limitations

- Data source is sporadic outbreak-report data, not routine incidence
  surveillance (F2) — any claim about true underlying case burden, as
  opposed to reporting behavior, is out of scope of what this project
  measured.
- Public data access itself is a limitation of the study design, not just
  the phenomenon studied: no open weekly/district feed exists to validate
  against (F1), and dengue-specific public granularity was not even
  independently confirmed (F1 marks this UNKNOWN).
- Only 3 of 6 target disease-state combinations could be modeled (F6);
  conclusions do not generalize to the excluded combinations by
  construction.
- COVID-era disruption (F5) required excluding all of 2020 and likely
  still affects 2021-2022 test-period stationarity assumptions —
  discussed as INTERPRETATION in F5, not confirmed causally.
- No explainability results exist yet (section 9 GAP) — any downstream
  claim about *why* the models perform as they do beyond the climate
  ablation (F9) is currently unsupported.

---

## 12. Conclusion

- Restate the core empirical result: reporting-propensity models beat
  naive baselines but only modestly, and mostly on the strength of
  reporting history rather than climate (F8, F9).
- Restate the scope constraint plainly: this is a 3-combination result,
  not a 6-combination one, because half the candidates failed a basic
  density screen (F6) — the generalization hypothesis (H2) is the weakest
  link, not the modeling approach itself.
- Name the two concrete next steps implied by the GAPs above: run SHAP
  (section 9) to address H3, and treat the F5 COVID-interpretation as a
  hypothesis to test explicitly (e.g. a structural-break or
  changepoint check) rather than an assumed explanation.
