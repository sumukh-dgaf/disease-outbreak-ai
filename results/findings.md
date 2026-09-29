# Findings

Research log for the India disease-outbreak-forecasting project. Every
number below is either quoted directly from a file in this repo, or
computed by a one-off read-only aggregation against a named parquet file
(marked "computed" in the source line, with the exact grouping stated so
it can be reproduced). Where a source file does not contain a number
needed for a finding, and no other file in the repo contains it either,
the finding says **UNKNOWN** rather than estimating. Claims that are
interpretation rather than direct measurement are marked **INTERPRETATION**.

Scope note: several findings below (F2, F3) were produced while the
pipeline still covered 6 state-disease combinations (Karnataka, Maharashtra,
West Bengal × Dengue, Acute Diarrhoeal Disease) and, in F2/F3's case, also
Malaria — before later steps narrowed modeling scope. They are reported as
measured at the time against the named source file, not restated to match
the current 3-combo model scope.

---

## F1 — Public surveillance data is not openly available at weekly, district-level granularity

**Claim:** The public-facing NCVBDC channel publishes vector-borne disease
data at district level but only monthly; the platform capable of finer
(near-real-time/weekly) granularity, IHIP, is login-gated with no public
open-data route.

**Supporting evidence (documentation, no numbers):**
- NCVBDC's malaria page publishes a "District Level Data Malaria Cases and
  Death" PDF and describes monthly reporting products (Monthly
  Epidemiological Situation, Monthly Malaria Information System); the page
  showed a last-updated date of 2026-09-10 at verification time.
- NCVBDC's dengue-specific page was not independently checked for the same
  detail in this pass — dengue's public granularity/frequency is
  **UNKNOWN** beyond what the disease-index page and malaria page show.
- IHIP's two official domains (`ihip.mohfw.gov.in/idsp/#!/login`,
  `ihip.nhp.gov.in/idsp/#!/login`) both route to a login screen; no public,
  credential-free data view was found for either domain in this search.

**Source:** Not a repo file — verified via web search/fetch on 2026-09-15
against ncvbdc.mohfw.gov.in and ihip.mohfw.gov.in / ihip.nhp.gov.in.

---

## F2 — EpiClim (`data/raw/Final_data.csv`) is sporadic outbreak-report data, not routine incidence surveillance

**Claim:** The dataset records isolated outbreak/alert events rather than a
continuous weekly case count — high-volume districts still only cover a
minority of possible weeks, and many district-weeks contain multiple
separate rows rather than one consolidated figure.

**Supporting numbers:**
- Total rows: 8,985. Date range: 2009-05-04 to 2022-06-29 (0 unparseable
  dates out of 8,985).
- Dengue: 1,619 rows total. Top district by row count is Nadia (West
  Bengal) with 36 rows, covering only **29 unique (year, week) pairs** —
  i.e. 7 of its 36 rows are repeats of a (year, week) already seen for
  that district.
- Malaria: 544 rows total. Top district is Gadchiroli (Maharashtra) with
  39 rows, covering **35 unique (year, week) pairs**.
- Across the whole file: 1,098 rows sit in a duplicate (year, week,
  district, Disease) group, spread over 497 such groups — e.g. Balod,
  Chhattisgarh, Acute Diarrhoeal Disease, 2013 week 36 alone has 10 rows.

**Source:** `results/metrics/epiclim_profile.json` — `basic.row_count`,
`basic.date_range`, `focus_diseases.Dengue.row_count`,
`focus_diseases.Dengue.top_district` (name/row_count/unique_year_week_pairs),
`focus_diseases.Malaria.top_district`, `duplicates.duplicate_row_count`,
`duplicates.duplicate_group_count`, `duplicates.top_duplicate_groups[0]`.

---

## F3 — EpiClim's bundled climate columns are unreliable and were dropped in favor of NASA POWER

**Claim:** The `Temp` column in EpiClim, even under the (documented)
assumption that it is Kelvin, produces implausible values for specific
district-weeks; the project therefore discarded EpiClim's `Temp`/`preci`/
`LAI` columns entirely and rebuilt climate features from NASA POWER instead.

**Supporting numbers:**
- Aggregate `Temp` column (assumed Kelvin): min 259.68 K, max 327.728 K,
  mean 304.52 K, median 303.445 K. Converting the extremes to Celsius gives
  a range of **-13.47°C to 54.58°C** across the dataset.
- Specific flagged rows judged not credible for their district/season: 
  Assam / Dhemaji, July 2021, converts to 1.76°C; Assam / Karbi Anglong,
  September 2015, converts to -11.85°C.
- Decision taken: drop `Temp`, `preci`, `LAI` from all downstream output;
  rebuild climate from NASA POWER instead (5 states × 731 ISO weeks =
  3,655 rows, one row per state per week, 2009-2022).

**Source:** aggregate Temp stats from
`results/metrics/epiclim_profile.json` → `climate_columns.Temp` (min, max,
mean, median, celsius_equivalent_range). Specific flagged district-weeks
and the drop decision from `src/build_state_weekly.py` (module docstring).
NASA POWER row count computed directly from
`data/processed/state_climate.parquet` (`len(df)` = 3655; grouped by
`state`, 731 rows each for Karnataka, Kerala, Maharashtra, Tamil Nadu, West
Bengal).

---

## F4 — Reporting coverage per state-disease is low everywhere, and varies roughly 3x across combos

**Claim:** No state-disease combination in the modeled data has a report
in even 40% of the 731 possible ISO weeks (2009-2022); coverage ranges
from 13.3% to 36.9%.

**Supporting numbers (reported weeks / 731 total weeks per combo):**

| State / Disease | Reported | Total | Rate |
|---|---|---|---|
| Karnataka / Acute Diarrhoeal Disease | 270 | 731 | 36.9% |
| West Bengal / Acute Diarrhoeal Disease | 263 | 731 | 36.0% |
| Maharashtra / Acute Diarrhoeal Disease | 259 | 731 | 35.4% |
| Maharashtra / Dengue | 215 | 731 | 29.4% |
| Karnataka / Dengue | 133 | 731 | 18.2% |
| West Bengal / Dengue | 97 | 731 | 13.3% |

**Source:** computed directly from `data/processed/model_dataset.parquet`
(grouped by `state`, `disease`; `reported.sum()` over `len(group)`, group
size confirmed to be 731 for all six combos). Reported-week counts match
`results/metrics/eda_summary.json` →
`cases_distribution_per_state_disease.<combo>.n_reported_weeks` exactly
(cross-checked for all six combos).

---

## F5 — A near-universal reporting collapse occurs between 2019 and 2020

**Claim:** Every one of the six state-disease combinations loses at least
42% of its 2019 reporting rate by 2020, with three combos losing more than
70%.

**Supporting numbers (reported weeks / weeks that year, and % change in rate):**

| State / Disease | 2019 | 2020 | 2021 | 2022 | 2019→2020 change |
|---|---|---|---|---|---|
| Karnataka / Acute Diarrhoeal Disease | 19/52 (36.5%) | 6/53 (11.3%) | 12/52 (23.1%) | 11/52 (21.2%) | -69.0% |
| Karnataka / Dengue | 22/52 (42.3%) | 6/53 (11.3%) | 4/52 (7.7%) | 2/52 (3.8%) | -73.2% |
| Maharashtra / Acute Diarrhoeal Disease | 17/52 (32.7%) | 10/53 (18.9%) | 8/52 (15.4%) | 3/52 (5.8%) | -42.3% |
| Maharashtra / Dengue | 24/52 (46.2%) | 7/53 (13.2%) | 23/52 (44.2%) | 3/52 (5.8%) | -71.4% |
| West Bengal / Acute Diarrhoeal Disease | 16/52 (30.8%) | 3/53 (5.7%) | 22/52 (42.3%) | 8/52 (15.4%) | -81.6% |
| West Bengal / Dengue | 4/52 (7.7%) | 0/53 (0.0%) | 2/52 (3.8%) | 1/52 (1.9%) | -100.0% |

**INTERPRETATION:** the simultaneity of the drop across all six
state-disease combinations in the same calendar year is consistent with a
shared administrative cause — most plausibly COVID-19-era disruption to
routine surveillance and reporting infrastructure — rather than six
independent epidemiological coincidences. This is an interpretation, not a
measurement; the data does not itself identify a cause.

**Source:** computed directly from `data/processed/model_dataset.parquet`
(grouped by `state`, `disease`, `year`, for years 2019-2022; `reported.sum()`
over `len(group)`). The 2019→2020 rate deltas match
`results/metrics/eda_summary.json` → `admin_jump_candidates` for the two
entries it flags (Karnataka: -0.2522 ADD / -0.3099 Dengue at
`admin_jump_threshold` 0.15; Maharashtra's listed jump there is for 2012,
not 2019→2020).

---

## F6 — Three combos were excluded from modeling for insufficient positive density

**Claim:** Karnataka/Dengue, West Bengal/Acute Diarrhoeal Disease, and
West Bengal/Dengue were dropped from the modeling scope; West Bengal/Dengue
in particular has as few as 3-4 positive weeks in some splits.

**Supporting numbers (positive count / split size, using the train
2009-2018 / validation 2019-2020 / test 2021-2022 split in effect at the
time this screening was done — this predates the later change to train
2009-2017 / validation 2018-2019 / test 2021-01-01–2022-06-29 with 2020
excluded):**

| State / Disease | Train (2009-2018) | Validation (2019-2020) | Test (2021-2022) |
|---|---|---|---|
| Karnataka / Dengue | 99/522 | 28/105 | 6/104 |
| West Bengal / Acute Diarrhoeal Disease | 214/522 | 19/105 | 30/104 |
| West Bengal / Dengue | 90/522 | 4/105 | 3/104 |

For comparison, the three combos that were **kept**:

| State / Disease | Train (2009-2018) | Validation (2019-2020) | Test (2021-2022) |
|---|---|---|---|
| Karnataka / Acute Diarrhoeal Disease | 222/522 | 25/105 | 23/104 |
| Maharashtra / Acute Diarrhoeal Disease | 221/522 | 27/105 | 11/104 |
| Maharashtra / Dengue | 158/522 | 31/105 | 26/104 |

**Source:** computed directly from `data/processed/model_dataset.parquet`
(grouped by `state`, `disease`, filtered by `year` into the three ranges
above; `reported.sum()` over `len(group)`).

---

## F7 — The seasonal baseline does not generalize out-of-sample

**Claim:** A climatology-style baseline (per-state-disease, per-week-of-year
historical reporting rate, fit on train only, thresholded at 0.5) shows the
largest train-to-test PR-AUC drop of the four baseline models.

**Supporting numbers (overall, across the 3 modeled combos):**
- Seasonal baseline PR-AUC: train 0.591, validation 0.466, test **0.273**
  — a drop of more than half from train to test.
- For comparison, logistic regression (same train/val/test split, full
  feature set): PR-AUC train 0.597, validation 0.578, test 0.473.
- Per-combo test PR-AUC for the seasonal baseline: Karnataka/ADD 0.283,
  Maharashtra/ADD 0.168, Maharashtra/Dengue 0.572 (the one combo where it
  holds up).

**Source:** `results/metrics/baseline_results.json` →
`models.seasonal.train.overall.pr_auc`,
`models.seasonal.validation.overall.pr_auc`,
`models.seasonal.test.overall.pr_auc`, and the corresponding
`models.seasonal.test.<combo>.pr_auc` / `models.logistic_regression.*`
entries.

---

## F8 — Model comparison: XGBoost and logistic regression both beat both baselines on test PR-AUC, by a similar margin

**Claim:** On the current scope (Karnataka/ADD, Maharashtra/Dengue,
Maharashtra/ADD; train 2009-2017, validation 2018-2019, test
2021-01-01–2022-06-29, 2020 excluded), both learned models outperform the
persistence and seasonal baselines on test PR-AUC, with XGBoost marginally
ahead.

**Supporting numbers (overall PR-AUC, across the 3 kept combos):**

| Model | Train | Validation | Test |
|---|---|---|---|
| Persistence | 0.465 | 0.434 | 0.344 |
| Seasonal | 0.591 | 0.466 | 0.273 |
| Logistic regression | 0.597 | 0.578 | 0.473 |
| XGBoost | 0.684 | 0.616 | **0.478** |

- Class balance (positive rate, target = next-week `reported`): train
  39.0% (n=1,386), validation 38.1% (n=312), test 25.6% (n=234).
- Rows dropped for falling entirely in 2020: 159 (53 per combo, exactly
  one year × 3 combos).
- A specific failure mode: Maharashtra/ADD has only 11 positive weeks out
  of 78 in test; both logistic regression and XGBoost predict 0 positives
  there at the default 0.5 threshold (precision = recall = 0.0), while
  still scoring 0.157-0.130 PR-AUC (better than random at that base rate,
  but 0 correct positive predictions at threshold 0.5).

**Source:** `results/metrics/baseline_results.json` →
`class_balance.*.overall`, `models.persistence.*.overall.pr_auc`,
`models.seasonal.*.overall.pr_auc`, `models.logistic_regression.*.overall.pr_auc`,
`models.xgboost.*.overall.pr_auc`, and
`models.{logistic_regression,xgboost}.test.Maharashtra / Acute Diarrhoeal Disease`
for the precision/recall/pr_auc figures. The 159-row 2020 exclusion count
is printed by `src/train_baseline.py` at run time and is not separately
stored in the JSON; it is reproducible by re-running that script.

---

## F9 — Climate adds only a marginal gain over reporting history and seasonality alone

**Claim:** For both logistic regression and XGBoost, adding the 15 climate
lag features (rainfall/temperature/humidity, lags 0-4) on top of reporting
history and calendar-seasonality features improves test PR-AUC only
slightly.

**Supporting numbers (overall test PR-AUC, reporting-history-only vs
+ climate):**

| Model | Reporting-history only | + Climate | Difference |
|---|---|---|---|
| Logistic regression | 0.459 | 0.473 | +0.014 |
| XGBoost | 0.477 | 0.478 | +0.001 |

**Source:** `results/metrics/baseline_results.json` →
`experiment_1_climate_ablation.logistic_regression.reporting_only.test.overall.pr_auc`,
`experiment_1_climate_ablation.logistic_regression.with_climate.test.overall.pr_auc`,
`experiment_1_climate_ablation.xgboost.reporting_only.test.overall.pr_auc`,
`experiment_1_climate_ablation.xgboost.with_climate.test.overall.pr_auc`.

---

## F10 — Reporting-only models only clear the base rate (with statistical confidence) for Maharashtra/Dengue, and only marginally for Karnataka/ADD

**Claim:** Across horizons h=1-4, the lower bound of the bootstrap 95% CI
on test PR-AUC exceeds the test-split base rate for both models on
Maharashtra/Dengue at every horizon tested, for XGBoost on Karnataka/ADD
at h=1 only, and never for either model on Maharashtra/ADD.

**Supporting numbers (max horizon, out of h=1-4, where PR-AUC's lower CI
bound > base rate; "none" if it never happens):**

| State / Disease | Logistic regression (reporting-only) | XGBoost (reporting-only) |
|---|---|---|
| Karnataka / Acute Diarrhoeal Disease | none | h=1 |
| Maharashtra / Acute Diarrhoeal Disease | none | none |
| Maharashtra / Dengue | h=4 | h=4 |

Underlying per-horizon values (test PR-AUC [CI lo, CI hi] vs. base rate):
Karnataka/ADD XGBoost h=1: 0.485 [0.311, 0.661] vs base rate 0.295 (CI lo
> base rate); h=2: 0.445 [0.261, 0.617] vs 0.295 (CI lo < base rate).
Maharashtra/Dengue logistic regression, all four horizons clear the base
rate of 0.333: h=1 0.643 [0.461, 0.831], h=2 0.618 [0.456, 0.839], h=3
0.625 [0.451, 0.831], h=4 0.637 [0.449, 0.824].

**Source:** computed directly from `results/metrics/horizon_results.json`
— for each horizon `h` in `"1".."4"`, `<model>.test.<combo>.pr_auc`,
`<model>.test.<combo>.pr_auc_ci_lo`, and `base_rate.test.<combo>`, for
`model` in `logistic_regression_reporting_only` and
`xgboost_reporting_only`.

---

## F11 — Maharashtra/Dengue's degradation curve is flat; Karnataka/ADD's (XGBoost) is monotonically decaying

**Claim:** As horizon increases from 1 to 4 weeks, Maharashtra/Dengue's
test PR-AUC (logistic regression) stays within a narrow band, while
Karnataka/ADD's test PR-AUC (XGBoost) decreases at every step.

**Supporting numbers:**
- Maharashtra/Dengue, logistic regression, test PR-AUC by horizon: h=1
  0.643, h=2 0.618, h=3 0.625, h=4 0.637 — range of 0.025 (0.618 to 0.643),
  not monotonic (dips at h=2, recovers by h=4).
- Karnataka/Acute Diarrhoeal Disease, XGBoost, test PR-AUC by horizon: h=1
  0.485, h=2 0.445, h=3 0.408, h=4 0.287 — strictly decreasing at every
  step, a drop of 0.198 from h=1 to h=4 (-40.8% relative to the h=1 value).

**INTERPRETATION:** a plausible mechanism for this contrast is that
Dengue's reporting pattern is dominated by a slow-moving seasonal regime
(so knowing the current reporting-history/seasonal state still carries
information 4 weeks out), while Acute Diarrhoeal Disease reporting is
driven more by short, localized outbreak dynamics whose informative signal
decays faster than 4 weeks. This is a proposed explanation, not something
measured directly by either model — the data only shows the two curves'
shapes, not their cause.

**Source:** computed directly from `results/metrics/horizon_results.json`
— `logistic_regression_reporting_only.test."Maharashtra / Dengue".pr_auc`
and `xgboost_reporting_only.test."Karnataka / Acute Diarrhoeal
Disease".pr_auc`, for `h` in `"1".."4"`.

---

## F12 — XGBoost's apparent Maharashtra/Dengue peak at h=3 is not distinguishable from h=1

**Claim:** XGBoost's test PR-AUC for Maharashtra/Dengue is highest at h=3
(0.717) among the four horizons tested, but its confidence interval
overlaps substantially with h=1's, so the data does not support treating
h=3 as a genuinely better horizon than h=1.

**Supporting numbers:**
- h=1: PR-AUC 0.639, 95% CI [0.469, 0.857].
- h=3: PR-AUC 0.717, 95% CI [0.530, 0.872].
- Overlap region: [0.530, 0.857] — h=3's entire CI lower portion up to
  0.857 sits inside h=1's CI, and h=1's point estimate (0.639) falls
  inside h=3's CI.

**Source:** computed directly from `results/metrics/horizon_results.json`
— `xgboost_reporting_only.test."Maharashtra / Dengue".pr_auc`,
`.pr_auc_ci_lo`, and `.pr_auc_ci_hi` for `h="1"` and `h="3"`.

---

## F13 — Permutation importance does not corroborate SHAP's feature attributions for the h=1 +climate XGBoost model

**Claim:** For the same trained model and the same test-split rows, mean
|SHAP| and permutation importance disagree sharply on which features
matter — including a full sign flip on SHAP's top-ranked feature for one
of the three combos — so the SHAP ranking reflects how this particular
model distributes credit internally, not an attribution corroborated by
an independent importance method.

**Supporting numbers:**
- `reported_roll_8` is SHAP's #1 feature (by mean |SHAP|) for all three
  combos. Its permutation-importance rank (out of 29 features) is #1 for
  Karnataka/ADD (mean +0.1136) and #1 for Maharashtra/Dengue (mean
  +0.1286) — but **#29 (dead last) for Maharashtra/ADD, with a negative
  mean (-0.0594)**: shuffling this "most important" feature slightly
  improved the model's test PR-AUC rather than degrading it.
- Across all three combos, permutation importance is nearly flat overall:
  11/29 (Karnataka/ADD), 11/29 (Maharashtra/ADD), and 8/29
  (Maharashtra/Dengue) features have |mean importance| < 0.001, and 6/29,
  17/29, and 16/29 features respectively have a *negative* mean
  importance (removing signal, by chance, helped the score).

**Source:** computed directly from `results/metrics/shap_summary.json`
(mean |SHAP| rank per feature per combo) and
`results/metrics/permutation_importance.json` (`importance_mean` and rank
per feature per combo) — both produced by `src/explainability.py` against
the same `models/xgboost_with_climate.joblib` model and the same test-split
rows. Cross-referenced: `reported_roll_8`'s entries in both files for all
three combo keys.

---

## F14 — XGBoost and logistic regression are not statistically distinguishable at any horizon, for any of the three combos

**Claim:** For every (combo, horizon) cell tested — 3 combos × 4 horizons
= 12 cells — the test-split bootstrap 95% CI on PR-AUC for XGBoost
(reporting-only) overlaps the CI for logistic regression (reporting-only).
No combo/horizon shows one model's CI fully separated from the other's, so
neither model can be called better than the other with the confidence
this bootstrap provides.

**Supporting numbers (PR-AUC 95% CI, XGBoost vs. logistic regression,
test split):**

| State / Disease | h | XGBoost CI | Logistic CI |
|---|---|---|---|
| Karnataka / Acute Diarrhoeal Disease | 1 | [0.311, 0.661] | [0.269, 0.634] |
| Karnataka / Acute Diarrhoeal Disease | 2 | [0.261, 0.617] | [0.229, 0.599] |
| Karnataka / Acute Diarrhoeal Disease | 3 | [0.245, 0.619] | [0.262, 0.618] |
| Karnataka / Acute Diarrhoeal Disease | 4 | [0.173, 0.473] | [0.205, 0.586] |
| Maharashtra / Acute Diarrhoeal Disease | 1 | [0.069, 0.246] | [0.084, 0.291] |
| Maharashtra / Acute Diarrhoeal Disease | 2 | [0.064, 0.279] | [0.086, 0.385] |
| Maharashtra / Acute Diarrhoeal Disease | 3 | [0.071, 0.255] | [0.086, 0.367] |
| Maharashtra / Acute Diarrhoeal Disease | 4 | [0.080, 0.277] | [0.085, 0.307] |
| Maharashtra / Dengue | 1 | [0.469, 0.857] | [0.461, 0.831] |
| Maharashtra / Dengue | 2 | [0.513, 0.864] | [0.456, 0.839] |
| Maharashtra / Dengue | 3 | [0.530, 0.872] | [0.451, 0.831] |
| Maharashtra / Dengue | 4 | [0.397, 0.786] | [0.449, 0.824] |

Every row overlaps (each pair's intervals share a non-empty range) — e.g.
Maharashtra/ADD h=2 is the widest apparent gap in point estimates but
still overlaps: XGBoost's upper bound (0.279) exceeds logistic's lower
bound (0.086), and logistic's upper bound (0.385) exceeds XGBoost's lower
bound (0.064).

**Source:** computed directly from `results/metrics/horizon_results.json`
— for each `h` in `"1".."4"` and each combo,
`xgboost_reporting_only.test.<combo>.pr_auc_ci_lo` /
`.pr_auc_ci_hi` and `logistic_regression_reporting_only.test.<combo>.pr_auc_ci_lo`
/ `.pr_auc_ci_hi`, compared pairwise for interval overlap.

---

## F15 — SARIMA (Fourier-seasonal) clears chance only for Maharashtra/Dengue, at every horizon; never for the two Acute Diarrhoeal Disease combos

**Claim:** A SARIMAX model per combo (order selected by train-only AIC grid
search over p,q in 0-2, d in 0-1, with a sin/cos Fourier pair at period 52
as exogenous seasonality, since a literal seasonal order of 52 is unstable
on weekly data) has its test-split bootstrap 95% CI lower bound on PR-AUC
exceed the test-split base rate only for Maharashtra/Dengue, at all four
horizons tested. It never clears the base rate for Karnataka/ADD or
Maharashtra/ADD, at any horizon.

**Orders selected (train-only AIC):** Karnataka/ADD (0,1,1), Maharashtra/
Dengue (1,1,1), Maharashtra/ADD (0,1,1).

**Supporting numbers (test PR-AUC [95% CI] vs. base rate, by horizon):**

| State / Disease | h=1 | h=2 | h=3 | h=4 | Base rate |
|---|---|---|---|---|---|
| Karnataka / Acute Diarrhoeal Disease | 0.341 [0.221, 0.557] | 0.374 [0.209, 0.558] | 0.296 [0.188, 0.483] | 0.314 [0.173, 0.502] | 0.295 (h1-2) / 0.282 (h3) / 0.269 (h4) |
| Maharashtra / Dengue | 0.575 [0.412, 0.798] | 0.555 [0.408, 0.783] | 0.591 [0.418, 0.780] | 0.596 [0.398, 0.775] | 0.333 |
| Maharashtra / Acute Diarrhoeal Disease | 0.209 [0.098, 0.410] | 0.186 [0.094, 0.358] | 0.185 [0.093, 0.371] | 0.179 [0.090, 0.352] | 0.141 |

Only Maharashtra/Dengue's lower CI bound (0.398-0.412 across horizons)
exceeds its base rate (0.333) at every horizon; Karnataka/ADD's lower
bounds (0.173-0.221) sit below its base rate (0.269-0.295) at every
horizon — notably, this means SARIMA does *not* clear chance for
Karnataka/ADD even at h=1, unlike XGBoost (reporting-only), which clears
it marginally at h=1 per F10.

**Comparison to XGBoost/logistic (reporting-only):** SARIMA's test PR-AUC
CI overlaps both XGBoost's and logistic's CI for every combo at every
horizon (12/12 cells each) — the same "not statistically distinguishable"
pattern F14 found between XGBoost and logistic themselves. SARIMA is not
shown to be worse or better than either learned model by this bootstrap.

**Source:** `src/train_sarima.py` (model fitting, walk-forward forecasting,
order selection), output at `results/metrics/sarima_results.json` —
`order_selected.<combo>`, and for each `h` in `"1".."4"`,
`<h>.sarima.test.<combo>.pr_auc` / `.pr_auc_ci_lo` / `.pr_auc_ci_hi`,
`<h>.base_rate.test.<combo>`. Compared against
`results/metrics/horizon_results.json` — `<h>.xgboost_reporting_only.test.<combo>`
and `<h>.logistic_regression_reporting_only.test.<combo>`
`.pr_auc_ci_lo` / `.pr_auc_ci_hi`, for interval overlap.

---

## F16 — Stochastic Gradient Boosting clears chance only for Maharashtra/Dengue too; climate adds little at h=1, matching F9's pattern for the other models

**Claim:** sklearn's `GradientBoostingClassifier` (reporting-only features,
balanced sample weights) shows the same combo pattern as XGBoost, logistic
regression, and SARIMA: its test-split bootstrap 95% CI lower bound on
PR-AUC clears the base rate only for Maharashtra/Dengue, at every horizon,
and never for Karnataka/ADD or Maharashtra/ADD.

**Supporting numbers (test PR-AUC [95% CI] vs. base rate, by horizon,
reporting-only feature set):**

| State / Disease | h=1 | h=2 | h=3 | h=4 | Base rate |
|---|---|---|---|---|---|
| Karnataka / Acute Diarrhoeal Disease | 0.452 [0.282, 0.633] | 0.461 [0.279, 0.637] | 0.407 [0.233, 0.593] | 0.321 [0.166, 0.518] | 0.295 (h1-2) / 0.282 (h3) / 0.269 (h4) |
| Maharashtra / Dengue | 0.624 [0.459, 0.819] | 0.683 [0.508, 0.849] | 0.644 [0.472, 0.845] | 0.625 [0.427, 0.804] | 0.333 |
| Maharashtra / Acute Diarrhoeal Disease | 0.117 [0.062, 0.216] | 0.133 [0.066, 0.257] | 0.142 [0.075, 0.288] | 0.129 [0.073, 0.234] | 0.141 |

Karnataka/ADD's lower CI bounds (0.166-0.282) never quite clear its base
rate (0.269-0.295) at any horizon — closer than SARIMA got, but still not
clearing, including at h=1 where XGBoost marginally does (F10).

**Climate ablation at h=1 (reporting-only vs. +climate, matching
train_baseline.py's Experiment 1):**

| State / Disease | Reporting-only | + Climate | Difference |
|---|---|---|---|
| Karnataka / Acute Diarrhoeal Disease | 0.452 | 0.580 | +0.128 |
| Maharashtra / Dengue | 0.624 | 0.637 | +0.013 |
| Maharashtra / Acute Diarrhoeal Disease | 0.117 | 0.119 | +0.002 |

For Maharashtra/Dengue and Maharashtra/ADD, climate adds about as little
as it did for XGBoost and logistic regression in F9 (+0.001 to +0.014
there). Karnataka/ADD is the exception — SGB gains +0.128 PR-AUC from
climate features, a larger jump than either XGBoost or logistic saw for
any combo in F9. **INTERPRETATION:** this could mean SGB's particular
boosting/regularization combination is better able to extract signal from
the climate lag features for this specific combo, or it could be noise
from a single train/test split — F9's own climate-ablation comparison
covers only XGBoost and logistic, so there is no other model's result to
corroborate this size of a jump against.

**Comparison to XGBoost/logistic (reporting-only):** SGB's test PR-AUC CI
overlaps both XGBoost's and logistic's CI for every combo at every horizon
(12/12 cells each), the same pattern as SARIMA in F15 — not statistically
distinguishable from either by this bootstrap.

**Source:** `src/train_sgb.py` (model fitting, feature reuse from
`train_baseline.py`), output at `results/metrics/sgb_results.json` — for
each `h` in `"1".."4"`, `<h>.sgb_reporting_only.test.<combo>.pr_auc` /
`.pr_auc_ci_lo` / `.pr_auc_ci_hi`, `<h>.base_rate.test.<combo>`; climate
ablation from `1.sgb_with_climate.test.<combo>.pr_auc`. Compared against
`results/metrics/horizon_results.json` (XGBoost/logistic CIs, as in F15)
and `results/metrics/baseline_results.json` →
`experiment_1_climate_ablation.{xgboost,logistic_regression}.*.test.overall.pr_auc`
(for the F9 climate-gain comparison).
