"""Precompute static JSON for the frontend/ static site.

Read-only: reuses src/train_baseline.py's feature list and preprocessing
helpers so numbers match backend/app.py's live /predict, /history, /shap
exactly (same models, same source files). No training, no writes outside
frontend/data/.

Writes:
  frontend/data/predictions.json — per (horizon, model), same fields as
    backend/app.py's /predict, keyed so the frontend can switch client-side
    without a live backend call.
  frontend/data/history.json     — same as backend/app.py's /history.
  frontend/data/shap.json        — same as backend/app.py's /shap.
  frontend/data/comparison.json  — one entry per state-disease combo tested
    in results/metrics/baseline_results.json (reporting-only feature set,
    matching the live models), with an outcome ("works"/"inconclusive"/
    "fails") derived from whether each of the two models' bootstrap CI
    lower bound clears the test-split base rate — not hand-typed labels.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import train_sarima  # noqa: E402 — module import (not just names) so DATASET_PATH can be pointed at Kerala's data
from train_baseline import REPORTING_ONLY_FEATURES, _xgb_frame, build_design_matrix  # noqa: E402
from train_sarima import load_series, split_masks, walk_forward_forecasts  # noqa: E402

STATE, DISEASE = "Maharashtra", "Dengue"
COMBO = f"{STATE} / {DISEASE}"

# Kerala/Dengue (screened VIABLE and trained in F17) added to the Forecast
# page alongside Maharashtra/Dengue. Its own data sources — distinct from
# Maharashtra's — so build_predictions()/build_history() (Maharashtra, left
# completely unmodified below) and the Kerala-specific functions never touch
# the same files.
KERALA_STATE, KERALA_DISEASE = "Kerala", "Dengue"
KERALA_COMBO = f"{KERALA_STATE} / {KERALA_DISEASE}"
FEATURES_EXTRA = ROOT / "data/processed/features_extra.parquet"
MODEL_DATASET_EXTRA = ROOT / "data/processed/model_dataset_extra.parquet"
MODELS_EXTRA_DIR = ROOT / "models/extra"

# EpiClim (the source data behind data/processed/model_dataset.parquet) stops
# reporting after this date; every week after it is a structural gap (no data
# collected), not an observed all-clear. Using the grid's literal last row as
# "the latest week" was picking up ~26 straight weeks of that gap, which fed
# SARIMA a run of manufactured zeros it (correctly) extrapolated toward 0, and
# fed every other model's lag/rolling features the same contaminated history.
# The actual latest row with real coverage is the last one at or before this
# date. Matches train_baseline.py's TEST_END exactly (not this file's
# BOUNDARIES.test_end below, which is one day later and only used for the
# About-page timeline's shaded test-region boundary, not for row selection).
COVERAGE_CUTOFF = "2022-06-29"

FEATURES_PATH = ROOT / "data/processed/features.parquet"
MODEL_DATASET_PATH = ROOT / "data/processed/model_dataset.parquet"
HORIZON_RESULTS_PATH = ROOT / "results/metrics/horizon_results.json"
SARIMA_RESULTS_PATH = ROOT / "results/metrics/sarima_results.json"
SGB_RESULTS_PATH = ROOT / "results/metrics/sgb_results.json"
SHAP_PATH = ROOT / "results/metrics/shap_summary.json"
PERM_PATH = ROOT / "results/metrics/permutation_importance.json"
BASELINE_RESULTS_PATH = ROOT / "results/metrics/baseline_results.json"
EXTRA_RESULTS_PATH = ROOT / "results/metrics/extra_combos_results.json"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "frontend/data"

# 4 combos screened VIABLE and trained in F17 (src/train_extra_combos.py),
# added alongside the original 3-combo scope. Their h=1 numbers (used on the
# Comparison page, matching the original 3 cards' h=1 basis) and all-horizon
# numbers (used on the Models page) both come from extra_combos_results.json,
# not baseline_results.json/horizon_results.json/sarima_results.json/
# sgb_results.json — those stay untouched and still drive the original 3.
EXTRA_COMBOS = [
    "West Bengal / Acute Diarrhoeal Disease",
    "Gujarat / Acute Diarrhoeal Disease",
    "Odisha / Acute Diarrhoeal Disease",
    "Kerala / Dengue",
]
NOT_VIABLE_COMBOS = [
    {"combo": "Tamil Nadu / Dengue", "test_positives": 9, "reason": "fewer than 15 test-split positives (viability bar)"},
]

BOUNDARIES = {
    "train_start": "2009-01-01", "train_end": "2017-12-31",
    "val_start": "2018-01-01", "val_end": "2019-12-31",
    "excluded_start": "2020-01-01", "excluded_end": "2020-12-31",
    "test_start": "2021-01-01", "test_end": "2022-06-30",
    "coverage_cutoff": COVERAGE_CUTOFF,
}

MODEL_KEYS = {"xgboost": "xgboost_reporting_only", "logistic": "logistic_regression_reporting_only"}
# sarima/sgb read from separate result files (src/train_sarima.py, src/train_sgb.py),
# added alongside horizon_results.json without touching it.

COMPARISON_COMBOS = [
    "Maharashtra / Dengue",
    "Karnataka / Acute Diarrhoeal Disease",
    "Maharashtra / Acute Diarrhoeal Disease",
]


def _latest_row(features_path: Path, state: str, disease: str) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    sub = df[(df.state == state) & (df.disease == disease) & (df.week_start_date <= COVERAGE_CUTOFF)]
    sub = sub.sort_values("week_start_date").tail(1).copy()
    sub["state"] = sub["state"].cat.remove_unused_categories()
    sub["disease"] = sub["disease"].cat.remove_unused_categories()
    return sub


def _latest_features_row() -> pd.DataFrame:
    return _latest_row(FEATURES_PATH, STATE, DISEASE)


def _sarima_forecast(dataset_path: Path, order, state: str, disease: str) -> dict:
    """SARIMA has no persisted model artifact, so its 'live' forecast can't be
    recomputed from a saved joblib like the other models. Reuses
    src/train_sarima.py's own functions (imported, not modified) to refit the
    already-selected order, then reads off the forecast from the walk-forward
    origin at/before COVERAGE_CUTOFF — same fix as _latest_row(), applied to
    SARIMA's own mechanism instead of a features-row lookup.
    """
    train_sarima.DATASET_PATH = dataset_path
    sub = load_series(state, disease)
    masks = split_masks(sub["week_start_date"])
    forecasts = walk_forward_forecasts(sub, tuple(order), masks["train"])

    cutoff_idx = sub.index[sub["week_start_date"] <= COVERAGE_CUTOFF][-1]
    return {h: float(np.clip(forecasts[cutoff_idx][h], 0.0, 1.0)) for h in (1, 2, 3, 4)}


def _latest_sarima_forecast() -> dict:
    order = json.loads(SARIMA_RESULTS_PATH.read_text())["order_selected"][COMBO]
    return _sarima_forecast(MODEL_DATASET_PATH, order, STATE, DISEASE)


def build_predictions() -> dict:
    row = _latest_features_row()
    horizon_results = json.loads(HORIZON_RESULTS_PATH.read_text())
    sarima_results = json.loads(SARIMA_RESULTS_PATH.read_text())
    sgb_results = json.loads(SGB_RESULTS_PATH.read_text())
    sarima_forecast = _latest_sarima_forecast()

    out = {}
    for horizon in (1, 2, 3, 4):
        h_entry = horizon_results[str(horizon)]
        out[str(horizon)] = {}

        for model_name, model_key in MODEL_KEYS.items():
            if model_name == "xgboost":
                # joblib.load here only ever reads models/*.joblib files this repo's own
                # training pipeline wrote (src/train_baseline.py) — not untrusted input.
                clf = joblib.load(MODELS_DIR / f"xgboost_reporting_only_h{horizon}.joblib")
                X = _xgb_frame(row, REPORTING_ONLY_FEATURES)
                proba = float(clf.predict_proba(X)[:, 1][0])
            else:
                bundle = joblib.load(MODELS_DIR / f"logistic_regression_reporting_only_h{horizon}.joblib")
                X = build_design_matrix(row, REPORTING_ONLY_FEATURES, columns=bundle["columns"], scaler=bundle["scaler"])
                proba = float(bundle["model"].predict_proba(X)[:, 1][0])

            metrics = h_entry[model_key]["test"][COMBO]
            base_rate = h_entry["base_rate"]["test"][COMBO]
            out[str(horizon)][model_name] = {
                "probability": proba,
                "horizon": horizon,
                "model": model_name,
                "pr_auc": metrics["pr_auc"],
                "pr_auc_ci_lo": metrics["pr_auc_ci_lo"],
                "pr_auc_ci_hi": metrics["pr_auc_ci_hi"],
                "base_rate": base_rate,
                "test_positives": metrics["n_positive"],
                "lower_ci_above_base_rate": metrics["pr_auc_ci_lo"] > base_rate,
            }

        # SARIMA: no persisted model artifact (see src/train_sarima.py's docstring) —
        # the "live" probability is the walk-forward forecast from the origin at/before
        # COVERAGE_CUTOFF (see _latest_sarima_forecast), the same fixed train-fit model.
        sarima_metrics = sarima_results[str(horizon)]["sarima"]["test"][COMBO]
        sarima_base_rate = sarima_results[str(horizon)]["base_rate"]["test"][COMBO]
        out[str(horizon)]["sarima"] = {
            "probability": sarima_forecast[horizon],
            "horizon": horizon,
            "model": "sarima",
            "pr_auc": sarima_metrics["pr_auc"],
            "pr_auc_ci_lo": sarima_metrics["pr_auc_ci_lo"],
            "pr_auc_ci_hi": sarima_metrics["pr_auc_ci_hi"],
            "base_rate": sarima_base_rate,
            "test_positives": sarima_metrics["n_positive"],
            "lower_ci_above_base_rate": sarima_metrics["pr_auc_ci_lo"] > sarima_base_rate,
        }

        # SGB: model artifact IS persisted (models/sgb_reporting_only_h{h}.joblib),
        # same live-prediction pattern as XGBoost/logistic.
        sgb_bundle = joblib.load(MODELS_DIR / f"sgb_reporting_only_h{horizon}.joblib")
        X_sgb = build_design_matrix(row, REPORTING_ONLY_FEATURES, columns=sgb_bundle["columns"])
        sgb_proba = float(sgb_bundle["model"].predict_proba(X_sgb)[:, 1][0])
        sgb_metrics = sgb_results[str(horizon)]["sgb_reporting_only"]["test"][COMBO]
        sgb_base_rate = sgb_results[str(horizon)]["base_rate"]["test"][COMBO]
        out[str(horizon)]["sgb"] = {
            "probability": sgb_proba,
            "horizon": horizon,
            "model": "sgb",
            "pr_auc": sgb_metrics["pr_auc"],
            "pr_auc_ci_lo": sgb_metrics["pr_auc_ci_lo"],
            "pr_auc_ci_hi": sgb_metrics["pr_auc_ci_hi"],
            "base_rate": sgb_base_rate,
            "test_positives": sgb_metrics["n_positive"],
            "lower_ci_above_base_rate": sgb_metrics["pr_auc_ci_lo"] > sgb_base_rate,
        }
    return out


def build_predictions_kerala() -> dict:
    """Kerala/Dengue's analogue of build_predictions() (Maharashtra, left
    untouched above). A dedicated function rather than a parameterized/shared
    one: Kerala's results live in ONE combined file
    (extra_combos_results.json, src/train_extra_combos.py) with different
    field names than Maharashtra's three separate result files
    (horizon_results.json/sarima_results.json/sgb_results.json), so forcing
    both through one generic path would risk the exact kind of subtle
    cross-wiring bug that could silently change Maharashtra's numbers —
    which must stay byte-identical to before this page existed.
    """
    row = _latest_row(FEATURES_EXTRA, KERALA_STATE, KERALA_DISEASE)
    extra_results = json.loads(EXTRA_RESULTS_PATH.read_text())
    sarima_order = extra_results["sarima_order_selected"][KERALA_COMBO]
    sarima_forecast = _sarima_forecast(MODEL_DATASET_EXTRA, sarima_order, KERALA_STATE, KERALA_DISEASE)

    out = {}
    for horizon in (1, 2, 3, 4):
        h_entry = extra_results[str(horizon)]
        out[str(horizon)] = {}

        clf = joblib.load(MODELS_EXTRA_DIR / f"xgboost_reporting_only_h{horizon}.joblib")
        X = _xgb_frame(row, REPORTING_ONLY_FEATURES)
        proba = float(clf.predict_proba(X)[:, 1][0])
        metrics = h_entry["xgboost_reporting_only"]["test"][KERALA_COMBO]
        base_rate = h_entry["base_rate"]["test"][KERALA_COMBO]
        out[str(horizon)]["xgboost"] = {
            "probability": proba, "horizon": horizon, "model": "xgboost",
            "pr_auc": metrics["pr_auc"], "pr_auc_ci_lo": metrics["pr_auc_ci_lo"], "pr_auc_ci_hi": metrics["pr_auc_ci_hi"],
            "base_rate": base_rate, "test_positives": metrics["n_positive"],
            "lower_ci_above_base_rate": metrics["pr_auc_ci_lo"] > base_rate,
        }

        bundle = joblib.load(MODELS_EXTRA_DIR / f"logistic_regression_reporting_only_h{horizon}.joblib")
        X = build_design_matrix(row, REPORTING_ONLY_FEATURES, columns=bundle["columns"], scaler=bundle["scaler"])
        proba = float(bundle["model"].predict_proba(X)[:, 1][0])
        metrics = h_entry["logistic_regression_reporting_only"]["test"][KERALA_COMBO]
        base_rate = h_entry["base_rate"]["test"][KERALA_COMBO]
        out[str(horizon)]["logistic"] = {
            "probability": proba, "horizon": horizon, "model": "logistic",
            "pr_auc": metrics["pr_auc"], "pr_auc_ci_lo": metrics["pr_auc_ci_lo"], "pr_auc_ci_hi": metrics["pr_auc_ci_hi"],
            "base_rate": base_rate, "test_positives": metrics["n_positive"],
            "lower_ci_above_base_rate": metrics["pr_auc_ci_lo"] > base_rate,
        }

        sarima_metrics = h_entry["sarima"]["test"][KERALA_COMBO]
        sarima_base_rate = h_entry["base_rate_sarima"]["test"][KERALA_COMBO]
        out[str(horizon)]["sarima"] = {
            "probability": sarima_forecast[horizon], "horizon": horizon, "model": "sarima",
            "pr_auc": sarima_metrics["pr_auc"], "pr_auc_ci_lo": sarima_metrics["pr_auc_ci_lo"], "pr_auc_ci_hi": sarima_metrics["pr_auc_ci_hi"],
            "base_rate": sarima_base_rate, "test_positives": sarima_metrics["n_positive"],
            "lower_ci_above_base_rate": sarima_metrics["pr_auc_ci_lo"] > sarima_base_rate,
        }

        sgb_bundle = joblib.load(MODELS_EXTRA_DIR / f"sgb_reporting_only_h{horizon}.joblib")
        X_sgb = build_design_matrix(row, REPORTING_ONLY_FEATURES, columns=sgb_bundle["columns"])
        sgb_proba = float(sgb_bundle["model"].predict_proba(X_sgb)[:, 1][0])
        sgb_metrics = h_entry["sgb_reporting_only"]["test"][KERALA_COMBO]
        sgb_base_rate = h_entry["base_rate"]["test"][KERALA_COMBO]
        out[str(horizon)]["sgb"] = {
            "probability": sgb_proba, "horizon": horizon, "model": "sgb",
            "pr_auc": sgb_metrics["pr_auc"], "pr_auc_ci_lo": sgb_metrics["pr_auc_ci_lo"], "pr_auc_ci_hi": sgb_metrics["pr_auc_ci_hi"],
            "base_rate": sgb_base_rate, "test_positives": sgb_metrics["n_positive"],
            "lower_ci_above_base_rate": sgb_metrics["pr_auc_ci_lo"] > sgb_base_rate,
        }
    return out


def compute_peak_week(dataset_path: Path, state: str, disease: str) -> dict:
    """ISO week-of-year with the highest historical reporting rate, computed
    directly (not hardcoded) — verified to reproduce eda_summary.json's
    Maharashtra/Dengue value (peak_week=20, rate=0.571) exactly before being
    trusted for Kerala, which has no eda_summary.json entry at all.
    """
    df = pd.read_parquet(dataset_path)
    sub = df[(df.state == state) & (df.disease == disease)]
    rate_by_week = sub.groupby("week")["reported"].mean()
    return {"peak_week": int(rate_by_week.idxmax()), "peak_week_rate": float(rate_by_week.max())}


def build_history() -> dict:
    df = pd.read_parquet(MODEL_DATASET_PATH)
    sub = df[(df.state == STATE) & (df.disease == DISEASE)].sort_values("week_start_date")
    series = [
        {
            "week_start_date": r.week_start_date.strftime("%Y-%m-%d"),
            "reported": bool(r.reported),
            "total_cases": None if pd.isna(r.total_cases) else float(r.total_cases),
        }
        for r in sub.itertuples()
    ]
    return {"series": series, "boundaries": BOUNDARIES}


def build_history_kerala() -> dict:
    df = pd.read_parquet(MODEL_DATASET_EXTRA)
    sub = df[(df.state == KERALA_STATE) & (df.disease == KERALA_DISEASE)].sort_values("week_start_date")
    series = [
        {
            "week_start_date": r.week_start_date.strftime("%Y-%m-%d"),
            "reported": bool(r.reported),
            "total_cases": None if pd.isna(r.total_cases) else float(r.total_cases),
        }
        for r in sub.itertuples()
    ]
    return {"series": series}


def build_shap() -> list:
    shap_data = json.loads(SHAP_PATH.read_text())[COMBO][:8]
    perm_data = {d["feature"]: d for d in json.loads(PERM_PATH.read_text())[COMBO]}
    return [
        {
            "feature": d["feature"],
            "mean_abs_shap": d["mean_abs_shap"],
            "permutation_importance_mean": perm_data.get(d["feature"], {}).get("importance_mean"),
        }
        for d in shap_data
    ]


def build_comparison() -> list:
    baseline = json.loads(BASELINE_RESULTS_PATH.read_text())
    ablation = baseline["experiment_1_climate_ablation"]
    sarima_results = json.loads(SARIMA_RESULTS_PATH.read_text())
    sgb_results = json.loads(SGB_RESULTS_PATH.read_text())
    out = []
    for combo in COMPARISON_COMBOS:
        base_rate = baseline["class_balance"]["test"][combo]["positive_rate"]

        xgb = ablation["xgboost"]["reporting_only"]["test"][combo]
        logit = ablation["logistic_regression"]["reporting_only"]["test"][combo]
        xgb_cleared = xgb["pr_auc_ci_lo"] > base_rate
        logit_cleared = logit["pr_auc_ci_lo"] > base_rate

        # Added alongside xgboost/logistic without touching the outcome/sentence
        # logic below, which stays based on the original two models exactly as before.
        sarima = sarima_results["1"]["sarima"]["test"][combo]
        sgb = sgb_results["1"]["sgb_reporting_only"]["test"][combo]
        sarima_cleared = sarima["pr_auc_ci_lo"] > base_rate
        sgb_cleared = sgb["pr_auc_ci_lo"] > base_rate

        if xgb_cleared and logit_cleared:
            outcome = "works"
            outcome_text = "Clears chance at horizon 1 for both models."
        elif xgb_cleared or logit_cleared:
            outcome = "inconclusive"
            outcome_text = "Only clears chance for one of the two models — marginal, not reliable."
        else:
            outcome = "fails"
            outcome_text = "Does not clear chance for either model — not usable for decision-making."

        sentence = (
            f"XGBoost PR-AUC {xgb['pr_auc']:.3f} [{xgb['pr_auc_ci_lo']:.3f}, {xgb['pr_auc_ci_hi']:.3f}] "
            f"vs base rate {base_rate:.3f}. {outcome_text}"
        )

        out.append({
            "combo": combo,
            "base_rate": base_rate,
            "xgboost": {
                "pr_auc": xgb["pr_auc"], "pr_auc_ci_lo": xgb["pr_auc_ci_lo"],
                "pr_auc_ci_hi": xgb["pr_auc_ci_hi"], "cleared": xgb_cleared,
            },
            "logistic": {
                "pr_auc": logit["pr_auc"], "pr_auc_ci_lo": logit["pr_auc_ci_lo"],
                "pr_auc_ci_hi": logit["pr_auc_ci_hi"], "cleared": logit_cleared,
            },
            "sarima": {
                "pr_auc": sarima["pr_auc"], "pr_auc_ci_lo": sarima["pr_auc_ci_lo"],
                "pr_auc_ci_hi": sarima["pr_auc_ci_hi"], "cleared": sarima_cleared,
            },
            "sgb": {
                "pr_auc": sgb["pr_auc"], "pr_auc_ci_lo": sgb["pr_auc_ci_lo"],
                "pr_auc_ci_hi": sgb["pr_auc_ci_hi"], "cleared": sgb_cleared,
            },
            "outcome": outcome,
            "sentence": sentence,
        })

    # Additive: the 4 new combos, same card schema, h=1 numbers from
    # extra_combos_results.json instead of baseline_results.json. The loop
    # above (original 3 combos) is untouched.
    extra = json.loads(EXTRA_RESULTS_PATH.read_text())["1"]
    for combo in EXTRA_COMBOS:
        base_rate = extra["base_rate"]["test"][combo]
        xgb = extra["xgboost_reporting_only"]["test"][combo]
        logit = extra["logistic_regression_reporting_only"]["test"][combo]
        sarima = extra["sarima"]["test"][combo]
        sgb = extra["sgb_reporting_only"]["test"][combo]
        xgb_cleared = xgb["pr_auc_ci_lo"] > base_rate
        logit_cleared = logit["pr_auc_ci_lo"] > base_rate
        sarima_cleared = sarima["pr_auc_ci_lo"] > base_rate
        sgb_cleared = sgb["pr_auc_ci_lo"] > base_rate
        cleared_count = sum([xgb_cleared, logit_cleared, sarima_cleared, sgb_cleared])

        if cleared_count == 4:
            outcome, outcome_text = "works", "Clears chance at horizon 1 for all four models."
        elif cleared_count == 0:
            outcome, outcome_text = "fails", "Does not clear chance for any of the four models — not usable for decision-making."
        else:
            outcome, outcome_text = "inconclusive", f"Only clears chance for {cleared_count} of 4 models — marginal, not reliable."

        sentence = (
            f"XGBoost PR-AUC {xgb['pr_auc']:.3f} [{xgb['pr_auc_ci_lo']:.3f}, {xgb['pr_auc_ci_hi']:.3f}] "
            f"vs base rate {base_rate:.3f}. {outcome_text}"
        )

        out.append({
            "combo": combo,
            "base_rate": base_rate,
            "xgboost": {"pr_auc": xgb["pr_auc"], "pr_auc_ci_lo": xgb["pr_auc_ci_lo"], "pr_auc_ci_hi": xgb["pr_auc_ci_hi"], "cleared": xgb_cleared},
            "logistic": {"pr_auc": logit["pr_auc"], "pr_auc_ci_lo": logit["pr_auc_ci_lo"], "pr_auc_ci_hi": logit["pr_auc_ci_hi"], "cleared": logit_cleared},
            "sarima": {"pr_auc": sarima["pr_auc"], "pr_auc_ci_lo": sarima["pr_auc_ci_lo"], "pr_auc_ci_hi": sarima["pr_auc_ci_hi"], "cleared": sarima_cleared},
            "sgb": {"pr_auc": sgb["pr_auc"], "pr_auc_ci_lo": sgb["pr_auc_ci_lo"], "pr_auc_ci_hi": sgb["pr_auc_ci_hi"], "cleared": sgb_cleared},
            "outcome": outcome,
            "sentence": sentence,
        })
    return out


def build_not_viable() -> list:
    return NOT_VIABLE_COMBOS


def build_models_comparison() -> dict:
    """All 4 models x all 3 combos x all 4 horizons, for the Model Comparison
    page. Unlike build_comparison() (h=1 only, from baseline_results.json),
    this covers every horizon, from horizon_results.json/sarima_results.json/
    sgb_results.json directly — nothing hand-typed.
    """
    horizon_results = json.loads(HORIZON_RESULTS_PATH.read_text())
    sarima_results = json.loads(SARIMA_RESULTS_PATH.read_text())
    sgb_results = json.loads(SGB_RESULTS_PATH.read_text())

    model_sources = {
        "xgboost": lambda h, c: (horizon_results[h]["xgboost_reporting_only"]["test"][c], horizon_results[h]["base_rate"]["test"][c]),
        "logistic": lambda h, c: (horizon_results[h]["logistic_regression_reporting_only"]["test"][c], horizon_results[h]["base_rate"]["test"][c]),
        "sarima": lambda h, c: (sarima_results[h]["sarima"]["test"][c], sarima_results[h]["base_rate"]["test"][c]),
        "sgb": lambda h, c: (sgb_results[h]["sgb_reporting_only"]["test"][c], sgb_results[h]["base_rate"]["test"][c]),
    }

    def fill(combo, source_fn):
        data[combo] = {}
        for h in ("1", "2", "3", "4"):
            data[combo][h] = {}
            for model_name, fn in source_fn.items():
                metrics, base_rate = fn(h, combo)
                data[combo][h][model_name] = {
                    "pr_auc": metrics["pr_auc"],
                    "pr_auc_ci_lo": metrics["pr_auc_ci_lo"],
                    "pr_auc_ci_hi": metrics["pr_auc_ci_hi"],
                    "base_rate": base_rate,
                    "test_positives": metrics["n_positive"],
                    "lower_ci_above_base": metrics["pr_auc_ci_lo"] > base_rate,
                }

    data = {}
    for combo in COMPARISON_COMBOS:
        fill(combo, model_sources)

    # Additive: the 4 new combos, same per-horizon shape, read from
    # extra_combos_results.json instead. The loop above is untouched.
    extra_results = json.loads(EXTRA_RESULTS_PATH.read_text())
    extra_sources = {
        "xgboost": lambda h, c: (extra_results[h]["xgboost_reporting_only"]["test"][c], extra_results[h]["base_rate"]["test"][c]),
        "logistic": lambda h, c: (extra_results[h]["logistic_regression_reporting_only"]["test"][c], extra_results[h]["base_rate"]["test"][c]),
        "sarima": lambda h, c: (extra_results[h]["sarima"]["test"][c], extra_results[h]["base_rate_sarima"]["test"][c]),
        "sgb": lambda h, c: (extra_results[h]["sgb_reporting_only"]["test"][c], extra_results[h]["base_rate"]["test"][c]),
    }
    for combo in EXTRA_COMBOS:
        fill(combo, extra_sources)

    return {
        "combos": COMPARISON_COMBOS + EXTRA_COMBOS,
        "models": ["xgboost", "logistic", "sarima", "sgb"],
        "data": data,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # predictions.json / history.json are now per-state (Forecast page state
    # selector). Maharashtra's own values come from build_predictions()/
    # build_history(), both left byte-for-byte unchanged above, so they
    # produce exactly what they did before Kerala existed.
    mh_history = build_history()
    kl_history = build_history_kerala()
    predictions = {
        "maharashtra": build_predictions(),
        "kerala": build_predictions_kerala(),
    }
    history = {
        "boundaries": mh_history["boundaries"],
        "maharashtra": {"series": mh_history["series"], **compute_peak_week(MODEL_DATASET_PATH, STATE, DISEASE)},
        "kerala": {"series": kl_history["series"], **compute_peak_week(MODEL_DATASET_EXTRA, KERALA_STATE, KERALA_DISEASE)},
    }

    artifacts = {
        "predictions": predictions,
        "history": history,
        "shap": build_shap(),
        "comparison": build_comparison(),
        "models": build_models_comparison(),
        "not_viable": build_not_viable(),
    }
    for name, data in artifacts.items():
        json_path = OUT_DIR / f"{name}.json"
        json_path.write_text(json.dumps(data, indent=2))
        print(f"wrote {json_path} ({json_path.stat().st_size} bytes)")

        # Browsers refuse fetch()/XHR reads of file:// JSON (no server involved),
        # so pages opened directly need this same data reachable via <script src>,
        # which local file loading does allow. Generated from the same JSON above
        # in the same step, so the two can't drift apart.
        js_path = OUT_DIR / f"{name}.js"
        js_path.write_text(f"window.__DATA__ = window.__DATA__ || {{}};\nwindow.__DATA__.{name} = {json.dumps(data)};\n")
        print(f"wrote {js_path} ({js_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
