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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train_baseline import REPORTING_ONLY_FEATURES, _xgb_frame, build_design_matrix  # noqa: E402

STATE, DISEASE = "Maharashtra", "Dengue"
COMBO = f"{STATE} / {DISEASE}"

FEATURES_PATH = ROOT / "data/processed/features.parquet"
MODEL_DATASET_PATH = ROOT / "data/processed/model_dataset.parquet"
HORIZON_RESULTS_PATH = ROOT / "results/metrics/horizon_results.json"
SARIMA_RESULTS_PATH = ROOT / "results/metrics/sarima_results.json"
SGB_RESULTS_PATH = ROOT / "results/metrics/sgb_results.json"
SHAP_PATH = ROOT / "results/metrics/shap_summary.json"
PERM_PATH = ROOT / "results/metrics/permutation_importance.json"
BASELINE_RESULTS_PATH = ROOT / "results/metrics/baseline_results.json"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "frontend/data"

BOUNDARIES = {
    "train_start": "2009-01-01", "train_end": "2017-12-31",
    "val_start": "2018-01-01", "val_end": "2019-12-31",
    "excluded_start": "2020-01-01", "excluded_end": "2020-12-31",
    "test_start": "2021-01-01", "test_end": "2022-06-30",
}

MODEL_KEYS = {"xgboost": "xgboost_reporting_only", "logistic": "logistic_regression_reporting_only"}
# sarima/sgb read from separate result files (src/train_sarima.py, src/train_sgb.py),
# added alongside horizon_results.json without touching it.

COMPARISON_COMBOS = [
    "Maharashtra / Dengue",
    "Karnataka / Acute Diarrhoeal Disease",
    "Maharashtra / Acute Diarrhoeal Disease",
]


def _latest_features_row() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    sub = df[(df.state == STATE) & (df.disease == DISEASE)].sort_values("week_start_date").tail(1).copy()
    sub["state"] = sub["state"].cat.remove_unused_categories()
    sub["disease"] = sub["disease"].cat.remove_unused_categories()
    return sub


def build_predictions() -> dict:
    row = _latest_features_row()
    horizon_results = json.loads(HORIZON_RESULTS_PATH.read_text())
    sarima_results = json.loads(SARIMA_RESULTS_PATH.read_text())
    sgb_results = json.loads(SGB_RESULTS_PATH.read_text())

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
        # the "live" probability reuses the most recent walk-forward forecast it
        # already computed for this horizon, from the same fixed train-fit model.
        sarima_metrics = sarima_results[str(horizon)]["sarima"]["test"][COMBO]
        sarima_base_rate = sarima_results[str(horizon)]["base_rate"]["test"][COMBO]
        out[str(horizon)]["sarima"] = {
            "probability": sarima_results["latest_forecast"][COMBO][str(horizon)],
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
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "predictions": build_predictions(),
        "history": build_history(),
        "shap": build_shap(),
        "comparison": build_comparison(),
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
