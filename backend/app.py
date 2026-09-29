"""Read-only FastAPI backend for the Maharashtra/Dengue forecast interface.

Scope: Maharashtra/Dengue only. Reuses src/train_baseline.py's feature list
and preprocessing helpers so predictions match exactly what produced
results/metrics/horizon_results.json. No training, no writes; every metric
field in /predict is read straight from horizon_results.json, never
recomputed (lower_ci_above_base_rate is a plain comparison of two numbers
already read from that file, not a new metric).
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train_baseline import REPORTING_ONLY_FEATURES, _xgb_frame, build_design_matrix  # noqa: E402

STATE, DISEASE = "Maharashtra", "Dengue"
COMBO = f"{STATE} / {DISEASE}"

FEATURES_PATH = ROOT / "data/processed/features.parquet"
MODEL_DATASET_PATH = ROOT / "data/processed/model_dataset.parquet"
HORIZON_RESULTS_PATH = ROOT / "results/metrics/horizon_results.json"
SHAP_PATH = ROOT / "results/metrics/shap_summary.json"
PERM_PATH = ROOT / "results/metrics/permutation_importance.json"
MODELS_DIR = ROOT / "models"

# Split boundaries per CLAUDE.md (train 2009-2017 / val 2018-2019 / 2020 excluded / test 2021-01-01..2022-06-30).
BOUNDARIES = {
    "train_start": "2009-01-01", "train_end": "2017-12-31",
    "val_start": "2018-01-01", "val_end": "2019-12-31",
    "excluded_start": "2020-01-01", "excluded_end": "2020-12-31",
    "test_start": "2021-01-01", "test_end": "2022-06-30",
}

MODEL_KEYS = {"xgboost": "xgboost_reporting_only", "logistic": "logistic_regression_reporting_only"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)  # ponytail: "*" is fine for a local read-only research demo; tighten if ever deployed


def _latest_features_row() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    sub = df[(df.state == STATE) & (df.disease == DISEASE)].sort_values("week_start_date").tail(1).copy()
    # Must match train_baseline.py's load_model_data(): drop unused categories so the
    # state/disease codebook XGBoost sees matches what it was trained on (3 categories,
    # not all 6 in the original parquet) — otherwise inplace_predict rejects the category.
    sub["state"] = sub["state"].cat.remove_unused_categories()
    sub["disease"] = sub["disease"].cat.remove_unused_categories()
    return sub


@app.get("/predict")
def predict(horizon: int, model: str):
    if horizon not in (1, 2, 3, 4):
        raise HTTPException(400, "horizon must be 1-4")
    if model not in MODEL_KEYS:
        raise HTTPException(400, "model must be 'xgboost' or 'logistic'")

    row = _latest_features_row()
    if model == "xgboost":
        # joblib.load here only ever reads models/*.joblib files this repo's own
        # training pipeline wrote (src/train_baseline.py) — not untrusted input.
        clf = joblib.load(MODELS_DIR / f"xgboost_reporting_only_h{horizon}.joblib")
        X = _xgb_frame(row, REPORTING_ONLY_FEATURES)
        proba = float(clf.predict_proba(X)[:, 1][0])
    else:
        bundle = joblib.load(MODELS_DIR / f"logistic_regression_reporting_only_h{horizon}.joblib")
        X = build_design_matrix(row, REPORTING_ONLY_FEATURES, columns=bundle["columns"], scaler=bundle["scaler"])
        proba = float(bundle["model"].predict_proba(X)[:, 1][0])

    h_entry = json.loads(HORIZON_RESULTS_PATH.read_text())[str(horizon)]
    metrics = h_entry[MODEL_KEYS[model]]["test"][COMBO]
    base_rate = h_entry["base_rate"]["test"][COMBO]

    return {
        "probability": proba,
        "horizon": horizon,
        "model": model,
        "pr_auc": metrics["pr_auc"],
        "pr_auc_ci_lo": metrics["pr_auc_ci_lo"],
        "pr_auc_ci_hi": metrics["pr_auc_ci_hi"],
        "base_rate": base_rate,
        "test_positives": metrics["n_positive"],
        "lower_ci_above_base_rate": metrics["pr_auc_ci_lo"] > base_rate,
    }


@app.get("/history")
def history():
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


@app.get("/shap")
def shap():
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
