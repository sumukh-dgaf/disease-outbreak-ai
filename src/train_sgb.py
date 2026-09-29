"""Stochastic Gradient Boosting (sklearn GradientBoostingClassifier) added
alongside the existing baseline/XGBoost/logistic models — nothing in
train_baseline.py is modified; its feature lists, split logic, horizon
target builder (with its leakage assertion), and evaluation functions are
imported and reused directly, not redefined.

Reporting-only features at h = 1-4 (multi-horizon, matching
train_baseline.py's run_horizon_experiment), plus a +climate variant at
h = 1 only, matching train_baseline.py's Experiment 1 climate-ablation
comparison (which is itself single-horizon in baseline_results.json).

Class imbalance is handled with sample_weight = compute_sample_weight(
'balanced', y_train) rather than XGBoost's scale_pos_weight, since
GradientBoostingClassifier has no built-in positive-class weight param.

sklearn's GradientBoostingClassifier (unlike XGBoost) has no native
categorical support, so state/disease are one-hot encoded via
train_baseline.build_design_matrix — the same helper logistic regression
uses — with no scaler (tree-based models don't need feature scaling).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import (  # noqa: E402
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    FULL_FEATURES,
    REPORTING_ONLY_FEATURES,
    build_design_matrix,
    build_horizon_data,
    chronological_split,
    combo_label,
    evaluate_model,
    to_native,
)

OUT_PATH = Path("results/metrics/sgb_results.json")
MODELS_DIR = Path("models")
HORIZONS = [1, 2, 3, 4]

SGB_PARAMS = dict(
    subsample=0.8,
    max_depth=3,
    n_estimators=200,
    learning_rate=0.05,
    random_state=42,
)


def fit_sgb(train: pd.DataFrame, feature_list: list, target_col: str):
    X_train = build_design_matrix(train, feature_list)
    y_train = train[target_col].astype(int)
    sample_weight = compute_sample_weight("balanced", y_train)
    model = GradientBoostingClassifier(**SGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, list(X_train.columns)


def sgb_predict(model, columns: list, df: pd.DataFrame, feature_list: list) -> tuple:
    X = build_design_matrix(df, feature_list, columns=columns)
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pd.Series(pred, index=df.index), pd.Series(proba, index=df.index)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Bootstrap: N={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, CI={BOOTSTRAP_CI}")
    print(f"SGB params: {SGB_PARAMS}, sample_weight=compute_sample_weight('balanced', y_train)\n")

    horizon_results = {}
    for h in HORIZONS:
        print(f"\n{'=' * 70}\nSGB horizon h={h}\n{'=' * 70}")
        model_df, target_col = build_horizon_data(h)
        splits = chronological_split(model_df)

        base_rate = {}
        for split_name, sub in splits.items():
            rate = sub[target_col].mean()
            base_rate[split_name] = {"overall": float(rate)}
            for (state, disease), g in sub.groupby(["state", "disease"], observed=True):
                base_rate[split_name][combo_label(state, disease)] = float(g[target_col].mean())

        model, columns = fit_sgb(splits["train"], REPORTING_ONLY_FEATURES, target_col)

        model_path = MODELS_DIR / f"sgb_reporting_only_h{h}.joblib"
        joblib.dump({"model": model, "columns": columns}, model_path)
        print(f"  saved trained model to {model_path}")

        def sgb_fn(df, _model=model, _columns=columns):
            return sgb_predict(_model, _columns, df, REPORTING_ONLY_FEATURES)

        reporting_only_results = evaluate_model(
            f"SGB (reporting-only, h={h})", splits, sgb_fn, target_col=target_col, verbose=True,
        )

        horizon_results[str(h)] = {
            "target_col": target_col,
            "base_rate": base_rate,
            "sgb_reporting_only": reporting_only_results,
        }

        if h == 1:
            # +climate variant, matching train_baseline.py's Experiment 1
            # climate ablation (itself only run at h=1 in baseline_results.json).
            climate_model, climate_columns = fit_sgb(splits["train"], FULL_FEATURES, target_col)

            def sgb_climate_fn(df, _model=climate_model, _columns=climate_columns):
                return sgb_predict(_model, _columns, df, FULL_FEATURES)

            climate_results = evaluate_model(
                "SGB (+ climate, h=1)", splits, sgb_climate_fn, target_col=target_col, verbose=True,
            )
            horizon_results["1"]["sgb_with_climate"] = climate_results

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_native(horizon_results), f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
