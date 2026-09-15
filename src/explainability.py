"""SHAP explainability for the trained +climate XGBoost model, per state-disease.

Loads models/xgboost_with_climate.joblib (produced by src/train_baseline.py
— not retrained here, since XGBoost's subsample/colsample_bytree make
refitting non-deterministic without a fixed seed). Computes SHAP values on
the test split only, separately for each of the 3 modeled state-disease
combinations.

Compatibility note: shap 0.49.1's UBJSON model loader cannot parse the
bracketed scientific-notation base_score string ("[5E-1]") that xgboost
3.2.0 emits, and raises ValueError before any SHAP computation happens.
_patch_shap_xgboost_base_score() works around this by fixing that one
field after shap's own UBJSON decode, before shap reads it. This is a
version-compatibility shim, not a modeling decision.

This script only computes and saves outputs. It does not interpret them.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from train_baseline import (
    CLIMATE_FEATURES,
    FULL_FEATURES,
    TARGET_COL,
    _xgb_frame,
    chronological_split,
    combo_label,
    load_model_data,
)

MODEL_PATH = Path("models/xgboost_with_climate.joblib")
FIGURES_DIR = Path("results/figures")
METRICS_PATH = Path("results/metrics/shap_summary.json")
PERMUTATION_METRICS_PATH = Path("results/metrics/permutation_importance.json")

TOP_N_DEPENDENCE = 3
PERMUTATION_N_REPEATS = 20
PERMUTATION_SEED = 42

REPORTING_HISTORY_FEATURES = [
    "reported_lag_1", "reported_lag_2", "reported_lag_3", "reported_lag_4",
    "reported_roll_4", "reported_roll_8",
]
SEASONALITY_FEATURES = [
    "week_of_year", "week_sin_52", "week_cos_52", "week_sin_26", "week_cos_26",
]
OTHER_FEATURES = [
    c for c in FULL_FEATURES
    if c not in REPORTING_HISTORY_FEATURES + SEASONALITY_FEATURES + CLIMATE_FEATURES
]


def _patch_shap_xgboost_base_score():
    """Fix shap's UBJSON decode of xgboost's bracketed base_score string.

    See module docstring. Patches shap.explainers._tree.decode_ubjson_buffer
    in place so XGBTreeModelLoader (which calls it by bare name) picks up
    the fix.
    """
    import shap.explainers._tree as shap_tree

    original_decode = shap_tree.decode_ubjson_buffer

    def patched_decode(fd):
        obj = original_decode(fd)
        try:
            param = obj["learner"]["learner_model_param"]
            base_score = param.get("base_score")
            if isinstance(base_score, str) and base_score.startswith("[") and base_score.endswith("]"):
                param["base_score"] = base_score[1:-1]
        except (KeyError, TypeError):
            pass
        return obj

    shap_tree.decode_ubjson_buffer = patched_decode


def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    return obj


def safe_filename(label: str) -> str:
    return label.replace(" ", "_").replace("/", "-")


def load_test_split() -> pd.DataFrame:
    model_df = load_model_data()
    splits = chronological_split(model_df)
    return splits["test"]


def make_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, label: str, out_path: Path):
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP summary — {label}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_bar_chart(feature_importance: list, label: str, out_path: Path):
    features, values = zip(*feature_importance)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(features[::-1], values[::-1])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Mean |SHAP| by feature — {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_dependence_plots(shap_values: np.ndarray, X: pd.DataFrame, top_features: list, label: str, safe_label: str):
    for feature in top_features:
        fig = plt.figure(figsize=(7, 5))
        shap.dependence_plot(feature, shap_values, X, show=False)
        plt.title(f"SHAP dependence: {feature} — {label}")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"shap_dependence_{safe_label}_{feature}.png", dpi=150)
        plt.close(fig)


def compute_permutation_importance(model, X: pd.DataFrame, y: pd.Series):
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=PERMUTATION_N_REPEATS,
        random_state=PERMUTATION_SEED,
    )
    return result.importances_mean, result.importances_std


def print_rank_comparison(label: str, feature_importance: list, perm_importance: list):
    shap_rank = {f: i + 1 for i, (f, _) in enumerate(feature_importance)}
    perm_rank = {f: i + 1 for i, (f, _, _) in enumerate(perm_importance)}

    print(f"\n{label} - SHAP rank vs permutation-importance rank:")
    print(f"  {'feature':<20} {'shap_rank':>10} {'perm_rank':>10} {'perm_mean':>12} {'perm_std':>10}")
    for feature, mean_v, std_v in perm_importance:
        print(f"  {feature:<20} {shap_rank[feature]:>10} {perm_rank[feature]:>10} "
              f"{mean_v:>12.4f} {std_v:>10.4f}")


def print_group_breakdown(label: str, method: str, values_by_feature: dict):
    # Note: permutation importance is signed (shuffling a feature can, by
    # chance, slightly improve the score, giving a negative value) and is not
    # guaranteed to sum the way non-negative mean |SHAP| does — these
    # fractions are computed the same way regardless, so a permutation-based
    # breakdown can include negative shares or not sum cleanly to 100%.
    total = sum(values_by_feature.values())
    reporting_sum = sum(values_by_feature[f] for f in REPORTING_HISTORY_FEATURES)
    climate_sum = sum(values_by_feature[f] for f in CLIMATE_FEATURES)
    seasonality_sum = sum(values_by_feature[f] for f in SEASONALITY_FEATURES)
    other_sum = sum(values_by_feature[f] for f in OTHER_FEATURES)

    print(f"{label} ({method}):")
    print(f"  reporting-history: {reporting_sum / total:.1%}")
    print(f"  climate: {climate_sum / total:.1%}")
    print(f"  seasonality: {seasonality_sum / total:.1%}")
    print(f"  other (state/disease/is_covid_period): {other_sum / total:.1%}")


def main():
    _patch_shap_xgboost_base_score()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    test_df = load_test_split()
    explainer = shap.TreeExplainer(model)

    summary = {}
    permutation_summary = {}

    for (state, disease), group in test_df.groupby(["state", "disease"], observed=True):
        label = combo_label(state, disease)
        safe_label = safe_filename(label)

        X = _xgb_frame(group, FULL_FEATURES)
        shap_values = explainer.shap_values(X)

        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_importance = sorted(zip(FULL_FEATURES, mean_abs), key=lambda t: -t[1])
        summary[label] = [
            {"feature": f, "mean_abs_shap": float(v)} for f, v in feature_importance
        ]

        make_beeswarm(shap_values, X, label, FIGURES_DIR / f"shap_beeswarm_{safe_label}.png")
        make_bar_chart(feature_importance, label, FIGURES_DIR / f"shap_bar_{safe_label}.png")

        top_features = [f for f, _ in feature_importance[:TOP_N_DEPENDENCE]]
        make_dependence_plots(shap_values, X, top_features, label, safe_label)

        mean_abs_by_feature = dict(zip(FULL_FEATURES, mean_abs))
        print_group_breakdown(label, "SHAP", mean_abs_by_feature)

        y = group[TARGET_COL].astype(int)
        perm_mean, perm_std = compute_permutation_importance(model, X, y)
        perm_importance = sorted(
            zip(FULL_FEATURES, perm_mean, perm_std), key=lambda t: -t[1]
        )
        permutation_summary[label] = [
            {
                "feature": f,
                "importance_mean": float(m),
                "importance_std": float(s),
                "rank": i + 1,
            }
            for i, (f, m, s) in enumerate(perm_importance)
        ]

        print_rank_comparison(label, feature_importance, perm_importance)

        perm_mean_by_feature = dict(zip(FULL_FEATURES, perm_mean))
        print_group_breakdown(label, "permutation importance", perm_mean_by_feature)
        print()

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(to_native(summary), f, indent=2, ensure_ascii=False)
    with open(PERMUTATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(to_native(permutation_summary), f, indent=2, ensure_ascii=False)
    print(f"Saved {METRICS_PATH}")
    print(f"Saved {PERMUTATION_METRICS_PATH}")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
