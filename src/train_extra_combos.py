"""Train and evaluate all 4 models (XGBoost, logistic regression, SARIMA, SGB)
on the 4 new combos built by build_extra_combos.py: West Bengal/ADD,
Gujarat/ADD, Odisha/ADD, Kerala/Dengue.

Reuses train_baseline.py's, train_sarima.py's, and train_sgb.py's own
functions directly — none of those files are modified, no existing model
file is overwritten, no existing results JSON is touched. The only
module-level constants monkeypatched (same technique as
build_extra_combos.py) are the data-source paths and the combo list; every
actual fitting/evaluation function (fit_xgboost, fit_logistic, fit_sgb,
evaluate_model, compute_metrics, bootstrap_pr_auc_ci, chronological_split,
load_series, select_order, walk_forward_forecasts) runs with identical
settings/splits/horizons/leakage checks/bootstrap parameters (1000 samples,
seed 42) as the original 3-combo run.

New models are persisted under models/extra/ — a separate directory from
the existing models/*.joblib files — so there is no path collision with
anything the original pipeline wrote.

Output: results/metrics/extra_combos_results.json. Schema: one dict per
horizon ("1".."4") with base_rate plus all 4 models' train/validation/test
metrics (same field names as horizon_results.json/sgb_results.json), plus
a top-level sarima_order_selected (per combo (p,d,q), same meaning as
sarima_results.json's order_selected).
"""
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_baseline  # noqa: E402
import train_sarima  # noqa: E402
import train_sgb  # noqa: E402

NEW_COMBOS = [
    ("West Bengal", "Acute Diarrhoeal Disease"),
    ("Gujarat", "Acute Diarrhoeal Disease"),
    ("Odisha", "Acute Diarrhoeal Disease"),
    ("Kerala", "Dengue"),
]
COMBO_LABELS = [train_baseline.combo_label(s, d) for s, d in NEW_COMBOS]

FEATURES_EXTRA = Path("data/processed/features_extra.parquet")
MODEL_DATASET_EXTRA = Path("data/processed/model_dataset_extra.parquet")
OUT_PATH = Path("results/metrics/extra_combos_results.json")
MODELS_DIR = Path("models/extra")
HORIZONS = [1, 2, 3, 4]


def train_xgb_logit_sgb() -> dict:
    train_baseline.FEATURES_PATH = FEATURES_EXTRA
    train_baseline.KEEP_COMBOS = NEW_COMBOS
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    horizon_results = {}
    for h in HORIZONS:
        print(f"\n{'=' * 70}\nExtra combos, horizon h={h}\n{'=' * 70}")
        model_df, target_col = train_baseline.build_horizon_data(h)
        splits = train_baseline.chronological_split(model_df)

        base_rate = {}
        for split_name, sub in splits.items():
            base_rate[split_name] = {"overall": float(sub[target_col].mean())}
            for (state, disease), g in sub.groupby(["state", "disease"], observed=True):
                base_rate[split_name][train_baseline.combo_label(state, disease)] = float(g[target_col].mean())

        logit_model, logit_columns, logit_scaler = train_baseline.fit_logistic(
            splits["train"], train_baseline.REPORTING_ONLY_FEATURES, target_col=target_col
        )
        joblib.dump(
            {"model": logit_model, "columns": list(logit_columns), "scaler": logit_scaler},
            MODELS_DIR / f"logistic_regression_reporting_only_h{h}.joblib",
        )

        def logistic_fn(df, _m=logit_model, _c=logit_columns, _s=logit_scaler):
            return train_baseline.logistic_predict(_m, df, train_baseline.REPORTING_ONLY_FEATURES, _c, _s)

        logistic_results = train_baseline.evaluate_model(
            f"Logistic (extra combos, h={h})", splits, logistic_fn, target_col=target_col, verbose=True,
        )

        xgb_model = train_baseline.fit_xgboost(
            splits["train"], splits["validation"], train_baseline.REPORTING_ONLY_FEATURES, target_col=target_col
        )
        joblib.dump(xgb_model, MODELS_DIR / f"xgboost_reporting_only_h{h}.joblib")

        def xgboost_fn(df, _m=xgb_model):
            return train_baseline.xgboost_predict(_m, df, train_baseline.REPORTING_ONLY_FEATURES)

        xgboost_results = train_baseline.evaluate_model(
            f"XGBoost (extra combos, h={h})", splits, xgboost_fn, target_col=target_col, verbose=True,
        )

        sgb_model, sgb_columns = train_sgb.fit_sgb(splits["train"], train_baseline.REPORTING_ONLY_FEATURES, target_col)
        joblib.dump({"model": sgb_model, "columns": sgb_columns}, MODELS_DIR / f"sgb_reporting_only_h{h}.joblib")

        def sgb_fn(df, _m=sgb_model, _c=sgb_columns):
            return train_sgb.sgb_predict(_m, _c, df, train_baseline.REPORTING_ONLY_FEATURES)

        sgb_results = train_baseline.evaluate_model(
            f"SGB (extra combos, h={h})", splits, sgb_fn, target_col=target_col, verbose=True,
        )

        horizon_results[str(h)] = {
            "target_col": target_col,
            "base_rate": base_rate,
            "logistic_regression_reporting_only": logistic_results,
            "xgboost_reporting_only": xgboost_results,
            "sgb_reporting_only": sgb_results,
        }
    return horizon_results


def train_sarima_extra() -> tuple:
    """Mirrors train_sarima.py's main() loop structure exactly (that loop isn't
    factored into a separate reusable function in train_sarima.py itself), but
    calls out to its module-level functions (load_series, split_masks,
    select_order, walk_forward_forecasts, build_horizon_frame, evaluate_split)
    for every actual computation.
    """
    import pandas as pd

    train_sarima.DATASET_PATH = MODEL_DATASET_EXTRA

    per_combo = {}
    orders = {}
    for state, disease in NEW_COMBOS:
        label = train_baseline.combo_label(state, disease)
        print(f"\n=== SARIMA (extra): {label} ===")
        sub = train_sarima.load_series(state, disease)
        masks = train_sarima.split_masks(sub["week_start_date"])
        order, aic = train_sarima.select_order(
            sub.loc[masks["train"], "reported"],
            sub.loc[masks["train"], ["fourier_sin_52", "fourier_cos_52"]],
        )
        print(f"  selected order (p,d,q)={order}, AIC={aic:.2f}")
        orders[label] = list(order)
        forecasts = train_sarima.walk_forward_forecasts(sub, order, masks["train"])
        per_combo[label] = {"sub": sub, "masks": masks, "forecasts": forecasts}

    sarima_by_horizon = {}
    for h in HORIZONS:
        print(f"\n--- SARIMA (extra) horizon h={h} ---")
        base_rate = {"train": {}, "validation": {}, "test": {}}
        sarima_results = {"train": {}, "validation": {}, "test": {}}
        pooled = {"train": [], "validation": [], "test": []}

        for label, entry in per_combo.items():
            sub, masks, forecasts = entry["sub"], entry["masks"], entry["forecasts"]
            for split_name in ("train", "validation", "test"):
                frame = train_sarima.build_horizon_frame(sub, forecasts, h, masks[split_name])
                if split_name == "train" and frame.empty:
                    frame = train_sarima.build_horizon_frame(sub, forecasts, 1, masks["train"])
                pooled[split_name].append(frame)

                metrics = train_sarima.evaluate_split(frame, with_ci=(split_name == "test"))
                sarima_results[split_name][label] = metrics
                rate = float(frame["y_true"].mean())
                base_rate[split_name][label] = rate
                print(f"  [{split_name}] {label}: pr_auc={metrics['pr_auc']}, base_rate={rate:.3f}, n={metrics['n']}")

        for split_name in ("train", "validation", "test"):
            combined = pd.concat(pooled[split_name], ignore_index=True)
            sarima_results[split_name]["overall"] = train_sarima.evaluate_split(combined, with_ci=(split_name == "test"))
            base_rate[split_name]["overall"] = float(combined["y_true"].mean())

        sarima_by_horizon[str(h)] = {"base_rate": base_rate, "sarima": sarima_results}
    return sarima_by_horizon, orders


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Bootstrap: N={train_baseline.BOOTSTRAP_N}, seed={train_baseline.BOOTSTRAP_SEED}, CI={train_baseline.BOOTSTRAP_CI}")
    print(f"New combos: {COMBO_LABELS}\n")

    horizon_results = train_xgb_logit_sgb()
    sarima_by_horizon, sarima_orders = train_sarima_extra()

    for h in (str(x) for x in HORIZONS):
        horizon_results[h]["sarima"] = sarima_by_horizon[h]["sarima"]
        # sarima's own base_rate is computed from walk-forward-evaluable rows,
        # which can differ by a row or two from the classifiers' dropna-filtered
        # rows; both are kept (already verified elsewhere to match almost
        # exactly), so this does not overwrite the classifiers' base_rate.
        horizon_results[h]["base_rate_sarima"] = sarima_by_horizon[h]["base_rate"]

    output = dict(horizon_results)
    output["sarima_order_selected"] = sarima_orders

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(train_baseline.to_native(output), f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
