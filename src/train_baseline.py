"""Baseline models for target_reported_next_week on data/processed/features.parquet.

Scope: Karnataka/Acute Diarrhoeal Disease, Maharashtra/Dengue, and
Maharashtra/Acute Diarrhoeal Disease only — the other three state-disease
combos (Karnataka/Dengue, West Bengal/ADD, West Bengal/Dengue) were dropped
after earlier reporting-density analysis showed them too sparse (in some
splits, single-digit positive weeks) to evaluate meaningfully.

Chronological split (never random, per CLAUDE.md): train 2009-2017,
validation 2018-2019, test 2021-01-01 to 2022-06-29. All of 2020 is
excluded entirely (COVID-era reporting disruption made it look like a
different data-generating process in earlier EDA) — the row count dropped
for this reason is printed explicitly, separate from the combo filtering.

cases_* columns and total_deaths are excluded from the model feature set —
this is a reporting-propensity model built from reporting history,
seasonality, climate, and a COVID-period flag, not a case-count model.

Four models, in increasing complexity:
1. Persistence: predict next week = this week's `reported`.
2. Seasonal: historical per-(state, disease, week-of-year) reporting rate,
   fit on TRAIN only, thresholded at 0.5.
3. Logistic regression (one-hot encoded state/disease, scaled numerics).
4. XGBoost: max_depth=3, n_estimators=200, learning_rate=0.05,
   min_child_weight=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight
   from train balance, early stopping on validation PR-AUC (aucpr). These
   are still not "tuned" in the sense of a search — they are fixed,
   regularization-leaning values chosen to counter the overfitting seen in
   the previous run (train PR-AUC of 1.0).

Experiment 1: for logistic regression and XGBoost, also fit and report a
reporting-history-only feature set (reporting lags/rolling rates, calendar
seasonality, COVID flag — no climate) alongside the full feature set
(+ climate lags), to see whether climate adds anything over reporting
history and seasonality alone.

PR-AUC is the headline metric because target_reported_next_week is
class-imbalanced (see the printed class balance) and ROC-AUC is overly
optimistic under imbalance.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from build_features import FEATURE_COLUMNS, TARGET_COL

FEATURES_PATH = Path("data/processed/features.parquet")
OUT_PATH = Path("results/metrics/baseline_results.json")
MODEL_PATH = Path("models/xgboost_with_climate.joblib")

KEEP_COMBOS = [
    ("Karnataka", "Acute Diarrhoeal Disease"),
    ("Maharashtra", "Dengue"),
    ("Maharashtra", "Acute Diarrhoeal Disease"),
]

CATEGORICAL_FEATURES = ["state", "disease"]
CLIMATE_VARS = ["rainfall", "temperature", "humidity"]

FULL_FEATURES = [c for c in FEATURE_COLUMNS if not c.startswith("cases_") and c != "total_deaths"]
CLIMATE_FEATURES = [c for c in FULL_FEATURES if c.split("_lag_")[0] in CLIMATE_VARS]
REPORTING_ONLY_FEATURES = [c for c in FULL_FEATURES if c not in CLIMATE_FEATURES]

TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2022-06-29")

XGB_PARAMS = dict(
    max_depth=3,
    n_estimators=200,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
)
XGB_EARLY_STOPPING_ROUNDS = 20  # not specified by the request; chosen as a reasonable default

HORIZONS = [1, 2, 3, 4]
HORIZON_OUT_PATH = Path("results/metrics/horizon_results.json")
HORIZON_FIGURE_PATH = Path("results/figures/horizon_degradation.png")


def to_native(obj):
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def combo_label(state, disease):
    return f"{state} / {disease}"


def _load_filtered_raw() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values(["state", "disease", "week_start_date"]).reset_index(drop=True)

    keep_mask = pd.Series(
        list(zip(df["state"], df["disease"])), index=df.index
    ).isin(KEEP_COMBOS)
    df = df[keep_mask].copy()
    df["state"] = df["state"].cat.remove_unused_categories()
    df["disease"] = df["disease"].cat.remove_unused_categories()
    return df


def load_model_data() -> pd.DataFrame:
    df = _load_filtered_raw()

    # Needed by the seasonal baseline: the week-of-year of the TARGET week,
    # built the same way the target itself was built (shift -1 per group).
    df["target_week_of_year"] = df.groupby(["state", "disease"], observed=True)["week_of_year"].shift(-1)

    model_df = df.dropna(subset=FULL_FEATURES + [TARGET_COL, "target_week_of_year"]).copy()
    model_df[TARGET_COL] = model_df[TARGET_COL].astype(bool)
    model_df["target_week_of_year"] = model_df["target_week_of_year"].astype(int)
    return model_df


def build_horizon_data(horizon: int) -> tuple:
    """Build a (state, disease)-grouped target at `horizon` weeks ahead:
    reported.shift(-horizon). Asserts the target week is exactly 7*horizon
    days after each row's own week_start_date — the multi-horizon analogue
    of build_features.py's single-horizon leakage assertion.
    """
    df = _load_filtered_raw()
    target_col = f"target_reported_h{horizon}"

    df[target_col] = df.groupby(["state", "disease"], observed=True)["reported"].shift(-horizon)
    df["_target_week_start_date"] = df.groupby(
        ["state", "disease"], observed=True
    )["week_start_date"].shift(-horizon)

    gap = df["_target_week_start_date"] - df["week_start_date"]
    valid_gap = gap.dropna()
    expected_gap = pd.Timedelta(days=7 * horizon)
    bad = valid_gap[valid_gap != expected_gap]
    if not bad.empty:
        sample = df.loc[
            bad.index, ["state", "disease", "week_start_date", "_target_week_start_date"]
        ].head()
        raise AssertionError(
            f"[horizon={horizon}] {len(bad)} row(s) have a target week that is not exactly "
            f"{7 * horizon} days after the feature row's own week — the target may include "
            f"same-week or later information. Sample:\n{sample.to_string(index=False)}"
        )
    df = df.drop(columns="_target_week_start_date")

    model_df = df.dropna(subset=REPORTING_ONLY_FEATURES + [target_col]).copy()
    model_df[target_col] = model_df[target_col].astype(bool)
    return model_df, target_col


def chronological_split(df: pd.DataFrame) -> dict:
    dropped_2020 = df[df["year"] == 2020]
    print(f"Rows dropped for falling in 2020 (excluded entirely): {len(dropped_2020)}")
    for (state, disease), g in dropped_2020.groupby(["state", "disease"], observed=True):
        print(f"  {combo_label(state, disease)}: {len(g)}")
    print()

    train = df[(df["year"] >= 2009) & (df["year"] <= 2017)].copy()
    validation = df[(df["year"] >= 2018) & (df["year"] <= 2019)].copy()
    test = df[(df["week_start_date"] >= TEST_START) & (df["week_start_date"] <= TEST_END)].copy()
    return {"train": train, "validation": validation, "test": test}


def print_class_balance(splits: dict) -> dict:
    balance = {}
    print("Class balance (fraction of target_reported_next_week == True):")
    for name, sub in splits.items():
        overall = sub[TARGET_COL].mean()
        print(f"  {name}: {overall:.1%} positive (n={len(sub)})")
        balance[name] = {"overall": {"positive_rate": float(overall), "n": int(len(sub))}}
        for (state, disease), g in sub.groupby(["state", "disease"], observed=True):
            rate = g[TARGET_COL].mean()
            print(f"    {combo_label(state, disease)}: {rate:.1%} positive (n={len(g)})")
            balance[name][combo_label(state, disease)] = {"positive_rate": float(rate), "n": int(len(g))}
    print()
    return balance


# --- Model 1: persistence -----------------------------------------------------

def persistence_fn(df: pd.DataFrame) -> tuple:
    pred = df["reported"].astype(int)
    return pred, pred.astype(float)


# --- Model 2: seasonal ---------------------------------------------------------

def fit_seasonal_baseline(train: pd.DataFrame) -> tuple:
    rate_table = train.groupby(["state", "disease", "week_of_year"], observed=True)["reported"].mean()
    fallback = train.groupby(["state", "disease"], observed=True)["reported"].mean()
    return rate_table, fallback


def seasonal_predict(df: pd.DataFrame, rate_table: pd.Series, fallback: pd.Series) -> pd.Series:
    scores = []
    n_fallback = 0
    for _, row in df.iterrows():
        key = (row["state"], row["disease"], row["target_week_of_year"])
        if key in rate_table.index:
            scores.append(rate_table.loc[key])
        else:
            n_fallback += 1
            scores.append(fallback.loc[(row["state"], row["disease"])])
    if n_fallback:
        print(f"  seasonal baseline: {n_fallback} row(s) had no train-set rate for their "
              f"(state, disease, week-of-year) — used the state-disease overall train rate instead.")
    return pd.Series(scores, index=df.index)


# --- Model 3: logistic regression ----------------------------------------------

def build_design_matrix(df: pd.DataFrame, feature_list: list, columns=None, scaler: StandardScaler = None):
    numeric_features = [c for c in feature_list if c not in CATEGORICAL_FEATURES]
    numeric = df[numeric_features].astype(float)
    if scaler is not None:
        numeric = pd.DataFrame(
            scaler.transform(numeric), columns=numeric.columns, index=numeric.index
        )
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES)
    X = pd.concat([numeric, dummies], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def fit_logistic(train: pd.DataFrame, feature_list: list, target_col: str = TARGET_COL):
    numeric_features = [c for c in feature_list if c not in CATEGORICAL_FEATURES]
    # Standard feature scaling for the solver's convergence — not hyperparameter
    # tuning, just putting features on comparable scales before an L2 solver.
    scaler = StandardScaler().fit(train[numeric_features].astype(float))
    X_train = build_design_matrix(train, feature_list, scaler=scaler)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train[target_col])
    return model, X_train.columns, scaler


def logistic_predict(model, df: pd.DataFrame, feature_list: list, columns, scaler: StandardScaler) -> tuple:
    X = build_design_matrix(df, feature_list, columns=columns, scaler=scaler)
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pd.Series(pred, index=df.index), pd.Series(proba, index=df.index)


# --- Model 4: XGBoost ----------------------------------------------------------

def _xgb_frame(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    X = df[feature_list].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def fit_xgboost(train: pd.DataFrame, validation: pd.DataFrame, feature_list: list, target_col: str = TARGET_COL):
    X_train = _xgb_frame(train, feature_list)
    y_train = train[target_col].astype(int)
    X_val = _xgb_frame(validation, feature_list)
    y_val = validation[target_col].astype(int)

    n_pos = int(y_train.sum())
    n_neg = int((~y_train.astype(bool)).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  XGBoost scale_pos_weight (train n_neg/n_pos = {n_neg}/{n_pos}): {scale_pos_weight:.3f}")

    model = XGBClassifier(
        **XGB_PARAMS,
        enable_categorical=True,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  best iteration (early stopping on validation aucpr): {model.best_iteration}")
    return model


def xgboost_predict(model, df: pd.DataFrame, feature_list: list) -> tuple:
    X = _xgb_frame(df, feature_list)
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pd.Series(pred, index=df.index), pd.Series(proba, index=df.index)


# --- Evaluation ------------------------------------------------------------

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI = (2.5, 97.5)


def bootstrap_pr_auc_ci(y_true: pd.Series, y_score: pd.Series) -> dict:
    """Bootstrap CI for PR-AUC: resample (y_true, y_score) pairs with
    replacement BOOTSTRAP_N times, recompute PR-AUC each time, and take the
    2.5th/97.5th percentiles. Resamples that land on a single class have an
    undefined PR-AUC and are skipped rather than assigned an invented value.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    n = len(y_true_arr)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    boot_scores = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_scores.append(average_precision_score(yt, y_score_arr[idx]))

    if not boot_scores:
        return {"pr_auc_ci_lo": None, "pr_auc_ci_hi": None, "pr_auc_ci_n_valid_resamples": 0}

    lo, hi = np.percentile(boot_scores, BOOTSTRAP_CI)
    return {
        "pr_auc_ci_lo": float(lo),
        "pr_auc_ci_hi": float(hi),
        "pr_auc_ci_n_valid_resamples": len(boot_scores),
    }


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    if y_true.nunique() < 2:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["note"] = "only one class present in y_true; AUC undefined"
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics


def evaluate_model(name: str, splits: dict, predict_fn, target_col: str = TARGET_COL, verbose: bool = True) -> dict:
    if verbose:
        print(f"=== {name} ===")
    results = {}
    for split_name, df in splits.items():
        y_pred, y_score = predict_fn(df)
        results[split_name] = {"overall": compute_metrics(df[target_col], y_pred, y_score)}
        for (state, disease), g in df.groupby(["state", "disease"], observed=True):
            idx = g.index
            metrics = compute_metrics(g[target_col], y_pred.loc[idx], y_score.loc[idx])
            if split_name == "test":
                metrics.update(bootstrap_pr_auc_ci(g[target_col], y_score.loc[idx]))
            results[split_name][combo_label(state, disease)] = metrics
        if verbose:
            overall = results[split_name]["overall"]
            print(f"  {split_name}: accuracy={overall['accuracy']:.3f} "
                  f"pr_auc={overall['pr_auc'] if overall['pr_auc'] is not None else float('nan'):.3f} "
                  f"roc_auc={overall['roc_auc'] if overall['roc_auc'] is not None else float('nan'):.3f}")
    if verbose:
        print()
    return results


def run_horizon_experiment() -> dict:
    """For h in HORIZONS, rebuild the target at h weeks ahead and refit
    logistic_regression+reporting_only and xgboost+reporting_only on the
    same splits/seed as the h=1 pipeline above.
    """
    horizon_results = {}
    for h in HORIZONS:
        print(f"\n{'=' * 70}\nHorizon h={h} (target = reported shifted -{h} week(s))\n{'=' * 70}")
        model_df, target_col = build_horizon_data(h)
        splits = chronological_split(model_df)

        base_rate = {}
        print(f"Base rate (fraction positive) at h={h}:")
        for split_name, sub in splits.items():
            rate = sub[target_col].mean()
            base_rate[split_name] = {"overall": float(rate)}
            print(f"  {split_name}: {rate:.1%} (n={len(sub)})")
            for (state, disease), g in sub.groupby(["state", "disease"], observed=True):
                combo_rate = g[target_col].mean()
                base_rate[split_name][combo_label(state, disease)] = float(combo_rate)

        logit_model, logit_columns, logit_scaler = fit_logistic(
            splits["train"], REPORTING_ONLY_FEATURES, target_col=target_col
        )

        logit_horizon_path = Path(f"models/logistic_regression_reporting_only_h{h}.joblib")
        logit_horizon_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": logit_model, "columns": list(logit_columns), "scaler": logit_scaler},
            logit_horizon_path,
        )
        print(f"  saved trained model to {logit_horizon_path}")

        def logistic_fn(df):
            return logistic_predict(logit_model, df, REPORTING_ONLY_FEATURES, logit_columns, logit_scaler)

        logistic_results = evaluate_model(
            f"Logistic regression (reporting-only, h={h})", splits, logistic_fn,
            target_col=target_col, verbose=True,
        )

        xgb_model = fit_xgboost(splits["train"], splits["validation"], REPORTING_ONLY_FEATURES, target_col=target_col)

        xgb_horizon_path = Path(f"models/xgboost_reporting_only_h{h}.joblib")
        xgb_horizon_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(xgb_model, xgb_horizon_path)
        print(f"  saved trained model to {xgb_horizon_path}")

        def xgboost_fn(df):
            return xgboost_predict(xgb_model, df, REPORTING_ONLY_FEATURES)

        xgboost_results = evaluate_model(
            f"XGBoost (reporting-only, h={h})", splits, xgboost_fn,
            target_col=target_col, verbose=True,
        )

        horizon_results[str(h)] = {
            "target_col": target_col,
            "base_rate": base_rate,
            "logistic_regression_reporting_only": logistic_results,
            "xgboost_reporting_only": xgboost_results,
        }
    return horizon_results


def plot_horizon_degradation(horizon_results: dict, combos: list) -> None:
    HORIZON_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 5), sharey=True)
    if len(combos) == 1:
        axes = [axes]

    model_keys = [
        ("logistic_regression_reporting_only", "Logistic regression (reporting-only)", "tab:blue"),
        ("xgboost_reporting_only", "XGBoost (reporting-only)", "tab:orange"),
    ]

    for ax, combo in zip(axes, combos):
        for model_key, model_label, color in model_keys:
            pr = []
            lo = []
            hi = []
            for h in HORIZONS:
                entry = horizon_results[str(h)][model_key]["test"][combo]
                pr.append(entry["pr_auc"])
                lo.append(entry["pr_auc_ci_lo"])
                hi.append(entry["pr_auc_ci_hi"])
            ax.plot(HORIZONS, pr, marker="o", color=color, label=model_label)
            ax.fill_between(HORIZONS, lo, hi, color=color, alpha=0.2)

        base_rates = [horizon_results[str(h)]["base_rate"]["test"][combo] for h in HORIZONS]
        ax.plot(HORIZONS, base_rates, linestyle="--", color="gray", label="base rate (test)")

        ax.set_xlabel("Horizon h (weeks ahead)")
        ax.set_xticks(HORIZONS)
        ax.set_title(combo, fontsize=10)

    axes[0].set_ylabel("Test PR-AUC")
    axes[0].legend(fontsize=8)
    fig.suptitle("Test PR-AUC vs. forecast horizon, reporting-history-only features\n"
                 "(shaded = 1000-sample bootstrap 95% CI; dashed = test-split base rate)")
    fig.tight_layout()
    fig.savefig(HORIZON_FIGURE_PATH, dpi=150)
    plt.close(fig)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_df = load_model_data()
    splits = chronological_split(model_df)
    class_balance = print_class_balance(splits)

    all_results = {"class_balance": class_balance, "models": {}}

    # 1. Persistence
    all_results["models"]["persistence"] = evaluate_model("Persistence baseline", splits, persistence_fn)

    # 2. Seasonal
    rate_table, fallback = fit_seasonal_baseline(splits["train"])

    def seasonal_fn(df):
        score = seasonal_predict(df, rate_table, fallback)
        pred = (score >= 0.5).astype(int)
        return pred, score

    all_results["models"]["seasonal"] = evaluate_model("Seasonal baseline", splits, seasonal_fn)

    # 3. Logistic regression (full feature set, incl. climate)
    logit_model, logit_columns, logit_scaler = fit_logistic(splits["train"], FULL_FEATURES)

    def logistic_fn(df):
        return logistic_predict(logit_model, df, FULL_FEATURES, logit_columns, logit_scaler)

    logistic_full_results = evaluate_model(
        "Logistic regression (+ climate)", splits, logistic_fn, verbose=True
    )
    all_results["models"]["logistic_regression"] = logistic_full_results

    # 4. XGBoost (full feature set, incl. climate)
    xgb_model = fit_xgboost(splits["train"], splits["validation"], FULL_FEATURES)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, MODEL_PATH)
    print(f"  saved trained model to {MODEL_PATH}")

    def xgboost_fn(df):
        return xgboost_predict(xgb_model, df, FULL_FEATURES)

    xgboost_full_results = evaluate_model("XGBoost (+ climate)", splits, xgboost_fn, verbose=True)
    all_results["models"]["xgboost"] = xgboost_full_results

    # --- Experiment 1: reporting-history only vs + climate ---
    print("### Experiment 1: reporting-history only vs + climate ###\n")

    logit_ro_model, logit_ro_columns, logit_ro_scaler = fit_logistic(splits["train"], REPORTING_ONLY_FEATURES)

    def logistic_reporting_only_fn(df):
        return logistic_predict(logit_ro_model, df, REPORTING_ONLY_FEATURES, logit_ro_columns, logit_ro_scaler)

    logistic_reporting_only_results = evaluate_model(
        "Logistic regression (reporting-only)", splits, logistic_reporting_only_fn, verbose=True
    )

    xgb_ro_model = fit_xgboost(splits["train"], splits["validation"], REPORTING_ONLY_FEATURES)

    def xgboost_reporting_only_fn(df):
        return xgboost_predict(xgb_ro_model, df, REPORTING_ONLY_FEATURES)

    xgboost_reporting_only_results = evaluate_model(
        "XGBoost (reporting-only)", splits, xgboost_reporting_only_fn, verbose=True
    )

    all_results["experiment_1_climate_ablation"] = {
        "logistic_regression": {
            "reporting_only": logistic_reporting_only_results,
            "with_climate": logistic_full_results,
        },
        "xgboost": {
            "reporting_only": xgboost_reporting_only_results,
            "with_climate": xgboost_full_results,
        },
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_native(all_results), f, indent=2, ensure_ascii=False)
    print(f"Saved results to {OUT_PATH}")

    # --- Multi-horizon experiment (h = 1, 2, 3, 4) ---
    horizon_results = run_horizon_experiment()

    HORIZON_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HORIZON_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_native(horizon_results), f, indent=2, ensure_ascii=False)
    print(f"Saved results to {HORIZON_OUT_PATH}")

    combo_labels = [combo_label(state, disease) for state, disease in KEEP_COMBOS]
    plot_horizon_degradation(horizon_results, combo_labels)
    print(f"Saved figure to {HORIZON_FIGURE_PATH}")


if __name__ == "__main__":
    main()
