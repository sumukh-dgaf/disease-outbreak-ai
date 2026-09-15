"""Streamlit dashboard for the disease-reporting-propensity forecasts.

Data/model provenance:
- data/processed/features.parquet: historical reported/not-reported series
  and the feature matrix used for live predictions.
- models/xgboost_reporting_only_h{1..4}.joblib and
  models/logistic_regression_reporting_only_h{1..4}.joblib: the two models
  selectable in the sidebar, one pair per horizon (see
  src/train_baseline.py's run_horizon_experiment). Both are the
  reporting-history-only feature set — neither is the "+climate" variant.
- results/metrics/horizon_results.json: every number in the status banner
  and metrics panel (PR-AUC, its bootstrap CI, base rate, positive counts)
  is read directly from this file, not recomputed here. The CI-overlap
  caption under the metrics panel is the one derived computation this app
  does with those numbers (a pairwise overlap check, not a new metric).
- results/figures/shap_bar_*.png: SHAP was only computed for the main h=1
  "+climate" XGBoost model (src/explainability.py) — not for either
  reporting-only model, and not per horizon. Shown only when XGBoost
  (reporting-only) is selected, with a note that it doesn't track the
  current horizon/model selection; hidden entirely for logistic regression.
- results/findings.md: the Data limitations expander shows findings F2,
  F4, F5, F6 verbatim, parsed from this file, not restated or summarized.
  F13 and F14 are cited (not reproduced) in the SHAP caption and the
  CI-overlap caption respectively.

No "risk level" label is computed anywhere in this app.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from train_baseline import (  # noqa: E402
    KEEP_COMBOS,
    REPORTING_ONLY_FEATURES,
    _xgb_frame,
    build_design_matrix,
    build_horizon_data,
    chronological_split,
    combo_label,
)

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
HORIZON_RESULTS_PATH = ROOT / "results" / "metrics" / "horizon_results.json"
FINDINGS_PATH = ROOT / "results" / "findings.md"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "models"

HORIZONS = [1, 2, 3, 4]
COMBOS = [combo_label(state, disease) for state, disease in KEEP_COMBOS]
LIMITATION_FINDINGS = [2, 4, 5, 6]

MODEL_OPTIONS = {
    "XGBoost (reporting-only)": "xgboost_reporting_only",
    "Logistic regression (reporting-only)": "logistic_regression_reporting_only",
}

TRAIN_START, TRAIN_END = pd.Timestamp("2009-01-01"), pd.Timestamp("2017-12-31")
VAL_START, VAL_END = pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-31")
EXCLUDED_START, EXCLUDED_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
TEST_START, TEST_END = pd.Timestamp("2021-01-01"), pd.Timestamp("2022-06-29")


st.set_page_config(page_title="Reporting-propensity dashboard", layout="wide")


@st.cache_data
def load_horizon_results():
    with open(HORIZON_RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_features():
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_xgboost_model(h: int):
    return joblib.load(MODELS_DIR / f"xgboost_reporting_only_h{h}.joblib")


@st.cache_resource
def load_logistic_bundle(h: int):
    return joblib.load(MODELS_DIR / f"logistic_regression_reporting_only_h{h}.joblib")


def predict_proba(model_key: str, h: int, df: pd.DataFrame):
    if model_key == "xgboost_reporting_only":
        model = load_xgboost_model(h)
        X = _xgb_frame(df, REPORTING_ONLY_FEATURES)
        return model.predict_proba(X)[:, 1]
    bundle = load_logistic_bundle(h)
    X = build_design_matrix(df, REPORTING_ONLY_FEATURES, columns=bundle["columns"], scaler=bundle["scaler"])
    return bundle["model"].predict_proba(X)[:, 1]


def ci_overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    return not (a_hi < b_lo or b_hi < a_lo)


@st.cache_data
def get_splits(h: int):
    model_df, target_col = build_horizon_data(h)
    splits = chronological_split(model_df)
    return splits, target_col


@st.cache_data
def load_limitation_findings():
    text = FINDINGS_PATH.read_text(encoding="utf-8")
    blocks = text.split("\n---\n")
    sections = {}
    for block in blocks:
        stripped = block.strip()
        for n in LIMITATION_FINDINGS:
            if stripped.startswith(f"## F{n} "):
                sections[n] = stripped
    return sections


def safe_filename(label: str) -> str:
    return label.replace(" ", "_").replace("/", "-")


# --- Sidebar -----------------------------------------------------------------

st.sidebar.title("Selection")
combo = st.sidebar.selectbox("State / Disease", COMBOS)
model_label = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()))
horizon = st.sidebar.selectbox("Horizon (weeks ahead)", HORIZONS)
model_key = MODEL_OPTIONS[model_label]
state, disease = combo.split(" / ", 1)

st.title("Disease-reporting forecast dashboard")
st.caption(f"Model: {model_label}, reporting-history-only features (no climate). "
           "Target: will a report occur h weeks from now.")

horizon_results = load_horizon_results()
h_entry = horizon_results[str(horizon)]
test_metrics = h_entry[model_key]["test"][combo]
base_rate = h_entry["base_rate"]["test"][combo]

pr_auc = test_metrics["pr_auc"]
ci_lo = test_metrics["pr_auc_ci_lo"]
ci_hi = test_metrics["pr_auc_ci_hi"]
n_positive = test_metrics["n_positive"]
n_test = test_metrics["n"]

# --- Status banner -------------------------------------------------------

above_chance = (ci_lo is not None) and (ci_lo > base_rate)
ci_lo_display = f"{ci_lo:.3f}" if ci_lo is not None else "undefined"

if above_chance:
    st.success(
        f"**{combo}, horizon {horizon} week(s):** the model's lower 95% bootstrap CI bound "
        f"({ci_lo_display}) exceeds the test-period base rate ({base_rate:.3f})."
    )
else:
    st.error(
        f"⚠️ **{combo}, horizon {horizon} week(s):** the model's lower 95% bootstrap "
        f"CI bound ({ci_lo_display}) does **not** exceed the test-period base rate "
        f"({base_rate:.3f}). This model does not demonstrate performance above chance for "
        f"this selection. Predictions shown below should not be used for decision-making."
    )

# --- Historical reported/not-reported timeline --------------------------

st.subheader("Historical reporting timeline")

features_df = load_features()
history = features_df[
    (features_df["state"] == state) & (features_df["disease"] == disease)
].sort_values("week_start_date")

fig_hist, ax_hist = plt.subplots(figsize=(12, 2.5))
ax_hist.axvspan(TRAIN_START, TRAIN_END, color="tab:blue", alpha=0.12, label="train")
ax_hist.axvspan(VAL_START, VAL_END, color="tab:orange", alpha=0.12, label="validation")
ax_hist.axvspan(EXCLUDED_START, EXCLUDED_END, color="gray", alpha=0.25, label="2020 excluded")
ax_hist.axvspan(TEST_START, TEST_END, color="tab:green", alpha=0.12, label="test")
ax_hist.scatter(
    history["week_start_date"], history["reported"].astype(int),
    s=8, color="black", label="reported (1) / not (0)",
)
ax_hist.set_yticks([0, 1])
ax_hist.set_ylim(-0.2, 1.2)
ax_hist.set_xlabel("Week")
ax_hist.set_title(f"Reported vs. not reported, {combo}")
ax_hist.legend(loc="upper left", fontsize=7, ncol=5)
fig_hist.tight_layout()
st.pyplot(fig_hist)
plt.close(fig_hist)

# --- Predicted probability vs actual (test period) -----------------------

st.subheader(f"Predicted probability, test period ({model_label}, horizon = {horizon} week(s))")

splits, target_col = get_splits(horizon)
test_split = splits["test"]
test_combo = test_split[(test_split["state"] == state) & (test_split["disease"] == disease)].sort_values(
    "week_start_date"
)

predicted_proba = predict_proba(model_key, horizon, test_combo)
actual = test_combo[target_col].astype(int).to_numpy()

fig_pred, ax_pred = plt.subplots(figsize=(12, 3.5))
ax_pred.plot(test_combo["week_start_date"], predicted_proba, color="tab:blue", label="predicted probability")
ax_pred.axhline(base_rate, color="gray", linestyle="--", label="base rate (test)")
ax_pred.scatter(
    test_combo.loc[actual == 1, "week_start_date"],
    [1.05] * int(actual.sum()),
    marker="|", color="black", label="actual: reported",
)
ax_pred.scatter(
    test_combo.loc[actual == 0, "week_start_date"],
    [-0.05] * int((actual == 0).sum()),
    marker="|", color="lightgray", label="actual: not reported",
)
ax_pred.set_ylim(-0.15, 1.15)
ax_pred.set_xlabel("Week")
ax_pred.set_ylabel("Predicted probability")
ax_pred.set_title(f"{combo} — predicted probability of a report {horizon} week(s) ahead")
ax_pred.legend(loc="upper left", fontsize=7, ncol=4)
fig_pred.tight_layout()
st.pyplot(fig_pred)
plt.close(fig_pred)

st.caption(
    "Predicted probability is shown with the model's aggregate uncertainty in the metrics "
    "panel below (the PR-AUC confidence interval) rather than as a per-week error band, since "
    "no per-prediction uncertainty estimate is stored in results/metrics/."
)

# --- Metrics panel ---------------------------------------------------------

st.subheader("Metrics (test split)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("PR-AUC", f"{pr_auc:.3f}" if pr_auc is not None else "undefined")
col2.metric("PR-AUC 95% CI", f"[{ci_lo_display}, {ci_hi:.3f}]" if ci_hi is not None else "undefined")
col3.metric("Base rate", f"{base_rate:.3f}")
col4.metric("Positives / N", f"{n_positive} / {n_test}")

overlap_flags = []
for h in HORIZONS:
    xgb_m = horizon_results[str(h)]["xgboost_reporting_only"]["test"][combo]
    log_m = horizon_results[str(h)]["logistic_regression_reporting_only"]["test"][combo]
    if xgb_m["pr_auc_ci_lo"] is None or log_m["pr_auc_ci_lo"] is None:
        overlap_flags.append(None)
    else:
        overlap_flags.append(
            ci_overlap(xgb_m["pr_auc_ci_lo"], xgb_m["pr_auc_ci_hi"], log_m["pr_auc_ci_lo"], log_m["pr_auc_ci_hi"])
        )

if all(overlap_flags):
    st.caption(
        f"XGBoost and logistic regression PR-AUC confidence intervals overlap at every horizon "
        f"(h=1-4) for {combo} — computed from results/metrics/horizon_results.json. No model is "
        f"statistically distinguishable from another for this combo (see F14 in "
        f"results/findings.md)."
    )
else:
    overlapping_h = [h for h, flag in zip(HORIZONS, overlap_flags) if flag]
    st.caption(
        f"XGBoost and logistic regression PR-AUC confidence intervals overlap at horizons "
        f"{overlapping_h} of h=1-4 for {combo} — computed from results/metrics/horizon_results.json."
    )

# --- SHAP panel -------------------------------------------------------------

st.subheader("SHAP feature importance")
if model_key == "xgboost_reporting_only":
    st.caption(
        "From the main model (h=1, reporting-history + climate features) — SHAP was not "
        "recomputed per horizon or per reporting-only variant, so this panel does not change "
        "with the horizon selector. Permutation importance on this same model and test split "
        "did not corroborate these attributions — most features sit near zero and several flip "
        "sign across combos (see F13 in results/findings.md) — so this chart shows how the "
        "model distributes credit internally, not evidence of causal drivers."
    )
    shap_path = FIGURES_DIR / f"shap_bar_{safe_filename(combo)}.png"
    if shap_path.exists():
        st.image(str(shap_path))
    else:
        st.info(f"No SHAP figure found for {combo} at {shap_path}.")
else:
    st.info(
        "SHAP was computed for the XGBoost \"+climate\" model at h=1 only "
        "(see results/figures/shap_bar_*.png) — not for logistic regression. "
        "Select XGBoost (reporting-only) in the sidebar to view it."
    )

# --- Data limitations --------------------------------------------------------

with st.expander("Data limitations"):
    sections = load_limitation_findings()
    for n in LIMITATION_FINDINGS:
        if n in sections:
            st.markdown(sections[n])
            st.markdown("---")
        else:
            st.warning(f"F{n} not found in {FINDINGS_PATH}")
