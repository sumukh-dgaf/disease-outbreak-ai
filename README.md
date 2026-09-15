# Disease Outbreak & Peak Forecasting — India

**Scope: India only.** District-level forecasting of dengue, malaria, and
influenza/ILI outbreaks.

This is a B.Tech research project exploring AI-based forecasting of disease
outbreaks and their peaks across Indian districts, using weekly (falling
back to monthly where necessary) epidemiological data. The project builds
up from classical time-series baselines (SARIMA) through gradient-boosted
trees (XGBoost) and deep learning (LSTM) to peak detection and SHAP-based
interpretability, with generalization tested across different
disease-state pairs (e.g. dengue in Karnataka vs. malaria in Goa), and
results surfaced through an interactive Streamlit dashboard.
