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

## Running

### Streamlit dashboard

```
streamlit run dashboard/app.py
```

### Liquid-glass interface (static 3-page site)

`frontend/` is a static site (Forecast / About / Comparison) that reads
precomputed data from `frontend/data/`. Regenerate it after any
model/metrics change:

```
python scripts/precompute.py
```

Then just open [frontend/index.html](frontend/index.html) directly in a
browser — no server needed. (The precompute step writes each JSON file
plus a `.js` sibling that assigns the same data to `window.__DATA__`,
loaded via `<script src>`, since browsers block `fetch()`/XHR of local
`file://` JSON entirely. The `.json` files stay as the readable/diffable
source of truth; the `.js` files are just how the static pages consume
them without a server.)

`backend/app.py` (the live FastAPI `/predict`, `/history`, `/shap` API
the frontend used before this restructure) is unchanged and still runs
standalone if you want to query the model live instead of via the
precomputed JSON:

```
uvicorn backend.app:app --reload
```
