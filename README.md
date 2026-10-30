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

## Submission contents

For the CSET225 submission:

- **Code**: [`src/`](src/) (data pipeline, feature engineering, model
  training — XGBoost, logistic regression, SARIMA, stochastic gradient
  boosting), [`scripts/`](scripts/) (precompute/export utilities),
  [`frontend/`](frontend/) (the static results site).
- **Dataset with train/validation/test splits**:
  [`data/epiclim_Final_data.csv`](data/epiclim_Final_data.csv) (the
  original EpiClim source data) and [`data/splits/`](data/splits/) (21
  CSVs — train/validation/test per state-disease combination; see
  [`data/splits/README.md`](data/splits/README.md) for the split
  boundaries and column documentation).
- **Papers**: [`papers/`](papers/) —
  [`base_paper_ElMorr2024_scoping_review.pdf`](papers/base_paper_ElMorr2024_scoping_review.pdf),
  [`base_paper_Kaur2025_EpiClim.pdf`](papers/base_paper_Kaur2025_EpiClim.pdf),
  [`our_paper_IEEE.pdf`](papers/our_paper_IEEE.pdf).
- **Slides**: [`papers/presentation_slides.pptx`](papers/presentation_slides.pptx)
  (.pptx, not .pdf).
- **Live demo**: [https://disease-outbreak-ai-nine.vercel.app](https://disease-outbreak-ai-nine.vercel.app)

**Team**: Sumukh Gupta (S24CSEU0570), Aaditya Goel (S24CSEU0614), Kushika
Mishra (S24CSEU0652), Neya Tripathi (S24CSEU0609).
