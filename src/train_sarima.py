"""SARIMA baseline for the same reporting-propensity targets used in
train_baseline.py's multi-horizon experiment (h = 1-4), added alongside
the existing models — nothing in train_baseline.py is modified.

Target series: the weekly `reported` indicator per (state, disease) from
data/processed/model_dataset.parquet (not features.parquet — SARIMA needs
the raw series, not the lag/rolling feature columns).

Same 3 combos and the same chronological split as train_baseline.py
(imported directly, not redefined): train 2009-2017, validation 2018-2019,
2020 kept in the series for continuity but excluded from evaluation, test
2021-01-01 to 2022-06-29.

Seasonal order 52 is unstable to estimate directly on weekly data (52 AR/MA
terms), so seasonality is instead carried by one Fourier pair (sin/cos at
period 52, computed from ISO week-of-year) passed as exogenous regressors —
purely calendar-based, so it carries no information about the target itself
and cannot leak.

(p, d, q) is chosen per combo by grid search over p, q in 0-2 and d in 0-1,
minimizing AIC on TRAIN data only. The model is then fit once on TRAIN with
that order, and forecasts are produced by walking forward week by week
through validation, 2020, and test using statsmodels' `append(refit=False)`
— which only updates the Kalman filter state with newly "known" actual
values, never re-estimates the ARIMA parameters. At each origin week t
(already-known, already appended), `get_forecast(steps=4)` gives the
h=1..4-week-ahead forecasts for weeks t+1..t+4, using only information
through week t — the same no-leakage property build_features.py's and
train_baseline.py's target-shift assertions enforce for the other models.
Forecast values are clipped to [0, 1] and used directly as the PR-AUC score.

ponytail: the "train" split's numbers reuse the initial in-sample
one-step-ahead fit (not a full walk-forward from an even-earlier origin),
since only test (and, incidentally, validation) walk-forward numbers were
asked for. Upgrade path: refit from a shorter initial window and walk
forward through all of train too, if train-split multi-horizon numbers are
ever needed.
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import (  # noqa: E402
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    KEEP_COMBOS,
    TEST_END,
    TEST_START,
    bootstrap_pr_auc_ci,
    combo_label,
    compute_metrics,
    to_native,
)

DATASET_PATH = Path("data/processed/model_dataset.parquet")
OUT_PATH = Path("results/metrics/sarima_results.json")
HORIZONS = [1, 2, 3, 4]
MAX_HORIZON = max(HORIZONS)

D_RANGE = [0, 1]
PQ_RANGE = [0, 1, 2]


def load_series(state: str, disease: str) -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    sub = df[(df["state"] == state) & (df["disease"] == disease)].copy()
    sub = sub.sort_values("week_start_date").reset_index(drop=True)
    gap = sub["week_start_date"].diff().dropna()
    bad = gap[gap != pd.Timedelta(days=7)]
    if not bad.empty:
        raise AssertionError(
            f"{combo_label(state, disease)}: {len(bad)} non-7-day gap(s) in the weekly "
            f"series — SARIMA needs an unbroken regular weekly grid."
        )
    sub["reported"] = sub["reported"].astype(float)
    sub["week_of_year"] = sub["week"].astype(int)
    sub["fourier_sin_52"] = np.sin(2 * np.pi * sub["week_of_year"] / 52)
    sub["fourier_cos_52"] = np.cos(2 * np.pi * sub["week_of_year"] / 52)
    return sub


def split_masks(dates: pd.Series) -> dict:
    years = dates.dt.year
    return {
        "train": (years >= 2009) & (years <= 2017),
        "validation": (years >= 2018) & (years <= 2019),
        "test": (dates >= TEST_START) & (dates <= TEST_END),
    }


def select_order(endog: pd.Series, exog: pd.DataFrame) -> tuple:
    best_aic, best_order = np.inf, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d, p, q in itertools.product(D_RANGE, PQ_RANGE, PQ_RANGE):
            try:
                res = SARIMAX(
                    endog, exog=exog, order=(p, d, q),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
            except Exception:
                continue
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic, best_order = res.aic, (p, d, q)
    if best_order is None:
        raise RuntimeError("No SARIMAX order converged during grid search.")
    return best_order, best_aic


def walk_forward_forecasts(sub: pd.DataFrame, order: tuple, train_mask: pd.Series) -> dict:
    """Fit once on TRAIN, then walk forward one week at a time (append,
    refit=False) through the rest of the series, recording h=1..4-step
    forecasts made from each origin. Returns {origin_idx: {h: forecast}}.
    """
    endog = sub["reported"]
    exog = sub[["fourier_sin_52", "fourier_cos_52"]]
    train_end = int(np.where(train_mask.values)[0].max()) + 1  # exclusive end

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = SARIMAX(
            endog.iloc[:train_end], exog=exog.iloc[:train_end], order=order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)

        forecasts = {}
        n = len(sub)
        k = train_end - 1  # last index already incorporated into `results`
        while k < n - 1:
            steps = min(MAX_HORIZON, n - 1 - k)
            fc = results.get_forecast(steps=steps, exog=exog.iloc[k + 1:k + 1 + steps])
            forecasts[k] = {h: float(fc.predicted_mean.iloc[h - 1]) for h in range(1, steps + 1)}
            k += 1
            results = results.append([endog.iloc[k]], exog=[exog.iloc[k].values], refit=False)
    return forecasts


def build_horizon_frame(sub: pd.DataFrame, forecasts: dict, horizon: int, mask: pd.Series) -> pd.DataFrame:
    rows = []
    for origin_idx in np.where(mask.values)[0]:
        target_idx = origin_idx + horizon
        if target_idx >= len(sub) or origin_idx not in forecasts or horizon not in forecasts[origin_idx]:
            continue
        rows.append({
            "origin_week": sub["week_start_date"].iloc[origin_idx],
            "y_true": sub["reported"].iloc[target_idx],
            "y_score": float(np.clip(forecasts[origin_idx][horizon], 0.0, 1.0)),
        })
    return pd.DataFrame(rows)


def evaluate_split(frame: pd.DataFrame, with_ci: bool) -> dict:
    y_true, y_score = frame["y_true"], frame["y_score"]
    y_pred = (y_score >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_score)
    if with_ci:
        metrics.update(bootstrap_pr_auc_ci(y_true, y_score))
    return metrics


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Bootstrap: N={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}, CI={BOOTSTRAP_CI}")

    per_combo = {}
    orders = {}
    latest_forecast = {}
    for state, disease in KEEP_COMBOS:
        label = combo_label(state, disease)
        print(f"\n=== SARIMA: {label} ===")
        sub = load_series(state, disease)
        masks = split_masks(sub["week_start_date"])

        order, aic = select_order(
            sub.loc[masks["train"], "reported"],
            sub.loc[masks["train"], ["fourier_sin_52", "fourier_cos_52"]],
        )
        print(f"  selected order (p,d,q)={order}, AIC={aic:.2f} (train-only grid search)")
        orders[label] = list(order)

        forecasts = walk_forward_forecasts(sub, order, masks["train"])
        per_combo[label] = {"sub": sub, "masks": masks, "forecasts": forecasts}

        # No SARIMA model artifact is persisted (unlike SGB/XGBoost/logistic),
        # so the frontend's "live" prediction reuses the most recent walk-forward
        # forecast available for each horizon — the same fixed train-fit model,
        # just the latest origin that still has an h-step-ahead forecast.
        latest_forecast[label] = {}
        for h in HORIZONS:
            origin = max(idx for idx, d in forecasts.items() if h in d)
            latest_forecast[label][str(h)] = float(np.clip(forecasts[origin][h], 0.0, 1.0))

    horizon_results = {}
    for h in HORIZONS:
        print(f"\n--- horizon h={h} ---")
        base_rate = {"train": {}, "validation": {}, "test": {}}
        sarima_results = {"train": {}, "validation": {}, "test": {}}

        pooled = {"train": [], "validation": [], "test": []}
        for label, entry in per_combo.items():
            sub, masks, forecasts = entry["sub"], entry["masks"], entry["forecasts"]
            for split_name in ("train", "validation", "test"):
                frame = build_horizon_frame(sub, forecasts, h, masks[split_name])
                if split_name == "train" and frame.empty:
                    # No walk-forward origins fall inside train (forecasting starts
                    # right after train ends) — fall back to the in-sample
                    # one-step-ahead fitted values as the train-split diagnostic.
                    frame = build_horizon_frame(sub, forecasts, 1, masks["train"])
                pooled[split_name].append(frame)

                metrics = evaluate_split(frame, with_ci=(split_name == "test"))
                sarima_results[split_name][label] = metrics
                rate = float(frame["y_true"].mean())
                base_rate[split_name][label] = rate
                print(f"  [{split_name}] {label}: pr_auc={metrics['pr_auc']}, base_rate={rate:.3f}, n={metrics['n']}")

        for split_name in ("train", "validation", "test"):
            combined = pd.concat(pooled[split_name], ignore_index=True)
            sarima_results[split_name]["overall"] = evaluate_split(combined, with_ci=(split_name == "test"))
            base_rate[split_name]["overall"] = float(combined["y_true"].mean())

        horizon_results[str(h)] = {
            "target_col": f"target_reported_h{h}",
            "base_rate": base_rate,
            "sarima": sarima_results,
        }

    # Top-level shape matches horizon_results.json ({h: {...}} keyed directly by
    # horizon) with one extra sibling key, order_selected, for the per-combo
    # (p,d,q) chosen by the train-only AIC search — not itself horizon-specific.
    output = dict(horizon_results)
    output["order_selected"] = orders
    output["latest_forecast"] = latest_forecast
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_native(output), f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
