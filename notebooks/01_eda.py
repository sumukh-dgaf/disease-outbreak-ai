"""EDA on data/processed/model_dataset.parquet.

Read-only against the processed dataset. Saves figures to results/figures/
and a summary JSON to results/metrics/eda_summary.json.

No smoothing (rolling means, interpolation, etc.) is applied anywhere in
this script. Every series plotted is the raw aggregate computed directly
from model_dataset.parquet.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATASET_PATH = Path("data/processed/model_dataset.parquet")
FIGURES_DIR = Path("results/figures")
METRICS_DIR = Path("results/metrics")

CLIMATE_VARS = ["rainfall", "temperature", "humidity"]
LAGS = [0, 1, 2, 3, 4]
ADMIN_JUMP_THRESHOLD = 0.15  # a chosen heuristic, not a fact — see docstring of detect_admin_jumps

STATE_COLORS = {"Karnataka": "tab:blue", "Maharashtra": "tab:orange", "West Bengal": "tab:green"}
DISEASE_LINESTYLE = {"Dengue": "-", "Acute Diarrhoeal Disease": "--"}


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


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    return df.sort_values(["state", "disease", "week_start_date"]).reset_index(drop=True)


# --- 1. Reporting rate by week-of-year --------------------------------------

def reporting_rate_by_week_of_year(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["state", "disease", "week"])["reported"]
        .mean()
        .reset_index(name="reporting_rate")
    )


def plot_reporting_rate_by_week(rate_by_week: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for (state, disease), sub in rate_by_week.groupby(["state", "disease"]):
        sub = sub.sort_values("week")
        ax.plot(
            sub["week"], sub["reporting_rate"],
            color=STATE_COLORS[state], linestyle=DISEASE_LINESTYLE[disease],
            label=combo_label(state, disease), marker="o", markersize=3,
        )
    ax.set_xlabel("ISO week of year")
    ax.set_ylabel("Reporting rate (fraction of years with a report)")
    ax.set_title("Reporting-rate seasonality by ISO week of year, per state-disease\n"
                  "(raw per-week average across years; not smoothed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reporting_rate_by_week_of_year.png", dpi=150)
    plt.close(fig)


# --- 2. Reporting rate by year -----------------------------------------------

def reporting_rate_by_year(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["state", "disease", "year"])["reported"]
        .mean()
        .reset_index(name="reporting_rate")
    )


def plot_reporting_rate_by_year(rate_by_year: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for (state, disease), sub in rate_by_year.groupby(["state", "disease"]):
        sub = sub.sort_values("year")
        ax.plot(
            sub["year"], sub["reporting_rate"],
            color=STATE_COLORS[state], linestyle=DISEASE_LINESTYLE[disease],
            label=combo_label(state, disease), marker="o", markersize=4,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Reporting rate (fraction of weeks with a report)")
    ax.set_title("Reporting-rate by year, per state-disease\n(raw yearly average; not smoothed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reporting_rate_by_year.png", dpi=150)
    plt.close(fig)


def detect_admin_jumps(rate_by_year: pd.DataFrame) -> list:
    """Flag candidate administrative step-change years: a year where BOTH
    diseases in the same state show a year-over-year reporting-rate jump of
    at least ADMIN_JUMP_THRESHOLD in the same direction. This is a heuristic
    threshold chosen for this report, not a validated fact — flagged
    candidates need human review, not automatic acceptance.
    """
    candidates = []
    for state in rate_by_year["state"].unique():
        state_sub = rate_by_year[rate_by_year["state"] == state]
        pivot = state_sub.pivot(index="year", columns="disease", values="reporting_rate").sort_index()
        deltas = pivot.diff()
        for year in deltas.index:
            row = deltas.loc[year].dropna()
            if len(row) < 2:
                continue
            if (row.abs() >= ADMIN_JUMP_THRESHOLD).all() and (np.sign(row) == np.sign(row.iloc[0])).all():
                candidates.append(
                    {
                        "state": state,
                        "year": int(year),
                        "deltas": row.to_dict(),
                    }
                )
    return candidates


# --- 3. Cases distribution among reported weeks ------------------------------

def plot_cases_histograms(df: pd.DataFrame) -> None:
    reported = df[df["reported"]]
    combos = sorted(reported[["state", "disease"]].drop_duplicates().itertuples(index=False))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (state, disease) in zip(axes.flat, combos):
        sub = reported[(reported["state"] == state) & (reported["disease"] == disease)]
        ax.hist(sub["total_cases"], bins=30, color=STATE_COLORS[state])
        ax.set_xscale("log")
        ax.set_xlabel("total_cases (log scale)")
        ax.set_ylabel("count of reported weeks")
        ax.set_title(combo_label(state, disease), fontsize=9)
    fig.suptitle("Distribution of total_cases among reported weeks only, per state-disease\n"
                 "(raw counts, log-scale x-axis; unreported weeks excluded, not imputed as 0)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cases_distribution_histograms.png", dpi=150)
    plt.close(fig)


def cases_summary_stats(df: pd.DataFrame) -> dict:
    reported = df[df["reported"]]
    out = {}
    for (state, disease), sub in reported.groupby(["state", "disease"]):
        out[combo_label(state, disease)] = {
            "n_reported_weeks": int(len(sub)),
            "min": float(sub["total_cases"].min()),
            "median": float(sub["total_cases"].median()),
            "mean": float(sub["total_cases"].mean()),
            "p90": float(sub["total_cases"].quantile(0.90)),
            "max": float(sub["total_cases"].max()),
        }
    return out


def plot_cases_median_p90_by_week(df: pd.DataFrame) -> None:
    reported = df[df["reported"]]
    combos = sorted(reported[["state", "disease"]].drop_duplicates().itertuples(index=False))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (state, disease) in zip(axes.flat, combos):
        sub = reported[(reported["state"] == state) & (reported["disease"] == disease)]
        stats = sub.groupby("week")["total_cases"].agg(median="median", p90=lambda s: s.quantile(0.90))
        stats = stats.sort_index()
        ax.plot(stats.index, stats["median"], label="median", color="black")
        ax.plot(stats.index, stats["p90"], label="p90", color="crimson", linestyle="--")
        ax.set_xlabel("ISO week of year")
        ax.set_ylabel("total_cases")
        ax.set_title(combo_label(state, disease), fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Median and p90 total_cases by ISO week of year, among reported weeks only\n"
                 "(raw per-week statistics; not smoothed)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cases_median_p90_by_week.png", dpi=150)
    plt.close(fig)


# --- 4. Climate seasonality ---------------------------------------------------

def climate_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    unique_climate = df.drop_duplicates(subset=["state", "week_start_date"])[
        ["state", "week"] + CLIMATE_VARS
    ]
    return unique_climate.groupby(["state", "week"])[CLIMATE_VARS].mean().reset_index()


def plot_climate_seasonality(seasonality: pd.DataFrame) -> None:
    units = {"rainfall": "mm/day (mean)", "temperature": "C", "humidity": "%"}
    for var in CLIMATE_VARS:
        fig, ax = plt.subplots(figsize=(10, 5))
        for state, sub in seasonality.groupby("state"):
            sub = sub.sort_values("week")
            ax.plot(sub["week"], sub[var], color=STATE_COLORS[state], label=state, marker="o", markersize=3)
        ax.set_xlabel("ISO week of year")
        ax.set_ylabel(f"{var} ({units[var]})")
        ax.set_title(f"{var.capitalize()} seasonality by ISO week of year, per state\n"
                     f"(raw weekly mean across years; not smoothed)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"climate_seasonality_{var}.png", dpi=150)
        plt.close(fig)


# --- 5. Lagged Spearman correlation: reporting vs climate ---------------------

def lagged_spearman(y: pd.Series, x: pd.Series, lag: int):
    x_lagged = x.shift(lag)
    mask = x_lagged.notna() & y.notna()
    if mask.sum() < 3:
        return None, None
    r, p = spearmanr(y[mask], x_lagged[mask])
    return float(r), float(p)


def lag_correlations(df: pd.DataFrame) -> dict:
    results = {}
    for (state, disease), sub in df.groupby(["state", "disease"]):
        sub = sub.sort_values("week_start_date")
        y = sub["reported"].astype(int)
        results[combo_label(state, disease)] = {}
        for var in CLIMATE_VARS:
            results[combo_label(state, disease)][var] = {}
            for lag in LAGS:
                r, p = lagged_spearman(y, sub[var], lag)
                results[combo_label(state, disease)][var][lag] = {"spearman_r": r, "p_value": p}
    return results


def plot_lag_correlation_heatmaps(correlations: dict) -> None:
    combos = list(correlations.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(wspace=0.6, hspace=0.35, left=0.06, right=0.88, top=0.85, bottom=0.08)
    for ax, combo in zip(axes.flat, combos):
        matrix = np.array(
            [[correlations[combo][var][lag]["spearman_r"] or np.nan for lag in LAGS] for var in CLIMATE_VARS]
        )
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(LAGS)))
        ax.set_xticklabels(LAGS)
        ax.set_yticks(range(len(CLIMATE_VARS)))
        ax.set_yticklabels(CLIMATE_VARS)
        ax.set_xlabel("Lag (weeks)")
        ax.set_title(combo, fontsize=9)
        for i in range(len(CLIMATE_VARS)):
            for j in range(len(LAGS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Spearman r")
    fig.suptitle("Spearman correlation of weekly `reported` (0/1) with lagged climate variables\n"
                 "per state-disease (lag = weeks the climate value precedes the reporting week)")
    fig.savefig(FIGURES_DIR / "lag_correlation_heatmaps.png", dpi=150)
    plt.close(fig)


# --- 6. Peak reporting week per state-disease --------------------------------

def peak_reporting_week(rate_by_week: pd.DataFrame) -> dict:
    out = {}
    for disease in ["Dengue", "Acute Diarrhoeal Disease"]:
        out[disease] = {}
        for state, sub in rate_by_week[rate_by_week["disease"] == disease].groupby("state"):
            top_row = sub.loc[sub["reporting_rate"].idxmax()]
            out[disease][state] = {
                "peak_week": int(top_row["week"]),
                "reporting_rate": float(top_row["reporting_rate"]),
            }
    return out


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    rate_by_week = reporting_rate_by_week_of_year(df)
    plot_reporting_rate_by_week(rate_by_week)

    rate_by_year = reporting_rate_by_year(df)
    plot_reporting_rate_by_year(rate_by_year)
    admin_jumps = detect_admin_jumps(rate_by_year)
    print(f"Candidate administrative step-change years (both diseases jump >= "
          f"{ADMIN_JUMP_THRESHOLD:.0%} in the same direction):")
    for c in admin_jumps:
        print(f"  {c['state']} / {c['year']}: {c['deltas']}")
    if not admin_jumps:
        print("  none found at this threshold")

    plot_cases_histograms(df)
    cases_stats = cases_summary_stats(df)
    plot_cases_median_p90_by_week(df)

    seasonality = climate_seasonality(df)
    plot_climate_seasonality(seasonality)

    correlations = lag_correlations(df)
    plot_lag_correlation_heatmaps(correlations)

    peak_weeks = peak_reporting_week(rate_by_week)
    print("\nPeak reporting week per state, by disease:")
    for disease, states in peak_weeks.items():
        for state, info in states.items():
            print(f"  {disease} / {state}: week {info['peak_week']} "
                  f"(reporting rate {info['reporting_rate']:.1%})")

    summary = {
        "reporting_rate_by_week_of_year": {
            combo_label(s, d): sub.set_index("week")["reporting_rate"].to_dict()
            for (s, d), sub in rate_by_week.groupby(["state", "disease"])
        },
        "reporting_rate_by_year": {
            combo_label(s, d): sub.set_index("year")["reporting_rate"].to_dict()
            for (s, d), sub in rate_by_year.groupby(["state", "disease"])
        },
        "admin_jump_candidates": admin_jumps,
        "admin_jump_threshold": ADMIN_JUMP_THRESHOLD,
        "cases_distribution_per_state_disease": cases_stats,
        "climate_seasonality_by_week": {
            state: sub.set_index("week")[CLIMATE_VARS].to_dict(orient="index")
            for state, sub in seasonality.groupby("state")
        },
        "lag_correlations_spearman": correlations,
        "peak_reporting_week": peak_weeks,
    }

    out_path = METRICS_DIR / "eda_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_native(summary), f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to {out_path}")
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
