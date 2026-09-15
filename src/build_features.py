"""Build model-ready features from data/processed/model_dataset.parquet.

For each (state, disease), sorted chronologically by week_start_date, every
feature at row t is computed from week t or earlier only. The target is
`reported` shifted forward one week (i.e. "does next week have a report") —
the TARGET is shifted, never the features. An explicit assertion checks
that each row's target week is exactly 7 days after that row's own week,
which would fail if a sort/group bug ever let a feature see next-week (or
later) information.

cases_* features are built directly from total_cases, which is already NaN
on unreported weeks — that NaN is never filled, so a lag/rolling window
touching an unreported week stays NaN rather than being silently averaged
from fewer observations.

Saved to data/processed/features.parquet (no output path was specified in
the request; this follows the data/processed/ convention used by every
other build script in this pipeline).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_PATH = Path("data/processed/model_dataset.parquet")
OUT_PATH = Path("data/processed/features.parquet")

CLIMATE_VARS = ["rainfall", "temperature", "humidity"]
CLIMATE_LAGS = [0, 1, 2, 3, 4]
REPORTED_CASES_LAGS = [1, 2, 3, 4]
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-06-30")
WARMUP_ROWS = 8  # longest trailing window: reported_roll_8

TARGET_COL = "target_reported_next_week"

FEATURE_COLUMNS = (
    ["state", "disease"]
    + [f"reported_lag_{k}" for k in REPORTED_CASES_LAGS]
    + ["reported_roll_4", "reported_roll_8"]
    + [f"cases_lag_{k}" for k in REPORTED_CASES_LAGS]
    + ["cases_roll_4_mean"]
    + ["week_of_year", "week_sin_52", "week_cos_52", "week_sin_26", "week_cos_26"]
    + [f"{var}_lag_{lag}" for var in CLIMATE_VARS for lag in CLIMATE_LAGS]
    + ["is_covid_period"]
)


def build_group_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("week_start_date").reset_index(drop=True)

    for lag in REPORTED_CASES_LAGS:
        g[f"reported_lag_{lag}"] = g["reported"].shift(lag).astype("boolean")
        g[f"cases_lag_{lag}"] = g["total_cases"].shift(lag)

    reported_shifted = g["reported"].shift(1)
    g["reported_roll_4"] = reported_shifted.rolling(4).mean()
    g["reported_roll_8"] = reported_shifted.rolling(8).mean()

    cases_shifted = g["total_cases"].shift(1)
    g["cases_roll_4_mean"] = cases_shifted.rolling(4).mean()

    g["week_of_year"] = g["week"].astype(int)
    g["week_sin_52"] = np.sin(2 * np.pi * g["week_of_year"] / 52)
    g["week_cos_52"] = np.cos(2 * np.pi * g["week_of_year"] / 52)
    g["week_sin_26"] = np.sin(2 * np.pi * g["week_of_year"] / 26)
    g["week_cos_26"] = np.cos(2 * np.pi * g["week_of_year"] / 26)

    for var in CLIMATE_VARS:
        for lag in CLIMATE_LAGS:
            g[f"{var}_lag_{lag}"] = g[var].shift(lag)

    g["is_covid_period"] = g["week_start_date"].between(COVID_START, COVID_END)

    # Target: shift the target forward, not the features.
    g[TARGET_COL] = g["reported"].shift(-1).astype("boolean")
    g["_target_week_start_date"] = g["week_start_date"].shift(-1)

    return g


def assert_no_leakage(df: pd.DataFrame) -> None:
    gap = df["_target_week_start_date"] - df["week_start_date"]
    valid_gap = gap.dropna()
    bad = valid_gap[valid_gap != pd.Timedelta(days=7)]
    if not bad.empty:
        sample = df.loc[
            bad.index, ["state", "disease", "week_start_date", "_target_week_start_date"]
        ].head()
        raise AssertionError(
            f"{len(bad)} row(s) have a target week that is not exactly 7 days after "
            f"the feature row's own week — the target may include same-week or later "
            f"information. Sample:\n{sample.to_string(index=False)}"
        )


def main():
    df = pd.read_parquet(DATASET_PATH)
    df = df.sort_values(["state", "disease", "week_start_date"]).reset_index(drop=True)

    featured = pd.concat(
        [build_group_features(g) for _, g in df.groupby(["state", "disease"], observed=True)],
        ignore_index=True,
    )

    assert_no_leakage(featured)
    featured = featured.drop(columns="_target_week_start_date")

    featured["state"] = featured["state"].astype("category")
    featured["disease"] = featured["disease"].astype("category")

    print("Feature list:")
    for col in FEATURE_COLUMNS:
        print(f"  {col}")
    print(f"\nTarget: {TARGET_COL} (reported, shifted -1 week within each state-disease group)")

    featured["_row_in_group"] = featured.groupby(["state", "disease"], observed=True).cumcount()
    usable_mask = (featured["_row_in_group"] >= WARMUP_ROWS) & featured[TARGET_COL].notna()
    usable = featured[usable_mask]

    cases_cols = [f"cases_lag_{k}" for k in REPORTED_CASES_LAGS] + ["cases_roll_4_mean"]
    cases_complete = usable[cases_cols].notna().all(axis=1)

    print(f"\nRows usable after {WARMUP_ROWS}-week lag warm-up (target non-null): {len(usable)}")
    print(f"Of those, rows with fully non-null cases_* features (limited by reporting "
          f"density, not by warm-up): {int(cases_complete.sum())} ({cases_complete.mean():.1%})")

    featured = featured.drop(columns="_row_in_group")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(featured)} rows (including warm-up rows) to {OUT_PATH}")


if __name__ == "__main__":
    main()
