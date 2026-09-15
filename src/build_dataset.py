"""Left-join disease reports with per-state weekly climate.

Joins data/processed/state_weekly.parquet to data/processed/state_climate.parquet
on (state, week_start_date). Every state_weekly row is expected to find a
matching climate row; an unmatched row would silently turn an otherwise
legitimate zero-report week into missing climate data, so the join asserts
zero unmatched rows and fails loudly instead of proceeding quietly.
"""
from pathlib import Path

import pandas as pd

STATE_WEEKLY_PATH = Path("data/processed/state_weekly.parquet")
STATE_CLIMATE_PATH = Path("data/processed/state_climate.parquet")
OUT_PATH = Path("data/processed/model_dataset.parquet")


def main():
    state_weekly = pd.read_parquet(STATE_WEEKLY_PATH)
    state_climate = pd.read_parquet(STATE_CLIMATE_PATH)

    # year/week are already carried by state_weekly and are redundant with
    # week_start_date; dropped here to avoid duplicate/suffixed columns.
    climate_cols = state_climate.drop(columns=["year", "week"])

    merged = state_weekly.merge(
        climate_cols, on=["state", "week_start_date"], how="left", indicator=True
    )

    unmatched = merged[merged["_merge"] == "left_only"]
    if len(unmatched) > 0:
        sample = unmatched[["state", "disease", "year", "week", "week_start_date"]].head(10)
        raise AssertionError(
            f"{len(unmatched)} state_weekly row(s) had no matching climate row on "
            f"(state, week_start_date). Examples:\n{sample.to_string(index=False)}"
        )
    merged = merged.drop(columns="_merge")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)

    print(f"Result shape: {merged.shape}")
    print("NaN counts per column:")
    for col, n in merged.isna().sum().items():
        print(f"  {col}: {n}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
