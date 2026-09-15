"""Build a complete (state, disease, year, week) grid from data/raw/Final_data.csv.

Read-only against data/raw/.

Decisions applied (previously surfaced for review, now resolved):
- Duplicate (state, disease, year, week) source rows are SUMMED, not
  dropped. `n_reports` records how many source rows contributed.
- Climate columns (Temp, preci, LAI) from EpiClim are DROPPED entirely and
  never reach the output. Investigation found implausible values under the
  assumed Kelvin->Celsius conversion (e.g. Assam/Dhemaji, July 2021,
  converted to 1.76C; Assam/Karbi Anglong, Sep 2015, to -11.85C) that are
  not credible for those districts/seasons. Rather than silently trust or
  patch a broken column, it is excluded until the source data is verified.
- Scope is narrowed to Dengue and Malaria (after alias merging) in five
  states: Maharashtra, Tamil Nadu, Karnataka, Kerala, West Bengal.
- Weeks are real ISO calendar weeks (Mon-start, 52 or 53 per year depending
  on the year), not a fixed 50-week epidemiological year. `week_start_date`
  is the actual Monday date for each (year, week).
- The grid is left-joined onto the aggregates: `total_cases`/`total_deaths`
  are left as NaN (not 0) where `reported` is False, since "no report" and
  "zero cases reported" are different facts.
"""
import re
from datetime import date
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/Final_data.csv")
OUT_PATH = Path("data/processed/state_weekly.parquet")

DENGUE_ALIASES = ["Dengue", "Dengue Fever", "Suspected Dengue"]
MALARIA_ALIASES = ["Malaria", "Malaria (PV)"]
DISEASE_MAPPING = {alias: "Dengue" for alias in DENGUE_ALIASES}
DISEASE_MAPPING.update({alias: "Malaria" for alias in MALARIA_ALIASES})

FOCUS_DISEASES = ["Dengue", "Malaria"]
FOCUS_STATES = ["Maharashtra", "Tamil Nadu", "Karnataka", "Kerala", "West Bengal"]
GRID_YEARS = range(2009, 2023)  # 2009-2022 inclusive


def parse_week_num(week_of_outbreak: str):
    match = re.match(r"(\d+)", str(week_of_outbreak))
    return int(match.group(1)) if match else None


def load_and_filter_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["week_num"] = df["week_of_outbreak"].apply(parse_week_num)
    df["disease_norm"] = df["Disease"].map(lambda d: DISEASE_MAPPING.get(d, d))

    df = df[df["disease_norm"].isin(FOCUS_DISEASES) & df["state_ut"].isin(FOCUS_STATES)].copy()

    df["Cases_numeric"] = pd.to_numeric(df["Cases"], errors="coerce")
    bad_mask = df["Cases_numeric"].isna()
    if bad_mask.any():
        print(f"Excluding {bad_mask.sum()} row(s) with non-numeric Cases:")
        for _, row in df[bad_mask].iterrows():
            print(f"  {row['state_ut']} / {row['district']} | {row['disease_norm']} | "
                  f"{row['year']} week {row['week_num']} | Cases={row['Cases']!r}")
    return df[~bad_mask].copy()


def aggregate_source(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (state, disease, year, week), group in df.groupby(
        ["state_ut", "disease_norm", "year", "week_num"]
    ):
        records.append(
            {
                "state": state,
                "disease": disease,
                "year": int(year),
                "week": int(week),
                "n_reports": int(len(group)),
                "total_cases": group["Cases_numeric"].sum(min_count=1),
                "total_deaths": group["Deaths"].sum(min_count=1),
            }
        )
    return pd.DataFrame.from_records(records)


def iso_weeks_in_year(year: int) -> int:
    # Dec 28 always falls in the last ISO week of its year.
    return date(year, 12, 28).isocalendar()[1]


def build_complete_grid() -> pd.DataFrame:
    week_rows = []
    for year in GRID_YEARS:
        for week in range(1, iso_weeks_in_year(year) + 1):
            week_rows.append((year, week, date.fromisocalendar(year, week, 1)))
    year_week_df = pd.DataFrame(week_rows, columns=["year", "week", "week_start_date"])
    year_week_df["week_start_date"] = pd.to_datetime(year_week_df["week_start_date"])

    combos = pd.MultiIndex.from_product(
        [FOCUS_STATES, FOCUS_DISEASES], names=["state", "disease"]
    ).to_frame(index=False)

    return combos.merge(year_week_df, how="cross")


def main():
    df = load_and_filter_raw()
    agg = aggregate_source(df)
    grid = build_complete_grid()

    merged = grid.merge(agg, on=["state", "disease", "year", "week"], how="left", indicator=True)
    if merged["_merge"].eq("right_only").any():
        print("WARNING: some aggregated rows did not match the grid — check week numbering.")
    merged["reported"] = merged["_merge"] == "both"
    merged = merged.drop(columns="_merge")
    merged["n_reports"] = merged["n_reports"].fillna(0).astype(int)
    # total_cases and total_deaths are intentionally left as NaN where reported is False.

    if merged["n_reports"].sum() != len(df):
        print(f"WARNING: sum(n_reports)={merged['n_reports'].sum()} does not match "
              f"source row count {len(df)} — some source rows were lost in the join.")

    merged = merged[
        ["state", "disease", "year", "week", "week_start_date", "reported",
         "n_reports", "total_cases", "total_deaths"]
    ].sort_values(["state", "disease", "year", "week"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)

    print(f"\nFinal shape: {merged.shape}")
    print(f"Saved to {OUT_PATH}\n")

    print("Weeks reported out of total, per state-disease:")
    for (state, disease), sub in merged.groupby(["state", "disease"]):
        total = len(sub)
        reported = int(sub["reported"].sum())
        print(f"  {state} / {disease}: {reported}/{total} ({reported / total:.1%})")


if __name__ == "__main__":
    main()
