"""Read-only profiling of data/raw/Final_data.csv.

Prints a human-readable summary and writes the same data as JSON to
results/metrics/epiclim_profile.json. Never modifies or writes into data/raw/.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/Final_data.csv")
OUT_PATH = Path("results/metrics/epiclim_profile.json")

DUP_KEY_COLS = ["year", "week_num", "district", "Disease"]
FOCUS_DISEASES = ["Dengue", "Malaria"]  # exact-label match only; see rows_per_disease
# for the full set of near-duplicate disease labels (e.g. "Dengue Fever",
# "Suspected Dengue", "Malaria (PV)") present in the raw data.


def to_native(obj):
    """Recursively convert numpy/pandas scalars to plain Python types for JSON."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def parse_week_num(week_of_outbreak: str):
    match = re.match(r"(\d+)", str(week_of_outbreak))
    return int(match.group(1)) if match else None


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["week_num"] = df["week_of_outbreak"].apply(parse_week_num)
    df["exact_date"] = pd.to_datetime(
        dict(year=df["year"], month=df["mon"], day=df["day"]), errors="coerce"
    )
    df["Cases_numeric"] = pd.to_numeric(df["Cases"], errors="coerce")
    return df


def profile_basic(df: pd.DataFrame) -> dict:
    return {
        "row_count": len(df),
        "column_count": df.shape[1] - 3,  # exclude the 3 derived helper columns
        "columns": [c for c in df.columns if c not in ("week_num", "exact_date", "Cases_numeric")],
        "dtypes": {
            c: str(df[c].dtype)
            for c in df.columns
            if c not in ("week_num", "exact_date", "Cases_numeric")
        },
        "date_range": {
            "min": df["exact_date"].min(),
            "max": df["exact_date"].max(),
            "unparseable_date_rows": int(df["exact_date"].isna().sum()),
        },
    }


def profile_disease_counts(df: pd.DataFrame) -> dict:
    rows_per_disease = df["Disease"].value_counts().to_dict()
    rows_per_disease_per_year = (
        df.groupby(["Disease", "year"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    return {
        "rows_per_disease": rows_per_disease,
        "rows_per_disease_per_year": rows_per_disease_per_year,
    }


def profile_geography(df: pd.DataFrame) -> dict:
    return {
        "unique_districts": int(df["district"].nunique()),
        "unique_states": int(df["state_ut"].nunique()),
        "unique_state_district_pairs": int(df[["state_ut", "district"]].drop_duplicates().shape[0]),
        "districts_per_disease": df.groupby("Disease")["district"].nunique().to_dict(),
    }


def profile_focus_disease(df: pd.DataFrame, disease: str) -> dict:
    sub = df[df["Disease"] == disease]
    rows_per_year = sub.groupby("year").size().to_dict()

    district_counts = sub["district"].value_counts()
    top_10 = district_counts.head(10).to_dict()

    result = {
        "row_count": int(len(sub)),
        "rows_per_year": rows_per_year,
        "top_10_districts": top_10,
    }

    if len(district_counts) > 0:
        top_district = district_counts.index[0]
        top_sub = sub[sub["district"] == top_district]
        year_week_pairs = sorted(set(zip(top_sub["year"], top_sub["week_num"])))
        result["top_district"] = {
            "name": top_district,
            "row_count": int(len(top_sub)),
            "unique_year_week_pairs": len(year_week_pairs),
            "year_week_pairs": [[int(y), int(w)] for y, w in year_week_pairs],
        }
    return result


def profile_cases_distribution(df: pd.DataFrame) -> dict:
    out = {}
    non_numeric_examples = df.loc[df["Cases_numeric"].isna(), "Cases"].unique().tolist()
    for disease, sub in df.groupby("Disease"):
        values = sub["Cases_numeric"].dropna()
        out[disease] = {
            "n": int(len(sub)),
            "non_numeric_excluded": int(sub["Cases_numeric"].isna().sum()),
            "min": values.min() if len(values) else None,
            "median": values.median() if len(values) else None,
            "mean": values.mean() if len(values) else None,
            "p90": values.quantile(0.90) if len(values) else None,
            "max": values.max() if len(values) else None,
        }
    out["_non_numeric_example_values_overall"] = non_numeric_examples
    return out


def profile_duplicates(df: pd.DataFrame) -> dict:
    dup_mask = df.duplicated(subset=DUP_KEY_COLS, keep=False)
    dup_rows = df[dup_mask]
    top_groups = (
        dup_rows.groupby(DUP_KEY_COLS).size().sort_values(ascending=False).head(10)
    )
    examples = [
        {"year": int(y), "week_num": int(w), "district": d, "Disease": dis, "row_count": int(cnt)}
        for (y, w, d, dis), cnt in top_groups.items()
    ]
    return {
        "key_columns": DUP_KEY_COLS,
        "duplicate_row_count": int(dup_mask.sum()),
        "duplicate_group_count": int(dup_rows.groupby(DUP_KEY_COLS).ngroups),
        "top_duplicate_groups": examples,
    }


def profile_missing(df: pd.DataFrame) -> dict:
    cols = [c for c in df.columns if c not in ("week_num", "exact_date", "Cases_numeric")]
    return df[cols].isna().sum().to_dict()


def profile_climate(df: pd.DataFrame) -> dict:
    def col_range(col):
        return {
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": df[col].mean(),
            "missing": int(df[col].isna().sum()),
        }

    temp_median = df["Temp"].median()
    unit_guess = "Kelvin" if temp_median > 100 else "Celsius"
    temp_stats = col_range("Temp")
    temp_stats["unit_guess"] = unit_guess
    temp_stats["median"] = temp_median
    if unit_guess == "Kelvin":
        temp_stats["celsius_equivalent_range"] = [
            float(df["Temp"].min() - 273.15),
            float(df["Temp"].max() - 273.15),
        ]

    return {
        "preci": col_range("preci"),
        "LAI": col_range("LAI"),
        "Temp": temp_stats,
    }


def build_profile(df: pd.DataFrame) -> dict:
    profile = {}
    profile.update({"basic": profile_basic(df)})
    profile.update({"disease_counts": profile_disease_counts(df)})
    profile.update({"geography": profile_geography(df)})
    profile["focus_diseases"] = {
        disease: profile_focus_disease(df, disease) for disease in FOCUS_DISEASES
    }
    profile["cases_distribution_per_disease"] = profile_cases_distribution(df)
    profile["duplicates"] = profile_duplicates(df)
    profile["missing_values_per_column"] = profile_missing(df)
    profile["climate_columns"] = profile_climate(df)
    return profile


def print_summary(profile: dict) -> None:
    basic = profile["basic"]
    print(f"Rows: {basic['row_count']}, Columns: {basic['column_count']}")
    print(f"Columns: {basic['columns']}")
    print(f"Dtypes: {basic['dtypes']}")
    print(f"Date range: {basic['date_range']['min']} to {basic['date_range']['max']} "
          f"(unparseable: {basic['date_range']['unparseable_date_rows']})")
    print()

    print("Rows per disease:")
    for disease, count in profile["disease_counts"]["rows_per_disease"].items():
        print(f"  {disease}: {count}")
    print()

    geo = profile["geography"]
    print(f"Unique districts: {geo['unique_districts']}")
    print(f"Unique states: {geo['unique_states']}")
    print(f"Unique (state, district) pairs: {geo['unique_state_district_pairs']}")
    print()

    for disease in FOCUS_DISEASES:
        fd = profile["focus_diseases"][disease]
        print(f"--- {disease} ---")
        print(f"  rows: {fd['row_count']}")
        print(f"  rows per year: {fd['rows_per_year']}")
        print(f"  top 10 districts: {fd['top_10_districts']}")
        if "top_district" in fd:
            td = fd["top_district"]
            print(f"  top district: {td['name']} ({td['row_count']} rows, "
                  f"{td['unique_year_week_pairs']} unique (year, week) pairs)")
        print()

    print("Duplicate check (year, week_num, district, Disease):")
    dup = profile["duplicates"]
    print(f"  duplicate rows: {dup['duplicate_row_count']} across {dup['duplicate_group_count']} groups")
    print()

    print("Missing values per column:")
    for col, n in profile["missing_values_per_column"].items():
        print(f"  {col}: {n}")
    print()

    print("Climate columns:")
    for col, stats in profile["climate_columns"].items():
        print(f"  {col}: {stats}")


def main():
    df = load_raw()
    profile = build_profile(df)
    print_summary(profile)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_native(profile), f, indent=2, ensure_ascii=False)
    print(f"\nSaved profile to {OUT_PATH}")


if __name__ == "__main__":
    main()
