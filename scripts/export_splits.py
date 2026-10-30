"""Export train/validation/test CSVs for the CSET225 submission's "dataset
with train/validation/test splits" requirement.

Read-only against existing processed data and existing code: reuses
train_baseline.py's build_horizon_data(1) (the exact function that already
produced horizon_results.json's/extra_combos_results.json's verified test
positive counts) and chronological_split(), unmodified — just points their
module-level FEATURES_PATH/KEEP_COMBOS at the two existing parquet files
(features.parquet for the original 3 combos, features_extra.parquet for the
4 added in F17) for this one read, then restores them. No retraining, no
writes to data/processed/ or data/raw/, no changes to any existing result.

Rows are never shuffled — build_horizon_data()'s own loader sorts by
week_start_date, and boolean-mask split filtering preserves that order;
each split is also explicitly re-sorted here as a defensive, zero-cost
double-check of that non-negotiable rule.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import train_baseline  # noqa: E402
from build_features import FEATURE_COLUMNS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data/processed/features.parquet"
FEATURES_EXTRA_PATH = ROOT / "data/processed/features_extra.parquet"
OUT_DIR = ROOT / "data/splits"

ORIGINAL_COMBOS = [
    ("Maharashtra", "Dengue"),
    ("Karnataka", "Acute Diarrhoeal Disease"),
    ("Maharashtra", "Acute Diarrhoeal Disease"),
]
EXTRA_COMBOS = [
    ("Kerala", "Dengue"),
    ("West Bengal", "Acute Diarrhoeal Disease"),
    ("Gujarat", "Acute Diarrhoeal Disease"),
    ("Odisha", "Acute Diarrhoeal Disease"),
]

HORIZON = 1
TARGET_COL = f"target_reported_h{HORIZON}"
ID_COLS = ["state", "disease", "week_start_date", "year", "week", "reported"]
EXPECTED_TEST_POSITIVES = {
    ("Maharashtra", "Dengue"): 26,
    ("Kerala", "Dengue"): 18,
}


def slug(state: str, disease: str) -> str:
    return f"{state}_{disease}".lower().replace(" ", "_").replace("/", "_")


def load_horizon_data(features_path: Path, combos: list) -> pd.DataFrame:
    """Reuses train_baseline.build_horizon_data(1) unmodified — only its
    module-level FEATURES_PATH/KEEP_COMBOS are pointed at the given file/combo
    list for this call, then restored, so this never leaves the module in a
    state that could affect anything else importing it in the same process.
    """
    orig_path, orig_combos = train_baseline.FEATURES_PATH, train_baseline.KEEP_COMBOS
    train_baseline.FEATURES_PATH = features_path
    train_baseline.KEEP_COMBOS = combos
    try:
        model_df, target_col = train_baseline.build_horizon_data(HORIZON)
    finally:
        train_baseline.FEATURES_PATH, train_baseline.KEEP_COMBOS = orig_path, orig_combos
    assert target_col == TARGET_COL
    return model_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original_df = load_horizon_data(FEATURES_PATH, ORIGINAL_COMBOS)
    extra_df = load_horizon_data(FEATURES_EXTRA_PATH, EXTRA_COMBOS)

    columns = ID_COLS + [c for c in FEATURE_COLUMNS if c not in ("state", "disease")] + [TARGET_COL]

    print(f"{'Combo':<42}{'Split':<12}{'Rows':<7}{'Positive':<10}{'Date range'}")
    print("-" * 110)

    assertion_failures = []
    for state, disease in ORIGINAL_COMBOS + EXTRA_COMBOS:
        source_df = original_df if (state, disease) in ORIGINAL_COMBOS else extra_df
        combo_df = source_df[(source_df.state == state) & (source_df.disease == disease)]
        splits = train_baseline.chronological_split(combo_df)

        base = slug(state, disease)
        for split_name, out_name in [("train", "train"), ("validation", "val"), ("test", "test")]:
            split_df = splits[split_name].sort_values("week_start_date")
            out_path = OUT_DIR / f"{base}_{out_name}.csv"
            split_df[columns].to_csv(out_path, index=False)

            n = len(split_df)
            n_pos = int(split_df[TARGET_COL].sum())
            date_range = (f"{split_df['week_start_date'].min().date()} to "
                          f"{split_df['week_start_date'].max().date()}" if n else "—")
            print(f"{state + ' / ' + disease:<42}{split_name:<12}{n:<7}{n_pos:<10}{date_range}")

            if split_name == "test" and (state, disease) in EXPECTED_TEST_POSITIVES:
                expected = EXPECTED_TEST_POSITIVES[(state, disease)]
                if n_pos != expected:
                    assertion_failures.append((state, disease, expected, n_pos))

    if assertion_failures:
        print("\nASSERTION FAILED — test positives do not match the project:")
        for state, disease, expected, actual in assertion_failures:
            print(f"  {state} / {disease}: expected {expected}, got {actual}")
        raise SystemExit(1)

    print(f"\nAll assertions passed. Wrote {7 * 3} CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
