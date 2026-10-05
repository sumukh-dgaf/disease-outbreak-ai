"""Build the data pipeline for the 4 new state-disease combos screened VIABLE
in results/findings.md (new finding, STEP 1): West Bengal/ADD, Gujarat/ADD,
Odisha/ADD, Kerala/Dengue.

Reuses build_state_weekly.py, fetch_climate.py, build_dataset.py, and
build_features.py exactly as written — none of those files are modified.
Each stage script's module-level path/scope constants (FOCUS_STATES,
FOCUS_DISEASES, STATE_WEEKLY_PATH, DATASET_PATH, OUT_PATH, etc.) are
monkeypatched at call time to point at new `_extra`-suffixed output paths
before calling that module's own main()/fetch_states(), so every row of
actual logic (cleaning, aggregation, the climate join's zero-unmatched-rows
assertion, feature engineering, the leakage assertion) runs unchanged.
Nothing under data/raw/ or any existing data/processed/*.parquet is touched.

West Bengal and Kerala already have NASA POWER climate fetched in the
existing data/processed/state_climate.parquet (both already in
fetch_climate.STATE_CENTROIDS) — those rows are reused directly rather than
re-fetched. Only Gujarat and Odisha need a fresh fetch_states() call, using
fetch_climate.py's own fetch_states() function with a temporary
STATE_CENTROIDS override and its own `out_path` parameter (no monkeypatch
needed there — it already accepts a path argument).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset  # noqa: E402
import build_features  # noqa: E402
import build_state_weekly  # noqa: E402
import fetch_climate  # noqa: E402

NEW_COMBOS = [
    ("West Bengal", "Acute Diarrhoeal Disease"),
    ("Gujarat", "Acute Diarrhoeal Disease"),
    ("Odisha", "Acute Diarrhoeal Disease"),
    ("Kerala", "Dengue"),
]
NEW_STATES = sorted({s for s, _ in NEW_COMBOS})
NEW_DISEASES = sorted({d for _, d in NEW_COMBOS})

# Approximate state geographic centroids, same precision/convention as the
# existing entries in fetch_climate.STATE_CENTROIDS.
NEW_STATE_CENTROIDS = {
    "Gujarat": (22.26, 71.19),
    "Odisha": (20.95, 85.10),
}

STATE_WEEKLY_EXTRA = Path("data/processed/state_weekly_extra.parquet")
STATE_CLIMATE_EXTRA = Path("data/processed/state_climate_extra.parquet")
MODEL_DATASET_EXTRA = Path("data/processed/model_dataset_extra.parquet")
FEATURES_EXTRA = Path("data/processed/features_extra.parquet")

EXISTING_STATE_CLIMATE = Path("data/processed/state_climate.parquet")


def step1_state_weekly():
    print("=== 1/4: state_weekly_extra.parquet (reusing build_state_weekly.main()) ===")
    build_state_weekly.FOCUS_STATES = NEW_STATES
    build_state_weekly.FOCUS_DISEASES = NEW_DISEASES
    build_state_weekly.OUT_PATH = STATE_WEEKLY_EXTRA
    build_state_weekly.main()

    # main() cross-joins FOCUS_STATES x FOCUS_DISEASES (8 combos here); keep
    # only the 4 actually screened VIABLE, same as the original pipeline
    # narrowing its own 3-state x 2-disease cross join down to 3 kept combos.
    df = pd.read_parquet(STATE_WEEKLY_EXTRA)
    keep_mask = pd.Series(list(zip(df["state"], df["disease"])), index=df.index).isin(NEW_COMBOS)
    df = df[keep_mask].reset_index(drop=True)
    df.to_parquet(STATE_WEEKLY_EXTRA, index=False)
    print(f"  narrowed to the 4 viable combos: {len(df)} rows -> {STATE_WEEKLY_EXTRA}\n")


def step2_climate():
    print("=== 2/4: state_climate_extra.parquet ===")
    needs_fetch = {s: c for s, c in NEW_STATE_CENTROIDS.items() if s in NEW_STATES}
    print(f"  fetching fresh NASA POWER data for: {list(needs_fetch)}")
    original_centroids = fetch_climate.STATE_CENTROIDS
    fetch_climate.STATE_CENTROIDS = needs_fetch
    try:
        fresh = fetch_climate.fetch_states(out_path=Path("data/processed/_fetch_tmp_unused.parquet"))
    finally:
        fetch_climate.STATE_CENTROIDS = original_centroids

    reused_states = [s for s in NEW_STATES if s not in needs_fetch]
    print(f"  reusing already-fetched climate for: {reused_states}")
    existing = pd.read_parquet(EXISTING_STATE_CLIMATE)
    reused = existing[existing["state"].isin(reused_states)].copy()

    combined = pd.concat([fresh, reused], ignore_index=True)
    combined = combined.sort_values(["state", "year", "week"]).reset_index(drop=True)
    combined.to_parquet(STATE_CLIMATE_EXTRA, index=False)
    Path("data/processed/_fetch_tmp_unused.parquet").unlink(missing_ok=True)
    print(f"  {len(combined)} rows -> {STATE_CLIMATE_EXTRA}\n")


def step3_model_dataset():
    print("=== 3/4: model_dataset_extra.parquet (reusing build_dataset.main()) ===")
    build_dataset.STATE_WEEKLY_PATH = STATE_WEEKLY_EXTRA
    build_dataset.STATE_CLIMATE_PATH = STATE_CLIMATE_EXTRA
    build_dataset.OUT_PATH = MODEL_DATASET_EXTRA
    build_dataset.main()
    print()


def step4_features():
    print("=== 4/4: features_extra.parquet (reusing build_features.main()) ===")
    build_features.DATASET_PATH = MODEL_DATASET_EXTRA
    build_features.OUT_PATH = FEATURES_EXTRA
    build_features.main()
    print()


def main():
    step1_state_weekly()
    step2_climate()
    step3_model_dataset()
    step4_features()
    print("Done. New files:")
    for p in (STATE_WEEKLY_EXTRA, STATE_CLIMATE_EXTRA, MODEL_DATASET_EXTRA, FEATURES_EXTRA):
        print(f"  {p}")


if __name__ == "__main__":
    main()
