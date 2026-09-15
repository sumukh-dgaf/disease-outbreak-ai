"""Fetch daily climate data from NASA POWER and aggregate to weekly (ISO, Mon-start)."""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M"]
POWER_COMMUNITY = "AG"
POWER_MISSING_VALUE = -999

# India bounding box (approximate), used to validate requested coordinates.
INDIA_LAT_RANGE = (6.5, 37.5)
INDIA_LON_RANGE = (68.0, 97.5)

# Indian district centroids for the study regions, keyed "State/District".
DISTRICTS = {
    "Goa/North Goa": (15.60, 73.90),
    "Goa/South Goa": (15.20, 74.05),
    "Karnataka/Bengaluru Urban": (12.97, 77.59),
    "Delhi/Delhi": (28.61, 77.21),
}

PROCESSED_DIR = Path("data/processed")


class NasaPowerAPIError(RuntimeError):
    """Raised when the NASA POWER API returns an error or an unusable response."""


def validate_india_bbox(lat: float, lon: float) -> None:
    lat_min, lat_max = INDIA_LAT_RANGE
    lon_min, lon_max = INDIA_LON_RANGE
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        raise ValueError(
            f"Coordinates ({lat}, {lon}) fall outside India's bounding box "
            f"(lat {lat_min}-{lat_max}, lon {lon_min}-{lon_max})."
        )


def fetch_daily_climate(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch raw daily climate data from NASA POWER for a single point.

    Returns a DataFrame indexed by date with columns matching POWER_PARAMETERS,
    with the API's -999 missing-value sentinel converted to NaN.
    """
    validate_india_bbox(lat, lon)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if start > end:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}.")

    params = {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": POWER_COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }

    logger.info("Requesting NASA POWER data for (%s, %s) from %s to %s", lat, lon, start_date, end_date)
    try:
        response = requests.get(POWER_BASE_URL, params=params, timeout=60)
    except requests.exceptions.RequestException as exc:
        raise NasaPowerAPIError(f"Network error while calling NASA POWER API: {exc}") from exc

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        raise NasaPowerAPIError(
            f"NASA POWER API rate limit exceeded (HTTP 429). Retry-After: {retry_after}."
        )
    if not response.ok:
        raise NasaPowerAPIError(
            f"NASA POWER API request failed with HTTP {response.status_code}: {response.text[:500]}"
        )

    try:
        payload = response.json()
        daily_params = payload["properties"]["parameter"]
    except (ValueError, KeyError) as exc:
        raise NasaPowerAPIError(f"Unexpected NASA POWER API response shape: {exc}") from exc

    df = pd.DataFrame({name: pd.Series(values) for name, values in daily_params.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.sort_index()
    df = df.replace(POWER_MISSING_VALUE, np.nan)
    return df


def aggregate_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily climate data to ISO weeks (Monday start).

    A week's value is computed only from the days actually present for that
    week; NaN days are excluded rather than averaged in, and gaps are never
    filled. `n_days` records how many calendar days contributed to each week
    so partial weeks (e.g. at the start/end of the requested range) stay visible.
    """
    iso = daily_df.index.isocalendar()
    grouped = daily_df.groupby([iso["year"], iso["week"]])

    records = []
    for (iso_year, iso_week), group in grouped:
        first_day = group.index.min()
        week_start = first_day - timedelta(days=first_day.isocalendar().weekday - 1)
        records.append(
            {
                "week_start": week_start,
                "year": int(iso_year),
                "week_of_year": int(iso_week),
                "temperature": group["T2M"].mean(),
                "temp_max": group["T2M_MAX"].mean(),
                "rainfall": group["PRECTOTCORR"].sum(min_count=1),
                "humidity": group["RH2M"].mean(),
                "n_days": int(group.shape[0]),
            }
        )

    weekly_df = pd.DataFrame.from_records(records)
    weekly_df = weekly_df.sort_values("week_start").reset_index(drop=True)
    return weekly_df


def fetch_and_aggregate(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    daily_df = fetch_daily_climate(lat, lon, start_date, end_date)
    return aggregate_to_weekly(daily_df)


def save_weekly_climate(weekly_df: pd.DataFrame, lat: float, lon: float) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"climate_{lat:.2f}_{lon:.2f}.parquet"
    weekly_df.to_parquet(out_path, index=False)
    logger.info("Saved %d weekly records to %s", len(weekly_df), out_path)
    return out_path


if __name__ == "__main__":
    PANAJI_LAT, PANAJI_LON = 15.49, 73.83
    weekly = fetch_and_aggregate(PANAJI_LAT, PANAJI_LON, "2015-01-01", "2023-12-31")
    save_weekly_climate(weekly, PANAJI_LAT, PANAJI_LON)
