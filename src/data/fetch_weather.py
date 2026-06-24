"""
fetch_weather.py -- Pull ERA5 reanalysis weather data from the Open-Meteo
Archive API (no authentication required).

Endpoint: https://archive-api.open-meteo.com/v1/archive

IMPORTANT: Preceding-hour accumulation convention
--------------------------------------------------
The variables `shortwave_radiation` and `precipitation` follow the
preceding-hour convention: the value at timestamp T represents the
accumulated quantity over the interval [T-1h, T).
For example, the row at 2013-01-01 01:00 UTC holds radiation/precip that
fell between 00:00 and 01:00 UTC.

This must be accounted for when lagging these features against demand
(which is an instantaneous reading at the start of the hour).

Wind speed
----------
MANDATORY parameter: wind_speed_unit=ms
The Open-Meteo default is km/h; the schema stores m/s. Omitting this
parameter silently corrupts the wind_speed_10m column.

Idempotency
-----------
The function checks the existing Parquet (if any) and only requests data
for (city, date) pairs not already present. Re-running with the same date
range is a no-op.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd

from src.config import (
    OPEN_METEO_ARCHIVE_URL,
    REGION_CITY_MAP,
    WEATHER_VARIABLES,
    CityInfo,
)

logger = logging.getLogger(__name__)

UTC = UTC

# Open-Meteo free tier: 10,000 API calls/day, ~600 calls/hour.
# With 12 cities and yearly chunks we need ~132 calls for full backfill.
# A small sleep between requests avoids hammering the endpoint.
_REQUEST_SLEEP_SECONDS: float = 0.3

# Open-Meteo returns at most a few years per call without issue.
# Chunk by year to keep response sizes manageable and allow partial retries.
_CHUNK_YEARS: int = 1


def fetch_city_weather(
    city_code: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch hourly ERA5 reanalysis weather for a single city from Open-Meteo.

    Parameters
    ----------
    city_code : str
        3-letter code used as the city identifier (e.g. 'PAR', 'LYO').
    lat : float
        Latitude in decimal degrees (WGS84).
    lon : float
        Longitude in decimal degrees (WGS84).
    start_date : date
        First date to fetch (inclusive).
    end_date : date
        Last date to fetch (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: ts_utc (datetime64[ns, UTC]), city_code (str),
                 temperature_2m (float32, degC),
                 wind_speed_10m (float32, m/s),
                 shortwave_radiation (float32, W/m2, preceding-hour accumulation),
                 cloud_cover (float32, %),
                 precipitation (float32, mm, preceding-hour accumulation).
        One row per UTC hour. Sorted ascending by ts_utc.

    Raises
    ------
    httpx.HTTPStatusError
        If the Open-Meteo API returns a non-2xx status.
    ValueError
        If the API response is missing the 'hourly' key.
    """
    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(WEATHER_VARIABLES),
        "wind_speed_unit": "ms",  # MANDATORY: default is km/h
        "timezone": "UTC",
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.get(OPEN_METEO_ARCHIVE_URL, params=params)
        resp.raise_for_status()

    payload = resp.json()

    if "hourly" not in payload:
        raise ValueError(
            f"Open-Meteo response missing 'hourly' key for city {city_code}: "
            f"{list(payload.keys())}"
        )

    hourly = payload["hourly"]
    times = hourly["time"]  # ISO strings in UTC, e.g. "2013-01-01T00:00"

    df = pd.DataFrame({"time": times})
    df["ts_utc"] = pd.to_datetime(df["time"], utc=True)

    for var in WEATHER_VARIABLES:
        if var in hourly:
            df[var] = hourly[var]
        else:
            logger.warning("Variable '%s' not in response for city %s", var, city_code)
            df[var] = float("nan")

    df["city_code"] = city_code

    # Cast to float32 as per schema
    for var in WEATHER_VARIABLES:
        df[var] = df[var].astype("float32")

    cols = ["ts_utc", "city_code"] + list(WEATHER_VARIABLES)
    return df[cols].sort_values("ts_utc").reset_index(drop=True)


def build_weather_hourly(
    processed_dir: Path,
    start_year: int = 2013,
    end_date: date | None = None,
    region_city_map: dict[str, CityInfo] | None = None,
) -> pd.DataFrame:
    """
    Build (or update) weather_city_hourly.parquet for all 12 city centroids.

    Idempotency: checks the existing Parquet for each (city_code, year) and
    skips any year already fully present. Only fetches missing year-chunks.

    Parameters
    ----------
    processed_dir : Path
        Output directory (created if absent).
    start_year : int
        First calendar year to fetch (default 2013).
    end_date : date | None
        Last date to fetch (inclusive). Defaults to yesterday UTC.
    region_city_map : dict | None
        Mapping of region_code -> {city_code, lat, lon, ...}.
        Defaults to REGION_CITY_MAP from src.config.

    Returns
    -------
    pd.DataFrame
        Full weather_city_hourly table.
    """
    if region_city_map is None:
        region_city_map = REGION_CITY_MAP

    if end_date is None:
        end_date = datetime.now(UTC).date() - pd.Timedelta(days=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "weather_city_hourly.parquet"

    existing: pd.DataFrame | None = None
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        logger.info("Loaded existing weather parquet: %d rows", len(existing))

    all_parts: list[pd.DataFrame] = []

    for region_code, info in region_city_map.items():
        city_code = info["city_code"]
        lat = info["lat"]
        lon = info["lon"]

        logger.info("Processing city %s (%s) lat=%.4f lon=%.4f", city_code, region_code, lat, lon)

        for year in range(start_year, end_date.year + 1):
            chunk_start = date(year, 1, 1)
            chunk_end = date(year, 12, 31)
            if chunk_end > end_date:
                chunk_end = end_date
            if chunk_start > end_date:
                break

            # Idempotency check: skip if this city/year already in parquet
            if existing is not None:
                city_existing = existing[existing["city_code"] == city_code]
                if len(city_existing) > 0:
                    city_max_ts = city_existing["ts_utc"].max()
                    if city_max_ts.year > year:
                        logger.debug("Skipping %s year %d (already in parquet)", city_code, year)
                        continue

            logger.info(
                "Fetching %s %d-%d (%s to %s)",
                city_code,
                year,
                year,
                chunk_start,
                chunk_end,
            )
            try:
                df = fetch_city_weather(city_code, lat, lon, chunk_start, chunk_end)
                all_parts.append(df)
                time.sleep(_REQUEST_SLEEP_SECONDS)
            except Exception as exc:
                logger.warning("Failed to fetch %s %d: %s", city_code, year, exc)

    if not all_parts:
        if existing is not None:
            logger.info("No new weather data fetched; returning existing parquet")
            return existing
        raise RuntimeError("No weather data fetched and no existing parquet found")

    new_data = pd.concat(all_parts, ignore_index=True)

    if existing is not None:
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    # Deduplicate: keep last value per (ts_utc, city_code)
    combined = combined.sort_values(["ts_utc", "city_code"]).drop_duplicates(
        subset=["ts_utc", "city_code"], keep="last"
    )
    combined = combined.reset_index(drop=True)

    combined.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(combined))
    return combined
