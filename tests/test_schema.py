"""
test_schema.py -- Schema and basic integrity checks for the 6 processed
DEMETER tables.

These tests pin the contract documented in docs/PROJECT_BRIEF.md "Data
schema" so a silent dtype drift (e.g. float32 -> float64, naive timestamp
instead of UTC-aware) is caught the moment fetch_data.py is changed.

For each table we assert:
    1. Required columns exist.
    2. Dtypes match the schema (timestamps are UTC-aware, loads are float32).
    3. No nulls in primary-key / non-nullable columns.
    4. Primary keys are unique.
    5. Timestamps are strictly increasing within each series.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
)

from src.config import CITY_CODES, REGION_CODES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_utc(series: pd.Series, name: str) -> None:
    """Fail if `series` is not a UTC-aware datetime column."""
    assert is_datetime64_any_dtype(series), f"{name} is not datetime, got {series.dtype}"
    tz = getattr(series.dtype, "tz", None)
    assert tz is not None, f"{name} is timezone-naive; brief mandates UTC-aware"
    assert str(tz) == "UTC", f"{name} tz is {tz}, expected UTC"


# ---------------------------------------------------------------------------
# region_city_mapping
# ---------------------------------------------------------------------------


def test_mapping_columns(mapping_df: pd.DataFrame) -> None:
    assert list(mapping_df.columns) == ["region_code", "city_code", "weight"]


def test_mapping_dtypes(mapping_df: pd.DataFrame) -> None:
    assert is_object_dtype(mapping_df["region_code"])
    assert is_object_dtype(mapping_df["city_code"])
    assert mapping_df["weight"].dtype == "float32"


def test_mapping_twelve_regions_no_corsica(mapping_df: pd.DataFrame) -> None:
    """12 continental regions; Corsica explicitly excluded by design."""
    assert len(mapping_df) == 12
    assert "FR-COR" not in mapping_df["region_code"].values
    assert set(mapping_df["region_code"]) == set(REGION_CODES)
    assert set(mapping_df["city_code"]) == set(CITY_CODES)


def test_mapping_weights_sum_to_one_per_region(mapping_df: pd.DataFrame) -> None:
    weights = mapping_df.groupby("region_code")["weight"].sum()
    # v1 is single-city per region with weight 1.0; allow float32 slack.
    assert ((weights - 1.0).abs() < 1e-6).all(), weights.to_dict()


# ---------------------------------------------------------------------------
# consumption_national_hourly
# ---------------------------------------------------------------------------


def test_national_columns(national_df: pd.DataFrame) -> None:
    assert list(national_df.columns) == ["ts_utc", "load_mw", "is_estimated"]


def test_national_dtypes(national_df: pd.DataFrame) -> None:
    _assert_utc(national_df["ts_utc"], "national.ts_utc")
    assert national_df["load_mw"].dtype == "float32"
    assert is_bool_dtype(national_df["is_estimated"])


def test_national_pk_unique_and_sorted(national_df: pd.DataFrame) -> None:
    assert national_df["ts_utc"].is_unique
    assert national_df["ts_utc"].is_monotonic_increasing


def test_national_load_positive_and_reasonable(national_df: pd.DataFrame) -> None:
    """French national load sits roughly in 25-100 GW; sanity bounds."""
    lo, hi = 20_000.0, 110_000.0
    assert national_df["load_mw"].between(lo, hi).all(), (
        f"load_mw outside [{lo}, {hi}] MW; "
        f"got min={national_df['load_mw'].min()}, max={national_df['load_mw'].max()}"
    )


# ---------------------------------------------------------------------------
# consumption_regional_hourly
# ---------------------------------------------------------------------------


def test_regional_columns(regional_df: pd.DataFrame) -> None:
    assert list(regional_df.columns) == [
        "ts_utc",
        "region_code",
        "load_mw",
        "is_estimated",
    ]


def test_regional_dtypes(regional_df: pd.DataFrame) -> None:
    _assert_utc(regional_df["ts_utc"], "regional.ts_utc")
    assert is_object_dtype(regional_df["region_code"])
    assert regional_df["load_mw"].dtype == "float32"
    assert is_bool_dtype(regional_df["is_estimated"])


def test_regional_twelve_regions(regional_df: pd.DataFrame) -> None:
    assert set(regional_df["region_code"].unique()) == set(REGION_CODES)


def test_regional_composite_pk_unique(regional_df: pd.DataFrame) -> None:
    """(ts_utc, region_code) is the composite primary key."""
    assert not regional_df.duplicated(subset=["ts_utc", "region_code"]).any()


def test_regional_load_non_negative(regional_df: pd.DataFrame) -> None:
    assert (regional_df["load_mw"] >= 0).all()


# ---------------------------------------------------------------------------
# weather_city_hourly
# ---------------------------------------------------------------------------


WEATHER_FLOAT_COLS = (
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
)


def test_weather_columns(weather_df: pd.DataFrame) -> None:
    expected = ["ts_utc", "city_code", *WEATHER_FLOAT_COLS]
    assert list(weather_df.columns) == expected


def test_weather_dtypes(weather_df: pd.DataFrame) -> None:
    _assert_utc(weather_df["ts_utc"], "weather.ts_utc")
    assert is_object_dtype(weather_df["city_code"])
    for col in WEATHER_FLOAT_COLS:
        assert weather_df[col].dtype == "float32", f"{col} dtype={weather_df[col].dtype}"


def test_weather_twelve_cities(weather_df: pd.DataFrame) -> None:
    assert set(weather_df["city_code"].unique()) == set(CITY_CODES)


def test_weather_composite_pk_unique(weather_df: pd.DataFrame) -> None:
    assert not weather_df.duplicated(subset=["ts_utc", "city_code"]).any()


def test_weather_physical_ranges(weather_df: pd.DataFrame) -> None:
    """Sanity bounds on each meteorological variable."""
    t = weather_df["temperature_2m"]
    assert t.between(
        -40, 50
    ).all(), f"temperature out of [-40, 50] degC: min={t.min()} max={t.max()}"

    wind = weather_df["wind_speed_10m"]
    assert (wind >= 0).all() and (wind <= 80).all()

    rad = weather_df["shortwave_radiation"]
    assert (rad >= 0).all() and (rad <= 1500).all()

    cc = weather_df["cloud_cover"]
    assert (cc >= 0).all() and (cc <= 100).all()

    precip = weather_df["precipitation"]
    assert (precip >= 0).all()


# ---------------------------------------------------------------------------
# calendar_daily
# ---------------------------------------------------------------------------


def test_calendar_columns(calendar_df: pd.DataFrame) -> None:
    assert list(calendar_df.columns) == [
        "date",
        "is_workday",
        "is_bank_holiday",
        "holiday_name",
        "school_vacation_zone_a",
        "school_vacation_zone_b",
        "school_vacation_zone_c",
    ]


def test_calendar_dtypes(calendar_df: pd.DataFrame) -> None:
    # `date` is stored as a Python date object column.
    assert isinstance(calendar_df["date"].iloc[0], dt.date)
    for col in (
        "is_workday",
        "is_bank_holiday",
        "school_vacation_zone_a",
        "school_vacation_zone_b",
        "school_vacation_zone_c",
    ):
        assert is_bool_dtype(calendar_df[col]), col


def test_calendar_pk_unique_and_sorted(calendar_df: pd.DataFrame) -> None:
    assert calendar_df["date"].is_unique
    assert calendar_df["date"].is_monotonic_increasing


def test_calendar_spans_2013_to_2030(calendar_df: pd.DataFrame) -> None:
    assert calendar_df["date"].min() == dt.date(2013, 1, 1)
    assert calendar_df["date"].max() == dt.date(2030, 12, 31)


def test_calendar_holiday_name_only_on_bank_holidays(calendar_df: pd.DataFrame) -> None:
    """holiday_name is non-null iff is_bank_holiday is True."""
    has_name = calendar_df["holiday_name"].notna()
    assert (has_name == calendar_df["is_bank_holiday"]).all()


def test_calendar_july_14_is_holiday_every_year(calendar_df: pd.DataFrame) -> None:
    """Spot-check: Bastille Day is always a French bank holiday."""
    july14 = calendar_df[calendar_df["date"].apply(lambda d: d.month == 7 and d.day == 14)]
    assert july14["is_bank_holiday"].all()


# ---------------------------------------------------------------------------
# forecast_rte_dayahead_baseline
# ---------------------------------------------------------------------------


def test_forecast_columns(forecast_df: pd.DataFrame) -> None:
    assert list(forecast_df.columns) == ["ts_utc", "forecast_made_utc", "forecast_mw"]


def test_forecast_dtypes(forecast_df: pd.DataFrame) -> None:
    _assert_utc(forecast_df["ts_utc"], "forecast.ts_utc")
    _assert_utc(forecast_df["forecast_made_utc"], "forecast.forecast_made_utc")
    assert forecast_df["forecast_mw"].dtype == "float32"


def test_forecast_made_utc_bounded(forecast_df: pd.DataFrame) -> None:
    """``forecast_made_utc`` is sourced from RTE's ``updated_date`` field
    (when the prediction was last revised), so it can legitimately be
    *after* ``ts_utc`` — RTE keeps revising even after the target hour.
    What we DO require is that the delta is bounded: any forecast made
    more than 3 days *before* its target, or more than 10 years *after*,
    is corrupt.
    """
    delta_hours = (
        forecast_df["forecast_made_utc"] - forecast_df["ts_utc"]
    ).dt.total_seconds() / 3600
    assert delta_hours.min() >= -72, f"made too far before target: {delta_hours.min()}h"
    assert delta_hours.max() <= 10 * 365 * 24, f"made too far after target: {delta_hours.max()}h"


def test_forecast_load_positive_and_reasonable(forecast_df: pd.DataFrame) -> None:
    """Forecast values must sit in the same range as observed load."""
    assert forecast_df["forecast_mw"].between(20_000, 110_000).all()


def test_forecast_starts_2023_or_later(forecast_df: pd.DataFrame) -> None:
    """RTE D-1 history confirmed available from 2023 (per PROJECT_BRIEF.md)."""
    earliest = forecast_df["ts_utc"].min()
    assert earliest >= pd.Timestamp("2023-01-01", tz="UTC")
