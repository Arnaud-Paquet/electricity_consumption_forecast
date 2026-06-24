"""
test_config.py -- Pure-Python sanity checks on src/config.py.

These tests do not depend on the processed Parquet files, so they always
run -- on a fresh clone, in CI, before any data is fetched.
"""

from __future__ import annotations

from src.config import (
    CITY_CODES,
    CITY_TO_REGION,
    DATA_START_YEAR,
    REGION_CITY_MAP,
    REGION_CODES,
    REGION_DIR_TO_CODE,
    RTE_CONSOLIDATION_LAG_DAYS,
    WEATHER_VARIABLES,
)


def test_twelve_continental_regions_no_corsica() -> None:
    """Brief: 12 continental regions, Corsica explicitly excluded."""
    assert len(REGION_CITY_MAP) == 12
    assert len(REGION_CODES) == 12
    assert "FR-COR" not in REGION_CODES


def test_region_codes_follow_iso_3166_2_fr() -> None:
    for code in REGION_CODES:
        assert code.startswith("FR-"), code
        assert len(code) == 6, code


def test_city_codes_unique_and_three_letters() -> None:
    assert len(set(CITY_CODES)) == 12
    for code in CITY_CODES:
        assert len(code) == 3, code
        assert code.isupper(), code


def test_city_to_region_inverse_consistent() -> None:
    """CITY_TO_REGION must be the exact inverse of REGION_CITY_MAP."""
    for region_code, info in REGION_CITY_MAP.items():
        city_code = info["city_code"]
        assert CITY_TO_REGION[city_code] == region_code


def test_lat_lon_in_metropolitan_france() -> None:
    """All 12 city centroids must land inside continental France's bbox."""
    for region_code, info in REGION_CITY_MAP.items():
        lat = info["lat"]
        lon = info["lon"]
        assert 41.0 <= lat <= 51.5, f"{region_code} lat={lat}"
        assert -5.5 <= lon <= 9.5, f"{region_code} lon={lon}"


def test_region_dir_map_covers_all_regions() -> None:
    assert set(REGION_DIR_TO_CODE.values()) == set(REGION_CODES)


def test_weather_variables_match_schema() -> None:
    """The five Open-Meteo variables in the brief's weather_city_hourly schema."""
    assert WEATHER_VARIABLES == [
        "temperature_2m",
        "wind_speed_10m",
        "shortwave_radiation",
        "cloud_cover",
        "precipitation",
    ]


def test_temporal_constants_sane() -> None:
    assert DATA_START_YEAR == 2013
    assert 1 <= RTE_CONSOLIDATION_LAG_DAYS <= 60
