"""
Shared pytest fixtures for the DEMETER test suite.

Loading processed Parquets is expensive (the regional and weather tables
are ~1.4M rows each), so fixtures are session-scoped: each table is read
from disk once per `pytest` invocation and reused across all tests.

Tests that depend on a processed table are skipped (not failed) when the
file is missing, so the suite is meaningful both in CI (where parquets
are fetched first) and on a fresh clone (where only unit tests run).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Project root is the parent of the tests/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"


def _load_or_skip(filename: str) -> pd.DataFrame:
    """Load a Parquet from data/processed, or skip the test if absent."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        pytest.skip(
            f"{filename} not found at {path}. "
            "Run `python scripts/fetch_data.py --backfill` first."
        )
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def national_df() -> pd.DataFrame:
    """consumption_national_hourly.parquet."""
    return _load_or_skip("consumption_national_hourly.parquet")


@pytest.fixture(scope="session")
def regional_df() -> pd.DataFrame:
    """consumption_regional_hourly.parquet."""
    return _load_or_skip("consumption_regional_hourly.parquet")


@pytest.fixture(scope="session")
def weather_df() -> pd.DataFrame:
    """weather_city_hourly.parquet."""
    return _load_or_skip("weather_city_hourly.parquet")


@pytest.fixture(scope="session")
def calendar_df() -> pd.DataFrame:
    """calendar_daily.parquet."""
    return _load_or_skip("calendar_daily.parquet")


@pytest.fixture(scope="session")
def mapping_df() -> pd.DataFrame:
    """region_city_mapping.parquet."""
    return _load_or_skip("region_city_mapping.parquet")


@pytest.fixture(scope="session")
def forecast_df() -> pd.DataFrame:
    """forecast_rte_dayahead_baseline.parquet."""
    return _load_or_skip("forecast_rte_dayahead_baseline.parquet")
