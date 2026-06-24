"""
Project-wide constants for DEMETER.

All coordinates are (latitude, longitude) in decimal degrees WGS84.
Region codes follow ISO 3166-2:FR. Corsica (FR-COR) is excluded by design:
the RTE consolidated national series covers continental France only.
"""

from __future__ import annotations

from typing import TypedDict

# ---------------------------------------------------------------------------
# Region -> city centroid mapping (one representative city per region, v1)
# Expandable to population-weighted multi-city in v2.
# ---------------------------------------------------------------------------


class CityInfo(TypedDict):
    """Per-region city centroid metadata.

    Typed so REGION_CITY_MAP values type-check downstream (fetch_weather.py
    etc.) without `# type: ignore[index]` escape hatches.
    """

    city_code: str
    city_name: str
    lat: float
    lon: float


REGION_CITY_MAP: dict[str, CityInfo] = {
    "FR-IDF": {"city_code": "PAR", "city_name": "Paris", "lat": 48.8566, "lon": 2.3522},
    "FR-ARA": {"city_code": "LYO", "city_name": "Lyon", "lat": 45.7640, "lon": 4.8357},
    "FR-HDF": {"city_code": "LIL", "city_name": "Lille", "lat": 50.6292, "lon": 3.0573},
    "FR-NAQ": {"city_code": "BOR", "city_name": "Bordeaux", "lat": 44.8378, "lon": -0.5792},
    "FR-OCC": {"city_code": "TLS", "city_name": "Toulouse", "lat": 43.6047, "lon": 1.4442},
    "FR-GES": {"city_code": "STR", "city_name": "Strasbourg", "lat": 48.5734, "lon": 7.7521},
    "FR-PAC": {"city_code": "MRS", "city_name": "Marseille", "lat": 43.2965, "lon": 5.3698},
    "FR-PDL": {"city_code": "NTE", "city_name": "Nantes", "lat": 47.2184, "lon": -1.5536},
    "FR-NOR": {"city_code": "CAE", "city_name": "Caen", "lat": 49.1829, "lon": -0.3707},
    "FR-BRE": {"city_code": "REN", "city_name": "Rennes", "lat": 48.1173, "lon": -1.6778},
    "FR-BFC": {"city_code": "DIJ", "city_name": "Dijon", "lat": 47.3220, "lon": 5.0415},
    "FR-CVL": {"city_code": "ORL", "city_name": "Orleans", "lat": 47.9029, "lon": 1.9093},
}

# Flat lookup: city_code -> region_code
CITY_TO_REGION: dict[str, str] = {v["city_code"]: k for k, v in REGION_CITY_MAP.items()}

# Ordered list of 12 region codes (no Corsica)
REGION_CODES: list[str] = list(REGION_CITY_MAP.keys())

# Ordered list of 12 city codes
CITY_CODES: list[str] = [v["city_code"] for v in REGION_CITY_MAP.values()]

# ---------------------------------------------------------------------------
# Directory mapping: regional raw sub-folder name -> region code
# The raw sub-folders under data/raw/consumption_regional_hourly/ are now named
# by ISO 3166-2:FR code (e.g. FR-IDF), so the lookup is the identity map.
# ---------------------------------------------------------------------------

REGION_DIR_TO_CODE: dict[str, str] = {code: code for code in REGION_CODES}

# ---------------------------------------------------------------------------
# Temporal scope
# ---------------------------------------------------------------------------

DATA_START_YEAR: int = 2013
PARIS_TZ: str = "Europe/Paris"
UTC_TZ: str = "UTC"

# ---------------------------------------------------------------------------
# Open-Meteo archive API
# ---------------------------------------------------------------------------

OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"

# Variables to pull. shortwave_radiation and precipitation are preceding-hour
# accumulations: a row at T covers the interval [T-1h, T).
WEATHER_VARIABLES: list[str] = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
]

# ---------------------------------------------------------------------------
# RTE API (incremental path only -- requires OAuth2)
# ---------------------------------------------------------------------------

RTE_API_BASE_URL: str = "https://digital.iservices.rte-france.com"
RTE_TOKEN_URL: str = f"{RTE_API_BASE_URL}/token/oauth/token"
RTE_CONSUMPTION_URL: str = (
    f"{RTE_API_BASE_URL}/open_api/consolidated_consumption/v1" "/consolidated_power_consumption"
)
# RTE Consumption API -- short-term forecast resource.
# Returns multiple forecast vintages (REALISED, ID, D-1, D-2, CORRECTED).
# We select type=D-1, the day-ahead forecast published around 23:55 Paris time
# the evening before the forecast day.  Max window per call: ~186 days.
RTE_SHORT_TERM_URL: str = f"{RTE_API_BASE_URL}/open_api/consumption/v1/short_term"

# The consolidated_power_consumption resource lags real time: definitive data
# is only published several weeks after the fact. Requesting an end_date more
# recent than the available term returns HTTP 400 (CONSOCONSU_COPCO_F04).
# Empirically (June 2026) data is available up to roughly now-50 days and the
# 400 boundary sits around now-15 days. We clamp the incremental end_date to
# now - RTE_CONSOLIDATION_LAG_DAYS to stay safely inside the recoverable term.
RTE_CONSOLIDATION_LAG_DAYS: int = 25

# ---------------------------------------------------------------------------
# Data paths (relative to project root; resolved at runtime by each script)
# ---------------------------------------------------------------------------

NATIONAL_RAW_DIR: str = "data/raw/consumption_national_hourly"
REGIONAL_RAW_DIR: str = "data/raw/consumption_regional_hourly"
PROCESSED_DIR: str = "data/processed"
INTERIM_DIR: str = "data/interim"
RAW_DIR: str = "data/raw"
