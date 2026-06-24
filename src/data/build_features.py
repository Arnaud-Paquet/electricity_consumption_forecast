"""
build_features.py -- Feature engineering for DEMETER forecasting models.

This module is a stub for W1. Full implementation in W3.

Feature categories (to be implemented)
---------------------------------------
1. Temporal: hour_of_day, day_of_week, month, quarter, is_dst.
2. Calendar: is_workday, is_bank_holiday, school_vacation_zone_* (from calendar_daily).
3. Load lags: load_mw at t-1h, t-2h, t-24h, t-48h, t-168h.
4. Rolling statistics: rolling mean/std over 24h, 48h, 168h windows.
5. Weather features: temperature_2m, wind_speed_10m, shortwave_radiation,
   cloud_cover, precipitation -- joined from weather_city_hourly via
   region_city_mapping. Units preserved as per schema (degC, m/s, W/m2, %, mm).

Note on shortwave_radiation and precipitation
----------------------------------------------
These variables follow the preceding-hour accumulation convention:
a row at timestamp T contains the value for the interval [T-1h, T).
When lagging these against demand (instantaneous at T), no additional
time shift is needed -- the value is already contemporaneous with T.
"""

from __future__ import annotations


def build_feature_matrix() -> None:
    """Placeholder -- full implementation in W3."""
    raise NotImplementedError("Feature engineering not implemented until W3")
