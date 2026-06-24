"""
test_calendar.py -- Property checks on the calendar_daily table.

Goes beyond the dtype/PK checks in test_schema.py to assert calendar
semantics: workday flag agrees with weekend + bank-holiday flags, every
calendar year is fully covered, and at least one entry exists in each
school-vacation zone.
"""

from __future__ import annotations

import pandas as pd


def test_workday_flag_matches_weekend_and_holiday(calendar_df: pd.DataFrame) -> None:
    """is_workday must be False on weekends AND on bank holidays."""
    df = calendar_df.copy()
    df["weekday"] = df["date"].apply(lambda d: d.weekday())
    is_weekend = df["weekday"] >= 5
    expected_workday = ~(is_weekend | df["is_bank_holiday"])
    assert (df["is_workday"] == expected_workday).all()


def test_every_year_fully_covered(calendar_df: pd.DataFrame) -> None:
    """Each year between min and max must have either 365 or 366 days."""
    years = pd.Series([d.year for d in calendar_df["date"]])
    counts = years.value_counts().sort_index()
    for year, count in counts.items():
        assert count in (365, 366), f"year {year} has {count} days"


def test_all_three_school_vacation_zones_used(calendar_df: pd.DataFrame) -> None:
    """At least one day flagged True in every zone."""
    for col in (
        "school_vacation_zone_a",
        "school_vacation_zone_b",
        "school_vacation_zone_c",
    ):
        assert calendar_df[col].any(), f"{col} is never True"


def test_christmas_day_is_a_bank_holiday(calendar_df: pd.DataFrame) -> None:
    """Spot-check a second bank holiday besides Bastille Day."""
    xmas = calendar_df[calendar_df["date"].apply(lambda d: d.month == 12 and d.day == 25)]
    assert xmas["is_bank_holiday"].all()
    assert xmas["holiday_name"].notna().all()
