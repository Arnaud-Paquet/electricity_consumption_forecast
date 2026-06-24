"""
fetch_calendar.py -- Build the calendar_daily table.

Sources
-------
1. French bank holidays: `holidays.France(years=range(2013, 2031))`.
   No network request required -- the `holidays` library ships the data.

2. School vacation zones A/B/C: data.education.gouv.fr "Calendrier scolaire" CSV.
   URL: https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/
        fr-en-calendrier-scolaire/exports/csv?lang=fr&timezone=Europe%2FParis

   The CSV has columns:
       description, population, zones, start_date, end_date, annee_scolaire
   where `zones` is a space-separated list like "Zone A Zone B" or "Zone C".

   We explode by zone and build boolean flags (school_vacation_zone_a,
   school_vacation_zone_b, school_vacation_zone_c).

   Fallback: if the download fails (no network), the zone columns are set to
   False and a warning is logged. The bank holidays and is_workday columns
   are still valid.

Output schema (calendar_daily)
-------------------------------
date                    : date       -- primary key, calendar date
is_workday              : bool       -- False on weekends and bank holidays
is_bank_holiday         : bool
holiday_name            : str | None -- name of the bank holiday (nullable)
school_vacation_zone_a  : bool
school_vacation_zone_b  : bool
school_vacation_zone_c  : bool
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta
from pathlib import Path

import holidays as hols
import httpx
import pandas as pd

logger = logging.getLogger(__name__)

SCHOOL_CAL_URL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-calendrier-scolaire/exports/csv"
    "?lang=fr&timezone=Europe%2FParis&delimiter=%2C"
)

START_YEAR: int = 2013
END_YEAR: int = 2030  # Inclusive; school calendar CSV typically covers to 2030


def _fetch_school_calendar_csv() -> pd.DataFrame | None:
    """
    Download the French school vacation calendar CSV from data.education.gouv.fr.

    Returns None if the download fails (caller handles gracefully).

    Returns
    -------
    pd.DataFrame | None
        Raw CSV as DataFrame, or None on failure.
    """
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(SCHOOL_CAL_URL)
            resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), dtype=str)
        logger.info("Downloaded school calendar CSV: %d rows, cols=%s", len(df), list(df.columns))
        return df
    except Exception as exc:
        logger.warning("Could not download school calendar CSV: %s", exc)
        return None


def _build_vacation_flags(
    school_df: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build a DataFrame with boolean vacation flags for each date in date_range.

    Parameters
    ----------
    school_df : pd.DataFrame
        Raw school calendar CSV (from data.education.gouv.fr).
    date_range : pd.DatetimeIndex
        All dates to cover.

    Returns
    -------
    pd.DataFrame
        Index = date, columns = school_vacation_zone_a/b/c (bool).
    """
    result = pd.DataFrame(
        {
            "school_vacation_zone_a": False,
            "school_vacation_zone_b": False,
            "school_vacation_zone_c": False,
        },
        index=date_range.date,
    )

    # Normalise column names to lowercase
    school_df.columns = pd.Index([c.strip().lower() for c in school_df.columns])

    # Identify relevant columns
    # Expected: description, population, zones, start_date, end_date, annee_scolaire
    zone_col = next((c for c in school_df.columns if "zone" in c), None)
    start_col = next((c for c in school_df.columns if "start" in c or c == "start_date"), None)
    end_col = next((c for c in school_df.columns if "end" in c or c == "end_date"), None)

    if zone_col is None or start_col is None or end_col is None:
        logger.warning(
            "School calendar CSV columns unexpected: %s; skipping vacation flags",
            list(school_df.columns),
        )
        return result

    # Keep only rows related to vacances scolaires (exclude Feries, Rentrée, etc.)
    if "description" in school_df.columns:
        school_df = school_df[
            school_df["description"].str.lower().str.contains("vacances", na=False)
        ]

    for _, row in school_df.iterrows():
        zones_raw = str(row.get(zone_col, "")).strip()
        start_raw = str(row.get(start_col, "")).strip()
        end_raw = str(row.get(end_col, "")).strip()

        if not zones_raw or not start_raw or not end_raw:
            continue

        try:
            # Dates may include time component (e.g. "2013-10-19T00:00:00+02:00")
            period_start = pd.Timestamp(start_raw).date()
            period_end = pd.Timestamp(end_raw).date()
        except Exception:
            continue

        # Parse zones: "Zone A", "Zone B", "Zone C" or combinations
        zones_upper = zones_raw.upper()
        is_a = "ZONE A" in zones_upper or "A" in zones_upper.split()
        is_b = "ZONE B" in zones_upper or "B" in zones_upper.split()
        is_c = "ZONE C" in zones_upper or "C" in zones_upper.split()

        # Mark each date in the period
        cur = period_start
        while cur <= period_end:
            if cur in result.index:
                if is_a:
                    result.at[cur, "school_vacation_zone_a"] = True
                if is_b:
                    result.at[cur, "school_vacation_zone_b"] = True
                if is_c:
                    result.at[cur, "school_vacation_zone_c"] = True
            cur = cur + timedelta(days=1)

    return result


def build_calendar_daily(
    processed_dir: Path,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> pd.DataFrame:
    """
    Build and write calendar_daily.parquet.

    Idempotent: if the file already exists and covers the requested range,
    it is returned as-is. If the range has extended (e.g. end_year updated),
    the new dates are appended.

    Parameters
    ----------
    processed_dir : Path
        Output directory.
    start_year : int
        First calendar year (default 2013).
    end_year : int
        Last calendar year, inclusive (default 2030).

    Returns
    -------
    pd.DataFrame
        Columns: date, is_workday, is_bank_holiday, holiday_name,
                 school_vacation_zone_a, school_vacation_zone_b,
                 school_vacation_zone_c.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "calendar_daily.parquet"

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing_min = existing["date"].min()
        existing_max = existing["date"].max()
        target_min = date(start_year, 1, 1)
        target_max = date(end_year, 12, 31)
        if existing_min <= target_min and existing_max >= target_max:
            logger.info("Calendar parquet already covers %s to %s; no-op", target_min, target_max)
            return existing
        logger.info("Calendar parquet exists but range extended; rebuilding")

    # Date range
    all_dates = pd.date_range(
        start=date(start_year, 1, 1),
        end=date(end_year, 12, 31),
        freq="D",
    )

    # Bank holidays using the `holidays` library. Country classes are
    # registered lazily and not surfaced to mypy through holidays==0.58's
    # type info, so attribute access on `hols.France` needs an explicit
    # ignore even though it works at runtime.
    fr_holidays = hols.France(years=range(start_year, end_year + 1))  # type: ignore[attr-defined]

    rows = []
    for dt in all_dates:
        d = dt.date()
        is_bh = d in fr_holidays
        holiday_name = fr_holidays.get(d, None)
        is_weekend = d.weekday() >= 5
        is_workday = not (is_weekend or is_bh)
        rows.append(
            {
                "date": d,
                "is_workday": is_workday,
                "is_bank_holiday": is_bh,
                "holiday_name": holiday_name,
                "school_vacation_zone_a": False,
                "school_vacation_zone_b": False,
                "school_vacation_zone_c": False,
            }
        )

    df = pd.DataFrame(rows)
    # Keep date column as Python date objects so PyArrow writes date32 (not timestamp).
    # pd.to_datetime would produce datetime64[ns] -> Parquet TIMESTAMP, which violates
    # the schema requirement of a plain date (date32).
    df["date"] = df["date"].apply(lambda d: d if isinstance(d, date) else d.date())

    # School vacation flags
    school_csv = _fetch_school_calendar_csv()
    if school_csv is not None:
        vacation_flags = _build_vacation_flags(school_csv, all_dates)
        for col in ["school_vacation_zone_a", "school_vacation_zone_b", "school_vacation_zone_c"]:
            # vacation_flags index contains Python date objects; df["date"] is also date
            # objects after our dtype fix.  Build a mapping dict to avoid index-type
            # mismatches that silently produce all-NaN when aligning a DatetimeIndex
            # against a date-object index.
            flag_map: dict[object, bool] = dict(
                zip(vacation_flags.index, vacation_flags[col], strict=False)
            )
            df[col] = df["date"].map(flag_map).fillna(False).astype(bool)
    else:
        logger.warning(
            "School vacation download failed; " "school_vacation_zone_* columns set to False"
        )

    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))
    return df
