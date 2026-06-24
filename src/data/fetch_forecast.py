"""
fetch_forecast.py -- Fetch RTE day-ahead (D-1) load forecasts and write the
``forecast_rte_dayahead_baseline`` Parquet table.

Resource
--------
RTE Consumption API, ``short_term`` resource:
    GET /open_api/consumption/v1/short_term
    (base: https://digital.iservices.rte-france.com)

The endpoint returns several forecast vintages under the ``short_term`` key:
    REALISED  -- actual measured load (intraday)
    ID        -- intraday updated forecast
    D-1       -- day-ahead forecast published ~23:55 Paris time (D-1 evening)
    D-2       -- two-days-ahead forecast
    CORRECTED -- corrected historical series (sparse)

This module selects the ``D-1`` vintage exclusively.  Each D-1 block covers
one calendar day in Europe/Paris time.  The ``updated_date`` field on every
value row records the publication timestamp of that specific forecast revision;
that becomes ``forecast_made_utc``.

Granularity
-----------
The API returns 15-minute steps (4 values per hour).  We resample to hourly
by taking the mean of the four MW readings per clock hour.  The hourly
timestamp (``ts_utc``) is the start-of-hour in UTC.

Idempotency
-----------
The composite key is ``(ts_utc, forecast_made_utc)``.  On each run we pull
a configurable trailing window and upsert: an existing row is replaced only
if the same ``ts_utc`` now carries a different ``forecast_made_utc`` (i.e.
the D-1 forecast was revised).  Rows that are identical in both key columns
are silently discarded before writing.

Authentication
--------------
OAuth2 client-credentials flow, identical to fetch_rte.py.  Credentials
(RTE_CLIENT_ID / RTE_CLIENT_SECRET) must be set in .env or the calling
environment -- never hardcoded here.

Units
-----
All ``forecast_mw`` values are in MW (megawatts), as returned by the API.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from src.config import (
    PROCESSED_DIR,
    RTE_SHORT_TERM_URL,
)
from src.data.fetch_rte import _get_rte_token

logger = logging.getLogger(__name__)

# The API accepts a maximum window of ~186 days per call.  We use 180 to stay
# safely under the limit regardless of DST edge cases.
_MAX_WINDOW_DAYS: int = 180

# The D-1 vintage identifier as returned by the API.
_FORECAST_TYPE_D1: str = "D-1"


def fetch_rte_dayahead_window(
    token: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Pull D-1 load forecasts from the RTE short_term API for a single window.

    The API is queried with ``type=D-1`` to receive only the day-ahead
    vintage.  Results are at 15-minute granularity.

    Parameters
    ----------
    token : str
        Bearer token obtained from ``fetch_rte._get_rte_token``.
    start_dt : datetime
        Inclusive window start (UTC-aware).  The forecast target timestamps
        (``ts_utc``) within this window are returned.
    end_dt : datetime
        Exclusive window end (UTC-aware).  Must satisfy
        ``end_dt - start_dt <= 180 days``.

    Returns
    -------
    pd.DataFrame
        Columns:
            ts_utc            datetime64[ns, UTC]  -- 15-min interval start
            forecast_made_utc datetime64[ns, UTC]  -- publication timestamp
            forecast_mw       float32              -- forecast load in MW
        Sorted ascending by ts_utc.  Not yet resampled to hourly.

    Raises
    ------
    ValueError
        If the window exceeds _MAX_WINDOW_DAYS.
    httpx.HTTPStatusError
        On non-2xx API response.
    """
    if (end_dt - start_dt).days > _MAX_WINDOW_DAYS:
        raise ValueError(
            f"Window of {(end_dt - start_dt).days} days exceeds the "
            f"{_MAX_WINDOW_DAYS}-day API limit.  Split into smaller chunks."
        )

    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    params = {
        "start_date": start_str,
        "end_date": end_str,
        "type": _FORECAST_TYPE_D1,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.get(RTE_SHORT_TERM_URL, headers=headers, params=params)
        resp.raise_for_status()

    payload = resp.json()
    blocks = payload.get("short_term", [])

    rows: list[dict[str, object]] = []
    for block in blocks:
        if block.get("type") != _FORECAST_TYPE_D1:
            # Guard: the type filter should already ensure this, but be explicit.
            continue
        for item in block.get("values", []):
            ts = pd.Timestamp(item["start_date"]).tz_convert("UTC")
            made = pd.Timestamp(item["updated_date"]).tz_convert("UTC")
            rows.append(
                {
                    "ts_utc": ts,
                    "forecast_made_utc": made,
                    "forecast_mw": float(item["value"]),
                }
            )

    if not rows:
        logger.warning(
            "fetch_rte_dayahead_window: no D-1 data returned for %s -> %s",
            start_dt.date(),
            end_dt.date(),
        )
        return pd.DataFrame(columns=["ts_utc", "forecast_made_utc", "forecast_mw"])

    df = pd.DataFrame(rows)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["forecast_made_utc"] = pd.to_datetime(df["forecast_made_utc"], utc=True)
    df["forecast_mw"] = df["forecast_mw"].astype("float32")
    return df.sort_values("ts_utc").reset_index(drop=True)


def _resample_15min_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a 15-minute D-1 forecast DataFrame to hourly resolution.

    Each output hour (start-of-hour in UTC) is the arithmetic mean of the
    four 15-minute MW readings that fall within that clock hour.  The
    ``forecast_made_utc`` is taken as the first (earliest) publication time
    among the four sub-hour readings; in practice all four share the same
    value since RTE publishes the D-1 forecast as a single batch.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: ts_utc (UTC, 15-min), forecast_made_utc (UTC), forecast_mw.

    Returns
    -------
    pd.DataFrame
        Columns: ts_utc (UTC, hourly), forecast_made_utc (UTC), forecast_mw.
        Sorted ascending by ts_utc.
    """
    df = df.copy()
    df["ts_hour"] = df["ts_utc"].dt.floor("h")

    hourly = (
        df.groupby("ts_hour", sort=True)
        .agg(
            forecast_mw=("forecast_mw", "mean"),
            forecast_made_utc=("forecast_made_utc", "min"),
        )
        .reset_index()
        .rename(columns={"ts_hour": "ts_utc"})
    )

    hourly["forecast_mw"] = hourly["forecast_mw"].astype("float32")
    hourly["ts_utc"] = pd.to_datetime(hourly["ts_utc"], utc=True)
    hourly["forecast_made_utc"] = pd.to_datetime(hourly["forecast_made_utc"], utc=True)
    cols = ["ts_utc", "forecast_made_utc", "forecast_mw"]
    return hourly[cols].sort_values("ts_utc").reset_index(drop=True)


def fetch_rte_dayahead_baseline(
    processed_dir: Path,
    client_id: str,
    client_secret: str,
    trailing_days: int = 60,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Pull RTE D-1 load forecasts and upsert into
    ``forecast_rte_dayahead_baseline.parquet``.

    Operating modes
    ---------------
    * Incremental (default): re-pull the last ``trailing_days`` and upsert.
    * Backfill: supply explicit ``start_date`` / ``end_date`` to populate a
      historical range.  Windows wider than ``_MAX_WINDOW_DAYS`` are split
      automatically into consecutive chunks.

    Idempotency
    -----------
    The composite key is ``(ts_utc, forecast_made_utc)``.  Rows already
    present with the same key are left unchanged.  A row is replaced only
    when the same ``ts_utc`` now has a different ``forecast_made_utc`` (i.e.
    a late revision was published by RTE).

    Parameters
    ----------
    processed_dir : Path
        Directory where ``forecast_rte_dayahead_baseline.parquet`` is stored
        (created if absent).
    client_id : str
        RTE OAuth2 client ID.  Read from .env / environment -- never
        hardcoded.
    client_secret : str
        RTE OAuth2 client secret.
    trailing_days : int
        Days to re-pull in incremental mode (default 60, captures late
        revisions).  Ignored when ``start_date`` / ``end_date`` are supplied.
    start_date : datetime | None
        Explicit backfill start (UTC-aware).  If supplied, ``end_date`` must
        also be supplied.
    end_date : datetime | None
        Explicit backfill end (UTC-aware, exclusive).

    Returns
    -------
    pd.DataFrame
        Full ``forecast_rte_dayahead_baseline`` table after the upsert.
        Columns: ts_utc, forecast_made_utc, forecast_mw.

    Raises
    ------
    ValueError
        If only one of ``start_date`` / ``end_date`` is supplied, or if
        ``start_date >= end_date``.
    """
    # Validate explicit date range
    if (start_date is None) != (end_date is None):
        raise ValueError("Provide both start_date and end_date, or neither.")
    if start_date is not None and end_date is not None and start_date >= end_date:
        raise ValueError("start_date must be strictly before end_date.")

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "forecast_rte_dayahead_baseline.parquet"

    existing: pd.DataFrame | None = None
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        logger.info("Loaded existing forecast parquet: %d rows", len(existing))

    # Determine the fetch window(s)
    if start_date is not None and end_date is not None:
        fetch_start = start_date
        fetch_end = end_date
    else:
        fetch_end = datetime.now(UTC)
        fetch_start = fetch_end - timedelta(days=trailing_days)

    # Obtain OAuth2 token (shared client-credentials flow with fetch_rte.py)
    token = _get_rte_token(client_id, client_secret)
    logger.info(
        "Fetching D-1 forecast %s -> %s",
        fetch_start.date(),
        fetch_end.date(),
    )

    # Split into chunks of at most _MAX_WINDOW_DAYS to respect the API limit
    chunk_start = fetch_start
    raw_parts: list[pd.DataFrame] = []
    while chunk_start < fetch_end:
        chunk_end = min(chunk_start + timedelta(days=_MAX_WINDOW_DAYS), fetch_end)
        logger.debug("Fetching chunk %s -> %s", chunk_start.date(), chunk_end.date())
        chunk_df = fetch_rte_dayahead_window(token, chunk_start, chunk_end)
        if not chunk_df.empty:
            raw_parts.append(chunk_df)
        chunk_start = chunk_end

    if not raw_parts:
        logger.warning("No D-1 forecast data returned for the requested window.")
        return (
            existing
            if existing is not None
            else pd.DataFrame(columns=["ts_utc", "forecast_made_utc", "forecast_mw"])
        )

    raw_15min = pd.concat(raw_parts, ignore_index=True)

    # Resample 15-min -> hourly
    new_hourly = _resample_15min_to_hourly(raw_15min)

    if existing is not None:
        combined = pd.concat([existing, new_hourly], ignore_index=True)
    else:
        combined = new_hourly

    # Deduplicate: keep the last occurrence of each (ts_utc, forecast_made_utc)
    # pair.  Sorting by ts_utc then forecast_made_utc ensures the most-recent
    # vintage wins when forecasts for the same hour were revised.
    combined = (
        combined.sort_values(["ts_utc", "forecast_made_utc"])
        .drop_duplicates(subset=["ts_utc"], keep="last")
        .reset_index(drop=True)
    )

    combined.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %s (%d rows, %s -> %s)",
        out_path,
        len(combined),
        combined["ts_utc"].min(),
        combined["ts_utc"].max(),
    )
    return combined


def backfill_rte_dayahead_baseline(
    processed_dir: Path,
    client_id: str,
    client_secret: str,
    start_year: int = 2023,
) -> pd.DataFrame:
    """
    Backfill the full D-1 forecast history from ``start_year`` to today.

    This is a convenience wrapper around ``fetch_rte_dayahead_baseline`` for
    initial population.  The RTE API retains D-1 data from at least 2023;
    earlier years require the Tempo/eCO2mix archive files which do not carry
    the day-ahead forecast separately.

    Parameters
    ----------
    processed_dir : Path
        Output directory.
    client_id : str
        RTE OAuth2 client ID.
    client_secret : str
        RTE OAuth2 client secret.
    start_year : int
        First calendar year to pull (default 2023 -- the earliest confirmed
        available year on the API).

    Returns
    -------
    pd.DataFrame
        Full ``forecast_rte_dayahead_baseline`` table.
    """
    start_dt = datetime(start_year, 1, 1, tzinfo=UTC)
    end_dt = datetime.now(UTC)
    logger.info("Starting D-1 forecast backfill: %d -> %s", start_year, end_dt.date())
    return fetch_rte_dayahead_baseline(
        processed_dir=processed_dir,
        client_id=client_id,
        client_secret=client_secret,
        start_date=start_dt,
        end_date=end_dt,
    )


def main() -> None:
    """
    CLI entry point: incremental pull using credentials from the environment.

    Usage:
        python -m src.data.fetch_forecast

    Environment variables required:
        RTE_CLIENT_ID
        RTE_CLIENT_SECRET

    Optional (via .env file in the project root).
    """
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    )

    client_id = os.environ.get("RTE_CLIENT_ID", "")
    client_secret = os.environ.get("RTE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        logger.error(
            "RTE_CLIENT_ID and RTE_CLIENT_SECRET must be set in the " "environment or .env file."
        )
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / PROCESSED_DIR

    result = fetch_rte_dayahead_baseline(
        processed_dir=processed_dir,
        client_id=client_id,
        client_secret=client_secret,
        trailing_days=60,
    )
    print(f"Done.  {len(result)} rows in " f"forecast_rte_dayahead_baseline.parquet")
    print(f"  ts_utc range: {result['ts_utc'].min()} -> {result['ts_utc'].max()}")


if __name__ == "__main__":
    main()
