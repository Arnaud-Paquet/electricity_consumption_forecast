"""
fetch_data.py -- Orchestrate backfill and incremental refresh of all six
DEMETER processed Parquet tables.

Tables produced (under data/processed/)
-----------------------------------------
1. region_city_mapping          -- static 12-row lookup (no network call)
2. calendar_daily               -- bank holidays + school vacations 2013-2030
3. consumption_national_hourly  -- eCO2mix .xls backfill + optional RTE API increment
4. consumption_regional_hourly  -- 12-region eCO2mix .xls backfill
5. weather_city_hourly          -- Open-Meteo ERA5 reanalysis, 12 city centroids
6. forecast_rte_dayahead_baseline -- RTE D-1 forecast (API, 2023-today)

Usage
-----
Backfill (full history; first run only -- long due to weather pull):
    python scripts/fetch_data.py --backfill

Incremental (nightly CI -- re-pulls a rolling trailing window):
    python scripts/fetch_data.py --incremental [--trailing-days N]

If neither flag is given, --incremental is the default (safe for scheduled runs).

Idempotency
-----------
All underlying fetchers are idempotent: re-running the same range is a no-op.
This orchestrator does NOT force-overwrite; it delegates idempotency to each
fetcher.

Authentication
--------------
RTE OAuth2 credentials (RTE_CLIENT_ID / RTE_CLIENT_SECRET) are read from the
.env file in the project root (or from the environment -- e.g. GitHub Actions
secrets). They are NEVER printed or logged.

Cost
----
$0: Open-Meteo requires no API key. RTE OAuth2 is free with registration.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root -- resolve relative to this file so the CLI works regardless
# of the CWD from which it is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Add project root to sys.path so `src.*` imports resolve when the package is
# not installed (e.g. raw `python scripts/fetch_data.py` invocations).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402 -- must follow sys.path mutation
    DATA_START_YEAR,
    NATIONAL_RAW_DIR,
    PROCESSED_DIR,
    REGION_CITY_MAP,
    REGION_DIR_TO_CODE,
    REGIONAL_RAW_DIR,
)
from src.data.fetch_calendar import build_calendar_daily  # noqa: E402
from src.data.fetch_forecast import (  # noqa: E402
    backfill_rte_dayahead_baseline,
    fetch_rte_dayahead_baseline,
)
from src.data.fetch_rte import (  # noqa: E402
    build_national_hourly,
    build_region_city_mapping,
    build_regional_hourly,
    incremental_national_update,
)
from src.data.fetch_weather import build_weather_hourly  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

UTC = UTC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_result(table_name: str, df: Any, t0: float) -> None:
    """Log row count and elapsed time after a table is built.

    Parameters
    ----------
    table_name : str
        Human-readable name of the table.
    df : Any
        The returned DataFrame (has len()).
    t0 : float
        Wall-clock start time from time.monotonic().
    """
    elapsed = time.monotonic() - t0
    logger.info(
        "%-42s  %8d rows   %.1fs",
        table_name,
        len(df),
        elapsed,
    )


def _require_rte_creds() -> tuple[str, str]:
    """Read RTE OAuth2 credentials from the environment.

    Returns
    -------
    tuple[str, str]
        (client_id, client_secret). Never logged or printed.

    Raises
    ------
    SystemExit
        If either variable is missing.
    """
    client_id = os.environ.get("RTE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("RTE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        logger.error(
            "RTE_CLIENT_ID and RTE_CLIENT_SECRET must be set in .env "
            "or the calling environment. Neither is committed to the repo."
        )
        sys.exit(1)
    return client_id, client_secret


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def run_region_city_mapping(processed_dir: Path) -> None:
    """Build static region_city_mapping.parquet (12 rows, no network call)."""
    t0 = time.monotonic()
    df = build_region_city_mapping(
        region_city_map=REGION_CITY_MAP,
        processed_dir=processed_dir,
    )
    _log_result("region_city_mapping", df, t0)


def run_calendar(processed_dir: Path) -> None:
    """Build calendar_daily.parquet (2013-2030, holidays library + HTTP)."""
    t0 = time.monotonic()
    df = build_calendar_daily(processed_dir=processed_dir)
    _log_result("calendar_daily", df, t0)


def run_national_backfill(
    raw_national_dir: Path,
    processed_dir: Path,
    start_year: int,
) -> None:
    """Parse all national eCO2mix .xls files and write consumption_national_hourly."""
    t0 = time.monotonic()
    df = build_national_hourly(
        raw_dir=raw_national_dir,
        processed_dir=processed_dir,
        start_year=start_year,
    )
    _log_result("consumption_national_hourly (backfill)", df, t0)


def run_regional_backfill(
    raw_regional_dir: Path,
    processed_dir: Path,
    start_year: int,
) -> None:
    """Parse all 12-region eCO2mix .xls files and write consumption_regional_hourly."""
    t0 = time.monotonic()
    df = build_regional_hourly(
        raw_dir=raw_regional_dir,
        processed_dir=processed_dir,
        region_dir_map=REGION_DIR_TO_CODE,
        start_year=start_year,
    )
    _log_result("consumption_regional_hourly (backfill)", df, t0)


def run_national_incremental(
    processed_dir: Path,
    client_id: str,
    client_secret: str,
    trailing_days: int,
) -> None:
    """Pull rolling trailing window from RTE API and upsert national hourly."""
    t0 = time.monotonic()
    df = incremental_national_update(
        processed_dir=processed_dir,
        client_id=client_id,
        client_secret=client_secret,
        trailing_days=trailing_days,
    )
    _log_result("consumption_national_hourly (incremental)", df, t0)


def run_weather(
    processed_dir: Path,
    start_year: int,
) -> None:
    """Fetch ERA5 weather for 12 city centroids via Open-Meteo (no auth).

    This is the long pole in the backfill: ~12 cities x 13 years of hourly
    data. Open-Meteo caches aggressively, so a second run is fast.
    """
    t0 = time.monotonic()
    df = build_weather_hourly(
        processed_dir=processed_dir,
        start_year=start_year,
    )
    _log_result("weather_city_hourly", df, t0)


def run_forecast_backfill(
    processed_dir: Path,
    client_id: str,
    client_secret: str,
    start_year: int = 2023,
) -> None:
    """Backfill RTE D-1 forecast from start_year (minimum 2023) to today."""
    t0 = time.monotonic()
    df = backfill_rte_dayahead_baseline(
        processed_dir=processed_dir,
        client_id=client_id,
        client_secret=client_secret,
        start_year=start_year,
    )
    _log_result("forecast_rte_dayahead_baseline (backfill)", df, t0)


def run_forecast_incremental(
    processed_dir: Path,
    client_id: str,
    client_secret: str,
    trailing_days: int,
) -> None:
    """Pull rolling trailing window for D-1 forecast and upsert."""
    t0 = time.monotonic()
    df = fetch_rte_dayahead_baseline(
        processed_dir=processed_dir,
        client_id=client_id,
        client_secret=client_secret,
        trailing_days=trailing_days,
    )
    _log_result("forecast_rte_dayahead_baseline (incremental)", df, t0)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_backfill(
    processed_dir: Path,
    raw_national_dir: Path,
    raw_regional_dir: Path,
    start_year: int,
    client_id: str,
    client_secret: str,
) -> None:
    """Run the full backfill pipeline to produce all 6 processed tables.

    Execution order: static/cheap tables first, then the long weather pull,
    and RTE API calls last (so a network failure does not abort local parsing).

    Parameters
    ----------
    processed_dir : Path
        Output directory for Parquet files.
    raw_national_dir : Path
        Directory containing national eCO2mix .xls files.
    raw_regional_dir : Path
        Root directory containing per-region sub-folders of .xls files.
    start_year : int
        First calendar year to include (from DATA_START_YEAR in config).
    client_id : str
        RTE OAuth2 client ID (read from env -- not logged).
    client_secret : str
        RTE OAuth2 client secret (read from env -- not logged).
    """
    wall_start = time.monotonic()
    logger.info("=== BACKFILL START (start_year=%d) ===", start_year)

    # Step 1: static mapping -- instant, no network
    logger.info("[1/6] Building region_city_mapping ...")
    run_region_city_mapping(processed_dir)

    # Step 2: calendar -- holidays library + one HTTP call for school vacations
    logger.info("[2/6] Building calendar_daily ...")
    run_calendar(processed_dir)

    # Step 3: national consumption -- parse .xls files on disk
    logger.info("[3/6] Building consumption_national_hourly (xls backfill) ...")
    run_national_backfill(raw_national_dir, processed_dir, start_year)

    # Step 4: regional consumption -- parse 12 * 13 years of .xls files
    logger.info("[4/6] Building consumption_regional_hourly (xls backfill) ...")
    run_regional_backfill(raw_regional_dir, processed_dir, start_year)

    # Step 5: weather -- Open-Meteo ERA5, 12 cities, long pole on first run
    logger.info("[5/6] Building weather_city_hourly (Open-Meteo ERA5) ...")
    run_weather(processed_dir, start_year)

    # Step 6: day-ahead forecast baseline -- RTE API, 2023-today
    logger.info("[6/6] Building forecast_rte_dayahead_baseline (RTE D-1 API) ...")
    run_forecast_backfill(
        processed_dir=processed_dir,
        client_id=client_id,
        client_secret=client_secret,
        start_year=2023,  # D-1 API history confirmed available from 2023
    )

    elapsed = time.monotonic() - wall_start
    logger.info(
        "=== BACKFILL COMPLETE  total wall-clock: %.0fs (%.1fmin) ===", elapsed, elapsed / 60
    )


def run_incremental(
    processed_dir: Path,
    raw_national_dir: Path,
    raw_regional_dir: Path,
    start_year: int,
    client_id: str,
    client_secret: str,
    trailing_days: int,
) -> None:
    """Run the incremental refresh pipeline (nightly CI mode).

    Re-pulls a rolling trailing window for live API sources. Re-parses the
    En-cours-Consolide .xls files for the .xls-based sources to capture late
    consolidation revisions. All operations are idempotent (no-ops when data
    is unchanged).

    Parameters
    ----------
    processed_dir : Path
        Output directory for Parquet files.
    raw_national_dir : Path
        Directory containing national eCO2mix .xls files.
    raw_regional_dir : Path
        Root directory containing per-region sub-folders.
    start_year : int
        Passed through to backfill helpers for any missing tables.
    client_id : str
        RTE OAuth2 client ID.
    client_secret : str
        RTE OAuth2 client secret.
    trailing_days : int
        How many trailing days to re-pull from live APIs (default 60).
    """
    wall_start = time.monotonic()
    logger.info("=== INCREMENTAL START (trailing_days=%d) ===", trailing_days)

    # Static mapping and calendar are always rebuilt idempotently (cheap)
    logger.info("[1/6] Refreshing region_city_mapping ...")
    run_region_city_mapping(processed_dir)

    logger.info("[2/6] Refreshing calendar_daily ...")
    run_calendar(processed_dir)

    # National: RTE API incremental path captures late revisions
    logger.info("[3/6] Incremental national consumption (RTE API) ...")
    run_national_incremental(
        processed_dir=processed_dir,
        client_id=client_id,
        client_secret=client_secret,
        trailing_days=trailing_days,
    )

    # Regional: re-parse the En-cours-Consolide .xls (already idempotent)
    logger.info("[4/6] Refreshing consumption_regional_hourly (xls) ...")
    run_regional_backfill(raw_regional_dir, processed_dir, start_year)

    # Weather: Open-Meteo will only fetch missing city/year chunks
    logger.info("[5/6] Refreshing weather_city_hourly (Open-Meteo) ...")
    run_weather(processed_dir, start_year)

    # Forecast: incremental trailing window
    logger.info("[6/6] Refreshing forecast_rte_dayahead_baseline (RTE D-1 API) ...")
    run_forecast_incremental(
        processed_dir=processed_dir,
        client_id=client_id,
        client_secret=client_secret,
        trailing_days=trailing_days,
    )

    elapsed = time.monotonic() - wall_start
    logger.info(
        "=== INCREMENTAL COMPLETE  total wall-clock: %.0fs (%.1fmin) ===",
        elapsed,
        elapsed / 60,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (default: sys.argv[1:]).

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="fetch_data.py",
        description=(
            "Produce all 6 DEMETER processed Parquet tables. "
            "Default mode is --incremental (safe for nightly scheduled runs). "
            "Use --backfill on the first run to populate full history."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "Full history from DATA_START_YEAR. "
            "Idempotent: re-running is a no-op for already-present data. "
            "Long on first run due to weather (Open-Meteo ERA5, 12 cities x 13 years)."
        ),
    )
    mode.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Rolling trailing window only (default). "
            "Runs in seconds for subsequent nightly calls once the parquets exist."
        ),
    )
    parser.add_argument(
        "--trailing-days",
        type=int,
        default=60,
        metavar="N",
        help=(
            "Days of history to re-pull in incremental mode (default: 60). "
            "A larger window captures more late-arriving RTE consolidation revisions."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] | None
        Argument list for unit-testing; defaults to sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    # Load .env from project root (no-op when vars are already in the environment)
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args(argv)

    # Resolve paths relative to project root (config values are bare strings)
    processed_dir = PROJECT_ROOT / PROCESSED_DIR
    raw_national_dir = PROJECT_ROOT / NATIONAL_RAW_DIR
    raw_regional_dir = PROJECT_ROOT / REGIONAL_RAW_DIR

    # Ensure output directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Default to incremental when no flag is given
    mode_is_backfill = args.backfill  # False when --incremental or neither flag set

    # Both modes need RTE creds (national API for incremental; forecast API for both)
    client_id, client_secret = _require_rte_creds()

    try:
        if mode_is_backfill:
            run_backfill(
                processed_dir=processed_dir,
                raw_national_dir=raw_national_dir,
                raw_regional_dir=raw_regional_dir,
                start_year=DATA_START_YEAR,
                client_id=client_id,
                client_secret=client_secret,
            )
        else:
            run_incremental(
                processed_dir=processed_dir,
                raw_national_dir=raw_national_dir,
                raw_regional_dir=raw_regional_dir,
                start_year=DATA_START_YEAR,
                client_id=client_id,
                client_secret=client_secret,
                trailing_days=args.trailing_days,
            )
    except Exception:
        logger.exception("fetch_data.py terminated with an unhandled error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
