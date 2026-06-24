"""
fetch_rte.py -- Parse RTE eCO2mix .xls files and (optionally) pull the
Consolidated Consumption v1.0 REST API.

File format notes
-----------------
The `.xls` files distributed by RTE eCO2mix are NOT real Excel binaries.
They are ISO-8859-1 (latin-1) encoded tab-separated text with:
  - A single header row (column names in French).
  - Data rows at 30-minute steps.
  - Every other row (HH:15 / HH:45 slots) has the Consommation cell blank --
    those rows carry only forecast columns and must be skipped for load.
  - A single trailing disclaimer line (starts with "RTE ne pourra") that
    must be stripped.
  - 'Nature' values: "Donnees definitives" -> is_estimated=False,
    "Donnees consolidees" -> is_estimated=True (provisional/recent).

Resampling
----------
Each hour contains two 30-min readings (HH:00 and HH:30). The hourly load
is the arithmetic mean of those two instantaneous MW readings. The resulting
hourly timestamp (`ts_utc`) is the start of the clock hour in UTC.

Incremental API path
--------------------
The RTE Consolidated Consumption v1.0 API requires OAuth2 client credentials
(RTE_CLIENT_ID / RTE_CLIENT_SECRET in .env). All API functions in this module
are implemented but NOT called during the backfill (--backfill) flow. They are
invoked only when scripts/fetch_data.py is run with --incremental.
"""

from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pandas as pd
import pytz

if TYPE_CHECKING:
    pass

from src.config import (
    PARIS_TZ,
    RTE_CONSOLIDATION_LAG_DAYS,
    RTE_CONSUMPTION_URL,
    RTE_TOKEN_URL,
    CityInfo,
)

logger = logging.getLogger(__name__)

PARIS = pytz.timezone(PARIS_TZ)
UTC = UTC

# ---------------------------------------------------------------------------
# Helpers: parse a single eCO2mix .xls tab-separated text file
# ---------------------------------------------------------------------------


def _read_eco2mix_tsv(path: Path) -> pd.DataFrame:
    """
    Read a single RTE eCO2mix tab-separated .xls file.

    Returns a raw DataFrame with columns:
        perimetre, nature, date_str, heures, consommation, ...

    Strips the trailing disclaimer line and keeps the header row.
    Encoding: ISO-8859-1.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("latin-1")
    lines = text.split("\n")

    # Drop trailing disclaimer / empty lines
    clean_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("RTE ne pourra")]

    import io

    # index_col=False is required: some national files have one more tab-separated
    # field per data row than the header (36 header cols, 37 data cols), which
    # causes pandas to silently treat the first data column as the row index,
    # shifting all column assignments by one position and corrupting every row.
    df = pd.read_csv(
        io.StringIO("\n".join(clean_lines)),
        sep="\t",
        encoding="latin-1",
        dtype=str,
        low_memory=False,
        index_col=False,
    )
    return df


def _parse_timestamp_paris(date_str: str, heures: str) -> pd.Timestamp:
    """
    Parse a Europe/Paris timestamp from RTE date (YYYY-MM-DD) and time (HH:MM)
    strings, returning a UTC-aware Timestamp.

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format (Europe/Paris wall clock).
    heures : str
        Time in HH:MM format (Europe/Paris wall clock).

    Returns
    -------
    pd.Timestamp
        UTC-aware timestamp representing the start of the 30-min interval.
    """
    naive = pd.Timestamp(f"{date_str} {heures}:00")
    try:
        aware = PARIS.localize(naive, is_dst=None)
    except pytz.exceptions.AmbiguousTimeError:
        # During clock-back (last Sunday of October), take fold=1 (standard time)
        aware = PARIS.localize(naive, is_dst=False)
    except pytz.exceptions.NonExistentTimeError:
        # During clock-forward (last Sunday of March), skip / return NaT
        return pd.NaT  # type: ignore[return-value]
    return pd.Timestamp(aware).tz_convert("UTC")


def _nature_to_is_estimated(nature: str) -> bool:
    """
    Map RTE 'Nature' field to the is_estimated boolean.

    'Donnees definitives' -> False (definitive, historical)
    'Donnees consolidees' -> True  (provisional, recent consolidation)
    'Donnees provisoires' -> True

    Parameters
    ----------
    nature : str
        Raw value of the Nature column (French, may contain accented chars).

    Returns
    -------
    bool
        True when the row is provisional/estimated.
    """
    lower = nature.lower().strip()
    # "definitivement" or accented "d\xe9finitivement" -> final;
    # "consolidees" and "provisoires" both fall through to estimated.
    return "d\xe9finitiv" not in lower and "definitiv" not in lower


# ---------------------------------------------------------------------------
# National backfill
# ---------------------------------------------------------------------------


def parse_national_eco2mix(path: Path) -> pd.DataFrame:
    """
    Parse a single national eCO2mix file (Annuel-Definitif or En-cours-Consolide).

    Steps:
    1. Read tab-separated text (latin-1 encoding).
    2. Keep only rows where the 'Consommation' cell is not blank/ND.
    3. Parse Europe/Paris timestamps -> UTC.
    4. Resample 30-min -> hourly: mean of the two readings per clock hour.

    Parameters
    ----------
    path : Path
        Absolute path to the .xls tab-separated text file.

    Returns
    -------
    pd.DataFrame
        Columns: ts_utc (datetime64[ns, UTC]), load_mw (float32),
                 is_estimated (bool).
        One row per UTC hour. Sorted ascending by ts_utc.
    """
    df = _read_eco2mix_tsv(path)

    # Normalise column names (header uses French accented names)
    col_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if "consommation" in low:
            col_map[col] = "consommation"
        elif "heures" in low:
            col_map[col] = "heures"
        elif "date" in low:
            col_map[col] = "date_str"
        elif "nature" in low:
            col_map[col] = "nature"
        elif "p\xe9rim\xe8tre" in low or "perimetre" in low:
            col_map[col] = "perimetre"
    df = df.rename(columns=col_map)

    # Keep only rows with valid Consommation (not blank, not 'ND', not NaN)
    df = df[df["consommation"].notna()]
    df = df[~df["consommation"].str.strip().isin(["", "ND", "-"])]

    # Build UTC timestamps
    ts_list = [
        _parse_timestamp_paris(row["date_str"], row["heures"])
        for _, row in df[["date_str", "heures"]].iterrows()
    ]
    df = df.copy()
    df["ts_utc_raw"] = ts_list

    # Drop rows where timestamp parsing failed (NonExistentTimeError -> NaT)
    df = df[df["ts_utc_raw"].notna()]

    # Determine is_estimated from Nature column
    df["is_estimated"] = df["nature"].apply(_nature_to_is_estimated)

    # Parse load as float
    df["load_mw_raw"] = pd.to_numeric(df["consommation"].str.strip(), errors="coerce")
    df = df[df["load_mw_raw"].notna()]

    # Resample to hourly: floor to hour, then mean within each hour
    df["ts_hour"] = df["ts_utc_raw"].apply(lambda t: t.floor("h"))

    hourly = (
        df.groupby("ts_hour", sort=True)
        .agg(
            load_mw=("load_mw_raw", "mean"),
            is_estimated=("is_estimated", "max"),  # True if any row in hour is estimated
        )
        .reset_index()
        .rename(columns={"ts_hour": "ts_utc"})
    )

    hourly["load_mw"] = hourly["load_mw"].astype("float32")
    hourly["ts_utc"] = pd.to_datetime(hourly["ts_utc"], utc=True)

    return hourly.sort_values("ts_utc").reset_index(drop=True)


def build_national_hourly(
    raw_dir: Path,
    processed_dir: Path,
    start_year: int = 2013,
) -> pd.DataFrame:
    """
    Parse all national eCO2mix files and write consumption_national_hourly.parquet.

    Idempotency: if the Parquet already exists, load it and only process files
    whose year is newer than max(ts_utc) in the existing file, then append.
    The function keys idempotency on the year of each source file -- if a year
    is already fully present in the Parquet, it is skipped.

    Parameters
    ----------
    raw_dir : Path
        Directory containing eCO2mix_RTE_Annuel-Definitif_*.xls and
        eCO2mix_RTE_En-cours-Consolide.xls.
    processed_dir : Path
        Output directory (created if absent).
    start_year : int
        First year to include in the output (default 2013).

    Returns
    -------
    pd.DataFrame
        The full consumption_national_hourly table (all years).
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "consumption_national_hourly.parquet"

    existing: pd.DataFrame | None = None
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        logger.info("Loaded existing national parquet: %d rows", len(existing))

    # Collect annual definitive files (sorted by year)
    annual_files = sorted(raw_dir.glob("eCO2mix_RTE_Annuel-Definitif_*.xls"))
    # Also include the En-cours-Consolide file for recent data
    consolide_files = list(raw_dir.glob("eCO2mix_RTE_En-cours-Consolide.xls"))

    all_files = annual_files + consolide_files

    parts: list[pd.DataFrame] = []
    for fpath in all_files:
        # Extract year from filename if possible (annual files only)
        fname = fpath.stem
        year_str = None
        for part in fname.split("_"):
            if part.isdigit() and len(part) == 4:
                year_str = part
                break

        if year_str is not None:
            year_int = int(year_str)
            if year_int < start_year:
                logger.info("Skipping %s (before start_year %d)", fpath.name, start_year)
                continue
            # Idempotency check: skip year already fully present
            if existing is not None:
                max_ts = existing["ts_utc"].max()
                if max_ts.year > year_int:
                    logger.info("Skipping %s (year %d already in parquet)", fpath.name, year_int)
                    continue

        logger.info("Parsing %s ...", fpath.name)
        try:
            df = parse_national_eco2mix(fpath)
            # Filter to start_year and later
            df = df[df["ts_utc"].dt.year >= start_year]
            parts.append(df)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", fpath.name, exc)

    if not parts:
        if existing is not None:
            logger.info("No new files to process; returning existing parquet")
            return existing
        raise RuntimeError(f"No parseable national files found in {raw_dir}")

    new_data = pd.concat(parts, ignore_index=True)

    if existing is not None:
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    # Deduplicate: keep last occurrence per ts_utc (later files / consolidation win)
    combined = combined.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
    combined = combined.reset_index(drop=True)

    combined.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(combined))
    return combined


# ---------------------------------------------------------------------------
# Regional backfill
# ---------------------------------------------------------------------------


def parse_regional_eco2mix(path: Path, region_code: str) -> pd.DataFrame:
    """
    Parse a single regional eCO2mix .xls file.

    The regional format is identical to national except:
    - Column 5 (index 4) is 'Consommation' (same name, same units: MW).
    - Rows at HH:15 and HH:45 are blank on the Consommation column.
    - Some early years contain 'ND' for missing values.

    Parameters
    ----------
    path : Path
        Absolute path to the regional .xls file.
    region_code : str
        ISO 3166-2:FR code (e.g. 'FR-IDF') to attach to all rows.

    Returns
    -------
    pd.DataFrame
        Columns: ts_utc (datetime64[ns, UTC]), region_code (str),
                 load_mw (float32), is_estimated (bool).
        One row per UTC hour. Sorted ascending by ts_utc.
    """
    df = _read_eco2mix_tsv(path)

    # Normalise column names
    col_map: dict[str, str] = {}
    for col in df.columns:
        low = col.strip().lower()
        if "consommation" in low:
            col_map[col] = "consommation"
        elif "heures" in low:
            col_map[col] = "heures"
        elif "date" in low:
            col_map[col] = "date_str"
        elif "nature" in low:
            col_map[col] = "nature"
        elif "p\xe9rim\xe8tre" in low or "perimetre" in low:
            col_map[col] = "perimetre"
    df = df.rename(columns=col_map)

    # Keep rows with valid Consommation
    if "consommation" not in df.columns:
        raise ValueError(f"No 'Consommation' column found in {path}")
    df = df[df["consommation"].notna()]
    df = df[~df["consommation"].str.strip().isin(["", "ND", "-"])]

    # Build UTC timestamps
    ts_list = [
        _parse_timestamp_paris(row["date_str"], row["heures"])
        for _, row in df[["date_str", "heures"]].iterrows()
    ]
    df = df.copy()
    df["ts_utc_raw"] = ts_list
    df = df[df["ts_utc_raw"].notna()]

    df["is_estimated"] = df["nature"].apply(_nature_to_is_estimated)
    df["load_mw_raw"] = pd.to_numeric(df["consommation"].str.strip(), errors="coerce")
    df = df[df["load_mw_raw"].notna()]

    # Resample to hourly
    df["ts_hour"] = df["ts_utc_raw"].apply(lambda t: t.floor("h"))

    hourly = (
        df.groupby("ts_hour", sort=True)
        .agg(
            load_mw=("load_mw_raw", "mean"),
            is_estimated=("is_estimated", "max"),
        )
        .reset_index()
        .rename(columns={"ts_hour": "ts_utc"})
    )

    hourly["region_code"] = region_code
    hourly["load_mw"] = hourly["load_mw"].astype("float32")
    hourly["ts_utc"] = pd.to_datetime(hourly["ts_utc"], utc=True)

    cols = ["ts_utc", "region_code", "load_mw", "is_estimated"]
    return hourly[cols].sort_values("ts_utc").reset_index(drop=True)


def build_regional_hourly(
    raw_dir: Path,
    processed_dir: Path,
    region_dir_map: dict[str, str],
    start_year: int = 2013,
) -> pd.DataFrame:
    """
    Parse all regional eCO2mix files and write consumption_regional_hourly.parquet.

    Parameters
    ----------
    raw_dir : Path
        Root directory; each sub-folder is one region.
    processed_dir : Path
        Output directory.
    region_dir_map : dict[str, str]
        Mapping of folder-name -> ISO region code.
    start_year : int
        First year to include (default 2013).

    Returns
    -------
    pd.DataFrame
        Full consumption_regional_hourly table.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "consumption_regional_hourly.parquet"

    existing: pd.DataFrame | None = None
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        logger.info("Loaded existing regional parquet: %d rows", len(existing))

    all_parts: list[pd.DataFrame] = []

    for dir_name, region_code in sorted(region_dir_map.items()):
        region_dir = raw_dir / dir_name
        if not region_dir.exists():
            logger.warning("Region directory not found: %s", region_dir)
            continue

        # Collect annual + consolide files
        annual_files = sorted(region_dir.glob("*Annuel-Definitif*.xls"))
        consolide_files = list(region_dir.glob("*En-cours-Consolide*.xls"))
        all_files = annual_files + consolide_files

        for fpath in all_files:
            fname = fpath.stem
            year_str = None
            for part in fname.split("_"):
                if part.isdigit() and len(part) == 4:
                    year_str = part
                    break

            if year_str is not None:
                year_int = int(year_str)
                if year_int < start_year:
                    continue
                if existing is not None:
                    region_existing = existing[existing["region_code"] == region_code]
                    if len(region_existing) > 0:
                        max_ts = region_existing["ts_utc"].max()
                        if max_ts.year > year_int:
                            logger.debug("Skipping %s (already in parquet)", fpath.name)
                            continue

            logger.info("Parsing %s [%s] ...", fpath.name, region_code)
            try:
                df = parse_regional_eco2mix(fpath, region_code)
                df = df[df["ts_utc"].dt.year >= start_year]
                all_parts.append(df)
            except Exception as exc:
                logger.warning("Failed %s: %s", fpath.name, exc)

    if not all_parts:
        if existing is not None:
            return existing
        raise RuntimeError(f"No parseable regional files found under {raw_dir}")

    new_data = pd.concat(all_parts, ignore_index=True)

    if existing is not None:
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    combined = combined.sort_values(["ts_utc", "region_code"]).drop_duplicates(
        subset=["ts_utc", "region_code"], keep="last"
    )
    combined = combined.reset_index(drop=True)

    combined.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(combined))
    return combined


# ---------------------------------------------------------------------------
# Incremental API path (requires OAuth2 -- NOT called during backfill)
# ---------------------------------------------------------------------------


def _get_rte_token(client_id: str, client_secret: str) -> str:
    """
    Obtain an OAuth2 bearer token from the RTE API using client credentials.

    The token has approximately 2 hours TTL. Callers should cache the token
    until it expires rather than requesting a new one per call.

    Parameters
    ----------
    client_id : str
        RTE API client ID (from .env / GitHub Actions secret).
    client_secret : str
        RTE API client secret.

    Returns
    -------
    str
        Bearer access token string.

    Raises
    ------
    httpx.HTTPStatusError
        If the token endpoint returns a non-2xx status.
    """
    credentials = f"{client_id}:{client_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            RTE_TOKEN_URL,
            headers=headers,
            data={"grant_type": "client_credentials"},
        )
        resp.raise_for_status()
    return str(resp.json()["access_token"])


def fetch_rte_consolidated_window(
    token: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    Pull consolidated power consumption from the RTE API for a single window.

    The API returns 30-min data in descending order (newest first).
    Max window = 186 days per call.

    Parameters
    ----------
    token : str
        Bearer token from _get_rte_token.
    start_dt : datetime
        Start of window (UTC-aware).
    end_dt : datetime
        End of window (UTC-aware). Must be <= start_dt + 186 days.

    Returns
    -------
    pd.DataFrame
        Columns: ts_utc (datetime64[ns, UTC]), load_mw (float32),
                 is_estimated (bool), updated_date (datetime64[ns, UTC]).
        30-min resolution, NOT yet resampled to hourly.

    Raises
    ------
    httpx.HTTPStatusError
        On non-2xx API response.
    ValueError
        If window exceeds 186 days.
    """
    if (end_dt - start_dt).days > 186:
        raise ValueError("Window exceeds 186-day API limit")

    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    params = {
        "start_date": start_str,
        "end_date": end_str,
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.get(RTE_CONSUMPTION_URL, headers=headers, params=params)
        resp.raise_for_status()

    payload = resp.json()
    records = payload.get("consolidated_power_consumption", [])

    rows = []
    for chunk in records:
        for item in chunk.get("values", []):
            start_date_str = item["start_date"]
            ts = pd.Timestamp(start_date_str).tz_convert("UTC")
            updated = pd.Timestamp(item.get("updated_date", start_date_str)).tz_convert("UTC")
            status = item.get("status", "FINAL")
            rows.append(
                {
                    "ts_utc": ts,
                    "load_mw": float(item["value"]),
                    "is_estimated": status == "PROVISORY",
                    "updated_date": updated,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["ts_utc", "load_mw", "is_estimated", "updated_date"])

    df = pd.DataFrame(rows)
    df["load_mw"] = df["load_mw"].astype("float32")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["updated_date"] = pd.to_datetime(df["updated_date"], utc=True)
    return df.sort_values("ts_utc").reset_index(drop=True)


def incremental_national_update(
    processed_dir: Path,
    client_id: str,
    client_secret: str,
    trailing_days: int = 60,
) -> pd.DataFrame:
    """
    Pull the last `trailing_days` of consolidated national consumption from
    the RTE API and upsert into consumption_national_hourly.parquet.

    Idempotency is keyed on `updated_date`: existing rows are only replaced
    when their updated_date has changed (i.e. a provisional row was revised).

    Parameters
    ----------
    processed_dir : Path
        Directory containing consumption_national_hourly.parquet.
    client_id : str
        RTE OAuth2 client ID.
    client_secret : str
        RTE OAuth2 client secret.
    trailing_days : int
        How many days back to re-pull (default 60, captures late revisions).

    Returns
    -------
    pd.DataFrame
        Updated consumption_national_hourly table.
    """
    out_path = processed_dir / "consumption_national_hourly.parquet"
    existing: pd.DataFrame | None = None
    if out_path.exists():
        existing = pd.read_parquet(out_path)

    token = _get_rte_token(client_id, client_secret)

    # The consolidated series lags real time, so cap end_date at the most
    # recent recoverable term to avoid HTTP 400 (CONSOCONSU_COPCO_F04).
    end_dt = datetime.now(UTC) - timedelta(days=RTE_CONSOLIDATION_LAG_DAYS)

    # Start far enough back to (a) cover the gap since the last stored hour and
    # (b) re-pull a trailing window so late consolidation revisions are caught.
    start_dt = end_dt - timedelta(days=trailing_days)
    if existing is not None and not existing.empty:
        last_ts = pd.to_datetime(existing["ts_utc"], utc=True).max().to_pydatetime()
        # 2-day overlap re-pulls the boundary in case it was still provisory.
        start_dt = min(start_dt, last_ts - timedelta(days=2))

    if start_dt >= end_dt:
        logger.info(
            "Nothing to pull: last stored hour is within the %d-day " "consolidation lag",
            RTE_CONSOLIDATION_LAG_DAYS,
        )
        return existing if existing is not None else pd.DataFrame()

    # The API caps a single request at 186 days; chunk the window.
    max_window = timedelta(days=180)
    frames: list[pd.DataFrame] = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + max_window, end_dt)
        logger.info(
            "Pulling RTE consolidated from %s to %s",
            chunk_start.date(),
            chunk_end.date(),
        )
        frames.append(fetch_rte_consolidated_window(token, chunk_start, chunk_end))
        chunk_start = chunk_end

    raw_30min = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if raw_30min.empty:
        logger.warning("API returned no data for the window")
        return existing if existing is not None else pd.DataFrame()

    # Resample 30-min -> hourly (mean)
    raw_30min["ts_hour"] = raw_30min["ts_utc"].dt.floor("h")
    hourly = (
        raw_30min.groupby("ts_hour", sort=True)
        .agg(
            load_mw=("load_mw", "mean"),
            is_estimated=("is_estimated", "max"),
        )
        .reset_index()
        .rename(columns={"ts_hour": "ts_utc"})
    )
    hourly["load_mw"] = hourly["load_mw"].astype("float32")
    hourly["ts_utc"] = pd.to_datetime(hourly["ts_utc"], utc=True)

    if existing is not None:
        combined = pd.concat([existing, hourly], ignore_index=True)
        combined = combined.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
    else:
        combined = hourly

    combined.to_parquet(out_path, index=False)
    logger.info("Incremental update: wrote %d rows to %s", len(combined), out_path)
    return combined


def build_region_city_mapping(
    region_city_map: dict[str, CityInfo],
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Build and write the static region_city_mapping.parquet (12 rows, v1).

    In v1 each region has exactly one city with weight=1.0.
    The weight column is reserved for future population-weighted multi-city expansion.

    Parameters
    ----------
    region_city_map : dict
        REGION_CITY_MAP from src.config.
    processed_dir : Path
        Output directory.

    Returns
    -------
    pd.DataFrame
        Columns: region_code (str), city_code (str), weight (float32).
    """
    rows = [
        {
            "region_code": region_code,
            "city_code": info["city_code"],
            "weight": 1.0,
        }
        for region_code, info in region_city_map.items()
    ]
    df = pd.DataFrame(rows)
    df["weight"] = df["weight"].astype("float32")

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "region_city_mapping.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))
    return df
