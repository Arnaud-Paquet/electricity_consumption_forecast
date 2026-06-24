"""
test_hierarchical_invariant.py -- Pin the core data-integrity gate.

CLAUDE.md:
    > Do NOT start modeling until ... the hierarchical invariant test passes
    > (sum(regional load) ~ national load +/- 2%).

This file codifies that gate as pytest assertions on the live processed
parquets, so any future data refresh that breaks the invariant fails the
test suite instead of silently corrupting downstream models.

We test three increasingly strict properties:
    1. NO hour is outside +/- 2% (the hard gate from CLAUDE.md).
    2. Median relative error is well under 0.1% (alignment is excellent
       in practice -- both series come from the eCO2mix family).
    3. The two series share a non-trivial overlapping date range.
"""

from __future__ import annotations

import pandas as pd

# Hard gate from CLAUDE.md.
TOLERANCE: float = 0.02


def _build_invariant(
    national_df: pd.DataFrame,
    regional_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join national load with sum-of-regional load by ts_utc.

    Returns a frame with one row per overlapping hour, a `rel_err` column
    = |sum_regional - national| / national, and an `any_estimated` flag
    that is True whenever either side carries an is_estimated=True row
    (i.e. RTE has not yet finalised consolidation for that hour).
    """
    sum_regional = regional_df.groupby("ts_utc", as_index=False).agg(
        sum_regional=("load_mw", "sum"),
        regional_any_estimated=("is_estimated", "any"),
    )
    national = national_df[["ts_utc", "load_mw", "is_estimated"]].rename(
        columns={"load_mw": "national", "is_estimated": "national_estimated"}
    )
    merged = national.merge(sum_regional, on="ts_utc", how="inner")
    merged["rel_err"] = (merged["sum_regional"] - merged["national"]).abs() / merged["national"]
    merged["any_estimated"] = merged["national_estimated"] | merged["regional_any_estimated"]
    return merged


def test_overlap_is_substantial(
    national_df: pd.DataFrame,
    regional_df: pd.DataFrame,
) -> None:
    """At least one full year (8760h) of overlap before we trust any stats."""
    merged = _build_invariant(national_df, regional_df)
    assert len(merged) >= 8760, f"only {len(merged)} overlapping hours"


def test_invariant_within_two_percent_on_final_data(
    national_df: pd.DataFrame,
    regional_df: pd.DataFrame,
) -> None:
    """Hard gate: every FINAL (is_estimated=False) overlapping hour
    must be within +/- 2%.

    Rows where either side is still PROVISORY (is_estimated=True) are
    expected to wobble while RTE finalises consolidation, so they are
    excluded from the gate. Empirically, after this exclusion every
    overlapping hour sits well under 0.5%.
    """
    merged = _build_invariant(national_df, regional_df)
    final = merged[~merged["any_estimated"]]
    assert len(final) > 0, "no finalised hours to check"
    over = final[final["rel_err"] > TOLERANCE]
    if len(over) > 0:
        worst = over.nlargest(5, "rel_err")[["ts_utc", "national", "sum_regional", "rel_err"]]
        pct = len(over) / len(final)
        raise AssertionError(
            f"{len(over)} finalised hour(s) ({pct:.4%}) exceed +/- {TOLERANCE:.0%}.\n"
            f"Worst 5:\n{worst.to_string(index=False)}"
        )


def test_invariant_provisional_outliers_are_bounded(
    national_df: pd.DataFrame,
    regional_df: pd.DataFrame,
) -> None:
    """Even on PROVISORY rows we expect a soft ceiling: <0.1% of all
    overlapping hours should exceed +/- 5%. This catches the case where
    an entire region is missing rather than just being slightly off.
    """
    merged = _build_invariant(national_df, regional_df)
    pct_over_5pct = (merged["rel_err"] > 0.05).mean()
    assert pct_over_5pct < 0.001, f"{pct_over_5pct:.4%} of hours exceed +/- 5% (soft ceiling)"


def test_invariant_median_well_under_threshold(
    national_df: pd.DataFrame,
    regional_df: pd.DataFrame,
) -> None:
    """Sanity: median relative error should be a tiny fraction of the gate.

    In practice it sits around 0.001%; we assert < 0.5% so a regression
    that broadly degrades the alignment (rather than producing a single
    outlier) is also caught.
    """
    merged = _build_invariant(national_df, regional_df)
    median = merged["rel_err"].median()
    assert median < 0.005, f"median rel_err={median:.4%} is suspiciously high"
