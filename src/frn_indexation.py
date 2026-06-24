"""FRN opening accrual helpers for forecast input construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from bisect import bisect_right

import pandas as pd


def load_frn_daily_indexes(path: str | Path) -> pd.DataFrame:
    """Load FiscalData/Treasury FRN daily index rows with normalized types."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"FRN daily index source is missing: {source_path}")
    frame = pd.read_csv(source_path)
    required = {
        "record_date",
        "cusip",
        "original_issue_date",
        "maturity_date",
        "spread",
        "start_of_accrual_period",
        "end_of_accrual_period",
        "daily_index",
        "daily_int_accrual_rate",
        "daily_accrued_int_per100",
        "accr_int_per100_pmt_period",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FRN daily index source is missing required columns: {missing}")
    normalized = frame.copy()
    normalized["cusip"] = normalized["cusip"].astype(str).str.strip()
    for column in (
        "record_date",
        "original_issue_date",
        "maturity_date",
        "start_of_accrual_period",
        "end_of_accrual_period",
    ):
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce").dt.normalize()
    for column in (
        "spread",
        "daily_index",
        "daily_int_accrual_rate",
        "daily_accrued_int_per100",
        "accr_int_per100_pmt_period",
    ):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    invalid = (
        normalized["cusip"].eq("")
        | normalized["record_date"].isna()
        | normalized["start_of_accrual_period"].isna()
        | normalized["end_of_accrual_period"].isna()
        | normalized["spread"].isna()
        | normalized["daily_index"].isna()
        | normalized["accr_int_per100_pmt_period"].isna()
    )
    if invalid.any():
        raise ValueError(f"FRN daily index source contains {int(invalid.sum())} unusable rows")
    return normalized


def build_forward_frn_rate_path_rows(
    *,
    scenario_id: str,
    periods: list[Any],
    yield_surface_rows: list[dict[str, Any]],
    observation_date: str,
    available_date: str,
    benchmark_tenor_years: float = 0.25,
) -> list[dict[str, Any]]:
    """Build a compact daily FRN benchmark path from the CBO-anchored yield surface."""

    if not periods:
        raise ValueError("FRN rate path requires at least one simulation period")
    if not yield_surface_rows:
        raise ValueError("FRN rate path requires yield surface rows")
    rate_by_date: dict[pd.Timestamp, float] = {}
    for row in yield_surface_rows:
        tenor = float(row.get("tenor_years", float("nan")))
        if abs(tenor - benchmark_tenor_years) > 1e-9:
            continue
        curve_date = pd.Timestamp(row["curve_date"]).normalize()
        if "nominal_rate_decimal" in row and pd.notna(row["nominal_rate_decimal"]):
            rate_decimal = float(row["nominal_rate_decimal"])
        else:
            rate_decimal = float(row["nominal_rate"]) / 100.0
        rate_by_date[curve_date] = rate_decimal
    if not rate_by_date:
        raise ValueError(f"yield surface has no {benchmark_tenor_years}Y rows for FRN rate path")
    curve_dates = sorted(rate_by_date)

    rows: list[dict[str, Any]] = []
    for period in periods:
        period_start = pd.Timestamp(period.period_start).date().isoformat()
        period_end_ts = pd.Timestamp(period.period_end).normalize()
        period_end = period_end_ts.date().isoformat()
        source_pos = bisect_right(curve_dates, period_end_ts) - 1
        if source_pos < 0:
            raise ValueError(f"FRN rate path has no prior yield-surface row for period_end={period_end}")
        source_curve_date = curve_dates[source_pos]
        rate_decimal = rate_by_date[source_curve_date]
        if abs(rate_decimal) > 1.0:
            raise ValueError("FRN rate path source rate must be a decimal runtime rate")
        rows.append(
            {
                "schema_version": "tdcsim_frn_rate_path_v1",
                "scenario_id": scenario_id,
                "period_start": period_start,
                "period_end": period_end,
                "rate_effective_start": period_start,
                "rate_effective_end": period_end,
                "benchmark_tenor_years": benchmark_tenor_years,
                "auction_high_rate_decimal": rate_decimal,
                "benchmark_rate_decimal": rate_decimal,
                "money_market_yield_decimal": rate_decimal,
                "spread_treatment": "add_security_fixed_spread_decimal",
                "day_count_basis": 360.0,
                "lockout_business_days": 2.0,
                "rate_source_family": "cbo_yield_surface_3m_tbill_anchor",
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "observation_date": observation_date,
                "available_date": available_date,
                "source_status": (
                    "cbo_3m_tbill_path_used_as_forward_13week_frn_high_rate_assumption_"
                    f"latest_curve_date_{source_curve_date.date().isoformat()}"
                ),
                "claim_boundary": (
                    "future_frn_resets_are_contract_mechanics_driven_by_explicit_cbo_3m_rate_path_not_observed_auction_results"
                ),
            }
        )
    return rows


def enrich_opening_frn_from_daily_indexes(
    portfolio: pd.DataFrame,
    *,
    frn_daily_indexes_path: str | Path,
    opening_date: str | pd.Timestamp,
    source_available_as_of: str | pd.Timestamp | None = None,
    cusip_column: str = "_SourceCUSIP",
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Set opening FRN spread, benchmark, and accrued interest from daily index rows."""

    if cusip_column not in portfolio.columns:
        raise ValueError(f"opening portfolio must include {cusip_column!r} for FRN enrichment")
    enriched = portfolio.copy()
    frn_mask = enriched["SecurityType"].astype(str).eq("FRN")
    if not frn_mask.any():
        return enriched, _empty_frn_metadata(frn_daily_indexes_path, opening_date, source_available_as_of), []

    source = load_frn_daily_indexes(frn_daily_indexes_path)
    opening_ts = pd.Timestamp(opening_date).normalize()
    if source_available_as_of is not None:
        available_ts = pd.Timestamp(source_available_as_of).normalize()
        source = source.loc[source["record_date"].le(available_ts)].copy()
    else:
        available_ts = source["record_date"].max()
    if source.empty:
        raise ValueError("FRN daily index source has no rows after applying source_available_as_of")

    source_rows = _select_opening_rows(source, opening_ts)
    frn_cusips = enriched.loc[frn_mask, cusip_column].astype(str).str.strip()
    missing_cusips = sorted(set(frn_cusips) - set(source_rows.index))
    if missing_cusips:
        raise ValueError(
            f"FRN daily index source is missing {len(missing_cusips)} opening FRN CUSIPs "
            f"for opening_date={opening_ts.date()}: {missing_cusips[:20]}"
        )

    diagnostics: list[dict[str, Any]] = []
    for idx in enriched.index[frn_mask]:
        cusip = str(enriched.at[idx, cusip_column]).strip()
        source_row = source_rows.loc[cusip]
        face_value = float(pd.to_numeric(enriched.at[idx, "FaceValue"], errors="coerce"))
        accrued_per100 = float(source_row["accr_int_per100_pmt_period"])
        accrued = face_value * accrued_per100 / 100.0
        spread_decimal = float(source_row["spread"]) / 100.0
        daily_index_decimal = float(source_row["daily_index"]) / 100.0
        daily_accrual_rate_decimal = float(source_row["daily_int_accrual_rate"]) / 100.0
        enriched.at[idx, "FixedSpread"] = spread_decimal
        enriched.at[idx, "BenchmarkRate_FRN"] = daily_index_decimal
        enriched.at[idx, "AccruedInterest_FRN"] = accrued
        enriched.at[idx, "LastAccrualDate"] = source_row["end_of_accrual_period"]
        enriched.at[idx, "IssueYieldAtIssue"] = daily_index_decimal
        diagnostics.append(
            {
                "bond_id": int(enriched.at[idx, "BondID"]) if not pd.isna(enriched.at[idx, "BondID"]) else "",
                "cusip": cusip,
                "holder_type": str(enriched.at[idx, "HolderType"]),
                "holder_subbucket": str(enriched.at[idx, "HolderSubBucket"]),
                "opening_date": opening_ts.date().isoformat(),
                "source_record_date": pd.Timestamp(source_row["record_date"]).date().isoformat(),
                "source_accrual_start": pd.Timestamp(source_row["start_of_accrual_period"]).date().isoformat(),
                "source_accrual_end": pd.Timestamp(source_row["end_of_accrual_period"]).date().isoformat(),
                "source_spread_percent": float(source_row["spread"]),
                "source_daily_index_percent": float(source_row["daily_index"]),
                "source_daily_interest_accrual_rate_percent": float(source_row["daily_int_accrual_rate"]),
                "source_accrued_interest_per100": accrued_per100,
                "face_value_bil": face_value,
                "accrued_interest_frn_bil": accrued,
                "source_status": "fiscaldata_frn_daily_indexes_opening_accrual_join",
            }
        )

    selected = source_rows.loc[frn_cusips]
    total_face = float(pd.to_numeric(enriched.loc[frn_mask, "FaceValue"], errors="coerce").sum())
    total_accrued = float(pd.to_numeric(enriched.loc[frn_mask, "AccruedInterest_FRN"], errors="coerce").sum())
    metadata = {
        "frn_daily_indexes_source": str(Path(frn_daily_indexes_path)),
        "frn_opening_date": opening_ts.date().isoformat(),
        "frn_source_available_as_of": pd.Timestamp(available_ts).date().isoformat(),
        "frn_rows_enriched": int(frn_mask.sum()),
        "frn_unique_cusips_enriched": int(frn_cusips.nunique()),
        "frn_source_record_dates": sorted(
            {pd.Timestamp(value).date().isoformat() for value in selected["record_date"].tolist()}
        ),
        "frn_source_accrual_end_dates": sorted(
            {pd.Timestamp(value).date().isoformat() for value in selected["end_of_accrual_period"].tolist()}
        ),
        "opening_frn_face_value_bil": total_face,
        "opening_frn_accrued_interest_bil": total_accrued,
        "opening_frn_benchmark_rate_decimal_min": float(selected["daily_index"].min() / 100.0),
        "opening_frn_benchmark_rate_decimal_max": float(selected["daily_index"].max() / 100.0),
        "opening_frn_fixed_spread_decimal_min": float(selected["spread"].min() / 100.0),
        "opening_frn_fixed_spread_decimal_max": float(selected["spread"].max() / 100.0),
        "frn_source_boundary_note": (
            "FRN opening accrued interest uses FiscalData frn_daily_indexes rows available as of "
            f"{pd.Timestamp(available_ts).date().isoformat()}, selected by latest "
            f"end_of_accrual_period on or before opening_date {opening_ts.date().isoformat()}; "
            "the selected source is on or before actuals_available_as_of and is contract-compliant."
        ),
        "frn_initialization": (
            "fiscaldata_frn_daily_indexes_join_sets_opening_benchmark_spread_and_accrued_interest; "
            "selected_source_row_has_latest_accrual_period_end_on_or_before_opening_date"
        ),
    }
    return enriched, metadata, diagnostics


def _select_opening_rows(source: pd.DataFrame, opening_date: pd.Timestamp) -> pd.DataFrame:
    eligible = source.loc[source["end_of_accrual_period"].le(opening_date)].copy()
    if eligible.empty:
        return pd.DataFrame().set_index(pd.Index([], name="cusip"))
    eligible = eligible.sort_values(["cusip", "end_of_accrual_period", "record_date", "start_of_accrual_period"])
    return eligible.groupby("cusip", as_index=False).tail(1).set_index("cusip")


def _empty_frn_metadata(
    frn_daily_indexes_path: str | Path,
    opening_date: str | pd.Timestamp,
    source_available_as_of: str | pd.Timestamp | None,
) -> dict[str, Any]:
    opening_ts = pd.Timestamp(opening_date).normalize()
    return {
        "frn_daily_indexes_source": str(Path(frn_daily_indexes_path)),
        "frn_opening_date": opening_ts.date().isoformat(),
        "frn_source_available_as_of": (
            pd.Timestamp(source_available_as_of).date().isoformat()
            if source_available_as_of is not None
            else None
        ),
        "frn_rows_enriched": 0,
        "frn_unique_cusips_enriched": 0,
        "frn_source_record_dates": [],
        "frn_source_accrual_end_dates": [],
        "opening_frn_face_value_bil": 0.0,
        "opening_frn_accrued_interest_bil": 0.0,
        "opening_frn_benchmark_rate_decimal_min": 0.0,
        "opening_frn_benchmark_rate_decimal_max": 0.0,
        "opening_frn_fixed_spread_decimal_min": 0.0,
        "opening_frn_fixed_spread_decimal_max": 0.0,
        "frn_source_boundary_note": "no_opening_frn_rows",
        "frn_initialization": "no_opening_frn_rows",
    }
