"""TIPS CPI indexation helpers for forecast input construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from sim_pricing import calculate_tips_auction_coupon_rate


def load_tips_cpi_detail(path: str | Path) -> pd.DataFrame:
    """Load TreasuryDirect TIPS CPI detail rows with normalized types."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"TIPS CPI detail source is missing: {source_path}")
    frame = pd.read_csv(source_path)
    required = {"cusip", "original_issue_date", "index_date", "ref_cpi", "index_ratio"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"TIPS CPI detail source is missing required columns: {missing}")
    frame = frame.copy()
    frame["cusip"] = frame["cusip"].astype(str).str.strip()
    frame["original_issue_date"] = pd.to_datetime(frame["original_issue_date"], errors="coerce")
    frame["index_date"] = pd.to_datetime(frame["index_date"], errors="coerce").dt.normalize()
    frame["ref_cpi"] = pd.to_numeric(frame["ref_cpi"], errors="coerce")
    frame["index_ratio"] = pd.to_numeric(frame["index_ratio"], errors="coerce")
    invalid = frame["cusip"].eq("") | frame["index_date"].isna() | frame["ref_cpi"].isna() | frame["index_ratio"].isna()
    if invalid.any():
        raise ValueError(f"TIPS CPI detail source contains {int(invalid.sum())} unusable rows")
    return frame


def enrich_opening_tips_from_cpi_detail(
    portfolio: pd.DataFrame,
    *,
    cpi_detail_path: str | Path,
    index_date: str | pd.Timestamp,
    cusip_column: str = "_SourceCUSIP",
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Set opening TIPS original par and index ratios from TreasuryDirect detail.

    The input opening cohort book is already scaled to MSPD adjusted principal.
    This helper preserves that adjusted stock and reconstructs source-backed
    original par as ``AdjustedPrincipal / IndexRatio``.
    """

    if cusip_column not in portfolio.columns:
        raise ValueError(f"opening portfolio must include {cusip_column!r} for TIPS enrichment")
    enriched = portfolio.copy()
    tips_mask = enriched["SecurityType"].astype(str).eq("TIPS")
    if not tips_mask.any():
        return enriched, _empty_tips_metadata(cpi_detail_path, index_date), []

    detail = load_tips_cpi_detail(cpi_detail_path)
    target_date = pd.Timestamp(index_date).normalize()
    detail_on_date = detail.loc[detail["index_date"].eq(target_date)].copy()
    if detail_on_date.empty:
        raise ValueError(f"TIPS CPI detail has no rows for index_date={target_date.date()}")
    duplicate_mask = detail_on_date.duplicated(["cusip"], keep=False)
    if duplicate_mask.any():
        duplicates = sorted(detail_on_date.loc[duplicate_mask, "cusip"].unique().tolist())
        raise ValueError(f"TIPS CPI detail has duplicate CUSIP rows for {target_date.date()}: {duplicates[:10]}")

    source_by_cusip = detail_on_date.set_index("cusip")
    tips_cusips = enriched.loc[tips_mask, cusip_column].astype(str).str.strip()
    missing_cusips = sorted(set(tips_cusips) - set(source_by_cusip.index))
    if missing_cusips:
        raise ValueError(
            f"TIPS CPI detail is missing {len(missing_cusips)} opening TIPS CUSIPs for "
            f"{target_date.date()}: {missing_cusips[:20]}"
        )

    diagnostics: list[dict[str, Any]] = []
    for idx in enriched.index[tips_mask]:
        cusip = str(enriched.at[idx, cusip_column]).strip()
        source = source_by_cusip.loc[cusip]
        index_ratio = float(source["index_ratio"])
        if index_ratio <= 0.0:
            raise ValueError(f"TIPS CPI detail has nonpositive index_ratio for {cusip}: {index_ratio}")
        source_ref_cpi = float(source["ref_cpi"])
        adjusted = float(pd.to_numeric(enriched.at[idx, "AdjustedPrincipal"], errors="coerce"))
        original = adjusted / index_ratio
        reference_cpi_issue = source_ref_cpi / index_ratio
        enriched.at[idx, "FaceValue"] = original
        enriched.at[idx, "OriginalPrincipal"] = original
        enriched.at[idx, "ReferenceCPI_Issue"] = reference_cpi_issue
        enriched.at[idx, "IndexRatio"] = index_ratio
        diagnostics.append(
            {
                "bond_id": int(enriched.at[idx, "BondID"]) if not pd.isna(enriched.at[idx, "BondID"]) else "",
                "cusip": cusip,
                "holder_type": str(enriched.at[idx, "HolderType"]),
                "holder_subbucket": str(enriched.at[idx, "HolderSubBucket"]),
                "maturity_date": pd.Timestamp(enriched.at[idx, "MaturityDate"]).date().isoformat(),
                "index_date": target_date.date().isoformat(),
                "source_original_issue_date": (
                    pd.Timestamp(source["original_issue_date"]).date().isoformat()
                    if not pd.isna(source["original_issue_date"])
                    else ""
                ),
                "source_ref_cpi": source_ref_cpi,
                "source_index_ratio": index_ratio,
                "adjusted_principal_bil": adjusted,
                "reconstructed_original_principal_bil": original,
                "reconstructed_reference_cpi_issue": reference_cpi_issue,
                "source_status": "treasurydirect_tips_cpi_detail_index_date_join",
            }
        )

    ref_cpi_values = sorted({round(float(v), 5) for v in source_by_cusip.loc[tips_cusips, "ref_cpi"].tolist()})
    total_adjusted = float(pd.to_numeric(enriched.loc[tips_mask, "AdjustedPrincipal"], errors="coerce").sum())
    total_original = float(pd.to_numeric(enriched.loc[tips_mask, "OriginalPrincipal"], errors="coerce").sum())
    metadata = {
        "tips_cpi_detail_source": str(Path(cpi_detail_path)),
        "tips_cpi_detail_index_date": target_date.date().isoformat(),
        "tips_rows_enriched": int(tips_mask.sum()),
        "tips_unique_cusips_enriched": int(tips_cusips.nunique()),
        "tips_source_ref_cpi_values": ref_cpi_values,
        "opening_tips_reference_cpi": float(ref_cpi_values[0]) if len(ref_cpi_values) == 1 else ref_cpi_values,
        "opening_tips_adjusted_principal_bil": total_adjusted,
        "opening_tips_reconstructed_original_principal_bil": total_original,
        "opening_tips_indexation_accretion_embedded_bil": total_adjusted - total_original,
        "tips_initialization": (
            "treasurydirect_tips_cpi_detail_join_preserves_mspd_adjusted_principal; "
            "opening_original_principal_reconstructed_as_adjusted_principal_divided_by_source_index_ratio"
        ),
    }
    return enriched, metadata, diagnostics


def build_projected_cpi_lookup_from_macro(
    macro_frame: pd.DataFrame | None,
    dates: Sequence[pd.Timestamp],
    *,
    scenario_id: str,
    default_value: float,
    lag_months: int = 0,
    anchor_date: str | pd.Timestamp | None = None,
    anchor_value: float | None = None,
) -> dict[pd.Timestamp, float]:
    """Build an interpolated CBO CPI lookup, optionally lagged and source-anchored."""

    if macro_frame is None or macro_frame.empty or "cbo_cpi_u_index" not in macro_frame.columns:
        return {}
    rows = _filter_scenario_rows(macro_frame, scenario_id)
    if rows.empty:
        return {}
    normalized = rows.copy()
    normalized["period_start_ts"] = pd.to_datetime(normalized.get("period_start"), errors="coerce").dt.normalize()
    normalized["cbo_cpi_u_index"] = pd.to_numeric(normalized["cbo_cpi_u_index"], errors="coerce")
    normalized = normalized.dropna(subset=["period_start_ts", "cbo_cpi_u_index"]).sort_values("period_start_ts")
    if normalized.empty:
        return {}

    x = normalized["period_start_ts"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    y = normalized["cbo_cpi_u_index"].to_numpy(dtype=float)

    def raw_value(value: str | pd.Timestamp) -> float:
        target = pd.Timestamp(value).normalize() - pd.DateOffset(months=int(lag_months))
        ordinal = float(target.toordinal())
        return float(np.interp(ordinal, x, y, left=y[0], right=y[-1]))

    scalar = 1.0
    if anchor_date is not None and anchor_value is not None:
        anchor_raw = raw_value(anchor_date)
        if anchor_raw > 0.0:
            scalar = float(anchor_value) / anchor_raw

    lookup: dict[pd.Timestamp, float] = {}
    for value in dates:
        target_date = pd.Timestamp(value).normalize()
        lookup[target_date] = raw_value(target_date) * scalar
    return lookup


def build_monthly_tips_cpi_path_rows(
    *,
    scenario_id: str,
    macro_forecast_rows: Sequence[Mapping[str, Any]],
    simulation_start_date: str | pd.Timestamp,
    simulation_end_date: str | pd.Timestamp,
    opening_reference_cpi: float,
    reference_lag_months: int = 3,
    pricing_horizon_end_date: str | pd.Timestamp | None = None,
    observation_date: str,
    available_date: str,
) -> list[dict[str, Any]]:
    """Build a compact monthly TIPS CPI path from CBO CPI-U anchors.

    CBO quarterly CPI-U anchors stop at the forecast horizon. TIPS pricing can
    require CPI years beyond that horizon, so months after the final CBO anchor
    are explicit TDCSIM terminal extrapolations using the final observed CBO CPI
    segment's annualized growth rate.
    """

    if not macro_forecast_rows:
        raise ValueError("TIPS CPI path requires macro forecast rows")
    macro = pd.DataFrame(macro_forecast_rows)
    rows = _filter_scenario_rows(macro, scenario_id)
    if rows.empty:
        raise ValueError(f"TIPS CPI path has no macro rows for scenario {scenario_id!r}")
    rows = rows.copy()
    rows["period_start_ts"] = pd.to_datetime(rows["period_start"], errors="coerce").dt.normalize()
    rows["cbo_cpi_u_index"] = pd.to_numeric(rows["cbo_cpi_u_index"], errors="coerce")
    rows = rows.dropna(subset=["period_start_ts", "cbo_cpi_u_index"]).sort_values("period_start_ts")
    if rows.empty:
        raise ValueError("TIPS CPI path macro rows have no usable CPI-U anchors")

    sim_start = pd.Timestamp(simulation_start_date).normalize()
    sim_end = pd.Timestamp(simulation_end_date).normalize()
    pricing_end = pd.Timestamp(pricing_horizon_end_date).normalize() if pricing_horizon_end_date is not None else sim_end
    first_month = (sim_start - pd.DateOffset(months=int(reference_lag_months))).replace(day=1)
    terminal_growth = _terminal_annualized_cpi_growth(rows)
    terminal_anchor_month = pd.Timestamp(rows.iloc[-1]["period_start_ts"]).normalize().replace(day=1)
    terminal_anchor_value = float(rows.iloc[-1]["cbo_cpi_u_index"])
    terminal_source_start = pd.Timestamp(rows.iloc[-2]["period_start_ts"]).normalize().date().isoformat()
    terminal_source_end = pd.Timestamp(rows.iloc[-1]["period_start_ts"]).normalize().date().isoformat()
    terminal_start_month = terminal_anchor_month + pd.DateOffset(months=1)
    # Include the following month so daily linear interpolation never needs to
    # synthesize a flat right endpoint at the maximum pricing horizon.
    last_required_date = max(sim_end, pricing_end)
    last_month = last_required_date.replace(day=1) + pd.DateOffset(months=1)
    months = pd.date_range(first_month, last_month, freq="MS")
    x = rows["period_start_ts"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    y = rows["cbo_cpi_u_index"].to_numpy(dtype=float)
    raw_by_month = {}
    for month in months:
        month_ts = pd.Timestamp(month).normalize()
        if month_ts <= terminal_anchor_month:
            raw_by_month[month_ts] = float(
                np.interp(float(month_ts.toordinal()), x, y, left=y[0], right=y[-1])
            )
        else:
            months_after_anchor = _months_between(terminal_anchor_month, month_ts)
            raw_by_month[month_ts] = float(
                terminal_anchor_value * ((1.0 + terminal_growth) ** (months_after_anchor / 12.0))
            )
    terminal_growth_label = f"{terminal_growth:.12f}"
    terminal_rule = (
        "tdcsim_terminal_extrapolation_beyond_cbo_horizon_final_cbo_cpi_segment_annualized_growth"
    )
    source_status_by_role = {
        "cbo_interpolated": "cbo_quarterly_cpi_u_interpolated_to_monthly_tips_nsa_assumption",
        "tdcsim_terminal": (
            f"{terminal_rule}_{terminal_growth_label}_"
            f"source_segment_{terminal_source_start}_to_{terminal_source_end}"
        ),
    }
    claim_boundary_by_role = {
        "cbo_interpolated": (
            "future_tips_cpi_uses_cbo_cpi_u_monthly_interpolation_not_observed_future_nsa_cpi"
        ),
        "tdcsim_terminal": (
            "post_cbo_horizon_tips_cpi_is_tdcsim_terminal_extrapolation_not_cbo_cpi_coverage"
        ),
    }
    anchor_raw = _reference_cpi_for_date_from_monthly(
        raw_by_month,
        sim_start,
        lag_months=reference_lag_months,
    )
    scale_factor = float(opening_reference_cpi) / anchor_raw if anchor_raw > 0.0 else 1.0
    output: list[dict[str, Any]] = []
    for month in months:
        month_ts = pd.Timestamp(month).normalize()
        cbo_value = raw_by_month[month_ts]
        horizon_role = "cbo_interpolated" if month_ts <= terminal_anchor_month else "tdcsim_terminal"
        output.append(
            {
                "schema_version": "tdcsim_tips_cpi_path_v1",
                "scenario_id": scenario_id,
                "month": month_ts.date().isoformat(),
                "cbo_cpi_u_index": cbo_value,
                "tips_cpi_u_index": cbo_value * scale_factor,
                "reference_lag_months": float(reference_lag_months),
                "interpolation_method": "treasury_daily_linear_reference_cpi",
                "anchor_date": sim_start.date().isoformat(),
                "anchor_reference_cpi": float(opening_reference_cpi),
                "scale_factor": scale_factor,
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "observation_date": observation_date,
                "available_date": available_date,
                "cpi_horizon_role": horizon_role,
                "terminal_cbo_anchor_month": terminal_anchor_month.date().isoformat(),
                "terminal_extrapolation_start_month": terminal_start_month.date().isoformat(),
                "terminal_annualized_cpi_growth_decimal": terminal_growth,
                "terminal_growth_source_period_start": terminal_source_start,
                "terminal_growth_source_period_end": terminal_source_end,
                "terminal_cpi_rule": terminal_rule,
                "source_status": source_status_by_role[horizon_role],
                "claim_boundary": claim_boundary_by_role[horizon_role],
            }
        )
    return output


def build_tips_daily_cpi_lookups_from_monthly_path(
    monthly_path: pd.DataFrame | None,
    dates: Sequence[pd.Timestamp],
    *,
    scenario_id: str,
    default_value: float,
    reference_lag_months: int = 3,
) -> tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, float]]:
    """Build daily CPI and TIPS reference-CPI lookups from a monthly CPI path."""

    if monthly_path is None or monthly_path.empty:
        return ({}, {})
    rows = _filter_scenario_rows(monthly_path, scenario_id)
    if rows.empty:
        return ({}, {})
    month_values = _monthly_tips_cpi_map(rows, default_value=default_value)
    cpi_lookup: dict[pd.Timestamp, float] = {}
    ref_lookup: dict[pd.Timestamp, float] = {}
    for value in dates:
        target = pd.Timestamp(value).normalize()
        cpi_lookup[target] = _daily_cpi_for_date_from_monthly(month_values, target)
        ref_lookup[target] = _reference_cpi_for_date_from_monthly(
            month_values,
            target,
            lag_months=reference_lag_months,
        )
    return cpi_lookup, ref_lookup


def build_tips_real_yield_path_rows(
    *,
    scenario_id: str,
    yield_surface_rows: Sequence[Mapping[str, Any]],
    tips_cpi_path_rows: Sequence[Mapping[str, Any]],
    observation_date: str,
    available_date: str,
) -> list[dict[str, Any]]:
    """Build real-yield assumptions from nominal CBO curve rows and expected CPI inflation."""

    if not yield_surface_rows:
        raise ValueError("TIPS real-yield path requires yield surface rows")
    if not tips_cpi_path_rows:
        raise ValueError("TIPS real-yield path requires TIPS CPI path rows")
    month_values = _monthly_tips_cpi_map(pd.DataFrame(tips_cpi_path_rows), default_value=100.0)
    terminal_rule = _terminal_rule_metadata_from_cpi_rows(tips_cpi_path_rows)
    output: list[dict[str, Any]] = []
    for row in yield_surface_rows:
        curve_date = pd.Timestamp(row["curve_date"]).normalize()
        tenor = float(row["tenor_years"])
        nominal_rate_decimal = (
            float(row["nominal_rate_decimal"])
            if "nominal_rate_decimal" in row and pd.notna(row["nominal_rate_decimal"])
            else float(row["nominal_rate"]) / 100.0
        )
        horizon_date = _expected_inflation_horizon_date(curve_date, tenor)
        expected_inflation = _expected_inflation_decimal(month_values, curve_date, tenor)
        real_yield = nominal_rate_decimal - expected_inflation
        output.append(
            {
                "schema_version": "tdcsim_tips_real_yield_path_v1",
                "scenario_id": scenario_id,
                "curve_date": curve_date.date().isoformat(),
                "tenor_years": tenor,
                "nominal_rate_decimal": nominal_rate_decimal,
                "expected_inflation_decimal": expected_inflation,
                "real_yield_decimal": real_yield,
                "real_coupon_decimal": calculate_tips_auction_coupon_rate(real_yield),
                "expected_inflation_horizon_date": horizon_date.date().isoformat(),
                "expected_inflation_cpi_terminal_rule": terminal_rule.get("terminal_cpi_rule", ""),
                "terminal_annualized_cpi_growth_decimal": terminal_rule.get(
                    "terminal_annualized_cpi_growth_decimal",
                    "",
                ),
                "terminal_cbo_anchor_month": terminal_rule.get("terminal_cbo_anchor_month", ""),
                "terminal_extrapolation_start_month": terminal_rule.get("terminal_extrapolation_start_month", ""),
                "pricing_method": "real_cashflow_present_value_semiannual",
                "coupon_rounding": "floor_to_1_8_percent_min_1_8_percent",
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "observation_date": observation_date,
                "available_date": available_date,
                "source_status": (
                    "nominal_cbo_yield_less_tdcsim_expected_inflation_with_declared_terminal_cpi_rule"
                ),
                "claim_boundary": (
                    "future_tips_real_yields_are_derived_assumptions_not_observed_tips_auction_high_yields;"
                    "post_cbo_horizon_expected_inflation_uses_tdcsim_terminal_cpi_extrapolation"
                ),
            }
        )
    return output


def _monthly_tips_cpi_map(frame: pd.DataFrame, *, default_value: float) -> dict[pd.Timestamp, float]:
    rows = frame.copy()
    rows["month_ts"] = pd.to_datetime(rows["month"], errors="coerce").dt.normalize()
    value_column = "tips_cpi_u_index" if "tips_cpi_u_index" in rows.columns else "cbo_cpi_u_index"
    rows[value_column] = pd.to_numeric(rows[value_column], errors="coerce")
    rows = rows.dropna(subset=["month_ts", value_column]).sort_values("month_ts")
    if rows.empty:
        return {}
    values = {
        pd.Timestamp(row["month_ts"]).normalize(): float(row[value_column])
        for _, row in rows.iterrows()
    }
    if not values:
        values[pd.Timestamp("1900-01-01")] = float(default_value)
    return values


def _daily_cpi_for_date_from_monthly(month_values: Mapping[pd.Timestamp, float], target_date: pd.Timestamp) -> float:
    month = pd.Timestamp(target_date).normalize().replace(day=1)
    next_month = month + pd.DateOffset(months=1)
    return _interpolate_month_pair(month_values, month, next_month, target_date)


def _reference_cpi_for_date_from_monthly(
    month_values: Mapping[pd.Timestamp, float],
    target_date: pd.Timestamp,
    *,
    lag_months: int,
) -> float:
    target = pd.Timestamp(target_date).normalize()
    lagged_month = (target.replace(day=1) - pd.DateOffset(months=int(lag_months))).normalize()
    next_lagged_month = lagged_month + pd.DateOffset(months=1)
    interpolation_date = lagged_month + pd.Timedelta(days=int(target.day) - 1)
    return _interpolate_month_pair(month_values, lagged_month, next_lagged_month, interpolation_date)


def _interpolate_month_pair(
    month_values: Mapping[pd.Timestamp, float],
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    target_date: pd.Timestamp,
) -> float:
    if not month_values:
        return 0.0
    start = pd.Timestamp(start_month).normalize()
    end = pd.Timestamp(end_month).normalize()
    available = sorted(pd.Timestamp(key).normalize() for key in month_values)

    def value_for(month: pd.Timestamp) -> float:
        if month in month_values:
            return float(month_values[month])
        prior = [candidate for candidate in available if candidate <= month]
        if prior:
            return float(month_values[prior[-1]])
        return float(month_values[available[0]])

    start_value = value_for(start)
    end_value = value_for(end)
    days_in_month = max(1, int((end - start).days))
    day_index = max(0, min(days_in_month, int((pd.Timestamp(target_date).normalize() - start).days)))
    return start_value + (end_value - start_value) * day_index / days_in_month


def _expected_inflation_decimal(
    month_values: Mapping[pd.Timestamp, float],
    curve_date: pd.Timestamp,
    tenor_years: float,
) -> float:
    horizon_years = max(1.0 / 12.0, float(tenor_years))
    start_value = _daily_cpi_for_date_from_monthly(month_values, curve_date)
    end_date = _expected_inflation_horizon_date(curve_date, horizon_years)
    _require_cpi_month_coverage(month_values, curve_date)
    _require_cpi_month_coverage(month_values, end_date)
    end_value = _daily_cpi_for_date_from_monthly(month_values, end_date)
    if start_value <= 0.0 or end_value <= 0.0:
        return 0.0
    return float((end_value / start_value) ** (1.0 / horizon_years) - 1.0)


def _expected_inflation_horizon_date(curve_date: pd.Timestamp, tenor_years: float) -> pd.Timestamp:
    horizon_years = max(1.0 / 12.0, float(tenor_years))
    return pd.Timestamp(curve_date).normalize() + relativedelta(months=max(1, int(round(horizon_years * 12))))


def _require_cpi_month_coverage(month_values: Mapping[pd.Timestamp, float], target_date: pd.Timestamp) -> None:
    if not month_values:
        raise ValueError("TIPS expected inflation requires a nonempty monthly CPI path")
    target = pd.Timestamp(target_date).normalize()
    target_month = target.replace(day=1)
    required_month = target_month if target.day == 1 else target_month + pd.DateOffset(months=1)
    available = sorted(pd.Timestamp(key).normalize() for key in month_values)
    if required_month > available[-1]:
        raise ValueError(
            "TIPS expected inflation CPI path ends before requested pricing horizon: "
            f"required_month={required_month.date().isoformat()} "
            f"last_available_month={available[-1].date().isoformat()}"
        )


def _terminal_annualized_cpi_growth(rows: pd.DataFrame) -> float:
    if len(rows) < 2:
        raise ValueError("TIPS terminal CPI rule requires at least two CBO CPI-U anchors")
    prior = rows.iloc[-2]
    latest = rows.iloc[-1]
    prior_value = float(prior["cbo_cpi_u_index"])
    latest_value = float(latest["cbo_cpi_u_index"])
    if prior_value <= 0.0 or latest_value <= 0.0:
        raise ValueError("TIPS terminal CPI rule requires positive final CBO CPI-U anchors")
    prior_date = pd.Timestamp(prior["period_start_ts"]).normalize()
    latest_date = pd.Timestamp(latest["period_start_ts"]).normalize()
    months = _months_between(prior_date, latest_date)
    if months <= 0:
        raise ValueError("TIPS terminal CPI rule requires increasing final CBO CPI-U anchor dates")
    return float((latest_value / prior_value) ** (12.0 / months) - 1.0)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    return (end_ts.year - start_ts.year) * 12 + end_ts.month - start_ts.month


def _terminal_rule_metadata_from_cpi_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if "terminal_cpi_rule" in row:
            return {
                "terminal_cpi_rule": row.get("terminal_cpi_rule", ""),
                "terminal_annualized_cpi_growth_decimal": row.get("terminal_annualized_cpi_growth_decimal", ""),
                "terminal_cbo_anchor_month": row.get("terminal_cbo_anchor_month", ""),
                "terminal_extrapolation_start_month": row.get("terminal_extrapolation_start_month", ""),
            }
    return {}


def _filter_scenario_rows(frame: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    if "scenario_id" not in frame.columns:
        return frame.copy()
    values = frame["scenario_id"].fillna("").astype(str)
    for candidate in (str(scenario_id), "all", "default", ""):
        rows = frame.loc[values.eq(candidate)]
        if not rows.empty:
            return rows.copy()
    return pd.DataFrame()


def _empty_tips_metadata(cpi_detail_path: str | Path, index_date: str | pd.Timestamp) -> dict[str, Any]:
    target_date = pd.Timestamp(index_date).normalize()
    return {
        "tips_cpi_detail_source": str(Path(cpi_detail_path)),
        "tips_cpi_detail_index_date": target_date.date().isoformat(),
        "tips_rows_enriched": 0,
        "tips_unique_cusips_enriched": 0,
        "tips_source_ref_cpi_values": [],
        "opening_tips_reference_cpi": None,
        "opening_tips_adjusted_principal_bil": 0.0,
        "opening_tips_reconstructed_original_principal_bil": 0.0,
        "opening_tips_indexation_accretion_embedded_bil": 0.0,
        "tips_initialization": "no_opening_tips_rows",
    }
