"""Dynamic yield-curve path helpers for tdcsim contract mode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_CURVE_COLUMNS = {
    "scenario_id",
    "curve_date",
    "tenor_years",
    "nominal_rate",
}

RATE_UNIT_PERCENT_POINTS = "percent_points"
RATE_UNIT_DECIMAL = "decimal"


def load_yield_curve_surface(path: str | Path | None) -> pd.DataFrame:
    """Load a date x tenor yield-curve surface."""

    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"yield curve surface file is missing: {path}")
    surface = pd.read_csv(path)
    missing = REQUIRED_CURVE_COLUMNS - set(surface.columns)
    if missing:
        raise ValueError(f"yield curve surface missing columns: {sorted(missing)}")
    surface = surface.copy()
    surface["curve_date"] = pd.to_datetime(surface["curve_date"])
    surface["tenor_years"] = pd.to_numeric(surface["tenor_years"], errors="raise")
    surface["nominal_rate"] = pd.to_numeric(surface["nominal_rate"], errors="raise")
    rate_unit = _single_rate_unit(surface)
    if "nominal_rate_decimal" in surface.columns:
        surface["nominal_rate_decimal"] = pd.to_numeric(surface["nominal_rate_decimal"], errors="raise")
        if rate_unit == RATE_UNIT_PERCENT_POINTS:
            expected = surface["nominal_rate"] / 100.0
        elif rate_unit == RATE_UNIT_DECIMAL:
            expected = surface["nominal_rate"]
        elif rate_unit:
            raise ValueError(
                "yield curve surface must use rate_unit percent_points/decimal when "
                "nominal_rate_decimal is present"
            )
        else:
            expected = _infer_legacy_decimal_rates(surface["nominal_rate"])
        max_error = (surface["nominal_rate_decimal"] - expected).abs().max()
        if pd.notna(max_error) and max_error > 1e-12:
            raise ValueError("yield curve nominal_rate_decimal must match nominal_rate and rate_unit")
    else:
        if rate_unit == RATE_UNIT_PERCENT_POINTS:
            surface["nominal_rate_decimal"] = surface["nominal_rate"] / 100.0
        elif rate_unit == RATE_UNIT_DECIMAL:
            surface["nominal_rate_decimal"] = surface["nominal_rate"]
        elif not rate_unit:
            surface["nominal_rate_decimal"] = _infer_legacy_decimal_rates(surface["nominal_rate"])
        else:
            raise ValueError(
                "yield curve surface must include nominal_rate_decimal or a rate_unit "
                "of percent_points/decimal"
            )
    if "runtime_rate_unit" in surface.columns:
        runtime_units = set(surface["runtime_rate_unit"].dropna().astype(str))
        if runtime_units and runtime_units != {RATE_UNIT_DECIMAL}:
            raise ValueError("yield curve runtime_rate_unit must be decimal")
    if (surface["nominal_rate_decimal"].abs() > 1.0).any():
        raise ValueError("yield curve decimal rates must not exceed 1.0 in absolute value")
    return surface.sort_values(["scenario_id", "curve_date", "tenor_years"]).reset_index(drop=True)


def curve_for_date(
    surface: pd.DataFrame,
    date,
    *,
    scenario_id: str | None = None,
) -> tuple[list[float], list[float], str]:
    """Return the latest curve on or before date for a scenario."""

    if surface is None or surface.empty:
        return ([], [], "static_curve_no_surface")
    current_date = pd.to_datetime(date)
    candidates = surface[surface["curve_date"] <= current_date].copy()
    if scenario_id and "scenario_id" in candidates.columns:
        scenario_candidates = candidates[candidates["scenario_id"].astype(str) == str(scenario_id)]
        if scenario_candidates.empty:
            raise ValueError(f"yield curve surface has no prior rows for scenario {scenario_id!r}")
        candidates = scenario_candidates
    if candidates.empty:
        return ([], [], "dynamic_curve_no_prior_date")
    curve_date = candidates["curve_date"].max()
    curve = candidates[candidates["curve_date"] == curve_date].sort_values("tenor_years")
    return (
        [float(v) for v in curve["tenor_years"].tolist()],
        [float(v) for v in curve["nominal_rate_decimal"].tolist()],
        f"dynamic_curve_surface:{curve_date.date()}",
    )


def _single_rate_unit(surface: pd.DataFrame) -> str:
    if "rate_unit" not in surface.columns:
        return ""
    units = set(surface["rate_unit"].dropna().astype(str))
    if len(units) != 1:
        raise ValueError("yield curve surface must use one rate_unit")
    return next(iter(units))


def _infer_legacy_decimal_rates(rates: pd.Series) -> pd.Series:
    nonnull = rates.dropna().abs()
    if nonnull.empty:
        raise ValueError("yield curve surface has no nominal_rate values")
    has_decimal_like = (nonnull <= 1.0).any()
    has_percent_like = (nonnull > 1.0).any()
    if has_decimal_like and has_percent_like:
        raise ValueError(
            "legacy yield curve surface has ambiguous mixed decimal and percent-point rates"
        )
    if has_percent_like:
        return rates / 100.0
    return rates
