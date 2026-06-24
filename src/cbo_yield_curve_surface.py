"""CBO yield-curve surface construction helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
import math
from typing import Any


SCHEMA_VERSION = "tdcsim_yield_curve_surface_v1"
CONSTRUCTION_METHOD = "frozen_reference_log_tenor_shift_tilt_pchip_v1"
TEMPORAL_FILL_METHOD = "piecewise_constant_within_quarter"
INTERPOLATION_METHOD = "pchip"
CURVE_BASIS = "nominal_par_yield_coupon_setting_surface"


def build_yield_curve_surface_rows(
    *,
    macro_forecast_rows: Iterable[Mapping[str, Any]],
    base_curve_rows: Iterable[Mapping[str, Any]],
    actuals_available_as_of: Any,
    output_tenors: Sequence[float] | None = None,
    allow_lookahead: bool = False,
) -> list[dict[str, Any]]:
    """Build quarterly CBO-anchored nominal par-yield surface rows.

    Rates are represented in percentage points, matching the CBO 3-month and
    10-year macro anchors.
    """

    cutoff = _to_date(actuals_available_as_of)
    base_curve = _normalize_base_curve(base_curve_rows)
    if not allow_lookahead:
        if base_curve["base_curve_date"] > cutoff:
            raise ValueError("base curve date is after actuals_available_as_of")
        if base_curve["available_date"] > cutoff:
            raise ValueError("base curve available_date is after actuals_available_as_of")

    tenors = tuple(float(value) for value in (output_tenors or base_curve["tenors"]))
    if 0.25 not in tenors or 10.0 not in tenors:
        raise ValueError("output_tenors must include 0.25 and 10.0 year anchors")
    _validate_positive_unique_tenors(tenors, label="output_tenors")

    base_3m = _pchip_value(base_curve["tenors"], base_curve["rates"], 0.25)
    base_10y = _pchip_value(base_curve["tenors"], base_curve["rates"], 10.0)
    rows: list[dict[str, Any]] = []
    for macro_row in macro_forecast_rows:
        scenario_id = str(macro_row["scenario_id"])
        curve_date = str(macro_row.get("period_start") or macro_row.get("curve_date"))
        anchor_3m = float(macro_row["cbo_3m_tbill_rate_pct"])
        anchor_10y = float(macro_row["cbo_10y_treasury_rate_pct"])
        available_date = str(macro_row.get("available_date") or base_curve["available_date"].isoformat())
        for tenor in tenors:
            base_rate = _pchip_value(base_curve["tenors"], base_curve["rates"], tenor)
            weight = _log_tenor_weight(tenor)
            nominal_rate = (
                base_rate
                + (1.0 - weight) * (anchor_3m - base_3m)
                + weight * (anchor_10y - base_10y)
            )
            if math.isclose(tenor, 0.25, rel_tol=0.0, abs_tol=1e-14):
                nominal_rate = anchor_3m
            elif math.isclose(tenor, 10.0, rel_tol=0.0, abs_tol=1e-14):
                nominal_rate = anchor_10y
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario_id": scenario_id,
                    "curve_date": curve_date,
                    "tenor_years": tenor,
                    "nominal_rate": nominal_rate,
                    "nominal_rate_decimal": nominal_rate / 100.0,
                    "rate_unit": "percent_points",
                    "runtime_rate_unit": "decimal",
                    "curve_basis": CURVE_BASIS,
                    "base_curve_date": base_curve["base_curve_date"].isoformat(),
                    "base_curve_source_key": base_curve["base_curve_source_key"],
                    "base_curve_sha256": base_curve["base_curve_sha256"],
                    "anchor_3m_pct": anchor_3m,
                    "anchor_10y_pct": anchor_10y,
                    "construction_method": CONSTRUCTION_METHOD,
                    "temporal_fill_method": TEMPORAL_FILL_METHOD,
                    "interpolation_method": INTERPOLATION_METHOD,
                    "source_role": "scenario_assumption",
                    "runtime_role": "hard_target",
                    "available_date": available_date,
                    "source_status": "frozen_reference_curve_cbo_quarterly_anchor",
                    "claim_boundary": "nominal_par_yield_surface_not_bootstrapped_zero_curve",
                }
            )
    if not rows:
        raise ValueError("macro_forecast_rows must not be empty")
    return rows


def _normalize_base_curve(base_curve_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(base_curve_rows)
    if not rows:
        raise ValueError("base_curve_rows must not be empty")
    first = rows[0]
    base_curve_date = _to_date(first.get("base_curve_date") or first.get("curve_date"))
    available_date = _to_date(first.get("available_date"))
    source_key = str(first.get("base_curve_source_key") or first.get("source_key") or "")
    sha256 = str(first.get("base_curve_sha256") or first.get("source_sha256") or first.get("sha256") or "")
    if not source_key:
        raise ValueError("base curve source key is required")
    if not sha256:
        raise ValueError("base curve SHA-256 is required")

    pairs: list[tuple[float, float]] = []
    for row in rows:
        row_date = _to_date(row.get("base_curve_date") or row.get("curve_date"))
        row_available = _to_date(row.get("available_date"))
        row_source_key = str(row.get("base_curve_source_key") or row.get("source_key") or "")
        row_sha256 = str(row.get("base_curve_sha256") or row.get("source_sha256") or row.get("sha256") or "")
        if row_date != base_curve_date or row_available != available_date:
            raise ValueError("base curve rows must share one observation and availability date")
        if row_source_key != source_key or row_sha256 != sha256:
            raise ValueError("base curve rows must share one source key and SHA-256")
        tenor = float(row["tenor_years"])
        rate = float(row.get("nominal_rate", row.get("nominal_rate_pct", row.get("rate_pct"))))
        pairs.append((tenor, rate))
    pairs.sort()
    tenors = tuple(pair[0] for pair in pairs)
    rates = tuple(pair[1] for pair in pairs)
    _validate_positive_unique_tenors(tenors, label="base_curve_rows")
    if 0.25 < tenors[0] or 10.0 > tenors[-1]:
        raise ValueError("base curve must cover the 0.25 and 10.0 year anchors")
    return {
        "base_curve_date": base_curve_date,
        "available_date": available_date,
        "base_curve_source_key": source_key,
        "base_curve_sha256": sha256,
        "tenors": tenors,
        "rates": rates,
    }


def _validate_positive_unique_tenors(tenors: Sequence[float], *, label: str) -> None:
    if any(value <= 0.0 for value in tenors):
        raise ValueError(f"{label} tenors must be positive")
    if len(set(tenors)) != len(tenors):
        raise ValueError(f"{label} tenors must be unique")


def _log_tenor_weight(tenor: float) -> float:
    return min(max(math.log(tenor / 0.25) / math.log(10.0 / 0.25), 0.0), 1.0)


def _to_date(value: Any) -> date:
    if value is None:
        raise ValueError("date value is required")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _pchip_value(x_values: Sequence[float], y_values: Sequence[float], x_new: float) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("PCHIP interpolation requires at least two x/y pairs")
    if x_new <= x_values[0]:
        return _linear_endpoint(x_values[0], y_values[0], x_values[1], y_values[1], x_new)
    if x_new >= x_values[-1]:
        return _linear_endpoint(x_values[-2], y_values[-2], x_values[-1], y_values[-1], x_new)

    slopes = _pchip_derivatives(x_values, y_values)
    idx = 0
    for candidate in range(len(x_values) - 1):
        if x_values[candidate] <= x_new <= x_values[candidate + 1]:
            idx = candidate
            break
    h = x_values[idx + 1] - x_values[idx]
    t = (x_new - x_values[idx]) / h
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t**2 * (3.0 - 2.0 * t)
    h11 = t**2 * (t - 1.0)
    return (
        h00 * y_values[idx]
        + h10 * h * slopes[idx]
        + h01 * y_values[idx + 1]
        + h11 * h * slopes[idx + 1]
    )


def _pchip_derivatives(x_values: Sequence[float], y_values: Sequence[float]) -> tuple[float, ...]:
    n = len(x_values)
    h = [x_values[idx + 1] - x_values[idx] for idx in range(n - 1)]
    if any(value <= 0.0 for value in h):
        raise ValueError("PCHIP x values must be strictly increasing")
    delta = [(y_values[idx + 1] - y_values[idx]) / h[idx] for idx in range(n - 1)]
    if n == 2:
        return (delta[0], delta[0])

    derivatives = [0.0] * n
    derivatives[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    derivatives[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    for idx in range(1, n - 1):
        if delta[idx - 1] == 0.0 or delta[idx] == 0.0 or delta[idx - 1] * delta[idx] < 0.0:
            derivatives[idx] = 0.0
        else:
            w1 = 2.0 * h[idx] + h[idx - 1]
            w2 = h[idx] + 2.0 * h[idx - 1]
            derivatives[idx] = (w1 + w2) / (w1 / delta[idx - 1] + w2 / delta[idx])
    return tuple(derivatives)


def _pchip_endpoint_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
    if slope * delta0 <= 0.0:
        return 0.0
    if delta0 * delta1 < 0.0 and abs(slope) > abs(3.0 * delta0):
        return 3.0 * delta0
    return slope


def _linear_endpoint(x0: float, y0: float, x1: float, y1: float, x_new: float) -> float:
    return y0 + (y1 - y0) * (x_new - x0) / (x1 - x0)


__all__ = [
    "CONSTRUCTION_METHOD",
    "build_yield_curve_surface_rows",
]
