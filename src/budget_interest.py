"""Pure budget-interest diagnostic helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

DEFAULT_DAY_COUNT_BASIS = 365.0
NET_INTEREST_WARNING_FLOOR_BIL = 10.0
NET_INTEREST_WARNING_SHARE = 0.01
NET_INTEREST_RED_FLOOR_BIL = 25.0
NET_INTEREST_RED_SHARE = 0.025
AVERAGE_RATE_WARNING_THRESHOLD_PCT = 0.25
AVERAGE_RATE_RED_THRESHOLD_PCT = 0.50

REQUIRED_COMPLETE_SCOPE_COMPONENT_KEYS = frozenset(
    {
        "fixed_coupon_accrual",
        "bill_discount_amortization",
        "frn_accrual",
        "signed_tips_principal_indexation",
    }
)
SIGNED_COMPONENT_KEYS = frozenset({"signed_tips_principal_indexation", "offsetting_interest_receipts"})


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def day_count_fraction(start_date: Any, end_date: Any, *, basis: float = DEFAULT_DAY_COUNT_BASIS) -> float:
    """Actual-day fraction for deterministic accrual helpers."""
    days = (_to_date(end_date) - _to_date(start_date)).days
    if days < 0:
        raise ValueError("end_date must be on or after start_date")
    return days / float(basis)


def fixed_coupon_accrual(face_value: float, coupon_rate: float, day_fraction: float) -> float:
    """Accrue fixed coupon interest using annual decimal coupon rates."""
    return _as_float(face_value) * _as_float(coupon_rate) * _as_float(day_fraction)


def bill_discount_amortization(
    face_value: float,
    issue_proceeds: float,
    elapsed_days: float,
    term_days: float,
) -> float:
    """Recognize bill discount over the bill term without changing principal."""
    term = _as_float(term_days)
    if term <= 0.0:
        raise ValueError("term_days must be positive")
    fraction = min(max(_as_float(elapsed_days) / term, 0.0), 1.0)
    return (_as_float(face_value) - _as_float(issue_proceeds)) * fraction


def frn_accrual(
    face_value: float,
    benchmark_rate: float,
    fixed_spread: float,
    day_fraction: float,
) -> float:
    """Accrue FRN interest from benchmark plus fixed spread."""
    benchmark = _require_decimal_rate(benchmark_rate, "benchmark_rate")
    spread = _require_decimal_rate(fixed_spread, "fixed_spread")
    return _as_float(face_value) * (benchmark + spread) * _as_float(day_fraction)


def tips_principal_indexation(face_value: float, beginning_index_ratio: float, ending_index_ratio: float) -> float:
    """Signed TIPS principal indexation; deflation is intentionally not clipped."""
    return _as_float(face_value) * (_as_float(ending_index_ratio) - _as_float(beginning_index_ratio))


def component_total(components: Iterable[Mapping[str, Any]]) -> float:
    """Sum explicit budget-interest components by amount, with no residual plug."""
    return sum(_as_float(component.get("amount_bil", component.get("amount"))) for component in components)


def simulated_average_interest_rate_pct(
    *,
    modeled_net_interest_bil: float,
    average_debt_held_public_bil: float,
) -> float:
    """Compute modeled fiscal-year average interest rate as percentage points."""
    average_debt = float(average_debt_held_public_bil)
    if average_debt <= 0.0:
        raise ValueError("average_debt_held_public_bil must be positive")
    return float(modeled_net_interest_bil) / average_debt * 100.0


def build_net_interest_diagnostic(
    *,
    cbo_net_interest_bil: float,
    modeled_net_interest_bil: float | None = None,
    components: Iterable[Mapping[str, Any]] | None = None,
    fiscal_year: int | None = None,
    scope_status: str = "partial",
    calibration_mode: str = "diagnostic_only",
) -> dict[str, Any]:
    """Compare modeled net interest to CBO as a nonbinding diagnostic."""
    component_rows = list(components or [])
    if scope_status == "complete":
        validate_complete_scope_components(component_rows)
    if modeled_net_interest_bil is None:
        modeled_net_interest_bil = component_total(component_rows)
    cbo_value = float(cbo_net_interest_bil)
    modeled_value = float(modeled_net_interest_bil)
    residual = cbo_value - modeled_value
    residual_pct = 0.0 if cbo_value == 0.0 else abs(residual) / abs(cbo_value) * 100.0
    warning_threshold = net_interest_warning_threshold_bil(cbo_value)
    red_threshold = net_interest_red_threshold_bil(cbo_value)
    abs_residual = abs(residual)
    if abs_residual > red_threshold:
        threshold_status = "red"
    elif abs_residual > warning_threshold:
        threshold_status = "warning"
    else:
        threshold_status = "ok"
    if scope_status == "complete" and threshold_status == "red":
        claim_status = "hard_release_failure_complete_net_interest_reconciliation_claim"
    elif scope_status == "complete" and threshold_status == "warning":
        claim_status = "blocks_complete_net_interest_reproduction_claim"
    elif threshold_status == "red":
        claim_status = "red_diagnostic_only_nonbinding"
    elif threshold_status == "warning":
        claim_status = "warning_diagnostic_only_nonbinding"
    else:
        claim_status = "diagnostic_only_nonbinding"
    largest_residual_component = _largest_residual_component(component_rows)
    return {
        "fiscal_year": fiscal_year,
        "cbo_net_interest_bil": cbo_value,
        "modeled_net_interest_bil": modeled_value,
        "residual_bil": residual,
        "residual_pct": residual_pct,
        "scope_status": scope_status,
        "calibration_mode": calibration_mode,
        "threshold_status": threshold_status,
        "claim_status": claim_status,
        "largest_residual_component": largest_residual_component,
        "issue_id": None if threshold_status == "ok" else f"net_interest_residual_{threshold_status}",
    }


def build_average_interest_rate_diagnostic(
    *,
    fiscal_year: int | None = None,
    cbo_average_interest_rate_pct: float,
    simulated_average_rate_pct: float | None = None,
    modeled_net_interest_bil: float | None = None,
    average_debt_held_public_bil: float | None = None,
    warning_threshold_pct: float = AVERAGE_RATE_WARNING_THRESHOLD_PCT,
    red_threshold_pct: float = AVERAGE_RATE_RED_THRESHOLD_PCT,
) -> dict[str, Any]:
    """Compare modeled portfolio-cost rate to CBO's reporting-only average rate."""
    if simulated_average_rate_pct is None:
        if modeled_net_interest_bil is None or average_debt_held_public_bil is None:
            raise ValueError(
                "simulated_average_rate_pct or both modeled_net_interest_bil and "
                "average_debt_held_public_bil are required"
            )
        simulated_average_rate_pct = simulated_average_interest_rate_pct(
            modeled_net_interest_bil=modeled_net_interest_bil,
            average_debt_held_public_bil=average_debt_held_public_bil,
        )
    cbo_rate = float(cbo_average_interest_rate_pct)
    simulated_rate = float(simulated_average_rate_pct)
    residual = simulated_rate - cbo_rate
    abs_residual = abs(residual)
    if abs_residual > red_threshold_pct:
        threshold_status = "red"
    elif abs_residual > warning_threshold_pct:
        threshold_status = "warning"
    else:
        threshold_status = "ok"
    return {
        "fiscal_year": fiscal_year,
        "cbo_average_interest_rate_pct": cbo_rate,
        "simulated_average_interest_rate_pct": simulated_rate,
        "average_interest_rate_residual_pct": residual,
        "threshold_status": threshold_status,
        "runtime_role": "diagnostic_only",
        "claim_boundary": "cbo_average_interest_rate_reporting_only_not_yield_curve_anchor",
    }


def net_interest_warning_threshold_bil(cbo_net_interest_bil: float) -> float:
    return max(NET_INTEREST_WARNING_FLOOR_BIL, abs(float(cbo_net_interest_bil)) * NET_INTEREST_WARNING_SHARE)


def net_interest_red_threshold_bil(cbo_net_interest_bil: float) -> float:
    return max(NET_INTEREST_RED_FLOOR_BIL, abs(float(cbo_net_interest_bil)) * NET_INTEREST_RED_SHARE)


def validate_complete_scope_components(components: Iterable[Mapping[str, Any]]) -> None:
    """Reject false complete-scope budget-interest claims."""
    component_rows = list(components)
    included_keys = [
        str(component.get("component_key"))
        for component in component_rows
        if bool(component.get("include_in_budget_interest", True))
    ]
    missing = sorted(REQUIRED_COMPLETE_SCOPE_COMPONENT_KEYS.difference(included_keys))
    if missing:
        raise ValueError(f"complete budget-interest scope missing components: {', '.join(missing)}")
    duplicates = sorted({key for key in included_keys if included_keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"complete budget-interest scope double counts components: {', '.join(duplicates)}")
    for component in component_rows:
        if not bool(component.get("include_in_budget_interest", True)):
            continue
        component_key = str(component.get("component_key"))
        amount = float(component.get("amount_bil", component.get("amount", 0.0)))
        if component_key not in SIGNED_COMPONENT_KEYS and amount < 0.0:
            raise ValueError(f"component {component_key} has invalid negative expense amount")


def _require_decimal_rate(value: Any, field_name: str) -> float:
    rate = _as_float(value)
    if abs(rate) > 1.0:
        raise ValueError(f"{field_name} must be a decimal annual rate, not percentage points")
    return rate


def _largest_residual_component(components: Iterable[Mapping[str, Any]]) -> str | None:
    largest_key: str | None = None
    largest_abs_residual = 0.0
    for component in components:
        if "residual_bil" not in component:
            continue
        abs_residual = abs(float(component["residual_bil"]))
        if abs_residual > largest_abs_residual:
            largest_abs_residual = abs_residual
            largest_key = str(component.get("component_key"))
    return largest_key


__all__ = [
    "day_count_fraction",
    "fixed_coupon_accrual",
    "bill_discount_amortization",
    "frn_accrual",
    "tips_principal_indexation",
    "component_total",
    "simulated_average_interest_rate_pct",
    "build_net_interest_diagnostic",
    "build_average_interest_rate_diagnostic",
    "net_interest_warning_threshold_bil",
    "net_interest_red_threshold_bil",
    "validate_complete_scope_components",
]
