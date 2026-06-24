"""Rate, FRN, CPI, and TIPS scenario transforms.

These functions operate on lists of CSV-style row dictionaries. They do not read
or write files; the compiler is responsible for file resolution and hashing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from math import log
from typing import Any


Row = dict[str, Any]

PERCENT_RATE_COLUMNS = ("nominal_rate", "anchor_3m_pct", "anchor_10y_pct")
CPI_COLUMNS = ("cbo_cpi_u_index", "tips_cpi_u_index")
SCENARIO_SOURCE_ROLE = "scenario_assumption"


class TransformError(ValueError):
    """Raised when a scenario transform is malformed or unsupported."""


def apply_nominal_yield_curve_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[Row]:
    """Apply a nominal-yield-curve override to baseline surface rows."""

    mode = _required_mode(override)
    if mode == "parallel_bp":
        delta = _bp_to_decimal(_number(override, "shock_bp"))
        return [_mark(_shift_rate_row(row, delta, decimal_cols=("nominal_rate_decimal",), percent_cols=PERCENT_RATE_COLUMNS), mode) for row in rows]
    if mode == "key_rate_bp":
        shocks = _shock_points(override)
        interpolation = str(override.get("interpolation") or "log_tenor_linear")
        return [
            _mark(
                _shift_rate_row(
                    row,
                    _interpolated_bp(float(row["tenor_years"]), shocks, interpolation) / 10000.0,
                    decimal_cols=("nominal_rate_decimal",),
                    percent_cols=("nominal_rate",),
                ),
                mode,
            )
            for row in rows
        ]
    if mode == "full_surface_file":
        return _replacement_rows(replacement_rows, required_columns=("curve_date", "tenor_years", "nominal_rate_decimal"), mode=mode)
    if mode == "anchor_path_file":
        raise TransformError("anchor_path_file requires compiler-level yield-curve reconstruction")
    raise TransformError(f"unsupported nominal_yield_curve mode: {mode}")


def apply_frn_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    nominal_curve_rows: Iterable[Mapping[str, Any]] | None = None,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[Row]:
    """Apply an FRN benchmark override while preserving FRN contract fields."""

    mode = _required_mode(override)
    if mode == "parallel_bp":
        delta = _bp_to_decimal(_number(override, "shock_bp"))
        return [_mark(_shift_rate_row(row, delta, decimal_cols=("benchmark_rate_decimal", "auction_high_rate_decimal", "money_market_yield_decimal")), mode) for row in rows]
    if mode == "linked_to_nominal_curve":
        if nominal_curve_rows is None:
            raise TransformError("linked_to_nominal_curve requires nominal_curve_rows")
        spread = _bp_to_decimal(float(override.get("spread_bp", 0.0)))
        tenor_map = _nominal_curve_map(nominal_curve_rows, target_tenor=0.25)
        out: list[Row] = []
        for row in rows:
            new = dict(row)
            key = _row_date_key(row, "period_start", "rate_effective_start", "period_end")
            benchmark = tenor_map.get(key)
            if benchmark is None:
                benchmark = _nearest_date_rate(tenor_map, key)
            new_rate = benchmark + spread
            for col in ("benchmark_rate_decimal", "auction_high_rate_decimal", "money_market_yield_decimal"):
                if col in new:
                    new[col] = new_rate
            new["rate_source_family"] = "scenario_nominal_curve_3m_linked"
            out.append(_mark(new, mode))
        return out
    if mode == "absolute_path_file":
        return _replacement_rows(replacement_rows, required_columns=("period_start", "period_end", "benchmark_rate_decimal"), mode=mode)
    raise TransformError(f"unsupported frn_benchmark mode: {mode}")


def apply_cpi_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[Row]:
    """Apply CPI path overrides without mutating opening reference-CPI fields."""

    mode = _required_mode(override)
    baseline = [dict(row) for row in rows]
    if mode == "monthly_path_file":
        return _replacement_rows(replacement_rows, required_columns=("month", "tips_cpi_u_index"), mode=mode)
    if not baseline:
        return []
    if mode == "cpi_level_scale":
        scale = _number(override, "scale")
        return [_mark(_scale_cpi_row(row, scale), mode) for row in baseline]
    if mode == "annualized_inflation_shift_bp":
        shock = _bp_to_decimal(_number(override, "shock_bp"))
        base_month = _parse_date(str(baseline[0]["month"]))
        out = []
        for row in baseline:
            current_month = _parse_date(str(row["month"]))
            months = (current_month.year - base_month.year) * 12 + (current_month.month - base_month.month)
            scalar = (1.0 + shock) ** (months / 12.0)
            new = _scale_cpi_row(row, scalar)
            if "terminal_annualized_cpi_growth_decimal" in new:
                new["terminal_annualized_cpi_growth_decimal"] = float(new["terminal_annualized_cpi_growth_decimal"]) + shock
            if "terminal_rule" in override:
                new["scenario_terminal_rule"] = str(override["terminal_rule"])
            out.append(_mark(new, mode))
        return out
    raise TransformError(f"unsupported inflation_cpi mode: {mode}")


def apply_tips_real_yield_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    nominal_curve_rows: Iterable[Mapping[str, Any]] | None = None,
    cpi_rows: Iterable[Mapping[str, Any]] | None = None,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[Row]:
    """Apply TIPS real-yield overrides to derived real-yield assumptions."""

    mode = _required_mode(override)
    if mode == "absolute_path_file":
        return _replacement_rows(replacement_rows, required_columns=("curve_date", "tenor_years", "real_yield_decimal"), mode=mode)
    if mode == "parallel_bp":
        delta = _bp_to_decimal(_number(override, "shock_bp"))
        return [_mark(_shift_rate_row(row, delta, decimal_cols=("real_yield_decimal", "real_coupon_decimal")), mode) for row in rows]
    if mode == "key_rate_bp":
        shocks = _shock_points(override)
        return [
            _mark(
                _shift_rate_row(
                    row,
                    _interpolated_bp(float(row["tenor_years"]), shocks, str(override.get("interpolation") or "log_tenor_linear")) / 10000.0,
                    decimal_cols=("real_yield_decimal", "real_coupon_decimal"),
                ),
                mode,
            )
            for row in rows
        ]
    if mode == "linked_recompute":
        nominal_map = _curve_rate_map(nominal_curve_rows) if nominal_curve_rows is not None else {}
        expected_inflation_map = _expected_inflation_by_curve_date(cpi_rows)
        add = _bp_to_decimal(float(override.get("additional_parallel_bp", 0.0)))
        out: list[Row] = []
        for row in rows:
            new = dict(row)
            key = (str(new["curve_date"]), float(new["tenor_years"]))
            nominal = nominal_map.get(key, float(new.get("nominal_rate_decimal", 0.0)))
            curve_date = str(new["curve_date"])
            expected_inflation = _nearest_date_rate(
                expected_inflation_map,
                curve_date,
            ) if expected_inflation_map else float(new.get("expected_inflation_decimal", 0.0))
            real_yield = nominal - expected_inflation + add
            new["nominal_rate_decimal"] = nominal
            new["expected_inflation_decimal"] = expected_inflation
            new["real_yield_decimal"] = real_yield
            if "real_coupon_decimal" in new:
                new["real_coupon_decimal"] = real_yield
            out.append(_mark(new, mode))
        return out
    raise TransformError(f"unsupported tips_real_yield mode: {mode}")


def _required_mode(override: Mapping[str, Any]) -> str:
    mode = override.get("mode")
    if not isinstance(mode, str) or not mode:
        raise TransformError("override mode is required")
    return mode


def _number(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping:
        raise TransformError(f"{key} is required")
    return float(mapping[key])


def _bp_to_decimal(bp: float) -> float:
    return bp / 10000.0


def _shift_rate_row(
    row: Mapping[str, Any],
    decimal_delta: float,
    *,
    decimal_cols: Sequence[str],
    percent_cols: Sequence[str] = (),
) -> Row:
    new = dict(row)
    for col in decimal_cols:
        if col in new and new[col] not in ("", None):
            new[col] = float(new[col]) + decimal_delta
    for col in percent_cols:
        if col in new and new[col] not in ("", None):
            new[col] = float(new[col]) + decimal_delta * 100.0
    return new


def _scale_cpi_row(row: Mapping[str, Any], scale: float) -> Row:
    new = dict(row)
    for col in CPI_COLUMNS:
        if col in new and new[col] not in ("", None):
            new[col] = float(new[col]) * scale
    return new


def _mark(row: Row, mode: str) -> Row:
    row["source_role"] = SCENARIO_SOURCE_ROLE
    row["runtime_role"] = "hard_target"
    row["claim_boundary"] = "scenario_assumption_not_official_cbo_source"
    row["scenario_transform"] = mode
    return row


def _replacement_rows(
    replacement_rows: Iterable[Mapping[str, Any]] | None,
    *,
    required_columns: Sequence[str],
    mode: str,
) -> list[Row]:
    if replacement_rows is None:
        raise TransformError(f"{mode} requires replacement_rows supplied by the compiler")
    out = []
    for row in replacement_rows:
        missing = [col for col in required_columns if col not in row]
        if missing:
            raise TransformError(f"{mode} replacement row is missing columns: {missing}")
        out.append(_mark(dict(row), mode))
    return out


def _shock_points(override: Mapping[str, Any]) -> list[tuple[float, float]]:
    shocks = override.get("shocks")
    if not isinstance(shocks, list) or not shocks:
        raise TransformError("key_rate_bp requires nonempty shocks")
    points = sorted((float(item["tenor_years"]), float(item["shock_bp"])) for item in shocks)
    if points[0][0] <= 0:
        raise TransformError("key-rate tenor_years must be positive")
    return points


def _interpolated_bp(tenor_years: float, points: Sequence[tuple[float, float]], interpolation: str) -> float:
    if tenor_years <= 0:
        raise TransformError("tenor_years must be positive")
    if tenor_years <= points[0][0]:
        return points[0][1]
    if tenor_years >= points[-1][0]:
        return points[-1][1]
    if interpolation not in {"log_tenor_linear", "pchip_log_tenor"}:
        raise TransformError(f"unsupported key-rate interpolation: {interpolation}")
    if interpolation == "pchip_log_tenor":
        try:
            from scipy.interpolate import PchipInterpolator  # type: ignore

            x = [log(point[0]) for point in points]
            y = [point[1] for point in points]
            return float(PchipInterpolator(x, y)(log(tenor_years)))
        except Exception:
            pass
    x = log(tenor_years)
    for (left_tenor, left_bp), (right_tenor, right_bp) in zip(points, points[1:]):
        if left_tenor <= tenor_years <= right_tenor:
            left_x = log(left_tenor)
            right_x = log(right_tenor)
            weight = (x - left_x) / (right_x - left_x)
            return left_bp + (right_bp - left_bp) * weight
    raise TransformError("failed to interpolate key-rate shock")


def _nominal_curve_map(rows: Iterable[Mapping[str, Any]], *, target_tenor: float) -> dict[str, float]:
    by_date: dict[str, tuple[float, float]] = {}
    for row in rows:
        tenor = float(row["tenor_years"])
        curve_date = str(row["curve_date"])
        rate = float(row["nominal_rate_decimal"])
        distance = abs(tenor - target_tenor)
        current = by_date.get(curve_date)
        if current is None or distance < current[0]:
            by_date[curve_date] = (distance, rate)
    if not by_date:
        raise TransformError("nominal_curve_rows contains no usable rates")
    return {key: value[1] for key, value in by_date.items()}


def _curve_rate_map(rows: Iterable[Mapping[str, Any]] | None) -> dict[tuple[str, float], float]:
    if rows is None:
        return {}
    return {(str(row["curve_date"]), float(row["tenor_years"])): float(row["nominal_rate_decimal"]) for row in rows}


def _expected_inflation_by_curve_date(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, float]:
    if rows is None:
        return {}
    points: list[tuple[date, float, float | None]] = []
    for row in rows:
        month = row.get("month")
        if month in ("", None):
            continue
        value = row.get("tips_cpi_u_index", row.get("cbo_cpi_u_index"))
        if value in ("", None):
            continue
        terminal = row.get("terminal_annualized_cpi_growth_decimal")
        points.append(
            (
                _parse_date(str(month)),
                float(value),
                None if terminal in ("", None) else float(terminal),
            )
        )
    if not points:
        return {}
    points.sort(key=lambda item: item[0])
    out: dict[str, float] = {}
    for point_date, _, _ in points:
        out[point_date.isoformat()] = _annualized_forward_cpi_growth(points, point_date)
    return out


def _annualized_forward_cpi_growth(points: Sequence[tuple[date, float, float | None]], start: date) -> float:
    start_date, start_value, start_terminal = _nearest_point(points, start)
    target = start + timedelta(days=365)
    future_candidates = [point for point in points if point[0] >= target]
    if future_candidates and start_value > 0:
        end_date, end_value, _ = future_candidates[0]
        years = max((end_date - start_date).days / 365.25, 1e-9)
        return (end_value / start_value) ** (1.0 / years) - 1.0
    terminal_candidates = [terminal for _, _, terminal in points if terminal is not None]
    if terminal_candidates:
        return terminal_candidates[-1]
    first_date, first_value, _ = points[0]
    last_date, last_value, _ = points[-1]
    if first_value <= 0 or last_date <= first_date:
        return 0.0
    years = max((last_date - first_date).days / 365.25, 1e-9)
    return (last_value / first_value) ** (1.0 / years) - 1.0


def _nearest_point(points: Sequence[tuple[date, float, float | None]], target: date) -> tuple[date, float, float | None]:
    before = [point for point in points if point[0] <= target]
    if before:
        return before[-1]
    return points[0]


def _row_date_key(row: Mapping[str, Any], *columns: str) -> str:
    for col in columns:
        if col in row and row[col] not in ("", None):
            return str(row[col])
    raise TransformError(f"row is missing date columns: {columns}")


def _nearest_date_rate(rates: Mapping[str, float], key: str) -> float:
    target = _parse_date(key)
    nearest = min(rates, key=lambda candidate: abs((_parse_date(candidate) - target).days))
    return rates[nearest]


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


__all__ = [
    "TransformError",
    "apply_cpi_override",
    "apply_frn_override",
    "apply_nominal_yield_curve_override",
    "apply_tips_real_yield_override",
]
