"""Fiscal, operating-cash, and cash-residual scenario transforms."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any


Row = dict[str, Any]
SCENARIO_SOURCE_ROLE = "scenario_assumption"
NO_FUNDING_FLAGS = {
    "affects_primary_deficit": False,
    "affects_net_interest": False,
    "affects_total_deficit": False,
    "affects_debt_target": False,
    "affects_issuance_size": False,
    "affects_tdc_fiscal_flow": False,
}


class FiscalTransformError(ValueError):
    """Raised when a fiscal scenario transform is malformed."""


def apply_operating_cash_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    inflation_rows: Iterable[Mapping[str, Any]] | None = None,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[Row]:
    """Apply operating-cash overrides without creating issuance effects."""

    mode = _required_mode(override)
    if mode == "aggregate_path_file":
        return _replacement_rows(
            replacement_rows,
            required_columns=("period_end", "operating_cash_target_bil"),
            mode=mode,
            claim_boundary="operating_cash_proxy_not_debt_target_or_issuance_supply",
        )
    if mode == "component_path_file":
        out = _replacement_rows(
            replacement_rows,
            required_columns=("period_end", "operating_cash_target_bil", "tga_target_bil", "ttl_target_bil", "other_operating_cash_target_bil"),
            mode=mode,
            claim_boundary="operating_cash_proxy_not_debt_target_or_issuance_supply",
        )
        for row in out:
            _assert_cash_identity(row)
        return out
    baseline = [dict(row) for row in rows]
    if not baseline:
        return []
    out: list[Row] = []
    base_values = {col: float(baseline[0].get(col, 0.0) or 0.0) for col in _cash_columns(baseline[0])}
    inflation_index = _inflation_index_by_date(inflation_rows)
    first_period = str(baseline[0].get("period_end") or "")
    base_index = _nearest_inflation_index(
        inflation_index,
        first_period,
        float(baseline[0].get("inflation_index_level", 0.0) or 0.0),
    )
    for row in baseline:
        new = dict(row)
        if mode == "constant_nominal":
            for col, value in base_values.items():
                new[col] = value
            new["construction_mode"] = "scenario_constant_nominal"
        elif mode == "constant_real":
            if base_index <= 0:
                raise FiscalTransformError("constant_real requires positive inflation_index_level in first row")
            key = str(new.get("period_end") or "")
            row_index = _nearest_inflation_index(
                inflation_index,
                key,
                float(new.get("inflation_index_level", base_index) or base_index),
            )
            scalar = row_index / base_index
            for col, value in base_values.items():
                new[col] = value * scalar
            new["inflation_index_level"] = row_index
            new["construction_mode"] = "scenario_constant_real"
        elif mode == "scale_baseline":
            scale = _number(override, "scale")
            for col in _cash_columns(new):
                new[col] = float(new[col]) * scale
            new["construction_mode"] = "scenario_scale_baseline"
        else:
            raise FiscalTransformError(f"unsupported operating_cash mode: {mode}")
        _assert_cash_identity(new)
        out.append(_mark(new, mode, claim_boundary="operating_cash_proxy_not_debt_target_or_issuance_supply"))
    return out


def apply_cash_residual_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    operating_cash_rows: Iterable[Mapping[str, Any]] | None = None,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[Row]:
    """Apply cash-residual overrides while keeping the residual non-funding."""

    mode = _required_mode(override)
    if mode == "explicit_path_file":
        out = _replacement_rows(
            replacement_rows,
            required_columns=("period_end", "cash_reconciliation_residual_bil"),
            mode=mode,
            runtime_role="reconciliation_only",
            claim_boundary="cash_residual_does_not_change_debt_issuance_primary_or_tdc_flows",
        )
        for row in out:
            row.update(NO_FUNDING_FLAGS)
            row["affects_operating_cash"] = True
        return out
    cash_delta = _cash_delta_by_date(operating_cash_rows) if operating_cash_rows is not None else {}
    out = []
    for row in rows:
        new = dict(row)
        if mode == "zero":
            new["cash_reconciliation_residual_bil"] = 0.0
        elif mode == "track_operating_cash_target":
            key = str(new.get("period_end") or "")
            new["cash_reconciliation_residual_bil"] = cash_delta.get(key, 0.0)
        else:
            raise FiscalTransformError(f"unsupported cash_reconciliation mode: {mode}")
        new.update(NO_FUNDING_FLAGS)
        new["affects_operating_cash"] = True
        out.append(_mark(new, mode, runtime_role="reconciliation_only", claim_boundary="cash_residual_does_not_change_debt_issuance_primary_or_tdc_flows"))
    return out


def apply_primary_deficit_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[Row]:
    """Apply primary-deficit path transforms to period-flow rows."""

    return _apply_path_override(
        rows,
        override,
        value_columns=("primary_deficit_bil", "annual_or_remaining_primary_deficit_bil"),
        mode_label="primary_deficit",
        replacement_rows=replacement_rows,
        annual_flow_anchor=True,
    )


def apply_debt_target_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[Row]:
    """Apply debt-target transforms to stock-path rows."""

    return _apply_path_override(
        rows,
        override,
        value_columns=("cbo_federal_debt_held_public_target_bil", "marketable_treasury_public_target_bil"),
        mode_label="debt_target",
        replacement_rows=replacement_rows,
        annual_flow_anchor=False,
    )


def apply_fiscal_incidence_override(rows: Iterable[Mapping[str, Any]], override: Mapping[str, Any]) -> list[Row]:
    """Apply a static fiscal-incidence split after exact share validation."""

    mode = _required_mode(override)
    if mode != "static_shares":
        raise FiscalTransformError(f"unsupported fiscal_incidence mode: {mode}")
    shares = {
        "du_share": _number(override, "domestic_ultimate_share"),
        "ru_share": _number(override, "rest_of_world_share"),
        "foreign_share": _number(override, "foreign_official_share"),
        "other_share": _number(override, "other_share"),
    }
    _assert_unit_sum(shares.values(), label="fiscal incidence shares")
    out = []
    for row in rows:
        new = dict(row)
        new.update(shares)
        new["recipient_share_source"] = "explicit_scenario_assumption"
        out.append(_mark(new, mode, runtime_role="memo_only", claim_boundary="recipient_incidence_assumption_does_not_change_aggregate_signed_flow"))
    return out


def _apply_path_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    value_columns: Sequence[str],
    mode_label: str,
    replacement_rows: Iterable[Mapping[str, Any]] | None,
    annual_flow_anchor: bool,
) -> list[Row]:
    mode = _required_mode(override)
    if mode == "absolute_path_file":
        return _replacement_rows(
            replacement_rows,
            required_columns=value_columns[:1],
            mode=mode,
            claim_boundary=f"{mode_label}_scenario_transform_no_plug",
        )
    baseline = [dict(row) for row in rows]
    if mode == "scale_path":
        scale = _number(override, "scale")
        return [_mark(_scale_values(row, value_columns, scale), mode, claim_boundary=f"{mode_label}_scenario_transform_no_plug") for row in baseline]
    if mode == "additive_bil":
        additive = _number(override, "additive_bil")
        return [_mark(_add_values(row, value_columns, additive), mode, claim_boundary=f"{mode_label}_scenario_transform_no_plug") for row in baseline]
    if mode == "fy_endpoint_anchors":
        anchors = _anchors(override)
        first_anchor = min(anchors)
        if annual_flow_anchor:
            return _apply_annual_flow_anchors(
                baseline,
                anchors,
                first_anchor=first_anchor,
                freeze_pre_start_actuals=bool(override.get("freeze_pre_start_actuals", False)),
                value_columns=value_columns,
                mode=mode,
                mode_label=mode_label,
            )
        out = []
        for row in baseline:
            fiscal_year = _fiscal_year(row)
            if bool(override.get("freeze_pre_start_actuals", False)) and fiscal_year < first_anchor:
                out.append(_mark(row, mode, claim_boundary=f"{mode_label}_pre_start_actuals_frozen"))
                continue
            target = _interpolate_anchor(fiscal_year, anchors)
            new = dict(row)
            primary_col = value_columns[0]
            base_value = float(new.get(primary_col, 0.0) or 0.0)
            ratio = 0.0 if base_value == 0 else target / base_value
            new[primary_col] = target
            for col in value_columns[1:]:
                if col in new:
                    new[col] = float(new[col]) * ratio
            out.append(_mark(new, mode, claim_boundary=f"{mode_label}_scenario_transform_no_plug"))
        return out
    raise FiscalTransformError(f"unsupported {mode_label} mode: {mode}")


def _apply_annual_flow_anchors(
    rows: Sequence[Mapping[str, Any]],
    anchors: Mapping[int, float],
    *,
    first_anchor: int,
    freeze_pre_start_actuals: bool,
    value_columns: Sequence[str],
    mode: str,
    mode_label: str,
) -> list[Row]:
    primary_col = value_columns[0]
    totals: dict[int, float] = {}
    for row in rows:
        fiscal_year = _fiscal_year(row)
        totals[fiscal_year] = totals.get(fiscal_year, 0.0) + float(row.get(primary_col, 0.0) or 0.0)
    out: list[Row] = []
    for row in rows:
        fiscal_year = _fiscal_year(row)
        if freeze_pre_start_actuals and fiscal_year < first_anchor:
            out.append(_mark(dict(row), mode, claim_boundary=f"{mode_label}_pre_start_actuals_frozen"))
            continue
        target = _interpolate_anchor(fiscal_year, anchors)
        base_total = totals.get(fiscal_year, 0.0)
        new = dict(row)
        if abs(base_total) > 1e-12:
            ratio = target / base_total
            new[primary_col] = float(new.get(primary_col, 0.0) or 0.0) * ratio
        else:
            year_rows = [candidate for candidate in rows if _fiscal_year(candidate) == fiscal_year]
            new[primary_col] = target / max(len(year_rows), 1)
        for col in value_columns[1:]:
            if col in new:
                new[col] = target
        out.append(_mark(new, mode, claim_boundary=f"{mode_label}_annual_flow_anchor_preserves_fy_sum"))
    return out


def _required_mode(override: Mapping[str, Any]) -> str:
    mode = override.get("mode")
    if not isinstance(mode, str) or not mode:
        raise FiscalTransformError("override mode is required")
    return mode


def _number(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping:
        raise FiscalTransformError(f"{key} is required")
    return float(mapping[key])


def _cash_columns(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        col
        for col in (
            "operating_cash_target_bil",
            "tga_target_bil",
            "ttl_target_bil",
            "other_operating_cash_target_bil",
        )
        if col in row
    )


def _assert_cash_identity(row: Mapping[str, Any]) -> None:
    if {"operating_cash_target_bil", "tga_target_bil", "ttl_target_bil", "other_operating_cash_target_bil"} <= set(row):
        total = float(row["tga_target_bil"]) + float(row["ttl_target_bil"]) + float(row["other_operating_cash_target_bil"])
        if abs(float(row["operating_cash_target_bil"]) - total) > 1e-9:
            raise FiscalTransformError("operating cash must equal TGA plus TOC components")


def _cash_delta_by_date(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, float]:
    if rows is None:
        return {}
    out: dict[str, float] = {}
    previous: float | None = None
    for row in rows:
        key = str(row.get("period_end") or "")
        value = float(row.get("operating_cash_target_bil", 0.0) or 0.0)
        out[key] = 0.0 if previous is None else value - previous
        previous = value
    return out


def _inflation_index_by_date(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, float]:
    if rows is None:
        return {}
    points: list[tuple[date, float]] = []
    for row in rows:
        key = row.get("month") or row.get("period_end")
        value = row.get("tips_cpi_u_index", row.get("cbo_cpi_u_index", row.get("inflation_index_level")))
        if key in ("", None) or value in ("", None):
            continue
        points.append((date.fromisoformat(str(key)[:10]), float(value)))
    if not points:
        return {}
    points.sort(key=lambda item: item[0])
    return {point[0].isoformat(): point[1] for point in points}


def _nearest_inflation_index(index: Mapping[str, float], key: str, default: float) -> float:
    if not index:
        return default
    if key in index:
        return index[key]
    target = date.fromisoformat(key[:10])
    nearest = min(index, key=lambda candidate: abs((date.fromisoformat(candidate) - target).days))
    return index[nearest]


def _scale_values(row: Mapping[str, Any], columns: Sequence[str], scale: float) -> Row:
    new = dict(row)
    for col in columns:
        if col in new:
            new[col] = float(new[col]) * scale
    return new


def _add_values(row: Mapping[str, Any], columns: Sequence[str], additive: float) -> Row:
    new = dict(row)
    for col in columns:
        if col in new:
            new[col] = float(new[col]) + additive
    return new


def _replacement_rows(
    replacement_rows: Iterable[Mapping[str, Any]] | None,
    *,
    required_columns: Sequence[str],
    mode: str,
    runtime_role: str | None = None,
    claim_boundary: str | None = None,
) -> list[Row]:
    if replacement_rows is None:
        raise FiscalTransformError(f"{mode} requires replacement_rows supplied by the compiler")
    out = []
    for row in replacement_rows:
        missing = [col for col in required_columns if col not in row]
        if missing:
            raise FiscalTransformError(f"{mode} replacement row is missing columns: {missing}")
        out.append(_mark(dict(row), mode, runtime_role=runtime_role, claim_boundary=claim_boundary))
    return out


def _anchors(override: Mapping[str, Any]) -> dict[int, float]:
    raw = override.get("anchors")
    if not isinstance(raw, list) or not raw:
        raise FiscalTransformError("fy_endpoint_anchors requires anchors")
    anchors = {int(item["fiscal_year"]): float(item["value_bil"]) for item in raw}
    if len(anchors) != len(raw):
        raise FiscalTransformError("duplicate fiscal-year anchors are not allowed")
    return dict(sorted(anchors.items()))


def _fiscal_year(row: Mapping[str, Any]) -> int:
    for key in ("source_fiscal_year", "fiscal_year"):
        if key in row and row[key] not in ("", None):
            return int(row[key])
    raise FiscalTransformError("row is missing source_fiscal_year")


def _interpolate_anchor(fiscal_year: int, anchors: Mapping[int, float]) -> float:
    years = sorted(anchors)
    if fiscal_year <= years[0]:
        return anchors[years[0]]
    if fiscal_year >= years[-1]:
        return anchors[years[-1]]
    for left, right in zip(years, years[1:]):
        if left <= fiscal_year <= right:
            weight = (fiscal_year - left) / (right - left)
            return anchors[left] + (anchors[right] - anchors[left]) * weight
    raise FiscalTransformError("failed to interpolate fiscal-year anchor")


def _assert_unit_sum(values: Iterable[float], *, label: str) -> None:
    total = sum(float(value) for value in values)
    if abs(total - 1.0) > 1e-9:
        raise FiscalTransformError(f"{label} must sum to 1.0, got {total}")


def _mark(
    row: Row,
    mode: str,
    *,
    runtime_role: str | None = None,
    claim_boundary: str | None = None,
) -> Row:
    row["source_role"] = SCENARIO_SOURCE_ROLE
    row["scenario_transform"] = mode
    row["runtime_role"] = runtime_role or "hard_target"
    if claim_boundary is not None:
        row["claim_boundary"] = claim_boundary
    return row


__all__ = [
    "FiscalTransformError",
    "apply_cash_residual_override",
    "apply_debt_target_override",
    "apply_fiscal_incidence_override",
    "apply_operating_cash_override",
    "apply_primary_deficit_override",
]
