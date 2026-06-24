"""Issuance, Fed-stock, and holder-preference scenario transforms."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


MARKETABLE_SECURITY_TYPES = ("bills", "notes", "bonds", "tips", "frn")
HOLDER_TYPES = ("Banks", "CB", "Foreign", "Private", "TrustFunds", "FedInternal")
SCENARIO_SOURCE_ROLE = "scenario_assumption"


class PortfolioTransformError(ValueError):
    """Raised when a portfolio scenario transform is malformed."""


@dataclass(frozen=True)
class IssuanceMix:
    """Validated marketable issuance mix with maturity distribution summary."""

    security_shares: Mapping[str, float]
    maturity_distributions: Mapping[str, tuple[Mapping[str, float], ...]]
    weighted_average_maturity_years: float
    negative_issuance_action: str


def compile_issuance_mix_override(override: Mapping[str, Any]) -> IssuanceMix:
    """Validate and summarize a replace-shares issuance-mix override."""

    if override.get("mode") != "replace_shares":
        raise PortfolioTransformError("issuance_mix mode must be replace_shares")
    tips_share = float(override["tips_share"])
    frn_share = float(override["frn_share"])
    if tips_share + frn_share > 1.0 + 1e-12:
        raise PortfolioTransformError("tips_share plus frn_share must not exceed 1.0")
    fixed = override.get("fixed_remainder_shares")
    if not isinstance(fixed, Mapping):
        raise PortfolioTransformError("fixed_remainder_shares must be a mapping")
    _assert_unit_sum((float(fixed["bills"]), float(fixed["notes"]), float(fixed["bonds"])), label="fixed_remainder_shares")
    remainder = 1.0 - tips_share - frn_share
    security_shares = {
        "bills": remainder * float(fixed["bills"]),
        "notes": remainder * float(fixed["notes"]),
        "bonds": remainder * float(fixed["bonds"]),
        "tips": tips_share,
        "frn": frn_share,
    }
    _assert_unit_sum(security_shares.values(), label="issuance security shares")
    distributions = _maturity_distributions(override.get("maturity_distributions"))
    wam = 0.0
    for security_type, share in security_shares.items():
        wam += share * sum(float(item["maturity_years"]) * float(item["share"]) for item in distributions[security_type])
    action = str(override["negative_issuance_action"])
    if action not in {"error", "retire_shortest_public_marketable"}:
        raise PortfolioTransformError(f"unsupported negative_issuance_action: {action}")
    return IssuanceMix(
        security_shares=security_shares,
        maturity_distributions=distributions,
        weighted_average_maturity_years=wam,
        negative_issuance_action=action,
    )


def apply_fed_holdings_override(
    rows: Iterable[Mapping[str, Any]],
    override: Mapping[str, Any],
    *,
    marketable_debt_rows: Iterable[Mapping[str, Any]] | None = None,
    replacement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Apply Fed-stock target overrides and enforce the marketable-debt bound."""

    mode = str(override.get("mode") or "")
    if mode == "absolute_path_file":
        out = _replacement_rows(
            replacement_rows,
            required_columns=("period_end", "holder_type", "cbo_fed_holdings_target_bil"),
            mode=mode,
        )
        _assert_fed_bounds(out, _marketable_debt_by_date(marketable_debt_rows))
        return out
    debt_bound = _marketable_debt_by_date(marketable_debt_rows)
    out = []
    for row in rows:
        new = dict(row)
        current = float(new["cbo_fed_holdings_target_bil"])
        if mode == "scale_path":
            new["cbo_fed_holdings_target_bil"] = current * float(override["scale"])
        elif mode == "additive_bil":
            new["cbo_fed_holdings_target_bil"] = current + float(override["additive_bil"])
        elif mode == "fy_endpoint_anchors":
            new["cbo_fed_holdings_target_bil"] = _interpolate_anchor(_fiscal_year(new), _anchors(override))
        else:
            raise PortfolioTransformError(f"unsupported fed_holdings mode: {mode}")
        value = float(new["cbo_fed_holdings_target_bil"])
        if value < -1e-12:
            raise PortfolioTransformError("Fed holdings target must be nonnegative")
        bound = debt_bound.get(str(new.get("period_end") or ""))
        if bound is not None and value > bound + 1e-9:
            raise PortfolioTransformError("Fed holdings target must not exceed marketable debt target")
        new["source_role"] = SCENARIO_SOURCE_ROLE
        new["scenario_transform"] = mode
        new["runtime_role"] = "hard_target"
        new["claim_boundary"] = "fed_holdings_path_guides_holder_allocation_not_total_issuance"
        out.append(new)
    return out


def _replacement_rows(
    replacement_rows: Iterable[Mapping[str, Any]] | None,
    *,
    required_columns: tuple[str, ...],
    mode: str,
) -> list[dict[str, Any]]:
    if replacement_rows is None:
        raise PortfolioTransformError(f"{mode} requires replacement_rows supplied by the compiler")
    out = []
    for row in replacement_rows:
        missing = [col for col in required_columns if col not in row]
        if missing:
            raise PortfolioTransformError(f"{mode} replacement row is missing columns: {missing}")
        new = dict(row)
        new["source_role"] = SCENARIO_SOURCE_ROLE
        new["scenario_transform"] = mode
        new["runtime_role"] = "hard_target"
        new["claim_boundary"] = "fed_holdings_path_guides_holder_allocation_not_total_issuance"
        out.append(new)
    return out


def _assert_fed_bounds(rows: Iterable[Mapping[str, Any]], debt_bound: Mapping[str, float]) -> None:
    for row in rows:
        value = float(row["cbo_fed_holdings_target_bil"])
        if value < -1e-12:
            raise PortfolioTransformError("Fed holdings target must be nonnegative")
        bound = debt_bound.get(str(row.get("period_end") or ""))
        if bound is not None and value > bound + 1e-9:
            raise PortfolioTransformError("Fed holdings target must not exceed marketable debt target")


def validate_holder_preferences(
    override: Mapping[str, Any],
    *,
    fed_stock_target_active: bool = False,
) -> list[dict[str, Any]]:
    """Validate static holder shares and return normalized row dictionaries."""

    mode = str(override.get("mode") or "")
    if mode == "quarterly_path_file":
        raise PortfolioTransformError("quarterly_path_file requires compiler-resolved replacement rows")
    if mode != "static_shares":
        raise PortfolioTransformError(f"unsupported holder_preferences mode: {mode}")
    rows = override.get("rows")
    if not isinstance(rows, list) or not rows:
        raise PortfolioTransformError("static_shares requires rows")
    out = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        security_type = str(row["security_type"])
        effective_quarter = str(row.get("effective_quarter") or "")
        key = (effective_quarter, security_type)
        if key in seen:
            raise PortfolioTransformError(f"duplicate holder preference row: {key}")
        seen.add(key)
        shares = row.get("shares")
        if not isinstance(shares, Mapping):
            raise PortfolioTransformError("holder preference shares must be a mapping")
        normalized = {holder: float(shares[holder]) for holder in HOLDER_TYPES}
        total = sum(normalized.values())
        if security_type == "nonmarketable":
            if abs(total) > 1e-12:
                raise PortfolioTransformError("nonmarketable auction holder shares must be zero")
        else:
            _assert_unit_sum(normalized.values(), label=f"{security_type} holder shares")
        if fed_stock_target_active and security_type != "nonmarketable" and normalized["CB"] != 0.0:
            raise PortfolioTransformError("CB auction share must be zero when Fed stock target is active")
        out.append(
            {
                "effective_quarter": effective_quarter or None,
                "security_type": security_type,
                "shares": normalized,
                "source_role": SCENARIO_SOURCE_ROLE,
                "runtime_role": "memo_only",
                "claim_boundary": "holder preference profile not exact holder ownership",
            }
        )
    return out


def reject_non_fed_stock_targets(targets: Mapping[str, Any]) -> None:
    """Reject stock-target controls outside the Fed path for CBO scenario v1."""

    unknown = sorted(key for key in targets if key not in {"CB", "FedInternal"})
    if unknown:
        raise PortfolioTransformError(f"non-Fed stock targets are not supported in CBO scenario v1: {unknown}")


def _maturity_distributions(raw: Any) -> Mapping[str, tuple[Mapping[str, float], ...]]:
    if not isinstance(raw, Mapping):
        raise PortfolioTransformError("maturity_distributions must be a mapping")
    out: dict[str, tuple[Mapping[str, float], ...]] = {}
    for security_type in MARKETABLE_SECURITY_TYPES:
        rows = raw.get(security_type)
        if not isinstance(rows, list) or not rows:
            raise PortfolioTransformError(f"{security_type} maturity distribution is required")
        normalized = []
        for item in rows:
            maturity = float(item["maturity_years"])
            share = float(item["share"])
            if maturity <= 0:
                raise PortfolioTransformError("maturity_years must be positive")
            normalized.append({"maturity_years": maturity, "share": share})
        _assert_unit_sum((item["share"] for item in normalized), label=f"{security_type} maturity distribution")
        out[security_type] = tuple(normalized)
    return out


def _marketable_debt_by_date(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, float]:
    if rows is None:
        return {}
    return {str(row["period_end"]): float(row["marketable_treasury_public_target_bil"]) for row in rows}


def _anchors(override: Mapping[str, Any]) -> dict[int, float]:
    raw = override.get("anchors")
    if not isinstance(raw, list) or not raw:
        raise PortfolioTransformError("fy_endpoint_anchors requires anchors")
    anchors = {int(item["fiscal_year"]): float(item["value_bil"]) for item in raw}
    if len(anchors) != len(raw):
        raise PortfolioTransformError("duplicate fiscal-year anchors are not allowed")
    return dict(sorted(anchors.items()))


def _fiscal_year(row: Mapping[str, Any]) -> int:
    for key in ("source_fiscal_year", "fiscal_year"):
        if key in row and row[key] not in ("", None):
            return int(row[key])
    raise PortfolioTransformError("row is missing source_fiscal_year")


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
    raise PortfolioTransformError("failed to interpolate fiscal-year anchor")


def _assert_unit_sum(values: Iterable[float], *, label: str) -> None:
    total = sum(float(value) for value in values)
    if abs(total - 1.0) > 1e-9:
        raise PortfolioTransformError(f"{label} must sum to 1.0, got {total}")


__all__ = [
    "IssuanceMix",
    "PortfolioTransformError",
    "apply_fed_holdings_override",
    "compile_issuance_mix_override",
    "reject_non_fed_stock_targets",
    "validate_holder_preferences",
]
