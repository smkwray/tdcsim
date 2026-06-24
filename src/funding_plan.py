"""Pure CBO debt-target funding-plan helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from sim_pricing import quote_issuance_from_face_target

DEFAULT_TOLERANCE = 1e-9
PUBLIC_MARKETABLE_SECURITY_TYPES = frozenset({"Fixed", "TIPS", "FRN", "Bill", "Note", "Bond"})
NONPUBLIC_HOLDERS = frozenset({"TrustFunds", "FedInternal", "Intragovernmental"})


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _security_type(row: Mapping[str, Any]) -> str:
    return str(row.get("security_type", row.get("SecurityType", "")))


def _holder_type(row: Mapping[str, Any]) -> str:
    return str(row.get("holder_type", row.get("HolderType", "")))


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("status", row.get("Status", "Active")))


def is_public_marketable_security(row: Mapping[str, Any]) -> bool:
    """Return whether a row belongs in the public marketable debt perimeter."""
    if _status(row) != "Active":
        return False
    security_type = _security_type(row)
    if security_type == "NonMarketable":
        return False
    if _holder_type(row) in NONPUBLIC_HOLDERS:
        return False
    return security_type in PUBLIC_MARKETABLE_SECURITY_TYPES


def controlled_debt_face_value(row: Mapping[str, Any]) -> float:
    """Debt-control face value, using adjusted principal for TIPS when present."""
    security_type = _security_type(row)
    if security_type == "TIPS":
        adjusted = row.get("adjusted_principal", row.get("AdjustedPrincipal"))
        if adjusted is not None:
            return _as_float(adjusted)
    return _as_float(row.get("face_value", row.get("FaceValue")))


def calculate_pre_issuance_controlled_debt(
    opening_positions: Iterable[Mapping[str, Any]],
) -> float:
    """Sum public marketable controlled debt before new issuance."""
    return sum(
        controlled_debt_face_value(position)
        for position in opening_positions
        if is_public_marketable_security(position)
    )


def calculate_required_face_issuance(
    controlled_debt_target: float,
    pre_issuance_controlled_debt: float,
    *,
    strict: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Return face issuance needed to hit a controlled-debt target."""
    required = float(controlled_debt_target) - float(pre_issuance_controlled_debt)
    if abs(required) <= tolerance:
        return 0.0
    if required < 0.0 and strict:
        raise ValueError(
            "negative required face issuance is unsupported without an explicit buyback or rolloff policy"
        )
    return required


def _normalize_funding_rows(
    funding_rows: Iterable[Mapping[str, Any]],
    *,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for index, row in enumerate(funding_rows):
        row_dict = dict(row)
        security_type = _security_type(row_dict)
        share = _as_float(row_dict.get("face_share", row_dict.get("share")), default=0.0)
        if share <= tolerance:
            diagnostics.append(
                {
                    "index": index,
                    "security_type": security_type,
                    "eligible": False,
                    "reason": "nonpositive_face_share",
                }
            )
            continue
        if not is_public_marketable_security(row_dict):
            diagnostics.append(
                {
                    "index": index,
                    "security_type": security_type,
                    "eligible": False,
                    "reason": "excluded_from_public_marketable_auction_supply",
                }
            )
            continue
        row_dict["face_share"] = share
        eligible.append(row_dict)
        diagnostics.append(
            {
                "index": index,
                "security_type": security_type,
                "eligible": True,
                "reason": "included_public_marketable",
            }
        )
    total_share = sum(row["face_share"] for row in eligible)
    if total_share > tolerance and abs(total_share - 1.0) > tolerance:
        diagnostics.append(
            {
                "eligible": True,
                "reason": "eligible_face_shares_normalized",
                "raw_share_sum": total_share,
            }
        )
        for row in eligible:
            row["face_share"] = row["face_share"] / total_share
    return eligible, diagnostics


def build_funding_plan(
    *,
    controlled_debt_target: float,
    pre_issuance_controlled_debt: float | None = None,
    opening_positions: Iterable[Mapping[str, Any]] | None = None,
    funding_rows: Iterable[Mapping[str, Any]],
    strict: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    holder_preferences: Mapping[str, Any] | None = None,
    cbo_net_interest_bil: float | None = None,
    quote_func: Callable[[str, float, float, float, float], tuple[float, float, float]] = quote_issuance_from_face_target,
) -> dict[str, Any]:
    """Build a deterministic face-sized funding plan.

    ``holder_preferences`` and ``cbo_net_interest_bil`` are accepted to make the
    nonbinding boundary explicit; they do not affect issuance sizing or pricing.
    """
    del holder_preferences, cbo_net_interest_bil
    if pre_issuance_controlled_debt is None:
        if opening_positions is None:
            raise ValueError("pre_issuance_controlled_debt or opening_positions is required")
        pre_issuance_controlled_debt = calculate_pre_issuance_controlled_debt(opening_positions)
    pre_issuance_controlled_debt = float(pre_issuance_controlled_debt)
    required_face_issuance = calculate_required_face_issuance(
        controlled_debt_target,
        pre_issuance_controlled_debt,
        strict=strict,
        tolerance=tolerance,
    )
    eligible_rows, eligibility_diagnostics = _normalize_funding_rows(funding_rows, tolerance=tolerance)
    if required_face_issuance > tolerance and not eligible_rows:
        raise ValueError("positive required face issuance has no eligible public marketable funding rows")

    face_by_security_and_maturity: dict[tuple[str, float], float] = {}
    quote_rows: list[dict[str, Any]] = []
    auction_proceeds = 0.0
    for row in eligible_rows:
        security_type = _security_type(row)
        maturity_years = _as_float(row.get("maturity_years", row.get("OriginalMaturityYears")))
        coupon_rate = _as_float(row.get("coupon_rate", row.get("CouponRate")))
        yield_at_issuance = _as_float(row.get("yield_at_issuance", row.get("IssueYieldAtIssue")))
        face_target = required_face_issuance * row["face_share"]
        quoted_face, proceeds, issue_price_ratio = quote_func(
            security_type,
            maturity_years,
            coupon_rate,
            yield_at_issuance,
            face_target,
        )
        key = (security_type, maturity_years)
        face_by_security_and_maturity[key] = face_by_security_and_maturity.get(key, 0.0) + quoted_face
        auction_proceeds += proceeds
        quote_rows.append(
            {
                "security_type": security_type,
                "maturity_years": maturity_years,
                "face_value": quoted_face,
                "auction_proceeds": proceeds,
                "issue_price_ratio": issue_price_ratio,
            }
        )

    issued_face = sum(face_by_security_and_maturity.values())
    issue_discount_or_premium = issued_face - auction_proceeds
    post_issuance_controlled_debt = pre_issuance_controlled_debt + issued_face
    target_error = post_issuance_controlled_debt - float(controlled_debt_target)
    return {
        "pre_issuance_controlled_debt": pre_issuance_controlled_debt,
        "required_face_issuance": required_face_issuance,
        "face_by_security_and_maturity": face_by_security_and_maturity,
        "auction_proceeds": auction_proceeds,
        "issue_discount_or_premium": issue_discount_or_premium,
        "post_issuance_controlled_debt": post_issuance_controlled_debt,
        "target_error": target_error,
        "eligibility_diagnostics": eligibility_diagnostics,
        "quote_rows": quote_rows,
    }


__all__ = [
    "calculate_pre_issuance_controlled_debt",
    "calculate_required_face_issuance",
    "build_funding_plan",
    "controlled_debt_face_value",
    "is_public_marketable_security",
]
