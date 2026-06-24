"""Pure helpers for CBO public-debt targeting calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

PUBLIC_MARKETABLE_SECURITY_TYPES = frozenset({"Fixed", "TIPS", "FRN"})
DEFAULT_TOLERANCE_BIL = 1e-9


class NegativeIssuanceError(ValueError):
    """Raised when strict debt-targeting would require negative issuance."""


@dataclass(frozen=True)
class DebtTargetDecomposition:
    cbo_federal_debt_held_public_target_bil: float
    public_nonmarketable_treasury_bil: float
    non_treasury_and_definition_residual_bil: float
    marketable_treasury_public_target_bil: float


def _finite_float(value: object, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} is required")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _get_value(row: Mapping[str, object], keys: Sequence[str], default: object = None) -> object:
    for key in keys:
        if key in row:
            return row[key]
    return default


def calculate_controlled_marketable_target(
    cbo_federal_debt_held_public_target_bil: object,
    public_nonmarketable_treasury_bil: object,
    non_treasury_and_definition_residual_bil: object,
) -> DebtTargetDecomposition:
    """Decompose CBO debt held by the public into the marketable target scope."""

    cbo_target = _finite_float(
        cbo_federal_debt_held_public_target_bil,
        "cbo_federal_debt_held_public_target_bil",
    )
    public_nonmarketable = _finite_float(
        public_nonmarketable_treasury_bil,
        "public_nonmarketable_treasury_bil",
    )
    residual = _finite_float(
        non_treasury_and_definition_residual_bil,
        "non_treasury_and_definition_residual_bil",
    )
    controlled_target = cbo_target - public_nonmarketable - residual
    if controlled_target < -DEFAULT_TOLERANCE_BIL:
        raise ValueError("controlled marketable target cannot be negative")
    if abs(controlled_target) <= DEFAULT_TOLERANCE_BIL:
        controlled_target = 0.0
    return DebtTargetDecomposition(
        cbo_federal_debt_held_public_target_bil=cbo_target,
        public_nonmarketable_treasury_bil=public_nonmarketable,
        non_treasury_and_definition_residual_bil=residual,
        marketable_treasury_public_target_bil=controlled_target,
    )


def adjusted_tips_principal(
    face_value: object,
    *,
    adjusted_principal: object | None = None,
    index_ratio: object | None = None,
) -> float:
    """Return TIPS adjusted principal from an explicit amount or index ratio."""

    face = _finite_float(face_value, "face_value")
    if adjusted_principal is not None:
        adjusted = _finite_float(adjusted_principal, "adjusted_principal")
    else:
        ratio = 1.0 if index_ratio is None else _finite_float(index_ratio, "index_ratio")
        adjusted = face * ratio
    if adjusted < -DEFAULT_TOLERANCE_BIL:
        raise ValueError("adjusted TIPS principal cannot be negative")
    return 0.0 if abs(adjusted) <= DEFAULT_TOLERANCE_BIL else adjusted


def is_public_marketable_security(security_type: object) -> bool:
    return str(security_type) in PUBLIC_MARKETABLE_SECURITY_TYPES


def is_active_position(row: Mapping[str, object]) -> bool:
    status = _get_value(row, ("Status", "status"), "Active")
    return str(status) == "Active"


def position_controlled_debt_base(row: Mapping[str, object]) -> float:
    """Return the controlled public-marketable debt base for one position."""

    if not is_active_position(row):
        return 0.0
    security_type = _get_value(row, ("SecurityType", "security_type"))
    if not is_public_marketable_security(security_type):
        return 0.0
    face_value = _get_value(row, ("FaceValue", "face_value", "face_bil"))
    if str(security_type) == "TIPS":
        return adjusted_tips_principal(
            face_value,
            adjusted_principal=_get_value(row, ("AdjustedPrincipal", "adjusted_principal"), None),
            index_ratio=_get_value(row, ("IndexRatio", "index_ratio"), None),
        )
    return _finite_float(face_value, "face_value")


def controlled_public_marketable_debt(positions: Iterable[Mapping[str, object]]) -> float:
    """Sum debt stock in CBO-controlled public marketable scope."""

    return sum(position_controlled_debt_base(row) for row in positions)


def auction_eligible_face(positions: Iterable[Mapping[str, object]]) -> float:
    """Sum face in the marketable auction supply scope, excluding nonmarketables."""

    total = 0.0
    for row in positions:
        if not is_active_position(row):
            continue
        security_type = _get_value(row, ("SecurityType", "security_type"))
        if not is_public_marketable_security(security_type):
            continue
        total += _finite_float(_get_value(row, ("FaceValue", "face_value", "face_bil")), "face_value")
    return total


def calculate_required_face_issuance(
    target_controlled_debt_bil: object,
    pre_issuance_controlled_debt_bil: object,
    *,
    tolerance_bil: float = DEFAULT_TOLERANCE_BIL,
    strict: bool = True,
) -> float:
    """Calculate face issuance needed to hit the controlled debt target."""

    target = _finite_float(target_controlled_debt_bil, "target_controlled_debt_bil")
    pre_issuance = _finite_float(pre_issuance_controlled_debt_bil, "pre_issuance_controlled_debt_bil")
    tolerance = _finite_float(tolerance_bil, "tolerance_bil")
    if tolerance < 0:
        raise ValueError("tolerance_bil must be nonnegative")
    required = target - pre_issuance
    if abs(required) <= tolerance:
        return 0.0
    if strict and required < 0:
        raise NegativeIssuanceError(
            f"required face issuance is negative: target={target}, pre_issuance={pre_issuance}"
        )
    return required
