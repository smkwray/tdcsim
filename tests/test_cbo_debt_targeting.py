from __future__ import annotations

import pytest

from debt_targeting import (
    NegativeIssuanceError,
    adjusted_tips_principal,
    auction_eligible_face,
    calculate_controlled_marketable_target,
    calculate_required_face_issuance,
    controlled_public_marketable_debt,
)


def test_cbo_target_decomposition_excludes_bridge_components() -> None:
    decomposition = calculate_controlled_marketable_target(
        cbo_federal_debt_held_public_target_bil=32_095.0,
        public_nonmarketable_treasury_bil=500.0,
        non_treasury_and_definition_residual_bil=45.0,
    )

    assert decomposition.marketable_treasury_public_target_bil == pytest.approx(31_550.0)
    assert decomposition.public_nonmarketable_treasury_bil == pytest.approx(500.0)
    assert decomposition.non_treasury_and_definition_residual_bil == pytest.approx(45.0)


def test_adjusted_tips_principal_uses_adjusted_amount_before_face_value() -> None:
    assert adjusted_tips_principal(100.0, adjusted_principal=112.5) == pytest.approx(112.5)
    assert adjusted_tips_principal(100.0, index_ratio=1.08) == pytest.approx(108.0)


def test_controlled_public_marketable_debt_uses_adjusted_tips_and_excludes_nonmarketables() -> None:
    positions = [
        {"SecurityType": "Fixed", "FaceValue": 100.0, "Status": "Active"},
        {"SecurityType": "TIPS", "FaceValue": 100.0, "AdjustedPrincipal": 112.5, "Status": "Active"},
        {"SecurityType": "FRN", "FaceValue": 25.0, "Status": "Active"},
        {"SecurityType": "NonMarketable", "FaceValue": 900.0, "Status": "Active"},
        {"SecurityType": "Fixed", "FaceValue": 20.0, "Status": "Matured"},
    ]

    assert controlled_public_marketable_debt(positions) == pytest.approx(237.5)


def test_public_nonmarketables_are_excluded_from_auction_supply() -> None:
    positions = [
        {"SecurityType": "Fixed", "FaceValue": 100.0, "Status": "Active"},
        {"SecurityType": "TIPS", "FaceValue": 100.0, "AdjustedPrincipal": 112.5, "Status": "Active"},
        {"SecurityType": "FRN", "FaceValue": 25.0, "Status": "Active"},
        {"SecurityType": "NonMarketable", "FaceValue": 900.0, "Status": "Active"},
    ]

    assert auction_eligible_face(positions) == pytest.approx(225.0)


def test_required_face_issuance_zeros_near_tolerance() -> None:
    assert calculate_required_face_issuance(100.0, 99.9999999995, tolerance_bil=1e-6) == 0.0


def test_required_face_issuance_fails_closed_when_negative_in_strict_mode() -> None:
    with pytest.raises(NegativeIssuanceError, match="required face issuance is negative"):
        calculate_required_face_issuance(100.0, 101.0, strict=True)


def test_required_face_issuance_can_report_negative_when_strict_is_disabled() -> None:
    assert calculate_required_face_issuance(100.0, 101.0, strict=False) == pytest.approx(-1.0)
