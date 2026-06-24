from __future__ import annotations

import pytest

from funding_plan import (
    build_funding_plan,
    calculate_pre_issuance_controlled_debt,
    calculate_required_face_issuance,
)
from sim_pricing import quote_issuance_from_face_target


def _bill_yield_for_price(price_ratio: float) -> float:
    return (1.0 / price_ratio) - 1.0


def test_quote_issuance_from_face_target_prices_face_into_proceeds():
    face, proceeds, price_ratio = quote_issuance_from_face_target(
        "Fixed",
        1.0,
        0.0,
        _bill_yield_for_price(0.98),
        100.0,
    )

    assert face == pytest.approx(100.0)
    assert proceeds == pytest.approx(98.0)
    assert price_ratio == pytest.approx(0.98)


def test_quote_issuance_from_face_target_does_not_floor_negative_bill_yield():
    face, proceeds, price_ratio = quote_issuance_from_face_target(
        "Fixed",
        1.0,
        0.0,
        -0.01,
        100.0,
    )

    assert face == pytest.approx(100.0)
    assert price_ratio == pytest.approx(1.0 / 0.99)
    assert proceeds == pytest.approx(100.0 / 0.99)


def test_funding_plan_distinguishes_face_control_from_cash_proceeds():
    plan = build_funding_plan(
        controlled_debt_target=1100.0,
        pre_issuance_controlled_debt=1000.0,
        funding_rows=[
            {
                "security_type": "Fixed",
                "maturity_years": 1.0,
                "coupon_rate": 0.0,
                "yield_at_issuance": _bill_yield_for_price(0.98),
                "face_share": 1.0,
            }
        ],
    )

    assert plan["required_face_issuance"] == pytest.approx(100.0)
    assert plan["face_by_security_and_maturity"][("Fixed", 1.0)] == pytest.approx(100.0)
    assert plan["post_issuance_controlled_debt"] == pytest.approx(1100.0)
    assert plan["auction_proceeds"] == pytest.approx(98.0)
    assert plan["issue_discount_or_premium"] == pytest.approx(2.0)
    assert plan["target_error"] == pytest.approx(0.0)


def test_holder_preferences_do_not_resize_fixed_funding_plan():
    kwargs = {
        "controlled_debt_target": 1100.0,
        "pre_issuance_controlled_debt": 1000.0,
        "funding_rows": [
            {"security_type": "Fixed", "maturity_years": 2.0, "coupon_rate": 0.03, "face_share": 0.4},
            {"security_type": "FRN", "maturity_years": 2.0, "face_share": 0.6},
        ],
    }
    banks_heavy = build_funding_plan(
        **kwargs,
        holder_preferences={"Banks": {"notes_pct": 1.0}, "Private": {"notes_pct": 0.0}},
    )
    private_heavy = build_funding_plan(
        **kwargs,
        holder_preferences={"Banks": {"notes_pct": 0.0}, "Private": {"notes_pct": 1.0}},
    )

    assert banks_heavy["required_face_issuance"] == pytest.approx(private_heavy["required_face_issuance"])
    assert banks_heavy["face_by_security_and_maturity"] == private_heavy["face_by_security_and_maturity"]
    assert banks_heavy["auction_proceeds"] == pytest.approx(private_heavy["auction_proceeds"])
    assert banks_heavy["post_issuance_controlled_debt"] == pytest.approx(
        private_heavy["post_issuance_controlled_debt"]
    )


def test_negative_required_face_issuance_fails_in_strict_mode():
    with pytest.raises(ValueError, match="negative required face issuance"):
        calculate_required_face_issuance(990.0, 1000.0, strict=True)


def test_public_nonmarketables_and_residual_bridges_do_not_enter_auction_supply():
    opening_positions = [
        {"SecurityType": "Fixed", "HolderType": "Private", "FaceValue": 1000.0},
        {"SecurityType": "NonMarketable", "HolderType": "Private", "FaceValue": 75.0},
        {"SecurityType": "ResidualBridge", "HolderType": "Private", "FaceValue": 20.0},
        {"SecurityType": "Fixed", "HolderType": "TrustFunds", "FaceValue": 50.0},
        {"SecurityType": "TIPS", "HolderType": "Foreign", "FaceValue": 100.0, "AdjustedPrincipal": 105.0},
    ]

    assert calculate_pre_issuance_controlled_debt(opening_positions) == pytest.approx(1105.0)

    plan = build_funding_plan(
        controlled_debt_target=1205.0,
        opening_positions=opening_positions,
        funding_rows=[
            {"security_type": "NonMarketable", "maturity_years": 1.0, "face_share": 0.25},
            {"security_type": "ResidualBridge", "maturity_years": 1.0, "face_share": 0.25},
            {"security_type": "Fixed", "maturity_years": 2.0, "coupon_rate": 0.03, "face_share": 0.5},
        ],
    )

    assert plan["required_face_issuance"] == pytest.approx(100.0)
    assert plan["face_by_security_and_maturity"] == {("Fixed", 2.0): pytest.approx(100.0)}
    assert any(
        diag["reason"] == "excluded_from_public_marketable_auction_supply"
        and diag["security_type"] == "NonMarketable"
        for diag in plan["eligibility_diagnostics"]
    )
    assert any(
        diag["reason"] == "excluded_from_public_marketable_auction_supply"
        and diag["security_type"] == "ResidualBridge"
        for diag in plan["eligibility_diagnostics"]
    )


def test_pre_issuance_controlled_debt_excludes_inactive_rows():
    opening_positions = [
        {"SecurityType": "Fixed", "HolderType": "Private", "FaceValue": 1000.0, "Status": "Active"},
        {"SecurityType": "Fixed", "HolderType": "Private", "FaceValue": 25.0, "Status": "Matured"},
        {"SecurityType": "TIPS", "HolderType": "Foreign", "FaceValue": 100.0, "AdjustedPrincipal": 105.0},
    ]

    assert calculate_pre_issuance_controlled_debt(opening_positions) == pytest.approx(1105.0)


def test_cbo_net_interest_target_is_diagnostic_only_for_funding_plan():
    kwargs = {
        "controlled_debt_target": 1100.0,
        "pre_issuance_controlled_debt": 1000.0,
        "funding_rows": [{"security_type": "Fixed", "maturity_years": 2.0, "coupon_rate": 0.03, "face_share": 1.0}],
    }

    assert build_funding_plan(**kwargs, cbo_net_interest_bil=100.0) == build_funding_plan(
        **kwargs,
        cbo_net_interest_bil=250.0,
    )
