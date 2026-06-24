from __future__ import annotations

import pytest

from cbo_portfolio_builder import (
    PUBLIC_MARKETABLE_CLASSES,
    build_cbo_opening_public_marketable_portfolio,
    build_holder_profile_rows,
    default_cbo_holder_preferences,
    extract_mspd_public_marketable_class_totals,
    validate_holder_preferences_non_degenerate,
)


def _mspd_rows():
    return [
        {
            "record_date": "2026-05-31",
            "security_type_desc": "Marketable",
            "security_class_desc": "Bills",
            "debt_held_public_mil_amt": "100000.0",
        },
        {
            "record_date": "2026-05-31",
            "security_type_desc": "Marketable",
            "security_class_desc": "Notes",
            "debt_held_public_mil_amt": "200000.0",
        },
        {
            "record_date": "2026-05-31",
            "security_type_desc": "Marketable",
            "security_class_desc": "Bonds",
            "debt_held_public_mil_amt": "300000.0",
        },
        {
            "record_date": "2026-05-31",
            "security_type_desc": "Marketable",
            "security_class_desc": "Treasury Inflation-Protected Securities (TIPS)",
            "debt_held_public_mil_amt": "50000.0",
        },
        {
            "record_date": "2026-05-31",
            "security_type_desc": "Marketable",
            "security_class_desc": "Floating Rate Notes",
            "debt_held_public_mil_amt": "25000.0",
        },
        {
            "record_date": "2026-05-31",
            "security_type_desc": "Total Marketable",
            "security_class_desc": "",
            "debt_held_public_mil_amt": "675000.0",
        },
        {
            "record_date": "2026-05-31",
            "security_type_desc": "Total Nonmarketable",
            "security_class_desc": "",
            "debt_held_public_mil_amt": "12000.0",
        },
    ]


def _class_total_from_portfolio(rows, class_name):
    if class_name == "Bills":
        selected = [
            row for row in rows if row["SecurityType"] == "Fixed" and row["MaturityCategory"] == "bills"
        ]
        return sum(float(row["FaceValue"]) for row in selected)
    if class_name == "Notes":
        selected = [
            row for row in rows if row["SecurityType"] == "Fixed" and row["MaturityCategory"] == "notes"
        ]
        return sum(float(row["FaceValue"]) for row in selected)
    if class_name == "Bonds":
        selected = [
            row for row in rows if row["SecurityType"] == "Fixed" and row["MaturityCategory"] == "bonds"
        ]
        return sum(float(row["FaceValue"]) for row in selected)
    if class_name == "TIPS":
        return sum(float(row["AdjustedPrincipal"]) for row in rows if row["SecurityType"] == "TIPS")
    if class_name == "FRN":
        return sum(float(row["FaceValue"]) for row in rows if row["SecurityType"] == "FRN")
    raise AssertionError(f"unknown class {class_name}")


def test_extracts_mspd_public_marketable_class_totals_and_bridge_components():
    extracted = extract_mspd_public_marketable_class_totals(_mspd_rows())

    assert extracted["class_totals_bil"] == {
        "Bills": pytest.approx(100.0),
        "Notes": pytest.approx(200.0),
        "Bonds": pytest.approx(300.0),
        "TIPS": pytest.approx(50.0),
        "FRN": pytest.approx(25.0),
    }
    assert extracted["total_marketable_bil"] == pytest.approx(675.0)
    assert extracted["nonmarketable_bridge_components"] == [
        {
            "source_selector": "security_type_desc=Total Nonmarketable",
            "amount_bil": pytest.approx(12.0),
            "source_role": "hard_actual_state",
            "runtime_role": "reconciliation_only",
            "claim_boundary": "public_nonmarketables_are_bridge_components_not_auction_securities",
        }
    ]


def test_opening_portfolio_reconciles_to_mspd_public_marketable_class_totals():
    bundle = build_cbo_opening_public_marketable_portfolio(
        mspd_table_1_rows=_mspd_rows(),
        opening_state_date="2026-05-31",
    )
    rows = bundle["portfolio_rows"]
    metadata = bundle["metadata"]

    for class_name in PUBLIC_MARKETABLE_CLASSES:
        assert _class_total_from_portfolio(rows, class_name) == pytest.approx(
            metadata["class_totals_bil"][class_name],
            abs=1e-9,
        )

    assert sum(_class_total_from_portfolio(rows, class_name) for class_name in PUBLIC_MARKETABLE_CLASSES) == (
        pytest.approx(675.0)
    )


def test_tips_adjusted_principal_represents_mspd_public_total():
    rows = build_cbo_opening_public_marketable_portfolio(
        mspd_table_1_rows=_mspd_rows(),
        opening_state_date="2026-05-31",
    )["portfolio_rows"]
    tips_rows = [row for row in rows if row["SecurityType"] == "TIPS"]

    assert tips_rows
    assert sum(float(row["AdjustedPrincipal"]) for row in tips_rows) == pytest.approx(50.0)
    assert all(float(row["AdjustedPrincipal"]) == pytest.approx(float(row["FaceValue"])) for row in tips_rows)
    assert all(float(row["IndexRatio"]) == pytest.approx(1.0) for row in tips_rows)


def test_nonmarketables_are_excluded_from_opening_public_marketable_portfolio():
    bundle = build_cbo_opening_public_marketable_portfolio(
        mspd_table_1_rows=_mspd_rows(),
        opening_state_date="2026-05-31",
    )
    rows = bundle["portfolio_rows"]
    metadata = bundle["metadata"]

    assert "NonMarketable" not in {row["SecurityType"] for row in rows}
    assert metadata["nonmarketable_bridge_components"][0]["amount_bil"] == pytest.approx(12.0)
    assert "not auction securities" in metadata["claim_boundary"]["excluded"]
    assert "not exact CUSIP" in metadata["claim_boundary"]["not_claimed"]


def test_holder_preferences_are_non_degenerate_and_use_engine_sector_names():
    preferences = default_cbo_holder_preferences(include_private_routes=False)

    validate_holder_preferences_non_degenerate(preferences)
    assert set(preferences) == {"Banks", "Private", "CB", "Foreign", "FedInternal", "TrustFunds"}
    for category in ("bills", "notes", "bonds", "tips", "frn"):
        key = f"{category}_pct"
        assert sum(float(preferences[holder][key]) for holder in preferences) == pytest.approx(1.0)
        positive_holders = [holder for holder in preferences if preferences[holder][key] > 0.0]
        assert positive_holders != ["Private"]
        assert "CB" not in positive_holders
        assert "FedInternal" not in positive_holders
        assert "TrustFunds" not in positive_holders


def test_holder_profile_rows_include_supported_private_routes_only():
    rows = build_holder_profile_rows(scenario_id="central")
    route_rows = [row for row in rows if row["holder_type"] == "Private" and row["holder_subbucket"]]

    assert {row["holder_subbucket"] for row in route_rows} == {
        "domestic_nonbank_deposit_funded",
        "mmf_cash_fund_route",
    }
    for category in ("bills", "notes", "bonds", "tips", "frn"):
        assert sum(float(row[f"{category}_route_share"]) for row in route_rows) == pytest.approx(1.0)
