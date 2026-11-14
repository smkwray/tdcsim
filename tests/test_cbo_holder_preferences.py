from __future__ import annotations

import pytest

from cbo_holder_preferences import (
    build_holder_profile_rows,
    default_cbo_holder_preferences,
    validate_holder_preferences_non_degenerate,
)


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
