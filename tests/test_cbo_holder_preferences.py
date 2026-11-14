from __future__ import annotations

import json

import pandas as pd
import pytest

from cbo_holder_preferences import (
    build_holder_profile_rows,
    holder_profile_from_opening_weights,
    validate_holder_preferences_non_degenerate,
)
from scripts.run_cbo_forecast_smoke import load_opening_portfolio
from tdcsim_cbo.runner import _holder_preferences

WEIGHTS = {
    "bills": {
        "Banks": 0.10,
        "Foreign": 0.20,
        "Private/domestic_nonbank_deposit_funded": 0.35,
        "Private/mmf_cash_fund_route": 0.35,
    },
    "nonbills": {
        "Banks": 0.08,
        "Foreign": 0.44,
        "Private/domestic_nonbank_deposit_funded": 0.43,
        "Private/mmf_cash_fund_route": 0.05,
    },
}


def test_new_issuance_follows_opening_holdings_of_bills_or_longer_securities():
    preferences, routes = holder_profile_from_opening_weights(WEIGHTS)

    validate_holder_preferences_non_degenerate(preferences)
    # Bills follow bill holdings; every other type follows holdings of longer securities.
    assert preferences["Foreign"]["bills_pct"] == pytest.approx(0.20)
    for category in ("notes", "bonds", "tips", "frn"):
        assert preferences["Foreign"][f"{category}_pct"] == pytest.approx(0.44)
        assert preferences["Private"][f"{category}_pct"] == pytest.approx(0.48)
    # Money funds take half of private bill purchases but about a tenth of private note purchases,
    # so their deposit pass-through applies mainly to bill flows.
    assert routes["bills"]["mmf_cash_fund_route"] == pytest.approx(0.5)
    assert routes["notes"]["mmf_cash_fund_route"] == pytest.approx(0.05 / 0.48)
    # The Fed follows its own stock path and takes no new issuance.
    assert all(value == 0.0 for value in preferences["CB"].values())


def test_profile_rows_carry_holder_and_route_shares():
    preferences, routes = holder_profile_from_opening_weights(WEIGHTS)
    rows = build_holder_profile_rows(scenario_id="central", holder_preferences=preferences, private_subbucket_shares=routes)
    route_rows = [row for row in rows if row["holder_type"] == "Private" and row["holder_subbucket"]]

    assert {row["holder_subbucket"] for row in route_rows} == {"domestic_nonbank_deposit_funded", "mmf_cash_fund_route"}
    for category in ("bills", "notes", "bonds", "tips", "frn"):
        assert sum(float(row[f"{category}_route_share"]) for row in route_rows) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "weights",
    [
        {"bills": WEIGHTS["bills"]},
        {**WEIGHTS, "bills": {**WEIGHTS["bills"], "Banks": 0.30}},
        {**WEIGHTS, "nonbills": {"Banks": 1.0}},
    ],
)
def test_malformed_opening_weights_raise(weights):
    with pytest.raises(ValueError, match="noncb_weights"):
        holder_profile_from_opening_weights(weights)


def test_runner_reads_the_profile_the_package_writes(tmp_path):
    # The package build and `tdcsim-cbo run` must divide new issuance the same way.

    preferences, routes = holder_profile_from_opening_weights(WEIGHTS)
    rows = build_holder_profile_rows(scenario_id="central", holder_preferences=preferences, private_subbucket_shares=routes)
    path = tmp_path / "tdcsim_holder_profile_assumptions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    loaded = _holder_preferences(path)
    for holder in ("Banks", "Foreign", "Private", "CB"):
        for category in ("bills", "notes", "frn"):
            assert loaded[holder][f"{category}_pct"] == pytest.approx(preferences[holder][f"{category}_pct"])
    assert loaded["__private_subbucket_shares__"]["bills"]["mmf_cash_fund_route"] == pytest.approx(0.5)


def test_package_build_rejects_an_opening_book_without_soma_holdings(tmp_path):
    book = tmp_path / "book.csv"
    book.write_text("BondID\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {"initial_portfolio": {"mspd_record_date": "2026-05-31"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="SOMA-based CB holdings"):
        load_opening_portfolio(book, simulation_start_date="2026-06-21", source_manifest=manifest)
