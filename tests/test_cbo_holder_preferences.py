from __future__ import annotations

import json

import pandas as pd
import pytest

from cbo_holder_preferences import (
    build_holder_profile_rows,
    holder_profile_from_opening_weights,
    validate_holder_preferences_non_degenerate,
)
from scripts.write_scenario_family import shifted_holder_rows
from scripts.run_cbo_forecast_smoke import load_opening_portfolio
from sim_engine import _private_subbucket_split, run_simulation
from tdc_shared import HOLDER_TYPES, PRIVATE_SUBBUCKET_SHARES_KEY
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
    # Money funds' calibrated share of new FRNs, from N-MFP.
    "frn_mmf_auction_share": 0.70,
}


def test_new_issuance_follows_opening_holdings_of_bills_or_longer_securities():
    preferences, routes = holder_profile_from_opening_weights(WEIGHTS)

    validate_holder_preferences_non_degenerate(preferences)
    # Bills keep their measured shares. Each longer security removes the money-fund share
    # while preserving the proportions among the other holders.
    assert preferences["Foreign"]["bills_pct"] == pytest.approx(0.20)
    for category in ("notes", "bonds", "tips"):
        assert preferences["Banks"][f"{category}_pct"] == pytest.approx(0.08 / 0.95)
        assert preferences["Foreign"][f"{category}_pct"] == pytest.approx(0.44 / 0.95)
        assert preferences["Private"][f"{category}_pct"] == pytest.approx(0.43 / 0.95)
        assert routes[category]["mmf_cash_fund_route"] == 0.0
        assert routes[category]["domestic_nonbank_deposit_funded"] == 1.0
    assert routes["bills"]["mmf_cash_fund_route"] == pytest.approx(0.5)
    # The Fed follows its own stock path and takes no new issuance.
    assert all(value == 0.0 for value in preferences["CB"].values())


def test_money_funds_take_the_calibrated_frn_share_and_others_scale_pro_rata():
    # Money funds hold about 70% of new FRN supply a month after issue, an order of magnitude
    # above their 5% WAMEST share of longer securities. The other holders keep their WAMEST
    # proportions over the remaining 30%.
    preferences, routes = holder_profile_from_opening_weights(WEIGHTS)
    mmf = preferences["Private"]["frn_pct"] * routes["frn"]["mmf_cash_fund_route"]
    dn = preferences["Private"]["frn_pct"] * routes["frn"]["domestic_nonbank_deposit_funded"]

    assert mmf == pytest.approx(0.70)
    scale = 0.30 / 0.95
    assert preferences["Banks"]["frn_pct"] == pytest.approx(0.08 * scale)
    assert preferences["Foreign"]["frn_pct"] == pytest.approx(0.44 * scale)
    assert dn == pytest.approx(0.43 * scale)
    assert sum(preferences[holder]["frn_pct"] for holder in HOLDER_TYPES) == pytest.approx(1.0)
    # Only FRNs change: notes still give money funds nothing, and bills keep their WAMEST split.
    other = holder_profile_from_opening_weights({**WEIGHTS, "frn_mmf_auction_share": 0.20})
    assert other[0]["Foreign"]["notes_pct"] == pytest.approx(preferences["Foreign"]["notes_pct"])
    assert other[1]["bills"] == pytest.approx(routes["bills"])
    assert other[0]["Foreign"]["frn_pct"] == pytest.approx(0.44 * 0.80 / 0.95)


def test_profile_rows_carry_holder_and_route_shares():
    preferences, routes = holder_profile_from_opening_weights(WEIGHTS)
    rows = build_holder_profile_rows(scenario_id="central", holder_preferences=preferences, private_subbucket_shares=routes)
    route_rows = [row for row in rows if row["holder_type"] == "Private" and row["holder_subbucket"]]

    assert {row["holder_subbucket"] for row in route_rows} == {"domestic_nonbank_deposit_funded", "mmf_cash_fund_route"}
    for category in ("bills", "notes", "bonds", "tips", "frn"):
        assert sum(float(row[f"{category}_route_share"]) for row in route_rows) == pytest.approx(1.0)


def test_foreign_retreat_replacement_notes_go_to_domestic_nonbanks():
    preferences, routes = holder_profile_from_opening_weights(WEIGHTS)
    notes = {holder: preferences[holder]["notes_pct"] for holder in HOLDER_TYPES}
    replacement = shifted_holder_rows(
        {"notes": notes}, holder="Foreign", multiplier=0.5, absorbers={"Private": 1.0},
    )[0]["shares"]
    split = _private_subbucket_split({PRIVATE_SUBBUCKET_SHARES_KEY: routes}, "notes")
    released = notes["Foreign"] - replacement["Foreign"]
    assert released > 0.0
    assert replacement["Private"] - notes["Private"] == pytest.approx(released)
    assert replacement["Private"] * split["mmf_cash_fund_route"] == 0.0
    assert replacement["Private"] * split["domestic_nonbank_deposit_funded"] == replacement["Private"]


def test_engine_issues_bills_and_frns_to_money_funds_and_routes_long_coupons_elsewhere(tmp_path):
    from test_cbo_engine_integration import (
        _build_temp_forecast_inputs,
        _minimal_engine_params,
        _write_tips_forward_paths,
    )

    paths = _build_temp_forecast_inputs(tmp_path, pre_issuance_controlled_debt_bil=0.0)
    paths.update(_write_tips_forward_paths(tmp_path))
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    preferences, routes = holder_profile_from_opening_weights(WEIGHTS)
    params["sector_preferences"] = {**preferences, PRIVATE_SUBBUCKET_SHARES_KEY: routes}
    profile = params["treasury_issuance_profile"]
    for category in ("bills", "notes", "bonds"):
        profile[category]["target_percentage_of_remainder"] = 1.0 / 3.0
    profile["TIPS"] = {"target_percentage": 0.2, "maturities": [10.0], "maturity_distribution": [1.0]}
    profile["FRN"] = {"target_percentage": 0.2, "maturities": [2.0], "maturity_distribution": [1.0]}
    _, portfolio = run_simulation(params, "2026-09-20", "2026-09-30", freq="10D", scenario_name="baseline")

    nominal = portfolio.loc[portfolio["SecurityType"].eq("Fixed")]
    assert set(nominal["MaturityCategory"]) == {"bills", "notes", "bonds"}
    assert set(portfolio["SecurityType"]) == {"Fixed", "FRN", "TIPS"}
    money_funds = portfolio.loc[portfolio["HolderSubBucket"].eq("mmf_cash_fund_route")]
    assert set(money_funds["SecurityType"]) == {"Fixed", "FRN"}
    assert set(money_funds.loc[money_funds["SecurityType"].eq("Fixed"), "MaturityCategory"]) == {"bills"}


@pytest.mark.parametrize(
    "weights",
    [
        {"bills": WEIGHTS["bills"]},
        {**WEIGHTS, "bills": {**WEIGHTS["bills"], "Banks": 0.30}},
        {**WEIGHTS, "nonbills": {"Banks": 1.0}},
        # The calibrated FRN share is required and must be a share.
        {"bills": WEIGHTS["bills"], "nonbills": WEIGHTS["nonbills"]},
        {**WEIGHTS, "frn_mmf_auction_share": 1.0},
        {**WEIGHTS, "frn_mmf_auction_share": "0.7"},
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
