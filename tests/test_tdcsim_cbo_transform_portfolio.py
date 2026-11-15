import pytest

from tdcsim_cbo.transforms.portfolio import (
    PortfolioTransformError,
    apply_fed_holdings_override,
    compile_holder_preference_events,
    compile_issuance_mix_override,
    reject_non_fed_stock_targets,
    validate_holder_preferences,
)


def _issuance_override() -> dict:
    return {
        "mode": "replace_shares",
        "tips_share": 0.08,
        "frn_share": 0.05,
        "fixed_remainder_shares": {"bills": 0.3, "notes": 0.5, "bonds": 0.2},
        "maturity_distributions": {
            "bills": [{"maturity_years": 0.5, "share": 1.0}],
            "notes": [{"maturity_years": 5.0, "share": 1.0}],
            "bonds": [{"maturity_years": 20.0, "share": 1.0}],
            "tips": [{"maturity_years": 10.0, "share": 1.0}],
            "frn": [{"maturity_years": 2.0, "share": 1.0}],
        },
        "negative_issuance_action": "retire_shortest_public_marketable",
    }


def test_issuance_mix_compiles_security_shares_and_wam() -> None:
    compiled = compile_issuance_mix_override(_issuance_override())

    assert compiled.security_shares["bills"] == pytest.approx(0.261)
    assert compiled.security_shares["notes"] == pytest.approx(0.435)
    assert compiled.security_shares["bonds"] == pytest.approx(0.174)
    assert compiled.security_shares["tips"] == pytest.approx(0.08)
    assert compiled.security_shares["frn"] == pytest.approx(0.05)
    assert compiled.weighted_average_maturity_years == pytest.approx(6.6855)


def test_issuance_mix_rejects_invalid_share_sums() -> None:
    override = _issuance_override()
    override["fixed_remainder_shares"] = {"bills": 0.4, "notes": 0.5, "bonds": 0.2}

    with pytest.raises(PortfolioTransformError, match="sum to 1.0"):
        compile_issuance_mix_override(override)


def test_issuance_mix_rejects_false_frn_maturity_knob() -> None:
    override = _issuance_override()
    override["maturity_distributions"]["frn"] = [{"maturity_years": 5.0, "share": 1.0}]

    with pytest.raises(PortfolioTransformError, match="frn maturity distribution"):
        compile_issuance_mix_override(override)


def test_issuance_mix_rejects_mislabeled_fixed_maturities() -> None:
    override = _issuance_override()
    override["maturity_distributions"]["notes"] = [{"maturity_years": 0.75, "share": 1.0}]

    with pytest.raises(PortfolioTransformError, match="notes maturity_years"):
        compile_issuance_mix_override(override)


def test_fed_holdings_scale_is_bounded_by_marketable_debt() -> None:
    rows = [{"period_end": "2027-01-01", "source_fiscal_year": 2027, "holder_type": "CB", "cbo_fed_holdings_target_bil": 1000.0}]
    debt = [{"period_end": "2027-01-01", "marketable_treasury_public_target_bil": 1200.0}]

    transformed = apply_fed_holdings_override(rows, {"mode": "scale_path", "scale": 1.1}, marketable_debt_rows=debt)

    assert transformed[0]["cbo_fed_holdings_target_bil"] == pytest.approx(1100.0)
    assert transformed[0]["claim_boundary"] == "fed_holdings_path_guides_holder_allocation_not_total_issuance"


def test_fed_holdings_rejects_exceeding_marketable_debt_bound() -> None:
    rows = [{"period_end": "2027-01-01", "source_fiscal_year": 2027, "holder_type": "CB", "cbo_fed_holdings_target_bil": 1000.0}]
    debt = [{"period_end": "2027-01-01", "marketable_treasury_public_target_bil": 1200.0}]

    with pytest.raises(PortfolioTransformError, match="must not exceed"):
        apply_fed_holdings_override(rows, {"mode": "scale_path", "scale": 1.3}, marketable_debt_rows=debt)


def test_fed_holdings_absolute_path_file_accepts_compiler_replacement_rows() -> None:
    replacement = [{"period_end": "2027-01-01", "holder_type": "CB", "cbo_fed_holdings_target_bil": 1000.0}]
    debt = [{"period_end": "2027-01-01", "marketable_treasury_public_target_bil": 1200.0}]

    transformed = apply_fed_holdings_override(
        [],
        {"mode": "absolute_path_file"},
        marketable_debt_rows=debt,
        replacement_rows=replacement,
    )

    assert transformed[0]["cbo_fed_holdings_target_bil"] == pytest.approx(1000.0)
    assert transformed[0]["scenario_transform"] == "absolute_path_file"


def test_holder_preferences_validate_unit_sums_and_cb_zero_with_fed_target() -> None:
    override = {
        "mode": "static_shares",
        "rows": [
            {
                "security_type": "bills",
                "shares": {
                    "Banks": 0.2,
                    "CB": 0.0,
                    "Foreign": 0.3,
                    "Private": 0.5,
                    "TrustFunds": 0.0,
                    "FedInternal": 0.0,
                },
            },
            {
                "security_type": "nonmarketable",
                "shares": {
                    "Banks": 0.0,
                    "CB": 0.0,
                    "Foreign": 0.0,
                    "Private": 0.0,
                    "TrustFunds": 0.0,
                    "FedInternal": 0.0,
                },
            },
        ],
    }

    rows = validate_holder_preferences(override, fed_stock_target_active=True)

    assert rows[0]["shares"]["Foreign"] == pytest.approx(0.3)
    assert rows[1]["security_type"] == "nonmarketable"


def test_holder_preferences_reject_cb_auction_share_when_fed_stock_target_active() -> None:
    override = {
        "mode": "static_shares",
        "rows": [
            {
                "security_type": "notes",
                "shares": {
                    "Banks": 0.2,
                    "CB": 0.1,
                    "Foreign": 0.2,
                    "Private": 0.5,
                    "TrustFunds": 0.0,
                    "FedInternal": 0.0,
                },
            }
        ],
    }

    with pytest.raises(PortfolioTransformError, match="CB auction share"):
        validate_holder_preferences(override, fed_stock_target_active=True)


def test_dated_holder_preferences_compile_to_sector_preference_events() -> None:
    override = {
        "mode": "dated_static_shares",
        "rows": [
            {
                "effective_date": "2030-01-15",
                "security_type": "notes",
                "shares": {
                    "Banks": 0.4,
                    "CB": 0.0,
                    "Foreign": 0.2,
                    "Private": 0.4,
                    "TrustFunds": 0.0,
                    "FedInternal": 0.0,
                },
            }
        ],
    }

    rows = validate_holder_preferences(override, fed_stock_target_active=True)
    events = compile_holder_preference_events(override, fed_stock_target_active=True)

    assert rows[0]["effective_date"] == "2030-01-15"
    assert rows[0]["runtime_role"] == "runtime_event"
    assert events == [
        {
            "id": "cbo_holder_preferences_2030-01-15",
            "date": "2030-01-15",
            "actions": [
                {
                    "parameter_path": "sector_preferences.Banks.notes_pct",
                    "new_value": 0.4,
                    "source_role": "scenario_assumption",
                    "runtime_role": "runtime_event",
                    "claim_boundary": "holder preference profile not exact holder ownership",
                },
                {
                    "parameter_path": "sector_preferences.CB.notes_pct",
                    "new_value": 0.0,
                    "source_role": "scenario_assumption",
                    "runtime_role": "runtime_event",
                    "claim_boundary": "holder preference profile not exact holder ownership",
                },
                {
                    "parameter_path": "sector_preferences.Foreign.notes_pct",
                    "new_value": 0.2,
                    "source_role": "scenario_assumption",
                    "runtime_role": "runtime_event",
                    "claim_boundary": "holder preference profile not exact holder ownership",
                },
                {
                    "parameter_path": "sector_preferences.Private.notes_pct",
                    "new_value": 0.4,
                    "source_role": "scenario_assumption",
                    "runtime_role": "runtime_event",
                    "claim_boundary": "holder preference profile not exact holder ownership",
                },
                {
                    "parameter_path": "sector_preferences.TrustFunds.notes_pct",
                    "new_value": 0.0,
                    "source_role": "scenario_assumption",
                    "runtime_role": "runtime_event",
                    "claim_boundary": "holder preference profile not exact holder ownership",
                },
                {
                    "parameter_path": "sector_preferences.FedInternal.notes_pct",
                    "new_value": 0.0,
                    "source_role": "scenario_assumption",
                    "runtime_role": "runtime_event",
                    "claim_boundary": "holder preference profile not exact holder ownership",
                },
            ],
        }
    ]


def test_dated_holder_preferences_reject_nonmarketable_rows() -> None:
    override = {
        "mode": "dated_static_shares",
        "rows": [
            {
                "effective_date": "2030-01-15",
                "security_type": "nonmarketable",
                "shares": {"Banks": 0.0, "CB": 0.0, "Foreign": 0.0, "Private": 0.0, "TrustFunds": 0.0, "FedInternal": 0.0},
            }
        ],
    }

    with pytest.raises(PortfolioTransformError, match="marketable security types only"):
        validate_holder_preferences(override)


def test_non_fed_stock_targets_are_rejected() -> None:
    with pytest.raises(PortfolioTransformError, match="non-Fed stock targets"):
        reject_non_fed_stock_targets({"Banks": 100.0})


def _bill_row_with_routes(private_routes: object, *, private: float = 0.5) -> dict:
    return {
        "mode": "dated_static_shares",
        "rows": [
            {
                "effective_date": "2027-01-15",
                "security_type": "bills",
                "shares": {
                    "Banks": 0.3,
                    "CB": 0.0,
                    "Foreign": 0.7 - private,
                    "Private": private,
                    "TrustFunds": 0.0,
                    "FedInternal": 0.0,
                },
                "private_routes": private_routes,
            }
        ],
    }


def test_private_routes_compile_into_route_share_events_on_the_row_date() -> None:
    # The money-fund scenario is only distinct from "Private buys more" if the extra
    # Private bills land in the money-fund route; the event must set both route shares
    # for that security type on the row's own date, after the holder shares.
    events = compile_holder_preference_events(
        _bill_row_with_routes({"domestic_nonbank_deposit_funded": 0.2, "mmf_cash_fund_route": 0.8})
    )

    assert [event["date"] for event in events] == ["2027-01-15"]
    route_actions = {
        action["parameter_path"]: action["new_value"]
        for action in events[0]["actions"]
        if "__private_subbucket_shares__" in action["parameter_path"]
    }
    assert route_actions == {
        "sector_preferences.__private_subbucket_shares__.bills.domestic_nonbank_deposit_funded": 0.2,
        "sector_preferences.__private_subbucket_shares__.bills.mmf_cash_fund_route": 0.8,
    }
    assert events[0]["actions"][-1]["parameter_path"].endswith("mmf_cash_fund_route")


def test_rows_without_private_routes_leave_route_shares_untouched() -> None:
    override = _bill_row_with_routes(None)
    del override["rows"][0]["private_routes"]

    actions = compile_holder_preference_events(override)[0]["actions"]

    assert not [action for action in actions if "__private_subbucket_shares__" in action["parameter_path"]]


@pytest.mark.parametrize(
    ("routes", "private", "match"),
    [
        ({"domestic_nonbank_deposit_funded": 0.3, "mmf_cash_fund_route": 0.6}, 0.5, "must sum to 1.0"),
        ({"domestic_nonbank_deposit_funded": 0.5, "money_funds": 0.5}, 0.5, "must have exactly"),
        ({"mmf_cash_fund_route": 1.0}, 0.5, "must have exactly"),
        ({"domestic_nonbank_deposit_funded": -0.5, "mmf_cash_fund_route": 1.5}, 0.5, r"in \[0, 1\]"),
        ({"domestic_nonbank_deposit_funded": "0.5", "mmf_cash_fund_route": 0.5}, 0.5, r"in \[0, 1\]"),
        ({"domestic_nonbank_deposit_funded": 0.0, "mmf_cash_fund_route": 1.0}, 0.0, "positive Private share"),
        ([0.5, 0.5], 0.5, "must have exactly"),
    ],
)
def test_invalid_private_routes_raise(routes: object, private: float, match: str) -> None:
    with pytest.raises(PortfolioTransformError, match=match):
        compile_holder_preference_events(_bill_row_with_routes(routes, private=private))


def test_private_routes_are_rejected_on_static_share_rows() -> None:
    override = _bill_row_with_routes({"domestic_nonbank_deposit_funded": 0.0, "mmf_cash_fund_route": 1.0})
    override["mode"] = "static_shares"
    del override["rows"][0]["effective_date"]

    with pytest.raises(PortfolioTransformError, match="only in dated_static_shares"):
        validate_holder_preferences(override)
