from __future__ import annotations

import pandas as pd

from historical_replay_observations import (
    build_historical_replay_observations,
    build_holder_basis_bridge,
)


def test_observation_registry_preserves_z1_lm_fl_valuation_basis():
    sector_levels = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "sector": "rest_of_world",
                "broad_holder_class": "foreign_international",
                "tdcsim_holder": "Foreign",
                "measure": "level",
                "value": 100.0,
                "z1_series": "LM263061105.Q",
                "z1_code": "263061105",
                "source_file": "z1_csv_files.zip::csv/l210.csv",
                "source_status": "z1_csv_files.zip::csv/l210.csv",
            },
            {
                "quarter": "2025Q1",
                "sector": "money_market_funds",
                "broad_holder_class": "money_market_cash",
                "tdcsim_holder": "Private",
                "measure": "level",
                "value": 40.0,
                "z1_series": "FL633061105.Q",
                "z1_code": "633061105",
                "source_file": "z1_csv_files.zip::csv/l210.csv",
                "source_status": "z1_csv_files.zip::csv/l210.csv",
            },
            {
                "quarter": "2025Q1",
                "sector": "treasury_flows",
                "broad_holder_class": "banks",
                "tdcsim_holder": "Banks",
                "measure": "transaction",
                "value": 5.0,
                "z1_series": "FA763061100.Q",
                "z1_code": "763061100",
                "source_file": "z1_csv_files.zip::csv/f.csv",
                "source_status": "z1_csv_files.zip::csv/f.csv",
            },
        ]
    )

    observations = build_historical_replay_observations(
        sector_levels,
        pd.DataFrame(),
        ffiec_path=None,
        ncua_path=None,
        tier2_constraints_path=None,
    )
    basis_by_sector = observations.set_index("source_row_key")["valuation_basis"].to_dict()

    assert basis_by_sector["2025Q1|rest_of_world|level|0"] == "z1_market_value_level"
    assert basis_by_sector["2025Q1|money_market_funds|level|1"] == "z1_book_or_par_level"
    assert basis_by_sector["2025Q1|treasury_flows|transaction|2"] == "z1_transaction_flow"

    bridge = build_holder_basis_bridge(observations)
    assert set(bridge["valuation_basis"]) >= {
        "z1_market_value_level",
        "z1_book_or_par_level",
        "z1_transaction_flow",
    }


def test_observation_registry_includes_ncua_and_tier2_interest_constraints(tmp_path):
    ncua_path = tmp_path / "ncua.csv"
    tier2_path = tmp_path / "tier2.csv"
    pd.DataFrame(
        [
            {
                "date": "2025-03-31",
                "total_treasuries_amortized_cost": 6_000_000_000.0,
                "total_treasuries_fair_value": 5_900_000_000.0,
                "investment_bucket_le_1y": 1_200_000_000.0,
                "fallback_split_basis": "fixture_ncua_basis",
            }
        ]
    ).to_csv(ncua_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-03-31",
                "sector_key": "credit_unions_marketable_proxy",
                "source_family": "NCUA_CALL_REPORT",
                "component_key": "coupon_accrual",
                "constraint_status": "usable_constraint",
                "level_mil": 6000.0,
                "fair_value_mil": 5900.0,
                "constraint_basis": "fixture_tier2_basis",
            }
        ]
    ).to_csv(tier2_path, index=False)

    observations = build_historical_replay_observations(
        pd.DataFrame(),
        pd.DataFrame(),
        ffiec_path=None,
        ncua_path=ncua_path,
        tier2_constraints_path=tier2_path,
        start_quarter="2025Q1",
        end_quarter="2025Q1",
    )

    assert set(observations["scope"]) >= {
        "ncua_credit_union_treasury_constraint",
        "ncua_credit_union_maturity_proxy",
        "tier2_interest_source_constraint",
        "mmf_treasury_total",
        "mmf_treasury_bill_component",
        "mmf_treasury_nonbill_component",
    }
    ncua_total = observations[
        observations["measure"] == "total_treasuries_amortized_cost"
    ].iloc[0]
    assert ncua_total["reported_value_mil"] == 6000.0
    assert ncua_total["valuation_basis"] == "ncua_amortized_cost"
    assert ncua_total["holder"] == "Banks"
    tier2 = observations[observations["scope"] == "tier2_interest_source_constraint"].iloc[0]
    assert tier2["holder"] == "Banks"
    assert tier2["valuation_basis"] == "fixture_tier2_basis"
