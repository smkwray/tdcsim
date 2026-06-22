from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_soma import build_soma_fixed_allocations, load_soma_treasury_holdings


def test_load_soma_treasury_holdings_selects_latest_observation_per_quarter(tmp_path):
    path = tmp_path / "soma.csv"
    pd.DataFrame(
        [
            {
                "As Of Date": "2022-02-28",
                "CUSIP": "'912810AA1'",
                "Security Type": "Note",
                "Maturity Date": "2032-02-15",
                "Current Face Value": "1,000,000",
                "Par Value": "1,000,000",
                "Inflation Compensation": "0",
            },
            {
                "As Of Date": "2022-03-31",
                "CUSIP": "'912810AA1'",
                "Security Type": "Note",
                "Maturity Date": "2032-02-15",
                "Current Face Value": "2,500,000",
                "Par Value": "2,500,000",
                "Inflation Compensation": "0",
            },
        ]
    ).to_csv(path, index=False)

    holdings = load_soma_treasury_holdings(path, start_quarter="2022Q1", end_quarter="2022Q1")

    assert len(holdings) == 1
    assert holdings.iloc[0]["quarter"] == "2022Q1"
    assert holdings.iloc[0]["as_of_date"] == pd.Timestamp("2022-03-31")
    assert holdings.iloc[0]["cusip"] == "912810AA1"
    assert holdings.iloc[0]["observed_face_value_mil"] == pytest.approx(2.5)


def test_build_soma_fixed_allocations_splits_duplicate_cusip_maturity_cohorts():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2022Q1",
                "cohort_id": "912810AA1|2020-02-15|2032-02-15",
                "cusip": "912810AA1",
                "maturity_date": "2032-02-15",
                "outstanding": 60.0,
            },
            {
                "quarter": "2022Q1",
                "cohort_id": "912810AA1|2021-02-15|2032-02-15",
                "cusip": "912810AA1",
                "maturity_date": "2032-02-15",
                "outstanding": 40.0,
            },
            {
                "quarter": "2022Q1",
                "cohort_id": "912828BB1|2021-02-15|2031-02-15",
                "cusip": "912828BB1",
                "maturity_date": "2031-02-15",
                "outstanding": 50.0,
            },
        ]
    )
    soma = pd.DataFrame(
        [
            {
                "quarter": "2022Q1",
                "as_of_date": pd.Timestamp("2022-03-31"),
                "cusip": "912810AA1",
                "maturity_date": "2032-02-15",
                "security_type": "Note",
                "observed_face_value_mil": 25.0,
                "observed_par_value_mil": 25.0,
                "inflation_compensation_mil": 0.0,
            }
        ]
    )

    allocations, diagnostics = build_soma_fixed_allocations(cohorts, soma)

    by_cohort = allocations.set_index("cohort_id")["allocated_outstanding"].sort_index()
    assert by_cohort["912810AA1|2020-02-15|2032-02-15"] == pytest.approx(15.0)
    assert by_cohort["912810AA1|2021-02-15|2032-02-15"] == pytest.approx(10.0)
    assert set(allocations["sector"]) == {"monetary_authority"}
    assert set(allocations["fixed_allocation_policy"]) == {"observed_soma_exact_block"}

    diag = diagnostics.iloc[0]
    assert diag["soma_face_mil"] == pytest.approx(25.0)
    assert diag["matched_face_mil"] == pytest.approx(25.0)
    assert diag["status"] == "matched"
