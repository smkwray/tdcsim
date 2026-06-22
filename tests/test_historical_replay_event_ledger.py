from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_event_ledger import (
    build_event_rollforward,
    build_historical_replay_event_ledger,
)


def test_event_ledger_uses_first_differences_and_startup_opening_stock():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "old_note",
                "cusip": "912345AA1",
                "security_type": "note",
                "issue_date": "2024-01-15",
                "maturity_date": "2029-01-15",
                "outstanding": 100.0,
                "issued_amt": 100.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
            },
            {
                "quarter": "2025Q2",
                "cohort_id": "old_note",
                "cusip": "912345AA1",
                "security_type": "note",
                "issue_date": "2024-01-15",
                "maturity_date": "2029-01-15",
                "outstanding": 80.0,
                "issued_amt": 100.0,
                "redeemed_amt": -20.0,
                "inflation_adj_amt": 99.0,
            },
            {
                "quarter": "2025Q2",
                "cohort_id": "new_note",
                "cusip": "912345BB2",
                "security_type": "note",
                "issue_date": "2025-05-15",
                "maturity_date": "2030-05-15",
                "outstanding": 50.0,
                "issued_amt": 50.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
            },
        ]
    )

    source_ledger = build_historical_replay_event_ledger(
        cohorts,
        pd.DataFrame(),
        start_quarter="2025Q1",
        end_quarter="2025Q2",
    )
    event_ledger, rollforward, unexplained = build_event_rollforward(
        cohorts,
        source_ledger,
        start_quarter="2025Q1",
    )

    old_start = rollforward[
        (rollforward["quarter"] == "2025Q1") & (rollforward["cohort_id"] == "old_note")
    ].iloc[0]
    assert old_start["opening_balance_mil"] == pytest.approx(100.0)
    assert old_start["source_issue_mil"] == pytest.approx(0.0)
    assert old_start["status"] == "closed_by_source_events"

    redemption = event_ledger[event_ledger["event_type"] == "principal_redemption"].iloc[0]
    assert redemption["cohort_id"] == "old_note"
    assert redemption["par_delta_mil"] == pytest.approx(-20.0)
    assert redemption["cash_amount_mil"] == pytest.approx(20.0)
    assert redemption["derivation"] == "first_difference_of_mspd_redeemed_amt"

    new_issue = event_ledger[event_ledger["event_type"] == "mspd_issuance"].iloc[0]
    assert new_issue["cohort_id"] == "new_note"
    assert new_issue["par_delta_mil"] == pytest.approx(50.0)
    assert new_issue["cash_amount_mil"] == pytest.approx(-50.0)

    q2_old = rollforward[
        (rollforward["quarter"] == "2025Q2") & (rollforward["cohort_id"] == "old_note")
    ].iloc[0]
    assert q2_old["opening_balance_mil"] == pytest.approx(100.0)
    assert q2_old["ending_balance_mil"] == pytest.approx(80.0)
    assert q2_old["source_redemption_mil"] == pytest.approx(-20.0)
    assert q2_old["unexplained_cohort_change_mil"] == pytest.approx(0.0)
    assert q2_old["status"] == "closed_by_source_events"

    assert event_ledger["event_type"].tolist().count("tips_principal_indexation") == 0
    assert unexplained.empty


def test_event_ledger_only_treats_tips_inflation_adjustment_as_indexation():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "tips_a",
                "cusip": "912345TI1",
                "security_type": "tips",
                "issue_date": "2024-01-15",
                "maturity_date": "2034-01-15",
                "outstanding": 100.0,
                "issued_amt": 95.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 5.0,
            },
            {
                "quarter": "2025Q2",
                "cohort_id": "tips_a",
                "cusip": "912345TI1",
                "security_type": "tips",
                "issue_date": "2024-01-15",
                "maturity_date": "2034-01-15",
                "outstanding": 103.0,
                "issued_amt": 95.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 8.0,
            },
            {
                "quarter": "2025Q2",
                "cohort_id": "note_a",
                "cusip": "912345NO1",
                "security_type": "note",
                "issue_date": "2024-01-15",
                "maturity_date": "2029-01-15",
                "outstanding": 100.0,
                "issued_amt": 100.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 15.0,
            },
        ]
    )

    source_ledger = build_historical_replay_event_ledger(
        cohorts,
        pd.DataFrame(),
        start_quarter="2025Q1",
        end_quarter="2025Q2",
    )

    indexation = source_ledger[source_ledger["event_type"] == "tips_principal_indexation"]
    assert indexation["cohort_id"].tolist() == ["tips_a"]
    assert indexation.iloc[0]["current_principal_delta_mil"] == pytest.approx(3.0)
    assert indexation.iloc[0]["par_delta_mil"] == pytest.approx(0.0)


def test_event_rollforward_treats_reopening_decomposition_as_source_reclassification():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q3",
                "cohort_id": "cusip|old|maturity",
                "cusip": "912345AA1",
                "security_type": "note",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "outstanding": 100.0,
                "issued_amt": 100.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
                "source_status": "mspd_cohort_row",
            },
            {
                "quarter": "2025Q4",
                "cohort_id": "cusip|old|maturity",
                "cusip": "912345AA1",
                "security_type": "note",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "outstanding": 40.0,
                "issued_amt": 40.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
                "source_status": "mspd_cohort_row;mspd_reopening_outstanding_decomposed",
            },
            {
                "quarter": "2025Q4",
                "cohort_id": "cusip|reopening|maturity",
                "cusip": "912345AA1",
                "security_type": "note",
                "issue_date": "2020-02-15",
                "maturity_date": "2030-01-15",
                "outstanding": 60.0,
                "issued_amt": 60.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
                "source_status": "mspd_cohort_row;mspd_reopening_outstanding_decomposed",
            },
        ]
    )

    source_ledger = build_historical_replay_event_ledger(
        cohorts,
        pd.DataFrame(),
        start_quarter="2025Q3",
        end_quarter="2025Q4",
    )
    _, rollforward, unexplained = build_event_rollforward(
        cohorts,
        source_ledger,
        start_quarter="2025Q3",
    )

    reclass = source_ledger[source_ledger["event_type"] == "mspd_source_reclassification"]
    assert reclass["par_delta_mil"].sum() == pytest.approx(0.0)
    assert sorted(reclass["par_delta_mil"].tolist()) == pytest.approx([-60.0, 60.0])
    q4 = rollforward[rollforward["quarter"] == "2025Q4"]
    assert set(q4["status"]) == {"closed_by_source_events"}
    assert q4["unexplained_cohort_change_mil"].abs().sum() == pytest.approx(0.0)
    assert q4["source_reclassification_mil"].abs().sum() == pytest.approx(120.0)
    assert unexplained.empty


def test_reclassification_does_not_duplicate_decomposed_source_flows():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "tips_reopening",
                "cusip": "912345TI1",
                "security_type": "tips",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "outstanding": 100.0,
                "issued_amt": 90.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 10.0,
                "source_status": "mspd_cohort_row;mspd_reopening_outstanding_decomposed",
            },
            {
                "quarter": "2025Q2",
                "cohort_id": "tips_reopening",
                "cusip": "912345TI1",
                "security_type": "tips",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "outstanding": 103.0,
                "issued_amt": 90.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 13.0,
                "source_status": "mspd_cohort_row;mspd_reopening_outstanding_decomposed",
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "note_reopening",
                "cusip": "912345NO1",
                "security_type": "note",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "outstanding": 100.0,
                "issued_amt": 100.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
                "source_status": "mspd_cohort_row;mspd_reopening_outstanding_decomposed",
            },
            {
                "quarter": "2025Q2",
                "cohort_id": "note_reopening",
                "cusip": "912345NO1",
                "security_type": "note",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "outstanding": 120.0,
                "issued_amt": 120.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
                "source_status": "mspd_cohort_row;mspd_reopening_outstanding_decomposed",
            },
        ]
    )

    source_ledger = build_historical_replay_event_ledger(
        cohorts,
        pd.DataFrame(),
        start_quarter="2025Q1",
        end_quarter="2025Q2",
    )
    _, rollforward, unexplained = build_event_rollforward(
        cohorts,
        source_ledger,
        start_quarter="2025Q1",
        end_quarter="2025Q2",
    )

    assert source_ledger[source_ledger["event_type"] == "mspd_source_reclassification"].empty
    assert source_ledger[source_ledger["event_type"] == "tips_principal_indexation"][
        "current_principal_delta_mil"
    ].sum() == pytest.approx(3.0)
    assert source_ledger[source_ledger["event_type"] == "mspd_issuance"][
        "par_delta_mil"
    ].sum() == pytest.approx(20.0)
    assert rollforward["unexplained_cohort_change_mil"].abs().sum() == pytest.approx(0.0)
    assert unexplained.empty


def test_event_rollforward_adds_terminal_maturity_zero_row():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "maturing_note",
                "cusip": "912345AA1",
                "security_type": "note",
                "issue_date": "2020-04-15",
                "maturity_date": "2025-04-15",
                "outstanding": 100.0,
                "issued_amt": 100.0,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
                "source_status": "mspd_cohort_row",
            }
        ]
    )

    source_ledger = build_historical_replay_event_ledger(
        cohorts,
        pd.DataFrame(),
        start_quarter="2025Q1",
        end_quarter="2025Q2",
    )
    _, rollforward, unexplained = build_event_rollforward(
        cohorts,
        source_ledger,
        start_quarter="2025Q1",
        end_quarter="2025Q2",
    )

    terminal_event = source_ledger[source_ledger["event_type"] == "maturity_redemption"].iloc[0]
    assert terminal_event["quarter"] == "2025Q2"
    assert terminal_event["current_principal_delta_mil"] == pytest.approx(-100.0)
    q2 = rollforward[rollforward["quarter"] == "2025Q2"].iloc[0]
    assert q2["opening_balance_mil"] == pytest.approx(100.0)
    assert q2["ending_balance_mil"] == pytest.approx(0.0)
    assert q2["source_redemption_mil"] == pytest.approx(-100.0)
    assert q2["status"] == "closed_by_source_events"
    assert unexplained.empty
