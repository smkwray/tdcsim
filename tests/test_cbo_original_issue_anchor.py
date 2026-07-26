from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cbo_policy_bundle import (
    build_fiscal_incidence_policy_rows,
    build_net_interest_bridge_rows,
)
from forecast_bundle_builders import (
    build_cash_reconciliation_residual_rows,
    build_debt_stock_path_rows,
    build_operating_cash_path_rows,
    build_primary_deficit_path_rows,
)
from sim_engine import run_simulation
from test_cbo_engine_integration import _minimal_engine_params, _single_period, _write_csv


AUCTION_DATE = "2025-05-08"
ISSUE_DATE = "2025-05-15"
MATURITY_DATE = pd.Timestamp("2055-05-15")
PUBLISHED_COUPON = 0.0475
PUBLISHED_HIGH_YIELD = 0.04819
PUBLISHED_PRICE_PER_100 = 98.911312
PUBLISHED_PRICE_RATIO = 0.98911312


def _auction_forecast_inputs(tmp_path: Path) -> dict[str, Path]:
    periods = _single_period(AUCTION_DATE, ISSUE_DATE)
    return {
        "debt_stock_path_file": _write_csv(
            tmp_path / "tdcsim_debt_stock_path.csv",
            build_debt_stock_path_rows(
                scenario_id="baseline",
                periods=periods,
                opening_state_date=AUCTION_DATE,
                opening_cbo_federal_debt_held_public_bil=125.0,
                cbo_fy_end_public_debt_targets_bil={2025: 225.0},
                public_nonmarketable_treasury_bil=100.0,
                non_treasury_and_definition_residual_bil=25.0,
                observation_date=AUCTION_DATE,
                available_date=AUCTION_DATE,
            ),
        ),
        "primary_deficit_path_file": _write_csv(
            tmp_path / "tdcsim_primary_deficit_path.csv",
            build_primary_deficit_path_rows(
                scenario_id="baseline",
                periods=periods,
                primary_deficit_by_fiscal_year_bil={2025: 0.0},
            ),
        ),
        "operating_cash_path_file": _write_csv(
            tmp_path / "tdcsim_operating_cash_path.csv",
            build_operating_cash_path_rows(
                scenario_id="baseline",
                periods=periods,
                base_date=AUCTION_DATE,
                base_balance_bil=800.0,
                inflation_scalar=0.0,
                observation_date=AUCTION_DATE,
                available_date=AUCTION_DATE,
            ),
        ),
        "cash_reconciliation_residual_file": _write_csv(
            tmp_path / "tdcsim_cash_reconciliation_residual.csv",
            build_cash_reconciliation_residual_rows(
                scenario_id="baseline",
                periods=periods,
                cash_reconciliation_residual_bil=0.0,
            ),
        ),
        "fiscal_incidence_policy_file": _write_csv(
            tmp_path / "tdcsim_fiscal_incidence_policy.csv",
            build_fiscal_incidence_policy_rows(
                scenario_id="baseline",
                signed_net_primary_flow_bil=0.0,
            ),
        ),
        "net_interest_bridge_file": _write_csv(
            tmp_path / "tdcsim_net_interest_bridge.csv",
            build_net_interest_bridge_rows(
                scenario_id="baseline",
                fiscal_year=2025,
                source_vintage="fixture_2025_05_08_30y_auction",
                cbo_reported_net_interest_bil=0.0,
                components=[{"component_key": "fixed_coupon_accrual", "amount_bil": 0.0}],
            ),
        ),
    }


def test_cbo_live_issuance_matches_published_2025_thirty_year_auction(tmp_path: Path) -> None:
    """The ordinary CBO issuance branch must carry Treasury's published terms into cash output."""

    params = _minimal_engine_params(
        _auction_forecast_inputs(tmp_path),
        opening_controlled_debt_bil=0.0,
    )
    params["data_vintage"] = {
        "actuals_available_as_of": ISSUE_DATE,
        "allow_lookahead": False,
    }
    params["yield_curve"]["rates"] = [PUBLISHED_HIGH_YIELD] * len(params["yield_curve"]["years"])
    params["treasury_issuance_profile"]["bills"]["target_percentage_of_remainder"] = 0.0
    params["treasury_issuance_profile"]["notes"]["target_percentage_of_remainder"] = 0.0
    params["treasury_issuance_profile"]["bonds"].update(
        {
            "target_percentage_of_remainder": 1.0,
            "maturities": [30.0],
            "maturity_distribution": [1.0],
        }
    )
    params["treasury_issuance_profile"]["remainder_maturity_years"] = 30.0

    results, portfolio = run_simulation(
        params,
        AUCTION_DATE,
        ISSUE_DATE,
        freq="7D",
        scenario_name="baseline",
    )

    issued_rows = portfolio.loc[portfolio["IssueDate"] == pd.Timestamp(ISSUE_DATE)]
    assert len(issued_rows) == 1
    issued = issued_rows.iloc[0]
    period = results.loc[pd.Timestamp(ISSUE_DATE)]
    issuance_flows = results.attrs["handoff_tables"]["tdcsim_period_issuance_flows"]
    assert len(issuance_flows) == 1
    issuance = issuance_flows[0]

    assert issuance["period_start"] == AUCTION_DATE
    assert issuance["period_end"] == ISSUE_DATE
    assert issued["IssueDate"] == pd.Timestamp(ISSUE_DATE)
    assert issued["DatedDate"] == pd.Timestamp(ISSUE_DATE)
    assert issued["MaturityDate"] == MATURITY_DATE

    assert issued["CouponRate"] == pytest.approx(PUBLISHED_COUPON, abs=1e-12)
    assert issued["IssueYieldAtIssue"] == pytest.approx(PUBLISHED_HIGH_YIELD, abs=1e-12)
    assert issuance["coupon_rate_decimal"] == pytest.approx(PUBLISHED_COUPON, abs=1e-12)
    assert issuance["issue_yield_decimal"] == pytest.approx(PUBLISHED_HIGH_YIELD, abs=1e-12)

    price_per_100 = float(issued["IssueProceeds"]) / float(issued["FaceValue"]) * 100.0
    assert round(price_per_100, 6) == PUBLISHED_PRICE_PER_100
    assert round(float(issued["IssuePriceRatio"]), 8) == PUBLISHED_PRICE_RATIO
    assert float(issued["IssueProceeds"]) / float(issued["FaceValue"]) == pytest.approx(
        PUBLISHED_PRICE_RATIO,
        abs=1e-12,
    )
    assert issued["IssuePriceRatio"] != pytest.approx(1.0, abs=1e-12)

    assert period["NewDebtIssued"] == pytest.approx(issued["FaceValue"], abs=1e-12)
    assert period["CBORequiredFaceIssuance"] == pytest.approx(issued["FaceValue"], abs=1e-12)
    assert period["AuctionProceeds"] == pytest.approx(issued["IssueProceeds"], abs=1e-12)
    assert issuance["face_issued_bil"] == pytest.approx(issued["FaceValue"], abs=1e-12)
    assert issuance["cash_proceeds_bil"] == pytest.approx(issued["IssueProceeds"], abs=1e-12)
