from __future__ import annotations

import copy

import pandas as pd
import pytest

from sim_engine import run_simulation
from tdcsim_cbo.output import _route_stock_closure_handoff_tables
from test_cbo_engine_integration import (
    _minimal_cash_mode_params,
    _opening_controlled_portfolio,
    _opening_tips_portfolio,
)


def _route_stock_row(date: str, amount: float) -> dict[str, object]:
    return {
        "date": date,
        "route_holder_sector": "Private",
        "route_holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "Fixed",
        "maturity_bucket": "notes",
        "debt_scope": "controlled_public_marketable",
        "route_debt_held_bil": amount,
    }


def _balanced_route_inputs() -> dict[str, list[dict[str, object]]]:
    return {
        "tdcsim_tdc_principal_route_stocks": [
            _route_stock_row("2026-09-20", 100.0),
            _route_stock_row("2026-09-30", 130.0),
        ],
        "tdcsim_period_issuance_flows": [
            {
                "period_start": "2026-09-20",
                "period_end": "2026-09-30",
                "holder_sector": "Private",
                "holder_subsector": "domestic_nonbank_deposit_funded",
                "instrument_type": "Fixed",
                "maturity_bucket": "notes",
                "face_issued_bil": 50.0,
            }
        ],
        "tdcsim_period_principal_flows": [
            {
                "period_start": "2026-09-20",
                "period_end": "2026-09-30",
                "tdc_principal_recipient_sector": "Private",
                "tdc_principal_recipient_subsector": "domestic_nonbank_deposit_funded",
                "instrument_type": "Fixed",
                "maturity_bucket": "notes",
                "face_redeemed_bil": 20.0,
            }
        ],
    }


@pytest.mark.parametrize("mutation", ["delete_issuance", "double_redemption"])
def test_route_stock_closure_fails_when_a_named_leg_is_deleted_or_doubled(
    mutation: str,
) -> None:
    raw = _balanced_route_inputs()
    balanced = _route_stock_closure_handoff_tables(copy.deepcopy(raw))[
        "tdcsim_tdc_principal_route_stock_closure"
    ][0]
    assert balanced["route_stock_residual_or_indexation_bil"] == pytest.approx(0.0)
    assert balanced["closure_identity_error_bil"] == pytest.approx(0.0)

    if mutation == "delete_issuance":
        raw["tdcsim_period_issuance_flows"] = []
    else:
        raw["tdcsim_period_principal_flows"][0]["face_redeemed_bil"] = 40.0

    row = _route_stock_closure_handoff_tables(raw)[
        "tdcsim_tdc_principal_route_stock_closure"
    ][0]

    # A missing or doubled named leg must remain an exposed conservation error.
    # It cannot be copied into an unrestricted residual and subtracted back out.
    assert row["route_stock_residual_or_indexation_bil"] == pytest.approx(0.0)
    assert abs(row["closure_identity_error_bil"]) > 1e-9


def test_trust_fund_issuance_cannot_raise_tga_without_a_named_payer_leg() -> None:
    params = _minimal_cash_mode_params()
    params["initial_values"]["tga"] = 50.0
    params["tga_params"]["target_balance"] = 100.0
    for holder, preferences in params["sector_preferences"].items():
        preferences["bills_pct"] = 1.0 if holder == "TrustFunds" else 0.0

    results, _ = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    final = results.iloc[-1]
    journal = results.attrs["handoff_tables"].get("tdcsim_accounting_journal", [])
    payer_legs = [
        row
        for row in journal
        if row.get("counterparty") == "TrustFunds"
        and row.get("leg_type") in {"fund_asset_debit", "external_cash_payer"}
    ]

    assert final["DebtHeld_TrustFunds"] > 0.0
    assert final["TGA"] == pytest.approx(50.0) or sum(
        abs(float(row["amount_bil"])) for row in payer_legs
    ) == pytest.approx(final["AuctionProceeds"])


def _maturing_tips_params(*, reference_cpi: float = 100.0) -> dict:
    params = _minimal_cash_mode_params()
    params["initial_values"] = {
        "reserves": 1_000.0,
        "tdc_level": 0.0,
        "tga": 1_000.0,
    }
    params["tga_params"] = {"target_balance": 0.0, "floor": 0.0}
    tips = _opening_tips_portfolio(100.0, reference_cpi)
    tips.loc[:, "MaturityDate"] = pd.Timestamp("2026-09-25")
    tips.loc[:, "CouponRate"] = 0.0
    tips.loc[:, "InterestPaymentFrequency"] = pd.NA
    params["initial_bonds_df"] = tips
    params["tips_params"] = {
        "cpi_start_level": reference_cpi,
        "reference_cpi_start_level": reference_cpi,
        "cpi_annual_inflation": 1.0,
        "ref_cpi_lag_months": 0,
        "default_real_coupon_rate": 0.0,
    }
    params["financing_cost_options"] = {"include_tips_inflation_accretion": True}
    return params


def test_tips_maturity_uses_exact_event_cpi_at_daily_weekly_and_monthly_steps() -> None:
    payments: dict[str, float] = {}
    for label, frequency in {
        "daily": "D",
        "weekly": "7D",
        "monthly": "30D",
    }.items():
        results, _ = run_simulation(
            copy.deepcopy(_maturing_tips_params()),
            "2026-09-20",
            "2026-10-20",
            freq=frequency,
            scenario_name=f"tips_{label}",
        )
        principal = results.attrs["handoff_tables"]["tdcsim_period_principal_flows"]
        assert len(principal) == 1
        payments[label] = float(principal[0]["cash_paid_bil"])

    exact_event_payment = 100.0 * (2.0 ** (5.0 / 365.25))
    assert payments == pytest.approx(
        {
            "daily": exact_event_payment,
            "weekly": exact_event_payment,
            "monthly": exact_event_payment,
        }
    )


def test_final_nonmarketable_capitalization_precedes_same_day_maturity_once() -> None:
    params = _minimal_cash_mode_params()
    params["initial_values"] = {
        "reserves": 1_000.0,
        "tdc_level": 0.0,
        "tga": 1_000.0,
    }
    params["tga_params"] = {"target_balance": 0.0, "floor": 0.0}
    nonmarketable = _opening_controlled_portfolio(100.0)
    nonmarketable.loc[:, "SecurityType"] = "NonMarketable"
    nonmarketable.loc[:, "HolderType"] = "TrustFunds"
    nonmarketable.loc[:, "HolderSubBucket"] = ""
    nonmarketable.loc[:, "TDCPrincipalHolderType"] = "TrustFunds"
    nonmarketable.loc[:, "TDCPrincipalHolderSubBucket"] = ""
    nonmarketable.loc[:, "IssueDate"] = pd.Timestamp("1996-12-31")
    nonmarketable.loc[:, "MaturityDate"] = pd.Timestamp("2026-12-31")
    nonmarketable.loc[:, "OriginalMaturityYears"] = 30.0
    nonmarketable.loc[:, "CouponRate"] = 0.0
    nonmarketable.loc[:, "MaturityCategory"] = pd.NA
    params["initial_bonds_df"] = nonmarketable
    params["nonmarketable_params"] = {
        "interest_crediting_frequency": "annual",
        "rate_setting_method": "yield_curve_points",
        "interest_rate_basis_maturities": [5.0, 10.0],
    }

    results, _ = run_simulation(
        params,
        "2026-12-30",
        "2027-01-02",
        freq="3D",
        scenario_name="nonmarketable_final_credit",
    )
    principal = results.attrs["handoff_tables"]["tdcsim_period_principal_flows"]

    assert results["NonMarketableInterestCapitalized_Period"].sum() == pytest.approx(5.0)
    assert results["NonMarketableInterestCapitalized_Cumulative"].iloc[-1] == pytest.approx(5.0)
    assert results["PrincipalPaid_Bonds"].sum() == pytest.approx(105.0)
    assert len(principal) == 1
    assert principal[0]["cash_paid_bil"] == pytest.approx(105.0)
