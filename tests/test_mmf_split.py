"""Regression tests for the Private/MMF pass-through split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from simulation_core import run_simulation
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)
from test_engine_and_validation import make_bond_row, minimal_params


MMF_DU_COLUMNS = [
    "TDC_AuctionAbsorption_MMF",
    "TDC_PrincipalToDU_MMF",
    "TDC_BillDiscountInterestToDU_MMF",
    "TDC_CouponInterestToDU_MMF",
    "TDC_FRNInterestToDU_MMF",
    "TDC_TIPSCouponInterestToDU_MMF",
    "TDC_TIPSInflationCompensationToDU_MMF",
    "TDC_InterestToDU_MMF",
    "TDC_SecondaryTrades_MMF",
    "TDC_GrossIssuanceProceedsAbsorbedByDU_MMF",
]


def _prefs(*, bills: float = 0.0, notes: float = 0.0, bonds: float = 0.0) -> dict[str, float]:
    return {
        "bills_pct": bills,
        "notes_pct": notes,
        "bonds_pct": bonds,
    }


def _write_holder_absorption_path(path: Path, *, split_to_mmf: bool) -> None:
    subbucket = PRIVATE_SUBBUCKET_MMF if split_to_mmf else PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
    pd.DataFrame(
        [
            {
                "scenario_id": "mmf_split_regression",
                "quarter": "2025Q1",
                "holder_type": "Private",
                "holder_subbucket": subbucket,
                "bills_pct": 1.0,
                "notes_pct": 1.0,
                "bonds_pct": 1.0,
            }
        ]
    ).to_csv(path, index=False)


def _split_params(tmp_path: Path, *, p: float, split_to_mmf: bool = True) -> dict:
    params = minimal_params()
    params["initial_values"] = {"reserves": 2000.0, "tdc_level": 0.0, "tga": 200.0}
    params["tga_params"] = {"target_balance": 500.0, "floor": 0.0}
    params["yield_curve"]["rates"] = [0.04] * len(params["yield_curve"]["years"])
    params["simulation_period"]["enable_preference_trading"] = True
    params["private_mmf_split"] = {"mmf_deposit_pass_through": p}
    params["auction_absorption_preferences"] = {
        "Banks": _prefs(),
        "CB": _prefs(),
        "Foreign": _prefs(),
        "FedInternal": _prefs(),
        "TrustFunds": _prefs(),
        "Private": _prefs(bills=1.0, notes=1.0, bonds=1.0),
    }
    params["secondary_target_preferences"] = {
        "Banks": _prefs(notes=1.0),
        "Private": _prefs(bonds=1.0),
        "Foreign": _prefs(bills=1.0),
        "CB": _prefs(),
        "FedInternal": _prefs(),
        "TrustFunds": _prefs(),
    }
    holder_path = tmp_path / ("holder_split.csv" if split_to_mmf else "holder_legacy.csv")
    _write_holder_absorption_path(holder_path, split_to_mmf=split_to_mmf)
    params["ratewall_input_paths"] = {"holder_absorption_path_file": str(holder_path)}

    subbucket = PRIVATE_SUBBUCKET_MMF if split_to_mmf else ""
    rows = [
        make_bond_row(
            BondID=1,
            SecurityType="Fixed",
            HolderType="Private",
            HolderSubBucket=subbucket,
            FaceValue=10.0,
            CouponRate=0.0,
            MaturityDate=pd.Timestamp("2025-01-06"),
            OriginalMaturityYears=0.25,
            MaturityCategory="bills",
            IssueProceeds=9.0,
            IssueYieldAtIssue=0.04,
        ),
        make_bond_row(
            BondID=2,
            SecurityType="Fixed",
            HolderType="Private",
            HolderSubBucket=subbucket,
            FaceValue=20.0,
            CouponRate=0.10,
            MaturityDate=pd.Timestamp("2025-01-06"),
            OriginalMaturityYears=2.0,
            MaturityCategory="notes",
        ),
        make_bond_row(
            BondID=3,
            SecurityType="FRN",
            HolderType="Private",
            HolderSubBucket=subbucket,
            FaceValue=30.0,
            CouponRate=0.0,
            MaturityDate=pd.Timestamp("2025-01-06"),
            OriginalMaturityYears=2.0,
            MaturityCategory=None,
            FixedSpread=0.0,
            AccruedInterest_FRN=2.0,
            LastAccrualDate=pd.Timestamp("2025-01-01"),
        ),
        make_bond_row(
            BondID=4,
            SecurityType="TIPS",
            HolderType="Private",
            HolderSubBucket=subbucket,
            FaceValue=40.0,
            CouponRate=0.10,
            MaturityDate=pd.Timestamp("2025-01-06"),
            OriginalMaturityYears=5.0,
            MaturityCategory="tips",
            OriginalPrincipal=40.0,
            AdjustedPrincipal=40.0,
            ReferenceCPI_Issue=90.0,
            IndexRatio=1.0,
        ),
        make_bond_row(
            BondID=5,
            SecurityType="Fixed",
            HolderType="Private",
            HolderSubBucket=subbucket,
            FaceValue=100.0,
            CouponRate=0.04,
            MaturityDate=pd.Timestamp("2027-01-01"),
            OriginalMaturityYears=2.0,
            MaturityCategory="notes",
        ),
        make_bond_row(
            BondID=6,
            SecurityType="Fixed",
            HolderType="Banks",
            FaceValue=100.0,
            CouponRate=0.0,
            MaturityDate=pd.Timestamp("2025-07-01"),
            OriginalMaturityYears=0.5,
            MaturityCategory="bills",
            IssueProceeds=98.0,
            IssueYieldAtIssue=0.04,
        ),
    ]
    params["initial_bonds_df"] = pd.DataFrame(rows, columns=BOND_PORTFOLIO_COLS).astype(
        PORTFOLIO_DTYPES,
        errors="ignore",
    )
    params["tips_params"] = {
        "cpi_start_level": 100.0,
        "cpi_annual_inflation": 0.0,
        "ref_cpi_lag_months": 3,
        "default_real_coupon_rate": 0.01,
    }
    params["financing_cost_options"] = {"include_tips_inflation_accretion": True}
    return params


def _run_case(tmp_path: Path, *, p: float, split_to_mmf: bool = True) -> pd.DataFrame:
    results, _ = run_simulation(
        _split_params(tmp_path, p=p, split_to_mmf=split_to_mmf),
        "2025-01-01",
        "2025-01-20",
        freq="W",
        scenario_name="mmf_split_regression",
    )
    return results


def test_full_mmf_pass_through_is_economically_identical_to_legacy_private(tmp_path):
    """At p=1.0, splitting Private into MMF and non-MMF cannot change TDC economics."""
    split = _run_case(tmp_path, p=1.0, split_to_mmf=True)
    legacy = _run_case(tmp_path, p=1.0, split_to_mmf=False)

    max_delta = (split["TDC_Change"] - legacy["TDC_Change"]).abs().max()
    assert max_delta < 1e-6
    assert split["TDC_AuctionAbsorption_MMF"].abs().sum() > 0.0
    assert split["TDC_PrincipalToDU_MMF"].abs().sum() > 0.0
    assert split["TDC_SecondaryTrades_MMF"].abs().sum() > 0.0


def test_zero_pass_through_zeros_all_mmf_du_legs(tmp_path):
    """At p=0.0, MMF cash recycling is plumbing only and must not hit DU deposits."""
    results = _run_case(tmp_path, p=0.0)

    assert (results[MMF_DU_COLUMNS] == 0.0).all().all()
    assert results["TDC_AuctionAbsorption_MMFPlumbing"].abs().sum() > 0.0
    assert results["TDC_DebtService_MMFPlumbing"].abs().sum() > 0.0
    assert results["TDC_SecondaryTrades_MMFPlumbing"].abs().sum() > 0.0


def test_mmf_pass_through_is_symmetric_and_conserved_across_all_channels(tmp_path):
    """The same p applies to auction drains, debt-service credits, and secondary trades."""
    p = 0.15
    results = _run_case(tmp_path, p=p)

    auction_gross = results["TDC_AuctionAbsorption_MMF"] + results["TDC_AuctionAbsorption_MMFPlumbing"]
    debt_service_du = results["TDC_PrincipalToDU_MMF"] + results["TDC_InterestToDU_MMF"]
    debt_service_gross = debt_service_du + results["TDC_DebtService_MMFPlumbing"]
    secondary_gross = results["TDC_SecondaryTrades_MMF"] + results["TDC_SecondaryTrades_MMFPlumbing"]

    assert auction_gross.abs().sum() > 0.0
    assert debt_service_gross.abs().sum() > 0.0
    assert secondary_gross.abs().sum() > 0.0
    pd.testing.assert_series_equal(
        results["TDC_AuctionAbsorption_MMF"],
        p * auction_gross,
        check_names=False,
        atol=1e-9,
        rtol=0.0,
    )
    pd.testing.assert_series_equal(
        debt_service_du,
        p * debt_service_gross,
        check_names=False,
        atol=1e-9,
        rtol=0.0,
    )
    pd.testing.assert_series_equal(
        results["TDC_SecondaryTrades_MMF"],
        p * secondary_gross,
        check_names=False,
        atol=1e-9,
        rtol=0.0,
    )


def test_mmf_du_effect_is_monotone_over_sensitivity_grid(tmp_path):
    """The sensitivity grid must move the MMF DU leg one-for-one with p."""
    principal_by_p = []
    for p in MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID:
        results = _run_case(tmp_path, p=p)
        principal_by_p.append(results["TDC_PrincipalToDU_MMF"].sum())

    assert MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID == [0.00, 0.15, 0.25, 0.50, 1.00]
    assert all(left < right for left, right in zip(principal_by_p, principal_by_p[1:]))
    gross_principal = principal_by_p[-1]
    assert gross_principal > 0.0
    assert principal_by_p == pytest.approx(
        [p * gross_principal for p in MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID],
        abs=1e-9,
        rel=0.0,
    )
