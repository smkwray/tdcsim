from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import historical_replay_solver
from historical_replay_solver import solve_sector_cohort_allocations


def test_solver_matches_sector_and_cohort_totals_when_balanced():
    cohorts = pd.DataFrame(
        [
            {"quarter": "2025Q1", "cohort_id": "c1", "outstanding": 40.0},
            {"quarter": "2025Q1", "cohort_id": "c2", "outstanding": 35.0},
            {"quarter": "2025Q1", "cohort_id": "c3", "outstanding": 25.0},
        ]
    )
    sector_stocks = pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "Banks", "outstanding": 30.0},
            {"quarter": "2025Q1", "sector": "MMF", "outstanding": 20.0},
            {"quarter": "2025Q1", "sector": "Foreign", "outstanding": 50.0},
        ]
    )

    allocations, diagnostics = solve_sector_cohort_allocations(
        cohorts,
        sector_stocks,
        quarter="2025Q1",
    )

    assert list(allocations.columns[:4]) == ["quarter", "sector", "cohort_id", "allocated_outstanding"]
    assert (allocations["allocated_outstanding"] >= 0.0).all()
    pd.testing.assert_series_equal(
        allocations.groupby("cohort_id")["allocated_outstanding"].sum().sort_index(),
        cohorts.groupby("cohort_id")["outstanding"].sum().sort_index(),
        atol=1e-9,
        rtol=0.0,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        allocations.groupby("sector")["allocated_outstanding"].sum().sort_index(),
        sector_stocks.groupby("sector")["outstanding"].sum().sort_index(),
        atol=1e-9,
        rtol=0.0,
        check_names=False,
    )

    total_balance = diagnostics[diagnostics["diagnostic_type"] == "total_balance"].iloc[0]
    assert total_balance["status"] == "balanced"
    assert abs(total_balance["residual"]) <= 1e-9


def test_solver_projects_zero_slack_highs_corner_to_prior_preserving_cells(monkeypatch):
    def fail_entropy(seed, row_targets, col_targets, coefficients, eligibility, row_constrained=None, *, tolerance, max_iterations=2_000):
        return np.asarray(seed, dtype=float), 1, False, 999.0

    def certify_slack(seed, row_targets, col_targets, *, tolerance, row_coefficients=None, eligibility=None, row_constrained=None):
        matrix = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=float,
        )
        return matrix, np.zeros(3), np.zeros(3), True, 0.0

    monkeypatch.setattr(historical_replay_solver, "_run_exact_weighted_entropy_solver", fail_entropy)
    monkeypatch.setattr(historical_replay_solver, "_run_min_slack_solver", certify_slack)

    cohorts = pd.DataFrame(
        [
            {"quarter": "2025Q1", "cohort_id": "c1", "outstanding": 10.0},
            {"quarter": "2025Q1", "cohort_id": "c2", "outstanding": 10.0},
            {"quarter": "2025Q1", "cohort_id": "c3", "outstanding": 10.0},
        ]
    )
    sector_stocks = pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "Banks", "outstanding": 10.0},
            {"quarter": "2025Q1", "sector": "Private", "outstanding": 10.0},
            {"quarter": "2025Q1", "sector": "Foreign", "outstanding": 10.0},
        ]
    )
    prior = pd.DataFrame(
        [
            {
                "quarter": "2024Q4",
                "sector": sector,
                "cohort_id": cohort,
                "allocated_outstanding": 10.0 / 3.0,
            }
            for sector in ["Banks", "Private", "Foreign"]
            for cohort in ["c1", "c2", "c3"]
        ]
    )

    allocations, diagnostics = solve_sector_cohort_allocations(
        cohorts,
        sector_stocks,
        quarter="2025Q1",
        prior_allocations=prior,
    )

    assert not allocations.empty
    solver = diagnostics[diagnostics["diagnostic_type"].eq("solver")].iloc[0]
    total_balance = diagnostics[diagnostics["diagnostic_type"].eq("total_balance")].iloc[0]
    assert total_balance["status"] == "prior_preserving_exact_fallback"
    assert solver["solver_method"] == "exact_prior_preserving_qp_projection_from_highs_feasible"
    assert solver["cell_portfolio_status"] == "cell_portfolio_prior_preserving_qp_projected"
    assert solver["continuity_prior_overlap_face"] == pytest.approx(30.0)
    assert solver["residual"] <= 1.0e-9

    pd.testing.assert_series_equal(
        allocations.groupby("cohort_id")["allocated_outstanding"].sum().sort_index(),
        cohorts.groupby("cohort_id")["outstanding"].sum().sort_index(),
        atol=1e-8,
        rtol=0.0,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        allocations.groupby("sector")["allocated_outstanding"].sum().sort_index(),
        sector_stocks.groupby("sector")["outstanding"].sum().sort_index(),
        atol=1e-8,
        rtol=0.0,
        check_names=False,
    )

    pivot = allocations.pivot_table(
        index="sector",
        columns="cohort_id",
        values="allocated_outstanding",
        aggfunc="sum",
        fill_value=0.0,
    )
    support = pivot.gt(1.0e-8).sum(axis=1)
    effective_cohorts = []
    tv_from_prior = []
    for _, row in pivot.iterrows():
        shares = row.to_numpy(dtype=float) / float(row.sum())
        effective_cohorts.append(1.0 / float((shares**2).sum()))
        prior_shares = np.full_like(shares, 1.0 / shares.size)
        tv_from_prior.append(0.5 * float(np.abs(shares - prior_shares).sum()))
    assert support.tolist() == [3, 3, 3]
    assert min(effective_cohorts) > 2.99
    assert max(tv_from_prior) < 0.002

    certified_corner = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=float,
    )
    corner_support = (certified_corner > 1.0e-12).sum(axis=1)
    corner_effective_cohorts = []
    corner_tv_from_prior = []
    for row in certified_corner:
        shares = row / row.sum()
        corner_effective_cohorts.append(1.0 / float((shares**2).sum()))
        prior_shares = np.full_like(shares, 1.0 / shares.size)
        corner_tv_from_prior.append(0.5 * float(np.abs(shares - prior_shares).sum()))
    assert corner_support.tolist() == [1, 1, 1]
    assert corner_effective_cohorts == pytest.approx([1.0, 1.0, 1.0])
    assert corner_tv_from_prior == pytest.approx([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])


def test_solver_adds_explicit_basis_residual_when_mspd_exceeds_z1_totals():
    cohorts = pd.DataFrame(
        [
            {"quarter": "2025Q2", "cohort_id": "c1", "outstanding": 70.0},
            {"quarter": "2025Q2", "cohort_id": "c2", "outstanding": 30.0},
        ]
    )
    sector_stocks = pd.DataFrame(
        [
            {"quarter": "2025Q2", "sector": "Banks", "outstanding": 24.0},
            {"quarter": "2025Q2", "sector": "DomesticNonbankExMMF", "outstanding": 56.0},
        ]
    )

    allocations, diagnostics = solve_sector_cohort_allocations(
        cohorts,
        sector_stocks,
        quarter="2025Q2",
    )

    sector_totals = allocations.groupby("sector")["allocated_outstanding"].sum().sort_index()
    expected = pd.Series(
        {
            "Banks": 24.0,
            "DomesticNonbankExMMF": 56.0,
            "MSPD_Z1_SourceBasisResidual": 20.0,
        }
    ).sort_index()
    pd.testing.assert_series_equal(
        sector_totals,
        expected,
        atol=1e-9,
        rtol=0.0,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        allocations.groupby("cohort_id")["allocated_outstanding"].sum().sort_index(),
        cohorts.groupby("cohort_id")["outstanding"].sum().sort_index(),
        atol=1e-9,
        rtol=0.0,
        check_names=False,
    )

    total_balance = diagnostics[diagnostics["diagnostic_type"] == "total_balance"].iloc[0]
    assert total_balance["status"] == "explicit_basis_residual"
    assert pd.isna(total_balance["scale_factor"])
    assert abs(total_balance["residual"] - 20.0) <= 1e-9

    banks_diag = diagnostics[
        (diagnostics["diagnostic_type"] == "sector_balance")
        & (diagnostics["subject"] == "Banks")
    ].iloc[0]
    assert abs(banks_diag["input_total"] - 24.0) <= 1e-9
    assert abs(banks_diag["achieved_total"] - 24.0) <= 1e-9
    assert abs(banks_diag["residual"]) <= 1e-9

    residual_diag = diagnostics[
        (diagnostics["diagnostic_type"] == "sector_balance")
        & (diagnostics["subject"] == "MSPD_Z1_SourceBasisResidual")
    ].iloc[0]
    assert residual_diag["status"] == "explicit_basis_residual"
    assert abs(residual_diag["achieved_total"] - 20.0) <= 1e-9


def test_solver_uses_market_value_coefficients_for_lm_sector_targets():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q2",
                "cohort_id": "discount_note",
                "outstanding": 100.0,
                "market_value_ratio": 0.8,
            },
        ]
    )
    sector_stocks = pd.DataFrame(
        [
            {
                "quarter": "2025Q2",
                "sector": "Foreign",
                "outstanding": 80.0,
                "valuation_basis": "z1_market_value_level",
            },
        ]
    )

    allocations, diagnostics = solve_sector_cohort_allocations(
        cohorts,
        sector_stocks,
        quarter="2025Q2",
    )

    assert allocations["allocated_outstanding"].sum() == pytest.approx(100.0)
    assert set(allocations["sector"]) == {"Foreign"}
    total_balance = diagnostics[diagnostics["diagnostic_type"] == "total_balance"].iloc[0]
    assert total_balance["basis_residual"] == 0.0
    assert total_balance["sector_face_equivalent_total"] == 100.0
    sector_diag = diagnostics[
        (diagnostics["diagnostic_type"] == "sector_balance")
        & (diagnostics["subject"] == "Foreign")
    ].iloc[0]
    assert sector_diag["achieved_total"] == pytest.approx(80.0)
    assert sector_diag["achieved_face_total"] == pytest.approx(100.0)
    assert sector_diag["status"] == "matched"
    assert total_balance["raw_mspd_minus_z1_total"] == 20.0
    assert total_balance["modeled_face_equivalent_basis_residual"] == 0.0


def test_solver_matches_mixed_basis_sector_constraints_with_nonuniform_prices():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q2",
                "cohort_id": "discount_note",
                "outstanding": 50.0,
                "market_value_ratio": 0.8,
                "prior_MarketA": 3.0,
                "prior_MarketB": 1.0,
                "prior_Book": 2.0,
            },
            {
                "quarter": "2025Q2",
                "cohort_id": "premium_note",
                "outstanding": 50.0,
                "market_value_ratio": 1.2,
                "prior_MarketA": 1.0,
                "prior_MarketB": 3.0,
                "prior_Book": 2.0,
            },
        ]
    )
    sector_stocks = pd.DataFrame(
        [
            {
                "quarter": "2025Q2",
                "sector": "MarketA",
                "outstanding": 28.0,
                "valuation_basis": "z1_market_value_level",
            },
            {
                "quarter": "2025Q2",
                "sector": "MarketB",
                "outstanding": 32.0,
                "valuation_basis": "z1_market_value_level",
            },
            {
                "quarter": "2025Q2",
                "sector": "Book",
                "outstanding": 40.0,
                "valuation_basis": "z1_book_or_par_level",
            },
        ]
    )

    allocations, diagnostics = solve_sector_cohort_allocations(
        cohorts,
        sector_stocks,
        quarter="2025Q2",
    )

    cohort_totals = allocations.groupby("cohort_id")["allocated_outstanding"].sum().sort_index()
    pd.testing.assert_series_equal(
        cohort_totals,
        cohorts.groupby("cohort_id")["outstanding"].sum().sort_index(),
        atol=1e-8,
        rtol=0.0,
        check_names=False,
    )
    prices = cohorts.set_index("cohort_id")["market_value_ratio"]
    weighted = allocations.assign(
        row_value=allocations["allocated_outstanding"] * allocations["cohort_id"].map(prices)
    )
    sector_values = weighted.groupby("sector")["row_value"].sum()
    assert sector_values["MarketA"] == pytest.approx(28.0, abs=1e-8)
    assert sector_values["MarketB"] == pytest.approx(32.0, abs=1e-8)
    assert allocations.loc[allocations["sector"].eq("Book"), "allocated_outstanding"].sum() == pytest.approx(
        40.0,
        abs=1e-8,
    )
    sector_diags = diagnostics[diagnostics["diagnostic_type"].eq("sector_balance")]
    assert pd.to_numeric(sector_diags["residual"], errors="coerce").abs().max() <= 1e-8
    solver = diagnostics[diagnostics["diagnostic_type"].eq("solver")].iloc[0]
    assert solver["solver_method"] == "exact_weighted_entropy_projection"
    assert solver["status"] == "converged"


def test_solver_continuity_prior_tilts_surviving_cohorts_without_breaking_totals():
    cohorts = pd.DataFrame(
        [
            {"quarter": "2025Q2", "cohort_id": "survivor_a", "outstanding": 50.0},
            {"quarter": "2025Q2", "cohort_id": "survivor_b", "outstanding": 50.0},
        ]
    )
    sector_stocks = pd.DataFrame(
        [
            {"quarter": "2025Q2", "sector": "Banks", "outstanding": 50.0},
            {"quarter": "2025Q2", "sector": "Private", "outstanding": 50.0},
        ]
    )
    prior = pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "Banks", "cohort_id": "survivor_a", "allocated_outstanding": 50.0},
            {"quarter": "2025Q1", "sector": "Private", "cohort_id": "survivor_b", "allocated_outstanding": 50.0},
        ]
    )

    allocations, diagnostics = solve_sector_cohort_allocations(
        cohorts,
        sector_stocks,
        quarter="2025Q2",
        prior_allocations=prior,
        continuity_weight=20.0,
    )

    pivot = allocations.pivot_table(
        index="sector",
        columns="cohort_id",
        values="allocated_outstanding",
        aggfunc="sum",
    )
    assert pivot.loc["Banks", "survivor_a"] > 45.0
    assert pivot.loc["Private", "survivor_b"] > 45.0
    pd.testing.assert_series_equal(
        allocations.groupby("cohort_id")["allocated_outstanding"].sum().sort_index(),
        cohorts.groupby("cohort_id")["outstanding"].sum().sort_index(),
        atol=1e-9,
        rtol=0.0,
        check_names=False,
    )
    solver = diagnostics[diagnostics["diagnostic_type"] == "solver"].iloc[0]
    assert solver["continuity_prior_overlap_face"] == 100.0
    assert solver["status"] == "converged"


def test_solver_uses_tdcsim_holder_prior_columns_without_collapsing_native_sectors():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "bank_tilt",
                "outstanding": 50.0,
                "prior_holder_Banks": 100.0,
                "prior_holder_Private": 1.0,
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "private_tilt",
                "outstanding": 50.0,
                "prior_holder_Banks": 1.0,
                "prior_holder_Private": 100.0,
            },
        ]
    )
    sector_stocks = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "sector": "us_chartered_depository_institutions",
                "tdcsim_holder": "Banks",
                "outstanding": 50.0,
            },
            {
                "quarter": "2025Q1",
                "sector": "household_sector",
                "tdcsim_holder": "Private",
                "outstanding": 50.0,
            },
        ]
    )

    allocations, diagnostics = solve_sector_cohort_allocations(
        cohorts,
        sector_stocks,
        quarter="2025Q1",
    )

    pivot = allocations.pivot_table(
        index="sector",
        columns="cohort_id",
        values="allocated_outstanding",
        aggfunc="sum",
    )
    assert pivot.loc["us_chartered_depository_institutions", "bank_tilt"] > 49.0
    assert pivot.loc["household_sector", "private_tilt"] > 49.0
    assert set(allocations["sector"]) == {
        "us_chartered_depository_institutions",
        "household_sector",
    }
    assert diagnostics.loc[diagnostics["diagnostic_type"] == "solver", "status"].iloc[0] == "converged"
