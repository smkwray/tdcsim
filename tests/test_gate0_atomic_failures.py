"""Gate 0 counterexamples for fail-closed, atomic production execution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import sim_engine
import tdcsim_cbo.runner as cbo_runner
import test_cbo_engine_integration as cbo_engine_fixtures
import test_engine_and_validation as engine_fixtures
import test_tdcsim_cbo_closeout_interface as cbo_fixtures
from bill_quote_basis import discount_price_ratio
from tdcsim_cbo import CboScenarioSpec
from tdcsim_cbo._json import read_json
from tdcsim_cbo.runner import RunnerError
from tdc_shared import BOND_PORTFOLIO_COLS, HOLDER_TYPES, PORTFOLIO_DTYPES


class InjectedMutationFailure(RuntimeError):
    """Sentinel failure injected after a period mutation has begun."""


def _assert_no_complete_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "tdcsim_cbo_run_manifest.json"
    if manifest_path.exists():
        assert read_json(manifest_path).get("status") != "complete"


@pytest.mark.parametrize("failure_kind", ["unbooked_residual", "negative_tga"])
def test_production_run_rejects_hard_cash_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """A hard cash breach must not return a complete, reusable run."""

    baseline, scenarios = cbo_fixtures._runner_baseline_and_scenarios(tmp_path)
    scenario_path = scenarios["cash" if failure_kind == "unbooked_residual" else "noop"]
    if failure_kind == "negative_tga":
        production_run_simulation = cbo_runner.run_simulation

        def return_overdrawn_results(*args, **kwargs):
            results, portfolio = production_run_simulation(*args, **kwargs)
            results.loc[results.index[-1], "TGA"] = -1_500.0
            return results, portfolio

        monkeypatch.setattr(cbo_runner, "run_simulation", return_overdrawn_results)

    run_dir = tmp_path / f"run-{failure_kind}"
    with pytest.raises(RunnerError):
        cbo_runner.run_cbo_scenario(
            baseline,
            CboScenarioSpec.from_file(scenario_path),
            run_dir,
        )
    _assert_no_complete_manifest(run_dir)


def test_production_cli_returns_nonzero_for_tga_only_residual(tmp_path: Path) -> None:
    """The +7.5 billion TGA-only residual must make the production CLI fail."""

    baseline, scenarios = cbo_fixtures._runner_baseline_and_scenarios(tmp_path)
    run_dir = tmp_path / "run-cli-cash-failure"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tdcsim_cbo.cli",
            "run",
            "--baseline",
            str(baseline.package_path),
            "--attestation",
            str(baseline.attestation.path),
            "--scenario",
            str(scenarios["cash"]),
            "--output-dir",
            str(run_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0, completed.stdout
    _assert_no_complete_manifest(run_dir)


def test_failed_portfolio_append_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issuance cash and liability mutations must not survive a failed append."""

    params = engine_fixtures.minimal_params()
    params["initial_bonds_df"] = pd.DataFrame(
        [engine_fixtures.make_bond_row(FaceValue=1.0)],
        columns=BOND_PORTFOLIO_COLS,
    ).astype(PORTFOLIO_DTYPES, errors="ignore")
    production_concat = sim_engine.pd.concat

    def fail_new_portfolio_append(objects, *args, **kwargs):
        frames = list(objects)
        if len(frames) == 2 and all("BondID" in frame.columns for frame in frames):
            raise InjectedMutationFailure("injected portfolio append failure")
        return production_concat(frames, *args, **kwargs)

    monkeypatch.setattr(sim_engine.pd, "concat", fail_new_portfolio_append)

    with pytest.raises(InjectedMutationFailure, match="portfolio append"):
        sim_engine.run_simulation(
            params,
            "2025-01-01",
            "2025-01-19",
            freq="W",
            scenario_name="gate0_append_failure",
        )


def test_failed_trading_mutation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trading exception must not be converted into a passed run."""

    params = engine_fixtures.minimal_params()
    params["simulation_period"]["enable_preference_trading"] = True
    params["initial_bonds_df"] = pd.DataFrame(
        [engine_fixtures.make_bond_row()],
        columns=BOND_PORTFOLIO_COLS,
    ).astype(PORTFOLIO_DTYPES, errors="ignore")

    def fail_after_mutating_portfolio(portfolio, *args, **kwargs):
        portfolio.loc[portfolio.index[0], "FaceValue"] = 999_999.0
        raise InjectedMutationFailure("injected trading mutation failure")

    monkeypatch.setattr(sim_engine, "execute_preference_trades", fail_after_mutating_portfolio)

    with pytest.raises(InjectedMutationFailure, match="trading mutation"):
        sim_engine.run_simulation(
            params,
            "2025-01-01",
            "2025-01-19",
            freq="W",
            scenario_name="gate0_trading_failure",
        )


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        ("not-a-date", "2025-01-19"),
        ("2025-01-19", "2025-01-01"),
        ("2025-01-01", "2025-01-01"),
    ],
    ids=["invalid", "reversed", "degenerate"],
)
def test_invalid_simulation_dates_fail(start_date: str, end_date: str) -> None:
    """Invalid or non-running date windows must fail instead of returning empty success."""

    with pytest.raises(ValueError, match="date|period"):
        sim_engine.run_simulation(
            engine_fixtures.minimal_params(),
            start_date,
            end_date,
            freq="W",
            scenario_name="gate0_invalid_dates",
        )


def test_generic_tips_issuance_requires_explicit_real_curve() -> None:
    """A nominal curve must never be silently reused to price new TIPS."""

    params = engine_fixtures.minimal_params()
    params["treasury_issuance_profile"]["TIPS"] = {
        "target_percentage": 1.0,
        "maturities": [10.0],
        "maturity_distribution": [1.0],
    }
    for holder, preferences in params["sector_preferences"].items():
        preferences["tips_pct"] = 1.0 if holder == "Private" else 0.0

    with pytest.raises(ValueError, match="real (yield )?curve"):
        sim_engine.run_simulation(
            params,
            "2025-01-01",
            "2025-01-19",
            freq="W",
            scenario_name="gate0_generic_tips_issuance",
        )


def test_generic_tips_trading_requires_explicit_real_curve() -> None:
    """Secondary TIPS settlement must not price real cash flows on a nominal curve."""

    portfolio = pd.DataFrame(
        [
            engine_fixtures.make_bond_row(
                BondID=801,
                SecurityType="TIPS",
                HolderType="Banks",
                FaceValue=200.0,
                CouponRate=0.005,
                IssueDate=pd.Timestamp("2023-01-01"),
                MaturityDate=pd.Timestamp("2033-01-01"),
                OriginalMaturityYears=10.0,
                MaturityCategory="tips",
                OriginalPrincipal=200.0,
                AdjustedPrincipal=210.0,
                ReferenceCPI_Issue=100.0,
                IndexRatio=1.05,
            ),
            engine_fixtures.make_bond_row(
                BondID=802,
                SecurityType="Fixed",
                HolderType="Private",
                FaceValue=200.0,
                CouponRate=0.0,
                IssueDate=pd.Timestamp("2024-07-01"),
                MaturityDate=pd.Timestamp("2025-07-01"),
                OriginalMaturityYears=1.0,
                MaturityCategory="bills",
                IssuePriceRatio=0.96,
                IssueProceeds=192.0,
            ),
        ],
        columns=BOND_PORTFOLIO_COLS,
    ).astype(PORTFOLIO_DTYPES, errors="ignore")
    preferences = {
        holder: {
            "bills_pct": 0.0,
            "notes_pct": 0.0,
            "bonds_pct": 0.0,
            "tips_pct": 0.0,
            "frn_pct": 0.0,
        }
        for holder in HOLDER_TYPES
    }
    preferences["Banks"]["bills_pct"] = 0.5
    preferences["Private"]["tips_pct"] = 0.5

    with pytest.raises(ValueError, match="real (yield )?curve"):
        sim_engine.execute_preference_trades(
            portfolio,
            pd.Timestamp("2025-01-15"),
            [0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            [0.04, 0.041, 0.042, 0.043, 0.045, 0.05],
            preferences,
            engine_fixtures.base_issuance_profile(),
            "gate0_generic_tips_trading",
        )


def test_cbo_bill_issue_price_uses_treasury_discount_quote_identity(tmp_path: Path) -> None:
    """A typed 13-week discount quote must set issue proceeds by 31 CFR 356."""

    paths = cbo_engine_fixtures._build_temp_forecast_inputs(
        tmp_path,
        pre_issuance_controlled_debt_bil=0.0,
    )
    params = cbo_engine_fixtures._minimal_engine_params(
        paths,
        opening_controlled_debt_bil=0.0,
    )
    params["yield_curve_surface"] = {
        "file": str(cbo_engine_fixtures._dynamic_surface_path(tmp_path)),
        "scenario_id": "baseline",
    }
    params["treasury_issuance_profile"]["bills"]["maturities"] = [0.25]
    params["treasury_issuance_profile"]["bills"]["maturity_distribution"] = [1.0]

    _, portfolio = sim_engine.run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    issued_bill = portfolio[
        portfolio["SecurityType"].eq("Fixed")
        & portfolio["CouponRate"].eq(0.0)
        & portfolio["IssueDate"].eq(pd.Timestamp("2026-09-30"))
    ].iloc[0]
    expected_price_ratio = discount_price_ratio(0.037, 91.0)

    assert issued_bill["IssuePriceRatio"] == pytest.approx(expected_price_ratio, abs=1e-12)
    assert issued_bill["IssueProceeds"] == pytest.approx(
        issued_bill["FaceValue"] * expected_price_ratio,
        abs=1e-10,
    )


def test_opening_fed_mismatch_is_rejected_or_restatement_is_nonsettling(
    tmp_path: Path,
) -> None:
    """An opening stock mismatch must never become an in-period Fed purchase."""

    paths = cbo_engine_fixtures._build_temp_forecast_inputs(
        tmp_path,
        cbo_public_debt_target_bil=1_250.0,
    )
    fed_rows = cbo_engine_fixtures.build_fed_holdings_path_rows(
        scenario_id="baseline",
        periods=cbo_engine_fixtures._single_period(),
        opening_state_date="2026-09-20",
        opening_cb_holdings_bil=0.0,
        cbo_fy_end_fed_holdings_bil={2026: 50.0},
        observation_date="2026-09-20",
        available_date="2026-09-20",
    )
    fed_rows[0]["cbo_fed_holdings_target_bil"] = 50.0
    paths["fed_holdings_path_file"] = cbo_engine_fixtures._write_csv(
        tmp_path / "tdcsim_fed_holdings_path.csv",
        fed_rows,
    )

    with pytest.raises((ValueError, RuntimeError), match=r"(?i)opening.*fed"):
        sim_engine.run_simulation(
            cbo_engine_fixtures._minimal_engine_params(paths),
            "2026-09-20",
            "2026-09-30",
            freq="10D",
            scenario_name="baseline",
        )
