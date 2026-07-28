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
import test_engine_and_validation as engine_fixtures
import test_tdcsim_cbo_closeout_interface as cbo_fixtures
from tdcsim_cbo import CboScenarioSpec
from tdcsim_cbo._json import read_json
from tdcsim_cbo.runner import RunnerError
from tdc_shared import BOND_PORTFOLIO_COLS, PORTFOLIO_DTYPES


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
