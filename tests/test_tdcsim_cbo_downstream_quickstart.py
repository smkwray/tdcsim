import json
import os
import subprocess
import sys
from pathlib import Path

from tdcsim_cbo import CboScenarioSpec, run_cbo_scenario
from test_tdcsim_cbo_closeout_interface import _runner_baseline_and_scenarios


def test_downstream_quickstart_generates_and_runs_public_examples(tmp_path: Path) -> None:
    baseline, _ = _runner_baseline_and_scenarios(tmp_path)
    scenarios_dir = tmp_path / "downstream-scenarios"
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_cbo_example_scenarios.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline",
            str(baseline.package_path),
            "--attestation",
            str(baseline.attestation.path),
            "--output-dir",
            str(scenarios_dir),
            "--start-date",
            "2026-09-20",
            "--end-date",
            "2026-09-30",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    scenario_paths = sorted(scenarios_dir.glob("*.json"))
    assert [path.name for path in scenario_paths] == [
        "00_baseline_noop.json",
        "01_rates_inflation_frn_tips.json",
        "02_issuance_maturity_mix.json",
        "03_sector_holders.json",
        "04_fiscal_fed_cash.json",
    ]
    assert completed.stdout.count(".json") == len(scenario_paths)

    noop = scenario_paths[0]
    noop_spec = CboScenarioSpec.from_file(noop)
    noop_spec.assert_baseline_matches(baseline)
    noop_run = run_cbo_scenario(baseline, noop_spec, tmp_path / f"run-{noop.stem}")
    _assert_readable_outputs(noop_run.output_dir)
    assert noop_run.run_manifest["scenario"]["scenario_id"] == noop_spec.scenario_id

    for scenario_path in scenario_paths[1:]:
        spec = CboScenarioSpec.from_file(scenario_path)
        spec.assert_baseline_matches(baseline)
        run_dir = tmp_path / f"run-{scenario_path.stem}"

        assert _run_cli(
            "validate",
            "--baseline",
            str(baseline.package_path),
            "--attestation",
            str(baseline.attestation.path),
            "--scenario",
            str(scenario_path),
        ) == "pass"
        run_sha = _run_cli(
            "run",
            "--baseline",
            str(baseline.package_path),
            "--attestation",
            str(baseline.attestation.path),
            "--scenario",
            str(scenario_path),
            "--output-dir",
            str(run_dir),
            "--profile",
            "compact",
        )
        assert len(_last_nonempty_line(run_sha)) == 64
        assert _last_nonempty_line(
            _run_cli(
                "verify",
                "--run-dir",
                str(run_dir),
                "--baseline",
                str(baseline.package_path),
                "--attestation",
                str(baseline.attestation.path),
            )
        ) == "pass"
        _assert_readable_outputs(run_dir)


def _run_cli(*args: str) -> str:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "tdcsim_cbo.cli", *args],
        check=True,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_readable_outputs(run_dir: Path) -> None:
    summary_path = run_dir / "outputs" / "summary.json"
    results_path = run_dir / "outputs" / "results_compact.csv"
    portfolio_path = run_dir / "outputs" / "final_portfolio_compact.csv"
    catalog_path = run_dir / "outputs" / "catalog.sqlite"

    assert summary_path.exists()
    assert results_path.exists()
    assert portfolio_path.exists()
    assert catalog_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rows"] > 0
    assert summary["final_portfolio_rows"] > 0


def _last_nonempty_line(text: str) -> str:
    return next(line for line in reversed(text.splitlines()) if line.strip())
