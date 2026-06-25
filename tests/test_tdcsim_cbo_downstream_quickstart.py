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
        "05_rate_down_25bp.json",
        "06_rate_up_25bp.json",
        "07_issuance_shorter.json",
        "08_issuance_longer.json",
        "09_private_holder_high.json",
        "10_private_holder_low.json",
        "11_primary_deficit_plus_1pct.json",
        "12_operating_cash_inflation_beta_50.json",
        "13_fed_holdings_scale_1.json",
    ]
    assert completed.stdout.count(".json") == len(scenario_paths)
    holder_example = json.loads((scenarios_dir / "03_sector_holders.json").read_text(encoding="utf-8"))
    assert holder_example["overrides"]["holder_preferences"]["mode"] == "dated_static_shares"
    assert {row["effective_date"] for row in holder_example["overrides"]["holder_preferences"]["rows"]} == {"2026-09-20"}
    _assert_matched_scenario_boundaries(scenarios_dir)

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
    results_path = _output_csv(run_dir, "results_compact")
    portfolio_path = _output_csv(run_dir, "final_portfolio_compact")
    catalog_path = run_dir / "outputs" / "catalog.sqlite"

    assert summary_path.exists()
    assert results_path.exists()
    assert portfolio_path.exists()
    assert catalog_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rows"] > 0
    assert summary["final_portfolio_rows"] > 0


def _output_csv(run_dir: Path, stem: str) -> Path:
    plain = run_dir / "outputs" / f"{stem}.csv"
    gzip = run_dir / "outputs" / f"{stem}.csv.gz"
    return gzip if gzip.exists() else plain


def _last_nonempty_line(text: str) -> str:
    return next(line for line in reversed(text.splitlines()) if line.strip())


def _assert_matched_scenario_boundaries(scenarios_dir: Path) -> None:
    scenarios = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(scenarios_dir.glob("*.json"))
    }
    assert scenarios["05_rate_down_25bp.json"]["scenario_id"] == "tdcsim_rate_down_25bp_v1"
    assert scenarios["06_rate_up_25bp.json"]["scenario_id"] == "tdcsim_rate_up_25bp_v1"
    assert set(scenarios["05_rate_down_25bp.json"]["overrides"]) == {
        "frn_benchmark",
        "nominal_yield_curve",
    }
    assert set(scenarios["06_rate_up_25bp.json"]["overrides"]) == {
        "frn_benchmark",
        "nominal_yield_curve",
    }
    assert scenarios["05_rate_down_25bp.json"]["overrides"]["nominal_yield_curve"]["shock_bp"] == -25
    assert scenarios["06_rate_up_25bp.json"]["overrides"]["nominal_yield_curve"]["shock_bp"] == 25

    shorter = scenarios["07_issuance_shorter.json"]["overrides"]
    longer = scenarios["08_issuance_longer.json"]["overrides"]
    assert set(shorter) == {"issuance_mix"}
    assert set(longer) == {"issuance_mix"}
    assert shorter["issuance_mix"]["tips_share"] == longer["issuance_mix"]["tips_share"]
    assert shorter["issuance_mix"]["frn_share"] == longer["issuance_mix"]["frn_share"]
    assert (
        shorter["issuance_mix"]["fixed_remainder_shares"]["bills"]
        > longer["issuance_mix"]["fixed_remainder_shares"]["bills"]
    )
    assert (
        shorter["issuance_mix"]["fixed_remainder_shares"]["bonds"]
        < longer["issuance_mix"]["fixed_remainder_shares"]["bonds"]
    )

    high_private = scenarios["09_private_holder_high.json"]["overrides"]
    low_private = scenarios["10_private_holder_low.json"]["overrides"]
    assert set(high_private) == {"holder_preferences"}
    assert set(low_private) == {"holder_preferences"}
    high_share = high_private["holder_preferences"]["rows"][0]["shares"]["Private"]
    low_share = low_private["holder_preferences"]["rows"][0]["shares"]["Private"]
    assert high_share == 0.70
    assert low_share == 0.30

    assert set(scenarios["11_primary_deficit_plus_1pct.json"]["overrides"]) == {"primary_deficit"}
    assert set(scenarios["12_operating_cash_inflation_beta_50.json"]["overrides"]) == {"operating_cash"}
    assert set(scenarios["13_fed_holdings_scale_1.json"]["overrides"]) == {"fed_holdings"}
