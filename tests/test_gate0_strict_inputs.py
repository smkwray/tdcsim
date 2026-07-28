from pathlib import Path

import pandas as pd
import pytest

from tdcsim_cbo import CboBaselinePackage, CboScenarioCompiler, CboScenarioSpec
from tdcsim_cbo._json import read_json, write_json
from tdcsim_cbo.runner import RunnerError, build_runtime_params
from test_tdcsim_cbo_closeout_interface import _write_runner_package, _write_scenario


def _materialized_inputs(tmp_path: Path) -> Path:
    package, attestation = _write_runner_package(tmp_path)
    baseline = CboBaselinePackage.open(package, attestation_path=attestation)
    scenario = _write_scenario(tmp_path / "noop.json", baseline, overrides={})
    return CboScenarioCompiler().compile(
        baseline,
        CboScenarioSpec.from_file(scenario),
        tmp_path / "compiled",
    ).forecast_inputs_dir


@pytest.mark.parametrize(
    ("case", "column", "value"),
    [
        ("malformed_date", "MaturityDate", "not-a-date"),
        ("malformed_numeric", "FaceValue", "not-a-number"),
        ("unknown_security", "SecurityType", "unknown-security"),
        ("unknown_holder", "HolderType", "unknown-holder"),
        ("unknown_status", "Status", "unknown-status"),
        ("nonfinite_stock", "FaceValue", float("inf")),
        ("negative_stock", "FaceValue", -1.0),
        ("duplicate_security_id", "BondID", None),
    ],
)
def test_runtime_rejects_malformed_opening_portfolio_before_simulation(
    tmp_path: Path,
    case: str,
    column: str,
    value: object,
) -> None:
    inputs = _materialized_inputs(tmp_path)
    path = inputs / "tdcsim_opening_portfolio.csv"
    portfolio = pd.read_csv(path)
    if case == "duplicate_security_id":
        portfolio = pd.concat([portfolio, portfolio.iloc[[0]]], ignore_index=True)
    else:
        portfolio[column] = portfolio[column].astype(object)
        portfolio.loc[0, column] = value
    portfolio.to_csv(path, index=False)

    with pytest.raises(RunnerError):
        build_runtime_params(inputs)


def test_fiscal_incidence_uses_explicit_policy_id_not_csv_row_order(tmp_path: Path) -> None:
    inputs = _materialized_inputs(tmp_path)
    path = inputs / "tdcsim_fiscal_incidence_policy.csv"
    policies = pd.read_csv(path)
    policy_id = "baseline_central_99du_1ru"
    assert policy_id in set(policies["policy_id"])
    policies.iloc[::-1].to_csv(path, index=False)
    runtime_path = inputs / "tdcsim_runtime_assumptions.json"
    runtime = read_json(runtime_path)
    runtime["fiscal_incidence_policy_id"] = policy_id
    write_json(runtime_path, runtime)

    assert build_runtime_params(inputs)["fiscal_incidence_policy"] == {
        "policy_id": policy_id,
        "mode": "explicit_scenario_assumption",
        "incidence_basis": "signed_net_primary_proxy",
        "du_share": 0.99,
        "ru_share": 0.01,
        "foreign_share": 0.0,
        "other_share": 0.0,
    }


def test_runtime_rejects_partial_explicit_opening_state(tmp_path: Path) -> None:
    inputs = _materialized_inputs(tmp_path)
    write_json(
        inputs / "tdcsim_opening_runtime_state.json",
        {
            "schema_version": "tdcsim_cbo_opening_runtime_state_v1",
            "opening_state_date": "2026-09-20",
            "initial_values": {"tdc_level": 0.0, "tga": 800.0},
        },
    )

    with pytest.raises(RunnerError, match="missing required keys"):
        build_runtime_params(inputs)


@pytest.mark.parametrize("case", ["missing", "malformed"])
def test_runtime_rejects_frn_without_finite_fixed_spread(
    tmp_path: Path,
    case: str,
) -> None:
    inputs = _materialized_inputs(tmp_path)
    path = inputs / "tdcsim_opening_portfolio.csv"
    portfolio = pd.read_csv(path)
    portfolio.loc[0, "SecurityType"] = "FRN"
    if case == "missing":
        portfolio = portfolio.drop(columns=["FixedSpread"])
    else:
        portfolio["FixedSpread"] = portfolio["FixedSpread"].astype(object)
        portfolio.loc[0, "FixedSpread"] = "not-a-spread"
    portfolio.to_csv(path, index=False)

    with pytest.raises(RunnerError, match="FixedSpread"):
        build_runtime_params(inputs)


def test_runtime_rejects_unknown_explicit_issuance_mode(tmp_path: Path) -> None:
    inputs = _materialized_inputs(tmp_path)
    write_json(
        inputs / "tdcsim_issuance_mix_assumptions.json",
        {
            "schema_version": "tdcsim_cbo_issuance_mix_assumptions_v1",
            "mode": "silently_use_defaults",
        },
    )

    with pytest.raises(RunnerError, match="unsupported compiled issuance mix mode"):
        build_runtime_params(inputs)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("tdcsim_opening_runtime_state.json", "required opening runtime state"),
        ("tdcsim_runtime_assumptions.json", "required compiled runtime assumptions"),
        ("tdcsim_issuance_mix_assumptions.json", "required compiled issuance mix assumptions"),
    ],
)
def test_runtime_rejects_missing_compiled_adapter_assumptions(
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    inputs = _materialized_inputs(tmp_path)
    (inputs / filename).unlink()

    with pytest.raises(RunnerError, match=message):
        build_runtime_params(inputs)


def test_runtime_rejects_duplicate_selected_fiscal_policy(tmp_path: Path) -> None:
    inputs = _materialized_inputs(tmp_path)
    path = inputs / "tdcsim_fiscal_incidence_policy.csv"
    policies = pd.read_csv(path)
    policy_id = "baseline_central_99du_1ru"
    selected = policies.loc[policies["policy_id"].eq(policy_id)]
    pd.concat([policies, selected], ignore_index=True).to_csv(path, index=False)
    runtime_path = inputs / "tdcsim_runtime_assumptions.json"
    runtime = read_json(runtime_path)
    runtime["fiscal_incidence_policy_id"] = policy_id
    write_json(runtime_path, runtime)

    with pytest.raises(RunnerError, match="select exactly one row"):
        build_runtime_params(inputs)
