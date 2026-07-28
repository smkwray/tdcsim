from pathlib import Path

import pandas as pd
import pytest

from tdcsim_cbo import CboBaselinePackage
from tdcsim_cbo._json import write_json
from tdcsim_cbo.runner import RunnerError, build_runtime_params
from test_tdcsim_cbo_closeout_interface import _write_runner_package


def _materialized_inputs(tmp_path: Path) -> Path:
    package, attestation = _write_runner_package(tmp_path)
    baseline = CboBaselinePackage.open(package, attestation_path=attestation)
    return baseline.materialize(tmp_path / "materialized") / "forecast_inputs"


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
    write_json(
        inputs / "tdcsim_runtime_assumptions.json",
        {
            "schema_version": "tdcsim_cbo_runtime_assumptions_v1",
            "fiscal_incidence_policy_id": policy_id,
        },
    )

    assert build_runtime_params(inputs)["fiscal_incidence_policy"] == {
        "policy_id": policy_id,
        "mode": "explicit_scenario_assumption",
        "incidence_basis": "signed_net_primary_proxy",
        "du_share": 0.99,
        "ru_share": 0.01,
        "foreign_share": 0.0,
        "other_share": 0.0,
    }
