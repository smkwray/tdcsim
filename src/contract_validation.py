"""Validation for RateWall contract exports."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd


IDENTITY_TOLERANCE = Decimal("1e-7")


def _decimal(value) -> Decimal:
    if value is None or value == "":
        return Decimal("0")
    return Decimal(str(value))


def validate_ratewall_contract(
    summary: pd.DataFrame,
    components: pd.DataFrame,
) -> dict[str, str]:
    """Validate RateWall contract summary/component identities."""

    failures: list[str] = []
    if summary.empty:
        failures.append("summary_empty")
    required_summary = {
        "scenario_id",
        "quarter",
        "tdc_change_bil",
        "tdc_fiscal_flow_bil",
        "tdc_debt_service_principal_to_du_bil",
        "tdc_debt_service_interest_to_du_bil",
        "tdc_auction_absorption_du_bil",
        "tdc_secondary_trades_bil",
        "tdc_other_bil",
        "overlap_cashflow_bil",
        "tdc_change_ex_overlap_bil",
    }
    missing_summary = required_summary - set(summary.columns)
    if missing_summary:
        failures.append(f"summary_missing:{','.join(sorted(missing_summary))}")
    required_components = {
        "scenario_id",
        "quarter",
        "component_key",
        "amount_bil",
        "enters_direct_interest_support",
        "enters_tdc_deposit_support_default",
    }
    missing_components = required_components - set(components.columns)
    if missing_components:
        failures.append(f"components_missing:{','.join(sorted(missing_components))}")
    if failures:
        return {"validation_status": "fail", "failure_reasons": ";".join(failures)}

    for _, row in summary.iterrows():
        expected = (
            _decimal(row["tdc_fiscal_flow_bil"])
            + _decimal(row["tdc_debt_service_principal_to_du_bil"])
            + _decimal(row["tdc_debt_service_interest_to_du_bil"])
            + _decimal(row["tdc_auction_absorption_du_bil"])
            + _decimal(row["tdc_secondary_trades_bil"])
            + _decimal(row["tdc_other_bil"])
        )
        actual = _decimal(row["tdc_change_bil"])
        if abs(expected - actual) > IDENTITY_TOLERANCE:
            failures.append(
                f"tdc_identity:{row['scenario_id']}:{row['quarter']}:{expected}:{actual}"
            )
        ex_overlap = _decimal(row["tdc_change_bil"]) - _decimal(row["overlap_cashflow_bil"])
        if abs(ex_overlap - _decimal(row["tdc_change_ex_overlap_bil"])) > IDENTITY_TOLERANCE:
            failures.append(f"overlap_identity:{row['scenario_id']}:{row['quarter']}")

    dual_entry = components[
        (components["enters_direct_interest_support"].astype(str) == "true")
        & (components["enters_tdc_deposit_support_default"].astype(str) == "true")
    ]
    if not dual_entry.empty:
        failures.append("component_dual_entry")

    tdc_default_components = components[
        components["enters_tdc_deposit_support_default"].astype(str).str.lower() == "true"
    ]
    default_sums: dict[tuple[str, str], Decimal] = {}
    for _, row in tdc_default_components.iterrows():
        key = (str(row["scenario_id"]), str(row["quarter"]))
        default_sums[key] = default_sums.get(key, Decimal("0")) + _decimal(row["amount_bil"])
    for _, row in summary.iterrows():
        key = (str(row["scenario_id"]), str(row["quarter"]))
        expected = _decimal(row["tdc_change_bil"]) - _decimal(row["overlap_cashflow_bil"])
        actual = default_sums.get(key, Decimal("0"))
        if abs(actual - expected) > IDENTITY_TOLERANCE:
            failures.append(
                f"component_tdc_default_identity:{row['scenario_id']}:{row['quarter']}:{actual}:{expected}"
            )

    return {
        "validation_status": "pass" if not failures else "fail",
        "failure_reasons": "" if not failures else ";".join(failures),
    }
