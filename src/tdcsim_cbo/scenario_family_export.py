"""Fiscal-year export for an illustrative scenario family run with opt-in TRC.

Each role must be a completed run whose bounded replay verified. Rows combine the
run's verified annual TDC and financing-cost summary, the fiscal-year TRC
projection, holder stocks at each fiscal-year end, and the 10-year nominal par
yield of the run's compiled curve in effect at each fiscal-year end. The export reports levels
per role; differences from the family baseline are left to consumers so that no
derived quantity hides its inputs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from holder_targets import PARTICIPANT_LABELS, STOCK_COLUMNS

from .open04_campaign import OPEN04_AGGREGATION_CLOCK_ID
from .marginal_tdc import INTEREST_PAYMENT_TYPES
from .trc_fy import read_output_csv, sha256_path, trc_fiscal_year_rows

FAMILY_FY_SCHEMA_VERSION = "tdcsim_scenario_family_fy_v1"
FAMILY_RECEIPT_SCHEMA_VERSION = "tdcsim_scenario_family_receipt_v1"
FED_INTEREST_REMITTANCE_SCHEMA_VERSION = "tdcsim_scenario_family_fed_interest_remittances_fy_v1"
PAYMENT_AGGREGATES_FILE = "tdcsim_period_payment_aggregates.csv.gz"
RUN_MANIFEST_FILE = "tdcsim_cbo_run_manifest.json"
CURVE_SURFACE = Path("compile/compiled/forecast_inputs/tdcsim_yield_curve_surface.csv")
ANNUAL_COLUMNS = (
    "tdc_change_bil",
    "cumulative_tdc_change_bil",
    "interest_outlay_bil",
    "modeled_financing_cost_bil",
    "cumulative_modeled_financing_cost_bil",
    "new_issuance_wam_years",
    "new_issuance_bill_share",
    "outstanding_controlled_wam_years",
)
HOLDING_COLUMNS = {
    "DebtHeld_Banks": "holdings_banks_end_bil",
    "DebtHeld_CentralBank": "holdings_fed_end_bil",
    "DebtHeld_Foreign": "holdings_foreign_end_bil",
    "DebtHeld_DomesticNonBanks": "holdings_domestic_nonbank_end_bil",
}


class ScenarioFamilyExportError(RuntimeError):
    """A scenario family cannot be exported."""


def _nominal_10y(run_dir: Path) -> pd.Series:
    surface = pd.read_csv(run_dir / CURVE_SURFACE, dtype={"curve_date": str})
    ten = surface[surface["tenor_years"].astype(float) == 10.0]
    if ten.empty or ten["curve_date"].duplicated().any():
        raise ScenarioFamilyExportError(f"{run_dir}: compiled curve lacks a unique 10-year point per date")
    return ten.set_index("curve_date")["nominal_rate"].astype(float).sort_index()


def _fed_interest_remittance_rows(
    payments: pd.DataFrame, results: pd.DataFrame, fy_rows: list[dict[str, Any]]
) -> list[dict[str, float]]:
    # Bill discount is paid within principal at maturity. The interest payment
    # types count that component once and exclude TIPS principal indexation.
    fed_payments = payments.loc[
        payments["holder_sector"].eq("CB") & payments["payment_type"].isin(INTEREST_PAYMENT_TYPES)
    ]
    payment_dates = pd.to_datetime(fed_payments["period_end"], errors="raise")
    result_dates = pd.to_datetime(results.index, errors="raise")
    remittances = pd.to_numeric(results["CBORemittanceCashEffect"], errors="raise")
    if not remittances.map(math.isfinite).all():
        raise ScenarioFamilyExportError("Fed remittance cash contains missing or nonfinite values")
    cash = pd.to_numeric(fed_payments["amount_bil"], errors="raise")
    if not cash.map(math.isfinite).all():
        raise ScenarioFamilyExportError("Treasury interest paid to the Fed contains missing or nonfinite values")
    rows = []
    for period in fy_rows:
        start = pd.Timestamp(period["coverage_start_exclusive"])
        end = pd.Timestamp(period["coverage_end_inclusive"])
        interest = float(cash[(payment_dates > start) & (payment_dates <= end)].sum())
        remitted = float(remittances[(result_dates > start) & (result_dates <= end)].sum())
        rows.append({
            "treasury_interest_paid_to_fed_bil": interest,
            "fed_remittances_to_tga_bil": remitted,
            "treasury_interest_less_fed_remittances_bil": interest - remitted,
        })
    return rows


def _role_rows(
    run_dir: Path, *, family_id: str, role: str, scenario_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    manifest_path = run_dir / RUN_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ScenarioFamilyExportError(f"{role}: run manifest is not complete")
    outputs = run_dir / "outputs"
    fy_rows, evidence = trc_fiscal_year_rows(outputs, label=role)
    annual = read_output_csv(outputs / "tdcsim_annual_economic_summary.csv.gz").set_index("period_label")
    results = read_output_csv(outputs / "results_compact.csv.gz").set_index("Date")
    payments_path = outputs / PAYMENT_AGGREGATES_FILE
    fed_flows = _fed_interest_remittance_rows(read_output_csv(payments_path), results, fy_rows)
    missing = [column for column in HOLDING_COLUMNS if column not in results.columns]
    if missing:
        raise ScenarioFamilyExportError(f"{role}: results lack holder stocks {missing}")
    nominal_10y = _nominal_10y(run_dir)
    rows = []
    fed_rows = []
    for fy_row, fed_flow in zip(fy_rows, fed_flows, strict=True):
        label = fy_row["period_label"]
        if label not in annual.index:
            raise ScenarioFamilyExportError(f"{role}: annual summary lacks {label}")
        end = fy_row["coverage_end_inclusive"]
        if end not in results.index:
            raise ScenarioFamilyExportError(f"{role}: results lack the {end} period end")
        row: dict[str, Any] = {
            "schema_version": FAMILY_FY_SCHEMA_VERSION,
            "clock_id": OPEN04_AGGREGATION_CLOCK_ID,
            "family_id": family_id,
            "scenario_role": role,
            "scenario_id": scenario_id,
            "claim_status": "illustrative_scenario_not_forecast_not_causal",
        }
        row.update({key: fy_row[key] for key in (
            "period_label", "period_status", "coverage_start_exclusive", "coverage_end_inclusive",
            "cumulative_origin_date",
        )})
        fed_rows.append({**row, "schema_version": FED_INTEREST_REMITTANCE_SCHEMA_VERSION, **fed_flow})
        for column in ANNUAL_COLUMNS:
            value = annual.loc[label, column]
            row[column] = float(value) if pd.notna(value) else None
        row.update({key: value for key, value in fy_row.items() if key not in row})
        for source, target in HOLDING_COLUMNS.items():
            row[target] = float(results.loc[end, source])
        in_effect = nominal_10y[nominal_10y.index <= str(end)[:10]]
        if in_effect.empty:
            raise ScenarioFamilyExportError(f"{role}: no curve date precedes {end}")
        row["nominal_10y_end_pct"] = float(in_effect.iloc[-1])
        rows.append(row)
    evidence.update(
        {
            "run_manifest_sha256": sha256_path(manifest_path),
            "results_compact_sha256": sha256_path(outputs / "results_compact.csv.gz"),
            "annual_summary_sha256": sha256_path(outputs / "tdcsim_annual_economic_summary.csv.gz"),
            "compiled_curve_sha256": sha256_path(run_dir / CURVE_SURFACE),
            "period_payment_aggregates_sha256": sha256_path(payments_path),
        }
    )
    return rows, fed_rows, evidence, results


def _holder_stock_rows(fy_rows: list[dict[str, Any]], results: pd.DataFrame, *, controlled: bool) -> list[dict[str, Any]]:
    trade_columns = ("BoughtFace", "SoldFace", "BoughtCash", "SoldCash")
    rows = []
    for fy in fy_rows:
        start, end = (fy[key] for key in ("coverage_start_exclusive", "coverage_end_inclusive"))
        window = results.loc[(results.index > start) & (results.index <= end)]
        for route, label in PARTICIPANT_LABELS.items():
            rows.append({
                **{key: fy[key] for key in (
                    "family_id", "scenario_role", "scenario_id", "period_label", "period_status",
                    "coverage_start_exclusive", "coverage_end_inclusive",
                )},
                "holder_type": route[0],
                "holder_subbucket": route[1],
                "closing_bill_stock_bil": float(results.loc[end, STOCK_COLUMNS["HolderBillStock"][route]]),
                "closing_controlled_treasury_stock_bil": float(results.loc[end, STOCK_COLUMNS["HolderStock"][route]]),
                **{
                    f"controller_{name}_bil": float(window[f"HolderTarget{stem}_{label}"].sum(skipna=False))
                    if controlled else 0.0
                    for name, stem in zip(("bought_face", "sold_face", "bought_cash", "sold_cash"), trade_columns)
                },
            })
    return rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ScenarioFamilyExportError(f"nonfinite export value {key}")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def export_scenario_family(
    *,
    family_spec: Mapping[str, Any],
    family_spec_sha256: str,
    runs: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the family fiscal-year tables and their receipt for verified runs."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ScenarioFamilyExportError(f"output directory already exists: {output_dir}")
    family_id = str(family_spec["family_id"])
    expected_roles = [str(item["role"]) for item in family_spec["scenarios"]]
    if list(runs) != expected_roles:
        raise ScenarioFamilyExportError(f"runs {list(runs)} differ from the family roles {expected_roles}")
    rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    fed_rows: list[dict[str, Any]] = []
    roles: dict[str, Any] = {}
    for role, record in runs.items():
        verification = record.get("verification")
        if not isinstance(verification, Mapping) or verification.get("status") != "pass":
            raise ScenarioFamilyExportError(f"{role}: bounded replay verification did not pass")
        role_rows, role_fed_rows, evidence, results = _role_rows(
            Path(record["run_dir"]), family_id=family_id, role=role, scenario_id=str(record["scenario_id"])
        )
        rows.extend(role_rows)
        fed_rows.extend(role_fed_rows)
        if family_spec.get("export_holder_stocks"):
            stock_rows.extend(_holder_stock_rows(
                role_rows, results, controlled="HolderTargetPhase" in results.columns,
            ))
        roles[role] = {
            "scenario_id": record["scenario_id"],
            "scenario_sha256": record["scenario_sha256"],
            "verification_status": verification.get("status"),
            "verification_grade": verification.get("verification_grade"),
            **evidence,
        }
    table_bytes = _csv_bytes(rows)
    fed_table_bytes = _csv_bytes(fed_rows)
    stock_bytes = _csv_bytes(stock_rows) if stock_rows else None
    output_dir.mkdir(parents=True)
    table_path = output_dir / "scenario_family_fy.csv"
    table_path.write_bytes(table_bytes)
    fed_table_path = output_dir / "scenario_family_fed_interest_remittances_fy.csv"
    fed_table_path.write_bytes(fed_table_bytes)
    receipt = {
        "schema_version": FAMILY_RECEIPT_SCHEMA_VERSION,
        "family_id": family_id,
        "family_spec_sha256": family_spec_sha256,
        "producer": dict(producer),
        "clock_id": OPEN04_AGGREGATION_CLOCK_ID,
        "output": {"path": table_path.name, "sha256": hashlib.sha256(table_bytes).hexdigest(), "rows": len(rows)},
        "fed_interest_remittances_output": {
            "path": fed_table_path.name,
            "sha256": hashlib.sha256(fed_table_bytes).hexdigest(),
            "rows": len(fed_rows),
        },
        "roles": roles,
        "boundaries": list(family_spec.get("boundaries", [])),
    }
    if stock_bytes is not None:
        stock_path = output_dir / "scenario_family_holder_stocks_fy.csv"
        stock_path.write_bytes(stock_bytes)
        receipt["holder_stock_output"] = {
            "path": stock_path.name, "sha256": hashlib.sha256(stock_bytes).hexdigest(), "rows": len(stock_rows),
        }
    (output_dir / "scenario_family_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt
