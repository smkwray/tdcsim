"""Thin fiscal-year export for the route-aware OPEN-04 TRC companion campaign.

The companion reruns the frozen three OPEN-04 paths with opt-in TRC routing.
Before projecting TRC, every legacy role output must reproduce the frozen
campaign byte for byte (resource telemetry excepted), so the companion adds
reserve incidence without revising any admitted TDC or cost value.
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


from .open04_campaign import OPEN04_AGGREGATION_CLOCK_ID, OPEN04_CAMPAIGN_ROLES
from .trc_fy import TrcFiscalYearError, sha256_path, trc_fiscal_year_rows

TRC_PATHS_SCHEMA_VERSION = "tdcsim_open04_trc_paths_fy_v1"
TRC_RECEIPT_SCHEMA_VERSION = "tdcsim_open04_trc_companion_receipt_v1"
CONTRACT_FILE = "open04_campaign_contract.json"
TELEMETRY_FILES = frozenset({"tdcsim_resource_samples.csv.gz", "tdcsim_resource_samples.csv"})
TRC_FILES = frozenset({"tdcsim_trc_results.csv.gz", "tdcsim_trc_metadata.json"})


class Open04TrcExportError(RuntimeError):
    """The companion campaign cannot support a TRC projection."""


def _sha256(path: Path) -> str:
    return sha256_path(path)


def _contract(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / CONTRACT_FILE
    if not path.is_file():
        raise Open04TrcExportError(f"campaign contract is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _role_outputs(campaign_root: Path, contract: Mapping[str, Any], role: str) -> Path:
    relative = contract["roles"][role]["run_relative_path"]
    outputs = campaign_root / relative / "outputs"
    if not outputs.is_dir():
        raise Open04TrcExportError(f"{role}: run outputs are missing: {outputs}")
    return outputs


def _legacy_identity(frozen: Path, companion: Path, role: str) -> dict[str, Any]:
    frozen_names = {path.name for path in frozen.iterdir() if path.is_file()}
    companion_names = {path.name for path in companion.iterdir() if path.is_file()}
    if companion_names - frozen_names != TRC_FILES:
        raise Open04TrcExportError(
            f"{role}: companion adds {sorted(companion_names - frozen_names)}, expected {sorted(TRC_FILES)}"
        )
    if frozen_names - companion_names:
        raise Open04TrcExportError(f"{role}: companion lacks {sorted(frozen_names - companion_names)}")
    compared = []
    for name in sorted(frozen_names - TELEMETRY_FILES - {"summary.json"}):
        if _sha256(frozen / name) != _sha256(companion / name):
            raise Open04TrcExportError(f"{role}: legacy output {name} differs from the frozen campaign")
        compared.append(name)
    frozen_summary = json.loads((frozen / "summary.json").read_text(encoding="utf-8"))
    companion_summary = json.loads((companion / "summary.json").read_text(encoding="utf-8"))
    for summary in (frozen_summary, companion_summary):
        summary.pop("peak_rss_bytes", None)
    if frozen_summary != companion_summary:
        raise Open04TrcExportError(f"{role}: summary.json differs beyond peak_rss_bytes")
    return {
        "byte_identical_files": compared,
        "summary_json_identical_except": ["peak_rss_bytes"],
        "telemetry_files_not_compared": sorted(TELEMETRY_FILES & frozen_names),
    }


def _role_rows(
    outputs: Path, *, role: str, scenario_id: str, campaign_id: str, frozen_campaign_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        fy_rows, evidence = trc_fiscal_year_rows(outputs, label=role)
    except TrcFiscalYearError as exc:
        raise Open04TrcExportError(str(exc)) from exc
    rows = []
    for fy_row in fy_rows:
        row = {
            "schema_version": TRC_PATHS_SCHEMA_VERSION,
            "clock_id": OPEN04_AGGREGATION_CLOCK_ID,
            "campaign_id": campaign_id,
            "frozen_campaign_id": frozen_campaign_id,
            "scenario_role": role,
            "scenario_id": scenario_id,
            "claim_status": "scenario_not_forecast_not_causal_not_optimization",
        }
        row.update({key: value for key, value in fy_row.items() if key != "foreign_gross_settlement_cash_bil"})
        rows.append(row)
    return rows, evidence


def export_open04_trc_companion(
    *,
    companion_campaign_root: Path,
    frozen_campaign_root: Path,
    output_dir: Path,
    producer_commit: str,
) -> dict[str, Any]:
    """Verify legacy identity, then write the FY TRC table and its receipt."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Open04TrcExportError(f"output directory already exists: {output_dir}")
    companion = _contract(Path(companion_campaign_root))
    frozen = _contract(Path(frozen_campaign_root))
    rows: list[dict[str, Any]] = []
    roles: dict[str, Any] = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        frozen_outputs = _role_outputs(Path(frozen_campaign_root), frozen, role)
        companion_outputs = _role_outputs(Path(companion_campaign_root), companion, role)
        identity = _legacy_identity(frozen_outputs, companion_outputs, role)
        scenario_id = str(companion["roles"][role].get("scenario_id") or "")
        role_rows, evidence = _role_rows(
            companion_outputs,
            role=role,
            scenario_id=scenario_id,
            campaign_id=companion["campaign_id"],
            frozen_campaign_id=frozen["campaign_id"],
        )
        rows.extend(role_rows)
        roles[role] = {"legacy_identity": identity, **evidence}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise Open04TrcExportError(f"nonfinite export value {key}")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    table_bytes = buffer.getvalue().encode("utf-8")
    output_dir.mkdir(parents=True)
    table_path = output_dir / "open04_trc_paths_fy.csv"
    table_path.write_bytes(table_bytes)
    receipt = {
        "schema_version": TRC_RECEIPT_SCHEMA_VERSION,
        "producer_commit": producer_commit,
        "companion_campaign_id": companion["campaign_id"],
        "companion_contract_sha256": _sha256(Path(companion_campaign_root) / CONTRACT_FILE),
        "frozen_campaign_id": frozen["campaign_id"],
        "frozen_contract_sha256": _sha256(Path(frozen_campaign_root) / CONTRACT_FILE),
        "clock_id": OPEN04_AGGREGATION_CLOCK_ID,
        "output": {"path": table_path.name, "sha256": hashlib.sha256(table_bytes).hexdigest(), "rows": len(rows)},
        "roles": roles,
        "boundaries": [
            "Every legacy role output reproduces the frozen campaign byte for byte except resource telemetry",
            "TRC is Treasury-attributed reserve incidence under stated routing assumptions, not a reserve-stock forecast",
            "Excludes Fed non-Treasury assets, currency, IORB, swaps and non-Treasury ON RRP dynamics",
        ],
    }
    (output_dir / "open04_trc_companion_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return receipt
