from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ratewall_assumption_mode_schema import validate_route_support_rows


REGISTRY_VERSION = "tdcsim_route_component_support_registry_v1"

DEFAULT_INPUTS = {
    "tdcest_private_route_support": Path(
        "../tdcest/data/processed/tdc_tdcsim_private_route_support_contract.csv"
    ),
    "rowflow_foreign_route_support": Path(
        "../rowflow/output/contracts/rowflow_ratewall_foreign_route_support.csv"
    ),
    "buycurve_auction_buyer_support": Path(
        "../buycurve/output/contracts/buycurve_ratewall_auction_buyer_mix_support.csv"
    ),
    "bidbridge_dealer_warehousing_support": Path(
        "../bidbridge/outputs/tables/bidbridge_ratewall_dealer_warehousing_support.csv"
    ),
}

REGISTRY_FIELDS = [
    "registry_row_id",
    "registry_version",
    "source_support_row_id",
    "producer_project",
    "producer_artifact",
    "route_component_id",
    "normalized_route_component_id",
    "object_family",
    "measurement_stage",
    "evidence_tier",
    "mapping_burden",
    "assumption_status",
    "admissible_use",
    "blocked_use",
    "source_backed_private_bucket_split_status",
    "source_backed_private_bucket_split_row",
    "canonical_tdc_math_change",
    "evidence_mode_enabled",
    "current_demand_eligible",
    "holder_allocation_enabled",
    "pricing_output_enabled",
    "incidence_claim_enabled",
    "welfare_claim_enabled",
    "tax_output_enabled",
    "mpc_output_enabled",
    "prior_narrowing_allowed",
    "period_start",
    "period_end",
    "ref_quarter",
    "amount",
    "share",
    "claim_boundary",
]

FALSE_GUARDRAILS = {
    "source_backed_private_bucket_split_row",
    "canonical_tdc_math_change",
    "evidence_mode_enabled",
    "current_demand_eligible",
    "holder_allocation_enabled",
    "pricing_output_enabled",
    "incidence_claim_enabled",
    "welfare_claim_enabled",
    "tax_output_enabled",
    "mpc_output_enabled",
    "prior_narrowing_allowed",
}

VERDICT_COMPONENTS = [
    "mmf_onrrp_like",
    "foreign_official_private_iro",
    "auction_buyer_mix",
    "dealer_warehousing",
    "bank_absorption",
    "domestic_nonbank_ex_mmf",
    "residual_unidentified",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_existing(inputs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key, path in inputs.items():
        if path.exists():
            frames[key] = pd.read_csv(path)
    return frames


def _normalized_component(source_key: str, row: pd.Series) -> str:
    route = str(row.get("route_component_id", ""))
    route_class = str(row.get("route_class", ""))
    if source_key == "rowflow_foreign_route_support":
        return "foreign_official_private_iro"
    if source_key == "buycurve_auction_buyer_support":
        return "auction_buyer_mix"
    if source_key == "bidbridge_dealer_warehousing_support":
        return "dealer_warehousing"
    if "mmf" in route_class or "mmf" in route:
        return "mmf_onrrp_like"
    if "non_deposit_funded_domestic_nonbank_ex_mmf" in route_class:
        return "domestic_nonbank_ex_mmf"
    if "deposit_funded_domestic_nonbank_possible" in route_class:
        return "domestic_nonbank_ex_mmf"
    return route or "residual_unidentified"


def _value(row: pd.Series, *columns: str) -> str:
    for column in columns:
        value = row.get(column, "")
        if value not in ("", None) and not pd.isna(value):
            return str(value)
    return ""


def _normalize_rows(source_key: str, frame: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        support_id = _value(row, "support_row_id", "allocation_row_id")
        normalized_component = _normalized_component(source_key, row)
        out = {
            "registry_row_id": f"tdcsim_support::{source_key}::{support_id}",
            "registry_version": REGISTRY_VERSION,
            "source_support_row_id": support_id,
            "producer_project": _value(row, "producer_project", "owner_project"),
            "producer_artifact": _value(row, "producer_artifact", "source_artifact"),
            "route_component_id": _value(row, "route_component_id", "route_class"),
            "normalized_route_component_id": normalized_component,
            "object_family": _value(row, "object_family"),
            "measurement_stage": _value(row, "measurement_stage"),
            "evidence_tier": _value(row, "evidence_tier"),
            "mapping_burden": _value(row, "mapping_burden"),
            "assumption_status": _value(row, "assumption_status"),
            "admissible_use": _value(row, "admissible_use", "allowed_use"),
            "blocked_use": _value(row, "blocked_use"),
            "source_backed_private_bucket_split_status": _value(
                row,
                "source_backed_private_bucket_split_status",
            )
            or "not_source_backed_private_bucket_split",
            "period_start": _value(row, "period_start", "month", "week_start"),
            "period_end": _value(row, "period_end", "month", "week_end"),
            "ref_quarter": _value(row, "ref_quarter"),
            "amount": _value(
                row,
                "amount_bil",
                "amount_usd_millions",
                "allotment_amount",
                "awarded_amount_total",
            ),
            "share": _value(
                row,
                "share_central",
                "share_of_foreign_total",
                "allotment_share",
                "dealer_allotment_share",
            ),
            "claim_boundary": _value(row, "claim_boundary"),
        }
        for field in FALSE_GUARDRAILS:
            out[field] = str(row.get(field, "false")).lower()
        rows.append(out)
    return rows


def build_support_registry(
    inputs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    input_paths = inputs or DEFAULT_INPUTS
    frames = _read_existing(input_paths)
    rows: list[dict[str, str]] = []
    for key, frame in frames.items():
        rows.extend(_normalize_rows(key, frame))
    registry = pd.DataFrame(rows, columns=REGISTRY_FIELDS)
    report = validate_route_support_rows(
        registry.rename(columns={"registry_row_id": "support_row_id"})
    )
    manifest = {
        "registry_version": REGISTRY_VERSION,
        "row_count": int(len(registry)),
        "validation": {
            "validation_status": report.validation_status,
            "row_count": report.row_count,
            "failure_reasons": list(report.failure_reasons),
        },
        "inputs": {
            key: {
                "path": str(path),
                "present": path.exists(),
                "sha256": _sha256(path) if path.exists() else "",
            }
            for key, path in input_paths.items()
        },
    }
    return registry, manifest


def build_route_component_verdict(registry: pd.DataFrame) -> pd.DataFrame:
    counts = registry["normalized_route_component_id"].value_counts().to_dict()
    verdicts = {
        "mmf_onrrp_like": (
            "source_backed_measurement",
            (
                "MMF deposit weight promoted only for the static central private-bucket "
                "split: ON-RRP-drained regime plus tdcest 2025Q4 source-of-funds "
                "anchor around 0.986, rounded conservatively to 0.97. Known limitation: "
                "SEC N-MFP fund scope does not identify the final deposit recipient."
            ),
        ),
        "foreign_official_private_iro": (
            "source_backed_measurement",
            "Rowflow supports foreign absorption accounting; domestic effects remain blocked.",
        ),
        "auction_buyer_mix": (
            "direct_input",
            "Buycurve directly measures initial auction allotment, not final holder funding.",
        ),
        "dealer_warehousing": (
            "bounded_proxy",
            "Bidbridge bounds transient dealer warehousing, not terminal ownership.",
        ),
        "bank_absorption": (
            "context_only",
            "No bank-level route contract is admitted in v1.",
        ),
        "domestic_nonbank_ex_mmf": (
            "unresolved_residual",
            "Domestic nonbank ex-MMF remains unresolved beyond bounded lambda support.",
        ),
        "residual_unidentified": (
            "unresolved_residual",
            "Residual unidentified route must remain explicit and non-promoting.",
        ),
    }
    rows = []
    for component in VERDICT_COMPONENTS:
        tier, boundary = verdicts[component]
        if component == "mmf_onrrp_like":
            admissible_use = "canonical_tdcsim_input;assumption_mode_support_ledger"
            blocked_use = "evidence_mode;final_current_demand;holder_allocation"
        else:
            admissible_use = "assumption_mode_support_ledger"
            blocked_use = (
                "source_backed_private_bucket_split;canonical_tdc_math;"
                "evidence_mode;final_current_demand;holder_allocation"
            )
        rows.append(
            {
                "route_component_id": component,
                "support_row_count": str(int(counts.get(component, 0))),
                "verdict_tier": tier,
                "admissible_use": admissible_use,
                "blocked_use": blocked_use,
                "claim_boundary": boundary,
            }
        )
    return pd.DataFrame(rows)


def write_support_registry_bundle(
    *,
    output_dir: Path | str = Path("output/ratewall_contract_assumption_mode"),
    inputs: dict[str, Path] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry, manifest = build_support_registry(inputs)
    verdict = build_route_component_verdict(registry)
    registry_path = output_dir / "tdcsim_route_component_support_registry.csv"
    verdict_path = output_dir / "tdcsim_route_component_verdict.csv"
    manifest_path = output_dir / "tdcsim_assumption_mode_manifest.json"
    registry.to_csv(registry_path, index=False)
    verdict.to_csv(verdict_path, index=False)
    manifest["outputs"] = {
        "support_registry": registry_path.name,
        "route_component_verdict": verdict_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "support_registry": registry_path,
        "route_component_verdict": verdict_path,
        "manifest": manifest_path,
    }
