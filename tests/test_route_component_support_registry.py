from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from route_component_support_registry import (
    build_route_component_verdict,
    build_support_registry,
    write_support_registry_bundle,
)


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_support_registry_validates_mixed_support_rows(tmp_path: Path) -> None:
    inputs = {
        "tdcest_private_route_support": _write(
            tmp_path / "tdcest.csv",
            [
                {
                    "support_row_id": "tdcest1",
                    "producer_project": "tdcest",
                    "producer_artifact": "tdcest.csv",
                    "route_component_id": "tdcsim_private_bucket_route_sensitivity",
                    "route_class": "mmf_onrrp_like_intermediated",
                    "object_family": "stock_interest_quarter_end",
                    "measurement_stage": "holder_stock",
                    "evidence_tier": "bounded_proxy",
                    "mapping_burden": "requires_unobserved_actor_split",
                    "assumption_status": "bounded_assumption",
                    "admissible_use": "assumption_mode_support_ledger",
                    "blocked_use": "source_backed_private_bucket_split",
                    "source_backed_private_bucket_split_status": "not_source_backed_private_bucket_split",
                }
            ],
        ),
        "rowflow_foreign_route_support": _write(
            tmp_path / "rowflow.csv",
            [
                {
                    "support_row_id": "rowflow1",
                    "producer_project": "rowflow",
                    "producer_artifact": "rowflow.csv",
                    "route_component_id": "foreign_official",
                    "object_family": "foreign_treasury_absorption",
                    "measurement_stage": "foreign_holder_flow",
                    "evidence_tier": "source_backed_measurement",
                    "mapping_burden": "mechanical_aggregation_only",
                    "assumption_status": "none_source_observed",
                    "admissible_use": "assumption_mode_support_ledger",
                    "blocked_use": "final_current_demand",
                }
            ],
        ),
    }

    registry, manifest = build_support_registry(inputs)

    assert len(registry) == 2
    assert manifest["validation"]["validation_status"] == "pass"
    assert set(registry["normalized_route_component_id"]) == {
        "mmf_onrrp_like",
        "foreign_official_private_iro",
    }
    assert set(registry["evidence_mode_enabled"]) == {"false"}


def test_route_component_verdict_has_expected_components(tmp_path: Path) -> None:
    inputs = {
        "buycurve_auction_buyer_support": _write(
            tmp_path / "buycurve.csv",
            [
                {
                    "support_row_id": "buycurve1",
                    "producer_project": "buycurve",
                    "producer_artifact": "buycurve.csv",
                    "route_component_id": "auction_primary_allocation",
                    "object_family": "auction_primary_market_allotment",
                    "measurement_stage": "auction_initial",
                    "evidence_tier": "direct_input",
                    "mapping_burden": "requires_terminal_holder_inference",
                    "assumption_status": "none_source_observed",
                    "admissible_use": "assumption_mode_support_ledger",
                    "blocked_use": "final_holder_split",
                }
            ],
        )
    }
    registry, _ = build_support_registry(inputs)
    verdict = build_route_component_verdict(registry)

    assert set(verdict["route_component_id"]) == {
        "mmf_onrrp_like",
        "foreign_official_private_iro",
        "auction_buyer_mix",
        "dealer_warehousing",
        "bank_absorption",
        "domestic_nonbank_ex_mmf",
        "residual_unidentified",
    }
    auction = verdict.loc[verdict["route_component_id"].eq("auction_buyer_mix")].iloc[0]
    assert auction["support_row_count"] == "1"
    assert auction["verdict_tier"] == "direct_input"


def test_write_support_registry_bundle_hashes_inputs(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "bidbridge.csv",
        [
            {
                "support_row_id": "bidbridge1",
                "producer_project": "bidbridge",
                "producer_artifact": "bidbridge.csv",
                "route_component_id": "dealer_warehousing",
                "object_family": "dealer_inventory_financing_bridge",
                "measurement_stage": "transient_bridge",
                "evidence_tier": "bounded_proxy",
                "mapping_burden": "requires_terminal_holder_inference",
                "assumption_status": "bounded_by_observed_context",
                "admissible_use": "assumption_mode_support_ledger",
                "blocked_use": "final_holder_split",
                "source_backed_private_bucket_split_status": "not_source_backed_private_bucket_split",
            }
        ],
    )
    paths = write_support_registry_bundle(
        output_dir=tmp_path / "out",
        inputs={"bidbridge_dealer_warehousing_support": source},
    )

    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    assert payload["inputs"]["bidbridge_dealer_warehousing_support"]["sha256"]
    assert paths["support_registry"].exists()
    assert paths["route_component_verdict"].exists()
