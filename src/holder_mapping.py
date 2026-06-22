"""Holder-to-RateWall perimeter mapping for contract exports."""

from __future__ import annotations

DEFAULT_HOLDER_PERIMETER = {
    "Private": "DU",
    "Banks": "RU",
    "CB": "RU",
    "Foreign": "RU",
    "FedInternal": "intragov",
    "TrustFunds": "intragov",
}

HOLDER_ROUTE_CONTRACT = [
    {
        "source_key": "Private",
        "ratewall_role": "domestic_nonbank_undifferentiated_current_contract",
        "central_default_eligible": "true",
        "sensitivity_only": "false",
        "binding_blocker": "",
    },
    {
        "source_key": "domestic_nonbank_non_deposit_funded",
        "ratewall_role": "target_route_not_current_holder_type",
        "central_default_eligible": "false",
        "sensitivity_only": "true",
        "binding_blocker": (
            "requires_source_backed_split_from_current_private_holder_bucket"
        ),
    },
    {
        "source_key": "mmf_cash_fund_route",
        "ratewall_role": "source_backed_mmf_cash_fund_route_static_0_97_deposit_weight",
        "central_default_eligible": "true",
        "sensitivity_only": "false",
        "binding_blocker": (
            "cleared_for_static_central_split_by_owner_t4_5;"
            "justification=on_rrp_drained_regime_tdcest_2025q4_source_of_funds_anchor_0_986_rounded_0_97;"
            "known_limitation=sec_nmfp_fund_scope_does_not_identify_final_deposit_recipient"
        ),
    },
]


def holder_perimeter(holder: str, mapping: dict[str, str] | None = None) -> str:
    """Return the RateWall perimeter bucket for a tdcsim holder."""

    if mapping and holder in mapping:
        return str(mapping[holder])
    return DEFAULT_HOLDER_PERIMETER.get(str(holder), "blocked")


def mapping_rows(mapping: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Return a stable source-registry representation of the perimeter map."""

    rows = []
    merged = dict(DEFAULT_HOLDER_PERIMETER)
    if mapping:
        merged.update({str(k): str(v) for k, v in mapping.items()})
    for holder, perimeter in sorted(merged.items()):
        central_default_allowed = "true" if perimeter in {"DU", "RU", "intragov"} else "false"
        rows.append(
            {
                "source_family": "holder_mapping",
                "source_key": holder,
                "source_status": "contract_mapping_present",
                "ratewall_role": perimeter,
                "central_default_eligible": central_default_allowed,
                "sensitivity_only": "false" if central_default_allowed == "true" else "true",
                "binding_blocker": ""
                if central_default_allowed == "true"
                else "holder_bucket_blocked_or_plumbing_not_central_default",
            }
        )
    rows.extend(
        {
            "source_family": "holder_route_contract",
            "source_key": route["source_key"],
            "source_status": "route_contract_present",
            "ratewall_role": route["ratewall_role"],
            "central_default_eligible": route["central_default_eligible"],
            "sensitivity_only": route["sensitivity_only"],
            "binding_blocker": route["binding_blocker"],
        }
        for route in HOLDER_ROUTE_CONTRACT
    )
    return rows
