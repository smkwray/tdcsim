"""Default holder preferences for new issuance in the CBO baseline package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tdc_shared import (
    HOLDER_TYPES,
    MARKETABLE_PREFERENCE_CATEGORIES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)

# Value kept for byte-stable holder-profile files in existing packages.
SCHEMA_VERSION = "tdcsim_cbo_opening_portfolio_v1"


def default_cbo_holder_preferences(*, include_private_routes: bool = True) -> dict[str, Any]:
    """Return nondegenerate auction/holder preferences using TDCSIM holder names."""

    preferences: dict[str, Any] = {
        "Banks": {
            "bills_pct": 0.17647058823529413,
            "notes_pct": 0.2105263157894737,
            "bonds_pct": 0.10526315789473685,
            "tips_pct": 0.11111111111111112,
            "frn_pct": 0.31578947368421056,
            "nonmarketable_pct": 0.0,
        },
        "Private": {
            "bills_pct": 0.5294117647058824,
            "notes_pct": 0.5263157894736842,
            "bonds_pct": 0.631578947368421,
            "tips_pct": 0.3888888888888889,
            "frn_pct": 0.4736842105263158,
            "nonmarketable_pct": 0.0,
        },
        "CB": {
            "bills_pct": 0.0,
            "notes_pct": 0.0,
            "bonds_pct": 0.0,
            "tips_pct": 0.0,
            "frn_pct": 0.0,
            "nonmarketable_pct": 0.0,
        },
        "Foreign": {
            "bills_pct": 0.29411764705882354,
            "notes_pct": 0.2631578947368421,
            "bonds_pct": 0.2631578947368421,
            "tips_pct": 0.5,
            "frn_pct": 0.2105263157894737,
            "nonmarketable_pct": 0.0,
        },
        "FedInternal": {
            "bills_pct": 0.0,
            "notes_pct": 0.0,
            "bonds_pct": 0.0,
            "tips_pct": 0.0,
            "frn_pct": 0.0,
            "nonmarketable_pct": 0.0,
        },
        "TrustFunds": {
            "bills_pct": 0.0,
            "notes_pct": 0.0,
            "bonds_pct": 0.0,
            "tips_pct": 0.0,
            "frn_pct": 0.0,
            "nonmarketable_pct": 0.0,
        },
    }
    if include_private_routes:
        preferences["__private_subbucket_shares__"] = default_private_subbucket_shares()
    return preferences


def default_private_subbucket_shares() -> dict[str, dict[str, float]]:
    """Return only Private subbucket routes supported by the current engine."""

    return {
        "bills": {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 0.75, PRIVATE_SUBBUCKET_MMF: 0.25},
        "notes": {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 0.90, PRIVATE_SUBBUCKET_MMF: 0.10},
        "bonds": {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 0.95, PRIVATE_SUBBUCKET_MMF: 0.05},
        "tips": {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 0.95, PRIVATE_SUBBUCKET_MMF: 0.05},
        "frn": {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 0.80, PRIVATE_SUBBUCKET_MMF: 0.20},
    }


def build_holder_profile_rows(
    *,
    scenario_id: str = "cbo_baseline",
    holder_preferences: Mapping[str, Mapping[str, float]] | None = None,
    private_subbucket_shares: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Return a compact row form for the holder preference/profile assumptions."""

    preferences = holder_preferences or default_cbo_holder_preferences(include_private_routes=False)
    route_shares = private_subbucket_shares or default_private_subbucket_shares()
    rows = []
    for holder in HOLDER_TYPES:
        holder_prefs = preferences.get(holder, {})
        row = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "holder_type": holder,
            "holder_subbucket": "",
            "source_role": "scenario_assumption",
            "runtime_role": "memo_only",
            "claim_boundary": "holder preference profile not exact holder ownership",
        }
        for category in MARKETABLE_PREFERENCE_CATEGORIES:
            row[f"{category}_pct"] = float(holder_prefs.get(f"{category}_pct", 0.0) or 0.0)
        rows.append(row)
    for subbucket in (PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, PRIVATE_SUBBUCKET_MMF):
        row = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "holder_type": "Private",
            "holder_subbucket": subbucket,
            "source_role": "scenario_assumption",
            "runtime_role": "memo_only",
            "claim_boundary": "private route split uses only current engine-supported subbuckets",
        }
        for category in MARKETABLE_PREFERENCE_CATEGORIES:
            row[f"{category}_route_share"] = float(route_shares.get(category, {}).get(subbucket, 0.0) or 0.0)
        rows.append(row)
    return rows


def validate_holder_preferences_non_degenerate(
    holder_preferences: Mapping[str, Mapping[str, float]],
    *,
    categories: Sequence[str] = MARKETABLE_PREFERENCE_CATEGORIES,
) -> None:
    """Fail when marketable category shares are all-private or do not sum to one."""

    public_holders = [holder for holder in HOLDER_TYPES if holder not in {"FedInternal", "TrustFunds"}]
    for category in categories:
        key = f"{category}_pct"
        shares = {holder: float(holder_preferences.get(holder, {}).get(key, 0.0) or 0.0) for holder in HOLDER_TYPES}
        total = sum(shares.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"holder preferences for {category} must sum to 1.0, got {total:.12f}")
        positive_public_holders = [holder for holder in public_holders if shares[holder] > 0.0]
        if len(positive_public_holders) < 2:
            raise ValueError(f"holder preferences for {category} are degenerate: {positive_public_holders}")
        if shares["FedInternal"] != 0.0 or shares["TrustFunds"] != 0.0:
            raise ValueError(f"intragovernmental holders must not absorb public marketable {category}")
