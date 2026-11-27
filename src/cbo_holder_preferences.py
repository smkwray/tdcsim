"""Holder shares of new issuance in the CBO baseline package.

New issuance is divided among non-Fed holders in proportion to their holdings in the
opening book: bills by holdings of bills, every other security type by holdings of
longer securities. Those weights come from WAMEST and are recorded in the opening-book
manifest. Floating-rate notes are the exception: money funds take the share of new FRN
supply calibrated from SEC N-MFP filings, also recorded there, and the other non-Fed
holders share the rest in their WAMEST proportions. The money-fund share of ineligible
issuance is redistributed to the other non-Fed holders. The Federal Reserve follows its
own stock path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

from issuance_calibration import BILL_TERM_WEEKS, COUPON_KIND_SECURITY_TYPE, COUPON_TENORS
from mmf_eligibility import mmf_may_acquire
from sim_pricing import get_security_category_for_prefs
from tdc_shared import (
    FRN_MMF_AUCTION_SHARE_KEY,
    HOLDER_TYPES,
    MARKETABLE_PREFERENCE_CATEGORIES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)

SCHEMA_VERSION = "tdcsim_cbo_holder_profile_v2"
_DOMESTIC_NONBANK_KEY = f"Private/{PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}"
_MMF_KEY = f"Private/{PRIVATE_SUBBUCKET_MMF}"
_WEIGHT_KEYS = ("Banks", "Foreign", _DOMESTIC_NONBANK_KEY, _MMF_KEY)


def _eligible_auction_categories() -> set[str]:
    # The baseline profile applies one share per category across the registered original
    # terms. Actual settlement-to-maturity dates are checked again when each lot issues.
    issue = date(2000, 1, 1)
    category_eligibility = {"bills": [mmf_may_acquire("Fixed", issue + timedelta(weeks=max(BILL_TERM_WEEKS)), issue)]}
    for kind, tenors in COUPON_TENORS.items():
        security_type = COUPON_KIND_SECURITY_TYPE[kind]
        for tenor in tenors:
            category = get_security_category_for_prefs(security_type, tenor, {})
            category_eligibility.setdefault(category, []).append(
                mmf_may_acquire(security_type, issue.replace(year=issue.year + tenor), issue)
            )
    return {category for category, eligible in category_eligibility.items() if all(eligible)}


def _with_mmf_share(shares: Mapping[str, float], mmf_share: float) -> dict[str, float]:
    """Give money funds ``mmf_share`` and scale the other holders pro rata to the rest."""

    others = sum(value for key, value in shares.items() if key != _MMF_KEY)
    if others <= 0.0:
        raise ValueError("opening-book noncb_weights give FRNs no holders besides money funds")
    return {key: mmf_share if key == _MMF_KEY else value * (1.0 - mmf_share) / others for key, value in shares.items()}


def holder_profile_from_opening_weights(
    noncb_weights: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return (holder preferences, private subbucket shares) from opening-book weights.

    ``noncb_weights`` maps ``bills`` and ``nonbills`` to shares keyed ``Banks``,
    ``Foreign``, ``Private/domestic_nonbank_deposit_funded`` and
    ``Private/mmf_cash_fund_route``, and ``frn_mmf_auction_share`` to money funds'
    calibrated share of new FRNs, as written by ``ratewall_input_builder``.
    """

    groups = {}
    for group in ("bills", "nonbills"):
        shares = noncb_weights.get(group)
        if not isinstance(shares, Mapping) or set(shares) != set(_WEIGHT_KEYS):
            raise ValueError(f"opening-book noncb_weights[{group!r}] must have exactly {list(_WEIGHT_KEYS)}")
        values = {key: float(shares[key]) for key in _WEIGHT_KEYS}
        if any(value < 0.0 for value in values.values()) or abs(sum(values.values()) - 1.0) > 1e-9:
            raise ValueError(f"opening-book noncb_weights[{group!r}] must be nonnegative and sum to 1")
        groups[group] = values
    mmf_frn_share = noncb_weights.get(FRN_MMF_AUCTION_SHARE_KEY)
    if isinstance(mmf_frn_share, bool) or not isinstance(mmf_frn_share, (int, float)) or not 0.0 < mmf_frn_share < 1.0:
        raise ValueError(f"opening-book noncb_weights[{FRN_MMF_AUCTION_SHARE_KEY!r}] must be a share between 0 and 1")

    preferences: dict[str, dict[str, float]] = {
        holder: {f"{category}_pct": 0.0 for category in (*MARKETABLE_PREFERENCE_CATEGORIES, "nonmarketable")}
        for holder in HOLDER_TYPES
    }
    private_subbucket_shares: dict[str, dict[str, float]] = {}
    eligible_categories = _eligible_auction_categories()
    for category in MARKETABLE_PREFERENCE_CATEGORIES:
        shares = groups["bills" if category == "bills" else "nonbills"]
        if category == "frn":
            shares = _with_mmf_share(shares, float(mmf_frn_share))
        if category not in eligible_categories:
            eligible_weight = sum(shares[key] for key in _WEIGHT_KEYS if key != _MMF_KEY)
            if eligible_weight <= 0.0:
                raise ValueError(f"opening-book noncb_weights give {category} no eligible holders")
            shares = {key: (0.0 if key == _MMF_KEY else value / eligible_weight) for key, value in shares.items()}
        private = shares[_DOMESTIC_NONBANK_KEY] + shares[_MMF_KEY]
        if private <= 0.0:
            raise ValueError(f"opening-book noncb_weights give private holders no {category} share")
        preferences["Banks"][f"{category}_pct"] = shares["Banks"]
        preferences["Foreign"][f"{category}_pct"] = shares["Foreign"]
        preferences["Private"][f"{category}_pct"] = private
        private_subbucket_shares[category] = {
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: shares[_DOMESTIC_NONBANK_KEY] / private,
            PRIVATE_SUBBUCKET_MMF: shares[_MMF_KEY] / private,
        }
    return preferences, private_subbucket_shares


def build_holder_profile_rows(
    *,
    scenario_id: str,
    holder_preferences: Mapping[str, Mapping[str, float]],
    private_subbucket_shares: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Return the holder profile file rows: one per holder plus one per private route."""

    rows = []
    for holder in HOLDER_TYPES:
        holder_prefs = holder_preferences.get(holder, {})
        row = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "holder_type": holder,
            "holder_subbucket": "",
            "source_role": "opening_book_stock_shares",
            "runtime_role": "new_issuance_holder_shares",
            "claim_boundary": (
                "new issuance follows WAMEST bill or nonbill shares among eligible non-Fed holders; "
                "money funds take the SEC N-MFP-calibrated share of new FRNs"
            ),
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
            "source_role": "opening_book_stock_shares",
            "runtime_role": "new_issuance_private_route_shares",
            "claim_boundary": "private route split follows WAMEST holdings among eligible private routes",
        }
        for category in MARKETABLE_PREFERENCE_CATEGORIES:
            row[f"{category}_route_share"] = float(private_subbucket_shares.get(category, {}).get(subbucket, 0.0) or 0.0)
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
