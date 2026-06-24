"""Pure CBO opening-portfolio helpers backed by MSPD Table 1 class totals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    HOLDER_TYPES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)


SCHEMA_VERSION = "tdcsim_cbo_opening_portfolio_v1"
MSPD_TABLE_1_SOURCE = "MSPD Table 1"
PUBLIC_MARKETABLE_CLASSES = ("Bills", "Notes", "Bonds", "TIPS", "FRN")
MARKETABLE_PREFERENCE_CATEGORIES = ("bills", "notes", "bonds", "tips", "frn")
CLASS_TO_CATEGORY = {
    "Bills": "bills",
    "Notes": "notes",
    "Bonds": "bonds",
    "TIPS": "tips",
    "FRN": "frn",
}
CLASS_TO_SECURITY_TYPE = {
    "Bills": "Fixed",
    "Notes": "Fixed",
    "Bonds": "Fixed",
    "TIPS": "TIPS",
    "FRN": "FRN",
}

DEFAULT_MATURITY_COUPON_PROFILE = {
    "Bills": (
        {"maturity_years": 4.0 / 52.0, "weight": 0.10, "coupon_rate": 0.0},
        {"maturity_years": 8.0 / 52.0, "weight": 0.10, "coupon_rate": 0.0},
        {"maturity_years": 0.25, "weight": 0.30, "coupon_rate": 0.0},
        {"maturity_years": 0.50, "weight": 0.30, "coupon_rate": 0.0},
        {"maturity_years": 1.00, "weight": 0.20, "coupon_rate": 0.0},
    ),
    "Notes": (
        {"maturity_years": 2.0, "weight": 0.20, "coupon_rate": 0.0350},
        {"maturity_years": 3.0, "weight": 0.20, "coupon_rate": 0.0350},
        {"maturity_years": 5.0, "weight": 0.25, "coupon_rate": 0.0375},
        {"maturity_years": 7.0, "weight": 0.15, "coupon_rate": 0.0375},
        {"maturity_years": 10.0, "weight": 0.20, "coupon_rate": 0.0400},
    ),
    "Bonds": (
        {"maturity_years": 20.0, "weight": 0.35, "coupon_rate": 0.0425},
        {"maturity_years": 30.0, "weight": 0.65, "coupon_rate": 0.0450},
    ),
    "TIPS": (
        {"maturity_years": 5.0, "weight": 0.25, "coupon_rate": 0.0125},
        {"maturity_years": 10.0, "weight": 0.50, "coupon_rate": 0.0150},
        {"maturity_years": 30.0, "weight": 0.25, "coupon_rate": 0.0175},
    ),
    "FRN": (
        {"maturity_years": 2.0, "weight": 1.00, "coupon_rate": 0.0, "fixed_spread": 0.0013},
    ),
}


def extract_mspd_public_marketable_class_totals(
    mspd_table_1_rows: Sequence[Mapping[str, Any]],
    *,
    required_classes: Sequence[str] = PUBLIC_MARKETABLE_CLASSES,
    tolerance_bil: float = 0.001,
) -> dict[str, Any]:
    """Extract MSPD public marketable class totals in billions.

    The amount basis is MSPD Table 1 ``debt_held_public_mil_amt`` when present.
    Public nonmarketable rows are retained as bridge components and never
    returned as auction or opening public-marketable securities.
    """

    class_totals = {class_name: 0.0 for class_name in PUBLIC_MARKETABLE_CLASSES}
    source_selectors: dict[str, list[str]] = {class_name: [] for class_name in PUBLIC_MARKETABLE_CLASSES}
    nonmarketable_bridge_components: list[dict[str, Any]] = []
    total_marketable_bil: float | None = None
    record_dates = set()

    for row in mspd_table_1_rows:
        record_date = _row_value(row, "record_date", "Record Date")
        if record_date not in (None, ""):
            record_dates.add(str(record_date))
        amount_bil = _mspd_public_amount_bil(row)
        row_class = _mspd_public_marketable_class(row)
        selector = _mspd_selector(row)
        text = _row_text(row)
        if "total marketable" in text:
            total_marketable_bil = amount_bil
            continue
        if row_class is None:
            if "nonmarketable" in text:
                nonmarketable_bridge_components.append(
                    {
                        "source_selector": selector,
                        "amount_bil": amount_bil,
                        "source_role": "hard_actual_state",
                        "runtime_role": "reconciliation_only",
                        "claim_boundary": "public_nonmarketables_are_bridge_components_not_auction_securities",
                    }
                )
            continue
        class_totals[row_class] += amount_bil
        source_selectors[row_class].append(selector)

    missing = [class_name for class_name in required_classes if class_totals.get(class_name, 0.0) <= 0.0]
    if missing:
        raise ValueError(f"missing MSPD public marketable class totals: {', '.join(missing)}")
    if len(record_dates) > 1:
        raise ValueError(f"MSPD class-total rows must share record_date: {sorted(record_dates)}")
    if total_marketable_bil is not None:
        class_sum = sum(class_totals.values())
        if abs(class_sum - total_marketable_bil) > tolerance_bil:
            raise ValueError(
                "MSPD public marketable class totals do not reconcile to Total Marketable "
                f"within {tolerance_bil} billion: class_sum={class_sum:.6f}, total={total_marketable_bil:.6f}"
            )

    return {
        "record_date": sorted(record_dates)[0] if record_dates else "",
        "class_totals_bil": class_totals,
        "source_selectors": source_selectors,
        "total_marketable_bil": total_marketable_bil,
        "nonmarketable_bridge_components": nonmarketable_bridge_components,
    }


def build_cbo_opening_public_marketable_portfolio(
    *,
    mspd_table_1_rows: Sequence[Mapping[str, Any]],
    opening_state_date: date | datetime | str,
    starting_bond_id: int = 1,
    scenario_id: str = "cbo_baseline",
    holder_preferences: Mapping[str, Mapping[str, float]] | None = None,
    private_subbucket_shares: Mapping[str, Mapping[str, float]] | None = None,
    maturity_coupon_profile: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
) -> dict[str, Any]:
    """Build an engine-shaped synthetic opening public-marketable portfolio."""

    opening_date = _as_date(opening_state_date)
    extracted = extract_mspd_public_marketable_class_totals(mspd_table_1_rows)
    class_totals = extracted["class_totals_bil"]
    preferences = holder_preferences or default_cbo_holder_preferences(include_private_routes=False)
    route_shares = private_subbucket_shares or default_private_subbucket_shares()
    profile = maturity_coupon_profile or DEFAULT_MATURITY_COUPON_PROFILE
    rows: list[dict[str, Any]] = []
    next_bond_id = int(starting_bond_id)

    for class_name in PUBLIC_MARKETABLE_CLASSES:
        class_total = float(class_totals[class_name])
        category = CLASS_TO_CATEGORY[class_name]
        maturity_rows = _normalized_profile_rows(profile[class_name], class_name)
        class_start_index = len(rows)
        for maturity_row in maturity_rows:
            maturity_amount = class_total * float(maturity_row["weight"])
            holder_shares = _normalized_holder_shares(preferences, category)
            for holder, holder_share in holder_shares.items():
                holder_amount = maturity_amount * holder_share
                if holder_amount <= 0.0:
                    continue
                for subbucket, route_share in _route_shares(holder, category, route_shares).items():
                    face_value = holder_amount * route_share
                    if face_value <= 0.0:
                        continue
                    rows.append(
                        _portfolio_row(
                            bond_id=next_bond_id,
                            class_name=class_name,
                            category=category,
                            opening_date=opening_date,
                            maturity_years=float(maturity_row["maturity_years"]),
                            face_value=face_value,
                            coupon_rate=float(maturity_row.get("coupon_rate", 0.0) or 0.0),
                            holder=holder,
                            holder_subbucket=subbucket,
                            fixed_spread=float(maturity_row.get("fixed_spread", 0.0) or 0.0),
                        )
                    )
                    next_bond_id += 1
        _force_class_reconciliation(rows[class_start_index:], class_name, class_total)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "source_family": "treasury_mspd",
        "source_table": MSPD_TABLE_1_SOURCE,
        "source_role": "hard_actual_state",
        "runtime_role": "hard_target",
        "record_date": extracted["record_date"],
        "opening_state_date": opening_date.isoformat(),
        "class_total_basis": "debt_held_public_mil_amt",
        "class_totals_bil": class_totals,
        "total_marketable_bil": extracted["total_marketable_bil"],
        "source_selectors": extracted["source_selectors"],
        "nonmarketable_bridge_components": extracted["nonmarketable_bridge_components"],
        "maturity_coupon_profile": {key: [dict(row) for row in value] for key, value in profile.items()},
        "holder_preferences": preferences,
        "private_subbucket_shares": route_shares,
        "claim_boundary": {
            "source_backed": "MSPD public marketable class totals are hard opening totals",
            "synthetic": "maturity coupon and holder attribution are scenario assumptions",
            "not_claimed": "not exact CUSIP composition or exact holder ownership",
            "excluded": "public nonmarketables remain bridge components not auction securities",
        },
    }
    return {"portfolio_rows": rows, "metadata": metadata}


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


def _portfolio_row(
    *,
    bond_id: int,
    class_name: str,
    category: str,
    opening_date: date,
    maturity_years: float,
    face_value: float,
    coupon_rate: float,
    holder: str,
    holder_subbucket: str,
    fixed_spread: float,
) -> dict[str, Any]:
    security_type = CLASS_TO_SECURITY_TYPE[class_name]
    maturity_date = _add_months(opening_date, max(1, int(round(maturity_years * 12.0))))
    original_principal = face_value if security_type == "TIPS" else 0.0
    adjusted_principal = face_value if security_type == "TIPS" else 0.0
    row = {
        "BondID": bond_id,
        "SecurityType": security_type,
        "IssueDate": opening_date.isoformat(),
        "MaturityDate": maturity_date.isoformat(),
        "OriginalMaturityYears": maturity_years,
        "FaceValue": face_value,
        "CouponRate": coupon_rate,
        "HolderType": holder,
        "HolderSubBucket": holder_subbucket if holder == "Private" else "",
        "Status": "Active",
        "MaturityCategory": category if security_type == "Fixed" else "",
        "OriginalPrincipal": original_principal,
        "AdjustedPrincipal": adjusted_principal,
        "ReferenceCPI_Issue": 100.0 if security_type == "TIPS" else 0.0,
        "IndexRatio": 1.0 if security_type == "TIPS" else 0.0,
        "FixedSpread": fixed_spread if security_type == "FRN" else 0.0,
        "AccruedInterest_FRN": 0.0,
        "BenchmarkRate_FRN": 0.0,
        "LastAccrualDate": opening_date.isoformat() if security_type == "FRN" else "",
        "IssuePriceRatio": 1.0,
        "IssueProceeds": face_value,
        "IssueYieldAtIssue": coupon_rate if security_type != "FRN" else 0.0,
        "TimeToMaturity": "",
        "DiscountYield": "",
        "CleanPrice": "",
        "AccruedInterest": "",
        "DirtyValue": "",
        "DirtyPriceRatio": "",
    }
    return {col: row.get(col, "") for col in BOND_PORTFOLIO_COLS}


def _force_class_reconciliation(rows: Sequence[dict[str, Any]], class_name: str, expected_total: float) -> None:
    if not rows:
        return
    actual = sum(_controlled_class_amount(row, class_name) for row in rows)
    delta = expected_total - actual
    if abs(delta) <= 1e-10:
        return
    last = rows[-1]
    last["FaceValue"] = float(last["FaceValue"]) + delta
    last["IssueProceeds"] = float(last["IssueProceeds"]) + delta
    if last["SecurityType"] == "TIPS":
        last["AdjustedPrincipal"] = float(last["AdjustedPrincipal"]) + delta
        last["OriginalPrincipal"] = float(last["OriginalPrincipal"]) + delta


def _controlled_class_amount(row: Mapping[str, Any], class_name: str) -> float:
    if class_name == "TIPS":
        return float(row["AdjustedPrincipal"])
    return float(row["FaceValue"])


def _route_shares(
    holder: str,
    category: str,
    private_subbucket_shares: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    if holder != "Private":
        return {"": 1.0}
    raw = private_subbucket_shares.get(category, {})
    domestic = max(0.0, float(raw.get(PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, 0.0) or 0.0))
    mmf = max(0.0, float(raw.get(PRIVATE_SUBBUCKET_MMF, 0.0) or 0.0))
    total = domestic + mmf
    if total <= 0.0:
        return {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 1.0}
    return {
        PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: domestic / total,
        PRIVATE_SUBBUCKET_MMF: mmf / total,
    }


def _normalized_holder_shares(
    holder_preferences: Mapping[str, Mapping[str, float]],
    category: str,
) -> dict[str, float]:
    key = f"{category}_pct"
    raw = {holder: max(0.0, float(holder_preferences.get(holder, {}).get(key, 0.0) or 0.0)) for holder in HOLDER_TYPES}
    raw["FedInternal"] = 0.0
    raw["TrustFunds"] = 0.0
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError(f"no positive public holder preference shares for {category}")
    return {holder: value / total for holder, value in raw.items()}


def _normalized_profile_rows(rows: Sequence[Mapping[str, float]], class_name: str) -> list[dict[str, float]]:
    normalized = []
    total_weight = sum(max(0.0, float(row.get("weight", 0.0) or 0.0)) for row in rows)
    if total_weight <= 0.0:
        raise ValueError(f"maturity/coupon profile for {class_name} has no positive weight")
    for row in rows:
        weight = max(0.0, float(row.get("weight", 0.0) or 0.0))
        if weight <= 0.0:
            continue
        out = dict(row)
        out["weight"] = weight / total_weight
        normalized.append(out)
    return normalized


def _mspd_public_marketable_class(row: Mapping[str, Any]) -> str | None:
    text = _row_text(row)
    if "nonmarketable" in text or "total marketable" in text or "total public debt" in text:
        return None
    if "floating rate" in text or "frn" in text:
        return "FRN"
    if "inflation" in text or "tips" in text:
        return "TIPS"
    if "bill" in text:
        return "Bills"
    if "note" in text:
        return "Notes"
    if "bond" in text:
        return "Bonds"
    return None


def _mspd_public_amount_bil(row: Mapping[str, Any]) -> float:
    if _row_value(row, "debt_held_public_bil", "amount_bil") not in (None, ""):
        return _as_float(_row_value(row, "debt_held_public_bil", "amount_bil"))
    if _row_value(row, "debt_held_public_mil_amt") not in (None, ""):
        return _as_float(_row_value(row, "debt_held_public_mil_amt")) / 1_000.0
    if _row_value(row, "debt_held_public_amt") not in (None, ""):
        value = _as_float(_row_value(row, "debt_held_public_amt"))
        return value / 1_000_000_000.0 if abs(value) > 10_000_000.0 else value
    raise ValueError(f"MSPD row missing public debt amount: {_mspd_selector(row)}")


def _mspd_selector(row: Mapping[str, Any]) -> str:
    parts = []
    for key in ("security_type_desc", "security_class_desc", "security_desc"):
        value = _row_value(row, key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else repr(dict(row))


def _row_text(row: Mapping[str, Any]) -> str:
    values = []
    for key in ("security_class_desc", "security_type_desc", "security_desc", "security_class", "security_type"):
        value = _row_value(row, key)
        if value not in (None, ""):
            values.append(str(value))
    return " ".join(values).strip().lower()


def _row_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _as_float(value: Any) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot parse numeric MSPD value {value!r}") from exc


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _month_days(year, month))
    return date(year, month, day)


def _month_days(year: int, month: int) -> int:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    if month in {4, 6, 9, 11}:
        return 30
    return 31
