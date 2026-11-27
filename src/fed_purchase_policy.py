"""New York Fed reserve-management secondary purchase policy."""

from dataclasses import dataclass

import pandas as pd


SOURCE_URL = "https://www.newyorkfed.org/markets/reserve-management-reinvestment-purchases-faq"
EXCLUDED_MATURITY_DAYS = 28


@dataclass(frozen=True)
class Sector:
    name: str
    min_months: int
    max_months: int
    weight: float


@dataclass(frozen=True)
class Limit:
    ownership_below: float
    increment: float | None
    post_operation_share: float


BILL_SECTORS = (Sector("bills_1_4m", 1, 4, 0.75), Sector("bills_4_12m", 4, 12, 0.25))
COUPON_SECTORS = (Sector("coupons_1_18m", 1, 18, 0.50), Sector("coupons_18_36m", 18, 36, 0.50))

MAX_SOMA_SHARE = 0.70
BILL_LIMITS = (
    Limit(0.15, None, 0.175),
    Limit(0.34, 0.025, 0.35),
    Limit(0.595, 0.01, 0.60),
    Limit(MAX_SOMA_SHARE, 0.005, MAX_SOMA_SHARE),
)
COUPON_LIMITS = (
    Limit(0.30, None, 0.35),
    Limit(0.475, 0.05, 0.50),
    Limit(0.59, 0.025, 0.60),
    Limit(MAX_SOMA_SHARE, 0.01, MAX_SOMA_SHARE),
)


def issue_ids(lots: pd.DataFrame, bills: pd.Series, *, require_identified: bool = False) -> pd.Series:
    """Use carried issue IDs, with contractual keys for wholly unidentified legacy books."""
    ids = lots.get("IssueID", pd.Series(pd.NA, index=lots.index)).astype("string").str.strip()
    missing = ids.isna() | ids.eq("")
    if missing.any() and (require_identified or not missing.all()):
        raise ValueError("marketable book requires complete issue IDs")
    for index, lot in lots.loc[missing].iterrows():
        maturity = pd.Timestamp(lot["MaturityDate"]).date()
        if bills.at[index]:
            ids.at[index] = f"bill-{maturity}"
            continue
        original = next(
            (lot[field] for field in ("OriginalDatedDate", "DatedDate", "IssueDate") if pd.notna(lot.get(field))),
            pd.NaT,
        )
        term = lot.get("OriginalMaturityYears")
        coupon = lot.get("CouponRate")
        detail = {"FRN": "FixedSpread", "TIPS": "ReferenceCPI_Issue"}.get(lot["SecurityType"])
        ids.at[index] = f"{lot['SecurityType']}-{original}-{maturity}-{term}-{coupon}-{lot.get(detail)}"
    return ids


def validate_issue_terms(lots: pd.DataFrame, ids: pd.Series, bills: pd.Series) -> None:
    terms = lots[["SecurityType", "MaturityDate", "CouponRate"]].copy()
    terms["CouponRate"] = terms["CouponRate"].round(12)
    for security_type, field in (("FRN", "FixedSpread"), ("TIPS", "ReferenceCPI_Issue")):
        if field in lots:
            terms[field] = lots[field].where(lots["SecurityType"].eq(security_type)).round(12)
    terms["bill"] = bills
    terms["issue_id"] = ids
    if terms.drop_duplicates()["issue_id"].duplicated().any():
        raise ValueError("a marketable issue ID has inconsistent security terms")


def purchase_sector(current: pd.Timestamp, maturity: pd.Timestamp, *, bill: bool) -> Sector | None:
    if (maturity - current).days <= EXCLUDED_MATURITY_DAYS:
        return None
    sectors = BILL_SECTORS if bill else COUPON_SECTORS
    # The shorter sector owns the shared endpoint. Bounds use calendar months.
    return next(
        (
            sector for sector in sectors
            if current + pd.DateOffset(months=sector.min_months) <= maturity
            <= current + pd.DateOffset(months=sector.max_months)
        ),
        None,
    )


def par_capacity(outstanding: float, soma: float, *, bill: bool) -> float:
    if outstanding <= 0.0:
        return 0.0
    for band in BILL_LIMITS if bill else COUPON_LIMITS:
        # At a shared ownership endpoint apply the more restrictive increment.
        if soma < band.ownership_below * outstanding:
            headroom = band.post_operation_share * outstanding - soma
            return max(0.0, headroom if band.increment is None else min(headroom, band.increment * outstanding))
    return 0.0
