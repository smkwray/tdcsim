"""Materialize coarse replay allocations into tdcsim portfolio rows."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from historical_replay_solver import _COHORT_ID_COL, _QUARTER_COL, _pick_column
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)


_ALLOCATION_COL = "allocated_outstanding"
_SECTOR_COL = "sector"
_COHORT_VALUE_CANDIDATES = (
    "cohort_outstanding",
    "outstanding",
    "outstanding_amount",
    "amount",
    "face_value",
    "face_value_bil",
    "adjusted_principal",
    "adjusted_principal_bil",
)


def _normalize_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def _prepare_cohorts(cohorts: pd.DataFrame) -> tuple[pd.DataFrame, str | None, list[str]]:
    if _COHORT_ID_COL not in cohorts.columns:
        raise ValueError(f"`cohorts` must include `{_COHORT_ID_COL}`")
    try:
        value_col = _pick_column(cohorts, _COHORT_VALUE_CANDIDATES, "cohort outstanding")
    except ValueError:
        value_col = None
    working = cohorts.copy()
    key_cols = [_COHORT_ID_COL]
    if _QUARTER_COL in working.columns:
        key_cols = [_QUARTER_COL, _COHORT_ID_COL]
    if not working.duplicated(subset=key_cols, keep=False).any():
        prepared = working.copy()
        if value_col is not None:
            prepared[value_col] = pd.to_numeric(prepared[value_col], errors="coerce").fillna(0.0)
        return prepared, value_col, key_cols
    agg_map = {}
    for col in working.columns:
        if col in key_cols:
            continue
        if col == value_col:
            agg_map[col] = "sum"
        else:
            agg_map[col] = _first_non_null
    prepared = working.groupby(key_cols, sort=False, dropna=False).agg(agg_map).reset_index()
    if value_col is not None:
        prepared[value_col] = pd.to_numeric(prepared[value_col], errors="coerce").fillna(0.0)
    return prepared, value_col, key_cols


def _security_type(row: pd.Series) -> str:
    for col in ("SecurityType", "security_type"):
        if col in row and pd.notna(row[col]):
            text = str(row[col]).strip()
            key = _normalize_label(text)
            if key in {
                "fixed",
                "nominal",
                "bill",
                "bills",
                "note",
                "notes",
                "bond",
                "bonds",
                "cashmanagementbill",
                "cmb",
                "federalfinancingbank",
            }:
                return "Fixed"
            if key in {"tips", "inflation", "inflationindexed", "treasuryinflationprotectedsecurity"}:
                return "TIPS"
            if key in {"frn", "floating", "floatingratenote"}:
                return "FRN"
            if key in {"nonmarketable", "nonmarketables"}:
                return "NonMarketable"
            return text
    category = str(row.get("MaturityCategory", row.get("maturity_category", ""))).lower()
    if category == "tips":
        return "TIPS"
    if category == "frn":
        return "FRN"
    if category == "nonmarketable":
        return "NonMarketable"
    return "Fixed"


def _maturity_category(row: pd.Series, security_type: str) -> str:
    for col in ("MaturityCategory", "maturity_category"):
        if col in row and pd.notna(row[col]):
            return str(row[col])
    if security_type == "TIPS":
        return "tips"
    if security_type == "FRN":
        return "frn"
    if security_type == "NonMarketable":
        return "nonmarketable"
    maturity_years = pd.to_numeric(row.get("OriginalMaturityYears", row.get("original_maturity_years")), errors="coerce")
    if pd.notna(maturity_years):
        if float(maturity_years) <= 1.0 + 1e-9:
            return "bills"
        if float(maturity_years) <= 10.0 + 1e-9:
            return "notes"
    return "bonds"


def _scale_amount(row: pd.Series, column: str, scale: float, default: float) -> float:
    if column in row and pd.notna(row[column]):
        return float(pd.to_numeric(row[column], errors="coerce")) * scale
    return default


def _first_numeric(row: pd.Series, columns: tuple[str, ...]):
    for column in columns:
        if column not in row or pd.isna(row[column]):
            continue
        value = pd.to_numeric(row[column], errors="coerce")
        if pd.notna(value):
            return float(value)
    return np.nan


def _decimal_rate(row: pd.Series, decimal_columns: tuple[str, ...], pct_columns: tuple[str, ...], default: float = 0.0) -> float:
    value = _first_numeric(row, decimal_columns)
    if pd.notna(value):
        return value
    pct_value = _first_numeric(row, pct_columns)
    if pd.notna(pct_value):
        return pct_value / 100.0
    return default


def _price_ratio(row: pd.Series, default: float = 1.0) -> float:
    value = _first_numeric(row, ("IssuePriceRatio", "issue_price_ratio"))
    if pd.notna(value):
        return value
    per100 = _first_numeric(row, ("price_per100", "avg_med_price", "high_price"))
    if pd.notna(per100):
        return per100 / 100.0
    return default


def _metadata_value(row: pd.Series, *columns: str, default=""):
    for column in columns:
        if column in row and pd.notna(row[column]):
            text = str(row[column]).strip()
            if text:
                return row[column]
    return default


def _value_from_columns(frame: pd.DataFrame, *columns: str):
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series(pd.NA, index=frame.index)


def _numeric_from_columns(frame: pd.DataFrame, *columns: str, default=np.nan) -> pd.Series:
    values = _value_from_columns(frame, *columns)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(default) if not pd.isna(default) else numeric


def _text_from_columns(frame: pd.DataFrame, *columns: str) -> pd.Series:
    values = _value_from_columns(frame, *columns)
    return values.where(values.notna(), "").astype(str).str.strip()


def _first_nonblank_text(frame: pd.DataFrame, *columns: str) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = frame[column].where(frame[column].notna(), "").astype(str).str.strip()
        result = result.where(result.astype(str).str.strip().ne(""), candidate)
    return result


def _decimal_rate_series(
    frame: pd.DataFrame,
    decimal_columns: tuple[str, ...],
    pct_columns: tuple[str, ...],
    *,
    default: float = 0.0,
) -> pd.Series:
    decimal = _numeric_from_columns(frame, *decimal_columns, default=np.nan)
    pct = _numeric_from_columns(frame, *pct_columns, default=np.nan) / 100.0
    return decimal.where(decimal.notna(), pct).fillna(default)


def _price_ratio_series(frame: pd.DataFrame, *, default: float = 1.0) -> pd.Series:
    ratio = _numeric_from_columns(frame, "IssuePriceRatio", "issue_price_ratio", default=np.nan)
    per100 = _numeric_from_columns(frame, "price_per100", "avg_med_price", "high_price", default=np.nan)
    return ratio.where(ratio.notna(), per100 / 100.0).fillna(default)


def _date_from_columns(frame: pd.DataFrame, *columns: str) -> pd.Series:
    return pd.to_datetime(_value_from_columns(frame, *columns), errors="coerce")


def _first_interest_from_pay_dates(
    frame: pd.DataFrame,
    issue_dates: pd.Series,
    maturity_dates: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    pay_date_columns = [f"interest_pay_date_{idx}" for idx in range(1, 5)]
    pay_dates = pd.DataFrame(
        {
            column: pd.to_datetime(_value_from_columns(frame, column), errors="coerce")
            for column in pay_date_columns
        },
        index=frame.index,
    )
    maturity_missing = pd.DataFrame(
        np.repeat(maturity_dates.isna().to_numpy()[:, None], len(pay_date_columns), axis=1),
        index=frame.index,
        columns=pay_date_columns,
    )
    valid_pay_dates = pay_dates.where(
        pay_dates.gt(issue_dates, axis=0) & (maturity_missing | pay_dates.le(maturity_dates, axis=0))
    )
    return valid_pay_dates.min(axis=1), valid_pay_dates.notna().sum(axis=1).astype(float)


def _security_type_from_label(value) -> str:
    text = str(value).strip()
    key = _normalize_label(text)
    if key in {
        "fixed",
        "nominal",
        "bill",
        "bills",
        "note",
        "notes",
        "bond",
        "bonds",
        "cashmanagementbill",
        "cmb",
        "federalfinancingbank",
    }:
        return "Fixed"
    if key in {"tips", "inflation", "inflationindexed", "treasuryinflationprotectedsecurity"}:
        return "TIPS"
    if key in {"frn", "floating", "floatingratenote"}:
        return "FRN"
    if key in {"nonmarketable", "nonmarketables"}:
        return "NonMarketable"
    return text or "Fixed"


def _map_holder(row: pd.Series) -> tuple[str, str]:
    explicit_holder = row.get("holder_type", row.get("HolderType"))
    explicit_subbucket = row.get("holder_subbucket", row.get("HolderSubBucket", ""))
    if pd.isna(explicit_holder):
        explicit_holder = row.get("tdcsim_holder")
        explicit_subbucket = row.get("tdcsim_holder_subbucket", explicit_subbucket)
    if pd.isna(explicit_holder):
        explicit_holder = row.get("broad_holder_class")
    if pd.notna(explicit_holder):
        holder = _holder_type_from_label(explicit_holder)
        if holder == "Private":
            subbucket = "" if pd.isna(explicit_subbucket) else str(explicit_subbucket)
            return holder, subbucket or PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
        return holder, ""

    sector_key = _normalize_label(row[_SECTOR_COL])
    if sector_key == "banks":
        return "Banks", ""
    if sector_key in {"cb", "fed", "federalreserve", "centralbank"}:
        return "CB", ""
    if sector_key in {"foreign", "foreigninternational", "row", "restofworld", "foreignofficial"}:
        return "Foreign", ""
    if sector_key == "trustfunds":
        return "TrustFunds", ""
    if sector_key == "fedinternal":
        return "FedInternal", ""
    if sector_key in {"mmf", "moneymarketcash", "moneymarketfunds"}:
        return "Private", PRIVATE_SUBBUCKET_MMF
    if sector_key in {
        "domesticnonbankexmmf",
        "domesticnonbank",
        "domesticnonbankother",
        "statelocalgovernment",
        "governmentsponsoredenterprises",
        "otherfinancial",
        "absissuers",
        "dealers",
        "investmentfunds",
        "pensionsinsurers",
        "individuals",
        "private",
    }:
        return "Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
    if sector_key in {"other", "residual", "otherresidual", "otherprivate", "residualother"}:
        return "Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
    if sector_key in {"sourcebasisresidual", "mspdz1sourcebasisresidual", "unallocatedbasisresidual"}:
        return "SourceBasisResidual", ""
    return "Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK


def _holder_type_from_label(value) -> str:
    key = _normalize_label(value)
    if key in {"banks", "bank", "depository", "depositoryinstitutions"}:
        return "Banks"
    if key in {"cb", "fed", "federalreserve", "monetaryauthority", "centralbank"}:
        return "CB"
    if key in {"foreign", "foreigninternational", "row", "restofworld", "foreignofficial"}:
        return "Foreign"
    if key in {"trustfunds", "trustfund", "treasurytrustfunds"}:
        return "TrustFunds"
    if key in {"fedinternal", "fedinternalaccounts"}:
        return "FedInternal"
    if key in {"sourcebasisresidual", "mspdz1sourcebasisresidual", "unallocatedbasisresidual"}:
        return "SourceBasisResidual"
    return "Private"


def materialize_portfolio(
    allocations: pd.DataFrame,
    cohorts: pd.DataFrame,
    *,
    start_bond_id: int = 1,
) -> pd.DataFrame:
    """Convert coarse sector/cohort allocations into tdcsim portfolio rows."""

    required_cols = {_SECTOR_COL, _COHORT_ID_COL, _ALLOCATION_COL}
    missing = required_cols.difference(allocations.columns)
    if missing:
        raise ValueError(f"`allocations` is missing required columns: {sorted(missing)}")

    prepared_cohorts, cohort_value_col, cohort_key_cols = _prepare_cohorts(cohorts)
    working_allocations = allocations.copy()
    working_allocations[_ALLOCATION_COL] = pd.to_numeric(
        working_allocations[_ALLOCATION_COL], errors="coerce"
    ).fillna(0.0)
    working_allocations = working_allocations.loc[working_allocations[_ALLOCATION_COL] > 0.0].copy()

    metadata_columns = [
        _QUARTER_COL,
        _COHORT_ID_COL,
        "cusip",
        "series_cd",
        "native_sector",
        "broad_holder_class",
        "tdcsim_holder",
        "tdcsim_holder_subbucket",
        "source_status",
        "security_enrichment_status",
        "source_sector",
        "allocated_outstanding_source",
        "interest_pay_date_1",
        "interest_pay_date_2",
        "interest_pay_date_3",
        "interest_pay_date_4",
    ]
    empty_columns = BOND_PORTFOLIO_COLS + metadata_columns
    if working_allocations.empty:
        empty = pd.DataFrame(columns=empty_columns)
        return empty.astype({**PORTFOLIO_DTYPES}, errors="ignore")

    merge_keys = [
        column
        for column in cohort_key_cols
        if column in working_allocations.columns and column in prepared_cohorts.columns
    ]
    if _COHORT_ID_COL not in merge_keys:
        merge_keys.append(_COHORT_ID_COL)
    merged = working_allocations.merge(
        prepared_cohorts,
        on=merge_keys,
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    if (merged["_merge"] != "both").any():
        missing_ids = merged.loc[merged["_merge"] != "both", _COHORT_ID_COL].tolist()
        raise ValueError(f"Could not find cohort metadata for cohort ids: {missing_ids}")
    merged = merged.drop(columns="_merge").reset_index(drop=True)

    allocation = pd.to_numeric(merged[_ALLOCATION_COL], errors="coerce").fillna(0.0)
    cohort_base = allocation.copy()
    if cohort_value_col is not None and cohort_value_col in merged.columns:
        cohort_base = pd.to_numeric(merged[cohort_value_col], errors="coerce").fillna(0.0)
        cohort_base = cohort_base.where(cohort_base > 0.0, allocation)
    scale = allocation / cohort_base.replace(0.0, np.nan)
    scale = scale.fillna(0.0)

    security_type = _text_from_columns(merged, "SecurityType", "security_type").map(_security_type_from_label)
    original_maturity_years = _numeric_from_columns(
        merged,
        "OriginalMaturityYears",
        "original_maturity_years",
        default=np.nan,
    )
    raw_maturity_category = _text_from_columns(merged, "MaturityCategory", "maturity_category")
    maturity_category = raw_maturity_category.copy()
    maturity_category = maturity_category.where(maturity_category.ne(""), "")
    maturity_category = maturity_category.where(~security_type.eq("TIPS"), maturity_category.where(maturity_category.ne(""), "tips"))
    maturity_category = maturity_category.where(~security_type.eq("FRN"), maturity_category.where(maturity_category.ne(""), "frn"))
    maturity_category = maturity_category.where(
        maturity_category.ne(""),
        np.where(original_maturity_years <= 1.0 + 1e-9, "bills", np.where(original_maturity_years <= 10.0 + 1e-9, "notes", "bonds")),
    )

    coupon_rate = _decimal_rate_series(
        merged,
        ("CouponRate", "coupon_rate", "coupon_rate_decimal"),
        ("interest_rate_pct", "int_rate"),
        default=0.0,
    )
    issue_yield = _decimal_rate_series(
        merged,
        ("IssueYieldAtIssue", "issue_yield_at_issue", "issue_yield_decimal"),
        ("yield_pct", "avg_med_yield", "high_yield"),
        default=np.nan,
    )
    issue_yield = issue_yield.where(issue_yield.notna(), coupon_rate)
    issue_price_ratio = _price_ratio_series(merged, default=1.0)
    bill_price_mask = (
        maturity_category.astype(str).map(_normalize_label).eq("bills")
        & (issue_price_ratio >= 1.0)
        & (issue_yield > 0.0)
        & original_maturity_years.notna()
        & (original_maturity_years > 0.0)
    )
    issue_price_ratio = issue_price_ratio.where(
        ~bill_price_mask,
        1.0 / (1.0 + issue_yield * original_maturity_years),
    )
    fixed_spread = _decimal_rate_series(
        merged,
        ("FixedSpread", "fixed_spread", "spread_decimal"),
        ("spread",),
        default=0.0,
    )

    index_ratio = _numeric_from_columns(
        merged,
        "IndexRatio",
        "index_ratio_on_issue_date",
        default=1.0,
    )
    index_ratio = index_ratio.where(index_ratio > 0.0, 1.0).fillna(1.0)
    explicit_adjusted_principal = _numeric_from_columns(merged, "AdjustedPrincipal", default=np.nan) * scale
    adjusted_principal = allocation.copy()
    adjusted_principal = adjusted_principal.where(
        ~explicit_adjusted_principal.notna() | (explicit_adjusted_principal <= 0.0),
        explicit_adjusted_principal,
    )
    adjusted_principal = adjusted_principal.where(security_type.eq("TIPS"), allocation)
    adjusted_principal = adjusted_principal.fillna(allocation)
    original_principal = _numeric_from_columns(merged, "OriginalPrincipal", default=np.nan) * scale
    original_principal = original_principal.where(
        ~security_type.eq("TIPS") | (original_principal > 0.0),
        adjusted_principal / index_ratio.replace(0.0, np.nan),
    )
    original_principal = original_principal.fillna(0.0)

    accrued_frn = _numeric_from_columns(merged, "AccruedInterest_FRN", default=np.nan) * scale
    accrued_frn_per100 = _numeric_from_columns(
        merged,
        "AccruedInterest_FRN_Per100",
        "frn_accrued_interest_per100",
        default=np.nan,
    )
    accrued_frn = accrued_frn.where(accrued_frn > 0.0, allocation * accrued_frn_per100 / 100.0)
    accrued_frn = accrued_frn.fillna(0.0)

    explicit_holder = _first_nonblank_text(
        merged,
        "holder_type",
        "HolderType",
        "tdcsim_holder",
        "broad_holder_class",
        _SECTOR_COL,
    )
    holder_type = explicit_holder.map(_holder_type_from_label)
    explicit_subbucket = _first_nonblank_text(
        merged,
        "holder_subbucket",
        "HolderSubBucket",
        "tdcsim_holder_subbucket",
    )
    sector_key = _first_nonblank_text(merged, _SECTOR_COL).map(_normalize_label)
    inferred_private_subbucket = pd.Series(PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, index=merged.index)
    inferred_private_subbucket = inferred_private_subbucket.where(
        ~sector_key.isin({"mmf", "moneymarketcash", "moneymarketfunds"}),
        PRIVATE_SUBBUCKET_MMF,
    )
    holder_subbucket = explicit_subbucket.where(explicit_subbucket.ne(""), inferred_private_subbucket)
    holder_subbucket = holder_subbucket.where(holder_type.eq("Private"), "")
    native_sector = _first_nonblank_text(merged, "native_sector", _SECTOR_COL)
    broad_holder_class = _first_nonblank_text(merged, "broad_holder_class")
    broad_holder_class = broad_holder_class.where(broad_holder_class.ne(""), holder_type)
    source_status = _first_nonblank_text(merged, "source_status", "source_status_y", "source_status_x").replace("", pd.NA)
    quarter_values = _first_nonblank_text(merged, _QUARTER_COL, f"{_QUARTER_COL}_x", f"{_QUARTER_COL}_y").replace("", pd.NA)

    issue_dates = _date_from_columns(merged, "IssueDate", "issue_date")
    maturity_dates = _date_from_columns(merged, "MaturityDate", "maturity_date")
    dated_dates = _date_from_columns(merged, "DatedDate", "dated_date").fillna(issue_dates)
    original_dated_dates = _date_from_columns(
        merged,
        "OriginalDatedDate",
        "original_dated_date",
    ).fillna(dated_dates)
    first_interest_from_schedule, schedule_frequency = _first_interest_from_pay_dates(
        merged,
        issue_dates,
        maturity_dates,
    )
    first_interest_payment_dates = _date_from_columns(
        merged,
        "FirstInterestPaymentDate",
        "first_interest_payment_date",
    ).fillna(first_interest_from_schedule)
    explicit_payment_frequency = _numeric_from_columns(
        merged,
        "InterestPaymentFrequency",
        "interest_payment_frequency",
        default=np.nan,
    )
    interest_payment_frequency = explicit_payment_frequency.where(
        explicit_payment_frequency.notna(),
        schedule_frequency.where(schedule_frequency > 0.0),
    )
    fixed_coupon_mask = security_type.isin(["Fixed", "TIPS"]) & (coupon_rate > 1e-12)
    interest_payment_frequency = interest_payment_frequency.where(
        interest_payment_frequency.notna(),
        np.where(security_type.eq("FRN"), 4.0, np.where(fixed_coupon_mask, 2.0, np.nan)),
    )

    materialized = pd.DataFrame(
        {
            "BondID": range(int(start_bond_id), int(start_bond_id) + len(merged.index)),
            "SecurityType": security_type,
            "IssueDate": issue_dates,
            "MaturityDate": maturity_dates,
            "DatedDate": dated_dates,
            "OriginalDatedDate": original_dated_dates,
            "FirstInterestPaymentDate": first_interest_payment_dates,
            "InterestPaymentFrequency": interest_payment_frequency,
            "OriginalMaturityYears": original_maturity_years,
            "FaceValue": allocation,
            "CouponRate": coupon_rate,
            "HolderType": holder_type,
            "HolderSubBucket": holder_subbucket,
            "Status": _first_nonblank_text(merged, "Status").replace("", "Active"),
            "MaturityCategory": maturity_category,
            "OriginalPrincipal": original_principal,
            "AdjustedPrincipal": adjusted_principal,
            "ReferenceCPI_Issue": _numeric_from_columns(merged, "ReferenceCPI_Issue", "ref_cpi_on_issue_date", default=0.0),
            "IndexRatio": index_ratio,
            "FixedSpread": fixed_spread,
            "AccruedInterest_FRN": accrued_frn,
            "BenchmarkRate_FRN": _numeric_from_columns(merged, "BenchmarkRate_FRN", default=0.0),
            "LastAccrualDate": pd.to_datetime(_value_from_columns(merged, "LastAccrualDate", "last_accrual_date"), errors="coerce"),
            "IssuePriceRatio": issue_price_ratio,
            "IssueProceeds": _numeric_from_columns(merged, "IssueProceeds", default=np.nan).mul(scale).where(
                _numeric_from_columns(merged, "IssueProceeds", default=np.nan).notna(),
                allocation * issue_price_ratio,
            ),
            "IssueYieldAtIssue": issue_yield,
            "TimeToMaturity": np.nan,
            "DiscountYield": np.nan,
            "CleanPrice": np.nan,
            "AccruedInterest": np.nan,
            "DirtyValue": np.nan,
            "DirtyPriceRatio": np.nan,
            _QUARTER_COL: quarter_values,
            _COHORT_ID_COL: merged[_COHORT_ID_COL],
            "cusip": _first_nonblank_text(merged, "cusip", _COHORT_ID_COL),
            "series_cd": _first_nonblank_text(merged, "series_cd"),
            "native_sector": native_sector,
            "broad_holder_class": broad_holder_class,
            "tdcsim_holder": holder_type,
            "tdcsim_holder_subbucket": holder_subbucket,
            "source_status": source_status,
            "security_enrichment_status": _first_nonblank_text(merged, "security_enrichment_status").replace("", pd.NA),
            "source_sector": merged[_SECTOR_COL],
            "allocated_outstanding_source": allocation,
            "interest_pay_date_1": _first_nonblank_text(merged, "interest_pay_date_1"),
            "interest_pay_date_2": _first_nonblank_text(merged, "interest_pay_date_2"),
            "interest_pay_date_3": _first_nonblank_text(merged, "interest_pay_date_3"),
            "interest_pay_date_4": _first_nonblank_text(merged, "interest_pay_date_4"),
        }
    )
    materialized = materialized[BOND_PORTFOLIO_COLS + metadata_columns]
    materialized[BOND_PORTFOLIO_COLS] = materialized[BOND_PORTFOLIO_COLS].astype(PORTFOLIO_DTYPES, errors="ignore")
    return materialized
