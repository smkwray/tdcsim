"""Quarterly stock-only interest detail helpers for historical replay snapshots."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from tdc_shared import DAYS_PER_YEAR_ACTUAL, TGA_FLOOR_TOLERANCE


DETAIL_COLUMNS = [
    "quarter",
    "cusip/cohort_id",
    "native_sector",
    "broad_holder_class",
    "tdcsim_holder",
    "security_type",
    "component",
    "issue_date",
    "maturity_date",
    "exposure_start",
    "exposure_end",
    "days_held",
    "exposure_fraction",
    "face_begin",
    "face_end",
    "exposure_base",
    "coupon_rate_decimal",
    "issue_price_ratio",
    "issue_yield_decimal",
    "fixed_spread",
    "index_ratio_start",
    "index_ratio_end",
    "ref_cpi_start",
    "ref_cpi_end",
    "derivation_status",
    "source_status",
    "interest_amount",
    "excluded_from_default_canonical",
]

TREASURY_INTEREST_EXPENSE_DIAGNOSTIC_COLUMNS = [
    "quarter",
    "official_pool",
    "official_interest_expense_mil",
    "official_row_count",
    "replay_component_interest_mil",
    "tdcest_candidate_component_mil",
    "replay_minus_official_mil",
    "tdcest_candidate_minus_official_mil",
    "diagnostic_status",
]

_PRIVATE_PREFIX = "Private"
_DEFAULT_SOURCE_STATUS = "materialized_portfolio_snapshot"


def build_quarterly_interest_detail(
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    """Build a quarter/security/component interest detail table from snapshots."""

    frames: list[pd.DataFrame] = []
    for quarter, snapshot in _iter_snapshots(portfolio_snapshots):
        quarter_detail = _build_quarter_interest_detail_frame(quarter, snapshot)
        if not quarter_detail.empty:
            frames.append(quarter_detail)
    if not frames:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    detail = pd.concat(frames, ignore_index=True)
    detail["days_held"] = pd.to_numeric(detail["days_held"], errors="coerce").fillna(0).astype(int)
    detail["excluded_from_default_canonical"] = detail["excluded_from_default_canonical"].astype(bool)
    return detail.sort_values(
        ["quarter", "broad_holder_class", "tdcsim_holder", "cusip/cohort_id", "component"],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_stock_only_interest(
    detail: pd.DataFrame,
    *,
    holder_column: str = "tdcsim_holder",
    include_excluded: bool = False,
) -> pd.DataFrame:
    """Aggregate stock-only interest detail by quarter and holder."""

    _require_columns(detail, ["quarter", holder_column, "interest_amount"])
    working = detail.copy()
    if not include_excluded and "excluded_from_default_canonical" in working.columns:
        working = working.loc[~working["excluded_from_default_canonical"].fillna(False)].copy()
    if working.empty:
        return pd.DataFrame(columns=["quarter", holder_column, "stock_only_interest_proxy"])
    aggregated = (
        working.groupby(["quarter", holder_column], dropna=False, sort=True)["interest_amount"]
        .sum()
        .reset_index()
        .rename(columns={"interest_amount": "stock_only_interest_proxy"})
    )
    return aggregated


def align_stock_only_interest_proxy(
    detail: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    holder_column: str = "tdcsim_holder",
    reference_mapping: Mapping[str, str] | None = None,
    reference_quarter_column: str = "quarter",
    include_excluded: bool = False,
) -> pd.DataFrame:
    """Aggregate stock-only interest and optionally align it to reference columns."""

    proxy = aggregate_stock_only_interest(
        detail,
        holder_column=holder_column,
        include_excluded=include_excluded,
    )
    if reference is None:
        return proxy
    if not reference_mapping:
        raise ValueError("reference_mapping is required when a reference frame is supplied")
    _require_columns(reference, [reference_quarter_column, *reference_mapping.values()])

    aligned_frames: list[pd.DataFrame] = []
    for holder_value, reference_column in reference_mapping.items():
        holder_proxy = proxy.loc[proxy[holder_column].astype(str) == str(holder_value)].copy()
        holder_proxy = holder_proxy.rename(columns={"stock_only_interest_proxy": "stock_only_interest_proxy"})
        holder_reference = reference[[reference_quarter_column, reference_column]].copy()
        holder_reference = holder_reference.rename(
            columns={
                reference_quarter_column: "quarter",
                reference_column: "reference_interest_proxy",
            }
        )
        merged = holder_reference.merge(holder_proxy, on="quarter", how="outer")
        merged[holder_column] = holder_value
        merged["reference_column"] = reference_column
        merged["difference_vs_reference"] = (
            pd.to_numeric(merged["stock_only_interest_proxy"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["reference_interest_proxy"], errors="coerce").fillna(0.0)
        )
        aligned_frames.append(merged)
    if not aligned_frames:
        return pd.DataFrame(
            columns=[
                "quarter",
                holder_column,
                "reference_column",
                "reference_interest_proxy",
                "stock_only_interest_proxy",
                "difference_vs_reference",
            ]
        )
    return (
        pd.concat(aligned_frames, ignore_index=True)
        .sort_values(["quarter", holder_column], kind="stable")
        .reset_index(drop=True)
    )


def build_treasury_interest_expense_diagnostic(
    treasury_interest_expense: pd.DataFrame,
    interest_detail: pd.DataFrame,
    *,
    tier2_interest_candidate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare official Treasury interest pools to replay and TDC-EST diagnostics."""

    official = _official_treasury_interest_pools(treasury_interest_expense)
    replay = _replay_interest_pools(interest_detail)
    candidate = _tdcest_candidate_interest_pools(tier2_interest_candidate)
    merged = official.merge(replay, on=["quarter", "official_pool"], how="outer")
    merged = merged.merge(candidate, on=["quarter", "official_pool"], how="outer")
    for column in [
        "official_interest_expense_mil",
        "official_row_count",
        "replay_component_interest_mil",
        "tdcest_candidate_component_mil",
    ]:
        if column not in merged.columns:
            merged[column] = np.nan
    merged["replay_minus_official_mil"] = (
        pd.to_numeric(merged["replay_component_interest_mil"], errors="coerce")
        - pd.to_numeric(merged["official_interest_expense_mil"], errors="coerce")
    )
    merged["tdcest_candidate_minus_official_mil"] = (
        pd.to_numeric(merged["tdcest_candidate_component_mil"], errors="coerce")
        - pd.to_numeric(merged["official_interest_expense_mil"], errors="coerce")
    )
    merged["diagnostic_status"] = np.where(
        merged["official_interest_expense_mil"].notna(),
        "official_pool_present_diagnostic_only",
        "no_official_pool_for_component",
    )
    if merged.empty:
        return pd.DataFrame(columns=TREASURY_INTEREST_EXPENSE_DIAGNOSTIC_COLUMNS)
    return (
        merged.loc[:, TREASURY_INTEREST_EXPENSE_DIAGNOSTIC_COLUMNS]
        .sort_values(["quarter", "official_pool"], kind="stable")
        .reset_index(drop=True)
    )


def _iter_snapshots(
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(portfolio_snapshots, pd.DataFrame):
        _require_columns(portfolio_snapshots, ["quarter"])
        snapshots: list[tuple[str, pd.DataFrame]] = []
        for quarter, frame in portfolio_snapshots.groupby("quarter", sort=True, dropna=False):
            snapshots.append((str(quarter), frame.copy()))
        return snapshots
    snapshots = []
    for quarter, frame in portfolio_snapshots.items():
        working = frame.copy()
        if "quarter" not in working.columns:
            working["quarter"] = str(quarter)
        snapshots.append((str(quarter), working))
    snapshots.sort(key=lambda item: pd.Period(str(item[0]), freq="Q"))
    return snapshots


def _official_treasury_interest_pools(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "official_interest_expense_mil", "official_row_count"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"expense_catg_desc", "expense_group_desc", "expense_type_desc", "month_expense_amt"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    working = frame.copy()
    if "quarter" not in working.columns:
        if "record_date" not in working.columns:
            return pd.DataFrame(columns=columns)
        working["quarter"] = pd.to_datetime(working["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    public_issue = working["expense_catg_desc"].astype(str).str.upper().eq("INTEREST EXPENSE ON PUBLIC ISSUES")
    working = working.loc[public_issue].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["official_pool"] = working.apply(_official_interest_pool, axis=1)
    working = working.loc[working["official_pool"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["month_expense_amt"] = pd.to_numeric(working["month_expense_amt"], errors="coerce")
    out = (
        working.groupby(["quarter", "official_pool"], dropna=False, sort=False)
        .agg(
            official_interest_expense_mil=("month_expense_amt", lambda values: values.sum(min_count=1) / 1_000_000.0),
            official_row_count=("month_expense_amt", "size"),
        )
        .reset_index()
    )
    return out.loc[:, columns]


def _official_interest_pool(row: pd.Series):
    group = str(row.get("expense_group_desc", "")).strip().lower()
    expense_type = str(row.get("expense_type_desc", "")).strip().lower()
    if "int. expense inflation compensation" in expense_type:
        return "tips_inflation_compensation"
    if "treasury bills" in expense_type and group == "amortized discount":
        return "bill_discount"
    if "treasury floating rate notes" in expense_type and group == "accrued interest expense":
        return "frn_interest"
    if "inflation protected securities" in expense_type and group == "accrued interest expense":
        return "tips_coupon_accrual"
    if expense_type in {"treasury notes", "treasury bonds"} and group == "accrued interest expense":
        return "coupon_accrual"
    return pd.NA


def _replay_interest_pools(detail: pd.DataFrame) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "replay_component_interest_mil"]
    if detail is None or detail.empty or not {"quarter", "component", "interest_amount"}.issubset(detail.columns):
        return pd.DataFrame(columns=columns)
    working = detail.copy()
    if "excluded_from_default_canonical" in working.columns:
        working = working.loc[~working["excluded_from_default_canonical"].fillna(False)].copy()
    component_map = {
        "bill_discount_amortization": "bill_discount",
        "fixed_coupon_accrual": "coupon_accrual",
        "frn_accrued_interest_pass_through": "frn_interest",
        "tips_coupon_accrual": "tips_coupon_accrual",
    }
    working["official_pool"] = working["component"].map(component_map)
    working = working.loc[working["official_pool"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["interest_amount"] = pd.to_numeric(working["interest_amount"], errors="coerce")
    return (
        working.groupby(["quarter", "official_pool"], dropna=False, sort=False)["interest_amount"]
        .sum(min_count=1)
        .rename("replay_component_interest_mil")
        .reset_index()
        .loc[:, columns]
    )


def _tdcest_candidate_interest_pools(candidate: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "tdcest_candidate_component_mil"]
    if candidate is None or candidate.empty:
        return pd.DataFrame(columns=columns)
    required = {"component_family", "component_anchored_interest_mil"}
    if not required.issubset(candidate.columns):
        return pd.DataFrame(columns=columns)
    working = candidate.copy()
    if "quarter" not in working.columns:
        if "date" not in working.columns:
            return pd.DataFrame(columns=columns)
        working["quarter"] = pd.to_datetime(working["date"], errors="coerce").dt.to_period("Q").astype(str)
    family_map = {
        "bill_discount": "bill_discount",
        "coupon_accrual": "coupon_accrual",
        "frn_interest": "frn_interest",
    }
    working["official_pool"] = working["component_family"].map(family_map)
    working = working.loc[working["official_pool"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["component_anchored_interest_mil"] = pd.to_numeric(
        working["component_anchored_interest_mil"],
        errors="coerce",
    )
    return (
        working.groupby(["quarter", "official_pool"], dropna=False, sort=False)[
            "component_anchored_interest_mil"
        ]
        .sum(min_count=1)
        .rename("tdcest_candidate_component_mil")
        .reset_index()
        .loc[:, columns]
    )


def _build_quarter_interest_detail_frame(quarter: str, snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    quarter_start, quarter_end, quarter_days = _quarter_bounds(quarter)
    frame = snapshot.copy()
    security_type = _text_from_columns(frame, "SecurityType", "security_type")
    active = security_type.str.strip().ne("")
    if not active.any():
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    frame = frame.loc[active].copy()
    security_type = security_type.loc[active]

    issue_date = pd.to_datetime(_value_from_columns(frame, "IssueDate", "issue_date"), errors="coerce").dt.normalize()
    maturity_date = pd.to_datetime(_value_from_columns(frame, "MaturityDate", "maturity_date"), errors="coerce").dt.normalize()
    exposure_start = pd.Series(quarter_start, index=frame.index, dtype="datetime64[ns]")
    issue_after_start = issue_date.notna() & (issue_date > quarter_start)
    exposure_start.loc[issue_after_start] = issue_date.loc[issue_after_start]
    exposure_end = pd.Series(quarter_end, index=frame.index, dtype="datetime64[ns]")
    maturity_before_end = maturity_date.notna() & (maturity_date < quarter_end)
    exposure_end.loc[maturity_before_end] = maturity_date.loc[maturity_before_end]
    days_held = ((exposure_end - exposure_start).dt.days + 1).where(exposure_end >= exposure_start, 0)
    held = days_held > 0
    if not held.any():
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    frame = frame.loc[held].copy()
    security_type = security_type.loc[held]
    issue_date = issue_date.loc[held]
    maturity_date = maturity_date.loc[held]
    exposure_start = exposure_start.loc[held]
    exposure_end = exposure_end.loc[held]
    days_held = days_held.loc[held]

    face_value = _numeric_from_columns(frame, "FaceValue", "face_value", default=0.0)
    coupon_rate = _numeric_from_columns(frame, "CouponRate", "coupon_rate", default=0.0)
    issue_price_ratio = _numeric_from_columns(frame, "IssuePriceRatio", "issue_price_ratio", default=np.nan)
    issue_proceeds = _numeric_from_columns(frame, "IssueProceeds", "issue_proceeds", default=np.nan)
    inferred_price = issue_proceeds / face_value.replace(0.0, np.nan)
    issue_price_ratio = issue_price_ratio.where(issue_price_ratio.notna(), inferred_price)
    issue_yield = _numeric_from_columns(
        frame,
        "IssueYieldAtIssue",
        "issue_yield_decimal",
        "IssueYield",
        default=np.nan,
    )
    fixed_spread = _numeric_from_columns(frame, "FixedSpread", "fixed_spread", default=np.nan)
    index_ratio_end = _numeric_from_columns(
        frame,
        "IndexRatioEnd",
        "IndexRatio",
        "index_ratio_end",
        "index_ratio",
        default=np.nan,
    )
    index_ratio_start = _numeric_from_columns(frame, "IndexRatioStart", "index_ratio_start", default=np.nan)
    index_ratio_start = index_ratio_start.where(index_ratio_start.notna(), index_ratio_end)
    ref_cpi_start = _numeric_from_columns(frame, "ReferenceCPI_Start", "ref_cpi_start", default=np.nan)
    ref_cpi_end = _numeric_from_columns(frame, "ReferenceCPI_End", "ref_cpi_end", default=np.nan)
    base_cpi = _numeric_from_columns(frame, "ReferenceCPI_Issue", "reference_cpi_issue", default=np.nan)
    ref_cpi_start = ref_cpi_start.where(ref_cpi_start.notna(), ref_cpi_end)
    ref_cpi_end = ref_cpi_end.where(ref_cpi_end.notna(), ref_cpi_start)
    ref_cpi_start = ref_cpi_start.where(ref_cpi_start.notna(), base_cpi * index_ratio_start)
    ref_cpi_end = ref_cpi_end.where(ref_cpi_end.notna(), base_cpi * index_ratio_end)
    adjusted_principal = _numeric_from_columns(frame, "AdjustedPrincipal", "adjusted_principal", default=np.nan)
    original_principal = _numeric_from_columns(frame, "OriginalPrincipal", "original_principal", default=np.nan)
    derived_adjusted = original_principal * index_ratio_end
    adjusted_principal = adjusted_principal.where(adjusted_principal > TGA_FLOOR_TOLERANCE, derived_adjusted)
    adjusted_principal = adjusted_principal.where(adjusted_principal > TGA_FLOOR_TOLERANCE, face_value)

    broad_holder = _text_from_columns(frame, "broad_holder_class", "HolderType", "holder_type")
    holder_subbucket = _text_from_columns(frame, "HolderSubBucket", "holder_subbucket")
    tdcsim_holder = _text_from_columns(frame, "tdcsim_holder")
    private_with_subbucket = tdcsim_holder.eq("") & broad_holder.eq(_PRIVATE_PREFIX) & holder_subbucket.ne("")
    tdcsim_holder = tdcsim_holder.where(tdcsim_holder.ne(""), broad_holder)
    tdcsim_holder = tdcsim_holder.where(~private_with_subbucket, _PRIVATE_PREFIX + ":" + holder_subbucket)
    residual_holder = tdcsim_holder.astype(str).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True).isin(
        {"sourcebasisresidual", "mspdz1sourcebasisresidual"}
    )
    face_begin = face_value.where(~(issue_date.notna() & (issue_date > quarter_start)), 0.0)
    face_end = face_value.where(~(maturity_date.notna() & (maturity_date < quarter_end)), 0.0)

    base = pd.DataFrame(
        {
            "quarter": quarter,
            "cusip/cohort_id": _first_nonblank_text(frame, "cohort_id", "CohortID", "cusip", "CUSIP"),
            "native_sector": _first_nonblank_text(frame, "native_sector", "source_sector", "sector"),
            "broad_holder_class": broad_holder.replace("", pd.NA),
            "tdcsim_holder": tdcsim_holder.replace("", pd.NA),
            "security_type": security_type,
            "component": pd.NA,
            "issue_date": issue_date,
            "maturity_date": maturity_date,
            "exposure_start": exposure_start,
            "exposure_end": exposure_end,
            "days_held": days_held,
            "exposure_fraction": days_held / float(quarter_days),
            "face_begin": face_begin,
            "face_end": face_end,
            "exposure_base": adjusted_principal.where(security_type.eq("TIPS"), face_value),
            "coupon_rate_decimal": coupon_rate,
            "issue_price_ratio": issue_price_ratio,
            "issue_yield_decimal": issue_yield,
            "fixed_spread": fixed_spread,
            "index_ratio_start": index_ratio_start,
            "index_ratio_end": index_ratio_end,
            "ref_cpi_start": ref_cpi_start,
            "ref_cpi_end": ref_cpi_end,
            "derivation_status": pd.NA,
            "source_status": _text_from_columns(frame, "source_status").replace("", _DEFAULT_SOURCE_STATUS),
            "interest_amount": np.nan,
            "excluded_from_default_canonical": residual_holder,
        },
        index=frame.index,
    )

    component_frames: list[pd.DataFrame] = []
    maturity_category = _text_from_columns(frame, "MaturityCategory", "maturity_category").str.lower()
    original_maturity = _numeric_from_columns(frame, "OriginalMaturityYears", "original_maturity_years", default=np.nan)
    bill_like = (
        security_type.eq("Fixed")
        & (
            maturity_category.eq("bills")
            | (
                (coupon_rate <= TGA_FLOOR_TOLERANCE)
                & original_maturity.notna()
                & (original_maturity <= 1.0 + TGA_FLOOR_TOLERANCE)
            )
        )
    )
    issue_proceeds_for_bills = issue_proceeds.where(issue_proceeds.notna(), face_value * issue_price_ratio)
    discount = face_value - issue_proceeds_for_bills
    life_days = ((maturity_date - issue_date).dt.days + 1).where(maturity_date.notna() & issue_date.notna(), 0)
    bill_mask = bill_like & (discount > TGA_FLOOR_TOLERANCE) & (life_days > 0)
    if bill_mask.any():
        component_frames.append(
            _component_frame(
                base,
                bill_mask,
                component="bill_discount_amortization",
                derivation_status="stock_only_bill_discount_life_fraction",
                interest_amount=discount * days_held / life_days.replace(0, np.nan),
                coupon_rate_decimal=0.0,
            )
        )

    fixed_mask = (
        security_type.eq("Fixed")
        & ~bill_like
        & (coupon_rate > TGA_FLOOR_TOLERANCE)
        & (face_value > TGA_FLOOR_TOLERANCE)
    )
    if fixed_mask.any():
        component_frames.append(
            _component_frame(
                base,
                fixed_mask,
                component="fixed_coupon_accrual",
                derivation_status="stock_only_fixed_coupon_act365",
                interest_amount=face_value * coupon_rate * days_held / DAYS_PER_YEAR_ACTUAL,
            )
        )

    accrued_frn = _numeric_from_columns(frame, "AccruedInterest_FRN", "accrued_interest_frn", default=0.0)
    frn_mask = security_type.eq("FRN") & (accrued_frn > TGA_FLOOR_TOLERANCE)
    if frn_mask.any():
        component_frames.append(
            _component_frame(
                base,
                frn_mask,
                component="frn_accrued_interest_pass_through",
                derivation_status="snapshot_frn_accrued_interest_pass_through",
                interest_amount=accrued_frn,
                coupon_rate_decimal=0.0,
            )
        )

    tips_mask = (
        security_type.eq("TIPS")
        & (coupon_rate > TGA_FLOOR_TOLERANCE)
        & (adjusted_principal > TGA_FLOOR_TOLERANCE)
    )
    if tips_mask.any():
        component_frames.append(
            _component_frame(
                base,
                tips_mask,
                component="tips_coupon_accrual",
                derivation_status="stock_only_tips_coupon_on_adjusted_principal",
                interest_amount=adjusted_principal * coupon_rate * days_held / DAYS_PER_YEAR_ACTUAL,
            )
        )

    if not component_frames:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    return pd.concat(component_frames, ignore_index=True).loc[:, DETAIL_COLUMNS]


def _component_frame(
    base: pd.DataFrame,
    mask: pd.Series,
    *,
    component: str,
    derivation_status: str,
    interest_amount: pd.Series,
    coupon_rate_decimal: float | None = None,
) -> pd.DataFrame:
    out = base.loc[mask].copy()
    out["component"] = component
    out["derivation_status"] = derivation_status
    out["interest_amount"] = pd.to_numeric(interest_amount.loc[mask], errors="coerce")
    if coupon_rate_decimal is not None:
        out["coupon_rate_decimal"] = coupon_rate_decimal
    return out


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
    return result.replace("", pd.NA)


def _quarter_bounds(quarter: str) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    period = pd.Period(str(quarter), freq="Q")
    quarter_start = period.start_time.normalize()
    quarter_end = period.end_time.normalize()
    quarter_days = int((quarter_end - quarter_start).days) + 1
    return quarter_start, quarter_end, quarter_days


def _exposure_window(
    quarter_start: pd.Timestamp,
    quarter_end: pd.Timestamp,
    issue_date: pd.Timestamp | pd.NaT,
    maturity_date: pd.Timestamp | pd.NaT,
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    exposure_start = quarter_start if pd.isna(issue_date) else max(quarter_start, issue_date.normalize())
    exposure_end = quarter_end if pd.isna(maturity_date) else min(quarter_end, maturity_date.normalize())
    if exposure_end < exposure_start:
        return exposure_start, exposure_end, 0
    return exposure_start, exposure_end, int((exposure_end - exposure_start).days) + 1


def _base_record(
    *,
    quarter: str,
    row: pd.Series,
    security_type: str,
    issue_date: pd.Timestamp | pd.NaT,
    maturity_date: pd.Timestamp | pd.NaT,
    quarter_start: pd.Timestamp,
    quarter_end: pd.Timestamp,
    exposure_start: pd.Timestamp,
    exposure_end: pd.Timestamp,
    days_held: int,
    quarter_days: int,
    face_value: float,
    adjusted_principal: float,
) -> dict[str, object]:
    broad_holder = str(row.get("broad_holder_class", row.get("HolderType", row.get("holder_type", "")))).strip()
    holder_subbucket = str(row.get("HolderSubBucket", row.get("holder_subbucket", ""))).strip()
    if not broad_holder:
        broad_holder = pd.NA
    tdcsim_holder = row.get("tdcsim_holder")
    if pd.isna(tdcsim_holder) or str(tdcsim_holder).strip() == "":
        if broad_holder == _PRIVATE_PREFIX and holder_subbucket:
            tdcsim_holder = f"{_PRIVATE_PREFIX}:{holder_subbucket}"
        else:
            tdcsim_holder = broad_holder

    issue_price_ratio = _num(
        row.get("IssuePriceRatio", row.get("issue_price_ratio")),
        default=np.nan,
    )
    issue_proceeds = _num(
        row.get("IssueProceeds", row.get("issue_proceeds")),
        default=np.nan,
    )
    if pd.isna(issue_price_ratio) and face_value > TGA_FLOOR_TOLERANCE and pd.notna(issue_proceeds):
        issue_price_ratio = issue_proceeds / face_value

    index_ratio_start, index_ratio_end = _tips_index_ratios(row)
    ref_cpi_start, ref_cpi_end = _tips_reference_cpi(row, index_ratio_start, index_ratio_end)
    face_begin = 0.0 if pd.notna(issue_date) and issue_date.normalize() > quarter_start.normalize() else face_value
    face_end = 0.0 if pd.notna(maturity_date) and maturity_date.normalize() < quarter_end.normalize() else face_value

    return {
        "quarter": quarter,
        "cusip/cohort_id": _cusip_or_cohort_id(row),
        "native_sector": row.get("native_sector", row.get("source_sector", row.get("sector", pd.NA))),
        "broad_holder_class": broad_holder,
        "tdcsim_holder": tdcsim_holder,
        "security_type": security_type,
        "component": pd.NA,
        "issue_date": issue_date,
        "maturity_date": maturity_date,
        "exposure_start": exposure_start,
        "exposure_end": exposure_end,
        "days_held": days_held,
        "exposure_fraction": days_held / float(quarter_days) if quarter_days > 0 else np.nan,
        "face_begin": face_begin,
        "face_end": face_end,
        "exposure_base": adjusted_principal if security_type == "TIPS" else face_value,
        "coupon_rate_decimal": _num(row.get("CouponRate", row.get("coupon_rate")), default=0.0),
        "issue_price_ratio": issue_price_ratio,
        "issue_yield_decimal": _num(
            row.get("IssueYieldAtIssue", row.get("issue_yield_decimal", row.get("IssueYield", np.nan))),
            default=np.nan,
        ),
        "fixed_spread": _num(row.get("FixedSpread", row.get("fixed_spread")), default=np.nan),
        "index_ratio_start": index_ratio_start,
        "index_ratio_end": index_ratio_end,
        "ref_cpi_start": ref_cpi_start,
        "ref_cpi_end": ref_cpi_end,
        "derivation_status": pd.NA,
        "source_status": row.get("source_status", _DEFAULT_SOURCE_STATUS),
        "interest_amount": np.nan,
        "excluded_from_default_canonical": False,
    }


def _build_fixed_coupon_record(
    *,
    row: pd.Series,
    base_record: dict[str, object],
    days_held: int,
    security_type: str,
    face_value: float,
) -> dict[str, object] | None:
    if security_type != "Fixed" or _is_bill_like_row(row):
        return None
    coupon_rate = _num(base_record["coupon_rate_decimal"], default=0.0)
    if coupon_rate <= TGA_FLOOR_TOLERANCE or face_value <= TGA_FLOOR_TOLERANCE:
        return None
    record = dict(base_record)
    record["component"] = "fixed_coupon_accrual"
    record["derivation_status"] = "stock_only_fixed_coupon_act365"
    record["interest_amount"] = face_value * coupon_rate * days_held / DAYS_PER_YEAR_ACTUAL
    return record


def _build_bill_discount_record(
    *,
    row: pd.Series,
    base_record: dict[str, object],
    days_held: int,
    security_type: str,
    issue_date: pd.Timestamp | pd.NaT,
    maturity_date: pd.Timestamp | pd.NaT,
) -> dict[str, object] | None:
    if security_type != "Fixed" or not _is_bill_like_row(row):
        return None
    if pd.isna(issue_date) or pd.isna(maturity_date):
        return None
    face_value = _num(row.get("FaceValue", row.get("face_value")))
    issue_proceeds = _num(row.get("IssueProceeds", row.get("issue_proceeds")), default=np.nan)
    if pd.isna(issue_proceeds):
        issue_price_ratio = _num(row.get("IssuePriceRatio", row.get("issue_price_ratio")), default=np.nan)
        if face_value > TGA_FLOOR_TOLERANCE and pd.notna(issue_price_ratio):
            issue_proceeds = face_value * issue_price_ratio
    discount = face_value - issue_proceeds
    life_days = int((maturity_date.normalize() - issue_date.normalize()).days) + 1
    if discount <= TGA_FLOOR_TOLERANCE or life_days <= 0:
        return None
    record = dict(base_record)
    record["component"] = "bill_discount_amortization"
    record["coupon_rate_decimal"] = 0.0
    record["derivation_status"] = "stock_only_bill_discount_life_fraction"
    record["interest_amount"] = discount * days_held / float(life_days)
    return record


def _build_frn_record(
    *,
    row: pd.Series,
    base_record: dict[str, object],
    security_type: str,
) -> dict[str, object] | None:
    if security_type != "FRN":
        return None
    accrued_interest = _num(row.get("AccruedInterest_FRN", row.get("accrued_interest_frn")), default=0.0)
    if accrued_interest <= TGA_FLOOR_TOLERANCE:
        return None
    record = dict(base_record)
    record["component"] = "frn_accrued_interest_pass_through"
    record["coupon_rate_decimal"] = 0.0
    record["derivation_status"] = "snapshot_frn_accrued_interest_pass_through"
    record["interest_amount"] = accrued_interest
    return record


def _build_tips_coupon_record(
    *,
    row: pd.Series,
    base_record: dict[str, object],
    days_held: int,
    security_type: str,
    adjusted_principal: float,
) -> dict[str, object] | None:
    if security_type != "TIPS":
        return None
    coupon_rate = _num(base_record["coupon_rate_decimal"], default=0.0)
    if coupon_rate <= TGA_FLOOR_TOLERANCE or adjusted_principal <= TGA_FLOOR_TOLERANCE:
        return None
    record = dict(base_record)
    record["component"] = "tips_coupon_accrual"
    record["derivation_status"] = "stock_only_tips_coupon_on_adjusted_principal"
    record["interest_amount"] = adjusted_principal * coupon_rate * days_held / DAYS_PER_YEAR_ACTUAL
    return record


def _tips_adjusted_principal(row: pd.Series, *, fallback_face: float) -> float:
    adjusted_principal = _num(row.get("AdjustedPrincipal", row.get("adjusted_principal")), default=np.nan)
    if pd.notna(adjusted_principal) and adjusted_principal > TGA_FLOOR_TOLERANCE:
        return adjusted_principal
    original_principal = _num(
        row.get("OriginalPrincipal", row.get("original_principal")),
        default=np.nan,
    )
    index_ratio = _num(row.get("IndexRatio", row.get("index_ratio")), default=np.nan)
    if pd.notna(original_principal) and pd.notna(index_ratio):
        derived = original_principal * index_ratio
        if derived > TGA_FLOOR_TOLERANCE:
            return derived
    return fallback_face


def _tips_index_ratios(row: pd.Series) -> tuple[float, float]:
    end_ratio = _num(row.get("IndexRatioEnd", row.get("IndexRatio", row.get("index_ratio_end", row.get("index_ratio")))), default=np.nan)
    start_ratio = _num(row.get("IndexRatioStart", row.get("index_ratio_start")), default=np.nan)
    if pd.isna(start_ratio):
        start_ratio = end_ratio
    return start_ratio, end_ratio


def _tips_reference_cpi(row: pd.Series, index_ratio_start: float, index_ratio_end: float) -> tuple[float, float]:
    start_cpi = _num(row.get("ReferenceCPI_Start", row.get("ref_cpi_start")), default=np.nan)
    end_cpi = _num(row.get("ReferenceCPI_End", row.get("ref_cpi_end")), default=np.nan)
    if pd.notna(start_cpi) or pd.notna(end_cpi):
        if pd.isna(start_cpi):
            start_cpi = end_cpi
        if pd.isna(end_cpi):
            end_cpi = start_cpi
        return start_cpi, end_cpi
    base_cpi = _num(
        row.get("ReferenceCPI_Issue", row.get("reference_cpi_issue")),
        default=np.nan,
    )
    if pd.isna(base_cpi):
        return np.nan, np.nan
    start_cpi = base_cpi * index_ratio_start if pd.notna(index_ratio_start) else np.nan
    end_cpi = base_cpi * index_ratio_end if pd.notna(index_ratio_end) else np.nan
    return start_cpi, end_cpi


def _cusip_or_cohort_id(row: pd.Series):
    for column in ("cusip", "CUSIP", "cohort_id", "CohortID"):
        value = row.get(column, pd.NA)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return pd.NA


def _is_bill_like_row(row: pd.Series) -> bool:
    security_type = str(row.get("SecurityType", row.get("security_type", ""))).strip()
    if security_type != "Fixed":
        return False
    maturity_category = str(row.get("MaturityCategory", row.get("maturity_category", ""))).strip().lower()
    if maturity_category == "bills":
        return True
    coupon_rate = _num(row.get("CouponRate", row.get("coupon_rate")), default=0.0)
    original_maturity_years = _num(
        row.get("OriginalMaturityYears", row.get("original_maturity_years")),
        default=np.nan,
    )
    return (
        coupon_rate <= TGA_FLOOR_TOLERANCE
        and pd.notna(original_maturity_years)
        and original_maturity_years <= 1.0 + TGA_FLOOR_TOLERANCE
    )


def _to_timestamp(value) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def _num(value, *, default=0.0):
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return default
    return float(converted)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


__all__ = [
    "DETAIL_COLUMNS",
    "TREASURY_INTEREST_EXPENSE_DIAGNOSTIC_COLUMNS",
    "aggregate_stock_only_interest",
    "align_stock_only_interest_proxy",
    "build_quarterly_interest_detail",
    "build_treasury_interest_expense_diagnostic",
]
