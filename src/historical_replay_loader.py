"""CSV loaders for bounded historical replay inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from historical_replay_contract import (
    AUCTION_FLOW_REQUIRED_COLUMNS,
    MSPD_COHORT_REQUIRED_COLUMNS,
    QUARTERLY_CASH_REQUIRED_COLUMNS,
    SECTOR_POSITION_REQUIRED_COLUMNS,
    filter_quarter_range,
    normalize_quarter_series,
    require_columns,
    validate_duplicate_keys,
    validate_evidence_labels,
    validate_numeric_columns,
    validate_source_status_columns,
)


_SOURCE_NULL_MARKERS = ("", "null", "none", "nan", "na", "n/a", "*")
_VALID_CUSIP_RE = r"^[A-Za-z0-9]{9}$"


def _read_csv(path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"historical replay file is missing: {csv_path}")
    return pd.read_csv(
        csv_path,
        low_memory=False,
        na_values=list(_SOURCE_NULL_MARKERS),
        keep_default_na=True,
    )


def _select_column(frame: pd.DataFrame, *candidates: str) -> pd.Series | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate]
    return None


def _require_alias(frame: pd.DataFrame, dataset_name: str, target: str, *candidates: str) -> pd.Series:
    column = _select_column(frame, *candidates)
    if column is None:
        raise ValueError(
            f"{dataset_name} missing required columns for {target}: {list(candidates)}"
        )
    return column


def _normalize_date_column(values: pd.Series, *, dataset_name: str, column_name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    invalid = parsed.isna() & values.notna() & values.astype(str).str.strip().ne("")
    if invalid.any():
        bad_values = sorted(values.loc[invalid].astype(str).unique().tolist())[:3]
        raise ValueError(f"{dataset_name} column {column_name} has invalid dates: {bad_values}")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalize_optional_date_column(values: pd.Series, *, dataset_name: str, column_name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    invalid = parsed.isna() & values.notna() & values.astype(str).str.strip().ne("")
    if invalid.any():
        bad_values = sorted(values.loc[invalid].astype(str).unique().tolist())[:3]
        raise ValueError(f"{dataset_name} column {column_name} has invalid dates: {bad_values}")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), "")


def _select_latest_record_in_quarter(
    frame: pd.DataFrame,
    *,
    date_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    parsed = pd.to_datetime(frame[date_column], errors="coerce")
    invalid = parsed.isna() & frame[date_column].notna() & frame[date_column].astype(str).str.strip().ne("")
    if invalid.any():
        bad_values = sorted(frame.loc[invalid, date_column].astype(str).unique().tolist())[:3]
        raise ValueError(f"{dataset_name} column {date_column} has invalid dates: {bad_values}")
    working = frame.copy()
    working["_replay_record_date"] = parsed
    working["_replay_quarter"] = parsed.dt.to_period("Q").astype(str)
    latest_dates = working.groupby("_replay_quarter")["_replay_record_date"].transform("max")
    working = working.loc[working["_replay_record_date"].eq(latest_dates)].copy()
    return working.drop(columns=["_replay_record_date", "_replay_quarter"])


def _coalesce_text_columns(*columns: pd.Series | None) -> pd.Series | None:
    result = None
    for column in columns:
        if column is None:
            continue
        text = column.where(column.notna(), "").astype(str).str.strip()
        if result is None:
            result = text
        else:
            result = result.where(result.ne(""), text)
    return result


def _valid_cusip_mask(values: pd.Series) -> pd.Series:
    return values.where(values.notna(), "").astype(str).str.strip().str.fullmatch(_VALID_CUSIP_RE)


def _default_evidence_label(frame: pd.DataFrame, *required_columns: str) -> pd.Series:
    available = pd.Series(True, index=frame.index)
    for column in required_columns:
        available &= frame[column].notna()
    return available.map(lambda is_present: "observed" if is_present else "not_available")


def _first_non_null(values: pd.Series):
    non_null = values.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def _sum_preserving_missing(values: pd.Series):
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.sum(min_count=1)


def _days_between(start_values: pd.Series, end_values: pd.Series) -> pd.Series:
    start = pd.to_datetime(start_values, errors="coerce")
    end = pd.to_datetime(end_values, errors="coerce")
    return (end - start).dt.days


def _percent_to_decimal(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric / 100.0


def _price_per100_to_ratio(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric / 100.0


def _aggregate_duplicate_rows(
    frame: pd.DataFrame,
    key_columns: list[str],
    *,
    numeric_columns: list[str],
) -> pd.DataFrame:
    if not frame.duplicated(subset=key_columns, keep=False).any():
        return frame
    numeric_set = set(numeric_columns)
    agg_map = {}
    for column in frame.columns:
        if column in key_columns:
            continue
        agg_map[column] = _sum_preserving_missing if column in numeric_set else _first_non_null
    return frame.groupby(key_columns, sort=False, dropna=False).agg(agg_map).reset_index()


def _decompose_mspd_reopening_outstanding(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"quarter", "cusip", "maturity_date", "issue_date", "outstanding", "issued_amt"}
    if not required.issubset(frame.columns):
        return frame
    working = frame.copy()
    for column in ["outstanding", "issued_amt", "redeemed_amt", "inflation_adj_amt"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        else:
            working[column] = 0.0
    for _, group in working.groupby(["quarter", "cusip", "maturity_date"], sort=False, dropna=False):
        if group["issue_date"].dropna().astype(str).nunique() <= 1:
            continue
        component_outstanding = (
            working.loc[group.index, "issued_amt"].fillna(0.0)
            + working.loc[group.index, "redeemed_amt"].fillna(0.0)
            + working.loc[group.index, "inflation_adj_amt"].fillna(0.0)
        )
        component_outstanding = component_outstanding.clip(lower=0.0)
        component_total = float(component_outstanding.sum())
        observed_outstanding = working.loc[group.index, "outstanding"].dropna()
        if observed_outstanding.empty or component_total <= 0.0:
            continue
        aggregate_outstanding = float(observed_outstanding.max())
        tolerance = max(1.0e-6, abs(aggregate_outstanding) * 1.0e-6)
        if abs(component_total - aggregate_outstanding) > tolerance:
            continue
        working.loc[group.index, "outstanding"] = component_outstanding
        if "source_status" in working.columns:
            source_status = working.loc[group.index, "source_status"].where(
                working.loc[group.index, "source_status"].notna(),
                "",
            ).astype(str)
            suffix = "mspd_reopening_outstanding_decomposed"
            working.loc[group.index, "source_status"] = source_status.map(
                lambda value: suffix if not value.strip() else f"{value};{suffix}"
            )
    return working


def _drop_mspd_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    masks = []
    for column in ["security_class1_desc", "security_class2_desc", "security_type_desc"]:
        if column not in frame.columns:
            continue
        text = frame[column].fillna("").astype(str).str.strip().str.lower()
        masks.append(
            text.str.startswith("total")
            | text.eq("federal financing bank")
        )
    if not masks:
        return frame
    summary = masks[0]
    for mask in masks[1:]:
        summary |= mask
    return frame.loc[~summary].copy()


def _canonical_security_type(value) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    if "cash management bill" in text or "cmb" in text:
        return "cash_management_bill"
    if "frn" in text or "floating" in text:
        return "frn"
    if "tips" in text or "inflation" in text:
        return "tips"
    if "bill" in text:
        return "bill"
    if "note" in text:
        return "note"
    if "bond" in text:
        return "bond"
    return text.replace(" ", "_").replace("-", "_")


def _canonical_measure(value) -> str:
    text = str(value).strip().lower()
    aliases = {
        "tx": "transaction",
        "transactions": "transaction",
        "flow": "transaction",
        "flows": "transaction",
        "level": "level",
        "levels": "level",
        "stock": "level",
        "stocks": "level",
    }
    return aliases.get(text, text)


def load_quarterly_cash(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load quarterly cash rows keyed by replay quarter."""

    frame = _read_csv(path)
    toc_transaction = _select_column(
        frame,
        "toc_transaction",
        "toc_tx",
        "selected_operating_cash_tx_mil",
        "z1_treasury_operating_cash_tx_mil",
    )
    normalized = pd.DataFrame(
        {
            "quarter": normalize_quarter_series(
                _require_alias(frame, "quarterly_cash", "quarter", "quarter", "quarter_end")
            ),
            "toc": _require_alias(
                frame,
                "quarterly_cash",
                "toc",
                "toc",
                "selected_operating_cash_level_mil",
            ),
            "tga": _require_alias(frame, "quarterly_cash", "tga", "tga", "dts_tga_mil"),
            "ttl": _require_alias(frame, "quarterly_cash", "ttl", "ttl", "dts_ttl_mil"),
            "toc_source_status": _require_alias(
                frame,
                "quarterly_cash",
                "toc_source_status",
                "toc_source_status",
                "selected_operating_cash_level_source",
            ),
        }
    )
    if toc_transaction is not None:
        normalized["toc_transaction"] = toc_transaction
        normalized["toc_transaction_source_status"] = _coalesce_text_columns(
            _select_column(
                frame,
                "toc_transaction_source_status",
                "toc_tx_source_status",
                "selected_operating_cash_tx_source",
            ),
            normalized["toc_transaction"].map(
                lambda value: "observed" if pd.notna(value) else "not_available"
            ),
        )
    normalized["tga_source_status"] = _coalesce_text_columns(
        _select_column(frame, "tga_source_status"),
        normalized["tga"].map(lambda value: "observed" if pd.notna(value) else "not_available"),
    )
    normalized["ttl_source_status"] = _coalesce_text_columns(
        _select_column(frame, "ttl_source_status"),
        normalized["ttl"].map(lambda value: "observed" if pd.notna(value) else "not_available"),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "toc"),
    )

    require_columns(normalized, QUARTERLY_CASH_REQUIRED_COLUMNS, dataset_name="quarterly_cash")
    normalized = validate_numeric_columns(
        normalized,
        [column for column in ["toc", "tga", "ttl", "toc_transaction"] if column in normalized.columns],
        dataset_name="quarterly_cash",
    )
    validate_source_status_columns(
        normalized,
        [
            column
            for column in [
                "toc_source_status",
                "tga_source_status",
                "ttl_source_status",
                "toc_transaction_source_status",
            ]
            if column in normalized.columns
        ],
        dataset_name="quarterly_cash",
    )
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="quarterly_cash")
    validate_duplicate_keys(normalized, ["quarter"], dataset_name="quarterly_cash")
    normalized = filter_quarter_range(
        normalized,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    return normalized.sort_values(["quarter"]).reset_index(drop=True)


def load_sector_positions(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load sector level/transaction rows normalized to quarter, sector, measure, value."""

    frame = _read_csv(path)
    native_sector = _coalesce_text_columns(
        _select_column(frame, "z1_sector", "sector"),
        _select_column(frame, "broad_investor_class"),
    )
    if native_sector is None:
        raise ValueError("sector_positions missing required columns for sector: ['z1_sector', 'sector', 'broad_investor_class']")
    broad_holder_class = _coalesce_text_columns(
        _select_column(frame, "broad_holder_class"),
        _select_column(frame, "broad_investor_class"),
        native_sector,
    )
    normalized = pd.DataFrame(
        {
            "quarter": normalize_quarter_series(
                _require_alias(frame, "sector_positions", "quarter", "quarter", "date", "record_date")
            ),
            "sector": native_sector.astype(str).str.strip(),
            "native_sector": native_sector.astype(str).str.strip(),
            "broad_holder_class": broad_holder_class.astype(str).str.strip(),
            "measure": _require_alias(frame, "sector_positions", "measure", "measure")
            .map(_canonical_measure),
            "value": _require_alias(frame, "sector_positions", "value", "value"),
        }
    )
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        _select_column(frame, "source_file"),
    )
    for target, aliases in {
        "z1_series": ("z1_series", "series_id", "fred_series_id"),
        "z1_code": ("z1_code", "series_code"),
        "source_file": ("source_file",),
    }.items():
        column = _select_column(frame, *aliases)
        if column is not None:
            normalized[target] = column.where(column.notna(), "").astype(str).str.strip()
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "value"),
    )

    require_columns(normalized, SECTOR_POSITION_REQUIRED_COLUMNS, dataset_name="sector_positions")
    normalized = validate_numeric_columns(
        normalized,
        ["value"],
        dataset_name="sector_positions",
    )
    normalized = _aggregate_duplicate_rows(
        normalized,
        ["quarter", "sector", "measure"],
        numeric_columns=["value"],
    )
    validate_source_status_columns(normalized, ["source_status"], dataset_name="sector_positions")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="sector_positions")
    validate_duplicate_keys(
        normalized,
        ["quarter", "sector", "measure"],
        dataset_name="sector_positions",
    )
    normalized = filter_quarter_range(
        normalized,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    return normalized.sort_values(["quarter", "sector", "measure"]).reset_index(drop=True)


def load_auction_flows(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load observed auction rows with normalized replay columns."""

    frame = _read_csv(path)
    auction_date = _require_alias(frame, "auction_flows", "auction_date", "auction_date", "record_date")
    issue_date = _require_alias(frame, "auction_flows", "issue_date", "issue_date")
    maturity_date = _require_alias(frame, "auction_flows", "maturity_date", "maturity_date", "mat_date")
    cusip = _require_alias(frame, "auction_flows", "cusip", "cusip", "announcemtd_cusip").astype(str).str.strip()
    quarter = _select_column(frame, "quarter")
    if quarter is None:
        quarter = issue_date
    normalized = pd.DataFrame(
        {
            "quarter": normalize_quarter_series(quarter),
            "auction_date": _normalize_date_column(
                auction_date,
                dataset_name="auction_flows",
                column_name="auction_date",
            ),
            "issue_date": _normalize_date_column(
                issue_date,
                dataset_name="auction_flows",
                column_name="issue_date",
            ),
            "maturity_date": _normalize_date_column(
                maturity_date,
                dataset_name="auction_flows",
                column_name="maturity_date",
            ),
            "cusip": cusip,
            "security_type": _require_alias(
                frame,
                "auction_flows",
                "security_type",
                "security_type",
                "security_type_desc",
            ).map(_canonical_security_type),
            "offering_amount": _require_alias(
                frame,
                "auction_flows",
                "offering_amount",
                "offering_amount",
                "offering_amt",
            ),
        }
    )
    normalized["original_maturity_days"] = _days_between(
        normalized["issue_date"],
        normalized["maturity_date"],
    )
    normalized["original_maturity_years"] = normalized["original_maturity_days"] / 365.25
    normalized["auction_id"] = _coalesce_text_columns(
        _select_column(frame, "auction_id"),
        normalized["cusip"] + "|" + normalized["auction_date"],
    )
    optional_numeric_aliases = {
        "total_accepted": ("total_accepted",),
        "avg_med_yield": ("avg_med_yield", "average_yield"),
        "int_rate": ("int_rate", "interest_rate_pct"),
        "price_per100": ("price_per100", "avg_med_price"),
        "allocation_pctage": ("allocation_pctage",),
    }
    for target, aliases in optional_numeric_aliases.items():
        column = _select_column(frame, *aliases)
        if column is not None:
            normalized[target] = column
    if "int_rate" in normalized.columns:
        normalized["coupon_rate_decimal"] = _percent_to_decimal(normalized["int_rate"])
    if "avg_med_yield" in normalized.columns:
        normalized["issue_yield_decimal"] = _percent_to_decimal(normalized["avg_med_yield"])
    if "price_per100" in normalized.columns:
        normalized["issue_price_ratio"] = _price_per100_to_ratio(normalized["price_per100"])
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        pd.Series(["fiscaldata_auction_row"] * len(normalized.index), index=normalized.index),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "offering_amount"),
    )

    require_columns(normalized, AUCTION_FLOW_REQUIRED_COLUMNS, dataset_name="auction_flows")
    numeric_columns = ["offering_amount", "original_maturity_days", "original_maturity_years"] + [
        column
        for column in [
            "total_accepted",
            "avg_med_yield",
            "int_rate",
            "price_per100",
            "allocation_pctage",
            "coupon_rate_decimal",
            "issue_yield_decimal",
            "issue_price_ratio",
        ]
        if column in normalized.columns
    ]
    normalized = validate_numeric_columns(
        normalized,
        numeric_columns,
        dataset_name="auction_flows",
    )
    validate_source_status_columns(normalized, ["source_status"], dataset_name="auction_flows")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="auction_flows")
    validate_duplicate_keys(
        normalized,
        ["quarter", "auction_id"],
        dataset_name="auction_flows",
    )
    normalized = filter_quarter_range(
        normalized,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    return normalized.sort_values(["quarter", "auction_date", "auction_id"]).reset_index(drop=True)


def load_mspd_cohorts(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load MSPD cohort rows with normalized quarter and cohort columns."""

    frame = _read_csv(path)
    frame = _drop_mspd_summary_rows(frame)
    if "quarter" not in frame.columns and "record_date" in frame.columns:
        frame = _select_latest_record_in_quarter(
            frame,
            date_column="record_date",
            dataset_name="mspd_cohorts",
        )
    issue_date = _require_alias(frame, "mspd_cohorts", "issue_date", "issue_date")
    maturity_date = _require_alias(
        frame,
        "mspd_cohorts",
        "maturity_date",
        "maturity_date",
        "mat_date",
    )
    normalized_issue_date = _normalize_date_column(
        issue_date,
        dataset_name="mspd_cohorts",
        column_name="issue_date",
    )
    normalized_maturity_date = _normalize_date_column(
        maturity_date,
        dataset_name="mspd_cohorts",
        column_name="maturity_date",
    )
    security_class2 = _select_column(frame, "security_class2_desc")
    real_cusip = _coalesce_text_columns(
        _select_column(frame, "cusip"),
        security_class2,
    )
    if real_cusip is None:
        real_cusip = pd.Series([""] * len(frame.index), index=frame.index)
    real_cusip = real_cusip.astype(str).str.strip().str.upper()
    source_cohort_id = _coalesce_text_columns(_select_column(frame, "cohort_id"))
    has_explicit_source_cohort = source_cohort_id is not None
    if source_cohort_id is not None:
        source_cohort_id = source_cohort_id.astype(str).str.strip()
    cohort_id = (
        real_cusip
        + "|"
        + normalized_issue_date.astype(str)
        + "|"
        + normalized_maturity_date.astype(str)
    )
    if source_cohort_id is not None:
        cohort_id = source_cohort_id.where(source_cohort_id.ne(""), cohort_id)
    normalized = pd.DataFrame(
        {
            "quarter": normalize_quarter_series(
                _require_alias(frame, "mspd_cohorts", "quarter", "quarter", "record_date")
            ),
            "cohort_id": cohort_id,
            "cusip": real_cusip,
            "series_cd": _coalesce_text_columns(
                _select_column(frame, "series_cd"),
                pd.Series([""] * len(frame.index), index=frame.index),
            ),
            "security_type": _require_alias(
                frame,
                "mspd_cohorts",
                "security_type",
                "security_type",
                "security_class1_desc",
                "security_type_desc",
            ).map(_canonical_security_type),
            "issue_date": normalized_issue_date,
            "maturity_date": normalized_maturity_date,
            "outstanding": _require_alias(
                frame,
                "mspd_cohorts",
                "outstanding",
                "outstanding",
                "outstanding_amt",
            ),
        }
    )
    positive_outstanding = pd.to_numeric(normalized["outstanding"], errors="coerce").fillna(0.0) > 0.0
    identifiable_security = (
        _valid_cusip_mask(normalized["cusip"])
        | (
            pd.Series(has_explicit_source_cohort, index=normalized.index)
            & normalized["cohort_id"].astype(str).str.strip().ne("")
        )
    ) & normalized["issue_date"].notna() & normalized["maturity_date"].notna()
    normalized = normalized.loc[~positive_outstanding | identifiable_security].copy()
    normalized["original_maturity_days"] = _days_between(
        normalized["issue_date"],
        normalized["maturity_date"],
    )
    normalized["original_maturity_years"] = normalized["original_maturity_days"] / 365.25
    optional_numeric_aliases = {
        "issued_amt": ("issued_amt",),
        "redeemed_amt": ("redeemed_amt",),
        "inflation_adj_amt": ("inflation_adj_amt",),
        "interest_rate_pct": ("interest_rate_pct", "int_rate"),
        "yield_pct": ("yield_pct", "avg_med_yield"),
    }
    for target, aliases in optional_numeric_aliases.items():
        column = _select_column(frame, *aliases)
        if column is not None:
            normalized[target] = column
    if "interest_rate_pct" in normalized.columns:
        normalized["coupon_rate_decimal"] = _percent_to_decimal(normalized["interest_rate_pct"])
    if "yield_pct" in normalized.columns:
        normalized["issue_yield_decimal"] = _percent_to_decimal(normalized["yield_pct"])
    for target, aliases in {
        "interest_pay_date_1": ("interest_pay_date_1",),
        "interest_pay_date_2": ("interest_pay_date_2",),
        "interest_pay_date_3": ("interest_pay_date_3",),
        "interest_pay_date_4": ("interest_pay_date_4",),
    }.items():
        column = _select_column(frame, *aliases)
        if column is not None:
            normalized[target] = column.where(column.notna(), "").astype(str).str.strip()
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        pd.Series(["mspd_cohort_row"] * len(normalized.index), index=normalized.index),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "outstanding"),
    )

    require_columns(normalized, MSPD_COHORT_REQUIRED_COLUMNS, dataset_name="mspd_cohorts")
    numeric_columns = ["outstanding", "original_maturity_days", "original_maturity_years"] + [
        column
        for column in [
            "issued_amt",
            "redeemed_amt",
            "inflation_adj_amt",
            "interest_rate_pct",
            "yield_pct",
            "coupon_rate_decimal",
            "issue_yield_decimal",
        ]
        if column in normalized.columns
    ]
    normalized = validate_numeric_columns(
        normalized,
        numeric_columns,
        dataset_name="mspd_cohorts",
    )
    normalized = _decompose_mspd_reopening_outstanding(normalized)
    additive_numeric_columns = [
        column
        for column in ["outstanding", "issued_amt", "redeemed_amt", "inflation_adj_amt"]
        if column in normalized.columns
    ]
    normalized = _aggregate_duplicate_rows(
        normalized,
        ["quarter", "cohort_id"],
        numeric_columns=additive_numeric_columns,
    )
    normalized = normalized.loc[
        pd.to_numeric(normalized["outstanding"], errors="coerce").fillna(0.0) > 0.0
    ].copy()
    validate_source_status_columns(normalized, ["source_status"], dataset_name="mspd_cohorts")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="mspd_cohorts")
    validate_duplicate_keys(
        normalized,
        ["quarter", "cohort_id"],
        dataset_name="mspd_cohorts",
    )
    normalized = filter_quarter_range(
        normalized,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    return normalized.sort_values(["quarter", "cohort_id"]).reset_index(drop=True)


def enrich_mspd_cohorts_with_security_sources(
    cohorts: pd.DataFrame,
    *,
    auction_path: str | Path | None = None,
    frn_daily_indexes_path: str | Path | None = None,
    tips_cpi_path: str | Path | None = None,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Add source-backed auction, FRN, and TIPS terms to loaded MSPD cohorts."""

    enriched = cohorts.copy()
    if enriched.empty:
        return enriched
    if "cusip" not in enriched.columns:
        enriched["cusip"] = enriched["cohort_id"].astype(str).str.strip()

    auction_terms = _load_auction_term_enrichment(auction_path)
    if not auction_terms.empty:
        enriched = enriched.merge(
            auction_terms,
            on=["cusip", "issue_date", "maturity_date"],
            how="left",
            validate="many_to_one",
        )
        enriched = _fill_from_enrichment(
            enriched,
            {
                "coupon_rate_decimal": "auction_coupon_rate_decimal",
                "issue_yield_decimal": "auction_issue_yield_decimal",
                "issue_price_ratio": "auction_issue_price_ratio",
                "FixedSpread": "auction_fixed_spread_decimal",
                "BenchmarkRate_FRN": "auction_benchmark_rate_decimal",
                "IndexRatio": "auction_index_ratio_on_issue_date",
                "ReferenceCPI_Issue": "auction_ref_cpi_on_issue_date",
                "security_term": "auction_security_term",
                "callable": "auction_callable",
                "call_date": "auction_call_date",
                "called_date": "auction_called_date",
            },
        )

    frn_terms = _load_frn_quarter_enrichment(
        frn_daily_indexes_path,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    if not frn_terms.empty:
        enriched = enriched.merge(
            frn_terms,
            on=["quarter", "cusip"],
            how="left",
            validate="many_to_one",
        )
        enriched = _overwrite_from_enrichment(
            enriched,
            {
                "AccruedInterest_FRN_Per100": "frn_accrued_interest_per100",
                "BenchmarkRate_FRN": "frn_benchmark_rate_decimal",
                "FixedSpread": "frn_fixed_spread_decimal",
            },
        )

    tips_terms = _load_tips_quarter_enrichment(
        tips_cpi_path,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    if not tips_terms.empty:
        enriched = enriched.merge(
            tips_terms,
            on=["quarter", "cusip"],
            how="left",
            validate="many_to_one",
        )
        enriched = _overwrite_from_enrichment(
            enriched,
            {
                "IndexRatio": "tips_index_ratio_end",
                "ReferenceCPI_End": "tips_ref_cpi_end",
            },
        )

    status_cols = [
        column
        for column in [
            "auction_enrichment_status",
            "frn_enrichment_status",
            "tips_enrichment_status",
        ]
        if column in enriched.columns
    ]
    if status_cols:
        enriched["security_enrichment_status"] = enriched[status_cols].apply(
            lambda row: ";".join(
                sorted(
                    {
                        str(value)
                        for value in row.dropna().tolist()
                        if str(value).strip()
                    }
                )
            ),
            axis=1,
        )
        enriched["security_enrichment_status"] = enriched["security_enrichment_status"].replace("", pd.NA)
    return enriched


def _load_auction_term_enrichment(path: str | Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    frame = _read_csv(path)
    required = ["cusip", "issue_date", "maturity_date"]
    if not set(required).issubset(frame.columns):
        return pd.DataFrame()
    normalized = pd.DataFrame(
        {
            "cusip": frame["cusip"].astype(str).str.strip(),
            "issue_date": _normalize_date_column(
                frame["issue_date"],
                dataset_name="auction_terms",
                column_name="issue_date",
            ),
            "maturity_date": _normalize_date_column(
                frame["maturity_date"],
                dataset_name="auction_terms",
                column_name="maturity_date",
            ),
        }
    )
    optional_text = {
        "auction_security_term": "security_term",
        "auction_callable": "callable",
    }
    for target, source in optional_text.items():
        if source in frame.columns:
            normalized[target] = frame[source].where(frame[source].notna(), "").astype(str).str.strip()
    for target, source in {
        "auction_call_date": "call_date",
        "auction_called_date": "called_date",
    }.items():
        if source in frame.columns:
            normalized[target] = _normalize_optional_date_column(
                frame[source],
                dataset_name="auction_terms",
                column_name=source,
            )
    if "int_rate" in frame.columns:
        normalized["auction_coupon_rate_decimal"] = _percent_to_decimal(frame["int_rate"])
    for target, sources in {
        "auction_issue_yield_decimal": ["avg_med_yield", "high_yield", "avg_med_investment_rate"],
        "auction_issue_price_ratio": ["price_per100", "avg_med_price", "high_price"],
        "auction_fixed_spread_decimal": ["spread"],
        "auction_benchmark_rate_decimal": ["frn_index_determination_rate"],
        "auction_index_ratio_on_issue_date": ["index_ratio_on_issue_date"],
        "auction_ref_cpi_on_issue_date": ["ref_cpi_on_issue_date"],
    }.items():
        source = _select_column(frame, *sources)
        if source is None:
            continue
        if target in {"auction_issue_price_ratio"}:
            normalized[target] = _price_per100_to_ratio(source)
        elif target in {
            "auction_issue_yield_decimal",
            "auction_fixed_spread_decimal",
            "auction_benchmark_rate_decimal",
        }:
            normalized[target] = _percent_to_decimal(source)
        else:
            normalized[target] = pd.to_numeric(source, errors="coerce")
    normalized["auction_enrichment_status"] = "auction_terms_joined"
    numeric_cols = [
        column
        for column in normalized.columns
        if column
        not in {
            "cusip",
            "issue_date",
            "maturity_date",
            "auction_security_term",
            "auction_callable",
            "auction_call_date",
            "auction_called_date",
            "auction_enrichment_status",
        }
    ]
    normalized = validate_numeric_columns(normalized, numeric_cols, dataset_name="auction_terms")
    return _aggregate_enrichment(normalized, ["cusip", "issue_date", "maturity_date"])


def _load_frn_quarter_enrichment(
    path: str | Path | None,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    frame = _read_csv(path)
    if not {"record_date", "cusip"}.issubset(frame.columns):
        return pd.DataFrame()
    parsed = pd.to_datetime(frame["record_date"], errors="coerce")
    normalized = pd.DataFrame(
        {
            "quarter": parsed.dt.to_period("Q").astype(str),
            "record_date": parsed,
            "cusip": frame["cusip"].astype(str).str.strip(),
        }
    )
    if "accr_int_per100_pmt_period" in frame.columns:
        normalized["frn_accrued_interest_per100"] = pd.to_numeric(
            frame["accr_int_per100_pmt_period"],
            errors="coerce",
        )
    if "daily_index" in frame.columns:
        normalized["frn_benchmark_rate_decimal"] = _percent_to_decimal(frame["daily_index"])
    if "spread" in frame.columns:
        normalized["frn_fixed_spread_decimal"] = _percent_to_decimal(frame["spread"])
    normalized["frn_enrichment_status"] = "frn_daily_index_joined"
    normalized = filter_quarter_range(
        normalized,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    if normalized.empty:
        return pd.DataFrame()
    latest = normalized.groupby(["quarter", "cusip"])["record_date"].transform("max")
    normalized = normalized.loc[normalized["record_date"].eq(latest)].copy()
    normalized = normalized.drop(columns=["record_date"])
    return _aggregate_enrichment(normalized, ["quarter", "cusip"])


def _load_tips_quarter_enrichment(
    path: str | Path | None,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    frame = _read_csv(path)
    if not {"cusip", "index_date"}.issubset(frame.columns):
        return pd.DataFrame()
    parsed = pd.to_datetime(frame["index_date"], errors="coerce")
    normalized = pd.DataFrame(
        {
            "quarter": parsed.dt.to_period("Q").astype(str),
            "index_date": parsed,
            "cusip": frame["cusip"].astype(str).str.strip(),
        }
    )
    if "ref_cpi" in frame.columns:
        normalized["tips_ref_cpi_end"] = pd.to_numeric(frame["ref_cpi"], errors="coerce")
    if "index_ratio" in frame.columns:
        normalized["tips_index_ratio_end"] = pd.to_numeric(frame["index_ratio"], errors="coerce")
    normalized["tips_enrichment_status"] = "tips_cpi_detail_joined"
    normalized = filter_quarter_range(
        normalized,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    if normalized.empty:
        return pd.DataFrame()
    latest = normalized.groupby(["quarter", "cusip"])["index_date"].transform("max")
    normalized = normalized.loc[normalized["index_date"].eq(latest)].copy()
    normalized = normalized.drop(columns=["index_date"])
    return _aggregate_enrichment(normalized, ["quarter", "cusip"])


def _aggregate_enrichment(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    agg = {}
    for column in frame.columns:
        if column in keys:
            continue
        agg[column] = _first_non_null
    return frame.groupby(keys, sort=False, dropna=False).agg(agg).reset_index()


def _fill_from_enrichment(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    for target, source in mapping.items():
        if source not in out.columns:
            continue
        if target not in out.columns:
            out[target] = out[source]
        else:
            out[target] = out[target].where(out[target].notna(), out[source])
    return out


def _overwrite_from_enrichment(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    for target, source in mapping.items():
        if source not in out.columns:
            continue
        if target not in out.columns:
            out[target] = out[source]
        else:
            out[target] = out[target].where(out[source].isna(), out[source])
    return out
