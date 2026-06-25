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


def _quarter_start_timestamp(quarter: str | None) -> pd.Timestamp | None:
    if quarter is None:
        return None
    return pd.Period(str(quarter), freq="Q").start_time.normalize()


def _quarter_end_exclusive_timestamp(quarter: str | None) -> pd.Timestamp | None:
    if quarter is None:
        return None
    return (pd.Period(str(quarter), freq="Q") + 1).start_time.normalize()


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


def load_interest_auction_lots(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load auction lots for aggregate interest-flow calculations.

    Unlike ``load_auction_flows``, this filters by accrual-window overlap rather
    than issue quarter so pre-window bills that accrue inside the replay window
    are retained.
    """

    frame = _read_csv(path)
    auction_date = _require_alias(frame, "interest_auction_lots", "auction_date", "auction_date", "record_date")
    issue_date = _require_alias(frame, "interest_auction_lots", "issue_date", "issue_date")
    maturity_date = _require_alias(frame, "interest_auction_lots", "maturity_date", "maturity_date", "mat_date")
    cusip = _require_alias(frame, "interest_auction_lots", "cusip", "cusip", "announcemtd_cusip").astype(str).str.strip()
    security_type = _require_alias(
        frame,
        "interest_auction_lots",
        "security_type",
        "security_type",
        "security_type_desc",
    ).map(_canonical_security_type)
    normalized = pd.DataFrame(
        {
            "auction_date": _normalize_date_column(
                auction_date,
                dataset_name="interest_auction_lots",
                column_name="auction_date",
            ),
            "issue_date": _normalize_date_column(
                issue_date,
                dataset_name="interest_auction_lots",
                column_name="issue_date",
            ),
            "maturity_date": _normalize_date_column(
                maturity_date,
                dataset_name="interest_auction_lots",
                column_name="maturity_date",
            ),
            "cusip": cusip,
            "security_type": security_type,
            "security_term": _coalesce_text_columns(_select_column(frame, "security_term")),
            "price_per100": _select_column(frame, "price_per100", "avg_med_price"),
            "total_accepted": _require_alias(
                frame,
                "interest_auction_lots",
                "total_accepted",
                "total_accepted",
            ),
            "offering_amt": _select_column(frame, "offering_amt", "offering_amount"),
        }
    )
    accepted_columns = [
        "comp_accepted",
        "noncomp_accepted",
        "soma_accepted",
        "fima_noncomp_accepted",
        "treas_retail_accepted",
    ]
    for column in accepted_columns:
        source = _select_column(frame, column)
        if source is not None:
            normalized[column] = source
    available_accepted = [column for column in accepted_columns if column in normalized.columns]
    if available_accepted:
        normalized["accepted_component_sum"] = normalized[available_accepted].apply(
            lambda row: pd.to_numeric(row, errors="coerce").sum(min_count=1),
            axis=1,
        )
    else:
        normalized["accepted_component_sum"] = pd.NA
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        pd.Series(["fiscaldata_auction_row"] * len(normalized.index), index=normalized.index),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "price_per100", "total_accepted"),
    )
    normalized = normalized.loc[normalized["security_type"].eq("bill")].copy()
    if normalized.empty:
        return normalized.assign(lot_id=pd.Series(dtype=str), quarter=pd.Series(dtype=str))
    normalized["lot_id"] = (
        normalized["cusip"].astype(str)
        + "|"
        + normalized["auction_date"].astype(str)
        + "|"
        + normalized["issue_date"].astype(str)
    )
    normalized["issue_quarter"] = normalize_quarter_series(normalized["issue_date"])
    numeric_columns = [
        "price_per100",
        "total_accepted",
        "offering_amt",
        "accepted_component_sum",
        *available_accepted,
    ]
    normalized = validate_numeric_columns(
        normalized,
        [column for column in numeric_columns if column in normalized.columns],
        dataset_name="interest_auction_lots",
    )
    validate_duplicate_keys(normalized, ["lot_id"], dataset_name="interest_auction_lots")
    validate_source_status_columns(normalized, ["source_status"], dataset_name="interest_auction_lots")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="interest_auction_lots")

    window_start = _quarter_start_timestamp(start_quarter)
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    issue_ts = pd.to_datetime(normalized["issue_date"], errors="coerce")
    maturity_ts = pd.to_datetime(normalized["maturity_date"], errors="coerce")
    if window_start is not None:
        normalized = normalized.loc[maturity_ts > window_start].copy()
        issue_ts = issue_ts.loc[normalized.index]
        maturity_ts = maturity_ts.loc[normalized.index]
    if window_end is not None:
        normalized = normalized.loc[issue_ts < window_end].copy()
    return normalized.sort_values(["issue_date", "auction_date", "lot_id"], kind="stable").reset_index(drop=True)


def load_nonbill_discount_premium_auction_lots(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load nonbill auction lots for source-led discount/premium amortization."""

    frame = _read_csv(path)
    output_columns = [
        "lot_id",
        "cusip",
        "instrument_family",
        "auction_date",
        "issue_date",
        "dated_date",
        "first_interest_payment_date",
        "maturity_date",
        "called_date",
        "is_reopening",
        "auction_format",
        "par_mil",
        "coupon_rate_decimal",
        "clean_price_per100",
        "price_source_tier",
        "high_price_per100",
        "low_price_per100",
        "reported_yield_pct",
        "index_ratio_on_issue_date",
        "ref_cpi_on_dated_date",
        "source_status",
        "evidence_label",
    ]
    auction_date = _require_alias(frame, "nonbill_discount_premium_auction_lots", "auction_date", "auction_date", "record_date")
    issue_date = _require_alias(frame, "nonbill_discount_premium_auction_lots", "issue_date", "issue_date")
    maturity_date = _require_alias(
        frame,
        "nonbill_discount_premium_auction_lots",
        "maturity_date",
        "maturity_date",
        "mat_date",
    )
    security_type = _require_alias(
        frame,
        "nonbill_discount_premium_auction_lots",
        "security_type",
        "security_type",
        "security_type_desc",
    ).map(_canonical_security_type)
    floating_rate = _coalesce_text_columns(_select_column(frame, "floating_rate"))
    inflation_index = _coalesce_text_columns(_select_column(frame, "inflation_index_security"))
    if floating_rate is None:
        floating_rate = pd.Series([""] * len(frame.index), index=frame.index)
    if inflation_index is None:
        inflation_index = pd.Series([""] * len(frame.index), index=frame.index)
    cusip = _require_alias(frame, "nonbill_discount_premium_auction_lots", "cusip", "cusip", "announcemtd_cusip").astype(str).str.strip().str.upper()
    auction_date_norm = _normalize_date_column(
        auction_date,
        dataset_name="nonbill_discount_premium_auction_lots",
        column_name="auction_date",
    )
    issue_date_norm = _normalize_date_column(
        issue_date,
        dataset_name="nonbill_discount_premium_auction_lots",
        column_name="issue_date",
    )
    price_per100 = pd.to_numeric(_select_column(frame, "price_per100"), errors="coerce")
    avg_med_price = pd.to_numeric(_select_column(frame, "avg_med_price"), errors="coerce")
    unadj_price = pd.to_numeric(_select_column(frame, "unadj_price"), errors="coerce")
    adj_price = pd.to_numeric(_select_column(frame, "adj_price"), errors="coerce")
    index_ratio = pd.to_numeric(_select_column(frame, "index_ratio_on_issue_date"), errors="coerce")
    clean_price = price_per100.where(price_per100.notna(), avg_med_price)
    price_source_tier = pd.Series("price_per100", index=frame.index, dtype=object)
    price_source_tier = price_source_tier.where(price_per100.notna(), "avg_med_price")
    tips_price = unadj_price.where(unadj_price.gt(0.0), adj_price / index_ratio.where(index_ratio.gt(0.0)))
    tips_price_tier = pd.Series("unadj_price", index=frame.index, dtype=object)
    tips_price_tier = tips_price_tier.where(unadj_price.gt(0.0), "adj_price_div_index_ratio")
    is_tips = security_type.eq("tips") | inflation_index.astype(str).str.lower().eq("yes")
    is_frn = security_type.eq("frn") | floating_rate.astype(str).str.lower().eq("yes")
    is_nominal = security_type.isin(["note", "bond"]) & ~is_tips & ~is_frn
    clean_price = clean_price.where(~is_tips, tips_price)
    price_source_tier = price_source_tier.where(~is_tips, tips_price_tier)
    normalized = pd.DataFrame(
        {
            "lot_id": cusip + "|" + auction_date_norm.astype(str) + "|" + issue_date_norm.astype(str),
            "cusip": cusip,
            "instrument_family": security_type.map(
                {
                    "note": "Treasury Notes",
                    "bond": "Treasury Bonds",
                    "tips": "Treasury Inflation Protected Securities (TIPS)",
                    "frn": "Treasury Floating Rate Notes (FRN)",
                }
            ),
            "auction_date": auction_date_norm,
            "issue_date": issue_date_norm,
            "dated_date": _normalize_optional_date_column(
                _select_column(frame, "dated_date", "original_dated_date")
                if _select_column(frame, "dated_date", "original_dated_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="nonbill_discount_premium_auction_lots",
                column_name="dated_date",
            ),
            "first_interest_payment_date": _normalize_optional_date_column(
                _select_column(frame, "first_int_payment_date")
                if _select_column(frame, "first_int_payment_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="nonbill_discount_premium_auction_lots",
                column_name="first_interest_payment_date",
            ),
            "maturity_date": _normalize_date_column(
                maturity_date,
                dataset_name="nonbill_discount_premium_auction_lots",
                column_name="maturity_date",
            ),
            "called_date": _normalize_optional_date_column(
                _select_column(frame, "called_date", "call_date")
                if _select_column(frame, "called_date", "call_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="nonbill_discount_premium_auction_lots",
                column_name="called_date",
            ),
            "is_reopening": _coalesce_text_columns(_select_column(frame, "reopening")).fillna("").str.lower().eq("yes")
            if _select_column(frame, "reopening") is not None
            else pd.Series([False] * len(frame.index), index=frame.index),
            "auction_format": _coalesce_text_columns(_select_column(frame, "auction_format")),
            "par_mil": pd.to_numeric(
                _require_alias(frame, "nonbill_discount_premium_auction_lots", "total_accepted", "total_accepted"),
                errors="coerce",
            )
            / 1_000_000.0,
            "coupon_rate_decimal": _percent_to_decimal(
                _select_column(frame, "int_rate", "interest_rate_pct")
                if _select_column(frame, "int_rate", "interest_rate_pct") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index)
            ),
            "clean_price_per100": clean_price,
            "price_source_tier": price_source_tier,
            "high_price_per100": _select_column(frame, "high_price"),
            "low_price_per100": _select_column(frame, "low_price"),
            "reported_yield_pct": _select_column(frame, "high_yield", "avg_med_yield"),
            "index_ratio_on_issue_date": _select_column(frame, "index_ratio_on_issue_date"),
            "ref_cpi_on_dated_date": _select_column(frame, "ref_cpi_on_dated_date"),
            "_include": is_nominal | is_tips | is_frn,
            "_issue_ts": pd.to_datetime(issue_date_norm, errors="coerce"),
            "_end_ts": pd.to_datetime(
                _normalize_date_column(
                    maturity_date,
                    dataset_name="nonbill_discount_premium_auction_lots",
                    column_name="maturity_date",
                ),
                errors="coerce",
            ),
        }
    )
    normalized.loc[is_frn, "instrument_family"] = "Treasury Floating Rate Notes (FRN)"
    normalized.loc[is_tips, "instrument_family"] = "Treasury Inflation Protected Securities (TIPS)"
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        pd.Series(["fiscaldata_nonbill_discount_premium_auction_row"] * len(normalized.index), index=normalized.index),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "par_mil", "clean_price_per100", "coupon_rate_decimal"),
    )
    normalized = normalized.loc[normalized["_include"] & _valid_cusip_mask(normalized["cusip"])].copy()
    if normalized.empty:
        return pd.DataFrame(columns=output_columns)
    numeric_columns = [
        "par_mil",
        "coupon_rate_decimal",
        "clean_price_per100",
        "high_price_per100",
        "low_price_per100",
        "reported_yield_pct",
        "index_ratio_on_issue_date",
        "ref_cpi_on_dated_date",
    ]
    normalized = validate_numeric_columns(
        normalized,
        numeric_columns,
        dataset_name="nonbill_discount_premium_auction_lots",
    )
    validate_duplicate_keys(normalized, ["lot_id"], dataset_name="nonbill_discount_premium_auction_lots")
    validate_source_status_columns(normalized, ["source_status"], dataset_name="nonbill_discount_premium_auction_lots")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="nonbill_discount_premium_auction_lots")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        normalized = normalized.loc[normalized["_end_ts"] > window_start].copy()
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[normalized["_issue_ts"] < window_end].copy()
    return (
        normalized.loc[:, output_columns]
        .sort_values(["issue_date", "auction_date", "lot_id"], kind="stable")
        .reset_index(drop=True)
    )


def load_fixed_coupon_monthly_stocks(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load monthly MSPD Notes/Bonds CUSIP stocks for aggregate coupon accrual."""

    frame = _read_csv(path)
    frame = _drop_mspd_summary_rows(frame)
    output_columns = [
        "record_date",
        "quarter",
        "cusip",
        "security_class",
        "outstanding_mil",
        "coupon_rate_decimal",
        "maturity_date",
        "interest_pay_date_1",
        "interest_pay_date_2",
        "source_status",
        "evidence_label",
    ]
    if _select_column(frame, "record_date") is None:
        return pd.DataFrame(columns=output_columns)
    record_date = _require_alias(frame, "fixed_coupon_monthly_stocks", "record_date", "record_date")
    security_type_desc = _require_alias(
        frame,
        "fixed_coupon_monthly_stocks",
        "security_type_desc",
        "security_type_desc",
    )
    security_class = _require_alias(
        frame,
        "fixed_coupon_monthly_stocks",
        "security_class",
        "security_class1_desc",
        "security_type",
    ).astype(str).str.strip()
    cusip = _require_alias(
        frame,
        "fixed_coupon_monthly_stocks",
        "cusip",
        "security_class2_desc",
        "cusip",
    ).astype(str).str.strip().str.upper()
    normalized = pd.DataFrame(
        {
            "record_date": _normalize_date_column(
                record_date,
                dataset_name="fixed_coupon_monthly_stocks",
                column_name="record_date",
            ),
            "security_type_desc": security_type_desc.astype(str).str.strip(),
            "security_class": security_class,
            "cusip": cusip,
            "outstanding_mil": _require_alias(
                frame,
                "fixed_coupon_monthly_stocks",
                "outstanding_mil",
                "outstanding_amt",
                "outstanding",
            ),
            "coupon_rate_decimal": _percent_to_decimal(
                _require_alias(
                    frame,
                    "fixed_coupon_monthly_stocks",
                    "interest_rate_pct",
                    "interest_rate_pct",
                    "int_rate",
                )
            ),
            "maturity_date": _normalize_date_column(
                _require_alias(
                    frame,
                    "fixed_coupon_monthly_stocks",
                    "maturity_date",
                    "maturity_date",
                    "mat_date",
                ),
                dataset_name="fixed_coupon_monthly_stocks",
                column_name="maturity_date",
            ),
            "interest_pay_date_1": _require_alias(
                frame,
                "fixed_coupon_monthly_stocks",
                "interest_pay_date_1",
                "interest_pay_date_1",
            ).where(lambda values: values.notna(), "").astype(str).str.strip(),
            "interest_pay_date_2": _require_alias(
                frame,
                "fixed_coupon_monthly_stocks",
                "interest_pay_date_2",
                "interest_pay_date_2",
            ).where(lambda values: values.notna(), "").astype(str).str.strip(),
        }
    )
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        pd.Series(["mspd_monthly_coupon_stock_row"] * len(normalized.index), index=normalized.index),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "outstanding_mil", "coupon_rate_decimal"),
    )
    normalized = normalized.loc[
        normalized["security_type_desc"].str.lower().eq("marketable")
        & normalized["security_class"].isin(["Notes", "Bonds"])
        & _valid_cusip_mask(normalized["cusip"])
    ].copy()
    if normalized.empty:
        return normalized.assign(quarter=pd.Series(dtype=str))
    normalized = validate_numeric_columns(
        normalized,
        ["outstanding_mil", "coupon_rate_decimal"],
        dataset_name="fixed_coupon_monthly_stocks",
    )
    normalized = normalized.loc[pd.to_numeric(normalized["outstanding_mil"], errors="coerce").fillna(0.0) > 0.0].copy()
    term_columns = [
        "security_class",
        "coupon_rate_decimal",
        "maturity_date",
        "interest_pay_date_1",
        "interest_pay_date_2",
    ]
    inconsistent_terms = []
    duplicate_mask = normalized.duplicated(["record_date", "cusip"], keep=False)
    if duplicate_mask.any():
        duplicates = normalized.loc[duplicate_mask].copy()
        for key, group in duplicates.groupby(["record_date", "cusip"], sort=False, dropna=False):
            for column in term_columns:
                values = group[column].dropna().astype(str).str.strip()
                values = values.loc[values.ne("")]
                if values.nunique() > 1:
                    inconsistent_terms.append((key, column, values.unique().tolist()[:3]))
    if inconsistent_terms:
        key, column, values = inconsistent_terms[0]
        raise ValueError(
            "fixed_coupon_monthly_stocks has inconsistent duplicate CUSIP-month "
            f"terms for {key} column {column}: {values}"
        )
    agg_map = {
        "security_type_desc": _first_non_null,
        "security_class": _first_non_null,
        "outstanding_mil": _sum_preserving_missing,
        "coupon_rate_decimal": _first_non_null,
        "maturity_date": _first_non_null,
        "interest_pay_date_1": _first_non_null,
        "interest_pay_date_2": _first_non_null,
        "source_status": _first_non_null,
        "evidence_label": _first_non_null,
    }
    normalized = (
        normalized.groupby(["record_date", "cusip"], sort=False, dropna=False)
        .agg(agg_map)
        .reset_index()
    )
    validate_source_status_columns(normalized, ["source_status"], dataset_name="fixed_coupon_monthly_stocks")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="fixed_coupon_monthly_stocks")
    validate_duplicate_keys(normalized, ["record_date", "cusip"], dataset_name="fixed_coupon_monthly_stocks")
    record_ts = pd.to_datetime(normalized["record_date"], errors="coerce")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        prior_month_end = window_start - pd.offsets.MonthEnd(1)
        normalized = normalized.loc[record_ts >= prior_month_end].copy()
        record_ts = record_ts.loc[normalized.index]
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[record_ts < window_end].copy()
    normalized["quarter"] = pd.to_datetime(normalized["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    return (
        normalized.loc[:, output_columns]
        .sort_values(["record_date", "cusip"], kind="stable")
        .reset_index(drop=True)
    )


def load_fixed_coupon_auction_events(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load nominal fixed-coupon Note/Bond auction events for interest accrual."""

    frame = _read_csv(path)
    auction_date = _require_alias(frame, "fixed_coupon_auction_events", "auction_date", "auction_date", "record_date")
    issue_date = _require_alias(frame, "fixed_coupon_auction_events", "issue_date", "issue_date")
    maturity_date = _require_alias(
        frame,
        "fixed_coupon_auction_events",
        "maturity_date",
        "maturity_date",
        "mat_date",
    )
    security_type = _require_alias(
        frame,
        "fixed_coupon_auction_events",
        "security_type",
        "security_type",
        "security_type_desc",
    ).map(_canonical_security_type)
    floating_rate = _coalesce_text_columns(_select_column(frame, "floating_rate"))
    inflation_index = _coalesce_text_columns(_select_column(frame, "inflation_index_security"))
    if floating_rate is None:
        floating_rate = pd.Series([""] * len(frame.index), index=frame.index)
    if inflation_index is None:
        inflation_index = pd.Series([""] * len(frame.index), index=frame.index)
    normalized = pd.DataFrame(
        {
            "event_id": (
                _require_alias(frame, "fixed_coupon_auction_events", "cusip", "cusip", "announcemtd_cusip")
                .astype(str)
                .str.strip()
                .str.upper()
                + "|"
                + _normalize_date_column(
                    auction_date,
                    dataset_name="fixed_coupon_auction_events",
                    column_name="auction_date",
                ).astype(str)
                + "|"
                + _normalize_date_column(
                    issue_date,
                    dataset_name="fixed_coupon_auction_events",
                    column_name="issue_date",
                ).astype(str)
            ),
            "cusip": _require_alias(frame, "fixed_coupon_auction_events", "cusip", "cusip", "announcemtd_cusip")
            .astype(str)
            .str.strip()
            .str.upper(),
            "security_class": security_type.map({"note": "Notes", "bond": "Bonds"}),
            "auction_date": _normalize_date_column(
                auction_date,
                dataset_name="fixed_coupon_auction_events",
                column_name="auction_date",
            ),
            "issue_date": _normalize_date_column(
                issue_date,
                dataset_name="fixed_coupon_auction_events",
                column_name="issue_date",
            ),
            "maturity_date": _normalize_date_column(
                maturity_date,
                dataset_name="fixed_coupon_auction_events",
                column_name="maturity_date",
            ),
            "called_date": _normalize_optional_date_column(
                _select_column(frame, "called_date", "call_date")
                if _select_column(frame, "called_date", "call_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="fixed_coupon_auction_events",
                column_name="called_date",
            ),
            "total_accepted_mil": pd.to_numeric(
                _require_alias(
                    frame,
                    "fixed_coupon_auction_events",
                    "total_accepted",
                    "total_accepted",
                ),
                errors="coerce",
            )
            / 1_000_000.0,
            "coupon_rate_decimal": _percent_to_decimal(
                _select_column(frame, "int_rate", "interest_rate_pct")
                if _select_column(frame, "int_rate", "interest_rate_pct") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index)
            ),
            "floating_rate": floating_rate.astype(str).str.strip(),
            "inflation_index_security": inflation_index.astype(str).str.strip(),
        }
    )
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        pd.Series(["fiscaldata_fixed_coupon_auction_row"] * len(normalized.index), index=normalized.index),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "total_accepted_mil"),
    )
    normalized = normalized.loc[
        normalized["security_class"].isin(["Notes", "Bonds"])
        & normalized["floating_rate"].str.lower().eq("no")
        & normalized["inflation_index_security"].str.lower().eq("no")
        & _valid_cusip_mask(normalized["cusip"])
    ].copy()
    if normalized.empty:
        return normalized.drop(columns=["floating_rate", "inflation_index_security"])
    normalized = validate_numeric_columns(
        normalized,
        ["total_accepted_mil", "coupon_rate_decimal"],
        dataset_name="fixed_coupon_auction_events",
    )
    validate_duplicate_keys(normalized, ["event_id"], dataset_name="fixed_coupon_auction_events")
    validate_source_status_columns(normalized, ["source_status"], dataset_name="fixed_coupon_auction_events")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="fixed_coupon_auction_events")
    issue_ts = pd.to_datetime(normalized["issue_date"], errors="coerce")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        normalized = normalized.loc[issue_ts >= window_start].copy()
        issue_ts = issue_ts.loc[normalized.index]
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[issue_ts < window_end].copy()
    normalized["quarter"] = pd.to_datetime(normalized["issue_date"], errors="coerce").dt.to_period("Q").astype(str)
    return (
        normalized.drop(columns=["floating_rate", "inflation_index_security"])
        .sort_values(["issue_date", "auction_date", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def load_frn_benchmark_auctions(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load 13-week bill auctions used as FRN benchmark-rate events."""

    frame = _read_csv(path)
    security_type = _require_alias(
        frame,
        "frn_benchmark_auctions",
        "security_type",
        "security_type",
        "security_type_desc",
    ).map(_canonical_security_type)
    security_term = _coalesce_text_columns(_select_column(frame, "security_term"))
    if security_term is None:
        security_term = pd.Series([""] * len(frame.index), index=frame.index)
    normalized = pd.DataFrame(
        {
            "auction_date": _normalize_date_column(
                _require_alias(frame, "frn_benchmark_auctions", "auction_date", "auction_date", "record_date"),
                dataset_name="frn_benchmark_auctions",
                column_name="auction_date",
            ),
            "issue_date": _normalize_date_column(
                _require_alias(frame, "frn_benchmark_auctions", "issue_date", "issue_date"),
                dataset_name="frn_benchmark_auctions",
                column_name="issue_date",
            ),
            "maturity_date": _normalize_date_column(
                _require_alias(frame, "frn_benchmark_auctions", "maturity_date", "maturity_date", "mat_date"),
                dataset_name="frn_benchmark_auctions",
                column_name="maturity_date",
            ),
            "security_type": security_type,
            "security_term": security_term.astype(str).str.strip(),
            "high_discnt_rate": _require_alias(
                frame,
                "frn_benchmark_auctions",
                "high_discnt_rate",
                "high_discnt_rate",
            ),
        }
    )
    normalized = normalized.loc[
        normalized["security_type"].eq("bill")
        & normalized["security_term"].str.lower().str.contains("13-week", na=False)
    ].copy()
    if normalized.empty:
        return normalized
    normalized = validate_numeric_columns(
        normalized,
        ["high_discnt_rate"],
        dataset_name="frn_benchmark_auctions",
    )
    issue = pd.to_datetime(normalized["issue_date"], errors="coerce")
    maturity = pd.to_datetime(normalized["maturity_date"], errors="coerce")
    normalized["term_days"] = (maturity - issue).dt.days
    discount = pd.to_numeric(normalized["high_discnt_rate"], errors="coerce")
    term_days = pd.to_numeric(normalized["term_days"], errors="coerce")
    normalized["benchmark_index_pct"] = (
        discount / (1.0 - (discount / 100.0) * term_days / 360.0)
    ).round(9)
    normalized["effective_date"] = (
        pd.to_datetime(normalized["auction_date"], errors="coerce") + pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        normalized = normalized.loc[pd.to_datetime(normalized["effective_date"], errors="coerce") >= window_start - pd.Timedelta(days=14)].copy()
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[pd.to_datetime(normalized["auction_date"], errors="coerce") < window_end].copy()
    return (
        normalized.loc[
            :,
            [
                "auction_date",
                "issue_date",
                "maturity_date",
                "security_term",
                "high_discnt_rate",
                "term_days",
                "benchmark_index_pct",
                "effective_date",
            ],
        ]
        .sort_values(["effective_date", "auction_date"], kind="stable")
        .reset_index(drop=True)
    )


def load_frn_auction_lots(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load FRN auction principal lots for source-only interest accrual."""

    frame = _read_csv(path)
    floating_rate = _coalesce_text_columns(_select_column(frame, "floating_rate"))
    if floating_rate is None:
        floating_rate = pd.Series([""] * len(frame.index), index=frame.index)
    cusip = _require_alias(frame, "frn_auction_lots", "cusip", "cusip", "announcemtd_cusip").astype(str).str.strip().str.upper()
    auction_date = _require_alias(frame, "frn_auction_lots", "auction_date", "auction_date", "record_date")
    issue_date = _require_alias(frame, "frn_auction_lots", "issue_date", "issue_date")
    maturity_date = _require_alias(frame, "frn_auction_lots", "maturity_date", "maturity_date", "mat_date")
    normalized = pd.DataFrame(
        {
            "lot_id": (
                cusip
                + "|"
                + _normalize_date_column(
                    auction_date,
                    dataset_name="frn_auction_lots",
                    column_name="auction_date",
                ).astype(str)
                + "|"
                + _normalize_date_column(
                    issue_date,
                    dataset_name="frn_auction_lots",
                    column_name="issue_date",
                ).astype(str)
            ),
            "cusip": cusip,
            "auction_date": _normalize_date_column(
                auction_date,
                dataset_name="frn_auction_lots",
                column_name="auction_date",
            ),
            "issue_date": _normalize_date_column(
                issue_date,
                dataset_name="frn_auction_lots",
                column_name="issue_date",
            ),
            "maturity_date": _normalize_date_column(
                maturity_date,
                dataset_name="frn_auction_lots",
                column_name="maturity_date",
            ),
            "dated_date": _normalize_optional_date_column(
                _select_column(frame, "dated_date")
                if _select_column(frame, "dated_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="frn_auction_lots",
                column_name="dated_date",
            ),
            "original_dated_date": _normalize_optional_date_column(
                _select_column(frame, "original_dated_date", "original_issue_date")
                if _select_column(frame, "original_dated_date", "original_issue_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="frn_auction_lots",
                column_name="original_dated_date",
            ),
            "security_term": _coalesce_text_columns(_select_column(frame, "security_term")),
            "floating_rate": floating_rate.astype(str).str.strip(),
            "reopening": _coalesce_text_columns(
                _select_column(frame, "reopening"),
                pd.Series([""] * len(frame.index), index=frame.index),
            ),
            "spread_pct": _select_column(frame, "spread"),
            "total_accepted_mil": pd.to_numeric(
                _require_alias(frame, "frn_auction_lots", "total_accepted", "total_accepted"),
                errors="coerce",
            )
            / 1_000_000.0,
            "frn_index_determination_date": _normalize_optional_date_column(
                _select_column(frame, "frn_index_determination_date")
                if _select_column(frame, "frn_index_determination_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="frn_auction_lots",
                column_name="frn_index_determination_date",
            ),
            "frn_index_determination_rate": _select_column(frame, "frn_index_determination_rate"),
        }
    )
    normalized = normalized.loc[
        normalized["floating_rate"].str.lower().eq("yes")
        & _valid_cusip_mask(normalized["cusip"])
    ].copy()
    if normalized.empty:
        return normalized.drop(columns=["floating_rate"])
    normalized["reopening"] = normalized["reopening"].where(normalized["reopening"].notna(), "").astype(str).str.strip()
    normalized["is_reopening"] = normalized["reopening"].str.lower().eq("yes")
    normalized = validate_numeric_columns(
        normalized,
        ["spread_pct", "total_accepted_mil", "frn_index_determination_rate"],
        dataset_name="frn_auction_lots",
    )
    validate_duplicate_keys(normalized, ["lot_id"], dataset_name="frn_auction_lots")
    issue_ts = pd.to_datetime(normalized["issue_date"], errors="coerce")
    maturity_ts = pd.to_datetime(normalized["maturity_date"], errors="coerce")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        normalized = normalized.loc[maturity_ts > window_start].copy()
        issue_ts = issue_ts.loc[normalized.index]
        maturity_ts = maturity_ts.loc[normalized.index]
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[issue_ts < window_end].copy()
    return (
        normalized.drop(columns=["floating_rate"])
        .sort_values(["issue_date", "auction_date", "lot_id"], kind="stable")
        .reset_index(drop=True)
    )


def load_frn_daily_index_validation(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load observed Treasury FRN daily index rows for reconstruction validation."""

    frame = _read_csv(path)
    normalized = pd.DataFrame(
        {
            "record_date": _normalize_date_column(
                _require_alias(frame, "frn_daily_index_validation", "record_date", "record_date"),
                dataset_name="frn_daily_index_validation",
                column_name="record_date",
            ),
            "cusip": _require_alias(frame, "frn_daily_index_validation", "cusip", "cusip").astype(str).str.strip().str.upper(),
            "start_of_accrual_period": _normalize_date_column(
                _require_alias(
                    frame,
                    "frn_daily_index_validation",
                    "start_of_accrual_period",
                    "start_of_accrual_period",
                ),
                dataset_name="frn_daily_index_validation",
                column_name="start_of_accrual_period",
            ),
            "end_of_accrual_period": _normalize_date_column(
                _require_alias(
                    frame,
                    "frn_daily_index_validation",
                    "end_of_accrual_period",
                    "end_of_accrual_period",
                ),
                dataset_name="frn_daily_index_validation",
                column_name="end_of_accrual_period",
            ),
            "daily_index": _require_alias(frame, "frn_daily_index_validation", "daily_index", "daily_index"),
            "spread": _require_alias(frame, "frn_daily_index_validation", "spread", "spread"),
            "daily_int_accrual_rate": _require_alias(
                frame,
                "frn_daily_index_validation",
                "daily_int_accrual_rate",
                "daily_int_accrual_rate",
            ),
            "daily_accrued_int_per100": _require_alias(
                frame,
                "frn_daily_index_validation",
                "daily_accrued_int_per100",
                "daily_accrued_int_per100",
            ),
        }
    )
    normalized = normalized.loc[_valid_cusip_mask(normalized["cusip"])].copy()
    normalized = validate_numeric_columns(
        normalized,
        ["daily_index", "spread", "daily_int_accrual_rate", "daily_accrued_int_per100"],
        dataset_name="frn_daily_index_validation",
    )
    start_ts = pd.to_datetime(normalized["start_of_accrual_period"], errors="coerce")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        normalized = normalized.loc[start_ts >= window_start].copy()
        start_ts = start_ts.loc[normalized.index]
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[start_ts < window_end].copy()
    normalized["quarter"] = pd.to_datetime(normalized["start_of_accrual_period"], errors="coerce").dt.to_period("Q").astype(str)
    return normalized.sort_values(["cusip", "start_of_accrual_period"], kind="stable").reset_index(drop=True)


def load_frn_mspd_principal_controls(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load monthly MSPD FRN principal controls for source-lot reconciliation."""

    frame = _read_csv(path)
    frame = _drop_mspd_summary_rows(frame)
    output_columns = [
        "record_date",
        "cusip",
        "control_outstanding_mil",
        "mspd_row_count",
        "quarter",
    ]
    if _select_column(frame, "record_date") is None:
        return pd.DataFrame(columns=output_columns)
    normalized = pd.DataFrame(
        {
            "record_date": _normalize_date_column(
                _require_alias(frame, "frn_mspd_principal_controls", "record_date", "record_date"),
                dataset_name="frn_mspd_principal_controls",
                column_name="record_date",
            ),
            "cusip": _require_alias(
                frame,
                "frn_mspd_principal_controls",
                "cusip",
                "security_class2_desc",
                "cusip",
            ).astype(str).str.strip().str.upper(),
            "security_class": _require_alias(
                frame,
                "frn_mspd_principal_controls",
                "security_class",
                "security_class1_desc",
                "security_type",
            ).astype(str).str.strip(),
            "issue_date": _normalize_optional_date_column(
                _select_column(frame, "issue_date")
                if _select_column(frame, "issue_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="frn_mspd_principal_controls",
                column_name="issue_date",
            ),
            "maturity_date": _normalize_optional_date_column(
                _select_column(frame, "maturity_date", "mat_date")
                if _select_column(frame, "maturity_date", "mat_date") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index),
                dataset_name="frn_mspd_principal_controls",
                column_name="maturity_date",
            ),
            "issued_amt_mil": _select_column(frame, "issued_amt"),
            "redeemed_amt_mil": _select_column(frame, "redeemed_amt"),
            "outstanding_mil": _select_column(frame, "outstanding_amt", "outstanding"),
        }
    )
    normalized = normalized.loc[
        normalized["security_class"].eq("Floating Rate Notes")
        & _valid_cusip_mask(normalized["cusip"])
    ].copy()
    if normalized.empty:
        return normalized
    normalized = validate_numeric_columns(
        normalized,
        ["issued_amt_mil", "redeemed_amt_mil", "outstanding_mil"],
        dataset_name="frn_mspd_principal_controls",
    )
    # Reopening rows often carry blank outstanding values. For each CUSIP-month,
    # prefer the observed outstanding row when present; only fall back to
    # issued/redeemed components when no outstanding control is reported.
    component = (
        pd.to_numeric(normalized["issued_amt_mil"], errors="coerce").fillna(0.0)
        + pd.to_numeric(normalized["redeemed_amt_mil"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    normalized["outstanding_mil"] = pd.to_numeric(normalized["outstanding_mil"], errors="coerce")
    normalized["component_control_mil"] = component
    normalized["control_outstanding_mil"] = pd.to_numeric(
        normalized["outstanding_mil"],
        errors="coerce",
    ).where(
        pd.to_numeric(normalized["outstanding_mil"], errors="coerce").notna(),
        component,
    )
    normalized["has_observed_outstanding"] = normalized["outstanding_mil"].notna()
    normalized = (
        normalized.groupby(["record_date", "cusip"], sort=False, dropna=False)
        .agg(
            observed_outstanding_mil=("outstanding_mil", "sum"),
            component_control_mil=("component_control_mil", "sum"),
            has_observed_outstanding=("has_observed_outstanding", "max"),
            mspd_row_count=("control_outstanding_mil", "size"),
        )
        .reset_index()
    )
    normalized["control_outstanding_mil"] = normalized["observed_outstanding_mil"].where(
        normalized["has_observed_outstanding"],
        normalized["component_control_mil"],
    )
    normalized = normalized.drop(
        columns=["observed_outstanding_mil", "component_control_mil", "has_observed_outstanding"],
    )
    record_ts = pd.to_datetime(normalized["record_date"], errors="coerce")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        normalized = normalized.loc[record_ts >= window_start].copy()
        record_ts = record_ts.loc[normalized.index]
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[record_ts < window_end].copy()
    normalized["quarter"] = pd.to_datetime(normalized["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    return normalized.loc[:, output_columns].sort_values(["record_date", "cusip"], kind="stable").reset_index(drop=True)


def load_tips_inflation_adjustment_stocks(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load monthly MSPD TIPS inflation-adjustment stocks by CUSIP lot."""

    frame = _read_csv(path)
    frame = _drop_mspd_summary_rows(frame)
    output_columns = [
        "record_date",
        "quarter",
        "cohort_id",
        "cusip",
        "issue_date",
        "maturity_date",
        "issued_amt_mil",
        "inflation_adjustment_mil",
        "redeemed_amt_mil",
        "outstanding_mil",
        "source_status",
        "evidence_label",
    ]
    if _select_column(frame, "record_date") is None:
        return pd.DataFrame(columns=output_columns)
    record_date = _normalize_date_column(
        _require_alias(frame, "tips_inflation_adjustment_stocks", "record_date", "record_date"),
        dataset_name="tips_inflation_adjustment_stocks",
        column_name="record_date",
    )
    issue_date = _normalize_date_column(
        _require_alias(frame, "tips_inflation_adjustment_stocks", "issue_date", "issue_date"),
        dataset_name="tips_inflation_adjustment_stocks",
        column_name="issue_date",
    )
    maturity_date = _normalize_date_column(
        _require_alias(frame, "tips_inflation_adjustment_stocks", "maturity_date", "maturity_date", "mat_date"),
        dataset_name="tips_inflation_adjustment_stocks",
        column_name="maturity_date",
    )
    cusip = _require_alias(
        frame,
        "tips_inflation_adjustment_stocks",
        "cusip",
        "security_class2_desc",
        "cusip",
    ).astype(str).str.strip().str.upper()
    security_class = _require_alias(
        frame,
        "tips_inflation_adjustment_stocks",
        "security_class",
        "security_class1_desc",
        "security_type",
    ).astype(str).str.strip()
    normalized = pd.DataFrame(
        {
            "record_date": record_date,
            "quarter": pd.to_datetime(record_date, errors="coerce").dt.to_period("Q").astype(str),
            "cohort_id": cusip + "|" + issue_date.astype(str) + "|" + maturity_date.astype(str),
            "cusip": cusip,
            "security_class": security_class,
            "issue_date": issue_date,
            "maturity_date": maturity_date,
            "issued_amt_mil": _select_column(frame, "issued_amt"),
            "inflation_adjustment_mil": _select_column(frame, "inflation_adj_amt"),
            "redeemed_amt_mil": _select_column(frame, "redeemed_amt"),
            "outstanding_mil": _select_column(frame, "outstanding_amt", "outstanding"),
        }
    )
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        pd.Series(["mspd_monthly_tips_inflation_adjustment_row"] * len(normalized.index), index=normalized.index),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(normalized, "inflation_adjustment_mil"),
    )
    normalized = normalized.loc[
        normalized["security_class"].astype(str).str.lower().str.contains("inflation", na=False)
        & _valid_cusip_mask(normalized["cusip"])
        & normalized["issue_date"].notna()
        & normalized["maturity_date"].notna()
    ].copy()
    if normalized.empty:
        return pd.DataFrame(columns=output_columns)
    normalized = validate_numeric_columns(
        normalized,
        ["issued_amt_mil", "inflation_adjustment_mil", "redeemed_amt_mil", "outstanding_mil"],
        dataset_name="tips_inflation_adjustment_stocks",
    )
    normalized = _aggregate_duplicate_rows(
        normalized.drop(columns=["security_class"]),
        ["record_date", "cohort_id"],
        numeric_columns=["issued_amt_mil", "inflation_adjustment_mil", "redeemed_amt_mil", "outstanding_mil"],
    )
    record_ts = pd.to_datetime(normalized["record_date"], errors="coerce")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        prior_month_end = window_start - pd.offsets.MonthEnd(1)
        normalized = normalized.loc[record_ts >= prior_month_end].copy()
        record_ts = record_ts.loc[normalized.index]
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[record_ts < window_end].copy()
    normalized["quarter"] = pd.to_datetime(normalized["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    validate_source_status_columns(normalized, ["source_status"], dataset_name="tips_inflation_adjustment_stocks")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="tips_inflation_adjustment_stocks")
    validate_duplicate_keys(normalized, ["record_date", "cohort_id"], dataset_name="tips_inflation_adjustment_stocks")
    return (
        normalized.loc[:, output_columns]
        .sort_values(["record_date", "cohort_id"], kind="stable")
        .reset_index(drop=True)
    )


def load_tips_auction_events(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load TIPS auction issue/reopening events for aggregate inflation and coupon flows."""

    frame = _read_csv(path)
    output_columns = [
        "event_id",
        "cusip",
        "auction_date",
        "issue_date",
        "maturity_date",
        "total_accepted_mil",
        "coupon_rate_decimal",
        "index_ratio_on_issue_date",
        "ref_cpi_on_dated_date",
        "ref_cpi_on_issue_date",
        "adj_accrued_int_per1000",
        "source_status",
        "evidence_label",
    ]
    auction_date = _require_alias(frame, "tips_auction_events", "auction_date", "auction_date", "record_date")
    issue_date = _require_alias(frame, "tips_auction_events", "issue_date", "issue_date")
    maturity_date = _require_alias(frame, "tips_auction_events", "maturity_date", "maturity_date", "mat_date")
    cusip = _require_alias(frame, "tips_auction_events", "cusip", "cusip", "announcemtd_cusip").astype(str).str.strip().str.upper()
    security_type = _require_alias(
        frame,
        "tips_auction_events",
        "security_type",
        "security_type",
        "security_type_desc",
    ).map(_canonical_security_type)
    inflation_index = _coalesce_text_columns(_select_column(frame, "inflation_index_security"))
    if inflation_index is None:
        inflation_index = pd.Series([""] * len(frame.index), index=frame.index)
    normalized_auction_date = _normalize_date_column(
        auction_date,
        dataset_name="tips_auction_events",
        column_name="auction_date",
    )
    normalized_issue_date = _normalize_date_column(
        issue_date,
        dataset_name="tips_auction_events",
        column_name="issue_date",
    )
    normalized = pd.DataFrame(
        {
            "event_id": cusip + "|" + normalized_auction_date.astype(str) + "|" + normalized_issue_date.astype(str),
            "cusip": cusip,
            "auction_date": normalized_auction_date,
            "issue_date": normalized_issue_date,
            "maturity_date": _normalize_date_column(
                maturity_date,
                dataset_name="tips_auction_events",
                column_name="maturity_date",
            ),
            "security_type": security_type,
            "inflation_index_security": inflation_index.astype(str).str.strip(),
            "total_accepted_mil": pd.to_numeric(
                _require_alias(frame, "tips_auction_events", "total_accepted", "total_accepted"),
                errors="coerce",
            )
            / 1_000_000.0,
            "coupon_rate_decimal": _percent_to_decimal(
                _select_column(frame, "int_rate", "interest_rate_pct")
                if _select_column(frame, "int_rate", "interest_rate_pct") is not None
                else pd.Series([pd.NA] * len(frame.index), index=frame.index)
            ),
            "index_ratio_on_issue_date": _select_column(frame, "index_ratio_on_issue_date"),
            "ref_cpi_on_dated_date": _select_column(frame, "ref_cpi_on_dated_date"),
            "ref_cpi_on_issue_date": _select_column(frame, "ref_cpi_on_issue_date"),
            "adj_accrued_int_per1000": _select_column(frame, "adj_accrued_int_per1000"),
        }
    )
    normalized["source_status"] = _coalesce_text_columns(
        _select_column(frame, "source_status"),
        pd.Series(["fiscaldata_tips_auction_row"] * len(normalized.index), index=normalized.index),
    )
    normalized["evidence_label"] = _coalesce_text_columns(
        _select_column(frame, "evidence_label"),
        _default_evidence_label(
            normalized,
            "total_accepted_mil",
            "ref_cpi_on_dated_date",
        ),
    )
    normalized = normalized.loc[
        (
            normalized["security_type"].eq("tips")
            | normalized["inflation_index_security"].str.lower().eq("yes")
        )
        & _valid_cusip_mask(normalized["cusip"])
    ].copy()
    if normalized.empty:
        return pd.DataFrame(columns=output_columns)
    normalized = validate_numeric_columns(
        normalized,
        [
            "total_accepted_mil",
            "coupon_rate_decimal",
            "index_ratio_on_issue_date",
            "ref_cpi_on_dated_date",
            "ref_cpi_on_issue_date",
            "adj_accrued_int_per1000",
        ],
        dataset_name="tips_auction_events",
    )
    validate_duplicate_keys(normalized, ["event_id"], dataset_name="tips_auction_events")
    validate_source_status_columns(normalized, ["source_status"], dataset_name="tips_auction_events")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="tips_auction_events")
    issue_ts = pd.to_datetime(normalized["issue_date"], errors="coerce")
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[issue_ts < window_end].copy()
    return (
        normalized.drop(columns=["security_type", "inflation_index_security"])
        .loc[:, output_columns]
        .sort_values(["issue_date", "auction_date", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def load_tips_cpi_reference_path(path, start_quarter=None, end_quarter=None) -> pd.DataFrame:
    """Load the universal daily TIPS Reference CPI path from FiscalData detail rows."""

    frame = _read_csv(path)
    output_columns = ["index_date", "ref_cpi", "source_status", "evidence_label"]
    if not {"index_date", "ref_cpi"}.issubset(frame.columns):
        return pd.DataFrame(columns=output_columns)
    normalized = pd.DataFrame(
        {
            "index_date": _normalize_date_column(
                frame["index_date"],
                dataset_name="tips_cpi_reference_path",
                column_name="index_date",
            ),
            "ref_cpi": frame["ref_cpi"],
        }
    )
    normalized["source_status"] = "fiscaldata_tips_cpi_detail_ref_cpi"
    normalized["evidence_label"] = _default_evidence_label(normalized, "ref_cpi")
    normalized = validate_numeric_columns(normalized, ["ref_cpi"], dataset_name="tips_cpi_reference_path")
    grouped = normalized.groupby("index_date", sort=False, dropna=False)["ref_cpi"]
    spread = grouped.transform("max") - grouped.transform("min")
    inconsistent = spread.abs() > 1.0e-9
    if inconsistent.any():
        sample = normalized.loc[inconsistent, ["index_date", "ref_cpi"]].head(5).to_dict("records")
        raise ValueError(f"tips_cpi_reference_path has inconsistent ref_cpi by date: {sample}")
    normalized = (
        normalized.groupby("index_date", sort=False, dropna=False)
        .agg(
            ref_cpi=("ref_cpi", _first_non_null),
            source_status=("source_status", _first_non_null),
            evidence_label=("evidence_label", _first_non_null),
        )
        .reset_index()
    )
    index_ts = pd.to_datetime(normalized["index_date"], errors="coerce")
    window_start = _quarter_start_timestamp(start_quarter)
    if window_start is not None:
        normalized = normalized.loc[index_ts >= window_start - pd.Timedelta(days=370)].copy()
        index_ts = index_ts.loc[normalized.index]
    window_end = _quarter_end_exclusive_timestamp(end_quarter)
    if window_end is not None:
        normalized = normalized.loc[index_ts < window_end + pd.Timedelta(days=370)].copy()
    validate_source_status_columns(normalized, ["source_status"], dataset_name="tips_cpi_reference_path")
    validate_evidence_labels(normalized, ["evidence_label"], dataset_name="tips_cpi_reference_path")
    validate_duplicate_keys(normalized, ["index_date"], dataset_name="tips_cpi_reference_path")
    return normalized.loc[:, output_columns].sort_values("index_date", kind="stable").reset_index(drop=True)


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
    parsed_record = pd.to_datetime(frame["record_date"], errors="coerce")
    if "start_of_accrual_period" in frame.columns:
        parsed_period = pd.to_datetime(frame["start_of_accrual_period"], errors="coerce")
        parsed_period = parsed_period.where(parsed_period.notna(), parsed_record)
    else:
        parsed_period = parsed_record
    normalized = pd.DataFrame(
        {
            "quarter": parsed_period.dt.to_period("Q").astype(str),
            "period_date": parsed_period,
            "record_date": parsed_record,
            "cusip": frame["cusip"].astype(str).str.strip(),
        }
    )
    if "accr_int_per100_pmt_period" in frame.columns:
        normalized["frn_accrued_interest_pmt_period_per100"] = pd.to_numeric(
            frame["accr_int_per100_pmt_period"],
            errors="coerce",
        )
    if "daily_accrued_int_per100" in frame.columns:
        normalized["frn_daily_accrued_interest_per100"] = pd.to_numeric(
            frame["daily_accrued_int_per100"],
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
    sort_columns = ["quarter", "cusip", "period_date", "record_date"]
    latest = (
        normalized.sort_values(sort_columns)
        .groupby(["quarter", "cusip"], sort=False, dropna=False)
        .tail(1)
        .copy()
    )
    latest = latest.drop(columns=["period_date", "record_date"])
    if "frn_accrued_interest_pmt_period_per100" in latest.columns:
        latest = latest.rename(
            columns={
                "frn_accrued_interest_pmt_period_per100": "frn_accrued_interest_per100"
            }
        )

    if "frn_daily_accrued_interest_per100" not in normalized.columns:
        return _aggregate_enrichment(latest, ["quarter", "cusip"])

    daily_flow = (
        normalized.groupby(["quarter", "cusip"], sort=False, dropna=False)[
            "frn_daily_accrued_interest_per100"
        ]
        .sum(min_count=1)
        .reset_index()
        .rename(
            columns={"frn_daily_accrued_interest_per100": "frn_accrued_interest_flow_per100"}
        )
    )
    latest = latest.drop(columns=["frn_daily_accrued_interest_per100"], errors="ignore")
    latest = latest.merge(daily_flow, on=["quarter", "cusip"], how="left", validate="one_to_one")
    if "frn_accrued_interest_per100" in latest.columns:
        latest["frn_accrued_interest_per100"] = latest[
            "frn_accrued_interest_flow_per100"
        ].combine_first(latest["frn_accrued_interest_per100"])
    else:
        latest["frn_accrued_interest_per100"] = latest["frn_accrued_interest_flow_per100"]
    latest = latest.drop(columns=["frn_accrued_interest_flow_per100"])
    return _aggregate_enrichment(latest, ["quarter", "cusip"])


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
