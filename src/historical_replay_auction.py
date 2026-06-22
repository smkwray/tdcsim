"""Auction allotment proxy helpers for historical replay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from historical_replay_contract import (
    filter_quarter_range,
    normalize_quarter_series,
    require_columns,
    validate_numeric_columns,
)


DEFAULT_ALLOTMENT_PANEL_PATH = Path(
    "data/historical_replay/imported/buycurve/auction_allotment_panel_base_slim.csv"
)
DEFAULT_AUCTION_TERMS_PATH = Path("data/historical_replay/raw/fiscaldata/auctions_query.csv")
DEFAULT_OUTPUT_PATH = Path("data/historical_replay/validation/auction_allotment_proxy.csv")
DEFAULT_ABSORPTION_OUTPUT_PATH = Path(
    "data/historical_replay/validation/auction_absorption_quarter_holder_reconciliation.csv"
)
AMOUNT_TO_MILLIONS = 1_000_000.0

AUCTION_ALLOTMENT_PROXY_COLUMNS = (
    "quarter",
    "cusip",
    "auction_date",
    "issue_date",
    "maturity_date",
    "security_type",
    "security_term",
    "raw_investor_class",
    "narrow_investor_class",
    "broad_investor_class",
    "is_bridge_class",
    "allotment_amount",
    "allotment_total_clean",
    "allotment_share_clean",
    "accepted_amount",
    "offering_amount",
    "reconciliation_gap",
    "source_status",
    "evidence_label",
)
AUCTION_ABSORPTION_RECONCILIATION_COLUMNS = (
    "quarter",
    "broad_investor_class",
    "tdcsim_holder",
    "tdc_absorption_role",
    "is_bridge_class",
    "included_in_identified_primary_allotment",
    "included_in_tdc_auction_absorption",
    "auction_count",
    "allotment_amount",
    "allotment_amount_mil",
    "signed_tdc_auction_absorption_mil",
    "unique_auction_accepted_amount_mil",
    "unique_auction_offering_amount_mil",
    "unique_auction_allotment_total_clean_mil",
    "quarter_allotment_reconciliation_gap_mil",
    "source_status",
    "evidence_label",
)
AUCTION_HOLDER_PRIOR_COLUMNS = (
    "cusip",
    "issue_date",
    "maturity_date",
    "prior_holder_Banks",
    "prior_holder_Private",
    "prior_holder_CB",
    "prior_holder_Foreign",
    "auction_prior_total_amount",
    "auction_prior_holder_count",
    "auction_prior_status",
)

_ALLOTMENT_REQUIRED_COLUMNS = (
    "cusip",
    "auction_date",
    "issue_date",
    "maturity_date",
    "security_type",
    "security_term",
    "raw_investor_class",
    "narrow_investor_class",
    "broad_investor_class",
    "is_bridge_class",
    "allotment_amount",
    "allotment_total_clean",
    "allotment_share_clean",
    "accepted_amount",
    "offering_amount",
)
_RAW_FALLBACK_NUMERIC_ALIASES = {
    "accepted_amount": ("total_accepted", "accepted_amount"),
    "offering_amount": ("offering_amt", "offering_amount"),
}
_SOURCE_NULL_MARKERS = ("", "null", "none", "nan", "na", "n/a", "*")
_BROAD_INVESTOR_TO_TDCSIM_HOLDER = {
    "banks": "Banks",
    "dealers": "dealer_bridge",
    "federal_reserve": "CB",
    "foreign_international": "Foreign",
    "individuals": "Private",
    "investment_funds": "Private",
    "other": "Private",
    "pensions_insurers": "Private",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
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


def _normalize_date_column(values: pd.Series, *, dataset_name: str, column_name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    invalid = parsed.isna() & values.notna() & values.astype(str).str.strip().ne("")
    if invalid.any():
        bad_values = sorted(values.loc[invalid].astype(str).unique().tolist())[:3]
        raise ValueError(f"{dataset_name} column {column_name} has invalid dates: {bad_values}")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalize_text(values: pd.Series) -> pd.Series:
    return values.where(values.notna(), "").astype(str).str.strip()


def _first_non_blank(left: pd.Series, right: pd.Series) -> pd.Series:
    left_text = _normalize_text(left)
    return left_text.where(left_text.ne(""), right)


def _normalize_boolean(values: pd.Series, *, dataset_name: str, column_name: str) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n"}
    result.loc[text.isin(truthy)] = True
    result.loc[text.isin(falsy)] = False
    invalid = values.notna() & ~text.isin(truthy | falsy)
    if invalid.any():
        bad_values = sorted(values.loc[invalid].astype(str).unique().tolist())[:3]
        raise ValueError(f"{dataset_name} column {column_name} has invalid booleans: {bad_values}")
    return result


def _first_non_null(values: pd.Series):
    non_null = values.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def _prepare_raw_auction_terms(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("cusip", "auction_date", "issue_date", "maturity_date")
    require_columns(frame, required, dataset_name="auction_terms")
    normalized = pd.DataFrame(
        {
            "cusip": _normalize_text(frame["cusip"]),
            "auction_date": _normalize_date_column(
                frame["auction_date"], dataset_name="auction_terms", column_name="auction_date"
            ),
            "issue_date": _normalize_date_column(
                frame["issue_date"], dataset_name="auction_terms", column_name="issue_date"
            ),
            "maturity_date": _normalize_date_column(
                frame["maturity_date"], dataset_name="auction_terms", column_name="maturity_date"
            ),
            "security_type__fallback": _normalize_text(
                _select_column(frame, "security_type") if _select_column(frame, "security_type") is not None else pd.Series("", index=frame.index)
            ),
            "security_term__fallback": _normalize_text(
                _select_column(frame, "security_term") if _select_column(frame, "security_term") is not None else pd.Series("", index=frame.index)
            ),
        }
    )
    for target, aliases in _RAW_FALLBACK_NUMERIC_ALIASES.items():
        column = _select_column(frame, *aliases)
        normalized[f"{target}__fallback"] = column if column is not None else pd.Series(pd.NA, index=frame.index)
    normalized = validate_numeric_columns(
        normalized,
        [f"{target}__fallback" for target in _RAW_FALLBACK_NUMERIC_ALIASES],
        dataset_name="auction_terms",
    )
    if not normalized.duplicated(
        subset=["cusip", "auction_date", "issue_date", "maturity_date"], keep=False
    ).any():
        return normalized
    aggregated = (
        normalized.groupby(
            ["cusip", "auction_date", "issue_date", "maturity_date"],
            sort=False,
            dropna=False,
        )
        .agg({column: _first_non_null for column in normalized.columns if column not in required})
        .reset_index()
    )
    return aggregated


def load_auction_allotment_proxy(
    allotment_panel_path: str | Path = DEFAULT_ALLOTMENT_PANEL_PATH,
    auction_terms_path: str | Path = DEFAULT_AUCTION_TERMS_PATH,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Load historical replay auction allotment proxy rows."""

    frame = _read_csv(allotment_panel_path)
    require_columns(frame, _ALLOTMENT_REQUIRED_COLUMNS, dataset_name="auction_allotment_proxy")
    normalized = pd.DataFrame(
        {
            "quarter": normalize_quarter_series(frame["issue_date"]),
            "cusip": _normalize_text(frame["cusip"]),
            "auction_date": _normalize_date_column(
                frame["auction_date"],
                dataset_name="auction_allotment_proxy",
                column_name="auction_date",
            ),
            "issue_date": _normalize_date_column(
                frame["issue_date"],
                dataset_name="auction_allotment_proxy",
                column_name="issue_date",
            ),
            "maturity_date": _normalize_date_column(
                frame["maturity_date"],
                dataset_name="auction_allotment_proxy",
                column_name="maturity_date",
            ),
            "security_type": _normalize_text(frame["security_type"]),
            "security_term": _normalize_text(frame["security_term"]),
            "raw_investor_class": _normalize_text(frame["raw_investor_class"]),
            "narrow_investor_class": _normalize_text(frame["narrow_investor_class"]),
            "broad_investor_class": _normalize_text(frame["broad_investor_class"]),
            "is_bridge_class": _normalize_boolean(
                frame["is_bridge_class"],
                dataset_name="auction_allotment_proxy",
                column_name="is_bridge_class",
            ),
            "allotment_amount": frame["allotment_amount"],
            "allotment_total_clean": frame["allotment_total_clean"],
            "allotment_share_clean": frame["allotment_share_clean"],
            "accepted_amount": frame["accepted_amount"],
            "offering_amount": frame["offering_amount"],
        }
    )
    normalized = validate_numeric_columns(
        normalized,
        [
            "allotment_amount",
            "allotment_total_clean",
            "allotment_share_clean",
            "accepted_amount",
            "offering_amount",
        ],
        dataset_name="auction_allotment_proxy",
    )

    fallback_used = pd.Series(False, index=normalized.index, dtype=bool)
    if auction_terms_path is not None:
        fallback = _prepare_raw_auction_terms(_read_csv(auction_terms_path))
        normalized = normalized.merge(
            fallback,
            on=["cusip", "auction_date", "issue_date", "maturity_date"],
            how="left",
            validate="many_to_one",
        )
        for column in ("security_type", "security_term"):
            fallback_column = f"{column}__fallback"
            blank = normalized[column].astype(str).str.strip().eq("")
            use_fallback = blank & normalized[fallback_column].astype(str).str.strip().ne("")
            normalized[column] = _first_non_blank(normalized[column], normalized[fallback_column])
            fallback_used |= use_fallback.fillna(False)
            normalized = normalized.drop(columns=fallback_column)
        for column in ("accepted_amount", "offering_amount"):
            fallback_column = f"{column}__fallback"
            use_fallback = normalized[column].isna() & normalized[fallback_column].notna()
            normalized[column] = normalized[column].where(~use_fallback, normalized[fallback_column])
            fallback_used |= use_fallback.fillna(False)
            normalized = normalized.drop(columns=fallback_column)

    gap_mask = normalized["accepted_amount"].notna() & normalized["allotment_total_clean"].notna()
    normalized["reconciliation_gap"] = pd.NA
    normalized.loc[gap_mask, "reconciliation_gap"] = (
        normalized.loc[gap_mask, "accepted_amount"] - normalized.loc[gap_mask, "allotment_total_clean"]
    )
    normalized["source_status"] = fallback_used.map(
        lambda used: "buycurve_allotment_panel_with_fiscaldata_fallback"
        if used
        else "buycurve_allotment_panel_observed"
    )
    normalized["evidence_label"] = "observed"

    normalized = filter_quarter_range(
        normalized,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    normalized = normalized.sort_values(
        ["quarter", "issue_date", "auction_date", "cusip", "raw_investor_class"]
    ).reset_index(drop=True)
    return normalized.loc[:, list(AUCTION_ALLOTMENT_PROXY_COLUMNS)]


def write_auction_allotment_proxy(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    allotment_panel_path: str | Path = DEFAULT_ALLOTMENT_PANEL_PATH,
    auction_terms_path: str | Path = DEFAULT_AUCTION_TERMS_PATH,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Write the auction allotment proxy CSV and return the normalized frame."""

    frame = load_auction_allotment_proxy(
        allotment_panel_path=allotment_panel_path,
        auction_terms_path=auction_terms_path,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return frame


def build_auction_absorption_reconciliation(proxy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate auction allotments into quarter/holder primary-allotment rows.

    Amounts in the source panel are dollars. The signed TDC contribution follows
    the simulator sign convention: DU/private purchases of newly issued debt
    reduce deposits, so the TDC auction-absorption contribution is negative.
    """

    columns = list(AUCTION_ABSORPTION_RECONCILIATION_COLUMNS)
    if proxy.empty:
        return pd.DataFrame(columns=columns)
    require_columns(
        proxy,
        [
            "quarter",
            "cusip",
            "auction_date",
            "issue_date",
            "maturity_date",
            "broad_investor_class",
            "is_bridge_class",
            "allotment_amount",
            "allotment_total_clean",
            "accepted_amount",
            "offering_amount",
            "source_status",
            "evidence_label",
        ],
        dataset_name="auction_allotment_proxy",
    )
    working = proxy.copy()
    working["broad_investor_class"] = _normalize_text(working["broad_investor_class"]).str.lower()
    unknown = sorted(set(working["broad_investor_class"]) - set(_BROAD_INVESTOR_TO_TDCSIM_HOLDER))
    if unknown:
        raise ValueError(f"auction_allotment_proxy has unmapped investor classes: {unknown}")
    working["tdcsim_holder"] = working["broad_investor_class"].map(_BROAD_INVESTOR_TO_TDCSIM_HOLDER)
    working["is_bridge_class"] = working["is_bridge_class"].astype("boolean").fillna(False).astype(bool)
    for column in ("allotment_amount", "allotment_total_clean", "accepted_amount", "offering_amount"):
        working[column] = pd.to_numeric(working[column], errors="coerce")

    auction_keys = ["quarter", "cusip", "auction_date", "issue_date", "maturity_date"]
    auction_totals = (
        working.drop_duplicates(subset=auction_keys, keep="first")
        .groupby("quarter", dropna=False, sort=False)
        .agg(
            auction_count=("cusip", "size"),
            unique_auction_accepted_amount=("accepted_amount", "sum"),
            unique_auction_offering_amount=("offering_amount", "sum"),
            unique_auction_allotment_total_clean=("allotment_total_clean", "sum"),
        )
        .reset_index()
    )
    auction_totals["unique_auction_accepted_amount_mil"] = (
        auction_totals["unique_auction_accepted_amount"] / AMOUNT_TO_MILLIONS
    )
    auction_totals["unique_auction_offering_amount_mil"] = (
        auction_totals["unique_auction_offering_amount"] / AMOUNT_TO_MILLIONS
    )
    auction_totals["unique_auction_allotment_total_clean_mil"] = (
        auction_totals["unique_auction_allotment_total_clean"] / AMOUNT_TO_MILLIONS
    )
    auction_totals["quarter_allotment_reconciliation_gap_mil"] = (
        auction_totals["unique_auction_accepted_amount"]
        - auction_totals["unique_auction_allotment_total_clean"]
    ) / AMOUNT_TO_MILLIONS
    auction_totals = auction_totals[
        [
            "quarter",
            "auction_count",
            "unique_auction_accepted_amount_mil",
            "unique_auction_offering_amount_mil",
            "unique_auction_allotment_total_clean_mil",
            "quarter_allotment_reconciliation_gap_mil",
        ]
    ]

    grouped = (
        working.groupby(
            ["quarter", "broad_investor_class", "tdcsim_holder", "is_bridge_class"],
            dropna=False,
            sort=True,
        )
        .agg(
            allotment_amount=("allotment_amount", "sum"),
            source_status=("source_status", _join_unique),
            evidence_label=("evidence_label", _join_unique),
        )
        .reset_index()
    )
    grouped = grouped.merge(auction_totals, on="quarter", how="left", validate="many_to_one")
    grouped["included_in_identified_primary_allotment"] = ~grouped["is_bridge_class"]
    grouped["included_in_tdc_auction_absorption"] = (
        grouped["included_in_identified_primary_allotment"]
        & grouped["tdcsim_holder"].eq("Private")
    )
    grouped["allotment_amount_mil"] = grouped["allotment_amount"] / AMOUNT_TO_MILLIONS
    grouped["signed_tdc_auction_absorption_mil"] = grouped["allotment_amount_mil"].where(
        grouped["included_in_tdc_auction_absorption"],
        0.0,
    ) * -1.0
    grouped["tdc_absorption_role"] = grouped.apply(_auction_absorption_role, axis=1)
    return grouped.loc[:, columns].sort_values(
        ["quarter", "is_bridge_class", "tdcsim_holder", "broad_investor_class"],
        kind="stable",
    ).reset_index(drop=True)


def build_auction_holder_prior_panel(proxy: pd.DataFrame) -> pd.DataFrame:
    """Build source-backed holder prior weights by security key."""

    columns = list(AUCTION_HOLDER_PRIOR_COLUMNS)
    if proxy.empty:
        return pd.DataFrame(columns=columns)
    require_columns(
        proxy,
        [
            "cusip",
            "issue_date",
            "maturity_date",
            "broad_investor_class",
            "is_bridge_class",
            "allotment_amount",
        ],
        dataset_name="auction_allotment_proxy",
    )
    working = proxy.copy()
    working["cusip"] = _normalize_text(working["cusip"]).str.upper()
    working["issue_date"] = _normalize_text(working["issue_date"])
    working["maturity_date"] = _normalize_text(working["maturity_date"])
    working["broad_investor_class"] = _normalize_text(working["broad_investor_class"]).str.lower()
    unknown = sorted(set(working["broad_investor_class"]) - set(_BROAD_INVESTOR_TO_TDCSIM_HOLDER))
    if unknown:
        raise ValueError(f"auction_allotment_proxy has unmapped investor classes: {unknown}")
    working["tdcsim_holder"] = working["broad_investor_class"].map(_BROAD_INVESTOR_TO_TDCSIM_HOLDER)
    working["is_bridge_class"] = working["is_bridge_class"].astype("boolean").fillna(False).astype(bool)
    working["allotment_amount"] = pd.to_numeric(working["allotment_amount"], errors="coerce").fillna(0.0)
    working = working.loc[
        ~working["is_bridge_class"]
        & working["tdcsim_holder"].isin(["Banks", "Private", "CB", "Foreign"])
        & working["cusip"].ne("")
        & working["issue_date"].ne("")
        & working["maturity_date"].ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        working.groupby(
            ["cusip", "issue_date", "maturity_date", "tdcsim_holder"],
            dropna=False,
            sort=False,
        )["allotment_amount"]
        .sum()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["cusip", "issue_date", "maturity_date"],
        columns="tdcsim_holder",
        values="allotment_amount",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    pivot.columns.name = None
    for holder in ["Banks", "Private", "CB", "Foreign"]:
        if holder not in pivot.columns:
            pivot[holder] = 0.0
        pivot[f"prior_holder_{holder}"] = pd.to_numeric(pivot[holder], errors="coerce").fillna(0.0)
    prior_cols = [f"prior_holder_{holder}" for holder in ["Banks", "Private", "CB", "Foreign"]]
    pivot["auction_prior_total_amount"] = pivot[prior_cols].sum(axis=1)
    pivot["auction_prior_holder_count"] = (pivot[prior_cols] > 0.0).sum(axis=1)
    pivot["auction_prior_status"] = "source_backed_nonbridge_auction_holder_prior"
    return pivot.loc[:, columns].sort_values(["issue_date", "cusip"], kind="stable").reset_index(drop=True)


def write_auction_absorption_reconciliation(
    output_path: str | Path = DEFAULT_ABSORPTION_OUTPUT_PATH,
    *,
    proxy: pd.DataFrame | None = None,
    allotment_panel_path: str | Path = DEFAULT_ALLOTMENT_PANEL_PATH,
    auction_terms_path: str | Path = DEFAULT_AUCTION_TERMS_PATH,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Write quarter/holder auction absorption reconciliation rows."""

    source = (
        proxy
        if proxy is not None
        else load_auction_allotment_proxy(
            allotment_panel_path=allotment_panel_path,
            auction_terms_path=auction_terms_path,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        )
    )
    frame = build_auction_absorption_reconciliation(source)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return frame


def _auction_absorption_role(row: pd.Series) -> str:
    investor_class = str(row.get("broad_investor_class", "")).strip().lower()
    holder = str(row.get("tdcsim_holder", "")).strip()
    if bool(row.get("is_bridge_class", False)):
        if investor_class == "dealers":
            return "dealer_bridge_not_final_holder"
        return "bridge_or_intermediary_not_tdc_deposit_absorption"
    if holder == "Private":
        return "identified_du_primary_allotment_gross_signed_negative"
    if holder in {"Banks", "CB", "Foreign"}:
        return "identified_ru_primary_allotment_not_tdc_deposit_absorption"
    return "unresolved_primary_class_not_allocated"


def _join_unique(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values.dropna().tolist() if str(value).strip()})
    return ";".join(unique)
