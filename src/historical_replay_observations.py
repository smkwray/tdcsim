"""Source-observation registry for quarterly historical replay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from historical_replay_contract import filter_quarter_range, normalize_quarter_value


OBSERVATION_REGISTRY_COLUMNS = [
    "observation_id",
    "quarter",
    "observation_date",
    "source_file",
    "source_row_key",
    "scope",
    "holder",
    "broad_holder",
    "auction_class",
    "security_id",
    "security_type",
    "maturity_bucket",
    "measure",
    "valuation_basis",
    "reported_value_mil",
    "lower_bound_mil",
    "upper_bound_mil",
    "absolute_tolerance_mil",
    "enforcement",
    "priority",
    "coverage_status",
    "transformation",
]

COVERAGE_COLUMNS = [
    "source",
    "scope",
    "quarter_start",
    "quarter_end",
    "observation_count",
    "reported_value_mil",
    "security_level_rows",
    "holder_security_rows",
    "solver_treatment",
    "coverage_status",
]

BASIS_BRIDGE_COLUMNS = [
    "quarter",
    "scope",
    "valuation_basis",
    "enforcement",
    "observation_count",
    "reported_value_mil",
    "solver_treatment",
    "basis_note",
]


_DEFAULT_FFIEC_PATH = Path(
    "data/historical_replay/imported/tdcest/ffiec_interest_constraints_normalized.csv"
)
_DEFAULT_NCUA_PATH = Path(
    "data/historical_replay/imported/tdcest/ncua_interest_constraints_normalized.csv"
)
_DEFAULT_TIER2_CONSTRAINTS_PATH = Path(
    "data/historical_replay/imported/tdcest/tier2_interest_source_constraints.csv"
)
_DEFAULT_QUARTERLY_INPUTS_PATH = Path(
    "data/historical_replay/imported/tdcest/quarterly_inputs.csv"
)
_AMOUNT_TO_MILLIONS = 1_000_000.0


def build_historical_replay_observations(
    sector_levels: pd.DataFrame,
    auction_allotment_proxy: pd.DataFrame,
    *,
    ffiec_path: str | Path | None = _DEFAULT_FFIEC_PATH,
    ncua_path: str | Path | None = _DEFAULT_NCUA_PATH,
    tier2_constraints_path: str | Path | None = _DEFAULT_TIER2_CONSTRAINTS_PATH,
    quarterly_inputs_path: str | Path | None = _DEFAULT_QUARTERLY_INPUTS_PATH,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Build a long-form registry of source observations entering replay.

    The registry separates source facts from solver outputs. Z.1 holder levels
    and FFIEC rows are aggregate observations; auction allotments are observed
    event facts. Synthetic sector-by-CUSIP allocations are intentionally not
    represented here as observed facts.
    """

    frames = [
        _sector_level_observations(
            sector_levels,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        ),
        _auction_allotment_observations(
            auction_allotment_proxy,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        ),
    ]
    ffiec = _ffiec_observations(
        ffiec_path,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    if not ffiec.empty:
        frames.append(ffiec)
    ncua = _ncua_observations(
        ncua_path,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    if not ncua.empty:
        frames.append(ncua)
    tier2_constraints = _tier2_interest_constraint_observations(
        tier2_constraints_path,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    if not tier2_constraints.empty:
        frames.append(tier2_constraints)
    mmf_components = _mmf_component_observations(
        quarterly_inputs_path,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    if not mmf_components.empty:
        frames.append(mmf_components)
    observations = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if observations.empty:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    observations = observations.loc[:, OBSERVATION_REGISTRY_COLUMNS].copy()
    observations["observation_id"] = [
        f"obs_{idx:09d}"
        for idx in range(1, len(observations.index) + 1)
    ]
    return observations.loc[:, OBSERVATION_REGISTRY_COLUMNS]


def build_exact_observation_coverage(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize which exact source observations are available."""

    if observations.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    working = observations.copy()
    working["reported_value_mil"] = pd.to_numeric(working["reported_value_mil"], errors="coerce").fillna(0.0)
    rows = []
    group_cols = ["source_file", "scope"]
    for (source_file, scope), group in working.groupby(group_cols, dropna=False, sort=False):
        quarter_values = group["quarter"].dropna().astype(str).tolist()
        security_rows = int(group["security_id"].notna().sum()) if "security_id" in group.columns else 0
        holder_security_rows = int(
            (
                group.get("holder", pd.Series(pd.NA, index=group.index)).notna()
                & group.get("security_id", pd.Series(pd.NA, index=group.index)).notna()
            ).sum()
        )
        rows.append(
            {
                "source": source_file,
                "scope": scope,
                "quarter_start": min(quarter_values) if quarter_values else pd.NA,
                "quarter_end": max(quarter_values) if quarter_values else pd.NA,
                "observation_count": int(len(group.index)),
                "reported_value_mil": float(group["reported_value_mil"].sum()),
                "security_level_rows": security_rows,
                "holder_security_rows": holder_security_rows,
                "solver_treatment": _solver_treatment_for_scope(str(scope)),
                "coverage_status": _coverage_status_for_scope(str(scope)),
            }
        )
    rows.append(
        {
            "source": "all_current_inputs",
            "scope": "observed_holder_by_security_positions",
            "quarter_start": working["quarter"].min(),
            "quarter_end": working["quarter"].max(),
            "observation_count": 0,
            "reported_value_mil": 0.0,
            "security_level_rows": 0,
            "holder_security_rows": 0,
            "solver_treatment": "not_observed_solved_endogenously",
            "coverage_status": "not_available_from_current_public_inputs",
        }
    )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_holder_basis_bridge(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize source valuation bases and how the solver should treat them."""

    if observations.empty:
        return pd.DataFrame(columns=BASIS_BRIDGE_COLUMNS)
    working = observations.copy()
    working["reported_value_mil"] = pd.to_numeric(working["reported_value_mil"], errors="coerce").fillna(0.0)
    grouped = (
        working.groupby(["quarter", "scope", "valuation_basis", "enforcement"], dropna=False, sort=False)
        .agg(
            observation_count=("observation_id", "size"),
            reported_value_mil=("reported_value_mil", "sum"),
        )
        .reset_index()
    )
    grouped["solver_treatment"] = grouped["scope"].map(_solver_treatment_for_scope)
    grouped["basis_note"] = grouped["valuation_basis"].map(_basis_note)
    return grouped.loc[:, BASIS_BRIDGE_COLUMNS]


def _sector_level_observations(
    sector_levels: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if sector_levels.empty:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    required = {"quarter", "sector", "measure", "value"}
    if not required.issubset(sector_levels.columns):
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame = sector_levels.copy()
    frame = filter_quarter_range(frame, start_quarter=start_quarter, end_quarter=end_quarter)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    rows = []
    for idx, row in frame.iterrows():
        quarter = str(row.get("quarter"))
        valuation_basis = _sector_valuation_basis(row)
        is_transaction_flow = valuation_basis == "z1_transaction_flow"
        rows.append(
            {
                "observation_id": pd.NA,
                "quarter": quarter,
                "observation_date": _quarter_end_date(quarter),
                "source_file": row.get("source_file", "z1_treasury_holders"),
                "source_row_key": f"{quarter}|{row.get('sector')}|{row.get('measure')}|{idx}",
                "scope": "z1_transaction_flow" if is_transaction_flow else "z1_holder_level",
                "holder": row.get("tdcsim_holder", pd.NA),
                "broad_holder": row.get("broad_holder_class", row.get("sector", pd.NA)),
                "auction_class": pd.NA,
                "security_id": pd.NA,
                "security_type": pd.NA,
                "maturity_bucket": pd.NA,
                "measure": row.get("measure"),
                "valuation_basis": valuation_basis,
                "reported_value_mil": row.get("value"),
                "lower_bound_mil": pd.NA if is_transaction_flow else row.get("value"),
                "upper_bound_mil": pd.NA if is_transaction_flow else row.get("value"),
                "absolute_tolerance_mil": pd.NA if is_transaction_flow else 0.0,
                "enforcement": "diagnostic_transition_reference" if is_transaction_flow else "penalized_aggregate_observation",
                "priority": 20,
                "coverage_status": (
                    "observed_aggregate_flow_not_transfer"
                    if is_transaction_flow
                    else "observed_aggregate_not_security_level"
                ),
                "transformation": row.get("source_status", "loaded_sector_level_value"),
            }
        )
    return pd.DataFrame(rows, columns=OBSERVATION_REGISTRY_COLUMNS)


def _auction_allotment_observations(
    auction_allotment_proxy: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if auction_allotment_proxy.empty:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    required = {"quarter", "cusip", "allotment_amount"}
    if not required.issubset(auction_allotment_proxy.columns):
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame = auction_allotment_proxy.copy()
    frame = filter_quarter_range(frame, start_quarter=start_quarter, end_quarter=end_quarter)
    frame["allotment_amount"] = pd.to_numeric(frame["allotment_amount"], errors="coerce")
    rows = []
    for idx, row in frame.iterrows():
        quarter = str(row.get("quarter"))
        value_mil = row.get("allotment_amount")
        if pd.notna(value_mil):
            value_mil = float(value_mil) / _AMOUNT_TO_MILLIONS
        rows.append(
            {
                "observation_id": pd.NA,
                "quarter": quarter,
                "observation_date": row.get("auction_date", row.get("issue_date", _quarter_end_date(quarter))),
                "source_file": "buycurve_auction_allotment_panel",
                "source_row_key": "|".join(
                    str(row.get(col, ""))
                    for col in ["cusip", "auction_date", "issue_date", "raw_investor_class"]
                ),
                "scope": "auction_allotment_event",
                "holder": _auction_holder(row.get("broad_investor_class")),
                "broad_holder": row.get("broad_investor_class", pd.NA),
                "auction_class": row.get("raw_investor_class", row.get("narrow_investor_class", pd.NA)),
                "security_id": row.get("cusip", pd.NA),
                "security_type": row.get("security_type", pd.NA),
                "maturity_bucket": row.get("security_term", pd.NA),
                "measure": "primary_issue_allotment",
                "valuation_basis": "par_at_issuance",
                "reported_value_mil": value_mil,
                "lower_bound_mil": value_mil,
                "upper_bound_mil": value_mil,
                "absolute_tolerance_mil": 0.0,
                "enforcement": "source_event_fact",
                "priority": 5,
                "coverage_status": "observed_security_event_not_position",
                "transformation": row.get("source_status", "buycurve_amount_dollars_to_millions"),
            }
        )
    return pd.DataFrame(rows, columns=OBSERVATION_REGISTRY_COLUMNS)


def _mmf_component_observations(
    quarterly_inputs_path: str | Path | None,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if quarterly_inputs_path is None:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    path = Path(quarterly_inputs_path)
    if not path.exists():
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame = pd.read_csv(path, low_memory=False)
    required = {"date", "mmf_tsy_level", "mmf_tsy_bills_level"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[frame["date"].notna()].copy()
    frame["quarter"] = frame["date"].dt.to_period("Q").astype(str)
    frame = filter_quarter_range(frame, start_quarter=start_quarter, end_quarter=end_quarter)
    frame["mmf_tsy_level"] = pd.to_numeric(frame["mmf_tsy_level"], errors="coerce")
    frame["mmf_tsy_bills_level"] = pd.to_numeric(frame["mmf_tsy_bills_level"], errors="coerce")
    rows = []
    for idx, row in frame.dropna(subset=["mmf_tsy_level", "mmf_tsy_bills_level"]).iterrows():
        quarter = str(row["quarter"])
        total = float(row["mmf_tsy_level"])
        bills = float(row["mmf_tsy_bills_level"])
        nonbill = max(total - bills, 0.0)
        for component, value, security_type in [
            ("mmf_treasury_total", total, pd.NA),
            ("mmf_treasury_bill_component", bills, "bill"),
            ("mmf_treasury_nonbill_component", nonbill, "nonbill"),
        ]:
            rows.append(
                {
                    "observation_id": pd.NA,
                    "quarter": quarter,
                    "observation_date": row["date"],
                    "source_file": str(path),
                    "source_row_key": f"{quarter}|{component}|{idx}",
                    "scope": component,
                    "holder": "Private",
                    "broad_holder": "money_market_cash",
                    "auction_class": pd.NA,
                    "security_id": pd.NA,
                    "security_type": security_type,
                    "maturity_bucket": pd.NA,
                    "measure": "treasury_security_component_level",
                    "valuation_basis": "z1_book_or_par_level",
                    "reported_value_mil": value,
                    "lower_bound_mil": value,
                    "upper_bound_mil": value,
                    "absolute_tolerance_mil": 0.0,
                    "enforcement": "hard_mmf_component_observation",
                    "priority": 8,
                    "coverage_status": "observed_aggregate_component_not_security_level",
                    "transformation": "tdcest_quarterly_inputs_mmf_total_and_bill_component",
                }
            )
    return pd.DataFrame(rows, columns=OBSERVATION_REGISTRY_COLUMNS)


def _ffiec_observations(
    ffiec_path: str | Path | None,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if ffiec_path is None:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    path = Path(ffiec_path)
    if not path.exists():
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    if "quarter" not in frame.columns:
        date_col = _first_present(frame, ["report_date", "date", "as_of_date"])
        if date_col is None:
            return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
        frame["quarter"] = pd.to_datetime(frame[date_col], errors="coerce").dt.to_period("Q").astype(str)
    frame = filter_quarter_range(frame, start_quarter=start_quarter, end_quarter=end_quarter)
    value_columns = [
        col
        for col in [
            "total_treasuries_fair_value",
            "total_treasuries_amortized_cost",
            "treasury_ladder_total",
            "treasury_bucket_3m_or_less",
            "treasury_bucket_3_12m",
            "treasury_bucket_1_3y",
            "treasury_bucket_3_5y",
            "treasury_bucket_5_15y",
            "treasury_bucket_over_15y",
        ]
        if col in frame.columns
    ]
    if not value_columns:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    rows = []
    group_cols = ["quarter"]
    bank_class_col = _first_present(frame, ["bank_class", "bank_class_name", "reporter_class"])
    if bank_class_col is not None:
        group_cols.append(bank_class_col)
    for group_key, group in frame.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        quarter = str(group_key[0])
        bank_class = group_key[1] if len(group_key) > 1 else "all_ffiec_banks"
        for value_col in value_columns:
            value = pd.to_numeric(group[value_col], errors="coerce").sum(min_count=1)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "observation_id": pd.NA,
                    "quarter": quarter,
                    "observation_date": _quarter_end_date(quarter),
                    "source_file": str(path),
                    "source_row_key": f"{quarter}|{bank_class}|{value_col}",
                    "scope": (
                        "ffiec_bank_maturity_prior"
                        if value_col.startswith("treasury_bucket_")
                        else "ffiec_bank_treasury_constraint"
                    ),
                    "holder": "Banks",
                    "broad_holder": bank_class,
                    "auction_class": pd.NA,
                    "security_id": pd.NA,
                    "security_type": pd.NA,
                    "maturity_bucket": _ffiec_maturity_bucket(value_col),
                    "measure": value_col,
                    "valuation_basis": _ffiec_valuation_basis(value_col),
                    "reported_value_mil": float(value) / 1000.0,
                    "lower_bound_mil": float(value) / 1000.0,
                    "upper_bound_mil": float(value) / 1000.0,
                    "absolute_tolerance_mil": 0.0,
                    "enforcement": (
                        "penalized_bank_maturity_prior"
                        if value_col.startswith("treasury_bucket_")
                        else "penalized_bank_subconstraint"
                    ),
                    "priority": 15,
                    "coverage_status": "observed_bank_aggregate_or_bucket",
                    "transformation": "ffiec_thousand_dollars_to_millions",
                }
            )
    return pd.DataFrame(rows, columns=OBSERVATION_REGISTRY_COLUMNS)


def _ncua_observations(
    ncua_path: str | Path | None,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if ncua_path is None:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    path = Path(ncua_path)
    if not path.exists():
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame["quarter"] = pd.to_datetime(frame["date"], errors="coerce").dt.to_period("Q").astype(str)
    frame = filter_quarter_range(frame, start_quarter=start_quarter, end_quarter=end_quarter)
    value_columns = [
        column
        for column in [
            "total_treasuries_amortized_cost",
            "total_treasuries_fair_value",
            "total_treasuries_level_proxy",
            "investment_ladder_total",
            "investment_bucket_le_1y",
            "investment_bucket_1_3y",
            "investment_bucket_3_5y",
            "investment_bucket_5_10y",
            "investment_bucket_over_10y",
        ]
        if column in frame.columns
    ]
    if not value_columns:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    rows = []
    for idx, row in frame.iterrows():
        quarter = str(row.get("quarter"))
        for value_col in value_columns:
            value = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            value_mil = float(value) / _AMOUNT_TO_MILLIONS
            rows.append(
                {
                    "observation_id": pd.NA,
                    "quarter": quarter,
                    "observation_date": row.get("date", _quarter_end_date(quarter)),
                    "source_file": str(path),
                    "source_row_key": f"{quarter}|credit_unions|{value_col}|{idx}",
                    "scope": (
                        "ncua_credit_union_maturity_proxy"
                        if value_col.startswith("investment_bucket")
                        else "ncua_credit_union_treasury_constraint"
                    ),
                    "holder": "Banks",
                    "broad_holder": "credit_unions",
                    "auction_class": pd.NA,
                    "security_id": pd.NA,
                    "security_type": pd.NA,
                    "maturity_bucket": _ncua_maturity_bucket(value_col),
                    "measure": value_col,
                    "valuation_basis": _ncua_valuation_basis(value_col),
                    "reported_value_mil": value_mil,
                    "lower_bound_mil": value_mil,
                    "upper_bound_mil": value_mil,
                    "absolute_tolerance_mil": 0.0,
                    "enforcement": "penalized_credit_union_subconstraint",
                    "priority": 15,
                    "coverage_status": "observed_credit_union_aggregate_or_bucket",
                    "transformation": str(row.get("fallback_split_basis", "ncua_dollars_to_millions")),
                }
            )
    return pd.DataFrame(rows, columns=OBSERVATION_REGISTRY_COLUMNS)


def _tier2_interest_constraint_observations(
    constraints_path: str | Path | None,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if constraints_path is None:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    path = Path(constraints_path)
    if not path.exists():
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    frame["quarter"] = pd.to_datetime(frame["date"], errors="coerce").dt.to_period("Q").astype(str)
    frame = filter_quarter_range(frame, start_quarter=start_quarter, end_quarter=end_quarter)
    value_columns = [column for column in ["level_mil", "fair_value_mil"] if column in frame.columns]
    if not value_columns:
        return pd.DataFrame(columns=OBSERVATION_REGISTRY_COLUMNS)
    rows = []
    for idx, row in frame.iterrows():
        quarter = str(row.get("quarter"))
        sector_key = str(row.get("sector_key", "")).strip()
        for value_col in value_columns:
            value = pd.to_numeric(pd.Series([row.get(value_col)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "observation_id": pd.NA,
                    "quarter": quarter,
                    "observation_date": row.get("date", _quarter_end_date(quarter)),
                    "source_file": str(path),
                    "source_row_key": f"{quarter}|{sector_key}|{row.get('component_key')}|{value_col}|{idx}",
                    "scope": "tier2_interest_source_constraint",
                    "holder": _tier2_holder(sector_key),
                    "broad_holder": sector_key,
                    "auction_class": pd.NA,
                    "security_id": pd.NA,
                    "security_type": pd.NA,
                    "maturity_bucket": pd.NA,
                    "measure": row.get("component_key", value_col),
                    "valuation_basis": row.get("constraint_basis", _tier2_constraint_basis(value_col)),
                    "reported_value_mil": float(value),
                    "lower_bound_mil": float(value),
                    "upper_bound_mil": float(value),
                    "absolute_tolerance_mil": 0.0,
                    "enforcement": "interest_proxy_reference",
                    "priority": 12,
                    "coverage_status": str(row.get("constraint_status", "documented_interest_constraint")),
                    "transformation": row.get("source_family", "tier2_interest_source_constraints"),
                }
            )
    return pd.DataFrame(rows, columns=OBSERVATION_REGISTRY_COLUMNS)


def _quarter_end_date(quarter: str) -> str:
    return pd.Period(normalize_quarter_value(quarter), freq="Q").end_time.strftime("%Y-%m-%d")


def _first_present(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _sector_valuation_basis(row: pd.Series) -> str:
    z1_series = str(row.get("z1_series", row.get("z1_code", ""))).strip().upper()
    if z1_series.startswith("LM"):
        return "z1_market_value_level"
    if z1_series.startswith("FL"):
        return "z1_book_or_par_level"
    if z1_series.startswith("FA") or str(row.get("measure", "")).lower() in {"flow", "transaction"}:
        return "z1_transaction_flow"
    return "published_aggregate_level"


def _ffiec_valuation_basis(measure: str) -> str:
    text = measure.lower()
    if "fair" in text:
        return "ffiec_fair_value"
    if "amortized" in text:
        return "ffiec_amortized_cost"
    return "ffiec_reported_ladder_total"


def _ffiec_maturity_bucket(measure: str) -> object:
    return {
        "treasury_bucket_3m_or_less": "le_3m",
        "treasury_bucket_3_12m": "3m_1y",
        "treasury_bucket_1_3y": "1_3y",
        "treasury_bucket_3_5y": "3_5y",
        "treasury_bucket_5_15y": "5_15y",
        "treasury_bucket_over_15y": "over_15y",
    }.get(str(measure), pd.NA)


def _ncua_valuation_basis(measure: str) -> str:
    text = measure.lower()
    if "fair" in text:
        return "ncua_fair_value"
    if "amortized" in text:
        return "ncua_amortized_cost"
    if "bucket" in text or "ladder" in text:
        return "ncua_maturity_bucket_proxy"
    return "ncua_reported_level_proxy"


def _ncua_maturity_bucket(measure: str) -> object:
    return {
        "investment_bucket_le_1y": "le_1y",
        "investment_bucket_1_3y": "1_3y",
        "investment_bucket_3_5y": "3_5y",
        "investment_bucket_5_10y": "5_10y",
        "investment_bucket_over_10y": "over_10y",
    }.get(measure, pd.NA)


def _tier2_holder(sector_key: str) -> str:
    text = sector_key.lower()
    if "row" in text or "foreign" in text:
        return "Foreign"
    if "bank" in text or "credit_union" in text or "depositor" in text:
        return "Banks"
    return "Private"


def _tier2_constraint_basis(value_col: str) -> str:
    return "tier2_interest_constraint_fair_value" if value_col == "fair_value_mil" else "tier2_interest_constraint_level"


def _auction_holder(value: object) -> object:
    text = str(value).strip().lower()
    return {
        "banks": "Banks",
        "federal_reserve": "CB",
        "foreign_international": "Foreign",
    }.get(text, "Private" if text else pd.NA)


def _solver_treatment_for_scope(scope: str) -> str:
    if scope == "auction_allotment_event":
        return "hard_event_evidence_in_event_ledger"
    if scope == "z1_holder_level":
        return "aggregate_holder_observation_with_basis_slack"
    if scope == "z1_transaction_flow":
        return "aggregate_transition_diagnostic_not_exact_transfer"
    if scope == "ffiec_bank_treasury_constraint":
        return "bank_subconstraint_reference_with_basis_slack"
    if scope == "ffiec_bank_maturity_prior":
        return "bank_maturity_prior_reference_with_basis_slack"
    if scope in {
        "mmf_treasury_total",
        "mmf_treasury_bill_component",
        "mmf_treasury_nonbill_component",
    }:
        return "hard_mmf_component_constraint"
    if scope in {
        "ncua_credit_union_treasury_constraint",
        "ncua_credit_union_maturity_proxy",
        "tier2_interest_source_constraint",
    }:
        return "interest_constraint_reference_with_basis_slack"
    if scope == "observed_holder_by_security_positions":
        return "not_observed_solved_endogenously"
    return "documented_source_observation"


def _coverage_status_for_scope(scope: str) -> str:
    if scope == "auction_allotment_event":
        return "observed_security_event_not_position"
    if scope == "z1_holder_level":
        return "observed_aggregate_not_security_level"
    if scope == "z1_transaction_flow":
        return "observed_aggregate_flow_not_security_level_transfer"
    if scope in {"ffiec_bank_treasury_constraint", "ffiec_bank_maturity_prior"}:
        return "observed_bank_aggregate_or_bucket"
    if scope == "ncua_credit_union_treasury_constraint":
        return "observed_credit_union_aggregate"
    if scope == "ncua_credit_union_maturity_proxy":
        return "observed_credit_union_bucket_proxy"
    if scope == "tier2_interest_source_constraint":
        return "documented_interest_constraint"
    if scope in {
        "mmf_treasury_total",
        "mmf_treasury_bill_component",
        "mmf_treasury_nonbill_component",
    }:
        return "observed_mmf_aggregate_component_not_security_level"
    return "documented"


def _basis_note(valuation_basis: object) -> str:
    text = str(valuation_basis)
    if text.startswith("z1_"):
        return "Z1 aggregate basis can differ from MSPD par/current principal."
    if text.startswith("ffiec_"):
        return "FFIEC bank basis is a bank subconstraint, not all-holder security ownership."
    if text.startswith("ncua_"):
        return "NCUA credit-union basis is a subconstraint, not all-holder security ownership."
    if "interest" in text or "constraint" in text:
        return "Tier 2 interest source constraint is a reference, not a holder-by-security fact."
    if text == "par_at_issuance":
        return "Auction allotment is an issuance event amount."
    return "Source basis documented in observation registry."


__all__ = [
    "BASIS_BRIDGE_COLUMNS",
    "COVERAGE_COLUMNS",
    "OBSERVATION_REGISTRY_COLUMNS",
    "build_exact_observation_coverage",
    "build_historical_replay_observations",
    "build_holder_basis_bridge",
]
