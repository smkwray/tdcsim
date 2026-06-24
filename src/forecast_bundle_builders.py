"""Pure row builders for non-engine CBO forecast bundle inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

from simulation_calendar import SimulationPeriod


DATE_TOLERANCE_BIL = 0.000001
CURRENT_FY_SPLICE_SCHEMA_VERSION = "tdcsim_current_fy_splice_v1"
DEBT_STOCK_PATH_SCHEMA_VERSION = "tdcsim_debt_stock_path_v1"
PRIMARY_DEFICIT_PATH_SCHEMA_VERSION = "tdcsim_primary_deficit_path_v1"
OPERATING_CASH_PATH_SCHEMA_VERSION = "tdcsim_operating_cash_path_v1"
CASH_RECONCILIATION_RESIDUAL_SCHEMA_VERSION = "tdcsim_cash_reconciliation_residual_v1"
FED_HOLDINGS_PATH_SCHEMA_VERSION = "tdcsim_fed_holdings_path_v1"
MTS_TABLE_1_DEFICIT_SELECTOR = "classification_desc=Year-to-Date;line_code_nbr=280;column=current_month_dfct_sur_amt"
MTS_TABLE_9_NET_INTEREST_SELECTOR = "classification_desc=Net Interest"
MSPD_TOTAL_NONMARKETABLE_SELECTOR = "security_type_desc=Total Nonmarketable"
MSPD_TOTAL_PUBLIC_DEBT_SELECTOR = "security_type_desc=Total Public Debt Outstanding"
DTS_TGA_CLOSING_BALANCE_SELECTOR = "account_type=Treasury General Account (TGA) Closing Balance"


def build_current_fy_splice_row(
    *,
    scenario_id: str,
    fiscal_year: int,
    simulation_start_date: date | datetime | str,
    fiscal_actuals_through: date | datetime | str,
    actuals_available_as_of: date | datetime | str,
    cbo_full_fy_primary_deficit_bil: float,
    actual_total_deficit_fytd_bil: float,
    actual_net_interest_fytd_bil: float,
    mts_table_1_selector: str,
    mts_table_9_selector: str,
    observation_date: date | datetime | str,
    available_date: date | datetime | str,
    allow_lookahead: bool = False,
    source_status: str = "mts_fytd_actuals_spliced_to_cbo_full_year_primary_deficit",
    claim_boundary: str = "current_fy_remaining_primary_deficit_only_after_fiscal_actuals",
) -> dict[str, Any]:
    """Build one ``tdcsim_current_fy_splice.csv`` row."""

    sim_start = _as_date(simulation_start_date)
    actuals_through = _as_date(fiscal_actuals_through)
    actuals_as_of = _as_date(actuals_available_as_of)
    observed = _as_date(observation_date)
    available = _as_date(available_date)
    fy_start = federal_fiscal_year_start(fiscal_year)
    fy_end = federal_fiscal_year_end(fiscal_year)

    if not fy_start < sim_start <= fy_end:
        raise ValueError("current-FY splice requires simulation_start_date inside the active fiscal year after Oct 1")
    if not fy_start <= actuals_through < sim_start:
        raise ValueError("fiscal_actuals_through must be inside the fiscal year and before simulation_start_date")
    if not allow_lookahead and actuals_as_of > sim_start:
        raise ValueError("actuals_available_as_of must be on or before simulation_start_date unless allow_lookahead")
    if not allow_lookahead and available > actuals_as_of:
        raise ValueError("available_date must be on or before actuals_available_as_of unless allow_lookahead")
    if observed > available:
        raise ValueError("observation_date must be on or before available_date")

    actual_primary = float(actual_total_deficit_fytd_bil) - float(actual_net_interest_fytd_bil)
    remaining = float(cbo_full_fy_primary_deficit_bil) - actual_primary
    return {
        "schema_version": CURRENT_FY_SPLICE_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "fiscal_year": fiscal_year,
        "simulation_start_date": sim_start.isoformat(),
        "fiscal_actuals_through": actuals_through.isoformat(),
        "actuals_available_as_of": actuals_as_of.isoformat(),
        "cbo_full_fy_primary_deficit_bil": float(cbo_full_fy_primary_deficit_bil),
        "actual_total_deficit_fytd_bil": float(actual_total_deficit_fytd_bil),
        "actual_net_interest_fytd_bil": float(actual_net_interest_fytd_bil),
        "actual_primary_deficit_fytd_bil": actual_primary,
        "remaining_cbo_primary_deficit_bil": remaining,
        "mts_table_1_selector": mts_table_1_selector,
        "mts_table_9_selector": mts_table_9_selector,
        "observation_date": observed.isoformat(),
        "available_date": available.isoformat(),
        "source_status": source_status,
        "claim_boundary": claim_boundary,
    }


def build_current_fy_splice_row_from_fiscaldata(
    *,
    scenario_id: str,
    fiscal_year: int,
    simulation_start_date: date | datetime | str,
    fiscal_actuals_through: date | datetime | str,
    actuals_available_as_of: date | datetime | str,
    cbo_full_fy_primary_deficit_bil: float,
    mts_table_1_row: Mapping[str, Any],
    mts_table_9_row: Mapping[str, Any],
    allow_lookahead: bool = False,
) -> dict[str, Any]:
    """Build the current-FY splice from exact MTS Table 1 and Table 9 rows."""

    deficit_bil = _mts_table_1_total_deficit_fytd_bil(mts_table_1_row)
    net_interest_bil = _mts_table_9_net_interest_fytd_bil(mts_table_9_row)
    observation_date = _matching_record_date(mts_table_1_row, mts_table_9_row)
    return build_current_fy_splice_row(
        scenario_id=scenario_id,
        fiscal_year=fiscal_year,
        simulation_start_date=simulation_start_date,
        fiscal_actuals_through=fiscal_actuals_through,
        actuals_available_as_of=actuals_available_as_of,
        cbo_full_fy_primary_deficit_bil=cbo_full_fy_primary_deficit_bil,
        actual_total_deficit_fytd_bil=deficit_bil,
        actual_net_interest_fytd_bil=net_interest_bil,
        mts_table_1_selector=MTS_TABLE_1_DEFICIT_SELECTOR,
        mts_table_9_selector=MTS_TABLE_9_NET_INTEREST_SELECTOR,
        observation_date=observation_date,
        available_date=actuals_available_as_of,
        allow_lookahead=allow_lookahead,
    )


def build_debt_stock_path_rows(
    *,
    scenario_id: str,
    periods: Sequence[SimulationPeriod],
    opening_state_date: date | datetime | str,
    opening_cbo_federal_debt_held_public_bil: float,
    cbo_fy_end_public_debt_targets_bil: Mapping[int, float],
    public_nonmarketable_treasury_bil: float,
    non_treasury_and_definition_residual_bil: float,
    observation_date: date | datetime | str,
    available_date: date | datetime | str,
    bridge_source: str = "debt_to_penny_and_mspd_actual_bridge",
    bridge_method: str = "constant_nominal_public_nonmarketable_and_definition_residual",
    source_status: str = "cbo_public_debt_target_bridged_to_marketable_treasury_public_target",
    claim_boundary: str = "bridge_components_not_auction_supply",
) -> list[dict[str, Any]]:
    """Build ``tdcsim_debt_stock_path.csv`` rows by actual-day interpolation."""

    if not periods:
        raise ValueError("periods must not be empty")
    opening = _as_date(opening_state_date)
    observed = _as_date(observation_date)
    available = _as_date(available_date)
    anchors = {opening: float(opening_cbo_federal_debt_held_public_bil)}
    for fiscal_year, value in cbo_fy_end_public_debt_targets_bil.items():
        anchors[federal_fiscal_year_end(int(fiscal_year))] = float(value)
    sorted_anchors = sorted(anchors.items())
    if sorted_anchors[0][0] != opening:
        raise ValueError("opening_state_date must be the first debt-stock interpolation anchor")

    rows: list[dict[str, Any]] = []
    public_nonmarketable = float(public_nonmarketable_treasury_bil)
    residual = float(non_treasury_and_definition_residual_bil)
    for period in periods:
        period_end = period.period_end
        public_target = _linear_interpolated_anchor_value(period_end, sorted_anchors)
        anchor_type = _debt_anchor_type(period_end, opening, cbo_fy_end_public_debt_targets_bil)
        source_role = "scenario_assumption"
        if anchor_type == "actual_as_of":
            source_role = "hard_actual_state"
        elif anchor_type == "cbo_fiscal_year_end":
            source_role = "official_hard_anchor"
        rows.append(
            {
                "schema_version": DEBT_STOCK_PATH_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "period_end": period_end.isoformat(),
                "cbo_federal_debt_held_public_target_bil": public_target,
                "public_nonmarketable_treasury_bil": public_nonmarketable,
                "non_treasury_and_definition_residual_bil": residual,
                "marketable_treasury_public_target_bil": public_target - public_nonmarketable - residual,
                "bridge_source": bridge_source,
                "bridge_method": bridge_method,
                "anchor_type": anchor_type,
                "interpolation_method": "linear_actual_days",
                "source_role": source_role,
                "runtime_role": "hard_target",
                "source_fiscal_year": federal_fiscal_year(period_end),
                "observation_date": observed.isoformat(),
                "available_date": available.isoformat(),
                "source_status": source_status,
                "claim_boundary": claim_boundary,
            }
        )
    return rows


def calculate_public_debt_bridge_components(
    *,
    cbo_actual_federal_debt_held_public_bil: float,
    debt_to_penny_row: Mapping[str, Any],
    mspd_rows: Sequence[Mapping[str, Any]],
    tolerance_bil: float = 0.001,
) -> dict[str, Any]:
    """Calculate the CBO public-debt to Treasury marketable-public bridge."""

    debt_date = _required_value(debt_to_penny_row, "record_date")
    debt_held_public_bil = _money_to_float(debt_to_penny_row, "debt_held_public_amt") / 1_000_000_000.0
    total_nonmarketable_row = _single_row(
        mspd_rows,
        key="security_type_desc",
        value="Total Nonmarketable",
        selector_name=MSPD_TOTAL_NONMARKETABLE_SELECTOR,
    )
    total_public_debt_row = _single_row(
        mspd_rows,
        key="security_type_desc",
        value="Total Public Debt Outstanding",
        selector_name=MSPD_TOTAL_PUBLIC_DEBT_SELECTOR,
    )
    mspd_date = _required_value(total_public_debt_row, "record_date")
    public_nonmarketable_bil = _money_to_float(total_nonmarketable_row, "debt_held_public_mil_amt") / 1_000.0
    mspd_public_debt_bil = _money_to_float(total_public_debt_row, "debt_held_public_mil_amt") / 1_000.0
    source_residual = mspd_public_debt_bil - debt_held_public_bil
    if abs(source_residual) > tolerance_bil:
        raise ValueError(
            "MSPD total public debt does not reconcile to Debt to the Penny public debt "
            f"within {tolerance_bil} billion: {source_residual:.6f}"
        )
    treasury_public_debt_bil = mspd_public_debt_bil
    date_alignment = "same_date"
    if str(mspd_date) != str(debt_date):
        date_alignment = "mspd_month_end_with_latest_prior_debt_to_penny_reconciliation"
    residual_bil = float(cbo_actual_federal_debt_held_public_bil) - treasury_public_debt_bil
    return {
        "record_date": str(mspd_date),
        "debt_to_penny_record_date": str(debt_date),
        "mspd_record_date": str(mspd_date),
        "date_alignment": date_alignment,
        "treasury_public_debt_bil": treasury_public_debt_bil,
        "debt_to_penny_public_debt_bil": debt_held_public_bil,
        "mspd_public_debt_bil": mspd_public_debt_bil,
        "mspd_debt_to_penny_residual_bil": source_residual,
        "public_nonmarketable_treasury_bil": public_nonmarketable_bil,
        "non_treasury_and_definition_residual_bil": residual_bil,
        "bridge_method": "latest_actual_constant_nominal_by_component",
        "bridge_source": "debt_to_penny_and_mspd_actual_bridge",
    }


def build_primary_deficit_path_rows(
    *,
    scenario_id: str,
    periods: Sequence[SimulationPeriod],
    primary_deficit_by_fiscal_year_bil: Mapping[int, float],
    allocation_method: str = "actual_day_weighted_equal_by_day",
    source_status: str = "annual_or_remaining_primary_deficit_allocated_by_actual_days",
    claim_boundary: str = "period_rows_sum_to_annual_or_remaining_primary_deficit",
    tolerance_bil: float = DATE_TOLERANCE_BIL,
) -> list[dict[str, Any]]:
    """Allocate annual or remaining primary deficits across period rows."""

    if not periods:
        raise ValueError("periods must not be empty")
    rows: list[dict[str, Any]] = []
    periods_by_fy: dict[int, list[SimulationPeriod]] = {}
    for period in periods:
        periods_by_fy.setdefault(federal_fiscal_year(period.period_end), []).append(period)

    for fiscal_year, fy_periods in sorted(periods_by_fy.items()):
        if fiscal_year not in primary_deficit_by_fiscal_year_bil:
            raise ValueError(f"missing primary deficit amount for FY{fiscal_year}")
        amount = float(primary_deficit_by_fiscal_year_bil[fiscal_year])
        total_days = sum(period.day_count for period in fy_periods)
        if total_days <= 0:
            raise ValueError(f"FY{fiscal_year} has no positive day-count periods")
        fy_rows: list[dict[str, Any]] = []
        allocated = 0.0
        for period in fy_periods:
            weight = period.day_count / total_days
            value = amount * weight
            allocated += value
            fy_rows.append(
                {
                    "schema_version": PRIMARY_DEFICIT_PATH_SCHEMA_VERSION,
                    "scenario_id": scenario_id,
                    "period_start": period.period_start.isoformat(),
                    "period_end": period.period_end.isoformat(),
                    "primary_deficit_bil": value,
                    "allocation_method": allocation_method,
                    "day_count_weight": weight,
                    "source_fiscal_year": fiscal_year,
                    "annual_or_remaining_primary_deficit_bil": amount,
                    "source_role": "hard_input",
                    "runtime_role": "hard_flow",
                    "source_status": source_status,
                    "claim_boundary": claim_boundary,
                }
            )
        if abs(allocated - amount) > tolerance_bil:
            raise AssertionError(
                f"FY{fiscal_year} primary deficit allocation residual {allocated - amount:.12f} exceeds tolerance"
            )
        rows.extend(fy_rows)
    return rows


def build_operating_cash_path_rows(
    *,
    scenario_id: str,
    periods: Sequence[SimulationPeriod],
    base_date: date | datetime | str,
    base_balance_bil: float,
    observation_date: date | datetime | str,
    available_date: date | datetime | str,
    inflation_scalar: float = 1.0,
    base_inflation_index_level: float | None = None,
    inflation_index_by_period_end: Mapping[date | str, float] | Callable[[date], float | None] | None = None,
    operating_cash_definition: str = "tga_only",
    reserve_settlement_component: str = "tga",
    component_coverage: str = "tga_only_frozen_input",
    source_status: str = "frozen_tga_opening_balance_policy_path",
    claim_boundary: str = "operating_cash_proxy_not_debt_target_or_issuance_supply",
) -> list[dict[str, Any]]:
    """Build operating-cash target rows with nominal or CPI-U indexed balance."""

    if not periods:
        raise ValueError("periods must not be empty")
    if operating_cash_definition != "tga_only":
        raise ValueError("first-tranche default builder supports tga_only operating cash")
    if reserve_settlement_component != "tga":
        raise ValueError("tga_only operating cash must use tga as the reserve settlement component")
    if inflation_scalar < 0.0:
        raise ValueError("inflation_scalar must be nonnegative")
    if inflation_scalar and base_inflation_index_level is None:
        raise ValueError("constant-real operating cash requires base_inflation_index_level")

    base = _as_date(base_date)
    observed = _as_date(observation_date)
    available = _as_date(available_date)
    construction_mode = "constant_nominal"
    if inflation_scalar:
        construction_mode = "constant_real_cbo_cpi_u" if inflation_scalar == 1.0 else "partial_cbo_cpi_u_indexation"

    rows: list[dict[str, Any]] = []
    for period in periods:
        cpi_level = _inflation_index_level(period.period_end, inflation_index_by_period_end)
        target = float(base_balance_bil)
        if inflation_scalar:
            if cpi_level is None:
                raise ValueError(f"missing CBO CPI-U level for {period.period_end.isoformat()}")
            target *= (cpi_level / float(base_inflation_index_level)) ** inflation_scalar
        rows.append(
            {
                "schema_version": OPERATING_CASH_PATH_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "period_end": period.period_end.isoformat(),
                "operating_cash_target_bil": target,
                "tga_target_bil": target,
                "ttl_target_bil": 0.0,
                "other_operating_cash_target_bil": 0.0,
                "operating_cash_definition": operating_cash_definition,
                "reserve_settlement_component": reserve_settlement_component,
                "construction_mode": construction_mode,
                "base_date": base.isoformat(),
                "base_balance_bil": float(base_balance_bil),
                "inflation_index": "cbo_cpi_u",
                "inflation_index_level": cpi_level if cpi_level is not None else "",
                "inflation_scalar": float(inflation_scalar),
                "component_coverage": component_coverage,
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "observation_date": observed.isoformat(),
                "available_date": available.isoformat(),
                "source_status": source_status,
                "claim_boundary": claim_boundary,
            }
        )
    return rows


def tga_closing_balance_bil_from_dts_row(row: Mapping[str, Any]) -> float:
    """Return the TGA closing balance in billions from the exact DTS cash row."""

    account_type = str(_required_value(row, "account_type"))
    if account_type != "Treasury General Account (TGA) Closing Balance":
        raise ValueError("DTS operating cash row must be Treasury General Account (TGA) Closing Balance")
    return _money_to_float(row, "open_today_bal") / 1_000.0


def build_cash_reconciliation_residual_rows(
    *,
    scenario_id: str,
    periods: Sequence[SimulationPeriod],
    cash_reconciliation_residual_bil: float | Mapping[date | str, float] | Callable[[SimulationPeriod], float],
    source_status: str = "unmodeled_nonbudget_cash_flow_reconciles_operating_cash_only",
    claim_boundary: str = "cash_residual_does_not_change_debt_issuance_primary_or_tdc_flows",
) -> list[dict[str, Any]]:
    """Build residual rows that affect operating cash only."""

    if not periods:
        raise ValueError("periods must not be empty")
    rows: list[dict[str, Any]] = []
    for period in periods:
        residual = _residual_value(period, cash_reconciliation_residual_bil)
        rows.append(
            {
                "schema_version": CASH_RECONCILIATION_RESIDUAL_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "period_start": period.period_start.isoformat(),
                "period_end": period.period_end.isoformat(),
                "cash_reconciliation_residual_bil": residual,
                "component_type": "unmodeled_nonbudget_cash_flow",
                "affects_operating_cash": True,
                "affects_primary_deficit": False,
                "affects_net_interest": False,
                "affects_total_deficit": False,
                "affects_debt_target": False,
                "affects_issuance_size": False,
                "affects_tdc_fiscal_flow": False,
                "source_role": "scenario_assumption",
                "runtime_role": "reconciliation_only",
                "source_status": source_status,
                "claim_boundary": claim_boundary,
            }
        )
    return rows


def build_fed_holdings_path_rows(
    *,
    scenario_id: str,
    periods: Sequence[SimulationPeriod],
    opening_state_date: date | datetime | str,
    opening_cb_holdings_bil: float,
    cbo_fy_end_fed_holdings_bil: Mapping[int, float],
    observation_date: date | datetime | str,
    available_date: date | datetime | str,
    source_status: str = "cbo_fed_holdings_interpolated_to_period_path",
    claim_boundary: str = "fed_holdings_path_guides_holder_allocation_not_total_issuance",
) -> list[dict[str, Any]]:
    """Build a period Fed/CB Treasury holdings target path.

    CBO reports Federal Reserve holdings at fiscal-year-end. The engine runs at
    a finer frequency, so the first source-backed tranche uses the same
    actual-day interpolation convention as the public-debt target.
    """

    if not periods:
        raise ValueError("periods must not be empty")
    opening = _as_date(opening_state_date)
    observed = _as_date(observation_date)
    available = _as_date(available_date)
    anchors = {opening: float(opening_cb_holdings_bil)}
    for fiscal_year, value in cbo_fy_end_fed_holdings_bil.items():
        anchors[federal_fiscal_year_end(int(fiscal_year))] = float(value)
    sorted_anchors = sorted(anchors.items())
    if sorted_anchors[0][0] != opening:
        raise ValueError("opening_state_date must be the first Fed holdings interpolation anchor")

    rows: list[dict[str, Any]] = []
    for period in periods:
        period_end = period.period_end
        target = _linear_interpolated_anchor_value(period_end, sorted_anchors)
        rows.append(
            {
                "schema_version": FED_HOLDINGS_PATH_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "period_end": period_end.isoformat(),
                "holder_type": "CB",
                "cbo_fed_holdings_target_bil": target,
                "interpolation_method": "linear_actual_days",
                "source_fiscal_year": federal_fiscal_year(period_end),
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "observation_date": observed.isoformat(),
                "available_date": available.isoformat(),
                "source_status": source_status,
                "claim_boundary": claim_boundary,
            }
        )
    return rows


def federal_fiscal_year(value: date | datetime | str) -> int:
    day = _as_date(value)
    return day.year + 1 if day.month >= 10 else day.year


def federal_fiscal_year_start(fiscal_year: int) -> date:
    return date(int(fiscal_year) - 1, 10, 1)


def federal_fiscal_year_end(fiscal_year: int) -> date:
    return date(int(fiscal_year), 9, 30)


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"expected date-like value, got {type(value).__name__}")


def _required_value(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"required FiscalData field missing: {key}")
    return value


def _money_to_float(row: Mapping[str, Any], key: str) -> float:
    value = _required_value(row, key)
    if str(value).lower() == "null":
        raise ValueError(f"FiscalData field is null: {key}")
    return float(str(value).replace(",", ""))


def _mts_table_1_total_deficit_fytd_bil(row: Mapping[str, Any]) -> float:
    desc = str(_required_value(row, "classification_desc"))
    line_code = str(_required_value(row, "line_code_nbr"))
    if desc != "Year-to-Date" or line_code != "280":
        raise ValueError(
            "MTS Table 1 row must be Year-to-Date line_code_nbr=280 using current_month_dfct_sur_amt"
        )
    amount = _money_to_float(row, "current_month_dfct_sur_amt")
    return amount / 1_000_000_000.0


def _mts_table_9_net_interest_fytd_bil(row: Mapping[str, Any]) -> float:
    desc = str(_required_value(row, "classification_desc"))
    if desc != "Net Interest":
        raise ValueError("MTS Table 9 row must be Net Interest")
    amount = _money_to_float(row, "current_fytd_rcpt_outly_amt")
    return amount / 1_000_000_000.0


def _matching_record_date(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    left_date = str(_required_value(left, "record_date"))
    right_date = str(_required_value(right, "record_date"))
    if left_date != right_date:
        raise ValueError("FiscalData source rows must share record_date")
    return left_date


def _single_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    value: str,
    selector_name: str,
) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get(key)) == value]
    if len(matches) != 1:
        raise ValueError(f"{selector_name} matched {len(matches)} rows")
    return matches[0]


def _linear_interpolated_anchor_value(target_date: date, anchors: Sequence[tuple[date, float]]) -> float:
    if target_date < anchors[0][0] or target_date > anchors[-1][0]:
        raise ValueError(f"period_end {target_date.isoformat()} falls outside debt-stock anchor range")
    for anchor_date, anchor_value in anchors:
        if target_date == anchor_date:
            return anchor_value
    previous_date, previous_value = anchors[0]
    for next_date, next_value in anchors[1:]:
        if previous_date <= target_date <= next_date:
            total_days = (next_date - previous_date).days
            elapsed_days = (target_date - previous_date).days
            if total_days <= 0:
                raise ValueError("debt-stock anchors must be strictly increasing by date")
            return previous_value + (next_value - previous_value) * (elapsed_days / total_days)
        previous_date, previous_value = next_date, next_value
    raise AssertionError("unreachable debt interpolation state")


def _debt_anchor_type(
    period_end: date,
    opening_state_date: date,
    cbo_fy_end_public_debt_targets_bil: Mapping[int, float],
) -> str:
    if period_end == opening_state_date:
        return "actual_as_of"
    cbo_anchor_dates = {federal_fiscal_year_end(int(fiscal_year)) for fiscal_year in cbo_fy_end_public_debt_targets_bil}
    if period_end in cbo_anchor_dates:
        return "cbo_fiscal_year_end"
    return "interpolated_assumption"


def _inflation_index_level(
    period_end: date,
    source: Mapping[date | str, float] | Callable[[date], float | None] | None,
) -> float | None:
    if source is None:
        return None
    if callable(source):
        value = source(period_end)
        return None if value is None else float(value)
    if period_end in source:
        return float(source[period_end])
    key = period_end.isoformat()
    if key in source:
        return float(source[key])
    return None


def _residual_value(
    period: SimulationPeriod,
    source: float | Mapping[date | str, float] | Callable[[SimulationPeriod], float],
) -> float:
    if callable(source):
        return float(source(period))
    if isinstance(source, Mapping):
        if period.period_end in source:
            return float(source[period.period_end])
        key = period.period_end.isoformat()
        if key in source:
            return float(source[key])
        raise ValueError(f"missing cash residual for {period.period_end.isoformat()}")
    return float(source)
