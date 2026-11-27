"""Pure row builders for non-engine CBO forecast bundle inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from forecast_paths import (
    CBO_FED_REMITTANCE_ROLE,
    FED_INCOME_PATH_SCHEMA_VERSION,
    FED_REMITTANCE_PATH_SCHEMA_VERSION,
    operating_cash_runtime_role,
)
from simulation_calendar import SimulationPeriod, is_fedwire_business_day


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
CBO_REMITTANCE_SOURCE_FILE = "cbo/fed_remittances_table_4_3.json"
CBO_REMITTANCE_PDF_SHA256 = "e101aa05f85c93f5c37667015bf8e15a17f618611a8942c58650efe2a4c4ba84"


def cbo_remittance_source_fixture() -> dict[str, Any]:
    """Transcription of the published levels and totals, in USD billions."""

    return {
        "source_url": "https://www.govinfo.gov/content/pkg/CMR-Y10-00199317/pdf/CMR-Y10-00199317.pdf",
        "source_sha256": CBO_REMITTANCE_PDF_SHA256,
        "pdf_page": 102,
        "printed_page": 94,
        "table": "4-3",
        "row": "Federal Reserve remittances",
        "levels_bil": {
            str(year): value
            for year, value in enumerate(
                ("7.9", "8.6", "9.2", "14.8", "37.9", "93.9", "129.7", "140.5", "152.9", "164.5", "176.4"),
                start=2026,
            )
        },
        "published_totals_bil": {"2027-2031": "164.3", "2027-2036": "928.4"},
    }


FED_INCOME_SOURCE_FILE = "fed/fed_income_sources.json"
FED_INCOME_DOCUMENTS = {
    "h41": "fed/h41_2026-06-18.htm",
    "fomc_implementation_note": "fed/fomc_implementation_note_2026-06-17.htm",
    "nyfed_soma_projection_data": "fed/nyfed_omo2025_projection_data.xlsx",
    "nyfed_soma_annual_report": "fed/nyfed_omo2025_report.pdf",
    "soma_agency_holdings": "fed/nyfed_soma_agency_holdings_2026-06-17.json",
    "reserve_banks_financial_statements": "fed/reserve_banks_combined_financial_statements_2025.pdf",
}
FED_INCOME_CLAIM_BOUNDARY = "fed_income_lines_from_stated_sources_cash_basis_not_a_fed_forecast"
DAYS_PER_YEAR = 365.25


def fed_income_source_fixture() -> dict[str, Any]:
    """Transcription of every Fed income and balance-sheet source, with each original's hash.

    Money amounts are USD millions as published unless the key says otherwise.
    """

    return {
        "schema_version": "tdcsim_fed_income_sources_v1",
        "documents": {
            "h41": {
                "file": FED_INCOME_DOCUMENTS["h41"],
                "url": "https://www.federalreserve.gov/releases/h41/20260618/",
                "sha256": "519777d077d4806251238aa67d97cf3f476a07673f6c3d2f323f0e26823ee2a6",
                "title": "H.4.1 Factors Affecting Reserve Balances, release of June 18, 2026",
            },
            "fomc_implementation_note": {
                "file": FED_INCOME_DOCUMENTS["fomc_implementation_note"],
                "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a1.htm",
                "sha256": "2db878e0d9d82665c907fe1475bff29f68377622120b7fb8df497ee8aea49d34",
                "title": "Implementation Note issued June 17, 2026",
            },
            "nyfed_soma_projection_data": {
                "file": FED_INCOME_DOCUMENTS["nyfed_soma_projection_data"],
                "url": "https://www.newyorkfed.org/medialibrary/media/markets/omo/omo2025-data.xlsx",
                "sha256": "ccafeef7f0107caf74397593575e42defaf69a16f7dbee2736370da3876b7ce3",
                "title": "Projections chart data for Open Market Operations during 2025 (released April 2026)",
            },
            "nyfed_soma_annual_report": {
                "file": FED_INCOME_DOCUMENTS["nyfed_soma_annual_report"],
                "url": "https://www.newyorkfed.org/medialibrary/media/markets/omo/omo2025-pdf.pdf",
                "sha256": "0eb6a6990dfc05bf80542cd54ef89565f8b6351c4c7e7d5d4138a93dad08d600",
                "title": "Open Market Operations during 2025, Federal Reserve Bank of New York",
            },
            "soma_agency_holdings": {
                "file": FED_INCOME_DOCUMENTS["soma_agency_holdings"],
                "url": "https://markets.newyorkfed.org/api/soma/agency/get/asof/2026-06-17.json",
                "sha256": "970a0622ddb2fbf880ddfe77f4edb44d3e209090306ea0ee424ca27ce3be2df7",
                "title": "SOMA agency holdings by CUSIP as of June 17, 2026",
            },
            "reserve_banks_financial_statements": {
                "file": FED_INCOME_DOCUMENTS["reserve_banks_financial_statements"],
                "url": "https://www.federalreserve.gov/aboutthefed/files/combinedfinstmt2025.pdf",
                "sha256": "3989982bc83546f15747ac86e98b695be0894d2fbd8d829b3000f3a9dcfd1ada",
                "title": "Federal Reserve Banks Combined Financial Statements, 2025 and 2024",
            },
        },
        "h41_wednesday_levels_mil": {
            "as_of_date": "2026-06-17",
            "reserve_balances_with_federal_reserve_banks": "2936355",
            "us_treasury_general_account": "956502",
            "reverse_repurchase_agreements": "335625",
            "reverse_repurchase_agreements_foreign_official": "328797",
            "reverse_repurchase_agreements_others": "6828",
            "federal_reserve_notes_net": "2419354",
            "mortgage_backed_securities": "1964798",
            "us_treasury_securities_held_outright": "4487319",
            "other_liabilities_and_capital": "-181188",
            "earnings_remittances_due_to_the_us_treasury": "-236697",
            "tables": "table 1 Wednesday level (reserves, TGA, reverse repos, other liabilities and capital); "
            "table 5 (notes, MBS, Treasury securities); table 6 Total (earnings remittances due)",
        },
        "administered_rates_pct": {
            "effective_date": "2026-06-18",
            "interest_on_reserve_balances": "3.65",
            "overnight_reverse_repo_offering_rate": "3.5",
            "federal_funds_target_range": "3.50-3.75",
        },
        "soma_agency_mbs": {
            "as_of_date": "2026-06-17",
            "current_face_mil": "1964797.733684066",
            "weighted_average_coupon_pct": "2.4985508468642283",
            "method": "current-face-weighted coupon of every MBS and CMBS CUSIP; the coupon is the percentage in "
            "the security description",
        },
        "nyfed_soma_projection": {
            "agency_mbs_year_end_bil": {
                "2025": "2039", "2026": "1861", "2027": "1692", "2028": "1533", "2029": "1387",
                "2030": "1249", "2031": "1112", "2032": "990", "2033": "877", "2034": "771", "2035": "677",
            },
            "soma_net_income_baseline_bil": {
                "2026": "37.2", "2027": "63.9", "2028": "78.5", "2029": "96.5", "2030": "112",
                "2031": "126.7", "2032": "139.8", "2033": "150.7", "2034": "161.4", "2035": "174.3",
            },
            "reserve_balances_year_end_bil": {
                "2026": "3068", "2027": "3195", "2028": "3325", "2029": "3457", "2030": "3604",
                "2031": "3748", "2032": "3897", "2033": "4053", "2034": "4216", "2035": "4385",
            },
            "liability_growth_pct": "4.0",
            "liability_growth_rule": "Federal Reserve liabilities grow from December 2025 levels in line with "
            "nominal GDP; long-run nominal GDP growth 4.0 percent (Appendix 4)",
            "deferred_asset_extinguished": "early 2030",
            "administered_rate_rule": "IORB 10 basis points below the top of the target range; ON RRP at the bottom",
            "sheets": "Chart 27 (Agency MBS), Chart 28 (Baseline), Chart 25 (Baseline)",
        },
        "reserve_banks_2025_mil": {
            "operating_expenses": "9800",
            "dividends_on_capital_stock": "1688",
            "other_income": {
                "loans_to_depository_institutions_and_other_loans": "216",
                "securities_purchased_under_agreements_to_resell": "70",
                "gse_debt_securities": "129",
                "foreign_currency_denominated_investments": "291",
                "central_bank_liquidity_swaps": "4",
                "treasury_securities_losses_net": "-42",
                "mbs_losses_net": "-72",
                "soma_other": "33",
                "income_from_services": "542",
                "reimbursable_services_to_government_agencies": "934",
                "other_components_of_net_benefit_costs": "88",
                "other": "80",
                "consolidated_variable_interest_entities_net": "-14",
            },
            "excluded_valuation_items": {
                "foreign_currency_translation_gains": "1572",
                "other_comprehensive_loss": "-796",
            },
            "remittances_transferred_while_deferred_asset_grew": "6281",
            "deferred_asset_year_end": "243481",
            "statement": "Combined Statements of Operations and Note 12, year ended December 31, 2025",
        },
        "rules": {
            "income_line_growth": "operating expenses, dividends and other income grow from their 2025 levels, "
            "centered on 2025-07-02, at the NY Fed projection's 4.0 percent liability growth",
            "currency": "Federal Reserve notes grow from the H.4.1 level at 4.0 percent a year",
            "mbs_runoff": "H.4.1 MBS scaled by the NY Fed year-end path, log-linear between year ends; after "
            "2035 the path keeps the 2034-2035 runoff rate",
            "mbs_income": "weighted-average coupon on period-start current face, actual/365.25",
            "short_rate": "the run's own nominal curve at 0.25 years; IORB and ON RRP keep their spreads to the "
            "package curve's 0.25-year rate at the simulation start",
            "bank_level_remittances": "dropped: the rule is system-level, so no remittance is paid while the "
            "system deferred asset is positive",
        },
    }


def _fed_growth(start: date, end: date, rate_pct: Decimal) -> float:
    return float(1 + rate_pct / 100) ** ((end - start).days / DAYS_PER_YEAR)


def _mbs_projection_level(day: date, year_end_bil: Mapping[str, str]) -> float:
    import math

    points = sorted((date(int(year), 12, 31), math.log(float(value))) for year, value in year_end_bil.items())
    if day < points[0][0]:
        raise ValueError("Fed MBS path starts after the requested date")
    for (left_day, left), (right_day, right) in zip(points, points[1:]):
        if day <= right_day:
            weight = (day - left_day).days / (right_day - left_day).days
            return math.exp(left + (right - left) * weight)
    (left_day, left), (right_day, right) = points[-2], points[-1]
    slope = (right - left) / (right_day - left_day).days
    return math.exp(right + slope * (day - right_day).days)


def build_fed_income_path_rows(
    *,
    scenario_id: str,
    periods: Sequence[SimulationPeriod],
    yield_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Daily Fed income lines and balance-sheet paths from the transcribed sources."""

    source = fed_income_source_fixture()
    h41 = source["h41_wednesday_levels_mil"]
    as_of = date.fromisoformat(h41["as_of_date"])
    projection = source["nyfed_soma_projection"]
    growth = Decimal(projection["liability_growth_pct"])
    banks = source["reserve_banks_2025_mil"]
    base_day = date(2025, 7, 2)
    other_income = sum(Decimal(value) for value in banks["other_income"].values()) / 1000
    operating = Decimal(banks["operating_expenses"]) / 1000
    dividends = Decimal(banks["dividends_on_capital_stock"]) / 1000
    wac = float(source["soma_agency_mbs"]["weighted_average_coupon_pct"]) / 100
    h41_mbs = float(Decimal(h41["mortgage_backed_securities"]) / 1000)
    h41_currency = float(Decimal(h41["federal_reserve_notes_net"]) / 1000)
    mbs_anchor = _mbs_projection_level(as_of, projection["agency_mbs_year_end_bil"])

    def mbs(day: date) -> float:
        return h41_mbs * _mbs_projection_level(day, projection["agency_mbs_year_end_bil"]) / mbs_anchor

    def currency(day: date) -> float:
        return h41_currency * _fed_growth(as_of, day, growth)

    start = _as_date(periods[0].period_start)
    curve_rows = [
        row for row in yield_rows
        if str(row["scenario_id"]) == scenario_id and _as_date(row["curve_date"]) <= start
        and abs(float(row["tenor_years"]) - 0.25) < 1e-12
    ]
    if not curve_rows:
        raise ValueError("the package curve has no 0.25-year rate at the simulation start")
    opening_short = max(curve_rows, key=lambda row: _as_date(row["curve_date"]))["nominal_rate"]
    rates = source["administered_rates_pct"]
    constants = {
        "h41_as_of_date": as_of.isoformat(),
        "h41_reserve_balances_bil": float(Decimal(h41["reserve_balances_with_federal_reserve_banks"]) / 1000),
        "h41_tga_bil": float(Decimal(h41["us_treasury_general_account"]) / 1000),
        "h41_on_rrp_bil": float(Decimal(h41["reverse_repurchase_agreements"]) / 1000),
        "h41_soma_mbs_bil": h41_mbs,
        "h41_currency_bil": h41_currency,
        "h41_earnings_remittances_due_bil": float(Decimal(h41["earnings_remittances_due_to_the_us_treasury"]) / 1000),
        "h41_other_liabilities_and_capital_bil": float(Decimal(h41["other_liabilities_and_capital"]) / 1000),
        "short_rate_tenor_years": 0.25,
        "iorb_spread_pct": float(Decimal(rates["interest_on_reserve_balances"]) - Decimal(str(opening_short))),
        "on_rrp_spread_pct": float(Decimal(rates["overnight_reverse_repo_offering_rate"]) - Decimal(str(opening_short))),
    }
    rows = []
    for period in periods:
        first, last = _as_date(period.period_start), _as_date(period.period_end)
        year_fraction = (last - first).days / DAYS_PER_YEAR
        scale = _fed_growth(base_day, last, growth) * year_fraction
        rows.append({
            "schema_version": FED_INCOME_PATH_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "period_start": first.isoformat(),
            "period_end": last.isoformat(),
            "soma_mbs_start_bil": mbs(first),
            "soma_mbs_end_bil": mbs(last),
            "soma_mbs_interest_bil": wac * mbs(first) * year_fraction,
            "currency_start_bil": currency(first),
            "currency_end_bil": currency(last),
            "other_income_bil": float(other_income) * scale,
            "operating_expenses_bil": float(operating) * scale,
            "dividends_bil": float(dividends) * scale,
            **constants,
            "source_role": "scenario_assumption",
            "runtime_role": "hard_flow",
            "source_status": "h41_nyfed_soma_projection_reserve_bank_2025_statements",
            "claim_boundary": FED_INCOME_CLAIM_BOUNDARY,
        })
    return rows


def read_dts_fed_earnings_rows(path: str | Path, fiscal_year: int) -> list[dict[str, str]]:
    """Read only the fiscal year's DTS Table II Fed earnings observations."""

    start = federal_fiscal_year_start(fiscal_year).isoformat()
    end = federal_fiscal_year_end(fiscal_year).isoformat()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        selector = {
            "table_nbr": "II",
            "account_type": "Treasury General Account (TGA)",
            "transaction_type": "Deposits",
            "transaction_catg": "Federal Reserve Earnings",
        }
        if not {*selector, "record_date", "transaction_fytd_amt"} <= set(reader.fieldnames or ()):
            raise ValueError("DTS Table II source is missing Federal Reserve Earnings columns")
        return [
            row for row in reader
            if all(row[key] == value for key, value in selector.items())
            and start <= row["record_date"] <= end
        ]


def read_dts_fed_earnings(path: str | Path, splice: Mapping[str, Any]) -> dict[str, str]:
    """Select the Table II TGA deposit row for the splice's last DTS date."""

    cutoff = _as_date(splice["fiscal_actuals_through"])
    dts_day = cutoff
    while not is_fedwire_business_day(dts_day):
        dts_day -= timedelta(days=1)
    matches = [
        row for row in read_dts_fed_earnings_rows(path, int(splice["fiscal_year"]))
        if row["record_date"] == dts_day.isoformat()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one DTS Federal Reserve Earnings row on {dts_day}; found {len(matches)}")
    return matches[0]


def build_fed_remittance_path_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any],
    dts_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply the primary flow's Fedwire-day weights to the fixed remittance levels."""

    amounts = {
        int(year): Decimal(value) for year, value in cbo_remittance_source_fixture()["levels_bil"].items()
    }
    splice_year = int(splice["fiscal_year"])
    try:
        fytd_mil = Decimal(dts_row["transaction_fytd_amt"])
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("invalid FYTD Federal Reserve Earnings in DTS Table II") from exc
    if not fytd_mil.is_finite() or fytd_mil < 0:
        raise ValueError("invalid FYTD Federal Reserve Earnings in DTS Table II")
    amounts[splice_year] -= fytd_mil / 1_000
    if amounts[splice_year] < 0:
        raise ValueError(f"remaining FY{splice_year} Fed remittances are negative")
    rows = []
    for primary in primary_rows:
        year = int(primary["source_fiscal_year"])
        annual = float(amounts[year])
        rows.append({
            "schema_version": FED_REMITTANCE_PATH_SCHEMA_VERSION,
            "scenario_id": primary["scenario_id"],
            "period_start": primary["period_start"],
            "period_end": primary["period_end"],
            "source_fiscal_year": year,
            "remittance_bil": annual * float(primary["day_count_weight"]),
            "annual_or_remaining_remittance_bil": annual,
            "source_role": "hard_input",
            "runtime_role": "hard_flow",
            "source_status": "cbo_table_4_3_net_of_dts_fytd" if year == splice_year else "cbo_table_4_3",
            "claim_boundary": CBO_FED_REMITTANCE_ROLE,
        })
    return rows


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
    # A retrospective opening can use pre-start observations released after the
    # start. The availability gate is the information vintage, not the model clock.
    if actuals_as_of < actuals_through:
        raise ValueError("actuals_available_as_of must be on or after fiscal_actuals_through")
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


def opening_cbo_public_debt_bil(marketable: float, nonmarketable: float, definition_residual: float) -> float:
    """Bridge the opening marketable debt base to CBO's public debt definition."""
    return marketable + nonmarketable + definition_residual


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
    allocation_method: str = "equal_by_fedwire_business_day",
    source_status: str = "annual_or_remaining_primary_deficit_allocated_by_fedwire_business_days",
    claim_boundary: str = "period_rows_sum_to_annual_or_remaining_primary_deficit",
    tolerance_bil: float = DATE_TOLERANCE_BIL,
) -> list[dict[str, Any]]:
    """Allocate annual or remaining primary deficits across period rows.

    Each fiscal year's amount is spread equally over its Fedwire business days, and a
    period takes the share of the business days it contains: Treasury cash moves only
    when Fedwire is open, so a weekend or holiday period books nothing.
    """

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
        business_days = {
            id(period): sum(
                1
                for offset in range(1, period.day_count + 1)
                if is_fedwire_business_day(period.period_start + timedelta(days=offset))
            )
            for period in fy_periods
        }
        total_days = sum(business_days.values())
        if total_days <= 0:
            raise ValueError(f"FY{fiscal_year} has no Fedwire business days")
        fy_rows: list[dict[str, Any]] = []
        allocated = 0.0
        for period in fy_periods:
            weight = business_days[id(period)] / total_days
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
    for row_index, period in enumerate(periods):
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
                "runtime_role": operating_cash_runtime_role(row_index),
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
    if periods[0].period_start != opening:
        raise ValueError(
            "opening_state_date must equal the first simulation period start"
        )

    rows: list[dict[str, Any]] = [
        {
            "schema_version": FED_HOLDINGS_PATH_SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "period_end": opening.isoformat(),
            "holder_type": "CB",
            "cbo_fed_holdings_target_bil": float(opening_cb_holdings_bil),
            "interpolation_method": "opening_state_identity",
            "source_fiscal_year": federal_fiscal_year(opening),
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "observation_date": observed.isoformat(),
            "available_date": available.isoformat(),
            "source_status": "opening_fed_stock_target_identity",
            "claim_boundary": claim_boundary,
        }
    ]
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
