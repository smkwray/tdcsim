"""CBO forecast source-contract parsing helpers.

This module intentionally keeps the first CBO parser slice dependency-light:
XLSX workbooks are read directly from the zipped XML package so the parser does
not require openpyxl.
"""

from __future__ import annotations

import hashlib
import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZipFile


BUDGET_WORKBOOK_SHA256 = "06593fcc3b8517806994090a6a9ffe748cfdd19514d2f9d2f18a1841b64b33a5"
ECONOMIC_WORKBOOK_SHA256 = "ae8f4920702fabf8fb3136bc94a42c53466cff5e890dec22396d8dc49dc2f776"
BUDGET_WORKBOOK_URL = "https://www.cbo.gov/system/files/2026-02/51118-2026-02-Budget-Projections.xlsx"
ECONOMIC_WORKBOOK_URL = "https://www.cbo.gov/system/files/2026-02/51135-2026-02-Economic-Projections.xlsx"
FORECAST_NAME = "cbo_2026_02_baseline"
SCHEMA_VERSION = "cbo_source_fixture_v1"
FISCAL_BASELINE_SCHEMA_VERSION = "tdcsim_cbo_fiscal_baseline_v1"
MACRO_FORECAST_SCHEMA_VERSION = "tdcsim_macro_forecast_path_v1"
FORECAST_PUBLICATION_DATE = "2026-02-11"
IDENTITY_TOLERANCE_BIL = 0.001


@dataclass(frozen=True)
class WorksheetRow:
    row_number: int
    values: tuple[str, ...]


@dataclass(frozen=True)
class SourceFixture:
    schema_version: str
    forecast_name: str
    source_family: str
    source_file: str
    source_sha256: str
    source_url: str
    source_sheet: str
    source_row_number: int
    source_row_selector: str
    source_unit_block: str
    source_year_or_period: str
    raw_value: float
    raw_sign_convention: str
    canonical_transform: str
    canonical_value: float
    observation_date: str
    available_date: str
    source_status: str


@dataclass(frozen=True)
class CboBudgetContract:
    source_path: Path
    source_sha256: str
    fiscal_years: dict[int, dict[str, float]]
    fixtures: dict[tuple[str, int], SourceFixture]

    def value(self, field: str, fiscal_year: int) -> float:
        return self.fiscal_years[fiscal_year][field]

    def fixture(self, field: str, fiscal_year: int) -> SourceFixture:
        return self.fixtures[(field, fiscal_year)]

    def fixture_rows(self) -> list[dict[str, str | float | int]]:
        return [_fixture_to_row(fixture) for fixture in self.fixtures.values()]


@dataclass(frozen=True)
class CboEconomicQuarterlyContract:
    source_path: Path
    source_sha256: str
    quarters: dict[str, dict[str, float]]
    fixtures: dict[tuple[str, str], SourceFixture]

    def value(self, field: str, quarter: str) -> float:
        return self.quarters[quarter][field]

    def fixture(self, field: str, quarter: str) -> SourceFixture:
        return self.fixtures[(field, quarter)]

    def fixture_rows(self) -> list[dict[str, str | float | int]]:
        return [_fixture_to_row(fixture) for fixture in self.fixtures.values()]


@dataclass(frozen=True)
class CboIdentityResidual:
    fiscal_year: int
    primary_deficit_bil: float
    cbo_net_interest_bil: float
    cbo_total_deficit_bil: float
    residual_bil: float


@dataclass(frozen=True)
class CboDebtContinuityResidual:
    fiscal_year: int
    debt_held_public_begin_bil: float
    cbo_total_deficit_bil: float
    cbo_other_means_financing_bil: float
    debt_identity_end_bil: float
    residual_bil: float


_BUDGET_SELECTORS = {
    "cbo_net_interest_bil": {
        "sheet": "Table 1-1",
        "row_number": 23,
        "label": "Net interest",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "positive",
        "canonical_transform": "preserve_raw_value",
    },
    "cbo_total_deficit_bil": {
        "sheet": "Table 1-1",
        "row_number": 27,
        "label": "Total deficit (-)",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "negative_deficit",
        "canonical_transform": "negate_raw_value",
    },
    "primary_deficit_bil": {
        "sheet": "Table 1-1",
        "row_number": 30,
        "label": "Primary deficit (-)",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "negative_deficit",
        "canonical_transform": "negate_raw_value",
    },
    "debt_held_public_end_bil": {
        "sheet": "Table 1-1",
        "row_number": 31,
        "label": "Debt held by the public",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "positive_stock",
        "canonical_transform": "preserve_raw_value",
    },
    "debt_held_public_begin_bil": {
        "sheet": "Table 1-3",
        "row_number": 9,
        "label": "Debt held by the public at the beginning of the year",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "positive_stock",
        "canonical_transform": "preserve_raw_value",
    },
    "cbo_other_means_financing_bil": {
        "sheet": "Table 1-3",
        "row_number": 13,
        "label": "Resulting from other means of financing",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "signed_source_value",
        "canonical_transform": "preserve_raw_value",
    },
    "debt_identity_end_bil": {
        "sheet": "Table 1-3",
        "row_number": 17,
        "label": "Debt held by the public at the end of the year",
        "numeric_label": "In billions of dollars",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "positive_stock",
        "canonical_transform": "preserve_raw_value",
    },
    "cbo_financial_assets_end_bil": {
        "sheet": "Table 1-3",
        "row_number": 21,
        "label": "Federal financial assets",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "positive_stock",
        "canonical_transform": "preserve_raw_value",
    },
    "cbo_fed_holdings_end_bil": {
        "sheet": "Table 1-3",
        "row_number": 27,
        "label": "Federal Reserve's holdings of debt held by the public",
        "unit_block": "billions_of_dollars",
        "raw_sign_convention": "positive_stock",
        "canonical_transform": "preserve_raw_value",
    },
    "cbo_average_interest_rate_pct": {
        "sheet": "Table 1-3",
        "row_number": 37,
        "label": "Average interest rate on debt held by the public (percent)",
        "unit_block": "percent",
        "raw_sign_convention": "percent_points",
        "canonical_transform": "preserve_raw_value",
    },
}

_ECONOMIC_QUARTERLY_SELECTORS = {
    "cbo_cpi_u_index": {
        "sheet": "1. Quarterly",
        "row_number": 53,
        "label": "Consumer price index, all urban consumers (CPI-U)",
        "unit_block": "1982-84=100",
        "raw_sign_convention": "index",
    },
    "cbo_10y_treasury_rate_pct": {
        "sheet": "1. Quarterly",
        "row_number": 103,
        "label": "10-Year Treasury note",
        "unit_block": "Percent",
        "raw_sign_convention": "percent_points",
    },
    "cbo_3m_tbill_rate_pct": {
        "sheet": "1. Quarterly",
        "row_number": 104,
        "label": "3-Month Treasury bill",
        "unit_block": "Percent",
        "raw_sign_convention": "percent_points",
    },
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_cbo_workbook_hashes(
    budget_workbook_path: str | Path,
    economic_workbook_path: str | Path,
) -> dict[str, str]:
    budget_sha = _verify_workbook_sha256(
        Path(budget_workbook_path),
        expected_sha256=BUDGET_WORKBOOK_SHA256,
        workbook_label="budget",
    )
    economic_sha = _verify_workbook_sha256(
        Path(economic_workbook_path),
        expected_sha256=ECONOMIC_WORKBOOK_SHA256,
        workbook_label="economic",
    )
    return {"budget": budget_sha, "economic": economic_sha}


def parse_cbo_budget_source_contract(
    workbook_path: str | Path,
    *,
    verify_hash: bool = True,
    expected_sha256: str = BUDGET_WORKBOOK_SHA256,
) -> CboBudgetContract:
    path = Path(workbook_path)
    source_sha256 = (
        _verify_workbook_sha256(path, expected_sha256=expected_sha256, workbook_label="budget")
        if verify_hash
        else sha256_file(path)
    )
    workbook = _read_xlsx(path)
    missing_sheets = {spec["sheet"] for spec in _BUDGET_SELECTORS.values()} - set(workbook)
    if missing_sheets:
        raise ValueError(f"CBO budget workbook missing required sheets: {sorted(missing_sheets)}")

    fiscal_years: dict[int, dict[str, float]] = {}
    fixtures: dict[tuple[str, int], SourceFixture] = {}
    sheet_headers = {
        "Table 1-1": _fiscal_year_columns(workbook["Table 1-1"], required_start=2026, required_end=2036),
        "Table 1-3": _fiscal_year_columns(workbook["Table 1-3"], required_start=2026, required_end=2036),
    }
    table_1_1_units = _table_1_1_unit_blocks(workbook["Table 1-1"])
    table_1_3_units = _table_1_3_unit_blocks(workbook["Table 1-3"])

    for field, spec in _BUDGET_SELECTORS.items():
        sheet_name = str(spec["sheet"])
        if field == "debt_identity_end_bil":
            numeric_rows = [row for row in workbook[sheet_name] if row.row_number == int(spec["row_number"])]
            if len(numeric_rows) != 1:
                raise ValueError(
                    "Expected exactly one CBO Table 1-3 debt identity numeric row "
                    f"{spec['row_number']}; found {len(numeric_rows)}"
                )
            row = numeric_rows[0]
            label_row = _require_row(workbook[sheet_name], 16, str(spec["label"]))
            if _normalize_label(_cell(row, 0)) != _normalize_label(str(spec["numeric_label"])):
                raise ValueError("CBO Table 1-3 debt identity numeric row is not the billions row")
            source_row_selector = _cell(label_row, 0)
        else:
            row = _require_row(workbook[sheet_name], int(spec["row_number"]), str(spec["label"]))
            source_row_selector = _cell(row, 0)

        unit_block = (
            table_1_1_units.get(row.row_number)
            if sheet_name == "Table 1-1"
            else table_1_3_units.get(row.row_number)
        )
        if unit_block != spec["unit_block"]:
            raise ValueError(
                f"{sheet_name} row {row.row_number} unit block mismatch: "
                f"expected {spec['unit_block']}, found {unit_block}"
            )

        for col_idx, fiscal_year in sheet_headers[sheet_name].items():
            raw = _parse_float(_cell(row, col_idx), sheet_name, row.row_number, fiscal_year)
            canonical = _canonical_value(raw, str(spec["canonical_transform"]))
            fiscal_years.setdefault(fiscal_year, {})[field] = canonical
            fixtures[(field, fiscal_year)] = _fixture(
                field=field,
                source_path=path,
                source_sha256=source_sha256,
                source_url=BUDGET_WORKBOOK_URL,
                source_family="cbo_budget_feb_2026_workbook",
                source_sheet=sheet_name,
                source_row_number=row.row_number,
                source_row_selector=source_row_selector,
                source_unit_block=str(spec["unit_block"]),
                source_year_or_period=str(fiscal_year),
                raw_value=raw,
                raw_sign_convention=str(spec["raw_sign_convention"]),
                canonical_transform=str(spec["canonical_transform"]),
                canonical_value=canonical,
                source_status="verified_feb_2026_cbo_workbook",
            )

    return CboBudgetContract(
        source_path=path,
        source_sha256=source_sha256,
        fiscal_years=fiscal_years,
        fixtures=fixtures,
    )


def parse_cbo_economic_quarterly_source_contract(
    workbook_path: str | Path,
    *,
    verify_hash: bool = True,
    expected_sha256: str = ECONOMIC_WORKBOOK_SHA256,
) -> CboEconomicQuarterlyContract:
    path = Path(workbook_path)
    source_sha256 = (
        _verify_workbook_sha256(path, expected_sha256=expected_sha256, workbook_label="economic")
        if verify_hash
        else sha256_file(path)
    )
    workbook = _read_xlsx(path)
    if "1. Quarterly" not in workbook:
        raise ValueError("CBO economic workbook missing required sheet: 1. Quarterly")

    rows = workbook["1. Quarterly"]
    quarter_cols = _quarter_columns(rows)
    quarters: dict[str, dict[str, float]] = {}
    fixtures: dict[tuple[str, str], SourceFixture] = {}

    for field, spec in _ECONOMIC_QUARTERLY_SELECTORS.items():
        row = _require_row(rows, int(spec["row_number"]), str(spec["label"]))
        if _cell(row, 1) != spec["unit_block"]:
            raise ValueError(
                f"1. Quarterly row {row.row_number} unit mismatch: "
                f"expected {spec['unit_block']}, found {_cell(row, 1)}"
            )
        for col_idx, quarter in quarter_cols.items():
            raw = _parse_float(_cell(row, col_idx), "1. Quarterly", row.row_number, quarter)
            quarters.setdefault(quarter, {})[field] = raw
            fixtures[(field, quarter)] = _fixture(
                field=field,
                source_path=path,
                source_sha256=source_sha256,
                source_url=ECONOMIC_WORKBOOK_URL,
                source_family="cbo_economic_feb_2026_workbook",
                source_sheet="1. Quarterly",
                source_row_number=row.row_number,
                source_row_selector=_cell(row, 0),
                source_unit_block=str(spec["unit_block"]),
                source_year_or_period=quarter,
                raw_value=raw,
                raw_sign_convention=str(spec["raw_sign_convention"]),
                canonical_transform="preserve_raw_value",
                canonical_value=raw,
                source_status="verified_feb_2026_cbo_quarterly_workbook",
            )

    return CboEconomicQuarterlyContract(
        source_path=path,
        source_sha256=source_sha256,
        quarters=quarters,
        fixtures=fixtures,
    )


def parse_cbo_grouped_table_2_1_exact_annual_data(_workbook_path: str | Path) -> None:
    raise ValueError(
        "Grouped Table 2-1 values must not be used as exact annual runtime data; "
        "use the economic workbook '1. Quarterly' sheet instead."
    )


def validate_cbo_deficit_identity(
    budget_contract: CboBudgetContract,
    *,
    tolerance_bil: float = IDENTITY_TOLERANCE_BIL,
) -> list[CboIdentityResidual]:
    residuals: list[CboIdentityResidual] = []
    failures: list[str] = []
    for fiscal_year in sorted(budget_contract.fiscal_years):
        values = budget_contract.fiscal_years[fiscal_year]
        required = {"primary_deficit_bil", "cbo_net_interest_bil", "cbo_total_deficit_bil"}
        if not required <= set(values):
            continue
        residual = (
            values["primary_deficit_bil"]
            + values["cbo_net_interest_bil"]
            - values["cbo_total_deficit_bil"]
        )
        residuals.append(
            CboIdentityResidual(
                fiscal_year=fiscal_year,
                primary_deficit_bil=values["primary_deficit_bil"],
                cbo_net_interest_bil=values["cbo_net_interest_bil"],
                cbo_total_deficit_bil=values["cbo_total_deficit_bil"],
                residual_bil=residual,
            )
        )
        if abs(residual) > tolerance_bil:
            failures.append(f"{fiscal_year}: {residual:.12f}")
    if failures:
        raise ValueError(
            "CBO deficit identity residual exceeds "
            f"{tolerance_bil:.6f} billion for fiscal years: {', '.join(failures)}"
        )
    return residuals


def validate_cbo_debt_continuity(
    budget_contract: CboBudgetContract,
    *,
    tolerance_bil: float = IDENTITY_TOLERANCE_BIL,
) -> list[CboDebtContinuityResidual]:
    residuals: list[CboDebtContinuityResidual] = []
    failures: list[str] = []
    for fiscal_year in sorted(budget_contract.fiscal_years):
        values = budget_contract.fiscal_years[fiscal_year]
        required = {
            "debt_held_public_begin_bil",
            "cbo_total_deficit_bil",
            "cbo_other_means_financing_bil",
            "debt_identity_end_bil",
        }
        if not required <= set(values):
            continue
        residual = (
            values["debt_held_public_begin_bil"]
            + values["cbo_total_deficit_bil"]
            + values["cbo_other_means_financing_bil"]
            - values["debt_identity_end_bil"]
        )
        residuals.append(
            CboDebtContinuityResidual(
                fiscal_year=fiscal_year,
                debt_held_public_begin_bil=values["debt_held_public_begin_bil"],
                cbo_total_deficit_bil=values["cbo_total_deficit_bil"],
                cbo_other_means_financing_bil=values["cbo_other_means_financing_bil"],
                debt_identity_end_bil=values["debt_identity_end_bil"],
                residual_bil=residual,
            )
        )
        if abs(residual) > tolerance_bil:
            failures.append(f"{fiscal_year}: {residual:.12f}")
    if failures:
        raise ValueError(
            "CBO debt continuity residual exceeds "
            f"{tolerance_bil:.6f} billion for fiscal years: {', '.join(failures)}"
        )
    return residuals


def build_cbo_fiscal_baseline_rows(
    budget_contract: CboBudgetContract,
    *,
    scenario_id: str = "baseline",
) -> list[dict[str, str | float | int]]:
    """Build compact `tdcsim_cbo_fiscal_baseline.csv` rows in memory."""

    deficit_residuals = {
        residual.fiscal_year: residual
        for residual in validate_cbo_deficit_identity(budget_contract)
    }
    debt_residuals = {
        residual.fiscal_year: residual
        for residual in validate_cbo_debt_continuity(budget_contract)
    }
    rows: list[dict[str, str | float | int]] = []
    for fiscal_year in sorted(budget_contract.fiscal_years):
        values = budget_contract.fiscal_years[fiscal_year]
        required = {
            "primary_deficit_bil",
            "cbo_net_interest_bil",
            "cbo_total_deficit_bil",
            "debt_held_public_begin_bil",
            "debt_held_public_end_bil",
            "debt_identity_end_bil",
            "cbo_other_means_financing_bil",
            "cbo_financial_assets_end_bil",
            "cbo_fed_holdings_end_bil",
            "cbo_average_interest_rate_pct",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(f"CBO fiscal baseline missing fields for FY{fiscal_year}: {sorted(missing)}")
        source_row_selector = "; ".join(
            budget_contract.fixture(field, fiscal_year).source_row_selector
            for field in (
                "primary_deficit_bil",
                "cbo_net_interest_bil",
                "cbo_total_deficit_bil",
                "debt_held_public_end_bil",
            )
        )
        rows.append(
            {
                "schema_version": FISCAL_BASELINE_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "fiscal_year": fiscal_year,
                "primary_deficit_bil": values["primary_deficit_bil"],
                "cbo_net_interest_bil": values["cbo_net_interest_bil"],
                "cbo_total_deficit_bil": values["cbo_total_deficit_bil"],
                "debt_held_public_begin_bil": values["debt_held_public_begin_bil"],
                "debt_held_public_end_bil": values["debt_held_public_end_bil"],
                "debt_identity_end_bil": values["debt_identity_end_bil"],
                "cbo_other_means_financing_bil": values["cbo_other_means_financing_bil"],
                "cbo_financial_assets_end_bil": values["cbo_financial_assets_end_bil"],
                "cbo_fed_holdings_end_bil": values["cbo_fed_holdings_end_bil"],
                "cbo_average_interest_rate_pct": values["cbo_average_interest_rate_pct"],
                "source_role": "hard_input",
                "runtime_role": "hard_flow",
                "source_vintage": FORECAST_NAME,
                "forecast_publication_date": FORECAST_PUBLICATION_DATE,
                "source_as_of": FORECAST_PUBLICATION_DATE,
                "observation_date": FORECAST_PUBLICATION_DATE,
                "available_date": FORECAST_PUBLICATION_DATE,
                "source_family": "cbo_budget_feb_2026_workbook",
                "source_table": "Table 1-1; Table 1-3",
                "source_row_selector": source_row_selector,
                "source_status": (
                    "verified_cbo_budget_fiscal_baseline;"
                    f"deficit_identity_residual_bil={deficit_residuals[fiscal_year].residual_bil:.12f};"
                    f"debt_continuity_residual_bil={debt_residuals[fiscal_year].residual_bil:.12f}"
                ),
                "claim_boundary": (
                    "cbo_primary_deficit_hard_input_net_interest_total_deficit_and_debt_continuity_checks"
                ),
            }
        )
    return rows


def build_cbo_macro_forecast_path_rows(
    economic_contract: CboEconomicQuarterlyContract,
    *,
    scenario_id: str = "baseline",
) -> list[dict[str, str | float]]:
    """Build compact `tdcsim_macro_forecast_path.csv` rows in memory."""

    rows: list[dict[str, str | float]] = []
    sorted_quarters = sorted(economic_contract.quarters, key=_quarter_sort_key)
    for quarter in sorted_quarters:
        values = economic_contract.quarters[quarter]
        required = {"cbo_3m_tbill_rate_pct", "cbo_10y_treasury_rate_pct", "cbo_cpi_u_index"}
        missing = required - set(values)
        if missing:
            raise ValueError(f"CBO macro path missing fields for {quarter}: {sorted(missing)}")
        previous_quarter = _previous_quarter(quarter)
        previous_cpi = economic_contract.quarters.get(previous_quarter, {}).get("cbo_cpi_u_index")
        inflation_pct = 0.0
        inflation_status = "first_available_quarter_no_prior_cpi"
        if previous_cpi:
            inflation_pct = (values["cbo_cpi_u_index"] / previous_cpi - 1.0) * 100.0
            inflation_status = "quarter_over_quarter_cpi_level_change"
        period_start, period_end = _quarter_dates(quarter)
        rows.append(
            {
                "schema_version": MACRO_FORECAST_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "cbo_3m_tbill_rate_pct": values["cbo_3m_tbill_rate_pct"],
                "cbo_10y_treasury_rate_pct": values["cbo_10y_treasury_rate_pct"],
                "cbo_cpi_u_index": values["cbo_cpi_u_index"],
                "cbo_cpi_u_inflation_pct": inflation_pct,
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "source_vintage": FORECAST_NAME,
                "forecast_publication_date": FORECAST_PUBLICATION_DATE,
                "source_table": "1. Quarterly",
                "source_row_selector": (
                    "Consumer price index, all urban consumers (CPI-U); "
                    "3-Month Treasury bill; 10-Year Treasury note"
                ),
                "observation_date": FORECAST_PUBLICATION_DATE,
                "available_date": FORECAST_PUBLICATION_DATE,
                "source_status": f"verified_cbo_quarterly_macro_workbook;{inflation_status}",
                "claim_boundary": "quarterly_cbo_macro_anchors_not_full_runtime_yield_curve",
            }
        )
    return rows


def write_forecast_rows_csv(
    path: str | Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        raise ValueError("Cannot write empty forecast CSV")
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _verify_workbook_sha256(path: Path, *, expected_sha256: str, workbook_label: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"CBO {workbook_label} workbook SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, found {actual}"
        )
    return actual


def _read_xlsx(path: Path) -> dict[str, list[WorksheetRow]]:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{main_ns}}}si"):
                shared_strings.append(
                    _clean("".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t")))
                )
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{pkg_rel_ns}}}Relationship")
        }
        sheets: dict[str, list[WorksheetRow]] = {}
        for sheet in workbook.findall(f".//{{{main_ns}}}sheet"):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{rel_ns}}}id"]
            target = rel_targets[rel_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            sheet_root = ET.fromstring(archive.read(target))
            rows: list[WorksheetRow] = []
            for row_node in sheet_root.findall(f".//{{{main_ns}}}row"):
                values: dict[int, str] = {}
                max_col = -1
                for cell in row_node.findall(f"{{{main_ns}}}c"):
                    col_idx = _cell_col_index(cell.attrib.get("r", ""))
                    if col_idx < 0:
                        continue
                    values[col_idx] = _cell_value(cell, shared_strings, main_ns)
                    max_col = max(max_col, col_idx)
                row_number = int(row_node.attrib.get("r", len(rows) + 1))
                rows.append(
                    WorksheetRow(
                        row_number=row_number,
                        values=tuple(values.get(idx, "") for idx in range(max_col + 1)),
                    )
                )
            sheets[name] = rows
    return sheets


def _cell_col_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return -1
    col_idx = 0
    for char in match.group(1):
        col_idx = col_idx * 26 + (ord(char) - ord("A") + 1)
    return col_idx - 1


def _cell_value(cell: ET.Element, shared_strings: list[str], main_ns: str) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return _clean("".join(node.text or "" for node in cell.iter(f"{{{main_ns}}}t")))
    value_node = cell.find(f"{{{main_ns}}}v")
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (IndexError, ValueError):
            return raw
    return _clean(raw)


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_label(value: object) -> str:
    return _clean(value).lower()


def _cell(row: WorksheetRow, col_idx: int) -> str:
    if col_idx >= len(row.values):
        return ""
    return row.values[col_idx]


def _require_row(rows: list[WorksheetRow], row_number: int, label: str) -> WorksheetRow:
    matches = [
        row
        for row in rows
        if row.row_number == row_number and _normalize_label(_cell(row, 0)) == _normalize_label(label)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one CBO source row {row_number} with label {label!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _fiscal_year_columns(
    rows: list[WorksheetRow],
    *,
    required_start: int,
    required_end: int,
) -> dict[int, int]:
    required = {str(year) for year in range(required_start, required_end + 1)}
    for row in rows:
        year_by_col = {
            value: idx
            for idx, value in enumerate(row.values)
            if re.fullmatch(r"20[0-9]{2}", _clean(value))
        }
        if required <= set(year_by_col):
            return {year_by_col[str(year)]: year for year in range(required_start, required_end + 1)}
    raise ValueError(f"Missing fiscal-year headers {required_start}-{required_end}")


def _quarter_columns(rows: list[WorksheetRow]) -> dict[int, str]:
    for row in rows:
        quarters = {
            idx: _clean(value)
            for idx, value in enumerate(row.values)
            if re.fullmatch(r"20[0-9]{2}Q[1-4]", _clean(value))
        }
        if "2026Q1" in quarters.values() and "2036Q4" in quarters.values():
            return quarters
    raise ValueError("Missing quarterly headers including 2026Q1 and 2036Q4")


def _table_1_1_unit_blocks(rows: list[WorksheetRow]) -> dict[int, str]:
    unit_by_row: dict[int, str] = {}
    current = ""
    for row in rows:
        joined = _normalize_label(" ".join(row.values))
        if "in billions of dollars" in joined:
            current = "billions_of_dollars"
        elif "as a percentage of gdp" in joined:
            current = "percent_of_gdp"
        unit_by_row[row.row_number] = current
    return unit_by_row


def _table_1_3_unit_blocks(rows: list[WorksheetRow]) -> dict[int, str]:
    unit_by_row: dict[int, str] = {}
    current = ""
    for row in rows:
        label = _cell(row, 0)
        joined = _normalize_label(" ".join(row.values))
        if joined == "billions of dollars" or label == "In billions of dollars":
            current = "billions_of_dollars"
        elif label == "Addendum:" or label == "Federal Reserve's holdings of debt held by the public":
            current = "billions_of_dollars"
        elif label == "As a percentage of GDP":
            current = "percent_of_gdp"
        if "(percent)" in _normalize_label(label):
            unit_by_row[row.row_number] = "percent"
        else:
            unit_by_row[row.row_number] = current
    return unit_by_row


def _parse_float(value: str, sheet: str, row_number: int, period: int | str) -> float:
    try:
        return float(value.replace(",", ""))
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric CBO value in {sheet} row {row_number}, period {period}: {value!r}"
        ) from exc


def _canonical_value(raw_value: float, canonical_transform: str) -> float:
    if canonical_transform == "negate_raw_value":
        return -raw_value
    if canonical_transform == "preserve_raw_value":
        return raw_value
    raise ValueError(f"Unsupported canonical transform: {canonical_transform}")


def _quarter_sort_key(quarter: str) -> tuple[int, int]:
    match = re.fullmatch(r"(20[0-9]{2})Q([1-4])", quarter)
    if not match:
        raise ValueError(f"Invalid CBO quarter label: {quarter!r}")
    return int(match.group(1)), int(match.group(2))


def _previous_quarter(quarter: str) -> str:
    year, qtr = _quarter_sort_key(quarter)
    if qtr == 1:
        return f"{year - 1}Q4"
    return f"{year}Q{qtr - 1}"


def _quarter_dates(quarter: str) -> tuple[date, date]:
    year, qtr = _quarter_sort_key(quarter)
    start_month = 1 + (qtr - 1) * 3
    start = date(year, start_month, 1)
    if qtr == 4:
        next_start = date(year + 1, 1, 1)
    else:
        next_start = date(year, start_month + 3, 1)
    return start, next_start - timedelta(days=1)


def _fixture(
    *,
    field: str,
    source_path: Path,
    source_sha256: str,
    source_url: str,
    source_family: str,
    source_sheet: str,
    source_row_number: int,
    source_row_selector: str,
    source_unit_block: str,
    source_year_or_period: str,
    raw_value: float,
    raw_sign_convention: str,
    canonical_transform: str,
    canonical_value: float,
    source_status: str,
) -> SourceFixture:
    return SourceFixture(
        schema_version=SCHEMA_VERSION,
        forecast_name=FORECAST_NAME,
        source_family=source_family,
        source_file=source_path.name,
        source_sha256=source_sha256,
        source_url=source_url,
        source_sheet=source_sheet,
        source_row_number=source_row_number,
        source_row_selector=source_row_selector,
        source_unit_block=source_unit_block,
        source_year_or_period=source_year_or_period,
        raw_value=raw_value,
        raw_sign_convention=raw_sign_convention,
        canonical_transform=canonical_transform,
        canonical_value=canonical_value,
        observation_date="2026-02-11",
        available_date="2026-02-11",
        source_status=f"{source_status}:{field}",
    )


def _fixture_to_row(fixture: SourceFixture) -> dict[str, str | float | int]:
    return {
        "schema_version": fixture.schema_version,
        "forecast_name": fixture.forecast_name,
        "source_family": fixture.source_family,
        "source_file": fixture.source_file,
        "source_sha256": fixture.source_sha256,
        "source_url": fixture.source_url,
        "source_sheet": fixture.source_sheet,
        "source_row_number": fixture.source_row_number,
        "source_row_selector": fixture.source_row_selector,
        "source_unit_block": fixture.source_unit_block,
        "source_year_or_period": fixture.source_year_or_period,
        "raw_value": fixture.raw_value,
        "raw_sign_convention": fixture.raw_sign_convention,
        "canonical_transform": fixture.canonical_transform,
        "canonical_value": fixture.canonical_value,
        "observation_date": fixture.observation_date,
        "available_date": fixture.available_date,
        "source_status": fixture.source_status,
    }
