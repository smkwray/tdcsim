import csv
import hashlib
import json
import platform
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
SRC_DIR = PROJECT_ROOT_FOR_IMPORT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.verify_cbo_forecast_package import REQUIRED_FILES, create_package_zip, verify_package
from cbo_opening_actuals import (
    ACTUALS_AVAILABLE_AS_OF, DTS_CASH_SOURCE, DTS_FISCAL_LINES, DTS_FLOWS_SOURCE,
    DTS_TGA_CLOSING_ACCOUNT, DTS_TGA_OPENING_ACCOUNT, OPENING_CLOSE_DATE,
    OPENING_PORTFOLIO_CLAIM_BOUNDARY, OPENING_ROLLFORWARD_STATUS, SOMA_SNAPSHOT_DATE, SOMA_SOURCE,
    build_opening_fy_splice, dts_fiscal_selectors, opening_cash_reconciliation, roll_opening_marketables,
)
from tdc_shared import PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, PRIVATE_SUBBUCKET_MMF
from cbo_yield_curve_surface import build_yield_curve_surface_rows
from forecast_bundle_builders import (
    CBO_REMITTANCE_SOURCE_FILE,
    FED_INCOME_SOURCE_FILE,
    build_fed_income_path_rows,
    build_fed_remittance_path_rows,
    build_primary_deficit_path_rows,
    cbo_remittance_source_fixture,
    fed_income_source_fixture,
    read_dts_fed_earnings,
)
from fed_income import (
    FED_INCOME_RESULT_COLUMNS,
    INTEREST_EXPENSE_DAY_COUNT,
    non_treasury_reserve_effect,
    remittance_week,
    settle_remittances,
)
from forecast_paths import (
    CBO_FED_REMITTANCE_EXCLUDED_CLAIMS,
    CBO_FED_REMITTANCE_OPENING_STATUS,
    CBO_FED_REMITTANCE_ROLE,
    CBO_FED_REMITTANCE_STATUS,
    FED_INCOME_PATH_FILE,
    FED_REMITTANCE_PATH_FILE,
)
from simulation_calendar import is_fedwire_business_day
from forecast_input_builder import (
    build_cbo_fiscal_baseline_rows,
    build_cbo_macro_forecast_path_rows,
    parse_cbo_budget_source_contract,
    parse_cbo_economic_quarterly_source_contract,
)
from tips_indexation import build_monthly_tips_cpi_path_rows, build_tips_real_yield_path_rows
from issuance_calibration import (
    SETTLEMENT_CALENDAR, build_issuance_calibration,
)
from sim_engine import _fiscal_year_close_dates
from simulation_calendar import build_simulation_calendar
from issuance_calibration import AUCTION_REQUIRED_COLUMNS
from test_issuance_rule import _fixture_auctions, _write_auctions


RUN_ROWS = 3755
BUDGET_WORKBOOK = PROJECT_ROOT_FOR_IMPORT.parent / "ratewall" / "data" / "raw" / "cbo" / "51118-2026-02-Budget-Projections.xlsx"
ECONOMIC_WORKBOOK = PROJECT_ROOT_FOR_IMPORT.parent / "ratewall" / "data" / "raw" / "cbo" / "51135-2026-02-Economic-Projections.xlsx"
FED_INCOME_SOURCES = PROJECT_ROOT_FOR_IMPORT / "data" / "fed_income_sources"
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _require_cbo_workbooks() -> None:
    missing = [path for path in (BUDGET_WORKBOOK, ECONOMIC_WORKBOOK) if not path.is_file()] + [
        FED_INCOME_SOURCES / Path(document["file"]).name
        for document in fed_income_source_fixture()["documents"].values()
        if not (FED_INCOME_SOURCES / Path(document["file"]).name).is_file()
    ]
    if missing:
        pytest.fail(
            "CBO package-verifier integration inputs are missing: "
            + ", ".join(path.name for path in missing),
            pytrace=False,
        )


RESULT_COLUMNS = [
    "Date",
    "TotalDebt_Agg",
    "CBOControlledDebtTarget",
    "CBOControlledDebtReference",
    "ScenarioControlledDebt",
    "DebtDriftFromReference",
    "CBOControlledDebtTargetApplicable",
    "CBOControlledDebtPostIssuance",
    "CBOControlledDebtTargetError",
    "CBORequiredFaceIssuance",
    "IssuanceNeedProceeds",
    "PendingIssuanceProceeds",
    "EarlySettlementProceeds",
    "CashManagementBillProceeds",
    "FedRolloverFace",
    "FedRolloverProceeds",
    "FedRolloverCashSizingDifference",
    "FedRedeemedFace",
    "CashFinancingProceeds",
    "NewDebtIssued",
    "AuctionProceeds",
    "IssueDiscountCost_Period",
    "InterestPaid_Bonds",
    "NonMarketableInterestCapitalized_Period",
    "PrimaryDeficit",
    "TGA",
    "CBOOperatingCashTarget",
    "CBOCashResidual",
    "CBOCashReconciliationResidual",
    "CBORemittanceCashEffect",
    "CBOFedHoldingsTarget",
    "CBOFedHoldingsTargetApplicable",
    "CBOFedHoldingsTargetError",
    "CBOFedAuctionShare",
    "CBOFedSecondaryPurchaseFace",
    "CBOFedSecondaryPurchaseCash",
    "CBOFedSecondaryPurchaseReserveEffect",
    "CBOFedSecondaryPurchaseDepositEffect",
    "CBOFedSecondarySaleCash",
    "CBOFedSecondarySaleReserveEffect",
    "CBOFedSecondarySaleDepositEffect",
    "CBOFedPrivateMaturityTDC",
    "CBOFedAcquisitionChannel",
    "CBOFedSecondarySaleBuyerMix",
    "CBOFedBeginStock",
    "CBOFedMaturitiesAndRedemptions",
    "CBOFedTipsPrincipalIndexation",
    "CBOFedAuctionRolloverAddons",
    "CBOFedSyntheticSecondaryPurchases",
    "CBOFedSyntheticSecondarySales",
    "CBOFedEndStock",
    "CBOFedGrossStockFlow",
    "CBOFedNetStockChange",
    "CBOFedStockMode",
    "CBOFedSettlementScope",
    "CBORemittanceStatus",
    "CB_TreasuryInterestCashReceived",
    "CB_NetIncome",
    "CB_Remittance",
    "CB_DeferredAsset",
    *FED_INCOME_RESULT_COLUMNS,
    "Reserves",
    "CBONetInterestDiagnostic",
    "CBOTotalDeficitDiagnostic",
    "NetInterestDiagnosticStatus",
    "DebtHeld_Banks",
    "DebtHeld_CentralBank",
    "DebtHeld_Foreign",
    "DebtHeld_DomesticNonBanks",
    "DebtHeldByType_Fixed",
    "DebtHeldByType_TIPS",
    "DebtHeldByType_FRN",
    "TIPSInflationAccretion_Period",
    "TIPSInflationAccretion_Cumulative",
    "CBOBuybackFaceRetired",
    "CBOBuybackCashPaid",
]


def test_cbo_package_verifier_checks_hashes_required_files_semantics_and_zip_unpack(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")

    summary = verify_package(package)
    assert summary["status"] == "pass"
    assert summary["artifact_hashes"] >= len(REQUIRED_FILES) - 2

    zip_path = create_package_zip(package, tmp_path / "pkg.zip")
    zip_summary = verify_package(zip_path)
    assert zip_summary["status"] == "pass"


def test_cbo_package_verifier_rejects_semantic_results_tampering_with_refreshed_hashes(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    # A drift that does not equal scenario debt less CBO's path is a fabricated diagnostic.
    _mutate_csv_cell(
        package / "exports_baseline" / "results_compact.csv",
        row_index=1,
        column="DebtDriftFromReference",
        value="-2.0",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="debt drift identity"):
        verify_package(package)


def test_cbo_package_verifier_rejects_pending_issuance_that_appears_from_nowhere(tmp_path: Path) -> None:
    """Pending issuance only grows by decided need and shrinks by what settles; a pending
    balance that jumps (with the TGA moved to hide it) is fabricated cash."""

    package = _write_valid_package(tmp_path / "pkg")
    results_path = package / "exports_baseline" / "results_compact.csv"
    rows = _read_csv(results_path)
    rows[10]["PendingIssuanceProceeds"] = "5.0"
    rows[10]["TGA"] = str(float(rows[10]["CBOOperatingCashTarget"]) - 5.0)
    rows[10]["CBOCashResidual"] = "-5.0"
    _write_csv(results_path, list(rows[0]), rows)
    _rewrite_summary(results_path, package / "exports_baseline" / "summary.json")
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="pending issuance identity"):
        verify_package(package)


def test_cbo_package_verifier_rejects_pending_cash_at_a_fiscal_year_end(tmp_path: Path) -> None:
    """Fiscal-year-end stocks are what scenarios compare; pending cash there must be zero."""

    package = _write_valid_package(tmp_path / "pkg")
    results_path = package / "exports_baseline" / "results_compact.csv"
    rows = _read_csv(results_path)
    index = next(i for i, row in enumerate(rows) if row["Date"] == "2027-09-30")
    # A run that skipped the close: three units stay pending, consistently booked.
    rows[index]["PendingIssuanceProceeds"] = "3.0"
    rows[index]["TGA"] = str(float(rows[index]["CBOOperatingCashTarget"]) - 3.0)
    rows[index]["CBOCashResidual"] = "-3.0"
    for column in ("AuctionProceeds", "CashFinancingProceeds"):
        rows[index][column] = "1.0"
    rows[index + 1]["IssuanceNeedProceeds"] = "1.0"
    _write_csv(results_path, list(rows[0]), rows)
    _rewrite_summary(results_path, package / "exports_baseline" / "summary.json")
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="pending issuance at the fiscal-year end"):
        verify_package(package)


def test_cbo_package_verifier_rejects_source_tampering_with_refreshed_manifests_and_hashes(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    source_path = package / "sources" / "fiscaldata" / "mts_table_1_deficit_2026-05-31.csv"
    _write_csv(
        source_path,
        ["record_date", "classification_desc", "line_code_nbr", "current_month_dfct_sur_amt"],
        [
            {
                "record_date": "2026-05-31",
                "classification_desc": "Year-to-Date",
                "line_code_nbr": "280",
                "current_month_dfct_sur_amt": "999.0",
            }
        ],
    )
    _refresh_source_package_hashes(package)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="current_month_dfct_sur_amt"):
        verify_package(package)


def test_cbo_package_verifier_rejects_fed_bridge_tampering_with_refreshed_hashes(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    _mutate_csv_cell(
        package / "exports_baseline" / "results_compact.csv",
        row_index=1,
        column="CBOFedSecondaryPurchaseFace",
        value="999.0",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="Fed secondary purchase face"):
        verify_package(package)


def _add_fed_rollover(package: Path, *, add_on: float) -> None:
    """Roll 2 of the Fed's maturing holdings on the second day into ``add_on`` of auction
    face at 0.99, booked consistently in issuance, cash, the auction share and the bridge."""

    results_path = package / "exports_baseline" / "results_compact.csv"
    rows = _read_csv(results_path)
    row = rows[1]
    issued = float(row["NewDebtIssued"]) + add_on
    row["FedRolloverFace"] = str(add_on)
    row["FedRolloverProceeds"] = str(add_on * 0.99)
    row["NewDebtIssued"] = str(issued)
    row["AuctionProceeds"] = str(float(row["AuctionProceeds"]) + add_on * 0.99)
    row["CBOFedAuctionShare"] = str(add_on / issued)
    row["CBOFedMaturitiesAndRedemptions"] = "2.0"
    row["CBOFedAuctionRolloverAddons"] = str(add_on)
    row["CBOFedEndStock"] = str(float(row["CBOFedEndStock"]) - 2.0 + add_on)
    row["DebtHeld_CentralBank"] = row["CBOFedEndStock"]
    row["CBOFedNetStockChange"] = str(float(row["CBOFedEndStock"]) - float(row["CBOFedBeginStock"]))
    row["CBOFedHoldingsTargetError"] = str(float(row["CBOFedEndStock"]) - float(row["CBOFedHoldingsTarget"]))
    row["CBOFedGrossStockFlow"] = str(float(row["CBOFedGrossStockFlow"]) + 2.0 + add_on)
    _write_csv(results_path, list(rows[0]), rows)
    _rewrite_summary(results_path, package / "exports_baseline" / "summary.json")
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_summaries"]["baseline"] = _summary(_read_csv(results_path))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_hashes(package)


def test_cbo_package_verifier_accepts_a_fed_rollover_and_rejects_one_above_its_maturities(tmp_path: Path) -> None:
    """The Fed's add-ons are auction cash the Treasury receives without a public need, so
    they enter auction proceeds and the auction share but not the public pending balance;
    they may not exceed what the Fed had maturing."""

    package = _write_valid_package(tmp_path / "pkg")
    _add_fed_rollover(package, add_on=2.0)
    assert verify_package(package)["status"] == "pass"

    over = _write_valid_package(tmp_path / "over")
    _add_fed_rollover(over, add_on=3.0)
    with pytest.raises(ValueError, match="exceed its maturing holdings"):
        verify_package(over)


def test_cbo_package_verifier_rejects_remittance_overclaim_with_refreshed_hashes(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    _mutate_csv_cell(
        package / "exports_baseline" / "results_compact.csv",
        row_index=1,
        column="CB_Remittance",
        value="1.0",
    )
    _refresh_manifest_hashes(package)

    # The deferred asset forbids any remittance on the first day.
    with pytest.raises(ValueError, match="CB_Remittance differs from its rebuild"):
        verify_package(package)


def test_cbo_package_verifier_requires_explicit_remittance_scope(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest["claim_boundary"]
    invalid_boundaries = [
        ({**boundary, "remittance_role": "not_modeled_in_cbo_lane"}, "remittance_role"),
        ({
            **boundary,
            "excluded_claims": [claim for claim in boundary["excluded_claims"]
                                if claim != "does_not_model_fed_valuation_gains_and_losses"],
        }, "excluded_claims"),
    ]
    for invalid, field in invalid_boundaries:
        _write_json(manifest_path, {**manifest, "claim_boundary": invalid})
        _refresh_manifest_hashes(package)
        with pytest.raises(ValueError, match=rf"claim_boundary\.{field}"):
            verify_package(package)


def test_cbo_package_verifier_rejects_deprecated_net_interest_file_with_refreshed_hashes(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    (package / "tdcsim_modeled_net_interest_diagnostic.csv").write_text("overclaim\n", encoding="utf-8")
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="deprecated/overclaim"):
        verify_package(package)


def test_cbo_package_verifier_rejects_coordinated_debt_target_path_mutation(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    for run_dir in ("exports_baseline",):
        results_path = package / run_dir / "results_compact.csv"
        for column in ("CBOControlledDebtTarget", "CBOControlledDebtReference"):
            _mutate_csv_cell(results_path, row_index=RUN_ROWS - 1, column=column, value="999999.0")
        _rewrite_summary(results_path, package / run_dir / "summary.json")
    debt_path = package / "forecast_inputs" / "tdcsim_debt_stock_path.csv"
    _mutate_csv_cell(
        debt_path,
        row_index=RUN_ROWS - 1,
        column="cbo_federal_debt_held_public_target_bil",
        value="999999.0",
    )
    _mutate_csv_cell(
        debt_path,
        row_index=RUN_ROWS - 1,
        column="marketable_treasury_public_target_bil",
        value="999999.0",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="actual-day interpolated public target"):
        verify_package(package)


def test_cbo_package_verifier_rejects_overclaim_manifest_fields_and_unlisted_readme(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["modeled_net_interest_diagnostic"] = "README.md"
    _write_json(manifest_path, manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    (package / "README.md").write_text("claims budgetary net interest closure\n", encoding="utf-8")
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="forbidden overclaim field"):
        verify_package(package)


def test_cbo_package_verifier_rejects_unlisted_readme_even_when_hashed(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    (package / "README.md").write_text("claims budgetary net interest closure\n", encoding="utf-8")
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="artifact_hashes inventory changed"):
        verify_package(package)


def test_cbo_package_verifier_rejects_manifest_authorized_readme_output(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["readme"] = "README.md"
    _write_json(manifest_path, manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    (package / "README.md").write_text("claims budgetary net interest closure\n", encoding="utf-8")
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="artifact_hashes inventory changed|manifest.outputs must exactly match approved"):
        verify_package(package)


def test_cbo_package_verifier_rejects_rehashed_cbo_workbook_substitution(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    (package / "sources" / "cbo" / "51118-2026-02-Budget-Projections.xlsx").write_bytes(
        b"substituted workbook bytes"
    )
    _refresh_source_package_hashes(package)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="source_path must equal|frozen source file identity changed"):
        verify_package(package)


def test_cbo_package_verifier_rejects_fixture_workbook_substitution_escape_hatch(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    (package / "sources" / "cbo" / "51118-2026-02-Budget-Projections.xlsx").write_bytes(
        b"budget workbook bytes"
    )
    (package / "sources" / "cbo" / "51135-2026-02-Economic-Projections.xlsx").write_bytes(
        b"economic workbook bytes"
    )
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    for rel in (
        "cbo/51118-2026-02-Budget-Projections.xlsx",
        "cbo/51135-2026-02-Economic-Projections.xlsx",
    ):
        manifest["source_package"]["source_files"][rel]["source_path"] = f"fixture/{rel}"
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_source_package_hashes(package)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="source_path must equal|frozen source file identity changed"):
        verify_package(package)


def test_cbo_package_verifier_rejects_unknown_false_release_claim(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["release_claim"] = "This package proves complete CBO budgetary net-interest replication."
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="unknown top-level fields|forbidden claim wording"):
        verify_package(package)


def test_cbo_package_verifier_rejects_portable_z1_opening_overclaim(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["opening_portfolio_construction"][
        "claim_boundary"
    ] = "fully portable source-backed Z.1 holder constraints and exact holder reproduction"
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="forbidden claim wording|opening_portfolio_construction.claim_boundary"):
        verify_package(package)


def test_cbo_package_verifier_rejects_rehashed_fiscal_baseline_mutation(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    _mutate_csv_cell(
        package / "forecast_inputs" / "tdcsim_cbo_fiscal_baseline.csv",
        row_index=0,
        column="primary_deficit_bil",
        value="999999.0",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="tdcsim_cbo_fiscal_baseline.csv row 1 primary_deficit_bil"):
        verify_package(package)


def test_cbo_package_verifier_rejects_coherent_nominal_and_tips_yield_mutation(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    yield_path = package / "forecast_inputs" / "tdcsim_yield_curve_surface.csv"
    yield_rows = _read_csv(yield_path)
    for row in yield_rows:
        row["nominal_rate"] = str(float(row["nominal_rate"]) + 50.0)
        row["nominal_rate_decimal"] = str(float(row["nominal_rate_decimal"]) + 0.5)
    _write_csv(yield_path, list(yield_rows[0]), yield_rows)

    real_path = package / "forecast_inputs" / "tdcsim_tips_real_yield_path.csv"
    real_rows = _read_csv(real_path)
    for row in real_rows:
        row["nominal_rate_decimal"] = str(float(row["nominal_rate_decimal"]) + 0.5)
        row["real_yield_decimal"] = str(float(row["real_yield_decimal"]) + 0.5)
    _write_csv(real_path, list(real_rows[0]), real_rows)

    frn_path = package / "forecast_inputs" / "tdcsim_frn_rate_path.csv"
    frn_rows = _read_csv(frn_path)
    for row in frn_rows:
        row["auction_high_rate_decimal"] = str(float(row["auction_high_rate_decimal"]) + 0.5)
        row["benchmark_rate_decimal"] = str(float(row["benchmark_rate_decimal"]) + 0.5)
        row["money_market_yield_decimal"] = str(float(row["money_market_yield_decimal"]) + 0.5)
    _write_csv(frn_path, list(frn_rows[0]), frn_rows)
    _refresh_manifest_hashes(package)

    # The Fed income path's IORB spread is set against this curve, so it catches the shift too.
    with pytest.raises(
        ValueError,
        match="tdcsim_yield_curve_surface.csv row 1 nominal_rate|tdcsim_tips_real_yield_path.csv"
        "|tdcsim_fed_income_path.csv row 1 iorb_spread_pct",
    ):
        verify_package(package)


def test_cbo_package_verifier_rejects_rehashed_interior_debt_path_and_results_mutation(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    for run_dir in ("exports_baseline",):
        results_path = package / run_dir / "results_compact.csv"
        for column in ("CBOControlledDebtTarget", "CBOControlledDebtReference"):
            _mutate_csv_cell(results_path, row_index=100, column=column, value="999999.0")
        _rewrite_summary(results_path, package / run_dir / "summary.json")
    debt_path = package / "forecast_inputs" / "tdcsim_debt_stock_path.csv"
    _mutate_csv_cell(
        debt_path,
        row_index=100,
        column="cbo_federal_debt_held_public_target_bil",
        value="999999.0",
    )
    _mutate_csv_cell(
        debt_path,
        row_index=100,
        column="marketable_treasury_public_target_bil",
        value="999999.0",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="actual-day interpolated public target"):
        verify_package(package)


def test_cbo_package_verifier_rejects_rehashed_treasury_curve_mutation(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    curve_path = package / "sources" / "treasury" / "base_curve_2026-06-16.json"
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    curve["rows"][1]["nominal_rate"] = 9.99
    _write_json(curve_path, curve)
    _refresh_source_package_hashes(package)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="Treasury base curve rows changed|tdcsim_yield_curve_surface.csv"):
        verify_package(package)


def test_cbo_package_verifier_rejects_coherent_tips_only_source_divergence(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    cpi_path = package / "forecast_inputs" / "tdcsim_tips_cpi_path.csv"
    cpi_rows = _read_csv(cpi_path)
    for row in cpi_rows:
        row["cbo_cpi_u_index"] = str(float(row["cbo_cpi_u_index"]) * 1.01)
        row["tips_cpi_u_index"] = str(float(row["tips_cpi_u_index"]) * 1.01)
    _write_csv(cpi_path, list(cpi_rows[0]), cpi_rows)

    real_rows = build_tips_real_yield_path_rows(
        scenario_id="cbo_full_horizon_local_smoke_3m",
        yield_surface_rows=_read_csv(package / "forecast_inputs" / "tdcsim_yield_curve_surface.csv"),
        tips_cpi_path_rows=cpi_rows,
        observation_date=cpi_rows[0]["observation_date"],
        available_date=cpi_rows[0]["available_date"],
    )
    _write_csv(package / "forecast_inputs" / "tdcsim_tips_real_yield_path.csv", list(real_rows[0]), real_rows)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="tdcsim_tips_cpi_path.csv row 1 cbo_cpi_u_index"):
        verify_package(package)


def test_cbo_package_verifier_rejects_unknown_nested_claim_field(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["tips_forward_paths"]["release_claim"] = "This package proves observed future TIPS real yields."
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="tips_forward_paths contains unknown fields"):
        verify_package(package)


def test_cbo_package_verifier_rejects_nested_source_manifest_disagreement(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    nested_path = package / "sources" / "source_manifest.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    del nested["files"]["cbo/51135-2026-02-Economic-Projections.xlsx"]
    _write_json(nested_path, nested)

    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest_entry = manifest["source_package"]["source_files"]["source_manifest.json"]
    source_manifest_entry["sha256"] = hashlib.sha256(nested_path.read_bytes()).hexdigest()
    source_manifest_entry["bytes"] = nested_path.stat().st_size
    _write_json(manifest_path, manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="source_manifest.json files must exactly match"):
        verify_package(package)


def test_cbo_package_verifier_rejects_unknown_source_package_field(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_package"]["release_claim"] = "all source provenance is externally reproducible"
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="source_package contains unknown fields"):
        verify_package(package)


def test_cbo_package_verifier_rejects_unknown_source_manifest_field(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    source_manifest_path = package / "sources" / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["release_claim"] = "silent skips allowed"
    _write_json(source_manifest_path, source_manifest)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_package"]["source_files"]["source_manifest.json"]["sha256"] = hashlib.sha256(
        source_manifest_path.read_bytes()
    ).hexdigest()
    manifest["source_package"]["source_files"]["source_manifest.json"]["bytes"] = source_manifest_path.stat().st_size
    manifest["source_package"]["source_manifest_sha256"] = manifest["source_package"]["source_files"][
        "source_manifest.json"
    ]["sha256"]
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="sources/source_manifest.json contains unknown fields"):
        verify_package(package)


def test_cbo_package_verifier_rejects_source_manifest_sha_mismatch(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    manifest["source_package"]["source_manifest_sha256"] = "0" * 64
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="source_package.source_manifest_sha256 mismatch"):
        verify_package(package)


def test_cbo_package_verifier_rejects_source_entry_path_mismatch(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    rel = "fiscaldata/mts_table_1_deficit_2026-05-31.csv"
    manifest["source_package"]["source_files"][rel]["source_path"] = f"fixture/{rel}"
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="source_path must equal fiscaldata/mts_table_1_deficit_2026-05-31.csv"):
        verify_package(package)


def test_cbo_package_verifier_rejects_incorrect_mts_selector_and_frozen_row(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_row_selectors"]["mts_table_1"] = "classification_desc=Wrong Row"
    _write_json(manifest_path, manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="source_row_selectors.mts_table_1"):
        verify_package(package)


def test_cbo_package_verifier_recursively_rejects_absolute_local_paths(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    metadata_path = package / "forecast_inputs" / "tdcsim_opening_portfolio_metadata.json"
    leaked_path = "/" + "home" + "/alice/tdcsim/data/source.csv"
    metadata_path.write_text(
        json.dumps({"source_file": leaked_path}, indent=2) + "\n",
        encoding="utf-8",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="machine-local path leaked"):
        verify_package(package)


def _write_valid_package(package: Path) -> Path:
    package.mkdir(parents=True)
    lock_path = package / "requirements.lock.txt"
    lock_path.write_text("locked test dependencies\n", encoding="utf-8")
    runs = {"baseline": "exports_baseline"}

    _write_opening_files(package)
    _write_auxiliary_required_files(package)
    source_package = _write_source_package(package)
    opening_manifest = _roll_fixture_opening_book(package)
    remittance_by_end = _write_fed_remittance_inputs(package)
    fed_income = _write_fed_income_path(package)
    debt_rows = _read_csv(package / "forecast_inputs" / "tdcsim_debt_stock_path.csv")
    run_summaries = {}
    run_rows = {}
    opening_cb_holdings = opening_manifest["opening_cb_holdings_bil"]
    for run_id, run_dir in runs.items():
        rows = _result_rows(debt_rows, opening_cb_holdings=opening_cb_holdings)
        for row in rows:
            row["CBOTable43RemittanceInFiscalFlow"] = remittance_by_end.get(row["Date"], 0.0)
        _apply_fed_income(rows, fed_income)
        run_rows[run_id] = rows
        _write_csv(package / run_dir / "results_compact.csv", RESULT_COLUMNS, rows)
        summary = _summary(rows)
        run_summaries[run_id] = summary
        _write_json(package / run_dir / "summary.json", summary)

    _write_csv(package / "exports_baseline" / "final_portfolio_active.csv", ["col"], [{"col": "value"}])
    _write_csv(package / "exports_baseline" / "maturity_ledger.csv", ["col"], [{"col": "value"}])
    _write_net_interest_diagnostic(package, run_rows["baseline"])
    calibration = build_issuance_calibration(
        package / "sources" / "fiscaldata" / "auctions_query.csv",
        simulation_start_date="2026-06-21",
        source_relative_path="sources/fiscaldata/auctions_query.csv",
        opening_portfolio=pd.read_csv(package / "forecast_inputs" / "tdcsim_opening_portfolio.csv"),
        check_calendar=False,
    )
    _write_json(package / "forecast_inputs" / "tdcsim_issuance_calibration.json", calibration)

    manifest = {
        "scenario_id": "cbo_full_horizon_local_smoke_3m",
        "date_range": {
            "simulation_start_date": "2026-06-21",
            "simulation_end_date": "2036-09-30",
            "opening_close_date": OPENING_CLOSE_DATE,
            "actuals_available_as_of": ACTUALS_AVAILABLE_AS_OF,
            "fiscal_actuals_through": OPENING_CLOSE_DATE,
        },
        "opening_portfolio_construction": opening_manifest,
        "source_values": {
            "row_counts": {
                "period_rows": RUN_ROWS,
                "frn_rate_path_rows": RUN_ROWS,
                "tips_cpi_path_rows": len(_read_csv(package / "forecast_inputs" / "tdcsim_tips_cpi_path.csv")),
                "tips_real_yield_path_rows": len(
                    _read_csv(package / "forecast_inputs" / "tdcsim_tips_real_yield_path.csv")
                ),
            }
        },
        "tips_forward_paths": {
            "runtime_effect": "TIPS indexation and TIPS auction proceeds only; not a cash, issuance, or net-interest plug"
        },
        "frn_forward_rate_path": {
            "runtime_effect": "FRN benchmark accrual only; not a cash, issuance, or net-interest plug"
        },
        "fed_holdings_path": {
            "claim_boundary": "fed_holdings_path_guides_beneficial_holder_settlement_not_total_debt_or_tga",
            "mode": "cbo_fed_holdings_endpoints_linear_actual_days_holder_target",
            "runtime_columns": [
                "CBOFedHoldingsTarget",
                "CBOFedHoldingsTargetError",
                "CBOFedAuctionShare",
            ],
            "runtime_effect": "holder_allocation_only_total_issuance_unchanged",
        },
        "issuance_mix": {
            "mode": "bill_share_target_v1",
            "calibration_file": "forecast_inputs/tdcsim_issuance_calibration.json",
            "calibration_sha256": hashlib.sha256(
                (package / "forecast_inputs" / "tdcsim_issuance_calibration.json").read_bytes()
            ).hexdigest(),
            "bill_share_target": calibration["bill_share_target"],
        },
        "run_summaries": run_summaries,
        "source_package": source_package,
        "source_row_selectors": _source_row_selectors(),
        "claim_boundary": _claim_boundary(),
        "trust_anchor": _trust_anchor(lock_path),
        "outputs": {
            "baseline_summary": "exports_baseline/summary.json",
            "partial_interest_financing_diagnostic": "tdcsim_partial_interest_financing_diagnostic.csv",
        },
        "artifacts": {
            "output_dir": ".",
            "forecast_inputs_dir": "forecast_inputs",
            "manifest": "manifest.json",
        },
        "artifact_hashes": {},
    }
    _write_json(package / "manifest.json", manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)
    _refresh_manifest_hashes(package)
    return package


def _source_row_selectors() -> dict:
    return {
        "mts_table_1": "classification_desc=Year-to-Date;line_code_nbr=280;column=current_month_dfct_sur_amt",
        "mts_table_9": "classification_desc=Net Interest;column=current_fytd_rcpt_outly_amt",
        "debt_to_the_penny": "record_date=2026-05-29;column=debt_held_public_amt",
        "mspd": [
            "security_type_desc=Total Nonmarketable",
            "security_type_desc=Total Public Debt Outstanding",
        ],
        "dts": f"account_type={DTS_TGA_CLOSING_ACCOUNT}",
        "dts_june_fiscal_stub": dts_fiscal_selectors(),
    }


def _claim_boundary() -> dict:
    return {
        "schema_version": "tdcsim_cbo_package_claim_boundary_v1",
        "package_role": "cbo_baseline_forecast_evidence_package",
        "net_interest_role": "partial_interest_financing_diagnostic_only",
        "issuance_mix_role": "tdcsim_assumption_not_cbo_prescription",
        "cash_role": "operating_cash_path_sizes_issuance_debt_drift_reported_against_cbo_path",
        "remittance_role": CBO_FED_REMITTANCE_ROLE,
        "excluded_claims": [
            "does_not_claim_exact_historical_replay",
            "does_not_claim_full_soma_transaction_replay",
            "does_not_claim_budgetary_net_interest_replication",
            "does_not_claim_generated_evidence_tracked_in_git",
            "does_not_claim_receipts_outlays_decomposition",
            "does_not_use_cbo_net_interest_as_cash_or_issuance_plug",
            "does_not_claim_cbo_issuance_mix",
            *CBO_FED_REMITTANCE_EXCLUDED_CLAIMS,
        ],
    }


def _trust_anchor(lock_path: Path) -> dict:
    verifier_path = PROJECT_ROOT_FOR_IMPORT / "scripts" / "verify_cbo_forecast_package.py"
    return {
        "schema_version": "tdcsim_cbo_package_trust_anchor_v1",
        "release_commit_status": "pending_until_release_commit",
        "release_commit_sha": "",
        "package_zip_sha256_status": "pending_until_release_commit",
        "package_zip_sha256": "",
        "dirty_state": True,
        "python_version": platform.python_version(),
        "verifier_version": "tdcsim_cbo_package_verifier_semantic_v2",
        "verifier_sha256": hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
        "verifier_runtime": platform.python_implementation(),
        "dependency_lock_status": "requirements_lock_txt_present",
        "requirements_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    }


FIXTURE_PRESTART_ROLLFORWARD_ROWS = 35
FIXTURE_OBSERVED_CPI = {"2026-03-01": 330.213, "2026-04-01": 333.020}


def _write_source_opening_book(package: Path) -> None:
    """The packaged source book behind the fixture opening portfolio: every opening row,
    with the first 35 fixed rows standing in for securities that matured before the start
    and were reinvested in the fixture's one bill auction settling before the start."""

    rows = []
    counters: dict[str, int] = {}
    for index, row in enumerate(_read_csv(package / "forecast_inputs" / "tdcsim_opening_portfolio.csv")):
        security_type = row["SecurityType"]
        number = counters.get(security_type, 0)
        counters[security_type] = number + 1
        cusip = {"TIPS": f"TIPS{number % 53:03d}", "FRN": f"FRN{number % 8:03d}"}.get(security_type, f"FIX{number:04d}")
        source_row = {**row, "cusip_or_synthetic_id": cusip}
        if security_type == "Fixed" and number < FIXTURE_PRESTART_ROLLFORWARD_ROWS:
            source_row["MaturityDate"] = "2026-06-18"
        if security_type == "TIPS":
            source_row["FaceValue"] = source_row["AdjustedPrincipal"]
        if source_row["HolderType"] == "Private":
            source_row["HolderSubBucket"] = PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
        rows.append(source_row)
    _write_csv(package / "sources" / "opening" / "tdcsim_initial_portfolio_cohorts.csv", list(rows[0]), rows)


def _write_opening_files(package: Path) -> None:
    rows = []
    holders = ["Banks", "CB", "Foreign", "Private"]
    bond_id = 1
    for index in range(1985):
        rows.append(_opening_row(bond_id, "Fixed", holders[index % len(holders)], face=1.0))
        bond_id += 1
    for index in range(265):
        rows.append(
            _opening_row(
                bond_id,
                "TIPS",
                holders[index % len(holders)],
                face=0.0,
                original=1.5,
                adjusted=2.0,
                index_ratio=1.25,
            )
        )
        bond_id += 1
    for index in range(40):
        rows.append(
            _opening_row(
                bond_id,
                "FRN",
                holders[index % len(holders)],
                face=3.0,
                accrued_frn=0.1,
                benchmark_frn=0.036,
            )
        )
        bond_id += 1
    _write_csv(package / "forecast_inputs" / "tdcsim_opening_portfolio.csv", list(rows[0]), rows)

    tips_diag = [
        {
            "bond_id": str(1986 + index),
            "cusip": f"TIPS{index % 53:03d}",
            "holder_type": holders[index % len(holders)],
            "holder_subbucket": "",
            "maturity_date": "2030-01-15",
            "index_date": "2026-06-21",
            "source_original_issue_date": "2020-01-15",
            "source_ref_cpi": "332.08433",
            "source_index_ratio": "1.25",
            "adjusted_principal_bil": "2.0",
            "reconstructed_original_principal_bil": "1.5",
            "reconstructed_reference_cpi_issue": "265.667464",
            "source_status": "treasurydirect_tips_cpi_detail_index_date_join",
        }
        for index in range(265)
    ]
    _write_csv(package / "forecast_inputs" / "tdcsim_opening_tips_indexation_diagnostics.csv", list(tips_diag[0]), tips_diag)

    frn_diag = [
        {
            "bond_id": str(2251 + index),
            "cusip": f"FRN{index % 8:03d}",
            "holder_type": holders[index % len(holders)],
            "holder_subbucket": "",
            "opening_date": "2026-06-21",
            "source_record_date": "2026-06-17",
            "source_accrual_start": "2026-06-20",
            "source_accrual_end": "2026-06-21",
            "source_spread_percent": "0.1",
            "source_daily_index_percent": "3.6",
            "source_daily_interest_accrual_rate_percent": "3.7",
            "source_accrued_interest_per100": "0.1",
            "face_value_bil": "3.0",
            "accrued_interest_frn_bil": "0.1",
            "source_status": "fiscaldata_frn_daily_indexes_opening_accrual_join",
        }
        for index in range(40)
    ]
    _write_csv(package / "forecast_inputs" / "tdcsim_opening_frn_indexation_diagnostics.csv", list(frn_diag[0]), frn_diag)

    _write_json(package / "forecast_inputs" / "tdcsim_opening_portfolio_metadata.json", {"source_file": "sources/opening.csv"})


def _write_auxiliary_required_files(package: Path) -> None:
    _write_cbo_source_fixtures(package)
    _write_fiscal_baseline(package)
    _write_debt_stock_path(package)
    _write_yield_and_frn_rate_paths(package)
    for rel in REQUIRED_FILES:
        path = package / rel
        if path.exists() or rel in {"manifest.json", "forecast_inputs/source_contract_smoke.json"}:
            continue
        if path.suffix == ".csv":
            _write_csv(path, ["col"], [{"col": "value"}])
        else:
            _write_json(path, {})


def _write_fed_income_path(package: Path) -> list[dict]:
    rows = build_fed_income_path_rows(
        scenario_id="cbo_full_horizon_local_smoke_3m",
        periods=build_simulation_calendar("2026-06-21", "2036-09-30", "daily"),
        yield_rows=_read_csv(package / "forecast_inputs" / "tdcsim_yield_curve_surface.csv"),
    )
    _write_csv(package / "forecast_inputs" / FED_INCOME_PATH_FILE, list(rows[0]), rows)
    return rows


def _apply_fed_income(rows: list[dict], path: list[dict]) -> None:
    """Fill the Fed columns the way the engine books them, for a Fed with no Treasury income."""

    first = path[0]
    reserves = first["h41_reserve_balances_bil"] - (float(rows[0]["TGA"]) - first["h41_tga_bil"]) + (
        first["soma_mbs_start_bil"] - first["h41_soma_mbs_bil"]
    ) - (first["currency_start_bil"] - first["h41_currency_bil"])
    on_rrp = first["h41_on_rrp_bil"]
    due = first["h41_earnings_remittances_due_bil"]
    week = remittance_week(rows[0]["Date"])
    short = 3.35
    for index, row in enumerate(rows):
        row.update({column: 0.0 for column in FED_INCOME_RESULT_COLUMNS if column != "CBOTable43RemittanceInFiscalFlow"})
        row.update({"Reserves": 3_000.0, "CB_ShortRatePct": short, "CB_IORBRatePct": short + first["iorb_spread_pct"],
                    "CB_ONRRPRatePct": short + first["on_rrp_spread_pct"], "CB_ONRRPBalances": on_rrp})
        line = path[max(0, index - 1)]
        if index == 0:
            row.update({"CB_MBSHoldings": line["soma_mbs_start_bil"], "CB_Currency": line["currency_start_bil"],
                        "CB_NetIncome": 0.0, "CB_Remittance": 0.0})
        else:
            accrual = 1.0 / INTEREST_EXPENSE_DAY_COUNT
            iorb = row["CB_IORBRatePct"] / 100.0 * reserves * accrual
            rrp = row["CB_ONRRPRatePct"] / 100.0 * on_rrp * accrual
            net = line["soma_mbs_interest_bil"] + line["other_income_bil"] - iorb - rrp - line[
                "operating_expenses_bil"] - line["dividends_bil"]
            pay_day = is_fedwire_business_day(date.fromisoformat(row["Date"])) and remittance_week(row["Date"]) > week
            if pay_day:
                week = remittance_week(row["Date"])
            due, paid = settle_remittances(due, net, pay_day)
            reserves += non_treasury_reserve_effect(
                mbs_start=line["soma_mbs_start_bil"], mbs_end=line["soma_mbs_end_bil"],
                currency_start=line["currency_start_bil"], currency_end=line["currency_end_bil"],
                mbs_income=line["soma_mbs_interest_bil"], other_income=line["other_income_bil"],
                iorb_expense=iorb, on_rrp_expense=rrp, operating_expenses=line["operating_expenses_bil"],
                dividends=line["dividends_bil"],
            )
            row.update({
                "CB_MBSIncome": line["soma_mbs_interest_bil"], "CB_OtherIncome": line["other_income_bil"],
                "CB_OperatingExpenses": line["operating_expenses_bil"], "CB_Dividends": line["dividends_bil"],
                "CB_IORBExpense": iorb, "CB_ONRRPExpense": rrp, "CB_NetIncome": net, "CB_Remittance": paid,
                "CB_MBSHoldings": line["soma_mbs_end_bil"], "CB_Currency": line["currency_end_bil"],
            })
        row.update({
            "CB_ReserveBalances": reserves, "CB_EarningsRemittancesDue": due, "CB_DeferredAsset": max(0.0, -due),
            "CB_OtherLiabilitiesAndCapital": first["h41_other_liabilities_and_capital_bil"] + due
            - first["h41_earnings_remittances_due_bil"],
            "CBORemittanceCashEffect": row["CB_Remittance"],
        })


def _write_fed_remittance_inputs(package: Path) -> dict[str, float]:
    inputs = package / "forecast_inputs"
    splice, = _read_csv(inputs / "tdcsim_current_fy_splice.csv")
    periods = build_simulation_calendar("2026-06-21", "2036-09-30", "daily")
    primary = build_primary_deficit_path_rows(
        scenario_id="cbo_full_horizon_local_smoke_3m",
        periods=periods,
        primary_deficit_by_fiscal_year_bil={year: 0.5 for year in range(2026, 2037)},
    )
    remittance = build_fed_remittance_path_rows(
        primary, splice, read_dts_fed_earnings(package / "sources" / DTS_FLOWS_SOURCE, splice),
    )
    _write_csv(inputs / "tdcsim_primary_deficit_path.csv", list(primary[0]), primary)
    _write_csv(inputs / FED_REMITTANCE_PATH_FILE, list(remittance[0]), remittance)
    return {row["period_end"]: row["remittance_bil"] for row in remittance}


def _write_cbo_source_fixtures(package: Path) -> None:
    if not BUDGET_WORKBOOK.exists() or not ECONOMIC_WORKBOOK.exists():
        raise AssertionError("CBO package-verifier integration workbooks are missing")
    budget_rows = parse_cbo_budget_source_contract(BUDGET_WORKBOOK).fixture_rows()
    economic_rows = parse_cbo_economic_quarterly_source_contract(ECONOMIC_WORKBOOK).fixture_rows()
    rows = budget_rows + economic_rows
    _write_csv(package / "forecast_inputs" / "source_fixtures.csv", list(rows[0]), rows)


def _write_fiscal_baseline(package: Path) -> None:
    rows = build_cbo_fiscal_baseline_rows(
        parse_cbo_budget_source_contract(BUDGET_WORKBOOK),
        scenario_id="cbo_full_horizon_local_smoke_3m",
    )
    _write_csv(package / "forecast_inputs" / "tdcsim_cbo_fiscal_baseline.csv", list(rows[0]), rows)


def _write_debt_stock_path(package: Path) -> None:
    rows = []
    start = date(2026, 6, 21)
    fiscal_rows = _read_csv(package / "forecast_inputs" / "tdcsim_cbo_fiscal_baseline.csv")
    anchors = [(start, 1_000.0)]
    for fiscal_row in fiscal_rows:
        fiscal_year = int(float(fiscal_row["fiscal_year"]))
        if 2026 <= fiscal_year <= 2036:
            anchors.append((date(fiscal_year, 9, 30), float(fiscal_row["debt_held_public_end_bil"])))
    anchors = sorted(dict(anchors).items())
    for index in range(RUN_ROWS):
        current = start + timedelta(days=index)
        target = _interpolated_anchor_value(anchors, current)
        anchor_type = "interpolated_assumption"
        source_role = "scenario_assumption"
        if index == 0:
            anchor_type = "actual_as_of"
            source_role = "hard_actual_state"
        elif current.month == 9 and current.day == 30:
            anchor_type = "cbo_fiscal_year_end"
            source_role = "official_hard_anchor"
        rows.append(
            {
                "schema_version": "tdcsim_debt_stock_path_v1",
                "scenario_id": "cbo_full_horizon_local_smoke_3m",
                "period_end": current.isoformat(),
                "cbo_federal_debt_held_public_target_bil": str(target),
                "public_nonmarketable_treasury_bil": "0.0",
                "non_treasury_and_definition_residual_bil": "0.0",
                "marketable_treasury_public_target_bil": str(target),
                "bridge_source": "debt_to_penny_and_mspd_actual_bridge",
                "bridge_method": "latest_actual_constant_nominal_by_component",
                "anchor_type": anchor_type,
                "interpolation_method": "linear_actual_days",
                "source_role": source_role,
                "runtime_role": "hard_target",
                "source_fiscal_year": str(current.year + int(current.month >= 10)),
                "observation_date": "2026-05-31",
                "available_date": "2026-06-17",
                "source_status": "cbo_public_debt_target_bridged_to_marketable_treasury_public_target",
                "claim_boundary": "bridge_components_not_auction_supply",
            }
        )
    _write_csv(package / "forecast_inputs" / "tdcsim_debt_stock_path.csv", list(rows[0]), rows)


def _interpolated_anchor_value(anchors: list[tuple[date, float]], current: date) -> float:
    for (start_date, start_value), (end_date, end_value) in zip(anchors, anchors[1:]):
        if start_date <= current <= end_date:
            span_days = (end_date - start_date).days
            elapsed_days = (current - start_date).days
            return start_value + (end_value - start_value) * (elapsed_days / span_days)
    return anchors[-1][1]


def _base_curve_rows() -> list[dict[str, object]]:
    payload = _base_curve_payload()
    rows_without_hash = [
        {
            "tenor_years": float(row["tenor_years"]),
            "nominal_rate": float(row["nominal_rate"]),
        }
        for row in payload["rows"]
    ]
    source_hash = hashlib.sha256(_json_bytes(payload)).hexdigest()
    return [
        {
            "base_curve_date": "2026-06-16",
            "available_date": "2026-06-16",
            "base_curve_source_key": "official_treasury_daily_nominal_par_curve_2026_06_16",
            "base_curve_sha256": source_hash,
            **row,
        }
        for row in rows_without_hash
    ]


def _base_curve_payload() -> dict[str, object]:
    rows_without_hash = [
        {"tenor_years": 1.0 / 12.0, "nominal_rate": 3.67},
        {"tenor_years": 0.25, "nominal_rate": 3.79},
        {"tenor_years": 0.5, "nominal_rate": 3.81},
        {"tenor_years": 1.0, "nominal_rate": 3.84},
        {"tenor_years": 2.0, "nominal_rate": 4.05},
        {"tenor_years": 3.0, "nominal_rate": 4.08},
        {"tenor_years": 5.0, "nominal_rate": 4.16},
        {"tenor_years": 7.0, "nominal_rate": 4.28},
        {"tenor_years": 10.0, "nominal_rate": 4.43},
        {"tenor_years": 20.0, "nominal_rate": 4.92},
        {"tenor_years": 30.0, "nominal_rate": 4.93},
    ]
    return {
        "rows": [
            {"base_curve_date": "2026-06-16", **row}
            for row in rows_without_hash
        ],
    }


def _write_yield_and_frn_rate_paths(package: Path) -> None:
    start = date(2026, 6, 21)
    macro_rows = build_cbo_macro_forecast_path_rows(
        parse_cbo_economic_quarterly_source_contract(ECONOMIC_WORKBOOK),
        scenario_id="cbo_full_horizon_local_smoke_3m",
    )
    _write_csv(package / "forecast_inputs" / "tdcsim_macro_forecast_path.csv", list(macro_rows[0]), macro_rows)
    base_curve_rows = _base_curve_rows()
    yield_rows = build_yield_curve_surface_rows(
        macro_forecast_rows=macro_rows,
        base_curve_rows=base_curve_rows,
        actuals_available_as_of=ACTUALS_AVAILABLE_AS_OF,
        output_tenors=(0.25, 10.0),
    )
    frn_rows = []
    real_yield_rows = []
    three_month_rates = sorted(
        (str(row["curve_date"]), float(row["nominal_rate_decimal"]))
        for row in yield_rows
        if abs(float(row["tenor_years"]) - 0.25) <= 1e-9
    )
    for index in range(RUN_ROWS):
        current = start + timedelta(days=index)
        previous = current - timedelta(days=1)
        source_date, rate_decimal = [
            (date_value, rate)
            for date_value, rate in three_month_rates
            if date_value <= current.isoformat()
        ][-1]
        frn_rows.append(
            {
                "schema_version": "tdcsim_frn_rate_path_v1",
                "scenario_id": "cbo_full_horizon_local_smoke_3m",
                "period_start": previous.isoformat(),
                "period_end": current.isoformat(),
                "rate_effective_start": previous.isoformat(),
                "rate_effective_end": current.isoformat(),
                "benchmark_tenor_years": "0.25",
                "auction_high_rate_decimal": str(rate_decimal),
                "benchmark_rate_decimal": str(rate_decimal),
                "money_market_yield_decimal": str(rate_decimal),
                "spread_treatment": "add_security_fixed_spread_decimal",
                "day_count_basis": "360.0",
                "lockout_business_days": "2.0",
                "rate_source_family": "cbo_yield_surface_3m_tbill_anchor",
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "observation_date": "2026-02-11",
                "available_date": "2026-02-11",
                "source_status": f"fixture_forward_13week_frn_high_rate_assumption_latest_curve_date_{source_date}",
                "claim_boundary": "future_frn_resets_are_explicit_forecast_assumptions",
            }
        )
    tips_cpi_rows = build_monthly_tips_cpi_path_rows(
        scenario_id="cbo_full_horizon_local_smoke_3m",
        macro_forecast_rows=macro_rows,
        simulation_start_date="2026-06-21",
        simulation_end_date="2036-09-30",
        opening_reference_cpi=331.80363,
        reference_cpi_anchor_date=OPENING_CLOSE_DATE,
        observed_cpi_by_month=FIXTURE_OBSERVED_CPI,
        reference_lag_months=3,
        pricing_horizon_end_date="2046-10-01",
        observation_date=OPENING_CLOSE_DATE,
        available_date=ACTUALS_AVAILABLE_AS_OF,
    )
    real_yield_rows = build_tips_real_yield_path_rows(
        scenario_id="cbo_full_horizon_local_smoke_3m",
        yield_surface_rows=yield_rows,
        tips_cpi_path_rows=tips_cpi_rows,
        observation_date=OPENING_CLOSE_DATE,
        available_date=ACTUALS_AVAILABLE_AS_OF,
    )
    _write_csv(package / "forecast_inputs" / "tdcsim_yield_curve_surface.csv", list(yield_rows[0]), yield_rows)
    _write_csv(package / "forecast_inputs" / "tdcsim_frn_rate_path.csv", list(frn_rows[0]), frn_rows)
    _write_csv(package / "forecast_inputs" / "tdcsim_tips_cpi_path.csv", list(tips_cpi_rows[0]), tips_cpi_rows)
    _write_csv(package / "forecast_inputs" / "tdcsim_tips_real_yield_path.csv", list(real_yield_rows[0]), real_yield_rows)


def _write_source_package(package: Path) -> dict:
    files = {
        "cbo/51118-2026-02-Budget-Projections.xlsx": BUDGET_WORKBOOK,
        "cbo/51135-2026-02-Economic-Projections.xlsx": ECONOMIC_WORKBOOK,
    }
    source_entries = {}
    for rel, source in files.items():
        if not source.exists():
            raise AssertionError("CBO package-verifier integration workbooks are missing")
        path = package / "sources" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        source_entries[rel] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "source_path": rel,
        }
    _write_csv(
        package / "sources" / "fiscaldata" / "mts_table_1_deficit_2026-05-31.csv",
        ["record_date", "classification_desc", "line_code_nbr", "current_month_dfct_sur_amt"],
        [
            {
                "record_date": "2026-05-31",
                "classification_desc": "Year-to-Date",
                "line_code_nbr": "280",
                "current_month_dfct_sur_amt": "1246203266386.93",
            }
        ],
    )
    _write_csv(
        package / "sources" / "fiscaldata" / "mts_table_9_net_interest_2026-05-31.csv",
        ["record_date", "classification_desc", "current_fytd_rcpt_outly_amt", "line_code_nbr"],
        [
            {
                "record_date": "2026-05-31",
                "classification_desc": "Net Interest",
                "current_fytd_rcpt_outly_amt": "722706511243.20",
                "line_code_nbr": "320",
            }
        ],
    )
    _write_csv(
        package / "sources" / DTS_CASH_SOURCE, ["record_date", "account_type", "open_today_bal"],
        [
            {"record_date": "2026-06-01", "account_type": DTS_TGA_OPENING_ACCOUNT, "open_today_bal": "980113"},
            {"record_date": OPENING_CLOSE_DATE, "account_type": DTS_TGA_CLOSING_ACCOUNT, "open_today_bal": "981113"},
        ],
    )
    _write_csv(
        package / "sources" / DTS_FLOWS_SOURCE,
        ["record_date", "table_nbr", "account_type", "transaction_type", "transaction_catg", "transaction_mtd_amt",
         "transaction_fytd_amt"],
        [
            *(
                {"record_date": OPENING_CLOSE_DATE, "table_nbr": "II", "account_type": account,
                 "transaction_type": kind, "transaction_catg": category, "transaction_mtd_amt": str(amount),
                 "transaction_fytd_amt": ""}
                for (kind, account, category), amount in zip(
                    DTS_FISCAL_LINES.values(), (50000, 49000, 25000, 21000, 2000)
                )
            ),
            {"record_date": OPENING_CLOSE_DATE, "table_nbr": "II", "account_type": "Treasury General Account (TGA)",
             "transaction_type": "Deposits", "transaction_catg": "Federal Reserve Earnings",
             "transaction_mtd_amt": "997", "transaction_fytd_amt": "6559"},
        ],
    )
    _write_csv(
        package / "sources" / "fiscaldata" / "mspd_table_1_debt_bridge_2026-05-31.csv",
        ["record_date", "security_type_desc", "debt_held_public_mil_amt"],
        [
            {
                "record_date": "2026-05-31",
                "security_type_desc": "Total Nonmarketable",
                "debt_held_public_mil_amt": "620821.77409502",
            },
            {
                "record_date": "2026-05-31",
                "security_type_desc": "Total Public Debt Outstanding",
                "debt_held_public_mil_amt": "31515369.91844520",
            },
        ],
    )
    _write_csv(
        package / "sources" / "fiscaldata" / "mspd_table_3_market_2026-05-31.csv",
        ["record_date", "security_type_desc", "security_class1_desc", "security_class2_desc",
         "issued_amt", "redeemed_amt", "outstanding_amt"],
        [
            {"record_date": "2026-05-31", "security_type_desc": "Marketable", "outstanding_amt": str(i)}
            for i in range(2000)
        ] + [
            {"record_date": "2026-05-31", "security_type_desc": "Marketable",
             "security_class1_desc": "Inflation-Protected Securities", "security_class2_desc": f"TIPS{i:03d}",
             "issued_amt": "1600", "redeemed_amt": "-100", "outstanding_amt": "2000"}
            for i in range(53)
        ],
    )
    _write_csv(
        package / "sources" / "treasury" / "debt_to_penny_public_debt_2026-05-29.csv",
        ["record_date", "debt_held_public_amt"],
        [{"record_date": "2026-05-29", "debt_held_public_amt": "31515369798622.98"}],
    )
    _write_source_opening_book(package)
    source_book = _read_csv(package / "sources" / "opening" / "tdcsim_initial_portfolio_cohorts.csv")
    fed_by_cusip = {}
    for row in source_book:
        if row["HolderType"] == "CB" and row["MaturityDate"] > OPENING_CLOSE_DATE:
            cusip = row["cusip_or_synthetic_id"]
            prior, _, _ = fed_by_cusip.get(cusip, (0, "", ""))
            fed_by_cusip[cusip] = (prior + float(row["FaceValue"]), row["SecurityType"], row["MaturityDate"])
    _write_csv(
        package / "sources" / SOMA_SOURCE,
        ["As Of Date", "CUSIP", "Security Type", "Maturity Date", "Par Value", "Inflation Compensation"],
        [
            {"As Of Date": SOMA_SNAPSHOT_DATE, "CUSIP": cusip, "Security Type": kind, "Maturity Date": maturity,
             "Par Value": str(amount * (0.75 if kind == "TIPS" else 1) * 1e9),
             "Inflation Compensation": str(amount * 0.25 * 1e9) if kind == "TIPS" else "0"}
            for cusip, (amount, kind, maturity) in fed_by_cusip.items()
        ],
    )
    shares = {
        "Banks": 0.25, "Foreign": 0.25, f"Private/{PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}": 0.5,
        f"Private/{PRIVATE_SUBBUCKET_MMF}": 0.0,
    }
    _write_json(
        package / "sources" / "opening" / "tdcsim_ratewall_input_manifest.json",
        {"artifacts": {"initial_portfolio": {"noncb_weights": {"bills": shares, "nonbills": shares, "frn_mmf_auction_share": 0.7}}}},
    )
    _write_csv(
        package / "sources" / "fiscaldata" / "tips_cpi_data_detail.csv",
        ["cusip", "original_issue_date", "index_date", "ref_cpi", "index_ratio"],
        [
            {"cusip": f"TIPS{index:03d}", "original_issue_date": "2026-01-01", "index_date": day,
             "ref_cpi": reference, "index_ratio": str(4 / 3)}
            for index in range(53)
            for day, reference in (
                ("2026-05-29", 329.88126), (OPENING_CLOSE_DATE, 331.80363), ("2026-06-21", 332.08433),
                ("2026-06-01", FIXTURE_OBSERVED_CPI["2026-03-01"]),
                ("2026-07-01", FIXTURE_OBSERVED_CPI["2026-04-01"]),
            )
        ],
    )
    _write_csv(
        package / "sources" / "fiscaldata" / "frn_daily_indexes.csv",
        ["cusip", "record_date", "daily_index_percent"],
        [{"cusip": "fixture", "record_date": "2026-06-17", "daily_index_percent": "3.6"}],
    )
    _write_json(
        package / "sources" / "treasury" / "base_curve_2026-06-16.json",
        _base_curve_payload(),
    )
    # The pre-start rollforward also reads each auction's CUSIP, original issue date, price
    # and rates; the fixture's one bill settling before the start takes the matured rows.
    rollforward_fields = {"original_issue_date": "null", "int_rate": "null", "price_per100": "99.0",
                          "high_investment_rate": "4.0", "high_yield": "null"}
    auction_rows = [
        {**row, "record_date": OPENING_CLOSE_DATE, "issue_date": OPENING_CLOSE_DATE, "maturity_date": "2026-09-17"}
        if row["issue_date"] == "2026-06-16" else row
        for row in _fixture_auctions()
    ]
    _write_auctions(
        package / "sources" / "fiscaldata" / "auctions_query.csv",
        [{"cusip": f"AUC{index:03d}", **rollforward_fields, **row} for index, row in enumerate(auction_rows)],
        columns=["record_date", *AUCTION_REQUIRED_COLUMNS, "cusip", *rollforward_fields],
    )
    _write_json(package / "sources" / CBO_REMITTANCE_SOURCE_FILE, cbo_remittance_source_fixture())
    fed_sources = fed_income_source_fixture()
    for document in fed_sources["documents"].values():
        target = package / "sources" / document["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((FED_INCOME_SOURCES / Path(document["file"]).name).read_bytes())
    _write_json(package / "sources" / FED_INCOME_SOURCE_FILE, fed_sources)
    for rel in (
        "opening/tdcsim_initial_portfolio_cohorts.csv",
        "opening/tdcsim_ratewall_input_manifest.json",
        "fiscaldata/tips_cpi_data_detail.csv",
        "fiscaldata/frn_daily_indexes.csv",
        "fiscaldata/auctions_query.csv",
        "fiscaldata/mts_table_1_deficit_2026-05-31.csv",
        "fiscaldata/mts_table_9_net_interest_2026-05-31.csv",
        DTS_CASH_SOURCE,
        DTS_FLOWS_SOURCE,
        SOMA_SOURCE,
        "fiscaldata/mspd_table_1_debt_bridge_2026-05-31.csv",
        "fiscaldata/mspd_table_3_market_2026-05-31.csv",
        "treasury/debt_to_penny_public_debt_2026-05-29.csv",
        "treasury/base_curve_2026-06-16.json",
        CBO_REMITTANCE_SOURCE_FILE,
        FED_INCOME_SOURCE_FILE,
        *(document["file"] for document in fed_sources["documents"].values()),
    ):
        path = package / "sources" / rel
        source_entries[rel] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "source_path": rel,
        }
    source_manifest = {
        "schema_version": "tdcsim_cbo_source_package_manifest_v1",
        "status": "package_relative_sources_for_cbo_forecast_smoke",
        "inventory_status": "complete_required_inventory_no_silent_skips",
        "files": source_entries,
    }
    manifest_bytes = json.dumps(source_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    manifest_path = package / "sources" / "source_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    source_entries = {
        **source_entries,
        "source_manifest.json": {
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "bytes": len(manifest_bytes),
            "source_path": "source_manifest.json",
            "origin_source_label": "package_generated_source_manifest",
        },
    }
    return {
        "sources_dir": "sources",
        "source_manifest": "sources/source_manifest.json",
        "source_manifest_sha256": source_entries["source_manifest.json"]["sha256"],
        "source_file_count": len(source_entries),
        "source_files": source_entries,
    }


def _frozen_treasury_observations() -> dict:
    return {
        "mts_table_1_deficit_row": {
            "record_date": "2026-05-31",
            "classification_desc": "Federal Surplus or Deficit",
            "current_fytd_dfct_sur_amt": "1246203266386.93",
        },
        "mts_table_9_net_interest_row": {
            "record_date": "2026-05-31",
            "classification_desc": "Net Interest",
            "current_fytd_rcpt_outly_amt": "722706511243.20",
        },
        "debt_to_penny_row": {
            "record_date": "2026-05-29",
            "debt_held_public_amt": "31515369798622.98",
        },
        "mspd_rows": [
            {
                "record_date": "2026-05-31",
                "security_type_desc": "Total Nonmarketable",
                "debt_held_public_mil_amt": "620821.77409502",
            },
            {
                "record_date": "2026-05-31",
                "security_type_desc": "Total Public Debt Outstanding",
                "debt_held_public_mil_amt": "30999912.58425522",
            },
        ],
        "source_status": "fixture_frozen_observation_extract",
    }


OPERATING_CASH_TARGET = 981.113


def _fiscal_year_end_dates() -> set[str]:
    days = _fiscal_year_close_dates({"settlement_calendar": SETTLEMENT_CALENDAR}, "2026-06-21", "2036-09-30")
    return {day.date().isoformat() for day in days}


def _result_rows(debt_rows: list[dict[str, str]], *, opening_cb_holdings: float) -> list[dict[str, str]]:
    """An operating-cash pass: one unit of need a day, settling with three units pending
    except at fiscal-year ends, and debt five below CBO's path."""

    start = date(2026, 6, 21)
    fiscal_year_ends = _fiscal_year_end_dates()
    previous_pending = 0.0
    rows = []
    for index in range(RUN_ROWS):
        current = start + timedelta(days=index)
        debt = float(debt_rows[index]["marketable_treasury_public_target_bil"])
        # The opening row is the engine's nonsettling identity: the target equals the
        # opening book's Fed holdings and nothing moves.
        fed_target = opening_cb_holdings + index
        fed_begin = 0.0 if index == 0 else opening_cb_holdings + index - 1
        fed_end = 0.0 if index == 0 else fed_target
        fed_purchase = fed_end - fed_begin
        residual = 0.0
        pending = 0.0 if index == 0 or current.isoformat() in fiscal_year_ends else 3.0
        # One unit settles a day; the need refills pending after a close and the close
        # settles what was pending, so pending(t) = pending(t-1) + need - settled.
        settled = 0.0 if index == 0 else 1.0 + max(0.0, previous_pending - pending)
        need = 0.0 if index == 0 else settled + pending - previous_pending
        previous_pending = pending
        tga = OPERATING_CASH_TARGET - pending
        scenario_debt = debt - 5.0
        row = {
            "Date": current.isoformat(),
            "TotalDebt_Agg": scenario_debt,
            "CBOControlledDebtTarget": debt,
            "CBOControlledDebtReference": debt,
            "ScenarioControlledDebt": scenario_debt,
            "DebtDriftFromReference": scenario_debt - debt,
            "CBOControlledDebtTargetApplicable": 0.0,
            "CBOControlledDebtPostIssuance": scenario_debt,
            "CBOControlledDebtTargetError": 0.0,
            "CBORequiredFaceIssuance": 0.0,
            "IssuanceNeedProceeds": need,
            "PendingIssuanceProceeds": pending,
            "EarlySettlementProceeds": 0.0,
            "CashManagementBillProceeds": 0.0,
            "FedRolloverFace": 0.0,
            "FedRolloverProceeds": 0.0,
            "FedRolloverCashSizingDifference": 0.0,
            "FedRedeemedFace": 0.0,
            "CashFinancingProceeds": settled,
            "NewDebtIssued": 0.0 if index == 0 else 1.0,
            "AuctionProceeds": settled,
            "IssueDiscountCost_Period": 0.0 if index == 0 else 0.01,
            "InterestPaid_Bonds": 0.0 if index == 0 else 0.2,
            "NonMarketableInterestCapitalized_Period": 0.0,
            "PrimaryDeficit": 0.0 if index == 0 else 0.5,
            "TGA": tga,
            "CBOOperatingCashTarget": OPERATING_CASH_TARGET,
            "CBOCashResidual": tga - OPERATING_CASH_TARGET,
            "CBOCashReconciliationResidual": residual,
            "CBORemittanceCashEffect": 0.0,
            "CBOFedHoldingsTarget": fed_target,
            "CBOFedHoldingsTargetApplicable": 1.0,
            "CBOFedHoldingsTargetError": 0.0,
            "CBOFedAuctionShare": 0.0,
            "CBOFedSecondaryPurchaseFace": 0.0 if index == 0 else fed_purchase,
            "CBOFedSecondaryPurchaseCash": 0.0 if index == 0 else fed_purchase * 0.99,
            "CBOFedSecondaryPurchaseReserveEffect": 0.0 if index == 0 else fed_purchase * 0.99,
            "CBOFedSecondaryPurchaseDepositEffect": 0.0 if index == 0 else fed_purchase * 0.80,
            "CBOFedSecondarySaleCash": 0.0,
            "CBOFedSecondarySaleReserveEffect": 0.0,
            "CBOFedSecondarySaleDepositEffect": 0.0,
            "CBOFedPrivateMaturityTDC": 0.0,
            "CBOFedAcquisitionChannel": (
                "synthetic_secondary_purchase_to_hit_cbo_stock_path" if index > 0 else ""
            ),
            "CBOFedSecondarySaleBuyerMix": "",
            "CBOFedBeginStock": fed_begin,
            "CBOFedMaturitiesAndRedemptions": 0.0,
            "CBOFedTipsPrincipalIndexation": 0.0,
            "CBOFedAuctionRolloverAddons": 0.0,
            "CBOFedSyntheticSecondaryPurchases": 0.0 if index == 0 else fed_purchase,
            "CBOFedSyntheticSecondarySales": 0.0,
            "CBOFedEndStock": fed_end,
            "CBOFedGrossStockFlow": 0.0 if index == 0 else abs(fed_purchase),
            "CBOFedNetStockChange": fed_end - fed_begin,
            "CBOFedStockMode": (
                "synthetic_cb_treasury_stock_target_beneficial_holder" if index > 0 else ""
            ),
            "CBOFedSettlementScope": (
                "beneficial_holder_with_model_dirty_value_settlement"
                if index > 0
                else "opening_stock_target_match_nonsettling"
            ),
            "CBORemittanceStatus": CBO_FED_REMITTANCE_STATUS if index > 0 else CBO_FED_REMITTANCE_OPENING_STATUS,
            "CB_TreasuryInterestCashReceived": 0.0,
            "CB_NetIncome": "",
            "CB_Remittance": 0.0,
            "CB_DeferredAsset": "",
            "CBONetInterestDiagnostic": 0.0,
            "CBOTotalDeficitDiagnostic": 0.0,
            "NetInterestDiagnosticStatus": "cbo_reported_check_only",
            "DebtHeld_Banks": 10.0 + index,
            "DebtHeld_CentralBank": fed_target,
            "DebtHeld_Foreign": 30.0 + index,
            "DebtHeld_DomesticNonBanks": 40.0 + index,
            "DebtHeldByType_Fixed": 50.0 + index,
            "DebtHeldByType_TIPS": 60.0 + index,
            "DebtHeldByType_FRN": 70.0 + index,
            "TIPSInflationAccretion_Period": 0.0 if index == 0 else 0.05,
            "TIPSInflationAccretion_Cumulative": float(index),
            "CBOBuybackFaceRetired": 0.0,
            "CBOBuybackCashPaid": 0.0,
        }
        rows.append({key: str(value) for key, value in row.items()})
    assert rows[-1]["Date"] == "2036-09-30"
    return rows


def _summary(rows: list[dict[str, str]]) -> dict:
    final = rows[-1]
    residuals = [float(row["CBOCashReconciliationResidual"]) for row in rows]
    return {
        "scenario_id": "cbo_full_horizon_local_smoke_3m",
        "rows": len(rows),
        "start_date": rows[0]["Date"],
        "final_date": final["Date"],
        "debt_drift_from_cbo_reference_bil": {
            "final": float(final["DebtDriftFromReference"]),
            "min": min(float(row["DebtDriftFromReference"]) for row in rows[1:]),
            "max": max(float(row["DebtDriftFromReference"]) for row in rows[1:]),
            "fiscal_year_end": {
                str(int(row["Date"][:4])): float(row["DebtDriftFromReference"])
                for row in rows
                if row["Date"] in _fiscal_year_end_dates()
            },
        },
        "tga_minus_operating_cash_target_at_fiscal_year_end_bil": {
            str(int(row["Date"][:4])): float(row["TGA"]) - float(row["CBOOperatingCashTarget"])
            for row in rows
            if row["Date"] in _fiscal_year_end_dates()
        },
        "max_pending_issuance_bil": max(float(row["PendingIssuanceProceeds"]) for row in rows),
        "early_settlement_days": sum(1 for row in rows if float(row["EarlySettlementProceeds"]) > 1e-9),
        "max_abs_cbo_fed_holdings_error_bil": max(abs(float(row["CBOFedHoldingsTargetError"])) for row in rows),
        "final_cbo_fed_holdings_error_bil": float(final["CBOFedHoldingsTargetError"]),
        "final_cbo_fed_holdings_target_bil": float(final["CBOFedHoldingsTarget"]),
        "final_cbo_reference_bil": float(final["CBOControlledDebtReference"]),
        "final_total_debt_bil": float(final["TotalDebt_Agg"]),
        "final_tga_bil": float(final["TGA"]),
        "final_operating_cash_target_bil": float(final["CBOOperatingCashTarget"]),
        "final_cash_residual_bil": float(final["CBOCashResidual"]),
        "net_cash_reconciliation_residual_bil": sum(residuals),
        "gross_abs_cash_reconciliation_residual_bil": sum(abs(value) for value in residuals),
        "positive_cash_reconciliation_residual_bil": sum(value for value in residuals if value > 0.0),
        "negative_cash_reconciliation_residual_bil": sum(value for value in residuals if value < 0.0),
        "nonzero_cash_reconciliation_residual_days": sum(1 for value in residuals if abs(value) > 1e-12),
        "total_face_issued_bil": sum(float(row["NewDebtIssued"]) for row in rows),
        "total_auction_proceeds_bil": sum(float(row["AuctionProceeds"]) for row in rows),
        "total_issue_discount_cost_bil": sum(float(row["IssueDiscountCost_Period"]) for row in rows),
        "total_primary_deficit_bil": sum(float(row["PrimaryDeficit"]) for row in rows),
        "total_cbo_buyback_face_retired_bil": sum(float(row["CBOBuybackFaceRetired"]) for row in rows),
        "tips_inflation_accretion_cumulative_bil": float(final["TIPSInflationAccretion_Cumulative"]),
    }


def _opening_row(
    bond_id: int,
    security_type: str,
    holder: str,
    *,
    face: float,
    original: float = 0.0,
    adjusted: float = 0.0,
    index_ratio: float = 0.0,
    accrued_frn: float = 0.0,
    benchmark_frn: float = 0.0,
) -> dict[str, str]:
    return {
        "BondID": str(bond_id),
        "SecurityType": security_type,
        "IssueDate": "2026-01-01",
        "MaturityDate": "2030-01-01",
        "OriginalMaturityYears": "1.0",
        "FaceValue": str(face),
        "CouponRate": "0.0",
        "HolderType": holder,
        "HolderSubBucket": "",
        "Status": "Active",
        "MaturityCategory": "bills",
        "OriginalPrincipal": str(original),
        "AdjustedPrincipal": str(adjusted),
        "ReferenceCPI_Issue": "0.0",
        "IndexRatio": str(index_ratio),
        "FixedSpread": "0.0",
        "AccruedInterest_FRN": str(accrued_frn),
        "BenchmarkRate_FRN": str(benchmark_frn),
        "LastAccrualDate": "",
        "IssuePriceRatio": "1.0",
        "IssueProceeds": str(face),
        "IssueYieldAtIssue": "0.0",
        "TimeToMaturity": "",
        "DiscountYield": "",
        "CleanPrice": "",
        "AccruedInterest": "",
        "DirtyValue": "",
        "DirtyPriceRatio": "",
    }


def _roll_fixture_opening_book(package: Path) -> dict:
    inputs = package / "forecast_inputs"
    sources = package / "sources"
    opening_path = inputs / "tdcsim_opening_portfolio.csv"
    opening_fields = list(_read_csv(opening_path)[0])
    source_path = sources / "opening" / "tdcsim_initial_portfolio_cohorts.csv"
    book = pd.read_csv(source_path, parse_dates=["IssueDate", "MaturityDate"])
    governed = json.loads((sources / "opening" / "tdcsim_ratewall_input_manifest.json").read_text())
    roll = roll_opening_marketables(
        book, cusip_column="cusip_or_synthetic_id",
        noncb_weights=governed["artifacts"]["initial_portfolio"]["noncb_weights"],
        auctions_query=sources / "fiscaldata" / "auctions_query.csv",
        soma_holdings=sources / SOMA_SOURCE,
        tips_cpi_detail=sources / "fiscaldata" / "tips_cpi_data_detail.csv",
    )
    rolled, summary, diagnostics = roll.book, roll.summary, roll.diagnostics
    rolled["IssueID"] = rolled["cusip_or_synthetic_id"]
    rolled[[*opening_fields, "IssueID"]].to_csv(opening_path, index=False)
    _write_json(inputs / "tdcsim_opening_portfolio_metadata.json", {
        "simulation_start_tips_observation": roll.tips_metadata["simulation_start_tips_observation"],
    })

    dts = opening_cash_reconciliation(sources / DTS_CASH_SOURCE, sources / DTS_FLOWS_SOURCE)
    diagnostics.append({"row_type": "dts_cash_bridge", **dts})
    fields = list(dict.fromkeys(key for row in diagnostics for key in row))
    _write_csv(inputs / "tdcsim_opening_rollforward_diagnostics.csv", fields, diagnostics)
    cash_row = {
        "base_date": OPENING_CLOSE_DATE, "observation_date": OPENING_CLOSE_DATE,
        "available_date": ACTUALS_AVAILABLE_AS_OF, "base_balance_bil": OPERATING_CASH_TARGET,
    }
    _write_csv(inputs / "tdcsim_operating_cash_path.csv", list(cash_row), [cash_row])
    fy2026, = [row for row in _read_csv(inputs / "tdcsim_cbo_fiscal_baseline.csv") if row["fiscal_year"] == "2026"]
    splice = build_opening_fy_splice(
        scenario_id="cbo_full_horizon_local_smoke_3m", cbo_primary_deficit_bil=float(fy2026["primary_deficit_bil"]),
        mts_deficit_bil=1246.20326638693, mts_interest_bil=722.70651124320, dts=dts,
    )
    _write_csv(inputs / "tdcsim_current_fy_splice.csv", list(splice), [splice])

    rows = _read_csv(opening_path)
    security_totals = {}
    holder_totals = {}
    for row in rows:
        debt = float(row["AdjustedPrincipal"]) if row["SecurityType"] == "TIPS" else float(row["FaceValue"])
        security_totals[row["SecurityType"]] = security_totals.get(row["SecurityType"], 0.0) + debt
        holder_totals[row["HolderType"]] = holder_totals.get(row["HolderType"], 0.0) + debt
    return {
        "record_date": "2026-05-31",
        "opening_close_date": OPENING_CLOSE_DATE,
        "soma_snapshot_date": SOMA_SNAPSHOT_DATE,
        "row_count": len(rolled),
        "source_rows_loaded": len(book),
        "prestart_matured_rows": summary["matured_rows"],
        "prestart_rollforward_rows": summary["settlement_lot_rows"],
        "prestart_rollforward_face_bil": summary["matured_face_bil"],
        "opening_rollforward_status": OPENING_ROLLFORWARD_STATUS,
        "opening_rollforward_source_debt_base_bil": summary["source_debt_base_bil"],
        "opening_rollforward_aligned_debt_base_bil": summary["opening_debt_base_bil"],
        "opening_rollforward_debt_base_delta_bil": summary["opening_debt_base_bil"] - summary["source_debt_base_bil"],
        "frn_source_boundary_note": "contract-compliant FRN opening accrual source",
        # Counted from the rolled book: aligning to SOMA merges same-terms lots, so the
        # enriched row counts follow the roll rather than the source book.
        "tips_rows_enriched": int(rolled["SecurityType"].eq("TIPS").sum()),
        "tips_unique_cusips_enriched": int(rolled.loc[rolled["SecurityType"].eq("TIPS"), "cusip_or_synthetic_id"].nunique()),
        "frn_rows_enriched": int(rolled["SecurityType"].eq("FRN").sum()),
        "frn_unique_cusips_enriched": int(rolled.loc[rolled["SecurityType"].eq("FRN"), "cusip_or_synthetic_id"].nunique()),
        "opening_tips_adjusted_principal_bil": 530.0,
        "opening_tips_reconstructed_original_principal_bil": 397.5,
        "opening_tips_indexation_accretion_embedded_bil": 132.5,
        "opening_tips_reference_cpi": 331.80363,
        "simulation_start_tips_adjusted_principal_bil": 530.0,
        "opening_frn_face_value_bil": 120.0,
        "opening_frn_accrued_interest_bil": 4.0,
        "security_type_totals_bil": security_totals,
        "holder_totals_bil": holder_totals,
        "opening_cb_holdings_bil": holder_totals["CB"],
        "simulation_start_cb_holdings_bil": holder_totals["CB"],
        "claim_boundary": OPENING_PORTFOLIO_CLAIM_BOUNDARY,
    }


def _write_net_interest_diagnostic(package: Path, rows: list[dict[str, str]]) -> None:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        year = int(row["Date"][:4])
        month = int(row["Date"][5:7])
        fiscal_year = year + int(month >= 10)
        grouped.setdefault(fiscal_year, []).append(row)
    output = []
    for fiscal_year in sorted(grouped):
        group = grouped[fiscal_year]
        coupon = sum(float(row["InterestPaid_Bonds"]) for row in group)
        discount = sum(float(row["IssueDiscountCost_Period"]) for row in group)
        tips = sum(float(row["TIPSInflationAccretion_Period"]) for row in group)
        nonmarketable = sum(float(row["NonMarketableInterestCapitalized_Period"]) for row in group)
        modeled = coupon + discount + tips + nonmarketable
        cbo = modeled + 1.0
        output.append(
            {
                "run_id": "baseline",
                "fiscal_year": str(fiscal_year),
                "cbo_net_interest_bil": str(cbo),
                "residual_bil": "1.0",
                "residual_pct": "0.0",
                "scope_status": "partial_market_securities_proxy",
                "calibration_mode": "diagnostic_only_no_cbo_interest_plug",
                "threshold_status": "ok",
                "claim_status": "partial_nonbinding_diagnostic_regardless_of_numeric_residual",
                "largest_residual_component": "",
                "issue_id": "",
                "fiscal_year_start": group[0]["Date"],
                "fiscal_year_end": group[-1]["Date"],
                "rows": str(len(group)),
                "coupon_cash_interest_bil": str(coupon),
                "issuance_discount_booked_at_issue_proxy_bil": str(discount),
                "tips_principal_indexation_bil": str(tips),
                "nonmarketable_interest_capitalized_bil": str(nonmarketable),
                "modeled_interest_related_proxy_bil": str(modeled),
                "average_total_debt_bil": "1.0",
                "modeled_average_interest_rate_pct": "0.0",
                "runtime_role": "diagnostic_only",
                "scope_completeness_status": "partial_interest_related_proxy_not_budgetary_net_interest",
                "numeric_residual_status": "ok",
                "source_status": "fixture",
                "claim_boundary": "no_cbo_net_interest_cash_or_issuance_plug",
            }
        )
    _write_csv(package / "tdcsim_partial_interest_financing_diagnostic.csv", list(output[0]), output)


def _mutate_csv_cell(path: Path, *, row_index: int, column: str, value: str) -> None:
    rows = _read_csv(path)
    rows[row_index][column] = value
    _write_csv(path, list(rows[0]), rows)


def _rewrite_summary(results_path: Path, summary_path: Path) -> None:
    _write_json(summary_path, _summary(_read_csv(results_path)))


def _refresh_source_package_hashes(package: Path) -> None:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_files = manifest["source_package"]["source_files"]
    for rel in list(source_files):
        if rel == "source_manifest.json":
            continue
        path = package / "sources" / rel
        source_files[rel]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        source_files[rel]["bytes"] = path.stat().st_size
    source_manifest = {
        "schema_version": "tdcsim_cbo_source_package_manifest_v1",
        "status": "package_relative_sources_for_cbo_forecast_smoke",
        "inventory_status": "complete_required_inventory_no_silent_skips",
        "files": {rel: entry for rel, entry in source_files.items() if rel != "source_manifest.json"},
    }
    _write_json(package / "sources" / "source_manifest.json", source_manifest)
    source_manifest_path = package / "sources" / "source_manifest.json"
    source_files["source_manifest.json"]["sha256"] = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    source_files["source_manifest.json"]["bytes"] = source_manifest_path.stat().st_size
    manifest["source_package"]["source_manifest_sha256"] = source_files["source_manifest.json"]["sha256"]
    manifest["source_package"]["source_file_count"] = len(source_files)
    _write_json(manifest_path, manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)


def _refresh_manifest_hashes(package: Path) -> None:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"] = _artifact_hashes(package)
    _write_json(manifest_path, manifest)
    _write_json(package / "forecast_inputs" / "source_contract_smoke.json", manifest)


def _artifact_hashes(package: Path) -> dict:
    artifact_hashes = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package).as_posix()
        if rel in {"manifest.json", "forecast_inputs/source_contract_smoke.json"}:
            continue
        entry = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                entry["rows_including_header"] = sum(1 for _ in csv.reader(handle))
        artifact_hashes[rel] = entry
    return artifact_hashes


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(data))


def _json_bytes(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
