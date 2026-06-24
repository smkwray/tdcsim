import csv
import hashlib
import json
import platform
from datetime import date, timedelta
from pathlib import Path
import sys

import pytest

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
SRC_DIR = PROJECT_ROOT_FOR_IMPORT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.verify_cbo_forecast_package import REQUIRED_FILES, create_package_zip, verify_package
from cbo_yield_curve_surface import build_yield_curve_surface_rows
from forecast_input_builder import (
    build_cbo_fiscal_baseline_rows,
    build_cbo_macro_forecast_path_rows,
    parse_cbo_budget_source_contract,
    parse_cbo_economic_quarterly_source_contract,
)
from tips_indexation import build_monthly_tips_cpi_path_rows, build_tips_real_yield_path_rows


RUN_ROWS = 3755
BUDGET_WORKBOOK = PROJECT_ROOT_FOR_IMPORT.parent / "ratewall" / "data" / "raw" / "cbo" / "51118-2026-02-Budget-Projections.xlsx"
ECONOMIC_WORKBOOK = PROJECT_ROOT_FOR_IMPORT.parent / "ratewall" / "data" / "raw" / "cbo" / "51135-2026-02-Economic-Projections.xlsx"
RESULT_COLUMNS = [
    "Date",
    "TotalDebt_Agg",
    "CBOControlledDebtTarget",
    "CBOControlledDebtTargetApplicable",
    "CBOControlledDebtPostIssuance",
    "CBOControlledDebtTargetError",
    "CBORequiredFaceIssuance",
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
    "CB_InterestIncome",
    "CB_NetIncome",
    "CB_Remittance",
    "CB_DeferredAsset",
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
    _mutate_csv_cell(
        package / "exports_zero_residual" / "results_compact.csv",
        row_index=1,
        column="CBOControlledDebtPostIssuance",
        value="999.0",
    )
    _mutate_csv_cell(
        package / "exports_zero_residual" / "results_compact.csv",
        row_index=1,
        column="CBOControlledDebtTargetError",
        value="-2.0",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="reported debt target error|recomputed max debt target error"):
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
        package / "exports_zero_residual" / "results_compact.csv",
        row_index=1,
        column="CBOFedSecondaryPurchaseFace",
        value="999.0",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="Fed secondary purchase face"):
        verify_package(package)


def test_cbo_package_verifier_rejects_remittance_overclaim_with_refreshed_hashes(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    _mutate_csv_cell(
        package / "exports_zero_residual" / "results_compact.csv",
        row_index=1,
        column="CB_Remittance",
        value="1.0",
    )
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="CB_Remittance must remain blank/NaN"):
        verify_package(package)


def test_cbo_package_verifier_rejects_deprecated_net_interest_file_with_refreshed_hashes(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    (package / "tdcsim_modeled_net_interest_diagnostic.csv").write_text("overclaim\n", encoding="utf-8")
    _refresh_manifest_hashes(package)

    with pytest.raises(ValueError, match="deprecated/overclaim"):
        verify_package(package)


def test_cbo_package_verifier_rejects_coordinated_debt_target_path_mutation(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    for run_dir in ("exports_zero_residual", "exports_constant_real_tga", "exports_constant_nominal_tga"):
        results_path = package / run_dir / "results_compact.csv"
        _mutate_csv_cell(results_path, row_index=RUN_ROWS - 1, column="CBOControlledDebtTarget", value="999999.0")
        _mutate_csv_cell(
            results_path,
            row_index=RUN_ROWS - 1,
            column="CBOControlledDebtPostIssuance",
            value="999999.0",
        )
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

    with pytest.raises(ValueError, match="tdcsim_yield_curve_surface.csv row 1 nominal_rate|tdcsim_tips_real_yield_path.csv"):
        verify_package(package)


def test_cbo_package_verifier_rejects_rehashed_interior_debt_path_and_results_mutation(tmp_path: Path) -> None:
    package = _write_valid_package(tmp_path / "pkg")
    for run_dir in ("exports_zero_residual", "exports_constant_real_tga", "exports_constant_nominal_tga"):
        results_path = package / run_dir / "results_compact.csv"
        _mutate_csv_cell(results_path, row_index=100, column="CBOControlledDebtTarget", value="999999.0")
        _mutate_csv_cell(results_path, row_index=100, column="CBOControlledDebtPostIssuance", value="999999.0")
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
    runs = {
        "zero_residual": ("exports_zero_residual", 783.093, 1239.186, 0.0),
        "constant_real_tga": ("exports_constant_real_tga", 1239.186, 1239.186, 456.0),
        "constant_nominal_tga": ("exports_constant_nominal_tga", 981.113, 981.113, 198.0),
    }

    _write_opening_files(package)
    _write_auxiliary_required_files(package)
    debt_rows = _read_csv(package / "forecast_inputs" / "tdcsim_debt_stock_path.csv")
    run_summaries = {}
    run_rows = {}
    for run_id, (run_dir, final_tga, final_cash_target, residual_total) in runs.items():
        rows = _result_rows(final_tga, final_cash_target, residual_total, debt_rows)
        run_rows[run_id] = rows
        _write_csv(package / run_dir / "results_compact.csv", RESULT_COLUMNS, rows)
        summary = _summary(rows)
        run_summaries[run_id] = summary
        _write_json(package / run_dir / "summary.json", summary)

    _write_csv(package / "exports_zero_residual" / "final_portfolio_active.csv", ["col"], [{"col": "value"}])
    _write_csv(package / "exports_zero_residual" / "maturity_ledger.csv", ["col"], [{"col": "value"}])
    _write_cash_sensitivity(package, run_rows)
    _write_net_interest_diagnostic(package, run_rows["zero_residual"])
    source_package = _write_source_package(package)

    opening_manifest = _opening_manifest(package / "forecast_inputs" / "tdcsim_opening_portfolio.csv")
    manifest = {
        "scenario_id": "cbo_full_horizon_local_smoke_3m",
        "date_range": {
            "simulation_start_date": "2026-06-21",
            "simulation_end_date": "2036-09-30",
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
            "claim_boundary": "fed_holdings_path_guides_cb_absorption_not_total_debt_or_cash",
            "mode": "cbo_fed_holdings_endpoints_linear_actual_days_holder_target",
            "runtime_columns": [
                "CBOFedHoldingsTarget",
                "CBOFedHoldingsTargetError",
                "CBOFedAuctionShare",
            ],
            "runtime_effect": "holder_allocation_only_total_issuance_unchanged",
        },
        "run_summaries": run_summaries,
        "source_package": source_package,
        "source_row_selectors": _source_row_selectors(),
        "claim_boundary": _claim_boundary(),
        "trust_anchor": _trust_anchor(lock_path),
        "outputs": {
            "zero_residual_summary": "exports_zero_residual/summary.json",
            "constant_real_tga_summary": "exports_constant_real_tga/summary.json",
            "constant_real_tga_cash_residual_file": "forecast_inputs/tdcsim_cash_reconciliation_residual_constant_real_tga.csv",
            "constant_nominal_tga_summary": "exports_constant_nominal_tga/summary.json",
            "constant_nominal_tga_cash_residual_file": "forecast_inputs/tdcsim_cash_reconciliation_residual_constant_nominal_tga.csv",
            "constant_nominal_tga_operating_cash_file": "forecast_inputs/tdcsim_operating_cash_path_constant_nominal.csv",
            "cash_sensitivity_comparison": "cash_sensitivity_comparison.csv",
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
        "dts": "account_type=Treasury General Account (TGA) Closing Balance",
    }


def _claim_boundary() -> dict:
    return {
        "schema_version": "tdcsim_cbo_package_claim_boundary_v1",
        "package_role": "cbo_baseline_forecast_evidence_package",
        "net_interest_role": "partial_interest_financing_diagnostic_only",
        "issuance_mix_role": "tdcsim_assumption_not_cbo_prescription",
        "cash_role": "operating_cash_sensitivity_only_not_debt_or_issuance_driver",
        "excluded_claims": [
            "does_not_claim_exact_historical_replay",
            "does_not_claim_full_soma_transaction_replay",
            "does_not_claim_budgetary_net_interest_replication",
            "does_not_claim_generated_evidence_tracked_in_git",
            "does_not_claim_receipts_outlays_decomposition",
            "does_not_use_cbo_net_interest_as_cash_or_issuance_plug",
            "does_not_claim_cbo_issuance_mix",
            "does_not_model_remittances_or_monetary_settlement_effects",
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

    rollforward = [
        {
            "row_type": "summary",
            "source_record_date": "2026-05-31",
            "simulation_start_date": "2026-06-21",
            "source_rows_maturing_on_or_before_start": "35",
            "replacement_rows": "35",
            "source_face_bil": "35.0",
            "replacement_face_bil": "35.0",
            "face_delta_bil": "0.0",
            "status": "balanced",
            "derivation": "source_rows_are_replaced_by_3_month_bills",
            "claim_boundary": "mechanical_rollforward",
            "source_bond_id": "",
            "replacement_bond_id": "",
            "source_cusip": "",
            "source_security_type": "",
            "replacement_security_type": "",
            "holder_type": "",
            "holder_subbucket": "",
            "source_maturity_date": "",
            "replacement_issue_date": "",
            "replacement_maturity_date": "",
        }
    ]
    for index in range(35):
        rollforward.append(
            {
                **{key: "" for key in rollforward[0]},
                "row_type": "replacement",
                "source_record_date": "2026-05-31",
                "simulation_start_date": "2026-06-21",
                "source_face_bil": "1.0",
                "replacement_face_bil": "1.0",
                "face_delta_bil": "0.0",
                "status": "balanced",
                "source_bond_id": str(index + 1),
                "replacement_bond_id": str(index + 2291),
                "source_security_type": "Fixed",
                "replacement_security_type": "Fixed",
                "holder_type": holders[index % len(holders)],
            }
        )
    _write_csv(package / "forecast_inputs" / "tdcsim_opening_rollforward_diagnostics.csv", list(rollforward[0]), rollforward)
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


def _write_cbo_source_fixtures(package: Path) -> None:
    if not BUDGET_WORKBOOK.exists() or not ECONOMIC_WORKBOOK.exists():
        pytest.skip("optional local CBO workbook fixtures are not present")
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
        actuals_available_as_of="2026-06-17",
        output_tenors=(0.25, 10.0),
    )
    frn_rows = []
    real_yield_rows = []
    three_month_rates = sorted(
        (str(row["curve_date"]), float(row["nominal_rate_decimal"]))
        for row in yield_rows
        if abs(float(row["tenor_years"]) - 0.25) <= 1e-9
    )
    nominal_by_key = {
        (str(row["curve_date"]), str(row["tenor_years"])): float(row["nominal_rate_decimal"])
        for row in yield_rows
    }
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
        opening_reference_cpi=332.08433,
        reference_lag_months=3,
        pricing_horizon_end_date="2046-10-01",
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )
    real_yield_rows = build_tips_real_yield_path_rows(
        scenario_id="cbo_full_horizon_local_smoke_3m",
        yield_surface_rows=yield_rows,
        tips_cpi_path_rows=tips_cpi_rows,
        observation_date="2026-02-11",
        available_date="2026-02-11",
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
            pytest.skip("optional local CBO workbook fixtures are not present")
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
        package / "sources" / "fiscaldata" / "dts_tga_closing_balance_2026-06-16.csv",
        ["record_date", "account_type", "open_today_bal"],
        [
            {
                "record_date": "2026-06-16",
                "account_type": "Treasury General Account (TGA) Closing Balance",
                "open_today_bal": "981113",
            }
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
        ["record_date", "security_type_desc", "outstanding_amt"],
        [
            {"record_date": "2026-05-31", "security_type_desc": "Marketable", "outstanding_amt": str(i)}
            for i in range(2000)
        ],
    )
    _write_csv(
        package / "sources" / "treasury" / "debt_to_penny_public_debt_2026-05-29.csv",
        ["record_date", "debt_held_public_amt"],
        [{"record_date": "2026-05-29", "debt_held_public_amt": "31515369798622.98"}],
    )
    _write_csv(
        package / "sources" / "opening" / "tdcsim_initial_portfolio_cohorts.csv",
        ["cohort_id", "status"],
        [{"cohort_id": "fixture", "status": "package_fixture"}],
    )
    _write_json(
        package / "sources" / "opening" / "tdcsim_ratewall_input_manifest.json",
        {"status": "package_fixture"},
    )
    _write_csv(
        package / "sources" / "fiscaldata" / "tips_cpi_data_detail.csv",
        ["cusip", "index_date", "ref_cpi", "index_ratio"],
        [{"cusip": "fixture", "index_date": "2026-06-21", "ref_cpi": "332.08433", "index_ratio": "1.25"}],
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
    for rel in (
        "opening/tdcsim_initial_portfolio_cohorts.csv",
        "opening/tdcsim_ratewall_input_manifest.json",
        "fiscaldata/tips_cpi_data_detail.csv",
        "fiscaldata/frn_daily_indexes.csv",
        "fiscaldata/mts_table_1_deficit_2026-05-31.csv",
        "fiscaldata/mts_table_9_net_interest_2026-05-31.csv",
        "fiscaldata/dts_tga_closing_balance_2026-06-16.csv",
        "fiscaldata/mspd_table_1_debt_bridge_2026-05-31.csv",
        "fiscaldata/mspd_table_3_market_2026-05-31.csv",
        "treasury/debt_to_penny_public_debt_2026-05-29.csv",
        "treasury/base_curve_2026-06-16.json",
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
        "dts_tga_closing_balance_row": {
            "record_date": "2026-06-16",
            "account_type": "Treasury General Account (TGA) Closing Balance",
            "open_today_bal": "981113",
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


def _result_rows(
    final_tga: float,
    final_cash_target: float,
    residual_total: float,
    debt_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    start = date(2026, 6, 21)
    residual_step = residual_total / (RUN_ROWS - 1) if residual_total else 0.0
    rows = []
    for index in range(RUN_ROWS):
        current = start + timedelta(days=index)
        debt = float(debt_rows[index]["marketable_treasury_public_target_bil"])
        fed_target = 0.0 if index == 0 else 100.0 + index
        fed_begin = 0.0 if index <= 1 else 100.0 + index - 1
        fed_end = fed_target
        fed_purchase = fed_end - fed_begin
        residual = 0.0 if index == 0 else residual_step
        row = {
            "Date": current.isoformat(),
            "TotalDebt_Agg": debt,
            "CBOControlledDebtTarget": debt,
            "CBOControlledDebtTargetApplicable": 0.0 if index == 0 else 1.0,
            "CBOControlledDebtPostIssuance": debt,
            "CBOControlledDebtTargetError": 0.0,
            "CBORequiredFaceIssuance": 0.0 if index == 0 else 1.0,
            "NewDebtIssued": 0.0 if index == 0 else 1.0,
            "AuctionProceeds": 0.0 if index == 0 else 0.99,
            "IssueDiscountCost_Period": 0.0 if index == 0 else 0.01,
            "InterestPaid_Bonds": 0.0 if index == 0 else 0.2,
            "NonMarketableInterestCapitalized_Period": 0.0,
            "PrimaryDeficit": 0.0 if index == 0 else 0.5,
            "TGA": final_tga,
            "CBOOperatingCashTarget": final_cash_target,
            "CBOCashResidual": final_tga - final_cash_target,
            "CBOCashReconciliationResidual": residual,
            "CBORemittanceCashEffect": 0.0,
            "CBOFedHoldingsTarget": fed_target,
            "CBOFedHoldingsTargetApplicable": 0.0 if index == 0 else 1.0,
            "CBOFedHoldingsTargetError": 0.0,
            "CBOFedAuctionShare": 0.0,
            "CBOFedSecondaryPurchaseFace": 0.0 if index == 0 else fed_purchase,
            "CBOFedSecondaryPurchaseCash": 0.0,
            "CBOFedSecondaryPurchaseReserveEffect": 0.0,
            "CBOFedSecondaryPurchaseDepositEffect": 0.0,
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
                "synthetic_cb_treasury_stock_target_par_reallocation" if index > 0 else ""
            ),
            "CBOFedSettlementScope": (
                "stock_reallocation_only_no_reserve_deposit_or_market_price_claim" if index > 0 else ""
            ),
            "CBORemittanceStatus": "not_modeled_cbo_primary_deficit_embeds_baseline_revenues",
            "CB_InterestIncome": 0.0,
            "CB_NetIncome": "" if index > 0 else 0.0,
            "CB_Remittance": "" if index > 0 else 0.0,
            "CB_DeferredAsset": "" if index > 0 else 0.0,
            "CBONetInterestDiagnostic": 0.0,
            "CBOTotalDeficitDiagnostic": 0.0,
            "NetInterestDiagnosticStatus": "cbo_reported_check_only",
            "DebtHeld_Banks": 10.0 + index,
            "DebtHeld_CentralBank": fed_end,
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
        "max_abs_target_error_bil": max(abs(float(row["CBOControlledDebtTargetError"])) for row in rows),
        "max_abs_cbo_fed_holdings_error_bil": max(abs(float(row["CBOFedHoldingsTargetError"])) for row in rows),
        "final_cbo_fed_holdings_error_bil": float(final["CBOFedHoldingsTargetError"]),
        "final_cbo_fed_holdings_target_bil": float(final["CBOFedHoldingsTarget"]),
        "final_cbo_target_bil": float(final["CBOControlledDebtTarget"]),
        "final_total_debt_bil": float(final["TotalDebt_Agg"]),
        "final_tga_bil": float(final["TGA"]),
        "final_operating_cash_target_bil": float(final["CBOOperatingCashTarget"]),
        "final_cash_residual_bil": float(final["CBOCashResidual"]),
        "net_cash_reconciliation_residual_bil": sum(residuals),
        "gross_abs_cash_reconciliation_residual_bil": sum(abs(value) for value in residuals),
        "positive_cash_reconciliation_residual_bil": sum(value for value in residuals if value > 0.0),
        "negative_cash_reconciliation_residual_bil": sum(value for value in residuals if value < 0.0),
        "nonzero_cash_reconciliation_residual_days": sum(1 for value in residuals if abs(value) > 1e-12),
        "total_required_face_issuance_bil": sum(float(row["CBORequiredFaceIssuance"]) for row in rows),
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


def _opening_manifest(opening_path: Path) -> dict:
    rows = _read_csv(opening_path)
    security_totals = {}
    holder_totals = {}
    for row in rows:
        debt = float(row["AdjustedPrincipal"]) if row["SecurityType"] == "TIPS" else float(row["FaceValue"])
        security_totals[row["SecurityType"]] = security_totals.get(row["SecurityType"], 0.0) + debt
        holder_totals[row["HolderType"]] = holder_totals.get(row["HolderType"], 0.0) + debt
    return {
        "row_count": 2290,
        "source_rows_loaded": 2290,
        "prestart_rollforward_rows": 35,
        "prestart_rollforward_face_bil": 35.0,
        "opening_rollforward_debt_base_delta_bil": 0.0,
        "opening_source_common_date_status": "no_same_day_cusip_level_mspd_table3_book_available_locally",
        "frn_source_boundary_note": "contract-compliant FRN opening accrual source",
        "tips_rows_enriched": 265,
        "tips_unique_cusips_enriched": 53,
        "frn_rows_enriched": 40,
        "frn_unique_cusips_enriched": 8,
        "opening_tips_adjusted_principal_bil": 530.0,
        "opening_tips_reconstructed_original_principal_bil": 397.5,
        "opening_tips_indexation_accretion_embedded_bil": 132.5,
        "opening_tips_reference_cpi": 332.08433,
        "opening_frn_face_value_bil": 120.0,
        "opening_frn_accrued_interest_bil": 4.0,
        "security_type_totals_bil": security_totals,
        "holder_totals_bil": holder_totals,
        "claim_boundary": (
            "prebuilt_mspd_table3_market_derived_cohort_book_with_z1_holder_provenance_metadata; "
            "z1_inputs_not_packaged_for_portable_holder_constraint_reproduction; "
            "securities_maturing_between_record_date_and_simulation_start_are_explicit_prestart_rollforward_bills"
        ),
    }


def _write_cash_sensitivity(package: Path, run_rows: dict[str, list[dict[str, str]]]) -> None:
    columns = [
        "run_id",
        "rows",
        "max_abs_delta_CBOControlledDebtTarget",
        "max_abs_delta_CBORequiredFaceIssuance",
        "max_abs_delta_NewDebtIssued",
        "max_abs_delta_AuctionProceeds",
        "max_abs_delta_DebtHeld_Banks",
        "max_abs_delta_DebtHeld_CentralBank",
        "max_abs_delta_DebtHeld_Foreign",
        "max_abs_delta_DebtHeld_DomesticNonBanks",
        "final_tga_bil",
        "final_operating_cash_target_bil",
        "net_cash_reconciliation_residual_bil",
        "gross_abs_cash_reconciliation_residual_bil",
    ]
    rows = []
    for run_id, results in run_rows.items():
        residuals = [float(row["CBOCashReconciliationResidual"]) for row in results]
        rows.append(
            {
                "run_id": run_id,
                "rows": str(len(results)),
                "max_abs_delta_CBOControlledDebtTarget": "0.0",
                "max_abs_delta_CBORequiredFaceIssuance": "0.0",
                "max_abs_delta_NewDebtIssued": "0.0",
                "max_abs_delta_AuctionProceeds": "0.0",
                "max_abs_delta_DebtHeld_Banks": "0.0",
                "max_abs_delta_DebtHeld_CentralBank": "0.0",
                "max_abs_delta_DebtHeld_Foreign": "0.0",
                "max_abs_delta_DebtHeld_DomesticNonBanks": "0.0",
                "final_tga_bil": results[-1]["TGA"],
                "final_operating_cash_target_bil": results[-1]["CBOOperatingCashTarget"],
                "net_cash_reconciliation_residual_bil": str(sum(residuals)),
                "gross_abs_cash_reconciliation_residual_bil": str(sum(abs(value) for value in residuals)),
            }
        )
    _write_csv(package / "cash_sensitivity_comparison.csv", columns, rows)


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
                "run_id": "zero_residual",
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
