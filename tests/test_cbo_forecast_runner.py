from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.run_cbo_forecast_smoke import (
    ACTUALS_AVAILABLE_AS_OF,
    DEFAULT_BUDGET_WORKBOOK,
    DEFAULT_ECONOMIC_WORKBOOK,
    PROJECT_ROOT,
    RUNNER_MTS_TABLE_1_DEFICIT_SELECTOR,
    RUNNER_MTS_TABLE_9_NET_INTEREST_SELECTOR,
    SOURCE_PACKAGE_INPUTS,
    build_forecast_bundle,
    parse_mts_table_1_deficit_row,
    parse_mts_table_9_net_interest_row,
    resolve_source_package_inputs,
    validate_yield_surface_rate_units,
)
from forecast_bundle_builders import build_current_fy_splice_row_from_fiscaldata
from forecast_paths import load_yield_curve_surface


WORKBOOKS_PRESENT = DEFAULT_BUDGET_WORKBOOK.exists() and DEFAULT_ECONOMIC_WORKBOOK.exists()


@pytest.mark.skipif(not WORKBOOKS_PRESENT, reason="optional local CBO workbook fixtures are not present")
def test_runner_builds_exact_manifest_and_inputs_without_engine_loop(tmp_path: Path) -> None:
    del tmp_path
    bundle = build_forecast_bundle(PROJECT_ROOT / "output" / "pytest_cbo_runner_manifest", clean=True)

    manifest = bundle.manifest
    assert manifest["scenario_id"] == "cbo_full_horizon_local_smoke_3m"
    assert manifest["date_range"]["simulation_start_date"] == "2026-06-21"
    assert manifest["date_range"]["simulation_end_date"] == "2036-09-30"
    assert manifest["source_row_selectors"]["mts_table_1"] == RUNNER_MTS_TABLE_1_DEFICIT_SELECTOR
    assert manifest["source_row_selectors"]["mts_table_9"] == RUNNER_MTS_TABLE_9_NET_INTEREST_SELECTOR
    assert manifest["source_row_selectors"]["debt_to_the_penny"] == (
        "record_date=2026-05-29;column=debt_held_public_amt"
    )
    assert manifest["rate_unit_declaration"]["runtime_rate_unit"] == "decimal"
    assert (
        manifest["opening_portfolio_construction"]["mode"]
        == "prebuilt_mspd_derived_cohort_book_with_explicit_prestart_rollforward"
    )
    assert manifest["opening_portfolio_construction"]["row_count"] == 2290
    assert manifest["opening_portfolio_construction"]["prestart_rollforward_rows"] == 35
    assert (
        manifest["opening_portfolio_construction"]["opening_rollforward_status"]
        == "mechanically_balanced_source_record_to_simulation_start"
    )
    assert manifest["opening_portfolio_construction"]["opening_rollforward_debt_base_delta_bil"] == pytest.approx(
        0.0,
        abs=1e-9,
    )
    assert "no_same_day_cusip_level_mspd_table3_book_available" in (
        manifest["opening_portfolio_construction"]["opening_source_common_date_status"]
    )
    assert manifest["opening_portfolio_construction"]["opening_tips_reference_cpi"] > 300.0
    assert "treasurydirect_tips_cpi_detail_join" in (
        manifest["opening_portfolio_construction"]["tips_initialization"]
    )
    assert manifest["opening_portfolio_construction"]["tips_cpi_detail_index_date"] == "2026-06-21"
    assert manifest["opening_portfolio_construction"]["tips_rows_enriched"] == 265
    assert manifest["opening_portfolio_construction"]["tips_unique_cusips_enriched"] == 53
    assert manifest["opening_portfolio_construction"]["opening_tips_indexation_accretion_embedded_bil"] > 300.0
    assert manifest["opening_portfolio_construction"]["frn_opening_date"] == "2026-06-21"
    assert manifest["opening_portfolio_construction"]["frn_source_available_as_of"] == "2026-06-17"
    assert manifest["opening_portfolio_construction"]["frn_rows_enriched"] == 40
    assert manifest["opening_portfolio_construction"]["frn_unique_cusips_enriched"] == 8
    assert manifest["opening_portfolio_construction"]["frn_source_accrual_end_dates"] == ["2026-06-21"]
    assert manifest["opening_portfolio_construction"]["opening_frn_accrued_interest_bil"] > 3.0
    assert "contract-compliant" in (
        manifest["opening_portfolio_construction"]["frn_source_boundary_note"]
    )
    assert manifest["source_values"]["debt_bridge"]["treasury_public_debt_bil"] == pytest.approx(
        31_515.369918445198
    )
    assert manifest["source_values"]["debt_bridge"]["debt_to_penny_public_debt_bil"] == pytest.approx(
        31_515.36979862298
    )
    assert manifest["source_values"]["debt_bridge"]["public_nonmarketable_treasury_bil"] == pytest.approx(
        620.82177409502
    )
    assert manifest["source_values"]["debt_bridge"]["date_alignment"] == (
        "mspd_month_end_with_latest_prior_debt_to_penny_reconciliation"
    )
    assert manifest["opening_portfolio_construction"]["security_type_totals_bil"]["TIPS"] > 2_000.0
    assert manifest["opening_portfolio_construction"]["security_type_totals_bil"]["FRN"] > 600.0
    assert set(manifest["opening_portfolio_construction"]["holder_totals_bil"]) == {
        "Banks",
        "CB",
        "Foreign",
        "Private",
    }
    assert manifest["issuance_mix"]["fixed_remainder_shares"] == {"bills": 0.25, "notes": 0.55, "bonds": 0.2}
    assert manifest["issuance_mix"]["bills"] == {"0.5y": 1.0}
    assert manifest["issuance_mix"]["notes"] == {"5.0y": 1.0}
    assert manifest["issuance_mix"]["bonds"] == {"20.0y": 1.0}
    assert manifest["issuance_mix"]["tips"]["target_percentage"] == 0.06
    assert manifest["issuance_mix"]["tips"]["10.0y"] == 1.0
    assert manifest["issuance_mix"]["frn"]["target_percentage"] == 0.04
    assert manifest["operating_cash_construction"]["construction_mode"] == "constant_real_cbo_cpi_u"
    assert "z1_holder_provenance_metadata" in manifest["opening_portfolio_construction"]["claim_boundary"]
    assert "z1_inputs_not_packaged_for_portable_holder_constraint_reproduction" in (
        manifest["opening_portfolio_construction"]["claim_boundary"]
    )
    assert manifest["claim_boundary"]["package_role"] == "cbo_baseline_forecast_evidence_package"
    assert manifest["source_package"]["source_file_count"] >= len(SOURCE_PACKAGE_INPUTS)
    source_files = manifest["source_package"]["source_files"]
    for rel_path in SOURCE_PACKAGE_INPUTS.values():
        assert rel_path in source_files
        assert source_files[rel_path]["source_path"] == rel_path
        assert not Path(source_files[rel_path]["source_path"]).is_absolute()
    assert source_files["opening/tdcsim_ratewall_input_manifest.json"]["portable_reserialized"] is True
    assert source_files["opening/tdcsim_ratewall_input_manifest.json"]["origin_repo_path"].endswith(
        "data/ratewall_inputs/tdcsim_ratewall_input_manifest.json"
    )
    assert source_files["fiscaldata/mts_table_1_deficit_2026-05-31.csv"]["origin_source_label"] == (
        "runner_canonical_exact_source_row_extract"
    )
    assert source_files["source_manifest.json"]["source_path"] == "source_manifest.json"
    assert (
        manifest["opening_portfolio_construction"]["source_manifest_original_sha256"]
        != manifest["opening_portfolio_construction"]["source_manifest_portable_sha256"]
    )
    assert manifest["opening_portfolio_construction"]["source_file"] == (
        "sources/opening/tdcsim_initial_portfolio_cohorts.csv"
    )
    assert manifest["opening_portfolio_construction"]["source_manifest"] == (
        "sources/opening/tdcsim_ratewall_input_manifest.json"
    )
    assert manifest["opening_portfolio_construction"]["tips_cpi_detail_source"] == (
        "sources/fiscaldata/tips_cpi_data_detail.csv"
    )
    assert manifest["opening_portfolio_construction"]["frn_daily_indexes_source"] == (
        "sources/fiscaldata/frn_daily_indexes.csv"
    )

    manifest_path = bundle.output_dir / "manifest.json"
    source_contract_path = bundle.inputs_dir / "source_contract_smoke.json"
    opening_portfolio_path = bundle.inputs_dir / "tdcsim_opening_portfolio.csv"
    tips_diagnostics_path = bundle.inputs_dir / "tdcsim_opening_tips_indexation_diagnostics.csv"
    frn_diagnostics_path = bundle.inputs_dir / "tdcsim_opening_frn_indexation_diagnostics.csv"
    rollforward_diagnostics_path = bundle.inputs_dir / "tdcsim_opening_rollforward_diagnostics.csv"
    assert manifest_path.exists()
    assert source_contract_path.exists()
    assert opening_portfolio_path.exists()
    assert tips_diagnostics_path.exists()
    assert frn_diagnostics_path.exists()
    assert rollforward_diagnostics_path.exists()
    assert set(bundle.initial_portfolio["SecurityType"].astype(str)) == {"Fixed", "TIPS", "FRN"}
    assert set(bundle.initial_portfolio["HolderType"].astype(str)) == {"Banks", "CB", "Foreign", "Private"}
    tips = bundle.initial_portfolio[bundle.initial_portfolio["SecurityType"].astype(str).eq("TIPS")]
    assert tips["IndexRatio"].min() > 1.0
    assert tips["OriginalPrincipal"].sum() < tips["AdjustedPrincipal"].sum()
    frn = bundle.initial_portfolio[bundle.initial_portfolio["SecurityType"].astype(str).eq("FRN")]
    assert frn["BenchmarkRate_FRN"].min() > 0.03
    assert frn["AccruedInterest_FRN"].sum() == pytest.approx(
        manifest["opening_portfolio_construction"]["opening_frn_accrued_interest_bil"]
    )
    assert bundle.params["tips_params"]["ref_cpi_lag_months"] == 3
    assert bundle.params["tips_params"]["reference_cpi_start_level"] == pytest.approx(
        manifest["opening_portfolio_construction"]["opening_tips_reference_cpi"]
    )

    surface = load_yield_curve_surface(
        bundle.inputs_dir / "tdcsim_yield_curve_surface.csv",
        actuals_available_as_of=ACTUALS_AVAILABLE_AS_OF,
    )
    assert set(surface["rate_unit"]) == {"percent_points"}
    assert set(surface["runtime_rate_unit"]) == {"decimal"}
    assert surface["nominal_rate_decimal"].max() < 0.10
    base_curve = surface[surface["curve_date"] == surface["curve_date"].min()]
    assert set(base_curve["base_curve_source_key"]) == {"official_treasury_daily_nominal_par_curve_2026_06_16"}


def test_runner_rate_unit_guardrails_reject_ambiguous_runtime_surface() -> None:
    good_row = {
        "nominal_rate": 3.70,
        "nominal_rate_decimal": 0.037,
        "rate_unit": "percent_points",
        "runtime_rate_unit": "decimal",
    }
    validate_yield_surface_rate_units([good_row])

    with pytest.raises(ValueError, match="runtime_rate_unit"):
        validate_yield_surface_rate_units([{**good_row, "runtime_rate_unit": "percent_points"}])

    with pytest.raises(ValueError, match="nominal_rate_decimal"):
        validate_yield_surface_rate_units([{**good_row, "nominal_rate_decimal": 3.70}])

    with pytest.raises(ValueError, match="nominal_rate_decimal"):
        bad_row = dict(good_row)
        bad_row.pop("nominal_rate_decimal")
        validate_yield_surface_rate_units([bad_row])


def test_runner_resolves_and_verifies_package_native_sources(tmp_path: Path) -> None:
    sources = tmp_path / "pkg" / "sources"
    files = {
        rel_path: rel_path.encode("utf-8")
        for rel_path in SOURCE_PACKAGE_INPUTS.values()
    }
    manifest_files = {}
    for rel_path, payload in files.items():
        path = sources / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest_files[rel_path] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    (sources / "source_manifest.json").write_text(
        json.dumps({"schema_version": "tdcsim_cbo_source_package_manifest_v1", "files": manifest_files}),
        encoding="utf-8",
    )

    resolved = resolve_source_package_inputs(tmp_path / "pkg")
    assert resolved["budget_workbook"] == sources / "cbo" / "51118-2026-02-Budget-Projections.xlsx"
    assert resolved["frn_daily_indexes"] == sources / "fiscaldata" / "frn_daily_indexes.csv"
    assert resolved["mspd_table_3_market"] == sources / "fiscaldata" / "mspd_table_3_market_2026-05-31.csv"
    assert resolved["treasury_base_curve"] == sources / "treasury" / "base_curve_2026-06-16.json"

    (sources / "fiscaldata" / "frn_daily_indexes.csv").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        resolve_source_package_inputs(sources)


def test_runner_source_parsers_require_exact_mts_rows(tmp_path: Path) -> None:
    table_1 = tmp_path / "mts_table_1.csv"
    table_1.write_text(
        "\n".join(
            [
                "record_date,classification_desc,line_code_nbr,current_fytd_dfct_sur_amt,current_month_dfct_sur_amt",
                "2026-05-31,Federal Surplus or Deficit,280,1246203266386.93,",
                "2026-05-31,Year-to-Date,280,,1246203266386.93",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert parse_mts_table_1_deficit_row(table_1)["classification_desc"] == "Year-to-Date"

    table_9 = tmp_path / "mts_table_9.csv"
    table_9.write_text(
        "\n".join(
            [
                "record_date,classification_desc,current_fytd_rcpt_outly_amt",
                "2026-05-31,Government Account Series,168566800000.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Net Interest"):
        parse_mts_table_9_net_interest_row(table_9)

    table_9.write_text(
        "\n".join(
            [
                "record_date,classification_desc,current_fytd_rcpt_outly_amt",
                "2026-05-31,Net Interest,722706511243.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert parse_mts_table_9_net_interest_row(table_9)["current_fytd_rcpt_outly_amt"] == "722706511243.20"


def test_runner_source_guardrail_requires_mts_table_9_net_interest_builder_contract() -> None:
    with pytest.raises(ValueError, match="Net Interest"):
        build_current_fy_splice_row_from_fiscaldata(
            scenario_id="baseline",
            fiscal_year=2026,
            simulation_start_date="2026-06-21",
            fiscal_actuals_through="2026-05-31",
            actuals_available_as_of=ACTUALS_AVAILABLE_AS_OF,
            cbo_full_fy_primary_deficit_bil=813.727,
            mts_table_1_row={
                "record_date": "2026-05-31",
                "classification_desc": "Year-to-Date",
                "line_code_nbr": "280",
                "current_month_dfct_sur_amt": "1246203266386.93",
            },
            mts_table_9_row={
                "record_date": "2026-05-31",
                "classification_desc": "Government Account Series",
                "current_fytd_rcpt_outly_amt": "168566800000.00",
            },
        )
