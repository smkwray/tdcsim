"""Producer-side ingest challenge for the RateWall export boundary."""

import csv
import json
import subprocess
from pathlib import Path

import pytest

from tdcsim_cbo._json import write_json
from tdcsim_cbo.consumer_challenge import (
    CHALLENGE_SCHEMA_VERSION,
    CHALLENGE_SCHEMA_VERSION_V1,
    AUDIT_ONLY_NON_SELECTED_FIELDS,
    CURRENT_SUMMARY_AUDIT_ONLY_FIELDS,
    CURRENT_SUMMARY_PROJECTION_FIELDS,
    EXPECTED_ABSENT_FIELDS,
    SELECTED_FIELD,
    build_handoff_package_manifest,
    build_ingest_challenge,
    collect_release_identity,
    read_ingest_challenge,
    verify_ingest_handoff,
)

CUMULATIVE_HEADER = [
    "period",
    "demand_conversion_case",
    SELECTED_FIELD,
    "delta_tdc_ex_overlap_interest_driven_excluded_bil",
    "tdc_materialized_deposit_stock_admissible_bil",
    "tdc_income_addendum_full_level_rate",
    "delta_tdc_ex_overlap_split_remainder_bil",
    "aggregation_reconciliation_status",
    "delta_tdc_ex_overlap_bil",
]
CURRENT_HEADER = [
    "period",
    "demand_conversion_case",
    SELECTED_FIELD,
    "delta_tdc_ex_overlap_interest_driven_excluded_bil",
    "tdc_materialized_deposit_stock_admissible_bil",
    "tdc_income_addendum_full_level_rate",
    "delta_tdc_ex_overlap_split_remainder_bil",
    "tdc_deposit_creation_split_schema_version",
    "tdc_income_addendum_admission_status",
    "tdc_income_addendum_collision_status",
    "delta_tdc_ex_overlap_bil",
    "legacy_chi_support_diagnostic_bil",
]


def _cumulative_rows(selected_2026="3.802346"):
    rows = []
    for year in range(2026, 2037):
        selected = selected_2026 if year == 2026 else f"{year - 2020}.125000"
        rows.append(
            {
                "period": str(year),
                "demand_conversion_case": "central",
                SELECTED_FIELD: selected,
                "delta_tdc_ex_overlap_interest_driven_excluded_bil": "1.542390",
                "tdc_materialized_deposit_stock_admissible_bil": "2.019246",
                "tdc_income_addendum_full_level_rate": "0.035",
                "delta_tdc_ex_overlap_split_remainder_bil": "0.0",
                "aggregation_reconciliation_status": "pass",
                "delta_tdc_ex_overlap_bil": "5.344736",
            }
        )
    return rows


def _current_rows():
    return [
        {
            "period": "2026",
            "demand_conversion_case": "central",
            SELECTED_FIELD: "3.802346",
            "delta_tdc_ex_overlap_interest_driven_excluded_bil": "1.542390",
            "tdc_materialized_deposit_stock_admissible_bil": "2.019246",
            "tdc_income_addendum_full_level_rate": "0.035",
            "delta_tdc_ex_overlap_split_remainder_bil": "0.0",
            "tdc_deposit_creation_split_schema_version": "tdc_deposit_creation_split_v1",
            "tdc_income_addendum_admission_status": "admitted_split_non_interest_bucket",
            "tdc_income_addendum_collision_status": "pass_split_collision_excluded",
            "delta_tdc_ex_overlap_bil": "5.344736",
            "legacy_chi_support_diagnostic_bil": "0.198432",
        }
    ]


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _producer_identity():
    return {
        "release_commit_sha": "a" * 40,
        "dirty_state": False,
        "runtime_identity_source": "clean_tracked_source_tree_files",
        "source_tree_sha256": "b" * 64,
        "source_tree_file_count": 10,
        "dependency_lock_files": [
            {"relative_path": "uv.lock", "sha256": "c" * 64, "bytes": 100}
        ],
        "dependency_lock_set_sha256": "d" * 64,
        "python_version": "3.13.5",
    }


def _package(tmp_path: Path, *, cumulative_rows=None, cumulative_header=None, current_header=None):
    root = tmp_path / "campaign"
    root.mkdir(parents=True)
    cumulative = _write_csv(
        root / "tdcsim_ratewall_cumulative_split_input.csv",
        cumulative_header or CUMULATIVE_HEADER,
        cumulative_rows or _cumulative_rows(),
    )
    pair_dir = root / "current_state_2026_plus_100bp_year_source_grade"
    pair_dir.mkdir()
    current = _write_csv(
        pair_dir / "tdcsim_ratewall_marginal_tdc_summary.csv",
        current_header or CURRENT_HEADER,
        _current_rows(),
    )
    pair_manifest = pair_dir / "tdcsim_ratewall_marginal_tdc_pair_manifest.json"
    write_json(
        pair_manifest,
        {
            "pair_id": "current_state_2026_plus_100bp_year_source_grade",
            "baseline_run": {"run_id": "baseline-run", "manifest_sha256": "e" * 64},
            "shock_run": {"run_id": "shock-run", "manifest_sha256": "f" * 64},
        },
    )
    pair_index = root / "tdcsim_ratewall_split_pair_index.csv"
    _write_csv(
        pair_index,
        ["pair_id", "verification_status"],
        [{"pair_id": "current_state_2026_plus_100bp_year_source_grade", "verification_status": "pass"}],
    )
    package_manifest = root / "tdcsim_ratewall_handoff_package_manifest.json"
    write_json(
        package_manifest,
        build_handoff_package_manifest(
            root,
            campaign_id="campaign-1",
            pair_index_path=pair_index,
            pair_manifest_paths=[pair_manifest],
            cumulative_csv=cumulative,
            current_summary_csv=current,
            producer_identity=_producer_identity(),
        ),
    )
    return root, cumulative, current, package_manifest


def _challenge(tmp_path, **kwargs):
    root, cumulative, current, package_manifest = _package(tmp_path, **kwargs)
    challenge = build_ingest_challenge(
        cumulative,
        current,
        package_manifest_path=package_manifest,
    )
    return root, cumulative, current, package_manifest, challenge


def test_v2_binds_both_files_full_projection_headers_counts_and_lineage(tmp_path):
    _, cumulative, current, _, challenge = _challenge(tmp_path)

    assert challenge["schema_version"] == CHALLENGE_SCHEMA_VERSION
    assert challenge["expected_selected_key_count"] == 11
    assert len(challenge["input_files"]) == 2
    assert {item["source_file_name"] for item in challenge["input_files"]} == {
        cumulative.name,
        current.name,
    }
    cumulative_record = challenge["input_files"][0]
    assert cumulative_record["pre_filter_row_count"] == 11
    assert cumulative_record["post_filter_row_count"] == 11
    assert cumulative_record["actual_header"] == CUMULATIVE_HEADER
    assert len(cumulative_record["header_digest_sha256"]) == 64
    assert "aggregation_reconciliation_status" in cumulative_record["required_projection_fields"]
    assert challenge["lineage"]["pair_manifest_identities"][0]["source_runs"][0]["run_id"] == "baseline-run"
    assert len(challenge["lineage"]["package_manifest_sha256"]) == 64
    assert challenge["producer_identity"]["dirty_state"] is False


def test_missing_gate_field_is_rejected(tmp_path):
    header = [field for field in CUMULATIVE_HEADER if field != "aggregation_reconciliation_status"]
    with pytest.raises(ValueError, match="missing required fields.*aggregation_reconciliation_status"):
        _challenge(tmp_path, cumulative_header=header)


def test_duplicate_canonical_key_is_rejected(tmp_path):
    rows = _cumulative_rows()
    rows[-1]["period"] = rows[-2]["period"]
    with pytest.raises(ValueError, match="must be unique"):
        _challenge(tmp_path, cumulative_rows=rows)


def test_blank_canonical_key_is_rejected(tmp_path):
    rows = _cumulative_rows()
    rows[-1]["period"] = "   "
    with pytest.raises(ValueError, match="must be nonblank"):
        _challenge(tmp_path, cumulative_rows=rows)


def test_expected_absent_and_audit_only_fields_are_distinct_invariants(tmp_path):
    header = [*CUMULATIVE_HEADER, EXPECTED_ABSENT_FIELDS[0]]
    rows = _cumulative_rows()
    for row in rows:
        row[EXPECTED_ABSENT_FIELDS[0]] = "0"
    with pytest.raises(ValueError, match="required to be absent"):
        _challenge(tmp_path, cumulative_header=header, cumulative_rows=rows)

    header_without_audit = [field for field in CUMULATIVE_HEADER if field != "delta_tdc_ex_overlap_bil"]
    with pytest.raises(ValueError, match="missing required fields.*delta_tdc_ex_overlap_bil"):
        _challenge(tmp_path / "second", cumulative_header=header_without_audit)


def test_retired_diagnostic_is_audit_only_not_required_absent(tmp_path):
    """The retired beta-chi diagnostic ships deliberately; requiring it absent destroys evidence.

    `legacy_chi_support_diagnostic_bil` was originally classified as required-absent, which
    failed the real campaign: the per-pair summary the fallback reads does carry it, exported
    under an explicitly retired name and paired with
    `legacy_chi_support_eligible_for_main_ratio=False` that the contract validator enforces.
    That pairing is what makes the retirement auditable rather than silent, and the consumer's
    parser references neither name. What must stay absent is the *neutral* name a consumer
    could mistake for the headline figure.

    The two consumed files also differ: the cumulative table is rebuilt from a fixed column
    list that omits the diagnostic, so the requirement is per file.
    """

    assert "marginal_tdc_support_bil" in EXPECTED_ABSENT_FIELDS
    assert "legacy_chi_support_diagnostic_bil" not in EXPECTED_ABSENT_FIELDS
    assert "legacy_chi_support_diagnostic_bil" in CURRENT_SUMMARY_AUDIT_ONLY_FIELDS
    assert "legacy_chi_support_diagnostic_bil" not in AUDIT_ONLY_NON_SELECTED_FIELDS

    # It is never selectable, whichever list it sits in.
    assert "legacy_chi_support_diagnostic_bil" != SELECTED_FIELD
    assert "legacy_chi_support_diagnostic_bil" not in CURRENT_SUMMARY_PROJECTION_FIELDS

    # Dropping it from the fallback file must fail: its presence is the retirement evidence.
    header = [f for f in CURRENT_HEADER if f != "legacy_chi_support_diagnostic_bil"]
    with pytest.raises(ValueError, match="missing required fields.*legacy_chi_support_diagnostic_bil"):
        _challenge(tmp_path, current_header=header)


def test_digest_is_over_original_decimal_strings_not_parsed_floats(tmp_path):
    assert float("3.8023460000000001") == float("3.8023460000000002")
    _, _, _, _, a = _challenge(tmp_path / "a", cumulative_rows=_cumulative_rows("3.8023460000000001"))
    _, _, _, _, b = _challenge(tmp_path / "b", cumulative_rows=_cumulative_rows("3.8023460000000002"))

    assert a["selected_projection_digest_sha256"] != b["selected_projection_digest_sha256"]


def test_last_decimal_drift_in_a_supporting_field_changes_projection_digest(tmp_path):
    rows_a = _cumulative_rows()
    rows_b = _cumulative_rows()
    rows_b[0]["tdc_income_addendum_full_level_rate"] = "0.0350000000000001"
    _, _, _, _, a = _challenge(tmp_path / "a", cumulative_rows=rows_a)
    _, _, _, _, b = _challenge(tmp_path / "b", cumulative_rows=rows_b)

    assert a["selected_projection_digest_sha256"] != b["selected_projection_digest_sha256"]


def test_dirty_tree_release_identity_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("lock\n", encoding="utf-8")
    (repo / "producer.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "uv.lock", "producer.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    assert collect_release_identity(repo)["dirty_state"] is False

    (repo / "uncommitted.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty_state == false"):
        collect_release_identity(repo)


def test_mutation_after_challenge_is_rejected_before_handoff(tmp_path):
    root, cumulative, _, _, challenge = _challenge(tmp_path)
    write_json(root / "tdcsim_ratewall_ingest_challenge.json", challenge)
    rows = _cumulative_rows("3.802347")
    _write_csv(cumulative, CUMULATIVE_HEADER, rows)

    with pytest.raises(ValueError, match="differs from recomputed packaged inputs"):
        verify_ingest_handoff(root)


def test_verified_handoff_manifest_contains_both_files_and_challenge_hash(tmp_path):
    root, _, _, _, challenge = _challenge(tmp_path)
    challenge_path = root / "tdcsim_ratewall_ingest_challenge.json"
    write_json(challenge_path, challenge)

    result = verify_ingest_handoff(root)

    assert result["status"] == "pass"
    manifest = result["manifest"]
    assert len(manifest["consumed_inputs"]) == 2
    assert manifest["ingest_challenge"]["relative_path"] == challenge_path.name
    assert len(manifest["ingest_challenge"]["sha256"]) == 64
    assert len(result["transfer_manifest_sha256"]) == 64


def test_v1_challenge_remains_readable(tmp_path):
    path = tmp_path / "challenge-v1.json"
    path.write_text(json.dumps({"schema_version": CHALLENGE_SCHEMA_VERSION_V1}), encoding="utf-8")

    assert read_ingest_challenge(path)["schema_version"] == CHALLENGE_SCHEMA_VERSION_V1
