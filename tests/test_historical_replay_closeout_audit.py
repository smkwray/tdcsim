from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from historical_replay import _build_replay_config_hash

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts" / "audit_historical_replay_closeout.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("historical_replay_closeout_audit", AUDIT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_required_artifact_status_fails_same_size_content_mutation(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(audit, "REQUIRED_ARTIFACTS", {"required.csv"})

    required = tmp_path / "required.csv"
    required.write_text("a,b\n1,2\n", encoding="utf-8")
    rel = audit._rel(required)
    manifest = pd.DataFrame(
        [
            {
                "artifact": "required",
                "path": rel,
                "status": "present",
                "bytes": required.stat().st_size,
                "sha256": _sha256(required),
                "row_count": 1,
                "column_count": 2,
                "header_sha256": hashlib.sha256("a,b".encode("utf-8")).hexdigest(),
                "run_id": "run123",
                "config_sha256": "cfg123",
            }
        ]
    )

    required.write_text("a,b\n9,2\n", encoding="utf-8")
    assert required.stat().st_size == int(manifest.iloc[0]["bytes"])

    status, observed, _ = audit._required_artifact_status(manifest)

    assert status == "fail"
    assert "sha256_mismatch" in observed


def test_required_artifact_status_fails_markdown_mutation(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(audit, "REQUIRED_ARTIFACTS", {"historical_replay_acceptance.md"})

    required = tmp_path / "historical_replay_acceptance.md"
    required.write_text("# Accepted\n", encoding="utf-8")
    rel = audit._rel(required)
    manifest = pd.DataFrame(
        [
            {
                "artifact": "historical_replay_acceptance",
                "path": rel,
                "status": "present",
                "bytes": required.stat().st_size,
                "sha256": _sha256(required),
                "row_count": pd.NA,
                "column_count": pd.NA,
                "header_sha256": pd.NA,
                "run_id": "run123",
                "config_sha256": "cfg123",
                "code_sha256": "code123",
            }
        ]
    )

    required.write_text("# Rejected\n", encoding="utf-8")
    status, observed, _ = audit._required_artifact_status(manifest)

    assert status == "fail"
    assert "sha256_mismatch" in observed


def test_required_input_status_recomputes_live_hash(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "EXPECTED_REQUIRED_INPUT_KEYS", {"input"})

    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {
                "source_key": "input",
                "path": "input.csv",
                "required": True,
                "required_for_claim": True,
                "source_vintage": "not_applicable",
                "status": "present",
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "row_count": 1,
            }
        ]
    )
    source.write_text("a,b\n9,2\n", encoding="utf-8")

    status, observed, _ = audit._required_input_status(manifest)

    assert status == "fail"
    assert "sha256_mismatch" in observed


def test_required_input_status_fails_required_flag_demotion(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "EXPECTED_REQUIRED_INPUT_KEYS", {"input"})

    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {
                "source_key": "input",
                "path": "input.csv",
                "required": False,
                "required_for_claim": False,
                "status": "present",
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "row_count": 1,
            }
        ]
    )

    status, observed, _ = audit._required_input_status(manifest)

    assert status == "fail"
    assert "demoted_required_for_claim" in observed


def test_required_artifact_status_fails_live_code_hash_mismatch(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(audit, "REQUIRED_ARTIFACTS", {"required.csv"})
    monkeypatch.setattr(audit, "_build_code_identity_hash", lambda: "live-code")

    required = tmp_path / "required.csv"
    required.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {
                "artifact": "required",
                "path": audit._rel(required),
                "status": "present",
                "bytes": required.stat().st_size,
                "sha256": _sha256(required),
                "row_count": 1,
                "column_count": 2,
                "header_sha256": hashlib.sha256("a,b".encode("utf-8")).hexdigest(),
                "run_id": "run123",
                "config_sha256": "cfg123",
                "code_sha256": "stale-code",
            }
        ]
    )

    status, observed, _ = audit._required_artifact_status(manifest)

    assert status == "fail"
    assert "live_code_sha256_mismatch" in observed


def test_artifact_integrity_manifest_status_fails_central_output_mutation(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "_build_code_identity_hash", lambda: "live-code")
    monkeypatch.setattr(audit, "_live_replay_config_hash", lambda: "live-config")

    ledger = tmp_path / "historical_replay_ledger.csv"
    ledger.write_text("quarter,debt_surface_role\n2025Q4,cell\n", encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {
                "artifact": "historical_replay_ledger",
                "path": "historical_replay_ledger.csv",
                "status": "present",
                "bytes": ledger.stat().st_size,
                "sha256": _sha256(ledger),
                "row_count": 1,
                "column_count": 2,
                "header_sha256": hashlib.sha256("quarter,debt_surface_role".encode("utf-8")).hexdigest(),
                "run_id": "run123",
                "config_sha256": "live-config",
                "code_sha256": "live-code",
            }
        ]
    )
    ledger.write_text("quarter,debt_surface_role\n2025Q4,stale\n", encoding="utf-8")

    status, observed, _ = audit._artifact_integrity_manifest_status(manifest)

    assert status == "fail"
    assert "historical_replay_ledger:sha256_mismatch" in observed


def test_artifact_integrity_manifest_status_fails_stale_config_identity(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "_build_code_identity_hash", lambda: "live-code")
    monkeypatch.setattr(audit, "_live_replay_config_hash", lambda: "live-config")

    results = tmp_path / "historical_replay_results.csv"
    results.write_text("date,TOC_Level\n2025-12-31,1\n", encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {
                "artifact": "historical_replay_results",
                "path": "historical_replay_results.csv",
                "status": "present",
                "bytes": results.stat().st_size,
                "sha256": _sha256(results),
                "row_count": 1,
                "column_count": 2,
                "header_sha256": hashlib.sha256("date,TOC_Level".encode("utf-8")).hexdigest(),
                "run_id": "run123",
                "config_sha256": "stale-config",
                "code_sha256": "live-code",
            }
        ]
    )

    status, observed, _ = audit._artifact_integrity_manifest_status(manifest)

    assert status == "fail"
    assert "live_config_sha256_mismatch" in observed


def test_replay_config_hash_canonicalizes_project_absolute_paths():
    manifest = pd.DataFrame([{"source_key": "input", "path": "data/input.csv"}])
    absolute_cfg = {
        "output_dir": str(ROOT / "data/historical_replay/validation"),
        "paths": {"cash": str(ROOT / "data/input.csv")},
    }
    relative_cfg = {
        "output_dir": "data/historical_replay/validation",
        "paths": {"cash": "data/input.csv"},
    }

    assert _build_replay_config_hash(
        absolute_cfg,
        start_quarter="2001Q1",
        end_quarter="2025Q4",
        replay_input_manifest=manifest,
    ) == _build_replay_config_hash(
        relative_cfg,
        start_quarter="2001Q1",
        end_quarter="2025Q4",
        replay_input_manifest=manifest,
    )


def test_acceptance_artifact_status_fails_stale_live_count(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "VALIDATION_DIR", tmp_path)
    (tmp_path / "historical_replay_acceptance.md").write_text(
        "This artifact is generated from live validation CSVs\n"
        "Replay input manifest rows: `99` total, `99` required\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"source_key": "a", "required_for_claim": True}, {"source_key": "b", "required_for_claim": False}]
    ).to_csv(tmp_path / "historical_replay_input_manifest.csv", index=False)
    pd.DataFrame([{"artifact": "x", "status": "present"}]).to_csv(tmp_path / "artifact_integrity.csv", index=False)
    pd.DataFrame([{"diagnostic_type": "solver", "solver_method": "exact_weighted_entropy_projection"}]).to_csv(
        tmp_path / "historical_replay_diagnostics.csv", index=False
    )
    pd.DataFrame([{"interval_quarters": 1, "high_tv_transition": False}]).to_csv(
        tmp_path / "portfolio_transition_diagnostics.csv", index=False
    )
    pd.DataFrame([{"sample_row_count": 1}]).to_csv(
        tmp_path / "historical_replay_large_artifact_sample_manifest.csv", index=False
    )
    pd.DataFrame([{"source_sector": "SourceBasisResidual"}]).to_csv(
        tmp_path / "historical_replay_final_portfolio.csv", index=False
    )

    status, observed, _ = audit._acceptance_artifact_status()

    assert status == "fail"
    assert "Replay input manifest rows" in observed


def test_source_role_consistency_fails_z1_lineage_marked_runtime(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(audit, "SOURCE_SCOPE_PATH", tmp_path / "source_scope.csv")
    pd.DataFrame(
        [
            {
                "source_key": "z1_raw_archive",
                "path": "data/z1.zip",
                "consumed_in_run": True,
                "required_for_claim": True,
                "source_usage": "runtime_consumed",
            }
        ]
    ).to_csv(tmp_path / "historical_replay_input_manifest.csv", index=False)
    pd.DataFrame(
        [{"source_path": "data/z1.zip", "closeout_role": "runtime_consumed"}]
    ).to_csv(tmp_path / "source_scope.csv", index=False)

    status, observed, _ = audit._source_role_consistency_status()

    assert status == "fail"
    assert "z1_raw_archive" in observed


def test_audit_main_returns_nonzero_when_required_check_fails(tmp_path, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "VALIDATION_DIR", tmp_path)
    monkeypatch.setattr(audit, "SOURCE_SCOPE_PATH", tmp_path / "source_scope.csv")
    monkeypatch.setattr(audit, "PLAUSIBILITY_PATH", tmp_path / "plausibility.csv")
    monkeypatch.setattr(audit, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(audit, "FINAL_EVIDENCE_MANIFEST_PATH", tmp_path / "final_manifest.csv")

    monkeypatch.setattr(audit, "build_source_scope_audit", lambda: pd.DataFrame())
    monkeypatch.setattr(
        audit,
        "build_plausibility_audit",
        lambda: pd.DataFrame(
            [
                {
                    "check": "required_artifacts_live",
                    "status": "fail",
                    "observed": "missing=['required.csv']",
                    "expected": "all required artifacts live",
                    "notes": "fixture",
                }
            ]
        ),
    )
    monkeypatch.setattr(audit, "write_report", lambda source_scope, plausibility: None)
    monkeypatch.setattr(audit, "write_final_evidence_manifest", lambda: None)

    assert audit.main() == 1
