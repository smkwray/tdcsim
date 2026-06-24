"""Deterministic CBO scenario compiler.

The compiler materializes a verified immutable baseline into a work directory,
copies its forecast inputs byte-for-byte, applies a sparse scenario overlay, and
writes a manifest with canonical input digests. It does not invoke the simulator.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json import canonical_json_sha256, sha256_file, write_json
from .baseline import CboBaselinePackage
from .contract import CboScenarioSpec
from .transforms.fiscal import (
    apply_cash_residual_override,
    apply_debt_target_override,
    apply_fiscal_incidence_override,
    apply_operating_cash_override,
    apply_primary_deficit_override,
)
from .transforms.portfolio import (
    apply_fed_holdings_override,
    compile_issuance_mix_override,
    validate_holder_preferences,
)
from .transforms.rates import (
    apply_cpi_override,
    apply_frn_override,
    apply_nominal_yield_curve_override,
    apply_tips_real_yield_override,
)


FORECAST_INPUTS = "forecast_inputs"
COMPILED_MANIFEST = "tdcsim_cbo_compiled_manifest.json"
ISSUANCE_MIX_FILE = "tdcsim_issuance_mix_assumptions.json"

INPUT_FILES = {
    "nominal_yield_curve": "tdcsim_yield_curve_surface.csv",
    "frn_benchmark": "tdcsim_frn_rate_path.csv",
    "inflation_cpi": "tdcsim_tips_cpi_path.csv",
    "tips_real_yield": "tdcsim_tips_real_yield_path.csv",
    "operating_cash": "tdcsim_operating_cash_path.csv",
    "cash_reconciliation": "tdcsim_cash_reconciliation_residual.csv",
    "primary_deficit": "tdcsim_primary_deficit_path.csv",
    "debt_target": "tdcsim_debt_stock_path.csv",
    "fed_holdings": "tdcsim_fed_holdings_path.csv",
    "fiscal_incidence": "tdcsim_fiscal_incidence_policy.csv",
}


class CompilerError(ValueError):
    """Raised when a CBO scenario cannot be compiled safely."""


@dataclass(frozen=True)
class CboCompiledScenario:
    """A compiled scenario input package."""

    work_dir: Path
    baseline_dir: Path
    compiled_dir: Path
    forecast_inputs_dir: Path
    manifest_path: Path
    baseline_forecast_inputs_digest: str
    compiled_inputs_digest: str
    scenario_sha256: str
    changed_inputs: tuple[str, ...]
    manifest: Mapping[str, Any]


class CboScenarioCompiler:
    """Compile sparse CBO scenario overlays into TDCSIM forecast inputs."""

    def compile(
        self,
        baseline: CboBaselinePackage,
        spec: CboScenarioSpec,
        work_dir: str | Path,
    ) -> CboCompiledScenario:
        if not baseline.is_zip:
            raise CompilerError("CBO scenario compiler requires a release-bound zip baseline package")
        spec.assert_baseline_matches(baseline)
        scenario = spec.data
        overrides = _overrides(scenario)
        coupling = _coupling(scenario)
        _validate_override_coupling(overrides, coupling)

        root = Path(work_dir).expanduser().resolve()
        baseline_dir = root / "baseline"
        compiled_dir = root / "compiled"
        forecast_inputs_dir = compiled_dir / FORECAST_INPUTS
        manifest_path = compiled_dir / COMPILED_MANIFEST
        _prepare_work_dir(root, baseline_dir, compiled_dir)

        original_package_sha = sha256_file(baseline.package_path) if baseline.is_zip else None
        materialized = baseline.materialize(baseline_dir)
        shutil.copytree(materialized / FORECAST_INPUTS, forecast_inputs_dir)

        baseline_digest = digest_input_tree(materialized / FORECAST_INPUTS)
        changed = _apply_overrides(forecast_inputs_dir, spec, overrides, coupling)
        compiled_digest = digest_input_tree(forecast_inputs_dir)
        if original_package_sha is not None and sha256_file(baseline.package_path) != original_package_sha:
            raise CompilerError("baseline package bytes changed during compilation")

        manifest = {
            "schema_version": "tdcsim_cbo_compiled_scenario_manifest_v1",
            "scenario_id": spec.scenario_id,
            "scenario_sha256": spec.canonical_sha256(),
            "baseline": {
                "package_id": baseline.package_id,
                "package_sha256": baseline.package_sha256,
                "manifest_sha256": baseline.manifest_sha256,
                "release_attestation_sha256": baseline.attestation.sha256,
            },
            "baseline_forecast_inputs_digest": baseline_digest,
            "compiled_inputs_digest": compiled_digest,
            "changed_inputs": sorted(changed),
            "overrides_applied": sorted(overrides),
            "coupling": dict(coupling),
            "claim_boundary": {
                "compiler_role": "scenario_input_overlay_only",
                "does_not_run_engine": True,
                "does_not_modify_baseline_package": True,
                "net_interest_role": "diagnostic_nonbinding",
                "fed_holdings_role": "holder_allocation_target_not_total_issuance",
                "operating_cash_role": "cash_path_not_issuance_plug",
            },
            "input_hashes": input_tree_hashes(forecast_inputs_dir),
        }
        write_json(manifest_path, manifest)
        return CboCompiledScenario(
            work_dir=root,
            baseline_dir=baseline_dir,
            compiled_dir=compiled_dir,
            forecast_inputs_dir=forecast_inputs_dir,
            manifest_path=manifest_path,
            baseline_forecast_inputs_digest=baseline_digest,
            compiled_inputs_digest=compiled_digest,
            scenario_sha256=spec.canonical_sha256(),
            changed_inputs=tuple(sorted(changed)),
            manifest=manifest,
        )


def digest_input_tree(path: str | Path) -> str:
    """Return a canonical digest of all files under a forecast-input tree."""

    return canonical_json_sha256(input_tree_hashes(path))


def input_tree_hashes(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    records = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append(
            {
                "path": file_path.relative_to(root).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return records


def _prepare_work_dir(root: Path, baseline_dir: Path, compiled_dir: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    blockers = [path for path in (baseline_dir, compiled_dir) if path.exists()]
    if blockers:
        raise CompilerError(f"compile work directory already contains generated paths: {[str(path) for path in blockers]}")


def _overrides(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    overrides = scenario.get("overrides")
    if not isinstance(overrides, Mapping):
        raise CompilerError("scenario overrides block must be a mapping")
    return overrides


def _coupling(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    coupling = scenario.get("coupling")
    if not isinstance(coupling, Mapping):
        raise CompilerError("scenario coupling block must be a mapping")
    return coupling


def _apply_overrides(
    forecast_inputs_dir: Path,
    spec: CboScenarioSpec,
    overrides: Mapping[str, Any],
    coupling: Mapping[str, Any],
) -> set[str]:
    changed: set[str] = set()

    def apply_csv(name: str, transform: Callable[[list[dict[str, str]], Mapping[str, Any]], list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
        file_name = INPUT_FILES[name]
        rows, header = _read_csv(forecast_inputs_dir / file_name)
        output = [dict(row) for row in transform(rows, _override_mapping(overrides[name]))]
        _write_csv(forecast_inputs_dir / file_name, output, preferred_header=header)
        changed.add(file_name)
        return output

    nominal_rows: list[dict[str, Any]] | None = None
    cpi_rows: list[dict[str, Any]] | None = None
    operating_cash_rows: list[dict[str, Any]] | None = None
    debt_rows: list[dict[str, Any]] | None = None
    fed_active = _fed_stock_target_present(forecast_inputs_dir / INPUT_FILES["fed_holdings"])

    if "nominal_yield_curve" in overrides:
        override = _override_mapping(overrides["nominal_yield_curve"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["nominal_yield_curve"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        nominal_rows = apply_nominal_yield_curve_override(rows, override, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["nominal_yield_curve"], nominal_rows, preferred_header=header)
        changed.add(INPUT_FILES["nominal_yield_curve"])
    elif _needs_nominal_rows(overrides, coupling):
        nominal_rows, _ = _read_csv(forecast_inputs_dir / INPUT_FILES["nominal_yield_curve"])

    if "inflation_cpi" in overrides:
        override = _override_mapping(overrides["inflation_cpi"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["inflation_cpi"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        cpi_rows = apply_cpi_override(rows, override, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["inflation_cpi"], cpi_rows, preferred_header=header)
        changed.add(INPUT_FILES["inflation_cpi"])
    elif coupling.get("operating_cash_inflation") == "scenario_cpi":
        cpi_rows, _ = _read_csv(forecast_inputs_dir / INPUT_FILES["inflation_cpi"])

    if "frn_benchmark" in overrides:
        override = _override_mapping(overrides["frn_benchmark"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["frn_benchmark"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_frn_override(rows, override, nominal_curve_rows=nominal_rows, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["frn_benchmark"], output, preferred_header=header)
        changed.add(INPUT_FILES["frn_benchmark"])

    if "tips_real_yield" in overrides:
        override = _override_mapping(overrides["tips_real_yield"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["tips_real_yield"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_tips_real_yield_override(
            rows,
            override,
            nominal_curve_rows=nominal_rows,
            cpi_rows=cpi_rows,
            replacement_rows=replacement,
        )
        _write_csv(forecast_inputs_dir / INPUT_FILES["tips_real_yield"], output, preferred_header=header)
        changed.add(INPUT_FILES["tips_real_yield"])
    elif coupling.get("tips_real_yield") == "recompute_from_nominal_and_scenario_inflation" and (
        "inflation_cpi" in overrides or "nominal_yield_curve" in overrides
    ):
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["tips_real_yield"])
        output = apply_tips_real_yield_override(
            rows,
            {"mode": "linked_recompute"},
            nominal_curve_rows=nominal_rows,
            cpi_rows=cpi_rows,
        )
        _write_csv(forecast_inputs_dir / INPUT_FILES["tips_real_yield"], output, preferred_header=header)
        changed.add(INPUT_FILES["tips_real_yield"])

    if "operating_cash" in overrides:
        override = _override_mapping(overrides["operating_cash"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["operating_cash"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        operating_cash_rows = apply_operating_cash_override(
            rows,
            override,
            inflation_rows=cpi_rows if coupling.get("operating_cash_inflation") == "scenario_cpi" else None,
            replacement_rows=replacement,
        )
        _write_csv(forecast_inputs_dir / INPUT_FILES["operating_cash"], operating_cash_rows, preferred_header=header)
        changed.add(INPUT_FILES["operating_cash"])

    if "cash_reconciliation" in overrides:
        override = _override_mapping(overrides["cash_reconciliation"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["cash_reconciliation"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_cash_residual_override(rows, override, operating_cash_rows=operating_cash_rows, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["cash_reconciliation"], output, preferred_header=header)
        changed.add(INPUT_FILES["cash_reconciliation"])

    if "primary_deficit" in overrides:
        override = _override_mapping(overrides["primary_deficit"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["primary_deficit"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_primary_deficit_override(rows, override, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["primary_deficit"], output, preferred_header=header)
        changed.add(INPUT_FILES["primary_deficit"])

    if "debt_target" in overrides:
        override = _override_mapping(overrides["debt_target"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["debt_target"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        debt_rows = apply_debt_target_override(rows, override, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["debt_target"], debt_rows, preferred_header=header)
        changed.add(INPUT_FILES["debt_target"])
    elif "fed_holdings" in overrides:
        debt_rows, _ = _read_csv(forecast_inputs_dir / INPUT_FILES["debt_target"])

    if "fed_holdings" in overrides:
        override = _override_mapping(overrides["fed_holdings"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["fed_holdings"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_fed_holdings_override(rows, override, marketable_debt_rows=debt_rows, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["fed_holdings"], output, preferred_header=header)
        changed.add(INPUT_FILES["fed_holdings"])

    if "fiscal_incidence" in overrides:
        apply_csv("fiscal_incidence", apply_fiscal_incidence_override)

    if "holder_preferences" in overrides:
        rows, header = _read_csv(forecast_inputs_dir / "tdcsim_holder_profile_assumptions.csv")
        output = _compile_holder_preferences(rows, header, _override_mapping(overrides["holder_preferences"]), fed_stock_target_active=fed_active)
        _write_csv(forecast_inputs_dir / "tdcsim_holder_profile_assumptions.csv", output, preferred_header=header)
        changed.add("tdcsim_holder_profile_assumptions.csv")

    if "issuance_mix" in overrides:
        issuance_mix = compile_issuance_mix_override(_override_mapping(overrides["issuance_mix"]))
        _write_issuance_mix(forecast_inputs_dir / ISSUANCE_MIX_FILE, issuance_mix)
        changed.add(ISSUANCE_MIX_FILE)

    if "net_interest_comparator" in overrides:
        net_interest = _override_mapping(overrides["net_interest_comparator"])
        if net_interest.get("role") != "diagnostic_nonbinding":
            raise CompilerError("net_interest_comparator role must remain diagnostic_nonbinding")

    return changed


def _validate_override_coupling(overrides: Mapping[str, Any], coupling: Mapping[str, Any]) -> None:
    for override_name, override in overrides.items():
        if isinstance(override, Mapping) and "file" in override and not _is_file_mode(str(override.get("mode") or "")):
            raise CompilerError(f"{override_name} file reference is only allowed for file-backed modes")
    frn = overrides.get("frn_benchmark")
    if isinstance(frn, Mapping):
        mode = frn.get("mode")
        if mode == "linked_to_nominal_curve" and coupling.get("frn_benchmark") != "derive_from_scenario_nominal_curve":
            raise CompilerError("linked FRN benchmark requires derive_from_scenario_nominal_curve coupling")
        if mode in {"parallel_bp", "absolute_path_file"} and coupling.get("frn_benchmark") != "independent_explicit_path":
            raise CompilerError("independent FRN benchmark overrides require independent_explicit_path coupling")
    tips = overrides.get("tips_real_yield")
    if isinstance(tips, Mapping):
        mode = tips.get("mode")
        if mode == "linked_recompute" and coupling.get("tips_real_yield") != "recompute_from_nominal_and_scenario_inflation":
            raise CompilerError("linked TIPS real-yield recompute requires recompute_from_nominal_and_scenario_inflation coupling")
        if mode in {"parallel_bp", "key_rate_bp", "absolute_path_file"} and coupling.get("tips_real_yield") != "independent_explicit_path":
            raise CompilerError("independent TIPS real-yield overrides require independent_explicit_path coupling")
    operating_cash = overrides.get("operating_cash")
    if isinstance(operating_cash, Mapping) and operating_cash.get("mode") == "constant_real":
        if coupling.get("operating_cash_inflation") not in {"baseline_cpi", "scenario_cpi", "independent_explicit_path"}:
            raise CompilerError("constant_real operating cash requires an explicit operating_cash_inflation coupling")
    if coupling.get("primary_deficit_to_debt_target") != "independent_no_plug":
        raise CompilerError("primary_deficit_to_debt_target must remain independent_no_plug")


def _is_file_mode(mode: str) -> bool:
    return mode in {
        "full_surface_file",
        "absolute_path_file",
        "monthly_path_file",
        "aggregate_path_file",
        "component_path_file",
        "explicit_path_file",
        "comparison_path_file",
    }


def _needs_nominal_rows(overrides: Mapping[str, Any], coupling: Mapping[str, Any]) -> bool:
    frn = overrides.get("frn_benchmark")
    tips = overrides.get("tips_real_yield")
    return (
        isinstance(frn, Mapping)
        and frn.get("mode") == "linked_to_nominal_curve"
        or isinstance(tips, Mapping)
        and tips.get("mode") == "linked_recompute"
        or coupling.get("frn_benchmark") == "derive_from_scenario_nominal_curve"
        and "frn_benchmark" in overrides
    )


def _override_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompilerError("override value must be a mapping")
    return value


def _csv_file_override_rows(
    spec: CboScenarioSpec,
    override: Mapping[str, Any],
    *,
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, str]] | None:
    file_ref = override.get("file")
    if file_ref is None:
        return None
    if not _is_file_mode(str(override.get("mode") or "")):
        raise CompilerError("file references require a file-backed override mode")
    if not isinstance(file_ref, Mapping):
        raise CompilerError("override file reference must be a mapping")
    if spec.path is None:
        raise CompilerError("file-based overrides require a scenario file path")
    rel = str(file_ref["relative_path"])
    path = (spec.path.parent / rel).resolve()
    if spec.path.parent.resolve() not in path.parents and path != spec.path.parent.resolve():
        raise CompilerError("scenario file reference escapes scenario directory")
    expected_sha = str(file_ref["sha256"])
    if sha256_file(path) != expected_sha:
        raise CompilerError(f"scenario file SHA-256 mismatch: {rel}")
    if file_ref.get("media_type", "text/csv") != "text/csv":
        raise CompilerError("compiler currently supports CSV file overrides only")
    rows, _ = _read_csv(path)
    _assert_replacement_coverage(rows, baseline_rows, mode=str(override.get("mode") or ""))
    return rows


def _assert_replacement_coverage(
    replacement_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    *,
    mode: str,
) -> None:
    if not replacement_rows:
        raise CompilerError(f"{mode} replacement file has no rows")
    if baseline_rows and len(replacement_rows) != len(baseline_rows):
        raise CompilerError(
            f"{mode} replacement row count must match baseline coverage: "
            f"{len(replacement_rows)} != {len(baseline_rows)}"
        )
    key_cols = _replacement_key_columns(baseline_rows, replacement_rows)
    if key_cols:
        baseline_keys = _unique_keys(baseline_rows, key_cols, label="baseline")
        replacement_keys = _unique_keys(replacement_rows, key_cols, label="replacement")
        if baseline_keys != replacement_keys:
            missing = sorted(baseline_keys - replacement_keys)[:5]
            extra = sorted(replacement_keys - baseline_keys)[:5]
            raise CompilerError(
                f"{mode} replacement key coverage mismatch for {key_cols}: "
                f"missing={missing}, extra={extra}"
            )


def _replacement_key_columns(
    baseline_rows: list[dict[str, str]],
    replacement_rows: list[dict[str, str]],
) -> tuple[str, ...]:
    if not baseline_rows or not replacement_rows:
        return ()
    baseline_cols = set(baseline_rows[0])
    replacement_cols = set(replacement_rows[0])
    for cols in (
        ("curve_date", "tenor_years"),
        ("period_start", "period_end"),
        ("period_end", "holder_type"),
        ("period_end",),
        ("month",),
        ("source_fiscal_year",),
        ("fiscal_year",),
    ):
        if set(cols) <= baseline_cols and set(cols) <= replacement_cols:
            return cols
    return ()


def _unique_keys(rows: list[dict[str, str]], cols: tuple[str, ...], *, label: str) -> set[tuple[str, ...]]:
    keys = [tuple(str(row.get(col) or "") for col in cols) for row in rows]
    if any(any(value == "" for value in key) for key in keys):
        raise CompilerError(f"{label} replacement coverage key has blank values for {cols}")
    out = set(keys)
    if len(out) != len(keys):
        raise CompilerError(f"{label} replacement coverage has duplicate keys for {cols}")
    return out


def _fed_stock_target_present(path: Path) -> bool:
    try:
        rows, _ = _read_csv(path)
    except CompilerError:
        return False
    return any("cbo_fed_holdings_target_bil" in row for row in rows)


def _write_issuance_mix(path: Path, issuance_mix: Any) -> None:
    if issuance_mix is None:
        payload = {
            "schema_version": "tdcsim_cbo_issuance_mix_assumptions_v1",
            "mode": "default_tdcsim_cbo_runner_profile",
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "claim_boundary": "issuance_mix_is_tdcsim_scenario_assumption_not_cbo_prescription",
        }
    else:
        payload = {
            "schema_version": "tdcsim_cbo_issuance_mix_assumptions_v1",
            "mode": "replace_shares",
            "security_shares": dict(issuance_mix.security_shares),
            "maturity_distributions": {
                key: [dict(item) for item in value]
                for key, value in issuance_mix.maturity_distributions.items()
            },
            "weighted_average_maturity_years": issuance_mix.weighted_average_maturity_years,
            "negative_issuance_action": issuance_mix.negative_issuance_action,
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "claim_boundary": "issuance_mix_is_tdcsim_scenario_assumption_not_cbo_prescription",
        }
    write_json(path, payload)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise CompilerError(f"required forecast input is missing: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CompilerError(f"CSV is missing a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], *, preferred_header: list[str]) -> None:
    materialized = [dict(row) for row in rows]
    header = list(preferred_header)
    extra = sorted({key for row in materialized for key in row} - set(header))
    fieldnames = header + extra
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def _compile_holder_preferences(
    baseline_rows: list[dict[str, str]],
    header: list[str],
    override: Mapping[str, Any],
    *,
    fed_stock_target_active: bool,
) -> list[dict[str, Any]]:
    validated = validate_holder_preferences(override, fed_stock_target_active=fed_stock_target_active)
    share_by_holder: dict[str, dict[str, float]] = {}
    for row in validated:
        security_type = str(row["security_type"])
        if security_type == "nonmarketable":
            continue
        column = f"{security_type}_pct"
        for holder, share in row["shares"].items():
            share_by_holder.setdefault(holder, {})[column] = share
    output: list[dict[str, Any]] = []
    seen_holders = set()
    for row in baseline_rows:
        holder = str(row.get("holder_type") or "")
        new = dict(row)
        if holder in share_by_holder:
            new.update(share_by_holder[holder])
            seen_holders.add(holder)
        new["source_role"] = "scenario_assumption"
        new["runtime_role"] = "memo_only"
        new["claim_boundary"] = "holder preference profile not exact holder ownership"
        new["scenario_transform"] = "static_shares"
        output.append(new)
    for holder in sorted(set(share_by_holder) - seen_holders):
        new = {field: "" for field in header}
        new["holder_type"] = holder
        new.update(share_by_holder[holder])
        new["source_role"] = "scenario_assumption"
        new["runtime_role"] = "memo_only"
        new["claim_boundary"] = "holder preference profile not exact holder ownership"
        new["scenario_transform"] = "static_shares"
        output.append(new)
    return output


__all__ = [
    "CboCompiledScenario",
    "CboScenarioCompiler",
    "CompilerError",
    "digest_input_tree",
    "input_tree_hashes",
]
