#!/usr/bin/env python3
"""Re-export existing RateWall marginal TDC pairs with the deposit-creation split."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tdcsim_cbo._json import read_json, write_json  # noqa: E402
from tdcsim_cbo.campaign_store import (  # noqa: E402
    locate_source_run_catalog,
    register_source_run,
    resolve_source_run,
)
from tdcsim_cbo.consumer_challenge import (  # noqa: E402
    build_handoff_package_manifest,
    build_ingest_challenge,
    collect_release_identity,
)
from tdcsim_cbo.marginal_tdc import (  # noqa: E402
    MANIFEST_FILE,
    SUMMARY_FILE,
    assemble_marginal_tdc_pair,
    verify_marginal_tdc_pair,
)


SOURCE_GRADE_ROOT = PROJECT_ROOT / "output" / "ratewall_source_grade_marginal_pairs_20260706"
FLOODED_ROOT = PROJECT_ROOT / "output" / "flooded_state_scenario_20260707"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "ratewall_split_pairs_20260707"
CUMULATIVE_SPLIT_FILE = "tdcsim_ratewall_cumulative_split_input.csv"
CHALLENGE_FILE = "tdcsim_ratewall_ingest_challenge.json"
PACKAGE_MANIFEST_FILE = "tdcsim_ratewall_handoff_package_manifest.json"
PAIR_INDEX_FILE = "tdcsim_ratewall_split_pair_index.csv"
CURRENT_SPLIT_PAIR_ID = "current_state_2026_plus_100bp_year_source_grade"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    producer_identity = collect_release_identity(PROJECT_ROOT)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"split output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    spec_out = output_root / "pair_specs"
    spec_out.mkdir(parents=True, exist_ok=True)

    source_specs = _specs(args.source_grade_root.expanduser().resolve())
    flooded_specs = _specs(args.flooded_root.expanduser().resolve())
    source_dir_map = _pair_dir_map(args.source_grade_root.expanduser().resolve())
    flooded_dir_map = _pair_dir_map(args.flooded_root.expanduser().resolve())

    index_rows: list[dict[str, Any]] = []
    cumulative_frames: list[pd.DataFrame] = []
    for group, specs, dir_map in (
        ("source_grade_cumulative_input", source_specs, source_dir_map),
        ("flooded_wave_scenario_surface", flooded_specs, flooded_dir_map),
    ):
        for spec_path in specs:
            spec = read_json(spec_path)
            if not isinstance(spec, Mapping):
                raise SystemExit(f"pair spec must be an object: {spec_path}")
            pair_id = str(spec["pair_id"])
            source_dir = dir_map.get(pair_id)
            if source_dir is None:
                raise SystemExit(f"missing existing pair directory for {pair_id}")
            pair_dir = output_root / source_dir.name
            copied_spec = spec_out / spec_path.name
            write_json(copied_spec, dict(spec))
            _import_source_runs(spec, locate_source_run_catalog(spec_path), output_root)
            result = assemble_marginal_tdc_pair(copied_spec, pair_dir)
            verified = verify_marginal_tdc_pair(result.output_dir)
            index_rows.append(
                {
                    "pair_id": pair_id,
                    "split_pair_dir": pair_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "source_pair_dir": source_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "pair_group": group,
                    "verification_status": verified["status"],
                    "rows": verified["rows"],
                }
            )
            if group == "source_grade_cumulative_input":
                cumulative_frames.append(pd.read_csv(result.summary_path))

    index_path = output_root / PAIR_INDEX_FILE
    index = pd.DataFrame(index_rows)
    index.to_csv(index_path, index=False)
    cumulative_path = output_root / CUMULATIVE_SPLIT_FILE
    cumulative = _cumulative_split_table(cumulative_frames)
    cumulative.to_csv(cumulative_path, index=False)

    generated_pair_dirs = _pair_dir_map(output_root)
    current_pair_dir = generated_pair_dirs.get(CURRENT_SPLIT_PAIR_ID)
    if current_pair_dir is None:
        raise SystemExit(f"missing current split pair for handoff: {CURRENT_SPLIT_PAIR_ID}")
    current_summary_path = current_pair_dir / SUMMARY_FILE
    pair_manifest_paths = sorted(output_root.glob(f"*/{MANIFEST_FILE}"))
    package_manifest_path = output_root / PACKAGE_MANIFEST_FILE
    write_json(
        package_manifest_path,
        build_handoff_package_manifest(
            output_root,
            campaign_id=args.campaign_id,
            pair_index_path=index_path,
            pair_manifest_paths=pair_manifest_paths,
            cumulative_csv=cumulative_path,
            current_summary_csv=current_summary_path,
            producer_identity=producer_identity,
        ),
    )
    # The challenge binds both files that the consumer hash-pins, the exact admitted projection,
    # and the package manifest carrying all pair/run lineage.
    write_json(
        output_root / CHALLENGE_FILE,
        build_ingest_challenge(
            cumulative_path,
            current_summary_path,
            package_manifest_path=package_manifest_path,
        ),
    )
    print(f"wrote {output_root}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-grade-root", default=SOURCE_GRADE_ROOT, type=Path)
    parser.add_argument("--flooded-root", default=FLOODED_ROOT, type=Path)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _import_source_runs(spec: Mapping[str, Any], source_catalog: Path, output_root: Path) -> None:
    for key in ("baseline_run_id", "shock_run_id"):
        source = resolve_source_run(source_catalog, str(spec[key]))
        if source.baseline_package is None or source.attestation is None:
            raise SystemExit(f"source campaign lacks replay inputs for {source.run_id}")
        register_source_run(
            output_root,
            source.root,
            baseline_package=source.baseline_package,
            attestation=source.attestation,
        )


def _specs(root: Path) -> list[Path]:
    spec_dir = root / "pair_specs"
    specs = sorted(spec_dir.glob("*.json"))
    if not specs:
        raise SystemExit(f"no pair specs found under {spec_dir}")
    return specs


def _pair_dir_map(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for manifest_path in sorted(root.glob(f"*/{MANIFEST_FILE}")):
        manifest = read_json(manifest_path)
        if isinstance(manifest, Mapping):
            out[str(manifest.get("pair_id") or "")] = manifest_path.parent
    return out


def _cumulative_split_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) != 11:
        raise SystemExit(f"expected 11 source-grade split summary frames, found {len(frames)}")
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != 11:
        raise SystemExit(f"expected 11 source-grade split rows, found {len(frame)}")
    rows = []
    for _, row in frame.sort_values(["period", "pair_id"]).iterrows():
        interest_excluded = float(row["delta_tdc_ex_overlap_interest_driven_excluded_bil"])
        admissible = float(row["delta_tdc_ex_overlap_non_interest_admissible_bil"])
        delta_ex = float(row["delta_tdc_ex_overlap_bil"])
        reconciled = interest_excluded + admissible
        identity_error = reconciled - delta_ex
        if abs(identity_error) > 1e-7:
            raise SystemExit(f"cumulative split identity failed for {row['pair_id']}: {identity_error}")
        rows.append(
            {
                "aggregation_schema_version": "tdcsim_ratewall_cumulative_split_input_v1",
                "aggregation_scope": "cumulative_split_input_for_ratewall_per_year_no_horizon_reweighting",
                "pair_id": row["pair_id"],
                "scenario_state_set_id": row["scenario_state_set_id"],
                "state_id": row["state_id"],
                "state_kind": row["state_kind"],
                "state_period": row["state_period"],
                "period": row["period"],
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "demand_conversion_case": row["demand_conversion_case"],
                "beta": float(row["beta"]),
                "delta_tdc_ex_overlap_bil": delta_ex,
                "delta_tdc_ex_overlap_interest_driven_excluded_bil": interest_excluded,
                "delta_tdc_ex_overlap_non_interest_admissible_bil": admissible,
                "delta_tdc_ex_overlap_split_remainder_bil": 0.0,
                "delta_tdc_ex_overlap_reconciled_bil": reconciled,
                "tdc_materialized_deposit_stock_admissible_bil": float(row["tdc_materialized_deposit_stock_admissible_bil"]),
                "tdc_materialized_deposit_stock_interest_excluded_bil": float(row["tdc_materialized_deposit_stock_interest_excluded_bil"]),
                "tdc_income_addendum_full_level_rate": float(row["tdc_income_addendum_full_level_rate"]),
                "tdc_income_addendum_gross_interest_bil": float(row["tdc_income_addendum_gross_interest_bil"]),
                "aggregation_identity_error_bil": 0.0,
                "aggregation_reconciliation_status": "pass",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
