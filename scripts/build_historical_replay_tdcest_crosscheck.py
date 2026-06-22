#!/usr/bin/env python3
"""Build a replay-versus-tdcest cross-check from local repo artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historical_replay import (  # noqa: E402
    _build_artifact_integrity,
    _build_code_identity_hash,
    _build_replay_config_hash,
    _build_run_id,
    run_historical_replay,
)
from historical_replay_tdcest import (  # noqa: E402
    DEFAULT_TDCEST_COLUMNS,
    build_selected_target_crosscheck,
    build_tdcest_crosscheck,
    load_tdcest_estimates,
    render_selected_target_crosscheck_markdown,
    render_tdcest_crosscheck_markdown,
    summarize_selected_target_crosscheck,
    summarize_tdcest_crosscheck,
)
from historical_replay_tdc import write_tdc_validation_artifacts  # noqa: E402


DEFAULT_CASH = ROOT / "data/historical_replay/processed/treasury_operating_cash_quarterly.csv"
DEFAULT_HOLDERS = ROOT / "data/historical_replay/processed/z1_treasury_holders_l210_full.csv"
DEFAULT_COHORTS = ROOT / "data/historical_replay/raw/fiscaldata/mspd_table_3_market.csv"
DEFAULT_TDCEST = ROOT / "data/historical_replay/imported/tdcest/tdc_estimates.csv"
DEFAULT_OUT = ROOT / "data/historical_replay/validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-quarter", default="2001Q1")
    parser.add_argument("--end-quarter", default="2025Q4")
    parser.add_argument("--cash", default=str(DEFAULT_CASH))
    parser.add_argument("--sector-positions", default=str(DEFAULT_HOLDERS))
    parser.add_argument("--cohorts", default=str(DEFAULT_COHORTS))
    parser.add_argument("--tdcest-estimates", default=str(DEFAULT_TDCEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--amount-unit-scale", type=float, default=1000.0)
    parser.add_argument("--sector-value-unit-scale", type=float, default=1_000_000.0)
    parser.add_argument("--tdcest-unit-scale", type=float, default=1000.0)
    parser.add_argument("--tdcest-column", action="append", dest="tdcest_columns")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "historical_replay": {
            "amount_unit_scale": args.amount_unit_scale,
            "sector_value_unit_scale": args.sector_value_unit_scale,
            "solver_tolerance": 1.0e-2,
            "start_quarter": args.start_quarter,
            "end_quarter": args.end_quarter,
            "output_dir": str(output_dir),
            "paths": {
                "cash": args.cash,
                "sector_positions": args.sector_positions,
                "cohorts": args.cohorts,
            },
        }
    }
    replay_results, _ = run_historical_replay(
        params,
        f"{args.start_quarter[:4]}-01-01",
        f"{args.end_quarter[:4]}-12-31",
        scenario_name="tdcest_crosscheck_replay",
    )
    selected_panel = replay_results.attrs.get("tdc_panel")
    if selected_panel is not None:
        write_tdc_validation_artifacts(
            output_dir,
            panel=selected_panel,
            formula_crosscheck=replay_results.attrs.get("tdc_formula_crosscheck"),
            source_manifest=replay_results.attrs.get("tdc_source_manifest"),
        )
        selected_crosscheck = build_selected_target_crosscheck(replay_results, selected_panel)
        selected_summary = summarize_selected_target_crosscheck(selected_crosscheck)
        selected_crosscheck_path = output_dir / "tdcest_selected_ladder_crosscheck.csv"
        selected_summary_path = output_dir / "tdcest_selected_ladder_crosscheck_summary.csv"
        selected_markdown_path = output_dir / "tdcest_selected_ladder_crosscheck.md"
        selected_crosscheck.to_csv(selected_crosscheck_path, index=False)
        selected_summary.to_csv(selected_summary_path, index=False)
        selected_markdown_path.write_text(
            render_selected_target_crosscheck_markdown(selected_summary),
            encoding="utf-8",
        )
        print(f"wrote {selected_crosscheck_path}")
        print(f"wrote {selected_summary_path}")
        print(f"wrote {selected_markdown_path}")
        print(selected_summary.to_string(index=False))

    tdcest = load_tdcest_estimates(args.tdcest_estimates)
    crosscheck = build_tdcest_crosscheck(
        replay_results,
        tdcest,
        tdcest_columns=args.tdcest_columns or DEFAULT_TDCEST_COLUMNS,
        tdcest_unit_scale=args.tdcest_unit_scale,
    )
    summary = summarize_tdcest_crosscheck(crosscheck)

    crosscheck_path = output_dir / "tdcest_replay_crosscheck.csv"
    summary_path = output_dir / "tdcest_replay_crosscheck_summary.csv"
    markdown_path = output_dir / "tdcest_replay_crosscheck.md"
    crosscheck.to_csv(crosscheck_path, index=False)
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(render_tdcest_crosscheck_markdown(summary), encoding="utf-8")
    _refresh_artifact_integrity(
        output_dir,
        cfg=params["historical_replay"],
        start_quarter=args.start_quarter,
        end_quarter=args.end_quarter,
    )
    _write_acceptance_artifact(output_dir)
    _refresh_artifact_integrity(
        output_dir,
        cfg=params["historical_replay"],
        start_quarter=args.start_quarter,
        end_quarter=args.end_quarter,
    )

    print(f"wrote {crosscheck_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {markdown_path}")
    print(summary.to_string(index=False))
    return 0


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _value_counts(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return "none"
    counts = frame[column].value_counts(dropna=False).sort_index()
    return ", ".join(f"{key}: {value}" for key, value in counts.items())


def _write_acceptance_artifact(output_dir: Path) -> None:
    input_manifest = _read_csv(output_dir / "historical_replay_input_manifest.csv")
    artifact_integrity = _read_csv(output_dir / "artifact_integrity.csv")
    selected = _read_csv(output_dir / "tdcest_selected_ladder_crosscheck_summary.csv")
    modern = _read_csv(output_dir / "tdcest_modern_formula_summary.csv")
    diagnostics = _read_csv(output_dir / "historical_replay_diagnostics.csv")
    transitions = _read_csv(output_dir / "portfolio_transition_diagnostics.csv")
    sample_manifest = _read_csv(output_dir / "historical_replay_large_artifact_sample_manifest.csv")
    valuation = _read_csv(output_dir / "valuation_basis_feasibility_certificate.csv")
    interest = _read_csv(output_dir / "interest_proxy_alignment.csv")
    portfolio = _read_csv(output_dir / "historical_replay_portfolio_snapshots.csv")
    final_portfolio = _read_csv(output_dir / "historical_replay_final_portfolio.csv")
    event_rollforward = _read_csv(output_dir / "historical_replay_event_rollforward.csv")
    unexplained = _read_csv(output_dir / "historical_replay_unexplained_change_ledger.csv")
    z1_flow = _read_csv(output_dir / "z1_transaction_flow_diagnostics.csv")

    required_inputs = (
        int(input_manifest.get("required_for_claim", pd.Series(dtype=bool)).astype(str).str.lower().isin({"true", "1"}).sum())
        if not input_manifest.empty
        else 0
    )
    solver = diagnostics[diagnostics.get("diagnostic_type", pd.Series(dtype=str)).astype(str).eq("solver")]
    aggregate_only = sorted(
        solver.loc[
            solver.get("solver_method", pd.Series(dtype=str)).astype(str).eq("aggregate_only_minimum_weighted_slack_certificate"),
            "quarter",
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    transition_rows = len(transitions.index)
    high_tv_rows = int(
        transitions.get("high_tv_transition", pd.Series(False, index=transitions.index))
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
        .sum()
    )
    transition_intervals = sorted(
        pd.to_numeric(transitions.get("interval_quarters", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    sample_min = (
        int(pd.to_numeric(sample_manifest.get("sample_row_count"), errors="coerce").min())
        if not sample_manifest.empty
        else 0
    )
    final_residual_rows = (
        int(final_portfolio.get("source_sector", pd.Series(dtype=str)).astype(str).str.contains("SourceBasisResidual", case=False, na=False).sum())
        if not final_portfolio.empty
        else 0
    )
    max_rollforward_residual = (
        float(pd.to_numeric(event_rollforward.get("rollforward_residual_mil"), errors="coerce").abs().max())
        if not event_rollforward.empty and "rollforward_residual_mil" in event_rollforward.columns
        else 0.0
    )
    z1_flow_observed = (
        int(pd.to_numeric(z1_flow.get("z1_transaction_flow_mil"), errors="coerce").notna().sum())
        if not z1_flow.empty
        else 0
    )
    selected_row = selected.iloc[0].to_dict() if not selected.empty else {}
    modern_row = modern.iloc[0].to_dict() if not modern.empty else {}
    valuation_rows = len(valuation.index)
    interest_feasible_column = (
        "within_feasible_bounds"
        if "within_feasible_bounds" in interest.columns
        else "target_within_feasible_bounds"
        if "target_within_feasible_bounds" in interest.columns
        else None
    )
    interest_infeasible = (
        int(
            interest[interest_feasible_column]
            .astype(str)
            .str.lower()
            .isin({"false", "0"})
            .sum()
        )
        if not interest.empty and interest_feasible_column is not None
        else 0
    )

    text = f"""# Historical Replay Acceptance Status

Status: **local evidence regenerated from current artifacts; external closeout verdict pending**.

This artifact is generated from live validation CSVs by `scripts/build_historical_replay_tdcest_crosscheck.py`. It is not a hand-maintained status note. The scoped claim remains quarterly aggregate-consistent synthetic replay, not exact transaction replay, exact holder-by-CUSIP history, or observed market-value reconstruction.

## Current Run

- Command: `uv run --with pandas --with numpy --with scipy --with pyyaml --with python-dateutil --with matplotlib python scripts/build_historical_replay_tdcest_crosscheck.py`
- Run range: `2001Q1` through `2025Q4`, quarterly end-of-period.
- Target-comparable acceptance window: `2002Q1` through `2025Q4`; four `2001` rows are warm-up rows with no selected empirical TDC target.
- Replay input manifest rows: `{len(input_manifest.index)}` total, `{required_inputs}` required for the scoped claim.
- Artifact integrity rows before final refresh: `{len(artifact_integrity.index)}`; status counts: `{_value_counts(artifact_integrity, "status")}`.
- Solver methods: `{_value_counts(solver, "solver_method")}`.
- Aggregate-only certificate quarters: `{len(aggregate_only)}` (`{", ".join(aggregate_only)}`).

## Evidence

- Selected TDC target wiring: `{int(selected_row.get("matched_rows", 0))}/{int(selected_row.get("compared_rows", 0))}` compared rows matched; mismatch rows `{int(selected_row.get("mismatch_rows", 0))}`; max absolute difference `{selected_row.get("max_abs_difference_bil", "n/a")}` billion.
- Modern canonical TDC formula: `{int(modern_row.get("matched_rows", 0)) if "matched_rows" in modern_row else int(modern_row.get("compared_rows", 0)) - int(modern_row.get("canonical_formula_mismatch_rows", 0))}/{int(modern_row.get("compared_rows", 0))}` compared rows matched; canonical formula mismatch rows `{int(modern_row.get("canonical_formula_mismatch_rows", 0))}`.
- Portfolio snapshots: `{len(portfolio.index)}` rows; final portfolio: `{len(final_portfolio.index)}` rows; final source-basis residual rows `{final_residual_rows}`.
- Portfolio transition diagnostics: `{transition_rows}` rows; high-TV rows `{high_tv_rows}`; interval_quarters values `{transition_intervals}`.
- Z.1 transaction-flow diagnostics: `{len(z1_flow.index)}` rows; rows with quarterly flow observations `{z1_flow_observed}`.
- Valuation-basis feasibility certificate rows: `{valuation_rows}`.
- Interest alignment infeasible rows under current bounds: `{interest_infeasible}`.
- Event rollforward rows: `{len(event_rollforward.index)}`; max rollforward residual `{max_rollforward_residual}` million.
- Unexplained change ledger rows: `{len(unexplained.index)}`.
- Deterministic large-artifact sample minimum row count: `{sample_min}`.

## Boundaries

- Aggregate-only certificate quarters suppress holder/security portfolio claims for those quarters.
- Z.1 transaction flows are aggregate transition diagnostics, not exact secondary-market transfers.
- Pricing values are model-implied diagnostics, not observed market quotes.
- Residual and unexplained cohort changes are disclosed accounting/model residuals, not observed transfers.
- FFIEC/NCUA maturity ladders are proxy constraints and priors, not exact holder-by-CUSIP observations.
"""
    (output_dir / "historical_replay_acceptance.md").write_text(text, encoding="utf-8")


def _refresh_artifact_integrity(
    output_dir: Path,
    *,
    cfg: dict,
    start_quarter: str,
    end_quarter: str,
) -> None:
    integrity_path = output_dir / "artifact_integrity.csv"
    if not integrity_path.exists():
        return
    integrity = pd.read_csv(integrity_path)
    input_manifest_path = output_dir / "historical_replay_input_manifest.csv"
    replay_input_manifest = (
        pd.read_csv(input_manifest_path, low_memory=False)
        if input_manifest_path.exists()
        else pd.DataFrame()
    )
    config_sha256 = _build_replay_config_hash(
        cfg,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
        replay_input_manifest=replay_input_manifest,
    )
    code_sha256 = _build_code_identity_hash()
    run_id = _build_run_id(config_sha256=config_sha256, code_sha256=code_sha256)
    paths: dict[str, str] = {}
    if {"artifact", "path"}.issubset(integrity.columns):
        for _, row in integrity.iterrows():
            artifact = str(row.get("artifact", "")).strip()
            path = str(row.get("path", "")).strip()
            if artifact and path and artifact != "artifact_integrity":
                paths[artifact] = str(ROOT / path)
    paths.update(
        {
            "selected_ladder_crosscheck": str(output_dir / "tdcest_selected_ladder_crosscheck.csv"),
            "selected_ladder_crosscheck_summary": str(output_dir / "tdcest_selected_ladder_crosscheck_summary.csv"),
            "selected_ladder_crosscheck_markdown": str(output_dir / "tdcest_selected_ladder_crosscheck.md"),
            "tdcest_replay_crosscheck": str(output_dir / "tdcest_replay_crosscheck.csv"),
            "tdcest_replay_crosscheck_summary": str(output_dir / "tdcest_replay_crosscheck_summary.csv"),
            "tdcest_replay_crosscheck_markdown": str(output_dir / "tdcest_replay_crosscheck.md"),
            "historical_replay_acceptance": str(output_dir / "historical_replay_acceptance.md"),
        }
    )
    _build_artifact_integrity(
        paths,
        run_id=run_id,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
    ).to_csv(integrity_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
