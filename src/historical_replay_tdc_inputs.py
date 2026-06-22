"""Source loading and provenance helpers for historical replay TDC inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TDCEST_PROCESSED = _PROJECT_ROOT.parent / "tdcest" / "data" / "processed"
_IMPORTED_TDCEST = _PROJECT_ROOT / "data" / "historical_replay" / "imported" / "tdcest"

SOURCE_SPECS = {
    "quarterly_inputs": {
        "filename": "quarterly_inputs.csv",
        "role": "canonical_formula_inputs",
        "units": "millions_of_dollars",
        "required": True,
    },
    "tdc_estimates": {
        "filename": "tdc_estimates.csv",
        "role": "published_tdc_estimate_rows",
        "units": "millions_of_dollars",
        "required": True,
    },
    "tdc_components": {
        "filename": "tdc_components.csv",
        "role": "tdc_component_and_decomposition_inputs",
        "units": "millions_of_dollars",
        "required": True,
    },
    "tdc_tier2_regression_series": {
        "filename": "tdc_tier2_regression_series.csv",
        "role": "selected_long_history_companion_rows",
        "units": "millions_of_dollars",
        "required": True,
    },
    "tdc_du_fiscal_flow_research": {
        "filename": "tdc_du_fiscal_flow_research.csv",
        "role": "fiscal_deficit_transfer_inputs",
        "units": "millions_of_dollars",
        "required": True,
    },
    "tdc_empirical_anchor": {
        "filename": "tdc_empirical_anchor.csv",
        "role": "modern_decomposition_anchor",
        "units": "millions_of_dollars",
        "required": True,
    },
    "tier2_interest_component_candidate": {
        "filename": "tier2_interest_component_candidate.csv",
        "role": "interest_component_alignment_reference",
        "units": "millions_of_dollars",
        "required": False,
    },
    "tier2_interest_source_constraints": {
        "filename": "tier2_interest_source_constraints.csv",
        "role": "interest_source_constraint_reference",
        "units": "millions_of_dollars",
        "required": False,
    },
    "treasury_interest_expense": {
        "filename": "treasury__interest_expense.csv",
        "role": "official_treasury_interest_expense_component_pool_reference",
        "units": "dollars",
        "required": False,
    },
    "ffiec_interest_constraints_normalized": {
        "filename": "ffiec_interest_constraints_normalized.csv",
        "role": "bank_security_interest_constraint_reference",
        "units": "reported_thousands_or_normalized_source_units",
        "required": False,
    },
    "ncua_interest_constraints_normalized": {
        "filename": "ncua_interest_constraints_normalized.csv",
        "role": "credit_union_interest_constraint_reference",
        "units": "reported_thousands_or_normalized_source_units",
        "required": False,
    },
    "tdc_mmf_rrp_quarterly_adjustments": {
        "filename": "tdc_mmf_rrp_quarterly_adjustments.csv",
        "role": "mmf_rrp_source_of_funds_adjustment_reference",
        "units": "millions_of_dollars_or_share",
        "required": False,
    },
    "tdc_mmf_rrp_quarterly_adjustments_sec_full": {
        "filename": "tdc_mmf_rrp_quarterly_adjustments_sec_full.csv",
        "role": "mmf_rrp_source_of_funds_adjustment_sec_full_reference",
        "units": "millions_of_dollars_or_share",
        "required": False,
    },
    "method_meta": {
        "filename": "method_meta.json",
        "role": "tdcest_method_metadata",
        "units": "not_applicable",
        "required": False,
    },
}


@dataclass(frozen=True)
class ReplaySource:
    """Resolved input file for a replay source."""

    key: str
    path: Path
    role: str
    units: str
    required: bool


def resolve_tdcest_source_paths(overrides: dict | None = None) -> dict[str, ReplaySource]:
    """Resolve TDC-EST source files using local imported copies before siblings."""

    overrides = overrides or {}
    resolved: dict[str, ReplaySource] = {}
    for key, spec in SOURCE_SPECS.items():
        override = overrides.get(key)
        path = Path(override) if override else _resolve_default_path(str(spec["filename"]))
        if spec["required"] and (path is None or not path.exists()):
            raise FileNotFoundError(f"required historical replay TDC source is missing: {key}")
        if path is None or not path.exists():
            continue
        resolved[key] = ReplaySource(
            key=key,
            path=path,
            role=str(spec["role"]),
            units=str(spec["units"]),
            required=bool(spec["required"]),
        )
    return resolved


def load_tdcest_replay_sources(overrides: dict | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load resolved TDC-EST replay sources plus a source manifest."""

    sources = resolve_tdcest_source_paths(overrides)
    frames: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, object]] = []
    for source in sources.values():
        if source.path.suffix.lower() != ".csv":
            manifest_rows.append(_manifest_row(source, row_count=pd.NA, first_quarter=pd.NA, last_quarter=pd.NA))
            continue
        frame = pd.read_csv(source.path, low_memory=False)
        frame = normalize_date_and_quarter(frame, dataset_name=source.key)
        frames[source.key] = frame
        first_quarter = frame["quarter"].dropna().min() if "quarter" in frame.columns else pd.NA
        last_quarter = frame["quarter"].dropna().max() if "quarter" in frame.columns else pd.NA
        manifest_rows.append(
            _manifest_row(
                source,
                row_count=len(frame.index),
                first_quarter=first_quarter,
                last_quarter=last_quarter,
            )
        )
    return frames, pd.DataFrame(manifest_rows)


def normalize_date_and_quarter(frame: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    """Return a copy with normalized quarter-end `date` and `quarter` columns."""

    out = frame.copy()
    if "date" in out.columns:
        parsed = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    elif "record_date" in out.columns:
        parsed = pd.to_datetime(out["record_date"], errors="coerce").dt.normalize()
    elif "quarter" in out.columns:
        quarter_text = out["quarter"].astype(str).str.replace("-", "", regex=False)
        parsed = pd.PeriodIndex(quarter_text, freq="Q").to_timestamp(how="end").normalize()
    else:
        raise ValueError(f"{dataset_name} must include `date` or `quarter`")
    if pd.isna(parsed).any():
        bad = out.loc[pd.isna(parsed)].head(3).to_dict(orient="records")
        raise ValueError(f"{dataset_name} contains invalid date/quarter values: {bad}")
    out["date"] = pd.to_datetime(parsed).dt.normalize()
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)
    return out


def write_source_manifest(manifest: pd.DataFrame, path: str | Path) -> None:
    """Write a source manifest CSV."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)


def _resolve_default_path(filename: str) -> Path | None:
    imported = _IMPORTED_TDCEST / filename
    if imported.exists():
        return imported
    sibling = _TDCEST_PROCESSED / filename
    if sibling.exists():
        return sibling
    return imported


def _manifest_row(
    source: ReplaySource,
    *,
    row_count: object,
    first_quarter: object,
    last_quarter: object,
) -> dict[str, object]:
    return {
        "source_key": source.key,
        "path": _display_path(source.path),
        "sha256": _sha256(source.path),
        "role": source.role,
        "units": source.units,
        "required": source.required,
        "row_count": row_count,
        "first_quarter": first_quarter,
        "last_quarter": last_quarter,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SOURCE_SPECS",
    "ReplaySource",
    "load_tdcest_replay_sources",
    "normalize_date_and_quarter",
    "resolve_tdcest_source_paths",
    "write_source_manifest",
]
