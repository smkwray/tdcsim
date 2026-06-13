"""Runtime readers for source-backed RateWall input paths.

The files loaded here are contracts produced before running tdcsim. They let
RateWall-oriented scenarios use quarter-specific fiscal-flow and holder-path
inputs without baking those assumptions into the simulator core.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from tdc_shared import (
    HOLDER_TYPES,
    PREFERENCE_CATEGORIES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
    PRIVATE_SUBBUCKETS,
)


def _resolve_path(path_value: str | None, base_dir: str | os.PathLike[str] | None = None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir:
        return Path(base_dir) / path
    return path


def _quarter_label(value) -> str:
    ts = pd.to_datetime(value)
    return f"{ts.year}Q{ts.quarter}"


def _scenario_candidates(scenario_name: str) -> list[str]:
    return [scenario_name, "all", "default", ""]


def load_primary_flow_path(config: dict, *, base_dir: str | os.PathLike[str] | None = None) -> dict[str, dict[str, float]]:
    """Load quarterly fiscal-flow path rows keyed by scenario then quarter."""

    path_value = config.get("primary_flow_to_du_file")
    path = _resolve_path(path_value, base_dir)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Primary flow path file is missing: {path}")
    frame = pd.read_csv(path)
    required = {"quarter", "primary_fiscal_flow_to_du_bil"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Primary flow path missing columns: {sorted(missing)}")

    scenario_col = "scenario_id" if "scenario_id" in frame.columns else None
    lookup: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        scenario_id = str(row.get(scenario_col, "all") if scenario_col else "all")
        quarter = str(row["quarter"])
        lookup.setdefault(scenario_id, {})[quarter] = float(row["primary_fiscal_flow_to_du_bil"])
    return lookup


def primary_flow_for_period(
    lookup: dict[str, dict[str, float]],
    *,
    scenario_name: str,
    current_date,
    period_counts_by_quarter: dict[str, int],
    warning_cache: set[str] | None = None,
) -> float | None:
    """Return the current period's fiscal-flow amount in billions."""

    if not lookup:
        return None
    quarter = _quarter_label(current_date)
    for scenario_id in _scenario_candidates(scenario_name):
        if quarter in lookup.get(scenario_id, {}):
            if scenario_id != scenario_name and warning_cache is not None:
                key = f"primary_flow:{scenario_name}:{scenario_id}"
                if key not in warning_cache:
                    print(
                        f"WARNING [{scenario_name}]: primary_flow path used fallback scenario "
                        f"'{scenario_id}' instead of '{scenario_name}'."
                    )
                    warning_cache.add(key)
            count = max(1, int(period_counts_by_quarter.get(quarter, 1)))
            return float(lookup[scenario_id][quarter]) / count
    return None


def load_holder_absorption_path(config: dict, *, base_dir: str | os.PathLike[str] | None = None) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Load holder preferences keyed by scenario, quarter, and holder."""

    path_value = config.get("holder_absorption_path_file")
    path = _resolve_path(path_value, base_dir)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Holder absorption path file is missing: {path}")
    frame = pd.read_csv(path)
    required = {"scenario_id", "quarter", "holder_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Holder absorption path missing columns: {sorted(missing)}")

    pref_cols = [f"{category}_pct" for category in PREFERENCE_CATEGORIES if f"{category}_pct" in frame.columns]
    if not pref_cols:
        raise ValueError("Holder absorption path has no *_pct preference columns.")

    lookup: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    route_values: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for _, row in frame.iterrows():
        holder = str(row["holder_type"])
        if holder not in HOLDER_TYPES:
            raise ValueError(f"Unsupported holder_type in holder path: {holder}")
        scenario_id = str(row["scenario_id"])
        quarter = str(row["quarter"])
        subbucket = str(row.get("holder_subbucket", "") or "")
        prefs = {
            col: float(row[col])
            for col in pref_cols
            if pd.notna(row[col])
        }
        holder_rows = lookup.setdefault(scenario_id, {}).setdefault(quarter, {})
        target = holder_rows.setdefault(holder, {})
        for col, value in prefs.items():
            target[col] = target.get(col, 0.0) + value
        if holder == "Private" and subbucket in PRIVATE_SUBBUCKETS:
            route_row = route_values.setdefault(scenario_id, {}).setdefault(quarter, {}).setdefault(
                subbucket,
                {},
            )
            for col, value in prefs.items():
                route_row[col] = route_row.get(col, 0.0) + value
    for scenario_id, quarters in route_values.items():
        for quarter, route_rows in quarters.items():
            private_prefs = lookup.setdefault(scenario_id, {}).setdefault(quarter, {}).setdefault("Private", {})
            shares_by_category = {}
            for category in PREFERENCE_CATEGORIES:
                col = f"{category}_pct"
                total = float(private_prefs.get(col, 0.0) or 0.0)
                if total <= 0.0:
                    continue
                shares_by_category[category] = {
                    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: float(
                        route_rows.get(PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, {}).get(col, 0.0) or 0.0
                    )
                    / total,
                    PRIVATE_SUBBUCKET_MMF: float(
                        route_rows.get(PRIVATE_SUBBUCKET_MMF, {}).get(col, 0.0) or 0.0
                    )
                    / total,
                }
            if shares_by_category:
                lookup[scenario_id][quarter]["__private_subbucket_shares__"] = shares_by_category
    return lookup


def holder_preferences_for_period(
    lookup: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    scenario_name: str,
    current_date,
    fallback_preferences: dict,
    warning_cache: set[str] | None = None,
) -> dict:
    """Return quarter-specific auction preferences when available."""

    if not lookup:
        return fallback_preferences
    quarter = _quarter_label(current_date)
    for scenario_id in _scenario_candidates(scenario_name):
        scenario_rows = lookup.get(scenario_id, {})
        if quarter not in scenario_rows:
            continue
        if scenario_id != scenario_name and warning_cache is not None:
            key = f"holder_absorption:{scenario_name}:{scenario_id}"
            if key not in warning_cache:
                print(
                    f"WARNING [{scenario_name}]: holder_absorption path used fallback scenario "
                    f"'{scenario_id}' instead of '{scenario_name}'."
                )
                warning_cache.add(key)
        prefs = {holder: dict(fallback_preferences.get(holder, {})) for holder in HOLDER_TYPES}
        for holder, holder_prefs in scenario_rows[quarter].items():
            if holder == "__private_subbucket_shares__":
                continue
            prefs.setdefault(holder, {}).update(holder_prefs)
        if "__private_subbucket_shares__" in scenario_rows[quarter]:
            prefs["__private_subbucket_shares__"] = scenario_rows[quarter]["__private_subbucket_shares__"]
        return prefs
    return fallback_preferences
