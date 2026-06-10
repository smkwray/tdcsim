"""Dynamic yield-curve path helpers for tdcsim contract mode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_CURVE_COLUMNS = {
    "scenario_id",
    "curve_date",
    "tenor_years",
    "nominal_rate",
}


def load_yield_curve_surface(path: str | Path | None) -> pd.DataFrame:
    """Load a date x tenor yield-curve surface."""

    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"yield curve surface file is missing: {path}")
    surface = pd.read_csv(path)
    missing = REQUIRED_CURVE_COLUMNS - set(surface.columns)
    if missing:
        raise ValueError(f"yield curve surface missing columns: {sorted(missing)}")
    surface = surface.copy()
    surface["curve_date"] = pd.to_datetime(surface["curve_date"])
    surface["tenor_years"] = pd.to_numeric(surface["tenor_years"], errors="raise")
    surface["nominal_rate"] = pd.to_numeric(surface["nominal_rate"], errors="raise")
    return surface.sort_values(["scenario_id", "curve_date", "tenor_years"]).reset_index(drop=True)


def curve_for_date(
    surface: pd.DataFrame,
    date,
    *,
    scenario_id: str | None = None,
) -> tuple[list[float], list[float], str]:
    """Return the latest curve on or before date for a scenario."""

    if surface is None or surface.empty:
        return ([], [], "static_curve_no_surface")
    current_date = pd.to_datetime(date)
    candidates = surface[surface["curve_date"] <= current_date].copy()
    if scenario_id and "scenario_id" in candidates.columns:
        scenario_candidates = candidates[candidates["scenario_id"].astype(str) == str(scenario_id)]
        if not scenario_candidates.empty:
            candidates = scenario_candidates
    if candidates.empty:
        return ([], [], "dynamic_curve_no_prior_date")
    curve_date = candidates["curve_date"].max()
    curve = candidates[candidates["curve_date"] == curve_date].sort_values("tenor_years")
    return (
        [float(v) for v in curve["tenor_years"].tolist()],
        [float(v) for v in curve["nominal_rate"].tolist()],
        f"dynamic_curve_surface:{curve_date.date()}",
    )
