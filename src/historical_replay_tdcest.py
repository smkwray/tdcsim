"""Cross-check historical replay outputs against tdcest estimate panels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_TDCEST_COLUMNS = (
    "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow",
    "tdc_base_bank_only_ru_flow",
    "tdc_base_broad_depository_np_cu_ru_flow",
    "tdc_tier2_interest_corrected_bank_only_ru_flow",
    "tdc_tier2_interest_corrected_broad_depository_np_cu_ru_flow",
    "tdc_tier3_fiscal_corrected_bank_only_ru_flow",
    "tdc_tier3_fiscal_corrected_broad_depository_np_cu_ru_flow",
)


def load_tdcest_estimates(path: str | Path) -> pd.DataFrame:
    """Load the wide tdcest estimates panel with a normalized date column."""

    frame = pd.read_csv(path, low_memory=False)
    if "date" not in frame.columns:
        raise ValueError("tdcest estimates must include a date column")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("tdcest estimates contains invalid date values")
    return frame


def build_tdcest_crosscheck(
    replay_results: pd.DataFrame,
    tdcest_estimates: pd.DataFrame,
    *,
    tdcest_columns: Iterable[str] = DEFAULT_TDCEST_COLUMNS,
    replay_column: str = "TDC_Change",
    tdcest_unit_scale: float = 1000.0,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Return a long-form comparison between replay TDC and selected tdcest series.

    `tdcest` values are stored in millions. tdcsim replay values are on the
    same scale as the replay result frame, so the default comparison converts
    tdcest millions to billions.
    """

    if replay_column not in replay_results.columns:
        raise ValueError(f"replay results must include {replay_column}")
    if "date" not in tdcest_estimates.columns:
        raise ValueError("tdcest estimates must include a date column")
    missing = [column for column in tdcest_columns if column not in tdcest_estimates.columns]
    if missing:
        raise ValueError(f"tdcest estimates missing comparison columns: {missing}")

    replay = replay_results.reset_index().rename(columns={"Date": "date"}).copy()
    if "date" not in replay.columns:
        replay = replay.rename(columns={replay.columns[0]: "date"})
    replay["date"] = pd.to_datetime(replay["date"], errors="coerce").dt.normalize()
    replay["quarter"] = replay["date"].dt.to_period("Q").astype(str)
    replay[replay_column] = pd.to_numeric(replay[replay_column], errors="coerce")

    tdcest = tdcest_estimates.copy()
    tdcest["date"] = pd.to_datetime(tdcest["date"], errors="coerce").dt.normalize()
    for column in tdcest_columns:
        tdcest[column] = pd.to_numeric(tdcest[column], errors="coerce") / float(tdcest_unit_scale)

    wide = replay[["date", "quarter", replay_column]].merge(
        tdcest[["date", *tdcest_columns]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for _, row in wide.iterrows():
        replay_value = row[replay_column]
        for column in tdcest_columns:
            tdcest_value = row[column]
            diff = replay_value - tdcest_value if pd.notna(replay_value) and pd.notna(tdcest_value) else pd.NA
            rows.append(
                {
                    "date": row["date"],
                    "quarter": row["quarter"],
                    "tdcsim_replay_column": replay_column,
                    "tdcest_series": column,
                    "tdcsim_value_bil": replay_value,
                    "tdcest_value_bil": tdcest_value,
                    "difference_bil": diff,
                    "absolute_difference_bil": abs(diff) if pd.notna(diff) else pd.NA,
                    "verdict": _comparison_verdict(
                        replay_value,
                        tdcest_value,
                        tolerance=tolerance,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_selected_target_crosscheck(
    replay_results: pd.DataFrame,
    selected_tdc_panel: pd.DataFrame,
    *,
    replay_column: str = "TDC_Change",
    selected_value_column: str = "selected_tdc_value_mil",
    selected_unit_scale: float = 1000.0,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Compare replay TDC to its one selected TDC-EST target per quarter."""

    if replay_column not in replay_results.columns:
        raise ValueError(f"replay results must include {replay_column}")
    required = {
        "date",
        "quarter",
        selected_value_column,
        "selected_tdc_series_key",
        "replay_tdc_method_label",
        "replay_tdc_method_tier",
    }
    missing = sorted(required.difference(selected_tdc_panel.columns))
    if missing:
        raise ValueError(f"selected TDC panel missing columns: {missing}")

    replay = replay_results.reset_index().rename(columns={"Date": "date"}).copy()
    if "date" not in replay.columns:
        replay = replay.rename(columns={replay.columns[0]: "date"})
    replay["date"] = pd.to_datetime(replay["date"], errors="coerce").dt.normalize()
    replay["quarter"] = replay["date"].dt.to_period("Q").astype(str)
    replay[replay_column] = pd.to_numeric(replay[replay_column], errors="coerce")

    selected = selected_tdc_panel.copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.normalize()
    selected["selected_tdc_value_bil"] = (
        pd.to_numeric(selected[selected_value_column], errors="coerce") / float(selected_unit_scale)
    )
    cols = [
        "date",
        "quarter",
        "selected_tdc_series_key",
        "replay_tdc_method_label",
        "replay_tdc_method_tier",
        "selected_tdc_value_bil",
    ]
    out = replay[["date", "quarter", replay_column]].merge(
        selected[cols],
        on=["date", "quarter"],
        how="left",
        validate="one_to_one",
    )
    out = out.rename(columns={replay_column: "tdcsim_value_bil"})
    out["validation_role"] = "selected_target_wiring_not_portfolio_replay_validation"
    out["difference_bil"] = out["tdcsim_value_bil"] - out["selected_tdc_value_bil"]
    out["absolute_difference_bil"] = out["difference_bil"].abs()
    out["verdict"] = out.apply(
        lambda row: _selected_verdict(
            row["tdcsim_value_bil"],
            row["selected_tdc_value_bil"],
            tolerance=tolerance,
        ),
        axis=1,
    )
    return out


def summarize_tdcest_crosscheck(crosscheck: pd.DataFrame) -> pd.DataFrame:
    """Summarize a long-form replay-versus-tdcest comparison by tdcest series."""

    if crosscheck.empty:
        return pd.DataFrame(
            columns=[
                "tdcest_series",
                "rows",
                "compared_rows",
                "matched_rows",
                "tdcsim_replay_tdc_not_implemented_rows",
                "mismatch_rows",
                "no_tdcest_value_rows",
                "first_compared_date",
                "last_compared_date",
                "max_abs_difference_bil",
                "mean_abs_difference_bil",
            ]
        )
    rows = []
    for series, group in crosscheck.groupby("tdcest_series", sort=False):
        compared = group.loc[group["tdcest_value_bil"].notna()].copy()
        rows.append(
            {
                "tdcest_series": series,
                "rows": int(len(group.index)),
                "compared_rows": int(len(compared.index)),
                "matched_rows": int((group["verdict"] == "matched").sum()),
                "tdcsim_replay_tdc_not_implemented_rows": int(
                    (group["verdict"] == "tdcsim_replay_tdc_not_implemented").sum()
                ),
                "mismatch_rows": int((group["verdict"] == "mismatch").sum()),
                "no_tdcest_value_rows": int((group["verdict"] == "no_tdcest_value").sum()),
                "first_compared_date": compared["date"].min() if not compared.empty else pd.NaT,
                "last_compared_date": compared["date"].max() if not compared.empty else pd.NaT,
                "max_abs_difference_bil": compared["absolute_difference_bil"].max()
                if not compared.empty
                else pd.NA,
                "mean_abs_difference_bil": compared["absolute_difference_bil"].mean()
                if not compared.empty
                else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def summarize_selected_target_crosscheck(crosscheck: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-target replay validation."""

    if crosscheck.empty:
        return pd.DataFrame(
            columns=[
                "rows",
                "compared_rows",
                "matched_rows",
                "missing_selected_target_rows",
                "mismatch_rows",
                "first_compared_date",
                "last_compared_date",
                "max_abs_difference_bil",
                "method_keys",
            ]
        )
    compared = crosscheck.loc[crosscheck["selected_tdc_value_bil"].notna()].copy()
    return pd.DataFrame(
        [
            {
                "rows": int(len(crosscheck.index)),
                "compared_rows": int(len(compared.index)),
                "matched_rows": int((crosscheck["verdict"] == "matched").sum()),
                "missing_selected_target_rows": int((crosscheck["verdict"] == "missing_selected_target").sum()),
                "mismatch_rows": int((crosscheck["verdict"] == "mismatch").sum()),
                "first_compared_date": compared["date"].min() if not compared.empty else pd.NaT,
                "last_compared_date": compared["date"].max() if not compared.empty else pd.NaT,
                "max_abs_difference_bil": compared["absolute_difference_bil"].max()
                if not compared.empty
                else pd.NA,
                "method_keys": ";".join(
                    sorted(crosscheck["selected_tdc_series_key"].dropna().astype(str).unique().tolist())
                ),
                "validation_role": ";".join(
                    sorted(crosscheck.get("validation_role", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
                ),
            }
        ]
    )


def render_tdcest_crosscheck_markdown(summary: pd.DataFrame) -> str:
    """Render a concise markdown summary for the replay-versus-tdcest check."""

    lines = [
        "# Historical Replay vs TDCest Broad Comparison Panel",
        "",
        "This comparison panel reports tdcsim historical replay `TDC_Change` against several `tdcest` estimate series, including alternatives that are not the replay's selected target.",
        "Use `tdcest_selected_ladder_crosscheck.md` as the primary selected-target wiring proof.",
        "Values are reported in billions of U.S. dollars.",
        "",
    ]
    if summary.empty:
        lines.append("No comparison rows were produced.")
        return "\n".join(lines) + "\n"

    for _, row in summary.iterrows():
        lines.append(f"## {row['tdcest_series']}")
        lines.append(f"- Compared rows: {int(row['compared_rows'])}")
        lines.append(f"- Matched rows: {int(row['matched_rows'])}")
        lines.append(
            "- Replay TDC not implemented rows: "
            f"{int(row['tdcsim_replay_tdc_not_implemented_rows'])}"
        )
        lines.append(f"- Mismatch rows: {int(row['mismatch_rows'])}")
        lines.append(f"- No tdcest value rows: {int(row['no_tdcest_value_rows'])}")
        if pd.notna(row["first_compared_date"]):
            first = pd.Timestamp(row["first_compared_date"]).date().isoformat()
            last = pd.Timestamp(row["last_compared_date"]).date().isoformat()
            lines.append(f"- Compared date range: {first} to {last}")
        if pd.notna(row["max_abs_difference_bil"]):
            lines.append(f"- Max absolute difference: {float(row['max_abs_difference_bil']):,.6f}")
            lines.append(f"- Mean absolute difference: {float(row['mean_abs_difference_bil']):,.6f}")
        lines.append("")
    return "\n".join(lines)


def render_selected_target_crosscheck_markdown(summary: pd.DataFrame) -> str:
    """Render selected-target cross-check summary."""

    lines = [
        "# Historical Replay Selected Target Wiring Check",
        "",
        "This checks that tdcsim loads the one selected TDC-EST target per quarter.",
        "It is not portfolio replay validation.",
        "Values are reported in billions of U.S. dollars.",
        "",
    ]
    if summary.empty:
        lines.append("No comparison rows were produced.")
        return "\n".join(lines) + "\n"
    row = summary.iloc[0]
    lines.append(f"- Compared rows: {int(row['compared_rows'])}")
    lines.append(f"- Matched rows: {int(row['matched_rows'])}")
    lines.append(f"- Missing selected target rows: {int(row['missing_selected_target_rows'])}")
    lines.append(f"- Mismatch rows: {int(row['mismatch_rows'])}")
    if pd.notna(row["first_compared_date"]):
        first = pd.Timestamp(row["first_compared_date"]).date().isoformat()
        last = pd.Timestamp(row["last_compared_date"]).date().isoformat()
        lines.append(f"- Compared date range: {first} to {last}")
    if pd.notna(row["max_abs_difference_bil"]):
        lines.append(f"- Max absolute difference: {float(row['max_abs_difference_bil']):,.12f}")
    lines.append(f"- Selected method keys: {row['method_keys']}")
    if "validation_role" in row and pd.notna(row["validation_role"]):
        lines.append(f"- Validation role: {row['validation_role']}")
    return "\n".join(lines) + "\n"


def _comparison_verdict(replay_value, tdcest_value, *, tolerance: float) -> str:
    if pd.isna(tdcest_value):
        return "no_tdcest_value"
    if pd.isna(replay_value):
        return "no_tdcsim_value"
    difference = float(replay_value) - float(tdcest_value)
    if abs(difference) <= tolerance:
        return "matched"
    if abs(float(replay_value)) <= tolerance and abs(float(tdcest_value)) > tolerance:
        return "tdcsim_replay_tdc_not_implemented"
    return "mismatch"


def _selected_verdict(replay_value, selected_value, *, tolerance: float) -> str:
    if pd.isna(selected_value):
        return "missing_selected_target"
    if pd.isna(replay_value):
        return "no_tdcsim_value"
    if abs(float(replay_value) - float(selected_value)) <= tolerance:
        return "matched"
    return "mismatch"


__all__ = [
    "DEFAULT_TDCEST_COLUMNS",
    "build_selected_target_crosscheck",
    "build_tdcest_crosscheck",
    "load_tdcest_estimates",
    "render_selected_target_crosscheck_markdown",
    "render_tdcest_crosscheck_markdown",
    "summarize_selected_target_crosscheck",
    "summarize_tdcest_crosscheck",
]
