"""The OPEN-04 TRC companion export refuses any revision of the frozen legacy outputs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from settlement_routes import TRC_FAMILIES, TRC_RESULT_COLUMNS
from tdcsim_cbo.open04_campaign import OPEN04_AGGREGATION_CLOCK_ID, OPEN04_CAMPAIGN_ROLES
from tdcsim_cbo.open04_trc_export import Open04TrcExportError, export_open04_trc_companion

DATES = ["2026-06-21", "2026-09-30", "2027-09-30"]


def _write_gz(path: Path, frame: pd.DataFrame) -> None:
    # Deterministic gzip header (mtime 0), as the production writer emits.
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(frame.to_csv(index=False).encode("utf-8"))


def _campaign(root: Path, campaign_id: str, *, with_trc: bool) -> Path:
    roles = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        relative = f"{role}-role/{role}-run"
        outputs = root / relative / "outputs"
        outputs.mkdir(parents=True)
        # Holder-routed legacy path: +10 then +5 reserve change; 2 of the first
        # period's change is an ON RRP placement, not a reserve posting.
        _write_gz(outputs / "results_compact.csv.gz",
                  pd.DataFrame({"Date": DATES, "Reserves": [100.0, 110.0, 115.0], "TDC_Change": [0.0, 3.0, 4.0]}))
        _write_gz(outputs / "tdcsim_annual_economic_summary.csv.gz", pd.DataFrame({
            "period_label": ["FY2026_PARTIAL_OPENING", "FY2027"],
            "period_start": DATES[:2],
            "period_end": DATES[1:],
            "is_partial_period": [True, False],
            "aggregation_clock_id": [OPEN04_AGGREGATION_CLOCK_ID] * 2,
        }))
        _write_gz(outputs / "tdcsim_resource_samples.csv.gz", pd.DataFrame({"rss": [hash(campaign_id) % 97]}))
        (outputs / "summary.json").write_text(json.dumps({"rows": 3, "peak_rss_bytes": len(campaign_id)}))
        if with_trc:
            rows = []
            for index, (date, trc, state) in enumerate(zip(DATES, [0.0, 8.0, 5.0], [0.0, 2.0, 0.0])):
                row = {column: 0.0 for column in TRC_RESULT_COLUMNS}
                row.update({"Date": date, "TRC_DebtService": trc, "TRC_Change": trc,
                            "TRC_RouteStateChange": state, "LegacyReserveChange_HolderRouted": trc + state})
                rows.append(row)
            frame = pd.DataFrame(rows, columns=list(TRC_RESULT_COLUMNS))
            frame["TRC_Cumulative"] = frame["TRC_Change"].cumsum()
            frame["ONRRP_MMF"] = 1.0 + frame["TRC_RouteStateChange"].cumsum()
            frame["ForeignOfficialFedDeposits"] = 9.0
            frame["ForeignOfficialFedRRP"] = 300.0
            frame["SyntheticReserveBalance_ExNonTreasuryFactors"] = 100.0 + frame["TRC_Cumulative"]
            _write_gz(outputs / "tdcsim_trc_results.csv.gz", frame)
            (outputs / "tdcsim_trc_metadata.json").write_text(json.dumps({
                "perimeter": "aggregate_reserve_balances_at_federal_reserve_banks",
                "routing": {"schema_version": "tdcsim_trc_routing_v1"},
                "routing_source": {"sha256": "0" * 64},
            }))
        roles[role] = {"run_relative_path": relative, "scenario_id": f"{role}_scenario"}
    (root / "open04_campaign_contract.json").write_text(json.dumps({"campaign_id": campaign_id, "roles": roles}))
    return root


def test_export_requires_legacy_identity_and_writes_fy_paths(tmp_path: Path) -> None:
    frozen = _campaign(tmp_path / "frozen", "frozen-campaign", with_trc=False)
    companion = _campaign(tmp_path / "companion", "companion-campaign", with_trc=True)
    receipt = export_open04_trc_companion(
        companion_campaign_root=companion,
        frozen_campaign_root=frozen,
        output_dir=tmp_path / "out",
        producer_commit="a" * 40,
    )
    table = pd.read_csv(tmp_path / "out" / "open04_trc_paths_fy.csv")
    assert len(table) == 2 * len(OPEN04_CAMPAIGN_ROLES)
    baseline = table[table.scenario_role == "baseline"].set_index("period_label")
    assert baseline.loc["FY2026_PARTIAL_OPENING", "trc_change_bil"] == pytest.approx(8.0)
    assert baseline.loc["FY2027", "cumulative_trc_bil"] == pytest.approx(13.0)
    # Legacy holder routing overstates reserve incidence by the ON RRP placement.
    assert baseline.loc["FY2026_PARTIAL_OPENING", "legacy_holder_routed_reserve_change_bil"] == pytest.approx(10.0)
    assert {f"trc_{family}" for family in ("debt_service", "fiscal_flow")} <= {
        column.removesuffix("_bil") for column in table.columns
    }
    assert len(TRC_FAMILIES) == 8
    for role in OPEN04_CAMPAIGN_ROLES:
        assert receipt["roles"][role]["legacy_identity"]["byte_identical_files"]


def test_export_rejects_any_revised_legacy_output(tmp_path: Path) -> None:
    frozen = _campaign(tmp_path / "frozen", "frozen-campaign", with_trc=False)
    companion = _campaign(tmp_path / "companion", "companion-campaign", with_trc=True)
    revised = companion / "candidate_a-role" / "candidate_a-run" / "outputs" / "results_compact.csv.gz"
    _write_gz(revised, pd.DataFrame({"Date": DATES, "Reserves": [100.0, 110.0, 115.0], "TDC_Change": [0.0, 3.0, 4.5]}))
    with pytest.raises(Open04TrcExportError, match="differs from the frozen campaign"):
        export_open04_trc_companion(
            companion_campaign_root=companion,
            frozen_campaign_root=frozen,
            output_dir=tmp_path / "out",
            producer_commit="a" * 40,
        )
    assert not (tmp_path / "out").exists()
