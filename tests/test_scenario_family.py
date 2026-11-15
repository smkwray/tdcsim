"""Scenario families: holder-share shifts and the verified fiscal-year export."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from write_scenario_family import (  # noqa: E402
    FAMILY_SPEC_SCHEMA,
    bill_holder_rows,
    bill_share_target_path,
    family_scenarios,
    rate_path_surface,
    shifted_holder_rows,
)

BASE = {
    "bills": {"Banks": 0.2, "CB": 0.0, "FedInternal": 0.0, "Foreign": 0.3, "Private": 0.5, "TrustFunds": 0.0},
    "bonds": {"Banks": 0.1, "CB": 0.0, "FedInternal": 0.0, "Foreign": 0.25, "Private": 0.65, "TrustFunds": 0.0},
}


def test_shift_moves_only_the_released_foreign_share() -> None:
    rows = {row["security_type"]: row["shares"] for row in shifted_holder_rows(
        BASE, holder="Foreign", multiplier=0.6, absorbers={"Private": 0.75, "Banks": 0.25})}
    assert rows["bills"]["Foreign"] == pytest.approx(0.18)
    assert rows["bills"]["Private"] == pytest.approx(0.5 + 0.12 * 0.75)
    assert rows["bills"]["Banks"] == pytest.approx(0.2 + 0.12 * 0.25)
    for shares in rows.values():
        assert sum(shares.values()) == pytest.approx(1.0)
        assert shares["CB"] == 0.0


def test_per_type_multipliers_release_official_shares_by_security() -> None:
    """Official holders own a smaller share of foreign bills than of coupons, so the cuts differ by type."""

    rows = {row["security_type"]: row["shares"] for row in shifted_holder_rows(
        BASE, holder="Foreign", multiplier={"bills": 0.9, "bonds": 0.5}, absorbers={"Banks": 1.0})}
    assert rows["bills"]["Foreign"] == pytest.approx(0.27)
    assert rows["bonds"]["Foreign"] == pytest.approx(0.125)
    assert rows["bonds"]["Banks"] == pytest.approx(0.1 + 0.125)
    with pytest.raises(ValueError, match="exactly"):
        shifted_holder_rows(BASE, holder="Foreign", multiplier={"bills": 0.9}, absorbers={"Banks": 1.0})


def _stocks(foreign_by_date: dict[str, float]) -> list[tuple[str, dict[str, float]]]:
    # Total marketable debt of 1,000 so that a shortfall of 10 is one percentage point.
    return [(day, {"Banks": 100.0, "CB": 100.0, "Foreign": value, "Private": 800.0 - value})
            for day, value in sorted(foreign_by_date.items())]


def test_rate_path_scales_with_shortfall_share_without_look_ahead() -> None:
    surface = [
        {"curve_date": day, "tenor_years": str(tenor), "nominal_rate": "4.0", "nominal_rate_decimal": "0.04"}
        for day in ("2026-04-01", "2026-07-01", "2026-10-01")
        for tenor in (1.0, 2.0, 5.0, 10.0, 30.0)
    ]
    baseline = _stocks({"2026-06-21": 300.0, "2026-09-30": 300.0, "2026-10-01": 300.0})
    # The shortfall is 10 (1 pp) by July and 20 (2 pp) only after the October curve date.
    source = _stocks({"2026-06-21": 300.0, "2026-06-30": 290.0, "2026-10-01": 290.0, "2026-10-02": 280.0})
    rows, path = rate_path_surface(
        surface, baseline_stocks=baseline, source_stocks=source, holder="Foreign", bp_per_share_point=7.0,
        key_rate_weights=[{"tenor_years": 2.0, "weight": 0.0}, {"tenor_years": 5.0, "weight": 0.5},
                          {"tenor_years": 10.0, "weight": 1.0}],
    )
    shift = {(row["curve_date"], float(row["tenor_years"])): (float(row["nominal_rate"]) - 4.0) * 100.0 for row in rows}
    assert [round(item["shift_10y_bp"], 9) for item in path] == [0.0, 7.0, 7.0]
    assert shift[("2026-07-01", 10.0)] == pytest.approx(7.0)
    assert shift[("2026-07-01", 30.0)] == pytest.approx(7.0)
    assert shift[("2026-07-01", 5.0)] == pytest.approx(3.5)
    assert shift[("2026-07-01", 2.0)] == shift[("2026-07-01", 1.0)] == 0.0
    for row in rows:
        assert float(row["nominal_rate_decimal"]) * 100.0 == pytest.approx(float(row["nominal_rate"]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"holder": "Foreign", "multiplier": 1.2, "absorbers": {"Private": 1.0}},
        {"holder": "Foreign", "multiplier": 0.5, "absorbers": {"Private": 0.5}},
        {"holder": "Foreign", "multiplier": 0.5, "absorbers": {"Foreign": 1.0}},
        {"holder": "Foreign", "multiplier": 0.5, "absorbers": {"Hedge": 1.0}},
    ],
)
def test_invalid_shift_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        shifted_holder_rows(BASE, **kwargs)


def _family(retreat_bp: float) -> dict:
    return {
        "schema_version": FAMILY_SPEC_SCHEMA,
        "family_id": "test_family",
        "as_of_date": "2026-10-08",
        "description": "test",
        "boundaries": ["test boundary"],
        "scenarios": [
            {"role": "baseline", "scenario_id": "test_family_baseline_v1", "label": "Baseline"},
            {
                "role": "retreat",
                "scenario_id": "test_family_retreat_v1",
                "label": "Foreign retreat",
                "holder_shift": {"holder": "Foreign", "multiplier": 0.5, "absorbers": {"Private": 1.0}},
                "key_rate_shocks_bp": [{"tenor_years": 2.0, "shock_bp": 0.0}, {"tenor_years": 10.0, "shock_bp": retreat_bp}],
            },
        ],
    }


def test_family_runs_verify_and_export(tmp_path: Path, monkeypatch) -> None:
    """Quantity runs feed the rate-path scenario; every role verifies and exports with its 10-year yield."""

    import tdcsim_cbo.runner as runner_module
    from tdcsim_cbo import CboScenarioSpec, run_cbo_scenario
    from tdcsim_cbo.scenario_family_export import ScenarioFamilyExportError, export_scenario_family
    from tdcsim_cbo.verifier import verify_scenario_run
    from test_tdcsim_cbo_closeout_interface import _runner_baseline_and_scenarios
    from test_trc_routing import _routing

    monkeypatch.setattr(runner_module, "host_available_memory_bytes", lambda: 1 << 50)
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    simulation = json.loads(Path(scenarios["noop"]).read_text()).get("simulation")
    family = _family(25.0)
    # The fixture's auctions go entirely to private holders, so the derived path shifts them.
    private_shift = {"holder": "Private", "multiplier": 0.5, "absorbers": {"Banks": 1.0}}
    family["scenarios"] += [
        {"role": "shift", "scenario_id": "test_family_shift_v1", "label": "Holder shift", "holder_shift": private_shift},
        {
            "role": "shift_rates",
            "scenario_id": "test_family_shift_rates_v1",
            "label": "Holder shift with a derived rate path",
            "holder_shift": private_shift,
            "rate_path": {"from_role": "shift", "bp_per_share_point": 7.0,
                          "key_rate_weights": [{"tenor_years": 2.0, "weight": 0.0}, {"tenor_years": 10.0, "weight": 1.0}]},
        },
    ]
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    written = family_scenarios(baseline, family)
    assert list(written) == ["00_baseline.json", "01_retreat.json", "02_shift.json"]
    retreat = written["01_retreat.json"]
    assert retreat["overrides"]["nominal_yield_curve"]["shocks"][-1]["shock_bp"] == 25.0
    assert retreat["provenance"]["kind"] == "external_source_assumption"
    with pytest.raises(ValueError, match="source runs"):
        family_scenarios(baseline, family, phase="rates")

    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps(_routing(on_rrp_opening=1_000_000.0)))
    runs = {}

    def run_written(items: dict) -> None:
        for name, data in items.items():
            role = name.split("_", 1)[1].removesuffix(".json")
            if simulation:
                data = {**data, "simulation": simulation}
            path = scenario_dir / name
            path.write_text(json.dumps(data))
            run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(path), tmp_path / "runs" / role / "run",
                                   trc_routing_path=routing)
            runs[role] = {
                "run_dir": run.output_dir,
                "scenario_id": data["scenario_id"],
                "scenario_sha256": "0" * 64,
                "verification": verify_scenario_run(run.output_dir),
            }

    run_written(written)
    # The fixture curve has one date before the window; add one inside it so the derived
    # path can respond to the shortfall that builds during the run.
    surface = tmp_path / "runs" / "baseline" / "run" / "compile" / "compiled" / "forecast_inputs" / "tdcsim_yield_curve_surface.csv"
    with surface.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header, curve = reader.fieldnames, list(reader)
    with surface.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=header, lineterminator="\n").writerows(
            {**row, "curve_date": "2026-09-27"} for row in curve)
    rated = family_scenarios(baseline, family, phase="rates", runs_root=tmp_path / "runs", output_dir=scenario_dir)
    assert list(rated) == ["03_shift_rates.json"]
    assert rated["03_shift_rates.json"]["overrides"]["nominal_yield_curve"]["mode"] == "full_surface_file"
    run_written(rated)
    receipt = export_scenario_family(
        family_spec=family, family_spec_sha256="1" * 64, runs=runs,
        output_dir=tmp_path / "export", producer={"repository_commit": "a" * 40},
    )
    with (tmp_path / "export" / "scenario_family_fy.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["scenario_role"] for row in rows} == {"baseline", "retreat", "shift", "shift_rates"}
    for column in ("cumulative_tdc_change_bil", "cumulative_modeled_financing_cost_bil", "cumulative_trc_bil",
                   "holdings_foreign_end_bil", "foreign_gross_settlement_cash_bil", "nominal_10y_end_pct"):
        assert all(row[column] != "" for row in rows), column
    by = {(row["scenario_role"], row["period_label"]): row for row in rows}
    rate_shifts = []
    for (role, label), row in by.items():
        shift = float(row["nominal_10y_end_pct"]) - float(by[("baseline", label)]["nominal_10y_end_pct"])
        if role == "retreat":
            assert shift == pytest.approx(0.25)
        if role == "shift_rates":
            rate_shifts.append(shift)
    # The derived path is the rule applied to the quantity run's shortfall at each curve date.
    shift_path = json.loads((scenario_dir / "shift_rates_rate_path.json").read_text())["shift_path"]
    assert max(item["shift_10y_bp"] for item in shift_path) > 0.0
    for item in shift_path:
        assert item["shift_10y_bp"] == pytest.approx(7.0 * 100.0 * item["shortfall_share"])
    assert max(rate_shifts) > 0.0
    assert {round(100.0 * s, 9) for s in rate_shifts} <= {round(item["shift_10y_bp"], 9) for item in shift_path}
    assert all(receipt["roles"][role]["verification_status"] == "pass" for role in runs)

    runs["retreat"]["verification"] = {"status": "fail"}
    with pytest.raises(ScenarioFamilyExportError, match="verification"):
        export_scenario_family(
            family_spec=family, family_spec_sha256="1" * 64, runs=runs,
            output_dir=tmp_path / "export2", producer={"repository_commit": "a" * 40},
        )


def test_runner_gives_each_role_its_own_parent(tmp_path: Path, monkeypatch) -> None:
    """The bounded runner claims a run's parent directory, so parallel roles need distinct parents."""

    import argparse

    import run_scenario_family as runner

    family = _family(0.0)
    family["scenarios"].append({"role": "later", "scenario_id": "later", "label": "Not yet written"})
    spec = tmp_path / "family.json"
    spec.write_text(json.dumps(family))
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    for index, item in enumerate(family["scenarios"][:2]):
        (scenario_dir / f"{index:02d}_{item['role']}.json").write_text("{}")
    launched = []

    class FakeProcess:
        pid = 1

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        launched.append(Path(command[command.index("--output-dir") + 1]))
        return FakeProcess()

    monkeypatch.setattr(runner, "_assert_installed_distribution_import", lambda: None)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    args = argparse.Namespace(family_spec=str(spec), scenario_dir=str(scenario_dir),
                              output_root=str(tmp_path / "runs"), trc_routing=str(spec), roles=None)
    from run_open04_reproduction import ReproductionError

    with pytest.raises(ReproductionError, match="missing"):
        runner.stage_run(args)
    args.roles = "baseline,retreat"
    assert runner.stage_run(args) == 0
    parents = [path.parent for path in launched]
    assert len(set(parents)) == len(parents) == 2
    assert all(path.name == "run" and not path.exists() and path.parent.is_dir() for path in launched)



def test_funding_closure_rules() -> None:
    from tdcsim_cbo.open04_campaign import Open04CampaignError, scenario_funding_closure_mode

    # Undeclared scenarios keep the TGA on its operating-cash path; debt-path modes stay selectable.
    assert scenario_funding_closure_mode({}) == "cbo_operating_cash_target_financing_v1"
    floor = "cbo_debt_reference_plus_tga_floor_financing_v1"
    assert scenario_funding_closure_mode({"funding_closure": {"mode": floor}}) == floor
    exact = "cbo_public_debt_target"
    assert scenario_funding_closure_mode({"funding_closure": {"mode": exact}}) == exact
    for bad in ({"funding_closure": {"mode": "plug"}}, {"funding_closure": {"mode": floor, "x": 1}}):
        with pytest.raises(Open04CampaignError):
            scenario_funding_closure_mode(bad)


def test_declared_closure_is_recorded_replayed_and_bound(tmp_path: Path, monkeypatch) -> None:
    """A rate-shock family needs TGA-floor financing; the verifier must replay the declared closure."""

    import tdcsim_cbo.runner as runner_module
    from tdcsim_cbo import CboScenarioSpec, run_cbo_scenario
    from tdcsim_cbo._json import read_json, write_json
    from tdcsim_cbo.verifier import VerificationError, verify_scenario_run
    from test_tdcsim_cbo_closeout_interface import _runner_baseline_and_scenarios

    monkeypatch.setattr(runner_module, "host_available_memory_bytes", lambda: 1 << 50)
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    simulation = json.loads(Path(scenarios["noop"]).read_text()).get("simulation")
    family = {**_family(25.0), "funding_closure_mode": "cbo_debt_reference_plus_tga_floor_financing_v1"}
    data = family_scenarios(baseline, family)["01_retreat.json"]
    assert data["funding_closure"] == {"mode": "cbo_debt_reference_plus_tga_floor_financing_v1"}
    if simulation:
        data = {**data, "simulation": simulation}
    path = tmp_path / "retreat.json"
    path.write_text(json.dumps(data))
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(path), tmp_path / "runs" / "retreat" / "run")
    manifest_path = run.output_dir / "tdcsim_cbo_run_manifest.json"
    manifest = read_json(manifest_path)
    assert manifest["funding_closure_mode"] == "cbo_debt_reference_plus_tga_floor_financing_v1"
    assert verify_scenario_run(run.output_dir)["status"] == "pass"
    manifest["funding_closure_mode"] = "cbo_public_debt_target"
    write_json(manifest_path, manifest)
    with pytest.raises(VerificationError):
        verify_scenario_run(run.output_dir)


def _bill_dollars(shares: dict[str, float], target: float) -> dict[str, float]:
    return {holder: share * target for holder, share in shares.items()}


@pytest.mark.parametrize("change", [0.05, -0.05])
def test_only_the_named_holders_take_up_a_change_in_the_bill_stock(change) -> None:
    # The deposit effect of an issuance change depends on who buys or gives up the bills.
    # Every other holder must keep its baseline bill holdings, or the scenario would
    # mix a holder-composition change into the issuance change.
    base_target = 0.225
    path = bill_share_target_path(
        base_target, [{"date": "2026-06-21", "change": 0.0}, {"date": "2027-06-30", "change": change}]
    )
    rows = bill_holder_rows(BASE["bills"], base_target=base_target, target_path=path, absorbers=["Banks", "Foreign"])

    assert rows[0]["effective_date"] == "2026-06-21"
    assert rows[-1]["effective_date"] == "2027-06-30"
    assert {row["security_type"] for row in rows} == {"bills"}
    final = rows[-1]["shares"]
    baseline = _bill_dollars(BASE["bills"], base_target)
    scenario = _bill_dollars(final, base_target + change)
    assert sum(final.values()) == pytest.approx(1.0)
    assert scenario["Private"] == pytest.approx(baseline["Private"])
    absorbed = {holder: scenario[holder] - baseline[holder] for holder in ("Banks", "Foreign")}
    assert sum(absorbed.values()) == pytest.approx(change)
    # Absorbers split the change in proportion to their baseline bill shares.
    assert absorbed["Banks"] / absorbed["Foreign"] == pytest.approx(0.2 / 0.3)


def test_bill_holder_shift_rejects_a_cut_larger_than_the_absorbers_hold() -> None:
    path = bill_share_target_path(0.225, [{"date": "2026-06-21", "change": -0.2}])
    with pytest.raises(ValueError, match="hold too few bills"):
        bill_holder_rows(BASE["bills"], base_target=0.225, target_path=path, absorbers=["Banks"])


def test_bill_share_changes_are_relative_to_the_package_opening_share() -> None:
    # The family spec states changes, so the opening share is read from the package rather than restated.
    path = bill_share_target_path(0.2252, [{"date": "2026-06-21", "change": 0.0}, {"date": "2027-06-30", "change": 0.05}])
    assert path == [
        {"date": "2026-06-21", "bill_share_target": 0.2252},
        {"date": "2027-06-30", "bill_share_target": pytest.approx(0.2752)},
    ]
    with pytest.raises(ValueError, match="exactly date and change"):
        bill_share_target_path(0.2252, [{"date": "2026-06-21", "target": 0.3}])
