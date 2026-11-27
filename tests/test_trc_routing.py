"""Treasury component of reserves (TRC): route registry, ledger and engine.

Reserve incidence must follow the settlement claim, not the holder's legal
sector, and adding TRC must leave every legacy TDC and reserve output unchanged.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from settlement_routes import (
    SETTLEMENT_ROUTES,
    TRC_ROUTE_STATES,
    TRC_ROUTING_SCHEMA_VERSION,
    TrcLedger,
    TrcRoutingError,
    parse_trc_routing,
)
from simulation_core import run_simulation
from tdc_shared import PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, PRIVATE_SUBBUCKET_MMF
from test_mmf_split import _split_params


def _routing(
    *,
    bank: float = 1.0,
    official_deposit: float = 0.0,
    official_rrp: float = 0.0,
    on_rrp_opening: float = 1_000.0,
    on_rrp_share: float = 0.0,
) -> dict:
    return {
        "schema_version": TRC_ROUTING_SCHEMA_VERSION,
        "foreign_settlement_mix": {
            "foreign_bank_settled": bank,
            "foreign_official_deposit": official_deposit,
            "foreign_official_rrp": official_rrp,
        },
        "mmf_cash_fund_route": {"on_rrp_share": on_rrp_share, "source": "test fixture"},
        "opening_fed_nonreserve_balances_bil": {
            "ONRRP_MMF": on_rrp_opening,
            "ForeignOfficialFedDeposits": 1_000.0,
            "ForeignOfficialFedRRP": 1_000.0,
        },
        "opening_balance_sources": {state: "test fixture" for state in TRC_ROUTE_STATES},
    }


def _ledger(pass_through: float = 1.0, **kwargs) -> TrcLedger:
    policy = parse_trc_routing(_routing(**kwargs), mmf_deposit_pass_through=pass_through)
    return TrcLedger(policy, opening_reserves=100.0)


def _trc(ledger: TrcLedger) -> float:
    return sum(ledger._period.values())


# --- Route table: each settlement route gives the reserve sign it should. ---


def test_spending_and_taxes_move_reserves_one_for_one() -> None:
    ledger = _ledger()
    ledger.record_public_cash("FiscalFlow", 1.0, "covered_deposit_settled")
    assert _trc(ledger) == pytest.approx(1.0)
    ledger.record_public_cash("FiscalFlow", -1.0, "covered_deposit_settled")
    assert _trc(ledger) == pytest.approx(0.0)


def test_auction_purchases_drain_reserves_only_when_reserve_settled() -> None:
    bank = _ledger()
    bank.record_holder_cash("AuctionSettlement", {"Banks": -1.0})
    assert _trc(bank) == pytest.approx(-1.0)

    deposit = _ledger()
    deposit.record_holder_cash(
        "AuctionSettlement", {"Private": -1.0}, {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: -1.0}
    )
    assert _trc(deposit) == pytest.approx(-1.0)

    on_rrp = _ledger(pass_through=0.0, on_rrp_share=1.0)
    on_rrp.record_holder_cash("AuctionSettlement", {"Private": -1.0}, {PRIVATE_SUBBUCKET_MMF: -1.0})
    assert _trc(on_rrp) == pytest.approx(0.0)
    assert on_rrp._state_delta["ONRRP_MMF"] == pytest.approx(-1.0)

    fed_addon = _ledger()
    fed_addon.record_holder_cash("AuctionSettlement", {"CB": -1.0})
    assert _trc(fed_addon) == pytest.approx(0.0)


def test_same_foreign_holder_differs_by_settlement_claim() -> None:
    """Holder sector alone cannot assign reserve incidence."""

    bank_settled = _ledger(bank=1.0)
    official = _ledger(bank=0.0, official_deposit=1.0)
    for ledger in (bank_settled, official):
        ledger.record_holder_cash("AuctionSettlement", {"Foreign": -1.0})
    assert _trc(bank_settled) == pytest.approx(-1.0)
    assert _trc(official) == pytest.approx(0.0)
    assert official._state_delta["ForeignOfficialFedDeposits"] == pytest.approx(-1.0)


def test_debt_service_routes() -> None:
    ledger = _ledger(pass_through=0.0, on_rrp_share=1.0, bank=0.5, official_rrp=0.5)
    ledger.record_holder_cash(
        "DebtService",
        {"Banks": 1.0, "Private": 2.0, "Foreign": 1.0, "CB": 5.0, "TrustFunds": 3.0},
        {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 1.0, PRIVATE_SUBBUCKET_MMF: 1.0},
    )
    # Bank +1, covered deposit +1, MMF placed into ON RRP 0, foreign half bank-settled +0.5,
    # SOMA and trust funds 0.
    assert _trc(ledger) == pytest.approx(2.5)
    assert ledger._state_delta["ONRRP_MMF"] == pytest.approx(1.0)
    assert ledger._state_delta["ForeignOfficialFedRRP"] == pytest.approx(0.5)


def test_soma_secondary_trades_and_holder_trades() -> None:
    ledger = _ledger()
    ledger.record_holder_cash("FedSecondary", {"Banks": 1.0})
    assert _trc(ledger) == pytest.approx(1.0)
    ledger.record_holder_cash("FedSecondary", {"Private": 1.0}, {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 1.0})
    assert _trc(ledger) == pytest.approx(2.0)
    ledger.record_holder_cash("FedSecondary", {"Banks": -1.0})
    assert _trc(ledger) == pytest.approx(1.0)

    trades = _ledger()
    trades.record_trade(
        "SecondaryTrades",
        buyer="Banks",
        buyer_private_route=None,
        seller="Private",
        seller_private_route=PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        amount=1.0,
    )
    assert _trc(trades) == pytest.approx(0.0)
    trades.record_trade(
        "SecondaryTrades",
        buyer="CB",
        buyer_private_route=None,
        seller="Banks",
        seller_private_route=None,
        amount=1.0,
    )
    assert _trc(trades) == pytest.approx(1.0)


def test_registry_reproduces_tdcsim_deposit_routing() -> None:
    """The route registry's deposit membership matches the engine's TDC routing."""

    policy = parse_trc_routing(_routing(), mmf_deposit_pass_through=1.0)
    assert policy.covered_deposit_share("Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK) == 1.0
    assert policy.routes_for("Private", PRIVATE_SUBBUCKET_MMF) == (("covered_deposit_settled", 1.0),)
    for holder in ("Banks", "Foreign", "CB", "FedInternal", "TrustFunds"):
        assert policy.covered_deposit_share(holder) == 0.0
    assert SETTLEMENT_ROUTES["tt_and_l"].reserve_settled_member == 1.0
    assert SETTLEMENT_ROUTES["tt_and_l"].covered_deposit_member == 0.0


@pytest.mark.parametrize("pass_through", [0.5, 0.9, 0.97])
def test_money_fund_cash_uses_exactly_the_two_declared_routes(pass_through) -> None:
    on_rrp_share = 1.0 - pass_through
    policy = parse_trc_routing(_routing(on_rrp_share=on_rrp_share), mmf_deposit_pass_through=pass_through)
    assert policy.routes_for("Private", PRIVATE_SUBBUCKET_MMF) == (
        ("covered_deposit_settled", pass_through),
        ("on_rrp_mmf", on_rrp_share),
    )
    assert policy.reserve_share("Private", PRIVATE_SUBBUCKET_MMF) == pass_through
    for bad_share in (on_rrp_share / 2, on_rrp_share * 1.01):
        with pytest.raises(TrcRoutingError, match="must sum to one"):
            parse_trc_routing(_routing(on_rrp_share=bad_share), mmf_deposit_pass_through=pass_through)


# --- Policy validation fails closed. ---


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: cfg["foreign_settlement_mix"].update(foreign_bank_settled=0.9),
        lambda cfg: cfg.update(extra=1),
        lambda cfg: cfg["opening_balance_sources"].update(ONRRP_MMF=""),
        lambda cfg: cfg["opening_fed_nonreserve_balances_bil"].update(ONRRP_MMF=-1.0),
        lambda cfg: cfg["mmf_cash_fund_route"].update(basis="complement_of_mmf_deposit_pass_through"),
        lambda cfg: cfg["mmf_cash_fund_route"].update(source=""),
        lambda cfg: cfg["mmf_cash_fund_route"].update(on_rrp_share=-0.1),
        lambda cfg: cfg.update(schema_version="v0"),
    ],
)
def test_invalid_routing_policy_is_rejected(mutate) -> None:
    config = _routing()
    mutate(config)
    with pytest.raises(TrcRoutingError):
        parse_trc_routing(config, mmf_deposit_pass_through=0.97)


# --- Ledger closure and capacity. ---


def test_closure_identity_and_capacity() -> None:
    ledger = _ledger(pass_through=0.0, on_rrp_share=1.0, on_rrp_opening=0.5)
    ledger.record_holder_cash("DebtService", {"Private": 1.0}, {PRIVATE_SUBBUCKET_MMF: 1.0})
    # Legacy holder routing booked +1 reserve; TRC is 0 and ON RRP rose by 1.
    row = ledger.close_period("2026-01-02", legacy_reserve_change=1.0, legacy_reserves=101.0)
    assert row["TRC_Change"] == pytest.approx(0.0)
    assert row["ONRRP_MMF"] == pytest.approx(1.5)
    assert row["SyntheticReserveBalance_ExNonTreasuryFactors"] == pytest.approx(100.0)

    ledger.record_holder_cash("AuctionSettlement", {"Private": -1.0}, {PRIVATE_SUBBUCKET_MMF: -1.0})
    with pytest.raises(TrcRoutingError, match="closure"):
        ledger.close_period("2026-01-03", legacy_reserve_change=0.0, legacy_reserves=101.0)

    short = _ledger(pass_through=0.0, on_rrp_share=1.0, on_rrp_opening=0.5)
    short.record_holder_cash("AuctionSettlement", {"Private": -1.0}, {PRIVATE_SUBBUCKET_MMF: -1.0})
    with pytest.raises(TrcRoutingError, match="capacity"):
        short.close_period("2026-01-02", legacy_reserve_change=-1.0, legacy_reserves=99.0)


# --- Engine integration on the existing MMF-split fixture. ---


def _run(tmp_path: Path, *, p: float, routing: dict | None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    params = _split_params(tmp_path, p=p)
    if routing is not None:
        params["trc_routing"] = copy.deepcopy(routing)
    results, portfolio = run_simulation(
        params, "2025-01-01", "2025-01-20", freq="W", scenario_name="mmf_split_regression"
    )
    return results, portfolio


def test_trc_mode_leaves_every_legacy_output_identical(tmp_path: Path) -> None:
    legacy, legacy_portfolio = _run(tmp_path / "a", p=1.0, routing=None)
    routed, routed_portfolio = _run(tmp_path / "b", p=1.0, routing=_routing())
    pd.testing.assert_frame_equal(routed, legacy, check_exact=True)
    pd.testing.assert_frame_equal(routed_portfolio, legacy_portfolio, check_exact=True)
    assert "trc_results" not in legacy.attrs
    trc = routed.attrs["trc_results"]
    assert len(trc) == len(routed)


def test_engine_trc_closes_to_the_legacy_reserve_path(tmp_path: Path) -> None:
    results, _ = _run(tmp_path, p=0.97, routing=_routing(on_rrp_share=0.03))
    trc = results.attrs["trc_results"].set_index("Date")
    legacy_change = results["ReserveChange"].to_numpy()[1:]
    trc_change = trc["TRC_Change"].to_numpy()[1:]
    state_change = trc["TRC_RouteStateChange"].to_numpy()[1:]
    assert legacy_change == pytest.approx(trc_change + state_change, abs=1e-9)
    # MMF settlements through ON RRP make TRC differ from the holder-routed path.
    assert abs(trc["ONRRP_MMF"].iloc[-1] - trc["ONRRP_MMF"].iloc[0]) > 0.0
    reserves = results["Reserves"].to_numpy()
    balances = trc["SyntheticReserveBalance_ExNonTreasuryFactors"].to_numpy() + (
        trc["ONRRP_MMF"].to_numpy() - trc["ONRRP_MMF"].iloc[0]
    )
    assert reserves == pytest.approx(balances, abs=1e-9)
    components = trc[[column for column in trc.columns if column.startswith("TRC_")
                      and column not in {"TRC_Change", "TRC_Cumulative", "TRC_RouteStateChange",
                                         "TRC_ForeignGrossSettlementCash", "TRC_LegacyClosureGap",
                                         "TRC_LegacyAliasGap"}]].sum(axis=1)
    assert components.to_numpy() == pytest.approx(trc["TRC_Change"].to_numpy(), abs=1e-12)


def test_full_pass_through_and_bank_settled_foreign_equal_legacy_reserves(tmp_path: Path) -> None:
    results, _ = _run(tmp_path, p=1.0, routing=_routing())
    trc = results.attrs["trc_results"]
    assert trc["TRC_Change"].to_numpy()[1:] == pytest.approx(
        results["ReserveChange"].to_numpy()[1:], abs=1e-9
    )
    assert trc["ONRRP_MMF"].nunique() == 1


def test_engine_fails_closed_when_on_rrp_capacity_is_exceeded(tmp_path: Path) -> None:
    ample, _ = _run(tmp_path / "ample", p=0.0, routing=_routing(on_rrp_opening=1_000.0, on_rrp_share=1.0))
    path = ample.attrs["trc_results"]["ONRRP_MMF"]
    drawdown = 1_000.0 - float(path.min())
    assert drawdown > 0.0, "fixture must route money-fund purchases through ON RRP"
    with pytest.raises(TrcRoutingError, match="capacity"):
        _run(tmp_path / "short", p=0.0, routing=_routing(on_rrp_opening=drawdown / 2.0, on_rrp_share=1.0))


# --- Output writer. ---


def test_trc_outputs_are_written_only_when_present(tmp_path: Path) -> None:
    from tdcsim_cbo.output import _write_trc_outputs

    plain = pd.DataFrame({"Date": ["2026-01-01"], "Reserves": [1.0]})
    outputs: dict = {}
    _write_trc_outputs(plain, tmp_path, outputs, suffix=".csv.gz", compression="gzip")
    assert outputs == {} and not list(tmp_path.iterdir())

    results, _ = _run(tmp_path / "run", p=1.0, routing=_routing())
    out = tmp_path / "out"
    out.mkdir()
    _write_trc_outputs(results, out, outputs, suffix=".csv.gz", compression="gzip")
    assert set(outputs) == {"tdcsim_trc_results", "tdcsim_trc_metadata"}
    with gzip.open(out / "tdcsim_trc_results.csv.gz", "rt") as handle:
        written = pd.read_csv(handle)
    assert list(written.columns)[:2] == ["Date", "TRC_FiscalFlow"]
    metadata = json.loads((out / "tdcsim_trc_metadata.json").read_text())
    assert metadata["perimeter"] == "aggregate_reserve_balances_at_federal_reserve_banks"
    assert "deprecated compatibility alias" in metadata["legacy_reserves_alias"]


# --- CBO runner: opt-in TRC output is additive and the bounded replay still verifies. ---


def test_cbo_run_with_trc_routing_keeps_legacy_outputs_byte_identical(tmp_path: Path, monkeypatch) -> None:
    import tdcsim_cbo.runner as runner_module
    from tdcsim_cbo import CboScenarioSpec, run_cbo_scenario
    from tdcsim_cbo._json import sha256_file
    from tdcsim_cbo.verifier import verify_scenario_run
    from test_tdcsim_cbo_closeout_interface import _runner_baseline_and_scenarios

    monkeypatch.setattr(runner_module, "host_available_memory_bytes", lambda: 1 << 50)
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(json.dumps(_routing(on_rrp_opening=1_000_000.0)))

    legacy = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "legacy")
    routed = run_cbo_scenario(
        baseline,
        CboScenarioSpec.from_file(scenarios["noop"]),
        tmp_path / "routed",
        trc_routing_path=routing_path,
    )
    legacy_files = {path.name for path in (legacy.output_dir / "outputs").iterdir()}
    routed_files = {path.name for path in (routed.output_dir / "outputs").iterdir()}
    results_name = next(name for name in legacy_files if name.startswith("results_"))
    suffix = ".csv.gz" if results_name.endswith(".csv.gz") else ".csv"
    assert routed_files - legacy_files == {f"tdcsim_trc_results{suffix}", "tdcsim_trc_metadata.json"}
    # Two otherwise identical runs differ only in resource telemetry.
    telemetry = {"tdcsim_resource_samples.csv.gz", "tdcsim_resource_samples.csv", "summary.json"}
    for name in sorted(legacy_files - telemetry):
        assert sha256_file(routed.output_dir / "outputs" / name) == sha256_file(
            legacy.output_dir / "outputs" / name
        ), name
    legacy_summary = json.loads((legacy.output_dir / "outputs" / "summary.json").read_text())
    routed_summary = json.loads((routed.output_dir / "outputs" / "summary.json").read_text())
    legacy_summary.pop("peak_rss_bytes")
    routed_summary.pop("peak_rss_bytes")
    assert routed_summary == legacy_summary
    metadata = json.loads((routed.output_dir / "outputs" / "tdcsim_trc_metadata.json").read_text())
    assert metadata["routing_source"]["sha256"] == sha256_file(routing_path)
    assert verify_scenario_run(routed.output_dir)["status"] == "pass"


def test_committed_onrrp_available_policy_requires_zero_deposit_pass_through() -> None:
    # The ON RRP money-fund case is defined as pass-through 0 with every money-fund flow
    # settling through ON RRP; the policy must refuse a run that still sends 97 percent of
    # money-fund flows through deposits, so the companion baseline cannot silently differ.
    config = json.loads((Path(__file__).resolve().parents[1] / "trc_routing_onrrp_available_v1.json").read_text())

    policy = parse_trc_routing(config, mmf_deposit_pass_through=0.0)

    assert policy.routes_for("Private", PRIVATE_SUBBUCKET_MMF) == (("on_rrp_mmf", 1.0),)
    assert policy.reserve_share("Private", PRIVATE_SUBBUCKET_MMF) == 0.0
    assert policy.opening_balances["ONRRP_MMF"] > 0.0
    with pytest.raises(TrcRoutingError, match="must sum to one"):
        parse_trc_routing(config, mmf_deposit_pass_through=0.97)
