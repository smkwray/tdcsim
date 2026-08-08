from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pytest

from forecast_bundle_builders import build_fed_holdings_path_rows
from sim_engine import run_simulation
from simulation_calendar import build_simulation_calendar
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)
from tdcsim_cbo.bounded_output import (
    BoundedEvidenceError,
    BoundedResourceLimits,
    BoundedScenarioEvidenceSink,
    _stock_closure_rows,
)
from tdcsim_cbo.output import (
    _accounting_closure_handoff_tables,
    _route_stock_closure_handoff_tables,
    _tdc_handoff_tables,
)
from test_cbo_engine_integration import (
    _build_temp_forecast_inputs,
    _minimal_cash_mode_params,
    _minimal_engine_params,
    _opening_controlled_portfolio,
    _opening_frn_portfolio,
    _opening_tips_portfolio,
    _single_period,
    _write_csv,
    _write_frn_rate_path,
    _write_macro_path,
)


_STOCK_NUMERIC = (
    "debt_held_bil",
    "face_stock_bil",
    "adjusted_principal_stock_bil",
)
_ROUTE_STOCK_NUMERIC = (
    "route_debt_held_bil",
    "route_face_stock_bil",
    "route_adjusted_principal_stock_bil",
)


def _limits() -> BoundedResourceLimits:
    return BoundedResourceLimits(
        minimum_available_bytes=0,
        application_abort_rss_bytes=0,
        parent_graceful_stop_rss_bytes=0,
        parent_kill_rss_bytes=0,
        acceptance_peak_rss_bytes=0,
        portfolio_row_budget=10_000,
        key_cardinality_budget=8_192,
    )


def _normalized_key_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _aggregate_row_map(
    rows: list[Mapping[str, Any]],
    *,
    numeric_columns: tuple[str, ...],
) -> dict[tuple[tuple[str, str], ...], tuple[float, ...]]:
    columns = sorted(
        {
            str(column)
            for row in rows
            for column in row
            if str(column) not in numeric_columns
        }
    )
    totals: dict[tuple[tuple[str, str], ...], list[float]] = {}
    for row in rows:
        key = tuple(
            (column, _normalized_key_value(row.get(column)))
            for column in columns
        )
        values = totals.setdefault(key, [0.0] * len(numeric_columns))
        for index, column in enumerate(numeric_columns):
            values[index] += float(row.get(column, 0.0) or 0.0)
    return {key: tuple(values) for key, values in totals.items()}


def _assert_aggregate_rows_close(
    actual: list[Mapping[str, Any]],
    expected: list[Mapping[str, Any]],
    *,
    numeric_columns: tuple[str, ...],
) -> None:
    actual_map = _aggregate_row_map(actual, numeric_columns=numeric_columns)
    expected_map = _aggregate_row_map(expected, numeric_columns=numeric_columns)
    assert actual_map.keys() == expected_map.keys()
    for key in expected_map:
        assert actual_map[key] == pytest.approx(
            expected_map[key],
            abs=1e-7,
            rel=1e-12,
        )


def _assert_evidence_table_close(
    path: Path,
    expected_rows: list[Mapping[str, Any]],
) -> pd.DataFrame:
    actual = pd.read_csv(path)
    expected = pd.DataFrame(expected_rows)
    assert list(actual.columns) == list(expected.columns)
    numeric = [column for column in actual.columns if column.endswith("_bil")]
    keys = [column for column in actual.columns if column not in numeric]
    for frame in (actual, expected):
        for column in keys:
            frame[column] = frame[column].fillna("").astype(str)
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    if keys:
        actual = actual.sort_values(keys, kind="mergesort").reset_index(drop=True)
        expected = expected.sort_values(keys, kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual[keys],
        expected[keys],
        check_dtype=False,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        actual[numeric],
        expected[numeric],
        check_dtype=False,
        check_exact=False,
        atol=1e-7,
        rtol=1e-12,
    )
    return actual


def _assert_optional_number(actual: Any, expected: Any) -> None:
    if pd.isna(expected):
        assert pd.isna(actual)
    else:
        assert float(actual) == pytest.approx(
            float(expected),
            abs=1e-7,
            rel=1e-12,
        )


def _assert_annual_matches_legacy(
    annual: pd.DataFrame,
    legacy_results: pd.DataFrame,
) -> None:
    assert len(annual) == 1
    row = annual.iloc[0]
    final = legacy_results.iloc[-1]
    tdc = _tdc_handoff_tables(legacy_results)["tdcsim_period_tdc_summary"]
    assert len(tdc) == len(legacy_results) - 1
    summary = {
        column: sum(float(item[column]) for item in tdc)
        for column in (
            "tdc_change_bil",
            "overlap_cashflow_bil",
            "tdc_change_ex_overlap_bil",
        )
    }
    period_results = legacy_results.iloc[1:]
    expected = {
        "tdc_change_bil": summary["tdc_change_bil"],
        "overlap_cashflow_bil": summary["overlap_cashflow_bil"],
        "tdc_change_ex_overlap_bil": summary["tdc_change_ex_overlap_bil"],
        "cumulative_tdc_change_bil": summary["tdc_change_bil"],
        "cumulative_overlap_cashflow_bil": summary["overlap_cashflow_bil"],
        "cumulative_tdc_change_ex_overlap_bil": summary[
            "tdc_change_ex_overlap_bil"
        ],
        "modeled_financing_cost_bil": period_results[
            "FinancingCost_Period"
        ].sum(),
        "cumulative_modeled_financing_cost_bil": period_results[
            "FinancingCost_Period"
        ].sum(),
        "outstanding_controlled_wam_years": final[
            "OutstandingControlledWAM"
        ],
        "outstanding_controlled_bill_share": final[
            "OutstandingControlledBillShare"
        ],
        "outstanding_controlled_short_maturity_share": final[
            "OutstandingControlledShortMaturityShare"
        ],
    }
    for column, expected_value in expected.items():
        _assert_optional_number(row[column], expected_value)
    issuance = legacy_results.attrs["handoff_tables"][
        "tdcsim_period_issuance_flows"
    ]
    issuance_face = sum(float(item["face_issued_bil"]) for item in issuance)
    if issuance_face > 1e-12:
        issuance_wam = sum(
            float(item["face_issued_bil"])
            * float(item["weighted_original_term_years"])
            for item in issuance
        ) / issuance_face
        bill_share = sum(
            float(item["face_issued_bil"])
            for item in issuance
            if item["instrument_type"] == "Fixed"
            and item["maturity_bucket"] == "bills"
        ) / issuance_face
        short_share = sum(
            float(item["face_issued_bil"])
            for item in issuance
            if float(item["weighted_original_term_years"]) <= 1.0 + 1e-9
        ) / issuance_face
        issuance_expected = {
            "new_issuance_wam_years": issuance_wam,
            "new_issuance_bill_share": bill_share,
            "new_issuance_short_maturity_share": short_share,
        }
        for column, expected_value in issuance_expected.items():
            _assert_optional_number(row[column], expected_value)
    else:
        assert pd.isna(row["new_issuance_wam_years"])
        assert pd.isna(row["new_issuance_bill_share"])
        assert pd.isna(row["new_issuance_short_maturity_share"])
    assert row["tdc_change_ex_overlap_bil"] == pytest.approx(
        row["tdc_change_bil"] - row["overlap_cashflow_bil"],
        abs=1e-7,
    )
    assert row["cumulative_tdc_change_ex_overlap_bil"] == pytest.approx(
        row["cumulative_tdc_change_bil"]
        - row["cumulative_overlap_cashflow_bil"],
        abs=1e-7,
    )


def _run_equivalence(
    tmp_path: Path,
    params: dict[str, Any],
    *,
    start: str = "2026-09-20",
    end: str = "2026-09-30",
    freq: str = "10D",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    BoundedScenarioEvidenceSink,
]:
    legacy_results, legacy_portfolio = run_simulation(
        copy.deepcopy(params),
        start,
        end,
        freq=freq,
        scenario_name="baseline",
    )
    sink = BoundedScenarioEvidenceSink(
        tmp_path / "bounded",
        limits=_limits(),
        rss_reader=lambda: 1,
        available_reader=lambda: 1,
        cpu_reader=lambda: 0.25,
    )
    try:
        bounded_results, bounded_portfolio = run_simulation(
            copy.deepcopy(params),
            start,
            end,
            freq=freq,
            scenario_name="baseline",
            handoff_sink=sink,
            require_bounded_handoff=True,
        )
    except BaseException as exc:
        sink.abort(exc)
        raise

    pd.testing.assert_frame_equal(
        bounded_results,
        legacy_results,
        check_exact=True,
        check_names=True,
    )
    pd.testing.assert_frame_equal(
        bounded_portfolio.reset_index(drop=True),
        legacy_portfolio.reset_index(drop=True),
        check_exact=True,
        check_names=True,
    )

    raw = legacy_results.attrs["handoff_tables"]
    final_date = str(pd.Timestamp(legacy_results.index[-1]).date())
    legacy_holder = [
        row
        for row in raw["tdcsim_holder_stocks"]
        if str(row.get("date")) == final_date
    ]
    bounded_holder = [
        row
        for row in sink._rows("tdcsim_holder_stocks")
        if str(row.get("date")) == final_date
    ]
    _assert_aggregate_rows_close(
        bounded_holder,
        legacy_holder,
        numeric_columns=_STOCK_NUMERIC,
    )
    legacy_route = [
        row
        for row in raw["tdcsim_tdc_principal_route_stocks"]
        if str(row.get("date")) == final_date
    ]
    bounded_route = [
        row
        for row in sink._rows("tdcsim_tdc_principal_route_stocks")
        if str(row.get("date")) == final_date
    ]
    _assert_aggregate_rows_close(
        bounded_route,
        legacy_route,
        numeric_columns=_ROUTE_STOCK_NUMERIC,
    )

    accounting = _assert_evidence_table_close(
        sink.output_dir / "tdcsim_period_accounting_closure.csv.gz",
        _accounting_closure_handoff_tables(legacy_results, raw)[
            "tdcsim_accounting_closure"
        ],
    )
    expected_stock: list[dict[str, Any]] = []
    period_dates = [
        str(pd.Timestamp(value).date()) for value in legacy_results.index
    ]
    for period_start, period_end in zip(period_dates, period_dates[1:]):
        period_raw = {
            "tdcsim_holder_stocks": [
                row
                for row in raw["tdcsim_holder_stocks"]
                if str(row.get("date")) in {period_start, period_end}
            ],
            "tdcsim_accounting_journal": [
                row
                for row in raw["tdcsim_accounting_journal"]
                if str(row.get("period_start")) == period_start
                and str(row.get("period_end")) == period_end
            ],
        }
        expected_stock.extend(
            _stock_closure_rows(period_raw, period_start, period_end)
        )
    stock = _assert_evidence_table_close(
        sink.output_dir / "tdcsim_period_stock_closure.csv.gz",
        expected_stock,
    )
    route = _assert_evidence_table_close(
        sink.output_dir / "tdcsim_period_route_stock_closure.csv.gz",
        _route_stock_closure_handoff_tables(raw)[
            "tdcsim_tdc_principal_route_stock_closure"
        ],
    )
    _assert_evidence_table_close(
        sink.output_dir / "tdcsim_period_tdc_summary.csv.gz",
        _tdc_handoff_tables(legacy_results)["tdcsim_period_tdc_summary"],
    )
    for column in (
        "face_stock_closure_error_bil",
        "adjusted_principal_closure_error_bil",
        "treasury_cash_closure_error_bil",
        "reserve_closure_error_bil",
        "deposit_closure_error_bil",
        "holder_total_error_bil",
        "instrument_total_error_bil",
        "unexplained_residual_bil",
    ):
        tolerance = 1e-6 if column == "treasury_cash_closure_error_bil" else 1e-7
        assert accounting[column].abs().max() <= tolerance
    for column in (
        "face_stock_closure_error_bil",
        "adjusted_principal_closure_error_bil",
        "debt_stock_closure_error_bil",
    ):
        assert stock[column].abs().max() <= 1e-7
    assert route["closure_identity_error_bil"].abs().max() <= 1e-7

    summary = bounded_results.attrs["bounded_handoff_summary"]
    assert summary["period_count"] == len(legacy_results) - 1
    assert summary["event_count"] > 0
    assert summary["max_portfolio_rows"] <= summary["portfolio_row_budget"]
    assert summary["max_key_cardinality"] <= summary["key_cardinality_budget"]
    annual = pd.read_csv(
        sink.output_dir / "tdcsim_annual_economic_summary.csv.gz"
    )
    _assert_annual_matches_legacy(annual, legacy_results)
    return bounded_results, bounded_portfolio, sink


def test_bounded_reference_financing_matches_legacy_and_retains_leg_lineage(
    tmp_path: Path,
) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs",
        cbo_public_debt_target_bil=1_145.0,
        pre_issuance_controlled_debt_bil=1_000.0,
        signed_primary_flow_bil=40.0,
    )
    params = _minimal_engine_params(paths)
    params["initial_values"]["tga"] = 10.0
    params["funding_rule"].update(
        {
            "mode": "cbo_debt_reference_plus_tga_floor_financing_v1",
            "cash_closure_target_bil": 0.0,
            "validation_floor_bil": -0.000001,
        }
    )

    results, _, sink = _run_equivalence(tmp_path, params)
    period = results.iloc[-1]
    issuance = pd.read_csv(
        tmp_path / "bounded" / "tdcsim_period_issuance_aggregates.csv.gz"
    )

    assert period["TGA"] == pytest.approx(0.0, abs=1e-9)
    assert set(issuance["issuance_leg"]) == {
        "cbo_reference_face_issuance",
        "tga_floor_cash_financing",
    }
    financing = issuance.loc[
        issuance["issuance_leg"].eq("tga_floor_cash_financing")
    ]
    assert financing["face_issued_bil"].sum() == pytest.approx(
        period["CashFinancingFaceIssued"]
    )
    assert financing["cash_proceeds_bil"].sum() == pytest.approx(
        period["CashFinancingProceeds"]
    )


class _AdversarialBoundedSink(BoundedScenarioEvidenceSink):
    """Test-only live sink mutation at the production append seam."""

    def __init__(self, *args: Any, mutation: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mutation = mutation
        self.mutated = False

    def _append_journal(
        self,
        row: Mapping[str, Any],
        *,
        event_seq: int | None = None,
    ) -> None:
        event = dict(row)
        if (
            not self.mutated
            and self.mutation in {"drop", "duplicate_fresh_sequence"}
            and str(event.get("event_type")) == "issuance"
        ):
            self.mutated = True
            if self.mutation == "drop":
                return
            super()._append_journal(event, event_seq=event_seq)
            super()._append_journal(event)
            return
        if (
            not self.mutated
            and self.mutation == "swap_holder_route"
            and str(event.get("holder_sector"))
            != str(event.get("route_holder_sector"))
            and abs(float(event.get("face_stock_change_bil", 0.0) or 0.0))
            > 1e-12
        ):
            self.mutated = True
            (
                event["holder_sector"],
                event["route_holder_sector"],
            ) = (
                event["route_holder_sector"],
                event["holder_sector"],
            )
            (
                event["holder_subsector"],
                event["route_holder_subsector"],
            ) = (
                event["route_holder_subsector"],
                event["holder_subsector"],
            )
        super()._append_journal(event, event_seq=event_seq)


def _assert_live_mutation_rejected(
    tmp_path: Path,
    params: dict[str, Any],
    *,
    mutation: str,
) -> None:
    sink = _AdversarialBoundedSink(
        tmp_path / mutation,
        mutation=mutation,
        limits=_limits(),
        rss_reader=lambda: 1,
        available_reader=lambda: 1,
        cpu_reader=lambda: 0.25,
    )
    try:
        with pytest.raises(BoundedEvidenceError, match="period .*closure failed"):
            run_simulation(
                copy.deepcopy(params),
                "2026-09-20",
                "2026-09-30",
                freq="10D",
                scenario_name="baseline",
                handoff_sink=sink,
                require_bounded_handoff=True,
            )
        assert sink.mutated
    finally:
        sink.abort(RuntimeError("test cleanup"))


@pytest.mark.parametrize("mutation", ["drop", "duplicate_fresh_sequence"])
def test_live_path_stock_event_mutation_fails_independent_closure(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _build_temp_forecast_inputs(tmp_path / "inputs")
    _assert_live_mutation_rejected(
        tmp_path,
        _minimal_engine_params(paths),
        mutation=mutation,
    )


def test_live_path_actual_holder_route_swap_fails_keyed_closure(
    tmp_path: Path,
) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs",
        cbo_public_debt_target_bil=975.0,
        pre_issuance_controlled_debt_bil=1_000.0,
        signed_primary_flow_bil=0.0,
    )
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _buyback_route_portfolio()
    params["funding_rule"][
        "negative_required_issuance_action"
    ] = "retire_shortest_public_marketable"
    _assert_live_mutation_rejected(
        tmp_path,
        params,
        mutation="swap_holder_route",
    )


def test_bounded_matches_legacy_thirty_period_production_cbo_fixture(
    tmp_path: Path,
) -> None:
    start = "2026-08-31"
    end = "2026-09-30"
    periods = build_simulation_calendar(start, end, "daily")
    assert len(periods) == 30
    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs",
        periods=periods,
        period_start=start,
        period_end=end,
    )
    results, portfolio, sink = _run_equivalence(
        tmp_path,
        _minimal_engine_params(paths),
        start=start,
        end=end,
        freq="D",
    )

    assert len(results) == 31
    assert not portfolio.empty
    summary = results.attrs["bounded_handoff_summary"]
    assert summary["period_count"] == 30
    assert summary["max_portfolio_rows"] <= summary["portfolio_row_budget"]
    commitments = pd.read_csv(
        sink.output_dir / "tdcsim_event_commitments.csv.gz"
    )
    accounting = pd.read_csv(
        sink.output_dir / "tdcsim_period_accounting_closure.csv.gz"
    )
    annual = pd.read_csv(
        sink.output_dir / "tdcsim_annual_economic_summary.csv.gz"
    )
    assert len(commitments) == 30
    assert len(accounting) == 30
    assert annual["coverage_days"].tolist() == [30]
    assert annual["snapshot_date"].tolist() == [end]


def test_bounded_matches_legacy_tips_indexation(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs",
        cbo_public_debt_target_bil=245.0,
        public_nonmarketable_bil=100.0,
        definition_residual_bil=25.0,
        pre_issuance_controlled_debt_bil=100.0,
        signed_primary_flow_bil=0.0,
    )
    paths["macro_forecast_path_file"] = _write_macro_path(
        tmp_path / "inputs",
        cpi_u_index=120.0,
    )
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["initial_bonds_df"] = _opening_tips_portfolio(
        face_value=100.0,
        reference_cpi=100.0,
    )
    params["tips_params"] = {
        "cpi_start_level": 100.0,
        "cpi_annual_inflation": 0.0,
        "ref_cpi_lag_months": 0,
        "default_real_coupon_rate": 0.01,
    }
    params["financing_cost_options"] = {
        "include_tips_inflation_accretion": True
    }

    results, portfolio, sink = _run_equivalence(tmp_path, params)

    assert results.iloc[-1]["TIPSInflationAccretion_Period"] == pytest.approx(
        20.0
    )
    assert portfolio.loc[
        portfolio["BondID"].eq(1), "AdjustedPrincipal"
    ].iloc[0] == pytest.approx(120.0)
    ledger = pd.read_csv(
        sink.output_dir / "tdcsim_period_ledger_totals.csv.gz"
    )
    indexation = ledger[ledger["event_type"].eq("indexation")]
    assert indexation["adjusted_principal_change_bil"].sum() == pytest.approx(
        20.0
    )


def test_bounded_matches_legacy_tips_deflation_floor_and_maturity(
    tmp_path: Path,
) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs",
        cbo_public_debt_target_bil=205.0,
        public_nonmarketable_bil=100.0,
        definition_residual_bil=25.0,
        pre_issuance_controlled_debt_bil=100.0,
        signed_primary_flow_bil=0.0,
    )
    paths["macro_forecast_path_file"] = _write_macro_path(
        tmp_path / "inputs",
        cpi_u_index=80.0,
    )
    opening = _opening_tips_portfolio(
        face_value=100.0,
        reference_cpi=100.0,
    )
    opening.loc[:, "MaturityDate"] = pd.Timestamp("2026-09-25")
    opening.loc[:, "AdjustedPrincipal"] = 80.0
    opening.loc[:, "IndexRatio"] = 0.8
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["initial_bonds_df"] = opening
    params["tips_params"] = {
        "cpi_start_level": 80.0,
        "cpi_annual_inflation": 0.0,
        "ref_cpi_lag_months": 0,
        "default_real_coupon_rate": 0.0,
    }

    _, _, sink = _run_equivalence(tmp_path, params)

    principal = pd.read_csv(
        sink.output_dir / "tdcsim_period_principal_aggregates.csv.gz"
    )
    maturity = principal[
        principal["redemption_type"].eq("scheduled_maturity")
        & principal["instrument_type"].eq("TIPS")
    ]
    assert maturity["adjusted_principal_stock_removed_bil"].sum() == pytest.approx(
        80.0
    )
    assert maturity["cash_paid_bil"].sum() == pytest.approx(100.0)
    payments = pd.read_csv(
        sink.output_dir / "tdcsim_period_payment_aggregates.csv.gz"
    )
    floor = payments[
        payments["payment_type"].eq("tips_deflation_floor_topup")
    ]
    assert floor["amount_bil"].sum() == pytest.approx(20.0)


def test_bounded_matches_legacy_scheduled_frn_maturity(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path / "inputs")
    paths["frn_rate_path_file"] = _write_frn_rate_path(
        tmp_path / "inputs",
        rate_decimal=0.05,
    )
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _opening_frn_portfolio(
        accrued_interest=0.25,
        first_interest_payment_date="2026-12-30",
        maturity_date="2026-09-30",
    )

    _, _, sink = _run_equivalence(tmp_path, params)

    principal = pd.read_csv(
        sink.output_dir / "tdcsim_period_principal_aggregates.csv.gz"
    )
    maturity = principal[
        principal["redemption_type"].eq("scheduled_maturity")
        & principal["instrument_type"].eq("FRN")
    ]
    assert maturity["face_redeemed_bil"].sum() == pytest.approx(1_000.0)
    assert maturity["cash_paid_bil"].sum() == pytest.approx(1_000.0)


def _buyback_route_portfolio() -> pd.DataFrame:
    base = _opening_controlled_portfolio(800.0).iloc[0].copy()
    rows = []
    for bond_id, face, maturity, holder, subbucket, route, route_subbucket in (
        (
            1,
            100.0,
            "2027-01-01",
            "CB",
            "",
            "Private",
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        ),
        (
            2,
            100.0,
            "2027-01-02",
            "Private",
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
            "CB",
            "",
        ),
        (
            3,
            800.0,
            "2027-09-30",
            "Private",
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
            "Private",
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        ),
    ):
        row = base.copy()
        row["BondID"] = bond_id
        row["FaceValue"] = face
        row["OriginalPrincipal"] = face
        row["AdjustedPrincipal"] = face
        row["IssueProceeds"] = face
        row["MaturityDate"] = pd.Timestamp(maturity)
        row["HolderType"] = holder
        row["HolderSubBucket"] = subbucket
        row["TDCPrincipalHolderType"] = route
        row["TDCPrincipalHolderSubBucket"] = route_subbucket
        row["MaturityCategory"] = "bills" if bond_id < 3 else "notes"
        row["OriginalMaturityYears"] = 0.25 if bond_id < 3 else 2.0
        row["CouponRate"] = 0.0
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=BOND_PORTFOLIO_COLS,
    ).reset_index(drop=True).astype(PORTFOLIO_DTYPES, errors="ignore")


def test_bounded_matches_legacy_explicit_buyback_and_route_override(
    tmp_path: Path,
) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs",
        cbo_public_debt_target_bil=975.0,
        pre_issuance_controlled_debt_bil=1_000.0,
        signed_primary_flow_bil=0.0,
    )
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _buyback_route_portfolio()
    params["funding_rule"][
        "negative_required_issuance_action"
    ] = "retire_shortest_public_marketable"

    results, _, sink = _run_equivalence(tmp_path, params)

    assert results.iloc[-1]["CBOBuybackFaceRetired"] == pytest.approx(150.0)
    principal = pd.read_csv(
        sink.output_dir / "tdcsim_period_principal_aggregates.csv.gz"
    )
    retirements = principal[
        principal["redemption_type"].eq("explicit_retirement_at_par")
    ]
    assert retirements["face_redeemed_bil"].sum() == pytest.approx(150.0)
    assert set(retirements["holder_sector"]) == {"CB", "Private"}
    assert set(retirements["tdc_principal_recipient_sector"]) == {"CB"}
    assert retirements["tdc_principal_cash_paid_to_du_bil"].sum() == pytest.approx(
        0.0
    )


def _fed_sale_params(tmp_path: Path) -> dict[str, Any]:
    paths = _build_temp_forecast_inputs(
        tmp_path,
        cbo_public_debt_target_bil=1_125.0,
        pre_issuance_controlled_debt_bil=1_000.0,
    )
    fed_rows = build_fed_holdings_path_rows(
        scenario_id="baseline",
        periods=_single_period(),
        opening_state_date="2026-09-20",
        opening_cb_holdings_bil=200.0,
        cbo_fy_end_fed_holdings_bil={2026: 150.0},
        observation_date="2026-09-20",
        available_date="2026-09-20",
    )
    paths["fed_holdings_path_file"] = _write_csv(
        tmp_path / "tdcsim_fed_holdings_path.csv",
        fed_rows,
    )
    rows = []
    for bond_id, holder, face in (
        (1, "CB", 200.0),
        (2, "Banks", 300.0),
        (3, "Private", 400.0),
        (4, "Foreign", 100.0),
    ):
        row = _opening_controlled_portfolio(face).iloc[0].copy()
        row["BondID"] = bond_id
        row["HolderType"] = holder
        row["HolderSubBucket"] = (
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
            if holder == "Private"
            else ""
        )
        row["TDCPrincipalHolderType"] = holder
        row["TDCPrincipalHolderSubBucket"] = row["HolderSubBucket"]
        rows.append(row)
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = pd.DataFrame(
        rows,
        columns=BOND_PORTFOLIO_COLS,
    ).reset_index(drop=True).astype(PORTFOLIO_DTYPES, errors="ignore")
    return params


def test_bounded_matches_legacy_fed_secondary_sale(tmp_path: Path) -> None:
    results, _, sink = _run_equivalence(
        tmp_path,
        _fed_sale_params(tmp_path / "inputs"),
    )

    final = results.iloc[-1]
    assert final["CBOFedSyntheticSecondarySales"] == pytest.approx(50.0)
    assert final["CBOFedSecondarySaleCash"] > 0.0
    ledger = pd.read_csv(
        sink.output_dir / "tdcsim_period_ledger_totals.csv.gz"
    )
    transfer = ledger[
        ledger["event_type"].eq("fed_secondary_transfer")
    ]
    assert not transfer.empty
    assert transfer["face_stock_change_bil"].abs().sum() > 0.0


def _route_divergence_portfolio() -> pd.DataFrame:
    rows = []
    for bond_id, face, holder, route, route_subbucket in (
        (
            1,
            400.0,
            "Banks",
            "Private",
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        ),
        (
            2,
            300.0,
            "Foreign",
            "Private",
            PRIVATE_SUBBUCKET_MMF,
        ),
        (3, 300.0, "Private", "Banks", ""),
    ):
        row = _opening_controlled_portfolio(face).iloc[0].copy()
        row["BondID"] = bond_id
        row["HolderType"] = holder
        row["HolderSubBucket"] = (
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
            if holder == "Private"
            else ""
        )
        row["TDCPrincipalHolderType"] = route
        row["TDCPrincipalHolderSubBucket"] = route_subbucket
        row["CouponRate"] = 0.0
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=BOND_PORTFOLIO_COLS,
    ).reset_index(drop=True).astype(PORTFOLIO_DTYPES, errors="ignore")


def test_bounded_matches_legacy_actual_holder_and_private_route_divergence(
    tmp_path: Path,
) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs",
        cbo_public_debt_target_bil=1_125.0,
        pre_issuance_controlled_debt_bil=1_000.0,
        signed_primary_flow_bil=1.0,
    )
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _route_divergence_portfolio()

    results, _, sink = _run_equivalence(tmp_path, params)

    date = str(pd.Timestamp(results.index[-1]).date())
    rows = [
        row
        for row in sink._rows("tdcsim_tdc_principal_route_stocks")
        if row["date"] == date
        and row["debt_scope"] == "controlled_public_marketable"
    ]
    route_totals: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (
            str(row["route_holder_sector"]),
            str(row["route_holder_subsector"]),
        )
        route_totals[key] = route_totals.get(key, 0.0) + float(
            row["route_debt_held_bil"]
        )
    assert route_totals == pytest.approx(
        {
            ("Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK): 400.0,
            ("Private", PRIVATE_SUBBUCKET_MMF): 300.0,
            ("Banks", ""): 300.0,
        }
    )


def _intragovernmental_payment_portfolio() -> pd.DataFrame:
    public = _opening_controlled_portfolio(1_000.0).iloc[0].copy()
    public["CouponRate"] = 0.0
    public["TDCPrincipalHolderType"] = "Private"
    public["TDCPrincipalHolderSubBucket"] = (
        PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
    )
    intragov = _opening_controlled_portfolio(50.0).iloc[0].copy()
    intragov["BondID"] = 2
    intragov["IssueDate"] = pd.Timestamp("2026-03-30")
    intragov["DatedDate"] = pd.Timestamp("2026-03-30")
    intragov["OriginalDatedDate"] = pd.Timestamp("2026-03-30")
    intragov["FirstInterestPaymentDate"] = pd.Timestamp("2026-09-30")
    intragov["InterestPaymentFrequency"] = 2.0
    intragov["MaturityDate"] = pd.Timestamp("2027-09-30")
    intragov["OriginalMaturityYears"] = 1.5
    intragov["CouponRate"] = 0.04
    intragov["HolderType"] = "FedInternal"
    intragov["HolderSubBucket"] = ""
    intragov["TDCPrincipalHolderType"] = "FedInternal"
    intragov["TDCPrincipalHolderSubBucket"] = ""
    return pd.DataFrame(
        [public, intragov],
        columns=BOND_PORTFOLIO_COLS,
    ).reset_index(drop=True).astype(PORTFOLIO_DTYPES, errors="ignore")


def test_bounded_matches_legacy_intragovernmental_payment(
    tmp_path: Path,
) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs",
        cbo_public_debt_target_bil=1_125.0,
        pre_issuance_controlled_debt_bil=1_000.0,
        signed_primary_flow_bil=1.0,
    )
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _intragovernmental_payment_portfolio()

    results, _, sink = _run_equivalence(tmp_path, params)

    payments = pd.read_csv(
        sink.output_dir / "tdcsim_period_payment_aggregates.csv.gz"
    )
    intragov = payments[payments["holder_sector"].eq("FedInternal")]
    assert not intragov.empty
    assert intragov["amount_bil"].sum() == pytest.approx(1.0)
    assert results.iloc[-1]["InterestPaid_Bonds"] == pytest.approx(1.0)
    assert results.iloc[-1]["TDC_InterestToDU"] == pytest.approx(0.0)
    assert results.iloc[-1]["TDC_DebtService"] == pytest.approx(0.0)
    assert intragov["accounting_basis"].eq("cash").all()
    assert intragov["is_additive_to_cash_total"].astype(
        str
    ).str.lower().eq(
        "true"
    ).all()


def test_bounded_matches_legacy_intragovernmental_noncash_issuance(
    tmp_path: Path,
) -> None:
    paths = _build_temp_forecast_inputs(tmp_path / "inputs")
    params = _minimal_cash_mode_params(paths)
    params["treasury_issuance_profile"]["NonMarketable"] = {
        "target_percentage": 1.0,
        "maturities": [30.0],
        "maturity_distribution": [1.0],
        "nominal_maturity_years": 30.0,
    }
    params["sector_preferences"]["TrustFunds"][
        "nonmarketable_pct"
    ] = 1.0
    params["nonmarketable_params"] = {"initial_holder": "TrustFunds"}
    opening = _opening_controlled_portfolio(1.0)
    opening["CouponRate"] = 0.0
    opening["FirstInterestPaymentDate"] = pd.NaT
    opening["InterestPaymentFrequency"] = float("nan")
    params["initial_bonds_df"] = opening

    results, portfolio, sink = _run_equivalence(tmp_path, params)

    issuance = pd.read_csv(
        sink.output_dir / "tdcsim_period_issuance_aggregates.csv.gz"
    )
    intragov = issuance[
        issuance["holder_sector"].eq("TrustFunds")
        & issuance["instrument_type"].eq("NonMarketable")
    ]
    ledger = pd.read_csv(
        sink.output_dir / "tdcsim_period_ledger_totals.csv.gz"
    )
    journal = ledger[
        ledger["event_type"].eq("intragovernmental_issuance")
    ]
    expected_face = float(results.iloc[-1]["NewDebtIssued"])
    assert expected_face > 0.0
    assert intragov["face_issued_bil"].sum() == pytest.approx(
        expected_face
    )
    assert intragov["cash_proceeds_bil"].sum() == pytest.approx(0.0)
    assert journal["face_stock_change_bil"].sum() == pytest.approx(
        expected_face
    )
    assert journal["treasury_cash_change_bil"].sum() == pytest.approx(0.0)
    assert journal["settlement_scope"].eq(
        "consolidated_noncash_intragovernmental"
    ).all()
    issued_portfolio = portfolio[
        portfolio["SecurityType"].eq("NonMarketable")
    ]
    assert len(issued_portfolio) == 1
    assert issued_portfolio["HolderType"].eq("TrustFunds").all()
    assert issued_portfolio["IssueProceeds"].sum() == pytest.approx(0.0)
    assert results.iloc[-1]["AuctionProceeds"] == pytest.approx(0.0)
    assert results.iloc[-1]["TGA"] == pytest.approx(
        results.iloc[0]["TGA"]
    )
