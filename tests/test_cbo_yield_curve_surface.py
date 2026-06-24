from __future__ import annotations

import pytest

from cbo_yield_curve_surface import CONSTRUCTION_METHOD, build_yield_curve_surface_rows
from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_yield_curve_surface
from yield_curve_path import curve_for_date, load_yield_curve_surface as load_runtime_yield_curve_surface


def _base_curve_rows(*, curve_date: str = "2026-02-10", available_date: str = "2026-02-10"):
    tenors_and_rates = [
        (0.25, 4.10),
        (1.0, 3.85),
        (5.0, 3.95),
        (10.0, 4.25),
        (30.0, 4.55),
    ]
    return [
        {
            "base_curve_date": curve_date,
            "available_date": available_date,
            "base_curve_source_key": "treasury_nominal_par_curve_2026_02_10",
            "base_curve_sha256": "abc123frozen",
            "tenor_years": tenor,
            "nominal_rate": rate,
        }
        for tenor, rate in tenors_and_rates
    ]


def _macro_rows():
    return [
        {
            "schema_version": "tdcsim_macro_forecast_path_v1",
            "scenario_id": "baseline",
            "period_start": "2026-04-01",
            "period_end": "2026-06-30",
            "cbo_3m_tbill_rate_pct": 3.70,
            "cbo_10y_treasury_rate_pct": 4.05,
            "cbo_cpi_u_index": 320.0,
            "cbo_cpi_u_inflation_pct": 0.0,
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "source_vintage": "cbo_2026_02_baseline",
            "forecast_publication_date": "2026-02-11",
            "source_table": "1. Quarterly",
            "source_row_selector": "3-Month Treasury bill; 10-Year Treasury note",
            "observation_date": "2026-02-11",
            "available_date": "2026-02-11",
            "source_status": "verified_cbo_quarterly_macro_workbook",
            "claim_boundary": "quarterly_cbo_macro_anchors_not_full_runtime_yield_curve",
        },
        {
            "schema_version": "tdcsim_macro_forecast_path_v1",
            "scenario_id": "baseline",
            "period_start": "2026-07-01",
            "period_end": "2026-09-30",
            "cbo_3m_tbill_rate_pct": 3.45,
            "cbo_10y_treasury_rate_pct": 3.95,
            "cbo_cpi_u_index": 322.0,
            "cbo_cpi_u_inflation_pct": 0.625,
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "source_vintage": "cbo_2026_02_baseline",
            "forecast_publication_date": "2026-02-11",
            "source_table": "1. Quarterly",
            "source_row_selector": "3-Month Treasury bill; 10-Year Treasury note",
            "observation_date": "2026-02-11",
            "available_date": "2026-02-11",
            "source_status": "verified_cbo_quarterly_macro_workbook",
            "claim_boundary": "quarterly_cbo_macro_anchors_not_full_runtime_yield_curve",
        },
    ]


def test_yield_surface_reproduces_cbo_anchors_and_loads_through_schema(tmp_path):
    rows = build_yield_curve_surface_rows(
        macro_forecast_rows=_macro_rows(),
        base_curve_rows=_base_curve_rows(),
        actuals_available_as_of="2026-02-20",
        output_tenors=(0.25, 1.0, 5.0, 10.0, 30.0),
    )
    csv_path = tmp_path / "tdcsim_yield_curve_surface.csv"
    write_forecast_rows_csv(csv_path, rows)

    loaded = load_yield_curve_surface(csv_path, actuals_available_as_of="2026-02-20")

    assert len(loaded) == 10
    assert set(loaded["construction_method"]) == {CONSTRUCTION_METHOD}
    assert set(loaded["base_curve_sha256"]) == {"abc123frozen"}
    for curve_date, expected_3m, expected_10y in [
        ("2026-04-01", 3.70, 4.05),
        ("2026-07-01", 3.45, 3.95),
    ]:
        quarter = loaded[loaded["curve_date"] == curve_date]
        rate_3m = quarter.loc[quarter["tenor_years"] == 0.25, "nominal_rate"].iloc[0]
        rate_10y = quarter.loc[quarter["tenor_years"] == 10.0, "nominal_rate"].iloc[0]
        rate_3m_decimal = quarter.loc[quarter["tenor_years"] == 0.25, "nominal_rate_decimal"].iloc[0]
        assert rate_3m == pytest.approx(expected_3m, abs=1e-14)
        assert rate_10y == pytest.approx(expected_10y, abs=1e-14)
        assert rate_3m_decimal == pytest.approx(expected_3m / 100.0, abs=1e-14)
    assert set(loaded["rate_unit"]) == {"percent_points"}
    assert set(loaded["runtime_rate_unit"]) == {"decimal"}


def test_runtime_loader_converts_generated_surface_to_decimal_rates(tmp_path):
    rows = build_yield_curve_surface_rows(
        macro_forecast_rows=_macro_rows()[:1],
        base_curve_rows=_base_curve_rows(),
        actuals_available_as_of="2026-02-20",
        output_tenors=(0.25, 1.0, 5.0, 10.0, 30.0),
    )
    csv_path = tmp_path / "tdcsim_yield_curve_surface.csv"
    write_forecast_rows_csv(csv_path, rows)

    surface = load_runtime_yield_curve_surface(csv_path)
    years, rates, status = curve_for_date(surface, "2026-04-15", scenario_id="baseline")

    assert status == "dynamic_curve_surface:2026-04-01"
    assert years[0] == pytest.approx(0.25)
    assert rates[0] == pytest.approx(0.037)
    assert rates[years.index(10.0)] == pytest.approx(0.0405)


def test_curve_for_date_rejects_missing_scenario(tmp_path):
    rows = build_yield_curve_surface_rows(
        macro_forecast_rows=_macro_rows()[:1],
        base_curve_rows=_base_curve_rows(),
        actuals_available_as_of="2026-02-20",
        output_tenors=(0.25, 10.0),
    )
    csv_path = tmp_path / "tdcsim_yield_curve_surface.csv"
    write_forecast_rows_csv(csv_path, rows)
    surface = load_runtime_yield_curve_surface(csv_path)

    with pytest.raises(ValueError, match="scenario 'other'"):
        curve_for_date(surface, "2026-04-15", scenario_id="other")


def test_runtime_loader_rejects_inconsistent_decimal_column(tmp_path):
    bad_path = tmp_path / "bad_surface.csv"
    bad_path.write_text(
        "\n".join(
            [
                "scenario_id,curve_date,tenor_years,nominal_rate,nominal_rate_decimal,rate_unit,runtime_rate_unit",
                "baseline,2026-04-01,0.25,3.70,0.370,percent_points,decimal",
                "baseline,2026-04-01,10.0,4.05,0.0405,percent_points,decimal",
            ]
        )
    )

    with pytest.raises(ValueError, match="nominal_rate_decimal"):
        load_runtime_yield_curve_surface(bad_path)


def test_runtime_loader_infers_legacy_percent_surface_and_rejects_mixed_units(tmp_path):
    legacy_path = tmp_path / "legacy_surface.csv"
    legacy_path.write_text(
        "\n".join(
            [
                "scenario_id,curve_date,tenor_years,nominal_rate",
                "baseline,2026-04-01,0.25,3.70",
                "baseline,2026-04-01,10.0,4.05",
            ]
        )
    )
    surface = load_runtime_yield_curve_surface(legacy_path)
    _, rates, _ = curve_for_date(surface, "2026-04-15", scenario_id="baseline")
    assert rates == pytest.approx([0.037, 0.0405])

    mixed_path = tmp_path / "mixed_surface.csv"
    mixed_path.write_text(
        "\n".join(
            [
                "scenario_id,curve_date,tenor_years,nominal_rate",
                "baseline,2026-04-01,0.25,0.037",
                "baseline,2026-04-01,10.0,4.05",
            ]
        )
    )
    with pytest.raises(ValueError, match="ambiguous mixed"):
        load_runtime_yield_curve_surface(mixed_path)


def test_yield_surface_rejects_base_curve_lookahead_unless_allowed():
    with pytest.raises(ValueError, match="base curve date"):
        build_yield_curve_surface_rows(
            macro_forecast_rows=_macro_rows(),
            base_curve_rows=_base_curve_rows(curve_date="2026-02-21", available_date="2026-02-21"),
            actuals_available_as_of="2026-02-20",
            output_tenors=(0.25, 10.0),
        )

    rows = build_yield_curve_surface_rows(
        macro_forecast_rows=_macro_rows()[:1],
        base_curve_rows=_base_curve_rows(curve_date="2026-02-21", available_date="2026-02-21"),
        actuals_available_as_of="2026-02-20",
        output_tenors=(0.25, 10.0),
        allow_lookahead=True,
    )

    assert rows[0]["base_curve_date"] == "2026-02-21"


def test_yield_surface_applies_no_implicit_zero_floor():
    base_curve = [
        {**row, "nominal_rate": -0.25}
        for row in _base_curve_rows()
    ]
    macro_row = {
        **_macro_rows()[0],
        "cbo_3m_tbill_rate_pct": -0.50,
        "cbo_10y_treasury_rate_pct": -0.10,
    }

    rows = build_yield_curve_surface_rows(
        macro_forecast_rows=[macro_row],
        base_curve_rows=base_curve,
        actuals_available_as_of="2026-02-20",
        output_tenors=(0.25, 1.0, 10.0),
    )

    assert min(row["nominal_rate"] for row in rows) < 0.0
    assert rows[0]["nominal_rate"] == pytest.approx(-0.50)
    assert rows[2]["nominal_rate"] == pytest.approx(-0.10)
