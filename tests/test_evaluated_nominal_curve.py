from __future__ import annotations

import copy
import json
import math
import struct
from pathlib import Path

import pytest

from evaluated_nominal_curve import (
    NOMINAL_EVALUATED_SHOCK_FILE,
    CurveContractError,
    EvaluatedNominalShock,
    build_evaluated_nominal_sidecar,
    evaluated_nominal_contract_metadata,
    load_evaluated_nominal_shock,
    normalize_open04_override,
)
from sim_pricing import evaluate_nominal_yield, get_yield_for_maturity


_CURVE_YEARS = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
_CURVE_RATES = [0.029, 0.031, 0.032, 0.034, 0.036, 0.041]


def _binary64(value: float) -> bytes:
    return struct.pack(">d", float(value))


def _override(sign: int = 1) -> dict:
    return {
        "mode": "evaluated_additive_key_rate_bp",
        "application": "post_baseline_evaluation",
        "interpolation": "log_tenor_linear",
        "lower_endpoint": "zero_at_or_below_first_key",
        "upper_endpoint": "flat_at_or_above_last_key",
        "time_profile": "constant_across_curve_dates",
        "compounding": "none",
        "shocks": [
            {"tenor_years": 10, "shock_bp": sign * 25},
            {"tenor_years": 2, "shock_bp": -0.0},
        ],
    }


def _surface(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "scenario_id,curve_date,tenor_years,nominal_rate_decimal",
                "baseline,2026-04-01,0.25,0.030",
                "baseline,2026-04-01,2,0.031",
                "baseline,2026-04-01,5,0.032",
                "baseline,2026-04-01,10,0.033",
                "baseline,2026-07-01,0.25,0.029",
                "baseline,2026-07-01,2,0.030",
                "baseline,2026-07-01,5,0.031",
                "baseline,2026-07-01,10,0.032",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_sidecar(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_shock_formula_has_exact_branches_and_log_tenor_interior() -> None:
    positive = EvaluatedNominalShock(25)
    negative = EvaluatedNominalShock(-25)

    for maturity in (0.0, 0.25, 1.999, 2.0):
        assert positive.shock_bp(maturity) == 0.0
        assert math.copysign(1.0, positive.shock_bp(maturity)) == 1.0
    assert positive.shock_bp(5.0) == pytest.approx(
        14.233086048165175,
        abs=1e-12,
        rel=0.0,
    )
    assert positive.shock_bp(10.0) == 25.0
    assert positive.shock_bp(30.0) == 25.0
    assert negative.shock_bp(5.0) == -positive.shock_bp(5.0)
    assert negative.shock_bp(10.0) == -25.0
    above_two = math.nextafter(2.0, math.inf)
    below_ten = math.nextafter(10.0, -math.inf)
    above_ten = math.nextafter(10.0, math.inf)
    assert positive.shock_bp(above_two) > 0.0
    assert negative.shock_bp(above_two) < 0.0
    assert 0.0 < positive.shock_bp(below_ten) <= 25.0
    assert positive.shock_bp(above_ten) == 25.0
    assert negative.shock_bp(above_ten) == -25.0
    assert 25.0 - positive.shock_bp(below_ten) <= 1e-12

    interior = [2.0 + index * 8.0 / 512.0 for index in range(1, 512)]
    values = [positive.shock_bp(maturity) for maturity in interior]
    assert all(0.0 < value < 25.0 for value in values)
    assert all(left < right for left, right in zip(values, values[1:]))


@pytest.mark.parametrize("value", [0, 24.999, -24.999, float("nan"), float("inf"), True])
def test_shock_rejects_unsupported_magnitude(value: object) -> None:
    with pytest.raises(CurveContractError):
        EvaluatedNominalShock(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("maturity", [-0.01, float("nan"), float("inf"), True])
def test_shock_rejects_unsupported_maturity(maturity: object) -> None:
    with pytest.raises(CurveContractError):
        EvaluatedNominalShock(25).shock_bp(maturity)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["linear", "pchip"])
@pytest.mark.parametrize("floor_zero", [False, True])
def test_nominal_wrapper_preserves_baseline_when_inactive(
    method: str,
    floor_zero: bool,
) -> None:
    for maturity in (0.0, 0.25, 1.375, 2.0, 5.0, 10.0, 45.0):
        baseline = get_yield_for_maturity(
            maturity,
            _CURVE_YEARS,
            _CURVE_RATES,
            method=method,
            floor_zero=floor_zero,
        )
        wrapped = evaluate_nominal_yield(
            maturity,
            _CURVE_YEARS,
            _CURVE_RATES,
            method=method,
            floor_zero=floor_zero,
            shock=None,
        )
        assert _binary64(wrapped) == _binary64(baseline)


def test_nominal_wrapper_is_bitwise_identical_on_dense_short_end() -> None:
    shocks = (EvaluatedNominalShock(-25.0), EvaluatedNominalShock(25.0))
    maturities = [
        *(2.0 * index / 1024.0 for index in range(1025)),
        math.nextafter(2.0, -math.inf),
    ]

    for maturity in maturities:
        baseline = get_yield_for_maturity(
            maturity,
            _CURVE_YEARS,
            _CURVE_RATES,
            method="pchip",
            floor_zero=False,
        )
        for shock in shocks:
            wrapped = evaluate_nominal_yield(
                maturity,
                _CURVE_YEARS,
                _CURVE_RATES,
                method="pchip",
                floor_zero=False,
                shock=shock,
            )
            assert _binary64(wrapped) == _binary64(baseline)


@pytest.mark.parametrize("signed_10y_bp", [-25.0, 25.0])
def test_nominal_wrapper_matches_analytic_delta_at_policy_boundaries(
    signed_10y_bp: float,
) -> None:
    shock = EvaluatedNominalShock(signed_10y_bp)
    maturities = (
        2.0,
        2.0 + 1e-8,
        5.0,
        math.nextafter(10.0, -math.inf),
        10.0,
        math.nextafter(10.0, math.inf),
        30.0,
        45.0,
    )

    for maturity in maturities:
        baseline = get_yield_for_maturity(
            maturity,
            _CURVE_YEARS,
            _CURVE_RATES,
            method="pchip",
            floor_zero=False,
        )
        wrapped = evaluate_nominal_yield(
            maturity,
            _CURVE_YEARS,
            _CURVE_RATES,
            method="pchip",
            floor_zero=False,
            shock=shock,
        )
        if maturity <= 2.0:
            assert _binary64(wrapped) == _binary64(baseline)
        else:
            assert wrapped - baseline == pytest.approx(
                shock.shock_bp(maturity) / 10_000.0,
                abs=1e-12,
                rel=0.0,
            )


@pytest.mark.parametrize(
    ("method", "floor_zero"),
    [
        ("linear", False),
        ("pchip", True),
    ],
)
def test_active_nominal_wrapper_rejects_nonbaseline_evaluator_contract(
    method: str,
    floor_zero: bool,
) -> None:
    with pytest.raises(
        CurveContractError,
        match="requires baseline pchip with floor_zero=false",
    ):
        evaluate_nominal_yield(
            5.0,
            _CURVE_YEARS,
            _CURVE_RATES,
            method=method,
            floor_zero=floor_zero,
            shock=EvaluatedNominalShock(25.0),
        )


def test_override_normalization_is_exact_and_canonical() -> None:
    normalized = normalize_open04_override(_override(-1))

    assert normalized["shocks"] == [
        {"tenor_years": 2.0, "shock_bp": 0.0},
        {"tenor_years": 10.0, "shock_bp": -25.0},
    ]
    assert math.copysign(1.0, normalized["shocks"][0]["shock_bp"]) == 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"extra": True}), "fields must be exact"),
        (lambda value: value.pop("compounding"), "fields must be exact"),
        (
            lambda value: value.update({"application": "stored_knot_transform"}),
            "application",
        ),
        (
            lambda value: value.update({"interpolation": "linear_tenor"}),
            "interpolation",
        ),
        (
            lambda value: value["shocks"][1].update({"shock_bp": 1}),
            "2-year shock",
        ),
        (
            lambda value: value["shocks"][0].update({"shock_bp": 40}),
            "10-year shock",
        ),
        (
            lambda value: value["shocks"].append(
                {"tenor_years": 30, "shock_bp": 25}
            ),
            "exactly two",
        ),
        (
            lambda value: value["shocks"][0].update({"tenor_years": 5}),
            "exactly 2.0 and 10.0",
        ),
    ],
)
def test_override_rejects_noncanonical_contract(mutation, message: str) -> None:
    value = _override()
    mutation(value)

    with pytest.raises(CurveContractError, match=message):
        normalize_open04_override(value)


def test_sidecar_binds_unchanged_surface_and_loads_signed_shock(tmp_path: Path) -> None:
    surface = _surface(tmp_path / "tdcsim_yield_curve_surface.csv")
    payload = build_evaluated_nominal_sidecar(_override(-1), surface)
    sidecar = _write_sidecar(tmp_path / NOMINAL_EVALUATED_SHOCK_FILE, payload)

    assert payload["baseline_surface"] == {
        "relative_path": "tdcsim_yield_curve_surface.csv",
        "sha256": payload["baseline_surface"]["sha256"],
        "row_count": 8,
        "curve_date_count": 2,
        "tenor_set": [0.25, 2.0, 5.0, 10.0],
    }
    assert payload["baseline_evaluator"] == {
        "interpolation_method": "pchip",
        "floor_zero": False,
        "endpoint_behavior": "clamp",
    }
    assert "curve_rows" not in payload
    assert load_evaluated_nominal_shock(sidecar, surface) == EvaluatedNominalShock(
        -25
    )


def test_metadata_has_bounded_hash_and_date_tenor_lineage(tmp_path: Path) -> None:
    surface = _surface(tmp_path / "tdcsim_yield_curve_surface.csv")
    payload = build_evaluated_nominal_sidecar(_override(), surface)
    sidecar = _write_sidecar(tmp_path / NOMINAL_EVALUATED_SHOCK_FILE, payload)

    first = evaluated_nominal_contract_metadata(sidecar, surface)
    second = evaluated_nominal_contract_metadata(sidecar, surface)

    assert first == second
    assert first["sidecar"]["relative_path"] == NOMINAL_EVALUATED_SHOCK_FILE
    assert first["sidecar"]["bytes"] == sidecar.stat().st_size
    assert len(first["sidecar"]["sha256"]) == 64
    assert len(first["sidecar"]["canonical_sha256"]) == 64
    assert first["baseline_surface"]["row_count"] == 8
    assert first["baseline_surface"]["curve_date_count"] == 2
    assert first["baseline_surface"]["tenor_count"] == 4
    assert len(first["baseline_surface"]["date_set_sha256"]) == 64
    assert len(first["baseline_surface"]["tenor_set_sha256"]) == 64
    assert len(first["canonical_curve_method_sha256"]) == 64
    assert set(first) == {
        "schema_version",
        "contract_id",
        "signed_10y_shock_bp",
        "sidecar",
        "baseline_surface",
        "baseline_evaluator",
        "canonical_curve_method_sha256",
    }


def test_sidecar_and_surface_mutations_fail_closed(tmp_path: Path) -> None:
    surface = _surface(tmp_path / "tdcsim_yield_curve_surface.csv")
    payload = build_evaluated_nominal_sidecar(_override(), surface)
    sidecar = _write_sidecar(tmp_path / NOMINAL_EVALUATED_SHOCK_FILE, payload)

    mutated = copy.deepcopy(payload)
    mutated["baseline_surface"]["sha256"] = "0" * 64
    _write_sidecar(sidecar, mutated)
    with pytest.raises(CurveContractError, match="baseline_surface binding"):
        load_evaluated_nominal_shock(sidecar, surface)

    _write_sidecar(sidecar, payload)
    surface.write_text(
        surface.read_text(encoding="utf-8").replace("0.032\n", "0.0321\n"),
        encoding="utf-8",
    )
    with pytest.raises(CurveContractError, match="baseline_surface binding"):
        load_evaluated_nominal_shock(sidecar, surface)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                ("2026-04-01", 0.25, 0.03),
                ("2026-04-01", 2.0, 0.03),
                ("2026-04-01", 2.0, 0.03),
                ("2026-04-01", 10.0, 0.03),
            ],
            "duplicate date/tenor",
        ),
        (
            [
                ("2026-04-01", 0.25, 0.03),
                ("2026-04-01", 10.0, 0.03),
            ],
            "2-year and 10-year",
        ),
        (
            [
                ("2026-04-01", 0.25, 0.03),
                ("2026-04-01", 2.0, 0.03),
                ("2026-04-01", 10.0, float("nan")),
            ],
            "finite number",
        ),
        (
            [
                ("2026-04-01", 0.25, 0.03),
                ("2026-04-01", 2.0, 0.03),
                ("2026-04-01", 10.0, 0.03),
                ("2026-07-01", 0.25, 0.03),
                ("2026-07-01", 2.0, 0.03),
                ("2026-07-01", 5.0, 0.03),
                ("2026-07-01", 10.0, 0.03),
            ],
            "identical tenor set",
        ),
    ],
)
def test_surface_validation_rejects_malformed_rows(
    tmp_path: Path,
    rows: list[tuple[str, float, float]],
    message: str,
) -> None:
    surface = tmp_path / "tdcsim_yield_curve_surface.csv"
    surface.write_text(
        "scenario_id,curve_date,tenor_years,nominal_rate_decimal\n"
        + "\n".join(
            f"baseline,{curve_date},{tenor},{rate}"
            for curve_date, tenor, rate in rows
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CurveContractError, match=message):
        build_evaluated_nominal_sidecar(_override(), surface)
