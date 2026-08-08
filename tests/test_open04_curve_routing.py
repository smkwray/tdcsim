from __future__ import annotations

import ast
from collections import Counter
import math
from pathlib import Path
import struct

import pandas as pd
import pytest

from evaluated_nominal_curve import CurveContractError, EvaluatedNominalShock
import sim_engine
from sim_pricing import evaluate_nominal_yield, get_yield_for_maturity


CURVE_YEARS = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
CURVE_RATES = [0.032, 0.033, 0.034, 0.035, 0.038, 0.041, 0.043, 0.044]


def _binary64(value: float) -> bytes:
    return struct.pack(">d", float(value))


_DIRECT_BASELINE_YIELD_CALLS = {
    (
        "src/csv_gen.py",
        "_get_config_derived_issue_terms",
        ("maturity_years", "yield_curve_years", "yield_curve_rates"),
        (),
    ): "opening_state_baseline",
    (
        "src/sim_engine.py",
        "_fed_secondary_dirty_value",
        ("time_to_maturity", "tips_real_curve_years", "tips_real_curve_rates"),
        (("method", "interpolation_method"), ("floor_zero", "False")),
    ): "tips_real",
    (
        "src/sim_engine.py",
        "_get_category_yield",
        ("maturity", "yield_curve_years", "yield_curve_rates"),
        (("method", "interpolation_method"), ("floor_zero", "floor_zero")),
    ): "disabled_rate_sensitive_demand",
    (
        "src/sim_engine.py",
        "_frn_reference_rate_for_date",
        (
            "benchmark_maturity_years",
            "yield_curve_years",
            "yield_curve_rates",
        ),
        (("method", "interpolation_method"), ("floor_zero", "floor_zero")),
    ): "frn_fail_closed_fallback",
    (
        "src/sim_engine.py",
        "run_simulation",
        (
            "frn_benchmark_mat",
            "current_yield_curve_years",
            "current_yield_curve_rates",
        ),
        (
            ("method", "yield_interpolation_method"),
            ("floor_zero", "yield_floor_zero"),
        ),
    ): "frn_fail_closed_fallback",
    (
        "src/sim_engine.py",
        "_build_issuance_supply_schedule",
        ("maturity_years", "tips_real_curve_years", "tips_real_curve_rates"),
        (("method", "yield_interpolation_method"), ("floor_zero", "False")),
    ): "tips_real",
    (
        "src/sim_engine.py",
        "run_simulation",
        (
            "item_maturity",
            "current_yield_curve_years",
            "current_yield_curve_rates",
        ),
        (
            ("method", "yield_interpolation_method"),
            ("floor_zero", "yield_floor_zero"),
        ),
    ): "disabled_rate_sensitive_demand",
    (
        "src/sim_pricing.py",
        "infer_issue_data_for_loaded_bill",
        (
            "original_maturity_years",
            "yield_curve_years or []",
            "yield_curve_rates or []",
        ),
        (),
    ): "opening_state_baseline",
    (
        "src/sim_pricing.py",
        "evaluate_nominal_yield",
        ("maturity_years", "curve_years", "curve_rates"),
        (("method", "method"), ("floor_zero", "floor_zero")),
    ): "nominal_wrapper_baseline_evaluation",
    (
        "src/sim_trading.py",
        "calculate_portfolio_value_and_composition.<lambda>",
        ("ttm", "yield_curve_years", "yield_curve_rates"),
        (("method", "'pchip'"),),
    ): "disabled_trading",
    (
        "src/sim_trading.py",
        "calculate_portfolio_value_and_composition.calculate_row_prices",
        (
            "row['TimeToMaturity']",
            "tips_real_curve_years",
            "tips_real_curve_rates",
        ),
        (("method", "'pchip'"), ("floor_zero", "False")),
    ): "tips_real_disabled_trading",
}

_DIRECT_PCHIP_CONSUMERS = Counter(
    {
        ("src/sim_pricing.py", "get_yield_for_maturity"): 1,
        ("src/tdcsim_cbo/curve_runtime.py", "_baseline_pchip_vector"): 1,
        ("src/tdcsim_cbo/transforms/rates.py", "_interpolated_bp"): 1,
    }
)


def _direct_baseline_yield_calls(project_root: Path) -> tuple[Counter, dict]:
    found: Counter = Counter()
    locations: dict[tuple, list[int]] = {}
    for source_path in sorted((project_root / "src").rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        direct_names = {"get_yield_for_maturity"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*" and node.module == "sim_pricing":
                        raise AssertionError(
                            f"{source_path}: wildcard sim_pricing import is not auditable"
                        )
                    if alias.name == "get_yield_for_maturity":
                        direct_names.add(alias.asname or alias.name)

        relative_path = source_path.relative_to(project_root).as_posix()
        scope: list[str] = []

        class CallVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                scope.append(node.name)
                self.generic_visit(node)
                scope.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Lambda(self, node: ast.Lambda) -> None:
                scope.append("<lambda>")
                self.generic_visit(node)
                scope.pop()

            def visit_Call(self, node: ast.Call) -> None:
                direct_call = (
                    isinstance(node.func, ast.Name)
                    and node.func.id in direct_names
                )
                qualified_call = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get_yield_for_maturity"
                )
                if direct_call or qualified_call:
                    key = (
                        relative_path,
                        ".".join(scope) or "<module>",
                        tuple(ast.unparse(arg) for arg in node.args),
                        tuple(
                            (
                                keyword.arg or "**",
                                ast.unparse(keyword.value),
                            )
                            for keyword in node.keywords
                        ),
                    )
                    found[key] += 1
                    locations.setdefault(key, []).append(node.lineno)
                self.generic_visit(node)

        CallVisitor().visit(tree)
    return found, locations


def _generic_evaluator_indirections_and_pchip_consumers(
    project_root: Path,
) -> tuple[list[tuple[str, int, str]], Counter]:
    indirections: list[tuple[str, int, str]] = []
    pchip_consumers: Counter = Counter()
    for source_path in sorted((project_root / "src").rglob("*.py")):
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        relative_path = source_path.relative_to(project_root).as_posix()
        generic_names = {"get_yield_for_maturity"}
        pchip_names = {"PchipInterpolator"}
        partial_names = {"partial"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "get_yield_for_maturity":
                        generic_names.add(alias.asname or alias.name)
                    if alias.name == "PchipInterpolator":
                        pchip_names.add(alias.asname or alias.name)
                    if node.module == "functools" and alias.name == "partial":
                        partial_names.add(alias.asname or alias.name)

        def is_generic_reference(node: ast.AST) -> bool:
            return (
                isinstance(node, ast.Name) and node.id in generic_names
            ) or (
                isinstance(node, ast.Attribute)
                and node.attr == "get_yield_for_maturity"
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                value = node.value
                if is_generic_reference(value):
                    indirections.append(
                        (
                            relative_path,
                            node.lineno,
                            "generic evaluator rebinding",
                        )
                    )
            if isinstance(node, ast.Call):
                is_partial = (
                    isinstance(node.func, ast.Name)
                    and node.func.id in partial_names
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "partial"
                )
                if (
                    is_partial
                    and node.args
                    and is_generic_reference(node.args[0])
                ):
                    indirections.append(
                        (
                            relative_path,
                            node.lineno,
                            "generic evaluator partial",
                        )
                    )

        scope: list[str] = []

        class PchipVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                scope.append(node.name)
                self.generic_visit(node)
                scope.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                direct = (
                    isinstance(node.func, ast.Name)
                    and node.func.id in pchip_names
                )
                qualified = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "PchipInterpolator"
                )
                if direct or qualified:
                    pchip_consumers[
                        (relative_path, ".".join(scope) or "<module>")
                    ] += 1
                self.generic_visit(node)

        PchipVisitor().visit(tree)
    return indirections, pchip_consumers


def test_direct_baseline_yield_callsites_are_explicitly_classified() -> None:
    project_root = Path(__file__).resolve().parents[1]
    actual, locations = _direct_baseline_yield_calls(project_root)
    expected = Counter(_DIRECT_BASELINE_YIELD_CALLS.keys())
    unexpected = actual - expected
    missing = expected - actual
    assert not unexpected and not missing, (
        "direct get_yield_for_maturity call-site inventory changed; route every "
        "new nominal use through evaluate_nominal_yield or explicitly classify "
        "a permitted real/fail-closed/inactive/baseline seam. "
        f"unexpected={[(key, locations.get(key, [])) for key in unexpected.elements()]}; "
        f"missing={list(missing.elements())}"
    )


def test_generic_evaluator_cannot_hide_behind_indirection_or_custom_pchip() -> None:
    project_root = Path(__file__).resolve().parents[1]
    indirections, pchip_consumers = (
        _generic_evaluator_indirections_and_pchip_consumers(project_root)
    )

    assert not indirections, (
        "get_yield_for_maturity must remain directly auditable; "
        f"found={indirections}"
    )
    assert pchip_consumers == _DIRECT_PCHIP_CONSUMERS, (
        "direct PCHIP consumer inventory changed; classify the baseline "
        f"or stored-knot-only seam explicitly: {pchip_consumers}"
    )


@pytest.mark.parametrize(
    "maturity",
    [
        0.0,
        0.25,
        1.75,
        math.nextafter(2.0, -math.inf),
        2.0,
    ],
)
def test_evaluated_nominal_wrapper_is_bitwise_identical_through_two_years(
    maturity: float,
) -> None:
    baseline = get_yield_for_maturity(
        maturity,
        CURVE_YEARS,
        CURVE_RATES,
        method="pchip",
        floor_zero=False,
    )
    candidate = evaluate_nominal_yield(
        maturity,
        CURVE_YEARS,
        CURVE_RATES,
        method="pchip",
        floor_zero=False,
        shock=EvaluatedNominalShock(25.0),
    )
    assert _binary64(candidate) == _binary64(baseline)


@pytest.mark.parametrize(
    ("maturity", "expected_bp"),
    [
        (5.0, 25.0 * math.log(2.5) / math.log(5.0)),
        (10.0, 25.0),
        (30.0, 25.0),
    ],
)
def test_evaluated_nominal_wrapper_applies_exact_analytic_long_end(
    maturity: float,
    expected_bp: float,
) -> None:
    baseline = get_yield_for_maturity(
        maturity,
        CURVE_YEARS,
        CURVE_RATES,
        method="pchip",
        floor_zero=False,
    )
    candidate = evaluate_nominal_yield(
        maturity,
        CURVE_YEARS,
        CURVE_RATES,
        method="pchip",
        floor_zero=False,
        shock=EvaluatedNominalShock(25.0),
    )
    assert candidate - baseline == pytest.approx(expected_bp / 10_000.0, abs=1e-15)


def _guard_params(
    *,
    rate_sensitive_demand: dict | None,
    simulation_period: dict | None,
) -> dict:
    return {
        "funding_rule": {"mode": sim_engine.CBO_FUNDING_MODE},
        "yield_curve_surface": {
            "interpolation_method": "pchip",
            "floor_zero": False,
            "evaluated_nominal_shock": EvaluatedNominalShock(25.0),
        },
        "rate_sensitive_demand": (
            {} if rate_sensitive_demand is None else rate_sensitive_demand
        ),
        "simulation_period": {} if simulation_period is None else simulation_period,
    }


@pytest.mark.parametrize(
    ("rate_sensitive_demand", "simulation_period", "message"),
    [
        (None, {"enable_preference_trading": False}, "rate_sensitive_demand.enabled=false"),
        ({"enabled": True}, {"enable_preference_trading": False}, "rate_sensitive_demand.enabled=false"),
        ({"enabled": False}, None, "enable_preference_trading=false"),
        ({"enabled": False}, {"enable_preference_trading": True}, "enable_preference_trading=false"),
    ],
)
def test_active_sidecar_requires_explicitly_disabled_behavioral_controls(
    monkeypatch: pytest.MonkeyPatch,
    rate_sensitive_demand: dict | None,
    simulation_period: dict | None,
    message: str,
) -> None:
    monkeypatch.setattr(sim_engine, "validate_run_params", lambda *args, **kwargs: None)
    with pytest.raises(CurveContractError, match=message):
        sim_engine.run_simulation(
            _guard_params(
                rate_sensitive_demand=rate_sensitive_demand,
                simulation_period=simulation_period,
            ),
            "2026-01-01",
            "2026-01-02",
            freq="D",
        )


def test_active_sidecar_accepts_exact_false_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReachedCboLoader(RuntimeError):
        pass

    monkeypatch.setattr(sim_engine, "validate_run_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sim_engine,
        "_load_cbo_engine_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(ReachedCboLoader()),
    )
    with pytest.raises(ReachedCboLoader):
        sim_engine.run_simulation(
            _guard_params(
                rate_sensitive_demand={"enabled": False},
                simulation_period={"enable_preference_trading": False},
            ),
            "2026-01-01",
            "2026-01-02",
            freq="D",
        )


@pytest.mark.parametrize(
    "event",
    [
        {
            "date": "2026-01-01",
            "actions": [
                {
                    "parameter_path": "rate_sensitive_demand",
                    "new_value": {"enabled": True},
                }
            ],
        },
        {
            "date": "2026-01-01",
            "actions": [
                {
                    "parameter_path": (
                        "simulation_period.enable_preference_trading"
                    ),
                    "new_value": True,
                }
            ],
        },
    ],
)
def test_active_sidecar_rejects_runtime_events_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    event: dict,
) -> None:
    monkeypatch.setattr(sim_engine, "validate_run_params", lambda *args, **kwargs: None)
    params = _guard_params(
        rate_sensitive_demand={"enabled": False},
        simulation_period={"enable_preference_trading": False},
    )
    params["events"] = [event]

    with pytest.raises(CurveContractError, match="empty events list"):
        sim_engine.run_simulation(
            params,
            "2026-01-01",
            "2026-01-02",
            freq="D",
        )


def test_active_sidecar_rejects_frn_nominal_curve_fallback() -> None:
    with pytest.raises(CurveContractError, match="explicit FRN benchmark coverage"):
        sim_engine._frn_reference_rate_for_date(
            {"frn_rate_path": pd.DataFrame()},
            "candidate_a",
            "2027-01-01",
            benchmark_maturity_years=0.25,
            yield_curve_years=CURVE_YEARS,
            yield_curve_rates=CURVE_RATES,
            interpolation_method="pchip",
            floor_zero=False,
            nominal_shock=EvaluatedNominalShock(-25.0),
        )


def test_frn_runtime_selects_one_scenario_before_exact_date_lookup() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario_id": "baseline",
                "period_start": "2026-01-01",
                "period_end": "2026-12-31",
                "benchmark_rate_decimal": 0.04,
            },
            {
                "scenario_id": "default",
                "period_start": "2026-06-01",
                "period_end": "2026-06-30",
                "benchmark_rate_decimal": 0.123,
            },
        ]
    )
    target = pd.Timestamp("2026-06-30")
    lookup = {("default", target): frame.iloc[1]}

    row = sim_engine._frn_rate_path_row_for_date(
        frame,
        "baseline",
        target,
        lookup=lookup,
    )

    assert row is not None
    assert row["scenario_id"] == "baseline"
    assert float(row["benchmark_rate_decimal"]) == 0.04


def test_nonmarketable_nominal_basis_consumes_evaluated_shock() -> None:
    portfolio = pd.DataFrame(
        [
            {
                "Status": "Active",
                "SecurityType": "NonMarketable",
                "MaturityDate": pd.Timestamp("2040-12-31"),
                "FaceValue": 100.0,
            }
        ]
    )
    params = {
        "interest_crediting_frequency": "semi-annual",
        "rate_setting_method": "yield_curve_points",
        "interest_rate_basis_maturities": [5.0, 10.0],
    }
    baseline = sim_engine._capitalize_nonmarketable_interest(
        portfolio.copy(),
        pd.Timestamp("2026-06-29"),
        pd.Timestamp("2026-06-30"),
        params,
        CURVE_YEARS,
        CURVE_RATES,
        interpolation_method="pchip",
        floor_zero=False,
    )
    candidate = sim_engine._capitalize_nonmarketable_interest(
        portfolio.copy(),
        pd.Timestamp("2026-06-29"),
        pd.Timestamp("2026-06-30"),
        params,
        CURVE_YEARS,
        CURVE_RATES,
        interpolation_method="pchip",
        floor_zero=False,
        nominal_shock=EvaluatedNominalShock(25.0),
    )
    expected_increment = (
        25.0 * math.log(2.5) / math.log(5.0) + 25.0
    ) / 2.0 / 10_000.0 * 100.0 * 0.5
    assert candidate - baseline == pytest.approx(expected_increment, abs=1e-14)


def test_fed_tips_uses_real_base_and_shocked_nominal_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    def fake_value_treasury_security(**kwargs):
        captured.update(kwargs)
        return {"clean": 105.0, "accrued": 0.0, "dirty": 105.0}

    monkeypatch.setattr(
        sim_engine,
        "value_treasury_security",
        fake_value_treasury_security,
    )
    row = pd.Series(
        {
            "SecurityType": "TIPS",
            "FaceValue": 100.0,
            "OriginalPrincipal": 100.0,
            "AdjustedPrincipal": 105.0,
            "CouponRate": 0.01,
            "IssueDate": pd.Timestamp("2020-01-01"),
            "MaturityDate": pd.Timestamp("2035-01-01"),
            "FirstInterestPaymentDate": pd.Timestamp("2020-07-01"),
            "AccruedInterest_FRN": 0.0,
        }
    )
    result = sim_engine._fed_secondary_dirty_value(
        row,
        105.0,
        pd.Timestamp("2025-01-01"),
        CURVE_YEARS,
        CURVE_RATES,
        interpolation_method="pchip",
        floor_zero=False,
        nominal_shock=EvaluatedNominalShock(25.0),
        tips_real_curve_years=[2.0, 10.0, 30.0],
        tips_real_curve_rates=[0.01, 0.01, 0.01],
    )
    assert captured["discount_yield"] == pytest.approx(0.01)
    remaining_years = (
        pd.Timestamp("2035-01-01") - pd.Timestamp("2025-01-01")
    ).total_seconds() / (sim_engine.DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
    expected_nominal = evaluate_nominal_yield(
        remaining_years,
        CURVE_YEARS,
        CURVE_RATES,
        method="pchip",
        floor_zero=False,
        shock=EvaluatedNominalShock(25.0),
    )
    assert captured["nominal_discount_yield"] == pytest.approx(expected_nominal)
    assert result["discount_yield"] == pytest.approx(0.01)
