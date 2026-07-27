"""Scenario-construction invariants for the flooded (COVID-scale) stress family.

The flooded surface injects $3tn of stylized emergency spending in 2028 and raises the debt
target by the same amount, so the borrowing is explicit rather than an implicit plug. The 2029
and 2031 states are then exported *from the injection run*, which means their opening
portfolios already carry the injected debt.

A year-gated override (`injection_paths if year == 2028 else None`) handed those later years
the unmodified CBO debt target. The scenario then said: hold $3tn of injection-created debt,
and simultaneously hit a debt target that never saw the injection. The only way to satisfy it
was to stop borrowing and pay maturities out of the operating account, which drove the TGA to
-2,032bn for 365 consecutive periods.

Nothing in the suite could catch that: the flooded writer had no tests, and the defect lived
in which override dict a year received rather than in any function's own behaviour. These
tests pin the construction rules directly.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_ratewall_flooded_state_pairs.py"
FLOODED_YEARS = (2028, 2029, 2031)


def _load_module():
    src_dir = str(ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("write_ratewall_flooded_state_pairs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def flooded():
    return _load_module()


def test_injection_overrides_carry_debt_deficit_and_fed_paths(flooded, tmp_path: Path) -> None:
    """The injection must move debt, deficit, and Fed holdings together.

    Raising the deficit without raising the debt target would make the extra spending an
    implicit plug, which the input's own claim boundary forbids.
    """

    paths = {}
    for key in ("primary_deficit", "debt_target", "fed_holdings"):
        path = tmp_path / f"{key}.csv"
        path.write_text("period_end,value\n2028-01-01,1.0\n", encoding="utf-8")
        paths[key] = path

    overrides = flooded._injection_overrides(paths)
    for key in ("primary_deficit", "debt_target", "fed_holdings"):
        assert key in overrides, f"injection overrides must set {key}"
        assert overrides[key]["mode"] == "absolute_path_file"
        assert len(overrides[key]["file"]["sha256"]) == 64


def test_injection_paths_are_not_gated_to_a_single_year(flooded) -> None:
    """Every flooded year inherits the injected portfolio, so every year needs its debt path.

    This is the exact defect: `injection_paths=injection_paths if year == 2028 else None`
    gave 2029 and 2031 an opening portfolio holding injected debt while targeting a debt
    stock that never saw the injection.
    """

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "injection_paths":
            continue
        # A bare name or None is fine; a year-conditional expression is the bug's shape.
        if isinstance(node.value, ast.IfExp):
            rendered = ast.unparse(node.value)
            pytest.fail(
                "injection_paths is passed conditionally on the year, which desynchronises "
                f"the injected portfolio from the injected debt target: {rendered}"
            )


def test_negative_issuance_action_differs_by_year(flooded) -> None:
    """2028 issues into the injection; later years unwind it.

    A negative required issuance is a real error while the injection is being funded, but is
    expected once the injected debt starts maturing. Collapsing these would either mask a
    genuine failure in 2028 or fail 2029/2031 spuriously.
    """

    bills_heavy = flooded._bills_heavy_issuance_mix()
    assert bills_heavy["negative_issuance_action"] == "error"

    unwind = flooded._default_issuance_mix(negative_issuance_action="retire_shortest_public_marketable")
    assert unwind["negative_issuance_action"] == "retire_shortest_public_marketable"
    assert bills_heavy["fixed_remainder_shares"]["bills"] > unwind["fixed_remainder_shares"]["bills"], (
        "the injection year should be bills-heavier than the unwind years"
    )


def test_injection_total_is_stated_once(flooded) -> None:
    """The $3tn size is a scenario constant, not a literal repeated across transforms."""

    assert flooded.INJECTION_TOTAL_BIL == 3000.0
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("3000.0") == 1, "injection size must be defined once and referenced"
