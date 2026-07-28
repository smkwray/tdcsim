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


def test_injection_files_are_trimmed_to_the_consuming_window(flooded, tmp_path: Path) -> None:
    """A later flooded state models fewer periods than the injection file spans.

    The injection paths are built once from the 2028 state and run 2028-01-01 to 2036-09-30.
    The compiler requires an `absolute_path_file` replacement to match its baseline coverage
    exactly, so handing the untrimmed file to the 2029 state raised

        CompilerError: absolute_path_file replacement row count must match baseline
        coverage: 3195 != 2829

    Trimming is safe because the injected debt levels are cumulative by date: dropping rows
    before the opening state removes periods that state does not model, without altering the
    value carried on any row it keeps.
    """

    import pandas as pd

    source = tmp_path / "fiscal_injection_debt_target_path.csv"
    dates = pd.date_range("2028-01-01", "2030-12-31", freq="D")
    pd.DataFrame(
        {"period_end": dates.strftime("%Y-%m-%d"), "cbo_federal_debt_held_public_target_bil": 1.0}
    ).to_csv(source, index=False)

    untrimmed_dir = tmp_path / "untrimmed"
    untrimmed_dir.mkdir()
    untrimmed = flooded._copy_injection_files({"debt_target": source}, untrimmed_dir)
    assert len(pd.read_csv(untrimmed["debt_target"])) == len(dates)

    window = [d.strftime("%Y-%m-%d") for d in dates if d >= pd.Timestamp("2029-01-01")]
    trimmed_dir = tmp_path / "trimmed"
    trimmed_dir.mkdir()
    trimmed = flooded._copy_injection_files(
        {"debt_target": source}, trimmed_dir, baseline_windows={"debt_target": window}
    )
    frame = pd.read_csv(trimmed["debt_target"])
    assert frame["period_end"].iloc[0] == "2029-01-01"
    assert len(frame) == len(window)
    # The retained values must be untouched -- trimming drops rows, never rescales them.
    assert set(frame["cbo_federal_debt_held_public_target_bil"]) == {1.0}


def test_each_injection_input_trims_to_its_own_window(flooded, tmp_path: Path) -> None:
    """The three overridden inputs do not share a period window.

    At the 2029 state the debt-stock and Fed-holdings paths open on 2029-01-01 with 2,830
    rows, while the primary-deficit path opens on 2029-01-02 with 2,829: a deficit is a flow
    accrued over a period, so it has no row for the opening instant the stock paths carry.
    Trimming all three to one shared date produced 2,830 deficit rows against a 2,829-row
    baseline and failed the compiler's coverage check by exactly one.
    """

    import pandas as pd

    dates = pd.date_range("2028-01-01", "2030-12-31", freq="D")
    sources = {}
    for key in ("primary_deficit", "debt_target"):
        path = tmp_path / f"{key}.csv"
        pd.DataFrame({"period_end": dates.strftime("%Y-%m-%d"), "v": 1.0}).to_csv(path, index=False)
        sources[key] = path

    stock_window = [d.strftime("%Y-%m-%d") for d in dates if d >= pd.Timestamp("2029-01-01")]
    flow_window = [d.strftime("%Y-%m-%d") for d in dates if d >= pd.Timestamp("2029-01-02")]
    assert len(flow_window) == len(stock_window) - 1

    out = tmp_path / "out"
    out.mkdir()
    copied = flooded._copy_injection_files(
        sources, out, baseline_windows={"debt_target": stock_window, "primary_deficit": flow_window}
    )

    assert len(pd.read_csv(copied["debt_target"])) == len(stock_window)
    assert len(pd.read_csv(copied["primary_deficit"])) == len(flow_window)
    assert pd.read_csv(copied["primary_deficit"])["period_end"].iloc[0] == "2029-01-02"


def test_injection_trim_fails_closed_when_a_baseline_period_is_absent(flooded, tmp_path: Path) -> None:
    """A window the injection file cannot cover must raise, not emit a short override."""

    import pandas as pd

    source = tmp_path / "fiscal_injection_debt_target_path.csv"
    pd.DataFrame({"period_end": ["2028-01-01"], "v": [1.0]}).to_csv(source, index=False)
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(SystemExit, match="missing 1 baseline period"):
        flooded._copy_injection_files(
            {"debt_target": source}, out, baseline_windows={"debt_target": ["2035-01-01"]}
        )


def test_injection_total_is_stated_once(flooded) -> None:
    """The $3tn size is a scenario constant, not a literal repeated across transforms."""

    assert flooded.INJECTION_TOTAL_BIL == 3000.0
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("3000.0") == 1, "injection size must be defined once and referenced"
