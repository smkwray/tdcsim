"""Static checks on the pair-writing scripts' spec builders.

These writers are only exercised by generating a real campaign, which takes hours, so a
plain NameError in one of them reached campaign generation undetected: the flooded writer's
`_pair_spec` read `shock_manifest["run_id"]` without ever loading it, while both sibling
writers loaded it correctly. The bug was introduced when pair specs moved from run
directories to run IDs and one of the three call sites was not converted.

A full functional test would need real run outputs. These checks are deliberately static and
cheap: they catch the divergence-between-siblings class of defect, which is what actually
bit, without pretending to prove the writers produce correct economics.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PAIR_WRITERS = (
    "write_ratewall_flooded_state_pairs.py",
    "write_ratewall_source_grade_marginal_pair.py",
    "write_ratewall_forecast_source_grade_marginal_pairs.py",
)

# Names these modules legitimately resolve from module scope or imports.
_MODULE_SCOPE_ALLOWED = {
    "pd",
    "Path",
    "Any",
    "read_json",
    "_read_json",
    "canonical_json_sha256",
    "sha256_file",
    "STATE_SET_ID",
    "SHOCK_PATH_ID",
    "OBJECT_ID",
    "DENOMINATOR_EQUIVALENCE_KEY",
    "NOMINAL_GDP_2026_BIL",
    "ForecastStateExport",
    "datetime",
    "timezone",
    "date",
}


def _pair_spec_functions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_pair_spec":
            found.append(node)
    return found


def _bound_names(func: ast.FunctionDef) -> set[str]:
    names = {arg.arg for arg in func.args.args} | {arg.arg for arg in func.args.kwonlyargs}
    if func.args.vararg:
        names.add(func.args.vararg.arg)
    if func.args.kwarg:
        names.add(func.args.kwarg.arg)
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            target = node.target
            names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            names |= {n.id for n in ast.walk(node.optional_vars) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


@pytest.mark.parametrize("script_name", PAIR_WRITERS)
def test_pair_spec_reads_no_unbound_name(script_name: str) -> None:
    """Every name a pair spec reads must be a parameter, a local, or module scope.

    This is the exact check that would have caught the flooded writer's missing
    `shock_manifest` load before it consumed hours of campaign compute.
    """

    path = ROOT / "scripts" / script_name
    functions = _pair_spec_functions(path)
    assert functions, f"{script_name} defines no _pair_spec to check"

    for func in functions:
        bound = _bound_names(func) | set(dir(builtins)) | _MODULE_SCOPE_ALLOWED
        # Names read but never bound anywhere in the function or its allowed scopes.
        read = {
            node.id
            for node in ast.walk(func)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        unbound = {name for name in read - bound if not name.startswith("_")}
        assert not unbound, f"{script_name}::_pair_spec reads unbound name(s): {sorted(unbound)}"


@pytest.mark.parametrize("script_name", PAIR_WRITERS)
def test_pair_spec_uses_every_run_dir_it_accepts(script_name: str) -> None:
    """A run-dir parameter that is accepted and never read is the shape of the real bug.

    The flooded writer took `shock_run_dir` and never touched it, so the shock run's identity
    silently came from nowhere. Accepting an input and ignoring it is the smell.
    """

    path = ROOT / "scripts" / script_name
    for func in _pair_spec_functions(path):
        params = {arg.arg for arg in func.args.args} | {arg.arg for arg in func.args.kwonlyargs}
        read = {
            node.id
            for node in ast.walk(func)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for param in sorted(params):
            if param.endswith("_run_dir"):
                assert param in read, (
                    f"{script_name}::_pair_spec accepts {param} but never reads it"
                )
