"""The handoff lookup for the consumer's fallback file resolves by directory, not pair_id.

The downstream consumer opens its fallback summary at a fixed relative path whose last
component is `current_state_2026_plus_100bp_year_source_grade`. That is a directory name. The
pair living there carries `pair_id = ratewall_current_2026_plus100bp_year_source_grade_pair_v1`.

Looking the directory name up in `_pair_dir_map`, which keys by `pair_id`, could never match.
It failed only at the very end of assembly -- after 22 source runs, 15 pairs, and every replay
verification had already completed -- because the lookup is the last step before the challenge
is written. That is the most expensive possible place to discover a name mismatch.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_ratewall_split_pairs.py"


def _module() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _constant(name: str) -> str:
    for node in ast.walk(_module()):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not defined in {SCRIPT.name}")


def test_current_split_pair_id_is_the_consumer_facing_directory_name() -> None:
    """It must stay the directory the consumer opens, not be 'corrected' to a pair_id."""

    value = _constant("CURRENT_SPLIT_PAIR_ID")
    assert value == "current_state_2026_plus_100bp_year_source_grade"
    # A pair_id in this project is prefixed and version-suffixed; this deliberately is not.
    assert not value.startswith("ratewall_")
    assert not value.endswith("_v1")


def test_handoff_does_not_resolve_the_directory_through_the_pair_id_map() -> None:
    """`_pair_dir_map` keys by pair_id, so it must not be asked for a directory name."""

    source = SCRIPT.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "_pair_dir_map" in stripped and "CURRENT_SPLIT_PAIR_ID" in stripped:
            raise AssertionError(
                "the handoff directory is being looked up in the pair_id map: " + stripped
            )
        # The specific shape of the original bug.
        if "generated_pair_dirs.get(CURRENT_SPLIT_PAIR_ID)" in stripped:
            raise AssertionError("handoff lookup uses the pair_id map: " + stripped)


def test_handoff_checks_the_directory_actually_holds_a_pair() -> None:
    """Resolving a path is not proof it exists; the manifest must be checked."""

    source = SCRIPT.read_text(encoding="utf-8")
    assert "current_pair_dir = output_root / CURRENT_SPLIT_PAIR_ID" in source
    assert "if not (current_pair_dir / MANIFEST_FILE).exists():" in source
