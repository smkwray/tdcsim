"""The installed distribution must carry every top-level source module."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_every_top_level_source_module_is_packaged() -> None:
    # Production runs import the installed wheel, not src/, so a module missing from
    # py-modules imports fine in tests and fails only in a release run.
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packaged = set(config["tool"]["setuptools"]["py-modules"])
    source = {path.stem for path in (PROJECT_ROOT / "src").glob("*.py")}
    assert source - packaged == set()
    assert packaged - source == set()
