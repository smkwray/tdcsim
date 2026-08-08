#!/usr/bin/env python3
"""Run a watched TDCSIM CBO scenario from a release-bound baseline package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RELEASE_WHEEL_RUN = bool(os.environ.get("TDCSIM_CBO_WHEEL_PATH"))
if not RELEASE_WHEEL_RUN and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tdcsim_cbo import cli as cbo_cli

if RELEASE_WHEEL_RUN:
    loaded_cli = Path(cbo_cli.__file__).resolve()
    try:
        loaded_cli.relative_to(SRC_DIR)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "release-bound CBO execution must import the installed wheel, "
            "not the project source tree"
        )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    return cbo_cli.main(["run", *raw_argv])


if __name__ == "__main__":
    raise SystemExit(main())
