#!/usr/bin/env python3
"""Verify TDCSIM CBO compiled scenario or scenario-run outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tdcsim_cbo.verifier import verify_compiled_scenario, verify_scenario_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-dir")
    parser.add_argument("--run-dir")
    args = parser.parse_args(argv)
    if bool(args.compiled_dir) == bool(args.run_dir):
        parser.error("provide exactly one of --compiled-dir or --run-dir")
    result = verify_compiled_scenario(args.compiled_dir) if args.compiled_dir else verify_scenario_run(args.run_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
