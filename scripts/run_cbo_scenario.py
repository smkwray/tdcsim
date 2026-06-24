#!/usr/bin/env python3
"""Run a TDCSIM CBO scenario from a release-bound baseline package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tdcsim_cbo.baseline import CboBaselinePackage
from tdcsim_cbo.contract import CboScenarioSpec
from tdcsim_cbo.runner import run_cbo_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--attestation", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=["summary", "compact", "audit"], default=None)
    args = parser.parse_args(argv)
    baseline = CboBaselinePackage.open(args.baseline, attestation_path=args.attestation)
    spec = CboScenarioSpec.from_file(args.scenario)
    run = run_cbo_scenario(baseline, spec, args.output_dir, output_profile=args.profile)
    print(run.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
