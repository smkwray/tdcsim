#!/usr/bin/env python3
"""Recompute the RateWall ingest challenge immediately before handoff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tdcsim_cbo.consumer_challenge import verify_ingest_handoff  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_ingest_handoff(args.package_root.expanduser().resolve())
    print(f"verified {result['transfer_manifest_path']}")
    print(f"transfer_manifest_sha256={result['transfer_manifest_sha256']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
