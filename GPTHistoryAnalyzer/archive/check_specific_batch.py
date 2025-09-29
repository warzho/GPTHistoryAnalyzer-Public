#!/usr/bin/env python3
"""Compatibility wrapper for checking a single batch via the new CLI."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so we can import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.batch_diagnostics import main as diagnostics_main

DEFAULT_BATCH_ID = "batch_68d82814b5bc8190840ac5f654c7c749"


def _build_args(argv: list[str]) -> list[str]:
    batch_ids = argv or [DEFAULT_BATCH_ID]
    args: list[str] = ["status"]
    for batch_id in batch_ids:
        args.extend(["--batch", batch_id])
    return args


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    return diagnostics_main(_build_args(argv))


if __name__ == "__main__":
    sys.exit(main())
