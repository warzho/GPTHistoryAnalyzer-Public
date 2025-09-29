#!/usr/bin/env python3
"""Compatibility wrapper for the unified batch diagnostics CLI."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so we can import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.batch_diagnostics import main as diagnostics_main


def _build_args(argv: list[str]) -> list[str]:
    if argv:
        return ["config", *argv]
    return ["config"]


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    return diagnostics_main(_build_args(argv))


if __name__ == "__main__":
    sys.exit(main())
