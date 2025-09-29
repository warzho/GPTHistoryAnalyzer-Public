#!/usr/bin/env python3
"""Compatibility shim for the unified batch diagnostics CLI."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so we can import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.batch_diagnostics import main as diagnostics_main


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    return diagnostics_main(["errors", *argv])


if __name__ == "__main__":
    sys.exit(main())
