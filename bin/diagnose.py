#!/usr/bin/env python3
"""
Diagnostic Entry Point
Unified interface for all diagnostic operations
"""

import sys
from pathlib import Path

# Add tools to path
tools_dir = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from batch_diagnostics import main as diagnostics_main


def main():
    """Main execution function for diagnostics"""
    # Pass all arguments to the unified diagnostics CLI
    return diagnostics_main()


if __name__ == "__main__":
    sys.exit(main())
