#!/usr/bin/env python3
"""
Debug Entry Point
Comprehensive debugging and testing interface
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Add project root to path for config
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the debug functionality
from utils.debug_staged_analysis import main as debug_main


def main():
    """Main execution function for debugging"""
    parser = argparse.ArgumentParser(
        description='Debug and test the analysis system'
    )
    parser.add_argument(
        '--test-only', 
        action='store_true',
        help='Only run tests, don\'t start analysis'
    )
    
    args = parser.parse_args()
    
    print("🔧 Starting Debug Mode")
    print("-" * 50)
    
    # Pass arguments to the debug script
    sys.argv = ['debug.py'] + (['--test-only'] if args.test_only else [])
    
    return debug_main()


if __name__ == "__main__":
    sys.exit(main())
