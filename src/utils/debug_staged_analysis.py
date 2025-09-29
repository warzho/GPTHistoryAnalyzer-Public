#!/usr/bin/env python3
"""
Diagnostic version of staged analysis script
This will help identify where the script is failing
"""

print("DEBUG: Script started")  # This should always print

import sys

print(f"DEBUG: Python version: {sys.version}")

try:
    import json

    print("DEBUG: json imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import json: {e}")
    sys.exit(1)

try:
    import time

    print("DEBUG: time imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import time: {e}")
    sys.exit(1)

try:
    from pathlib import Path

    print("DEBUG: Path imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import Path: {e}")
    sys.exit(1)

try:
    from datetime import datetime, timedelta

    print("DEBUG: datetime imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import datetime: {e}")
    sys.exit(1)

try:
    import argparse

    print("DEBUG: argparse imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import argparse: {e}")
    sys.exit(1)

# Check if staged_batch_analyzer exists
analyzer_path = Path("staged_batch_analyzer.py")
if analyzer_path.exists():
    print(f"DEBUG: Found staged_batch_analyzer.py at {analyzer_path.absolute()}")
else:
    print(f"ERROR: Cannot find staged_batch_analyzer.py in {Path.cwd()}")
    print("Make sure staged_batch_analyzer.py is in the same directory as this script")
    sys.exit(1)

# Add scripts directory to path
script_dir = Path(__file__).parent / "scripts"
print(f"DEBUG: Adding to path: {script_dir}")
sys.path.insert(0, str(script_dir))

# Try to import config
try:
    from config.analysis_config import AnalysisConfig

    print("DEBUG: AnalysisConfig imported successfully")
    print(f"DEBUG: Config has OPENAI_API_KEY: {'Yes' if hasattr(AnalysisConfig, 'OPENAI_API_KEY') else 'No'}")
    print(f"DEBUG: Config has CHAT_EXPORT_FILE: {'Yes' if hasattr(AnalysisConfig, 'CHAT_EXPORT_FILE') else 'No'}")
    if hasattr(AnalysisConfig, 'CHAT_EXPORT_FILE'):
        print(f"DEBUG: CHAT_EXPORT_FILE = {AnalysisConfig.CHAT_EXPORT_FILE}")
except ImportError as e:
    print(f"ERROR: Could not import AnalysisConfig: {e}")
    print("Make sure config/analysis_config.py exists and is properly formatted")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Problem with config file: {e}")
    sys.exit(1)

# Try to import data parser
try:
    from scripts.data_parser import load_and_parse_chat_export

    print("DEBUG: load_and_parse_chat_export imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import data_parser: {e}")
    print("Make sure scripts/data_parser.py exists")
    sys.exit(1)

# Try to import tiktoken
try:
    import tiktoken

    print("DEBUG: tiktoken imported successfully")
except ImportError as e:
    print(f"WARNING: tiktoken not installed: {e}")
    print("Install with: pip install tiktoken")
    # Don't exit - we'll handle this in the analyzer

# Try to import openai
try:
    from openai import OpenAI

    print("DEBUG: OpenAI imported successfully")
except ImportError as e:
    print(f"ERROR: OpenAI library not installed: {e}")
    print("Install with: pip install openai")
    sys.exit(1)

# Try to import the staged analyzer
try:
    from staged_batch_analyzer import StagedBatchAnalyzer

    print("DEBUG: StagedBatchAnalyzer imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import StagedBatchAnalyzer: {e}")
    print("Check that staged_batch_analyzer.py is in the current directory")
    import traceback

    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Problem with staged_batch_analyzer.py: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\nDEBUG: All imports successful!")
print("-" * 50)


def test_basic_functionality():
    """Test basic functionality before running main analysis"""
    print("\nDEBUG: Testing basic functionality...")

    # Test file access
    try:
        chat_file = Path(AnalysisConfig.CHAT_EXPORT_FILE)
        if chat_file.exists():
            print(f"✓ Chat export file exists: {chat_file}")
            print(f"  File size: {chat_file.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            print(f"✗ Chat export file not found: {chat_file}")
            print(f"  Current directory: {Path.cwd()}")
            print(f"  Looking for: {chat_file.absolute()}")
            return False
    except Exception as e:
        print(f"✗ Error checking chat file: {e}")
        return False

    # Test loading conversations
    try:
        print("\nDEBUG: Attempting to load conversations...")
        conversations = load_and_parse_chat_export(AnalysisConfig.CHAT_EXPORT_FILE)
        print(f"✓ Loaded {len(conversations)} conversations")

        if len(conversations) > 0:
            # Check structure of first conversation
            first_conv = conversations[0]
            print(f"  First conversation keys: {list(first_conv.keys())[:5]}...")

            # Check date field
            if 'create_time' in first_conv:
                create_time = first_conv['create_time']
                print(f"  create_time type: {type(create_time)}")
                print(f"  create_time value: {create_time}")
            else:
                print("  WARNING: No 'create_time' field found")
        else:
            print("✗ No conversations found in file")
            return False

    except Exception as e:
        print(f"✗ Error loading conversations: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test analyzer initialization
    try:
        print("\nDEBUG: Testing analyzer initialization...")
        analyzer = StagedBatchAnalyzer(AnalysisConfig)
        print("✓ Analyzer initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test date filtering
    try:
        print("\nDEBUG: Testing date filtering...")
        filtered = analyzer.filter_conversations_by_date(conversations[:5], months_back=12)
        print(f"✓ Date filtering works: {len(filtered)} conversations passed filter")
    except Exception as e:
        print(f"✗ Error in date filtering: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Main function with extensive debugging"""
    print("\nDEBUG: Entering main()")

    parser = argparse.ArgumentParser(description='Debug staged analysis')
    parser.add_argument('--test-only', action='store_true', help='Only run tests, don\'t start analysis')

    print("DEBUG: Parsing arguments...")
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: {args}")

    # Run basic tests
    if test_basic_functionality():
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)

        if not args.test_only:
            print("\nThe system appears to be working correctly.")
            print("To run the actual analysis, use:")
            print("  python run_staged_analysis.py --months 1")
        else:
            print("\nTest mode complete. Add --months flag to run actual analysis:")
            print("  python debug_staged_analysis.py --months 1")
    else:
        print("\n" + "=" * 50)
        print("❌ Tests failed - fix the issues above before proceeding")
        print("=" * 50)
        return 1

    return 0


if __name__ == "__main__":
    print("DEBUG: __name__ == '__main__' check passed")
    try:
        exit_code = main()
        print(f"\nDEBUG: Script completing with exit code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nERROR: Uncaught exception in main: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDEBUG: Script interrupted by user")
        sys.exit(1)