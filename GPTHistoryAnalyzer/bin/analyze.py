#!/usr/bin/env python3
"""
Main Analysis Entry Point
Clean interface for running chat history analysis
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

# Import modules
from utils.data_parser import load_and_parse_chat_export
from config.analysis_config import AnalysisConfig
from analyzers.staged_batch_analyzer import StagedBatchAnalyzer


def main():
    """Main execution function - your starting point for analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze chat history with automatic batch staging'
    )
    parser.add_argument(
        '--months', 
        type=int, 
        default=12,
        help='Number of months back to analyze (default: 12)'
    )
    parser.add_argument(
        '--test-only', 
        action='store_true',
        help='Only test configuration and imports, don\'t run analysis'
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Chat History Analysis")
    print(f"📅 Analyzing last {args.months} months")
    print("-" * 50)
    
    if args.test_only:
        print("🧪 Test mode - checking configuration...")
        try:
            # Test configuration
            config = AnalysisConfig()
            print(f"✅ Configuration loaded")
            print(f"📁 Chat file: {config.CHAT_EXPORT_FILE}")
            
            # Test data loading
            conversations = load_and_parse_chat_export(config.CHAT_EXPORT_FILE)
            print(f"✅ Loaded {len(conversations)} conversations")
            
            # Test analyzer
            analyzer = StagedBatchAnalyzer(config)
            print(f"✅ Analyzer initialized")
            
            print("\n🎉 All tests passed! Ready to run analysis.")
            print("Run without --test-only to start the actual analysis.")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return 1
    else:
        try:
            # Initialize analyzer
            analyzer = StagedBatchAnalyzer(AnalysisConfig())
            
            # Load and filter conversations
            conversations = load_and_parse_chat_export(AnalysisConfig.CHAT_EXPORT_FILE)
            filtered_conversations = analyzer.filter_conversations_by_date(
                conversations, months_back=args.months
            )
            
            print(f"📊 Processing {len(filtered_conversations)} conversations")
            
            # Run analysis
            analyzer.run_staged_analysis(filtered_conversations)
            
            print("\n✅ Analysis complete!")
            print("📁 Results saved to: output/batches/")
            print("🔍 Use 'python bin/diagnose.py' to check status")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
