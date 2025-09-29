#!/usr/bin/env python3
"""
Main Script for Running Staged Analysis
This is your primary interface for processing chat history within token limits
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add scripts directory to path for imports
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

# Import your existing modules
from scripts.data_parser import load_and_parse_chat_export
from config.analysis_config import AnalysisConfig

# Import the new staged analyzer (you'll need to move the class to scripts/)
from staged_batch_analyzer import StagedBatchAnalyzer


def main():
    """
    Main execution function - your starting point for analysis
    """
    parser = argparse.ArgumentParser(
        description='Analyze chat history with automatic batch staging'
    )

    parser.add_argument(
        '--months',
        type=int,
        default=12,
        help='Number of months of history to analyze (default: 12)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress if interrupted'
    )

    parser.add_argument(
        '--combine-only',
        action='store_true',
        help='Only combine existing results without new submissions'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Check status of current analysis progress'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("            STAGED CHAT ANALYSIS SYSTEM")
    print("=" * 70)
    print("\nThis system will automatically:")
    print("  1. Filter your chat history to the specified time period")
    print("  2. Break it into chunks that fit within token limits")
    print("  3. Submit batches sequentially to avoid queue overflow")
    print("  4. Track progress and allow resumption if interrupted")
    print("  5. Combine all results into a final analysis")
    print("\n" + "=" * 70)

    # Initialize the staged analyzer
    analyzer = StagedBatchAnalyzer(AnalysisConfig)

    # Handle different modes
    if args.status:
        show_progress_status(analyzer)
        return 0

    if args.combine_only:
        print("\n📊 Combining existing results...")
        results = analyzer.combine_all_results()
        print("\n✅ Results combined successfully!")
        return 0

    # Main analysis flow
    try:
        # Step 1: Load your chat data
        print(f"\n📂 Loading chat export from: {AnalysisConfig.CHAT_EXPORT_FILE}")
        conversations = load_and_parse_chat_export(AnalysisConfig.CHAT_EXPORT_FILE)
        print(f"   Found {len(conversations)} total conversations")

        # Step 2: Run the staged analysis
        print(f"\n🔍 Analyzing past {args.months} months of history")

        if args.resume:
            print("   Resuming from previous progress...")

        # This is where the magic happens - the analyzer handles everything
        # Fixing to properly handle month specification
        result = analyzer.submit_staged_batches(conversations, months_back=args.months)

        if result['success']:
            print("\n" + "=" * 70)
            print("           ✨ ANALYSIS COMPLETE! ✨")
            print("=" * 70)

            # Step 3: Combine all results
            print("\n📊 Combining all batch results into final analysis...")
            final_results = analyzer.combine_all_results()

            print("\n🎯 Next Steps:")
            print("  1. Check 'staged_results/' directory for all batch results")
            print("  2. Review 'staged_results/combined_analysis.json' for merged data")
            print("  3. Run your knowledge synthesis on the combined results")

            # Provide the command for synthesis
            print("\n💡 To create your knowledge base, run:")
            print("     python synthesize_knowledge.py --input staged_results/combined_analysis.json")

        else:
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
            print("\n💡 You can resume by running:")
            print("     python run_staged_analysis.py --resume")

    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        print("💡 You can resume by running:")
        print("     python run_staged_analysis.py --resume")
        return 1

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def show_progress_status(analyzer: StagedBatchAnalyzer):
    """
    Displays current progress of the analysis
    This helps you understand where you are in the process
    """
    progress = analyzer.load_progress()

    if not progress.get('batches'):
        print("\n📭 No analysis in progress")
        print("Start a new analysis with: python run_staged_analysis.py")
        return

    print("\n📊 ANALYSIS PROGRESS REPORT")
    print("=" * 50)

    started = progress.get('started_at', 'Unknown')
    print(f"Started: {started}")

    batches = progress.get('batches', {})
    total_batches = len(batches)
    completed = sum(1 for b in batches.values() if b.get('status') == 'completed')
    failed = sum(1 for b in batches.values() if b.get('status') == 'failed')
    in_progress = sum(1 for b in batches.values() if b.get('status') == 'submitted')

    print(f"\nBatch Statistics:")
    print(f"  Total Batches: {total_batches}")
    print(f"  ✅ Completed: {completed}")
    print(f"  ⏳ In Progress: {in_progress}")
    print(f"  ❌ Failed: {failed}")

    if total_batches > 0:
        completion_percent = (completed / total_batches) * 100
        print(f"  📈 Overall Progress: {completion_percent:.1f}%")

    print("\nBatch Details:")
    for batch_num, batch_info in sorted(batches.items(), key=lambda x: int(x[0])):
        status = batch_info.get('status', 'unknown')
        batch_id = batch_info.get('batch_id', 'N/A')

        status_icon = {
            'completed': '✅',
            'submitted': '⏳',
            'failed': '❌'
        }.get(status, '❓')

        print(f"  Batch {batch_num}: {status_icon} {status} (ID: {batch_id[:20]}...)")

    if in_progress > 0:
        print("\n💡 Batches are still processing. Check again later or run:")
        print("     python run_staged_analysis.py --resume")
    elif failed > 0:
        print("\n⚠️ Some batches failed. Review errors and consider resubmitting")
    elif completed == total_batches:
        print("\n✨ All batches completed! You can now combine results:")
        print("     python run_staged_analysis.py --combine-only")


if __name__ == "__main__":
    sys.exit(main())