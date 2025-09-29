#!/usr/bin/env python3
"""
Knowledge Synthesis Entry Point
Combines and synthesizes analysis results
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from synthesizers.synthesize_knowledge import main as synthesize_main


def main():
    """Main execution function for knowledge synthesis"""
    parser = argparse.ArgumentParser(
        description='Synthesize knowledge from analysis results'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path("output/batches"),
        help='Directory containing batch results (default: output/batches)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("output/knowledge"),
        help='Directory for knowledge output (default: output/knowledge)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4',
        help='Model to use for synthesis (default: gpt-4)'
    )
    
    args = parser.parse_args()
    
    print("🧠 Starting Knowledge Synthesis")
    print(f"📁 Input: {args.input_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🤖 Model: {args.model}")
    print("-" * 50)
    
    try:
        # Ensure output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run synthesis
        synthesize_main(args.input_dir, args.output_dir, args.model)
        
        print("\n✅ Knowledge synthesis complete!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
