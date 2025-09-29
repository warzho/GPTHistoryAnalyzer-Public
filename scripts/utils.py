"""Utility helpers for configuration validation and cost estimation."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken

from config.analysis_config import AnalysisConfig


class TokenCounter:
    """Utility for counting tokens with consistent special-token handling."""

    def __init__(self, model: str = AnalysisConfig.ANALYSIS_MODEL) -> None:
        candidates = [
            model,
            "gpt-4o-mini",
            "gpt-4o",
            "o200k_base",
            "cl100k_base",
        ]
        encoding = None
        for name in candidates:
            try:
                if name.endswith("_base"):
                    encoding = tiktoken.get_encoding(name)
                else:
                    encoding = tiktoken.encoding_for_model(name)
                break
            except Exception:
                continue

        if encoding is None:
            raise RuntimeError("Unable to initialise tokenizer for token planning")

        self.encoding = encoding
        # Treat exported special markers like ``<|endoftext|>`` as plain text so that
        # tokenisation never aborts when these markers appear in titles or messages.
        self.encode_kwargs = {"disallowed_special": ()}

    def count_text(self, text: str) -> int:
        """Return the number of tokens required to represent *text*."""

        if not text:
            return 0
        return len(self.encoding.encode(text, **self.encode_kwargs))

    def count_chat_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate chat-completion tokens following OpenAI's guidelines."""

        total = 0
        for message in messages:
            # Every message contributes a small fixed overhead plus the content.
            total += 4
            total += self.count_text(message.get("content", ""))
            if message.get("name"):
                # Name values reduce the overall cost by one token.
                total -= 1
        # The assistant reply adds two tokens.
        total += 2
        return max(total, 0)


def validate_file_exists(file_path: str) -> bool:
    """Validate that *file_path* exists and contains well-formed JSON."""

    path = Path(file_path).expanduser()
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        html_path = path.with_suffix(".html")
        if html_path.exists():
            print(f"   ℹ️  Detected HTML export at {html_path}. Please convert it to JSON.")
        else:
            print(f"   ℹ️  Working directory: {os.getcwd()}")
        return False

    if not path.is_file():
        print(f"❌ Path exists but is not a file: {file_path}")
        return False

    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"❌ File exists but is not valid JSON: {exc}")
        return False
    except UnicodeDecodeError as exc:
        print(f"❌ File exists but is not UTF-8 encoded: {exc}")
        return False

    print(f"✅ Valid JSON file found: {path.name}")
    return True


def create_backup_of_results() -> Optional[str]:
    """Create a timestamped backup of the processed data and results directories."""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = AnalysisConfig.PROJECT_ROOT / f"backup_{timestamp}"

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        if AnalysisConfig.RESULTS_DIR.exists():
            shutil.copytree(
                AnalysisConfig.RESULTS_DIR,
                backup_dir / "results",
                dirs_exist_ok=True,
            )
        if AnalysisConfig.DATA_PROCESSED_DIR.exists():
            shutil.copytree(
                AnalysisConfig.DATA_PROCESSED_DIR,
                backup_dir / "processed",
                dirs_exist_ok=True,
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"⚠️  Backup creation failed: {exc}")
        return None

    print(f"💾 Created backup at: {backup_dir}")
    return str(backup_dir)


def print_setup_validation() -> bool:
    """Run a comprehensive configuration validation workflow."""

    print("🔧 Validating analysis configuration...\n")
    ok = AnalysisConfig.validate_config()

    if ok:
        ok = validate_file_exists(AnalysisConfig.CHAT_EXPORT_FILE)

    if ok:
        print("\n✅ Setup validation passed!")
    else:
        print("\n❌ Setup validation failed. Please resolve the issues above.")
    return ok


def estimate_batch_analysis_cost(conversations: List[Dict]) -> Dict[str, any]:
    """Compute batch-processing cost estimates for *conversations*."""

    from scripts.cost_estimator import ComprehensiveCostEstimator

    estimator = ComprehensiveCostEstimator()
    return estimator.estimate_comprehensive_analysis_cost(conversations)


def print_cost_comparison_analysis(cost_details: Dict) -> None:
    """Pretty-print a previously computed cost analysis report."""

    summary = cost_details.get("conversation_summary", {})
    costs = cost_details.get("cost_breakdown", {})
    rec = cost_details.get("recommendations", {})

    print("\n" + "=" * 60)
    print("💰 COST ANALYSIS: BATCH vs REAL-TIME")
    print("=" * 60)
    print(
        f"Conversations: {summary.get('total_conversations', 0):,} | "
        f"Tokens: {summary.get('total_conversation_tokens', 0):,}"
    )
    print(f"Model: {cost_details.get('pricing_model', AnalysisConfig.ANALYSIS_MODEL)}")

    batch_pricing = costs.get("batch_pricing", {})
    regular_pricing = costs.get("regular_pricing", {})
    savings = costs.get("savings_analysis", {})

    print("\nRegular API pricing:")
    print(f"  Input cost:  ${regular_pricing.get('input_cost', 0):,.4f}")
    print(f"  Output cost: ${regular_pricing.get('output_cost', 0):,.4f}")
    print(f"  Total:       ${regular_pricing.get('total_cost', 0):,.4f}")

    print("\nBatch API pricing (50% discount assumed):")
    print(f"  Input cost:  ${batch_pricing.get('input_cost', 0):,.4f}")
    print(f"  Output cost: ${batch_pricing.get('output_cost', 0):,.4f}")
    print(f"  Total:       ${batch_pricing.get('total_cost', 0):,.4f}")

    print("\nSavings:")
    print(
        f"  Dollar savings: ${savings.get('dollar_savings', 0):,.4f}"
        f" ({savings.get('percentage_savings', 0):.1f}% reduction)"
    )

    if rec:
        print("\nRecommendation:")
        print(f"  Mode: {rec.get('recommended_mode', 'batch').upper()}")
        if rec.get("reasoning"):
            print(f"  Why: {rec['reasoning']}")

    print("=" * 60 + "\n")
