# Updated configuration for JSON-only processing
import math
import os
from pathlib import Path


class AnalysisConfig:
    """
    Updated configuration supporting JSON file processing with both batch and real-time API options.
    This version removes all HTML-specific references and expects JSON exports.
    """

    # =============================================================================
    # IMPORTANT: UPDATE THESE VALUES WITH YOUR ACTUAL SETTINGS
    # =============================================================================

    # Your OpenAI API Key - REPLACE WITH YOUR ACTUAL API KEY
    OPENAI_API_KEY = "sk-proj-YOUR_ACTUAL_API_KEY_HERE"

    # Path to your JSON chat export file - UPDATED TO USE JSON EXTENSION
    # Simply change the extension from .html to .json for your existing file
    CHAT_EXPORT_FILE = r"C:\path\to\your\chat_export.json"

    # Specific projects you want to track - REPLACE WITH YOUR ACTUAL PROJECT NAMES
    TARGET_PROJECTS = ["Project Alpha", "Project Beta", "Learning Initiative", "Research Topic",
                       "Personal Development", "Work Project", "Side Hustle", "Hobby Project"]

    # =============================================================================
    # PROCESSING MODE SETTINGS
    # =============================================================================

    # Choose processing mode: 'batch' for 50% cost savings, 'realtime' for immediate results
    PROCESSING_MODE = "batch"  # Options: "batch" or "realtime"

    # Models to use for different types of analysis
    ANALYSIS_MODEL = "gpt-4o-mini"  # Primary model for all analysis flows
    SYNTHESIS_MODEL = "gpt-4o-mini"  # Keep synthesis on the same model for consistency

    # For backward compatibility - maps to ANALYSIS_MODEL
    DEEP_RESEARCH_MODEL = ANALYSIS_MODEL

    # Model/token planning limits for gpt-4o-mini
    MAX_INPUT_TOKENS = 272_000  # Hard input ceiling aligned with gpt-4o-mini context window
    MAX_OUTPUT_TOKENS = 128_000
    CONTEXT_WINDOW = 400_000
    INPUT_USAGE_TARGET = 0.8  # Aim to stay within 80% of the input window
    SYSTEM_OVERHEAD_TOKENS = 3_000  # Reserve for system prompts + formatting

    # Throughput and queue guardrails for gpt-4o-mini
    TOKENS_PER_MINUTE_LIMIT = 500_000
    REQUESTS_PER_MINUTE_LIMIT = 500
    BATCH_TOKENS_PER_DAY_LIMIT = 1_500_000

    # Batch processing settings
    BATCH_SIZE = 100  # Number of conversations per batch job
    BATCH_COMPLETION_WINDOW = "24h"  # How long OpenAI has to complete the batch
    MAX_CONVERSATIONS_PER_BATCH = 100  # Maximum conversations to process in one batch
    MAX_REQUESTS_PER_JOB = 250  # Safety cap to keep JSONL payload sizes reasonable
    MAX_BATCH_FILE_BYTES = 180_000_000  # ~180 MB per upload to avoid "File is too large"
    # Reasonable output limit for batch analysis
    BATCH_MAX_OUTPUT_TOKENS = 4000

    # Real-time processing settings (fallback for small datasets)
    REALTIME_THRESHOLD = 10  # Use real-time if fewer conversations than this

    # =============================================================================
    # DIRECTORY AND FILE PATHS
    # =============================================================================

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    DATA_BATCH_DIR = PROJECT_ROOT / "data" / "batch"  # For batch files
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Batch processing file paths
    BATCH_REQUESTS_FILE = DATA_BATCH_DIR / "analysis_requests.jsonl"
    BATCH_JOBS_FILE = DATA_BATCH_DIR / "active_jobs.json"
    BATCH_RESULTS_DIR = DATA_BATCH_DIR / "results"

    @classmethod
    def get_input_token_budget(cls) -> int:
        """Return the target maximum tokens to allocate to conversation content per request."""

        base_limit = math.floor(cls.MAX_INPUT_TOKENS * cls.INPUT_USAGE_TARGET)
        # Always leave room for the reserved system/formatting overhead
        return max(base_limit - cls.SYSTEM_OVERHEAD_TOKENS, 1)

    @classmethod
    def clamp_input_tokens(cls, input_tokens: int) -> int:
        """Ensure input tokens never exceed the model hard limit."""

        return min(input_tokens, cls.MAX_INPUT_TOKENS)

    @classmethod
    def plan_output_tokens(cls, input_tokens: int) -> int:
        """Plan a conservative max_tokens value for a request based on the input size."""

        safe_input = cls.clamp_input_tokens(input_tokens)
        planned = min(int(round(0.2 * safe_input)), cls.MAX_OUTPUT_TOKENS)

        # Respect the overall context window
        if safe_input + planned > cls.CONTEXT_WINDOW:
            planned = max(cls.CONTEXT_WINDOW - safe_input, 0)

        return max(planned, 0)

    @classmethod
    def validate_config(cls):
        """
        Enhanced validation that checks for JSON files and batch processing setup.
        This version specifically validates JSON file format instead of HTML.
        """
        errors = []

        # Validate API key
        if cls.OPENAI_API_KEY.startswith("sk-proj-YOUR_ACTUAL"):
            errors.append("Please set your actual OpenAI API key in OPENAI_API_KEY")

        # Validate chat export file - now checking for JSON
        if not cls.CHAT_EXPORT_FILE.endswith('.json'):
            errors.append(f"Chat export file must be a JSON file, got: {cls.CHAT_EXPORT_FILE}")
            errors.append("Please ensure your JSON export file path ends with '.json'")

        # Check if the JSON file exists
        chat_file_path = Path(cls.CHAT_EXPORT_FILE)
        if not chat_file_path.exists():
            # Check if the old HTML file exists to provide helpful migration message
            html_path = Path(cls.CHAT_EXPORT_FILE.replace('.json', '.html'))
            if html_path.exists():
                errors.append(f"Found HTML file but JSON file missing: {cls.CHAT_EXPORT_FILE}")
                errors.append(f"Please extract the JSON data from your HTML file first")
            else:
                errors.append(f"Chat export file not found: {cls.CHAT_EXPORT_FILE}")
        else:
            # Validate that it's actually a JSON file
            try:
                import json
                with open(chat_file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"✅ Valid JSON file found: {chat_file_path.name}")
            except json.JSONDecodeError as e:
                errors.append(f"File exists but is not valid JSON: {e}")
                errors.append("Please ensure your export file contains valid JSON data")
            except Exception as e:
                errors.append(f"Error reading JSON file: {e}")

        # Create required directories
        try:
            cls.DATA_BATCH_DIR.mkdir(parents=True, exist_ok=True)
            cls.BATCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            cls.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            cls.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Could not create required directories: {e}")

        if errors:
            print("❌ Configuration errors found:")
            for error in errors:
                print(f"   - {error}")
            return False

        print("✅ Configuration validation passed!")
        print(f"📊 Processing mode: {cls.PROCESSING_MODE}")
        print(f"📁 JSON export file: {Path(cls.CHAT_EXPORT_FILE).name}")
        if cls.PROCESSING_MODE == "batch":
            print(f"💰 Cost savings: ~50% compared to real-time processing")
        return True

    @classmethod
    def migrate_from_html_config(cls, html_file_path: str):
        """
        Helper method to migrate from HTML to JSON configuration.
        This updates the configuration to use the JSON version of the file.
        """
        json_file_path = html_file_path.rsplit('.', 1)[0] + '.json'
        cls.CHAT_EXPORT_FILE = json_file_path
        print(f"📝 Configuration updated to use JSON file: {json_file_path}")
        return json_file_path