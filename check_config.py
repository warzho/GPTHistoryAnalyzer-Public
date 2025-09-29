#!/usr/bin/env python3
"""
Check what attributes are actually in your AnalysisConfig
This helps us understand your config structure
"""

import sys
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

print("Config Attribute Inspector")
print("=" * 50)

try:
    from config.analysis_config import AnalysisConfig

    print("✅ Successfully imported AnalysisConfig\n")

    print("Attributes found in your config:")
    print("-" * 40)

    # Get all attributes that don't start with underscore
    attributes = [attr for attr in dir(AnalysisConfig) if not attr.startswith('_')]

    # Categorize attributes
    prompt_attrs = []
    token_attrs = []
    file_attrs = []
    key_attrs = []
    other_attrs = []

    for attr in attributes:
        value = getattr(AnalysisConfig, attr)

        # Skip methods and classes
        if callable(value):
            continue

        attr_lower = attr.lower()

        if 'prompt' in attr_lower:
            prompt_attrs.append(
                (attr, type(value).__name__, str(value)[:100] + '...' if len(str(value)) > 100 else str(value)))
        elif 'token' in attr_lower:
            token_attrs.append((attr, type(value).__name__, value))
        elif 'file' in attr_lower or 'path' in attr_lower:
            file_attrs.append((attr, type(value).__name__, value))
        elif 'key' in attr_lower or 'api' in attr_lower:
            # Don't print actual API keys
            if isinstance(value, str) and len(value) > 20:
                key_attrs.append((attr, type(value).__name__, value[:10] + '...[hidden]'))
            else:
                key_attrs.append((attr, type(value).__name__, value))
        else:
            other_attrs.append((attr, type(value).__name__,
                                str(value)[:100] if isinstance(value, str) and len(str(value)) > 100 else value))

    # Print categorized attributes
    if prompt_attrs:
        print("\n📝 PROMPT ATTRIBUTES:")
        for name, type_name, value in prompt_attrs:
            print(f"  {name} ({type_name})")
            if isinstance(value, str) and len(value) > 50:
                print(f"    Preview: {value[:50]}...")
            else:
                print(f"    Value: {value}")

    if token_attrs:
        print("\n🔢 TOKEN ATTRIBUTES:")
        for name, type_name, value in token_attrs:
            print(f"  {name} ({type_name}): {value}")

    if file_attrs:
        print("\n📁 FILE ATTRIBUTES:")
        for name, type_name, value in file_attrs:
            print(f"  {name} ({type_name}): {value}")

    if key_attrs:
        print("\n🔑 API KEY ATTRIBUTES:")
        for name, type_name, value in key_attrs:
            print(f"  {name} ({type_name}): {value}")

    if other_attrs:
        print("\n📦 OTHER ATTRIBUTES:")
        for name, type_name, value in other_attrs:
            print(f"  {name} ({type_name}): {value}")

    # Check for the specific attributes we need
    print("\n" + "=" * 50)
    print("Checking for required attributes:")
    print("-" * 40)

    required = {
        'OPENAI_API_KEY': False,
        'CHAT_EXPORT_FILE': False,
        'Prompts (in staged_batch_analyzer.py)': False,
        'MAX_TOKENS or similar': False
    }

    # Check for API key
    if hasattr(AnalysisConfig, 'OPENAI_API_KEY'):
        required['OPENAI_API_KEY'] = True
    elif hasattr(AnalysisConfig, 'API_KEY'):
        print("  Note: Found API_KEY instead of OPENAI_API_KEY")
        required['OPENAI_API_KEY'] = True

    # Check for chat file
    if hasattr(AnalysisConfig, 'CHAT_EXPORT_FILE'):
        required['CHAT_EXPORT_FILE'] = True
    elif hasattr(AnalysisConfig, 'EXPORT_FILE'):
        print("  Note: Found EXPORT_FILE instead of CHAT_EXPORT_FILE")
        required['CHAT_EXPORT_FILE'] = True

    # Check for prompts in staged_batch_analyzer.py
    print("\n📝 Checking for prompts in staged_batch_analyzer.py...")
    staged_analyzer_path = Path("staged_batch_analyzer.py")
    if staged_analyzer_path.exists():
        with open(staged_analyzer_path, 'r', encoding='utf-8') as f:
            analyzer_content = f.read()

        # Check if the default prompts are defined
        if ('project_prompt' in analyzer_content and
                'inquiry_prompt' in analyzer_content and
                'knowledge_prompt' in analyzer_content):
            required['Prompts (in staged_batch_analyzer.py)'] = True
            print("  ✅ Found default prompts in staged_batch_analyzer.py")
        else:
            print("  ❌ Default prompts not found in staged_batch_analyzer.py")
    else:
        print("  ⚠️  staged_batch_analyzer.py not found")

    # Check for token limits
    for attr in attributes:
        if 'token' in attr.lower() and 'max' in attr.lower():
            required['MAX_TOKENS or similar'] = True
            break

    # Print results
    print("\nRequired attributes status:")
    for key, found in required.items():
        status = "✅" if found else "❌"
        print(f"  {status} {key}")

    # Suggest fixes if needed
    missing = [k for k, v in required.items() if not v]
    if missing:
        print("\n⚠️  Some required attributes may be missing.")

        if 'Prompts (in staged_batch_analyzer.py)' in missing:
            print("\n💡 The staged_batch_analyzer.py should contain default prompts.")
            print("Check that your staged_batch_analyzer.py has the prompt definitions")
            print("in the create_analysis_requests method. The prompts are now")
            print("embedded in the analyzer itself, not in the config file.")

        if 'OPENAI_API_KEY' in missing:
            print("\n💡 To add your OpenAI API key to config, add this line to analysis_config.py:")
            print("    OPENAI_API_KEY = 'your-api-key-here'")

        if 'CHAT_EXPORT_FILE' in missing:
            print("\n💡 To specify your chat export location, add this line to analysis_config.py:")
            print("    CHAT_EXPORT_FILE = 'path/to/your/conversations.json'")
    else:
        print("\n✅ All required attributes found!")
        print("The staged analyzer will use:")
        print("  - Your config for API key and file paths")
        print("  - Built-in default prompts from staged_batch_analyzer.py")
        print("  - Token limits from your config (with intelligent fallbacks)")

except ImportError as e:
    print(f"❌ Could not import AnalysisConfig: {e}")
    print("\nMake sure config/analysis_config.py exists")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error checking config: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
