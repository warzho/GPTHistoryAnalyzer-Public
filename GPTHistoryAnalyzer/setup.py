#!/usr/bin/env python3
"""
Setup script for GPT History Analyzer
Helps users configure the system for first-time use
"""

import os
import sys
from pathlib import Path


def main():
    print("🚀 GPT History Analyzer Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return 1
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check if config exists
    config_file = Path("config/analysis_config.py")
    if not config_file.exists():
        print("❌ Configuration file not found")
        return 1
    
    print("✅ Configuration file found")
    
    # Check for API key placeholder
    with open(config_file, 'r') as f:
        content = f.read()
        if "sk-proj-YOUR_ACTUAL_API_KEY_HERE" in content:
            print("⚠️  API key not configured")
            print("   Please update OPENAI_API_KEY in config/analysis_config.py")
        else:
            print("✅ API key appears to be configured")
    
    # Check for file path placeholder
    if r"C:\path\to\your\chat_export.json" in content:
        print("⚠️  Chat export file path not configured")
        print("   Please update CHAT_EXPORT_FILE in config/analysis_config.py")
    else:
        print("✅ Chat export file path appears to be configured")
    
    # Check for project names placeholder
    if "Project Alpha" in content:
        print("⚠️  Project names not customized")
        print("   Please update TARGET_PROJECTS in config/analysis_config.py")
    else:
        print("✅ Project names appear to be customized")
    
    print("\n📋 Next Steps:")
    print("1. Update config/analysis_config.py with your settings")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Test configuration: python check_config.py")
    print("4. Run analysis: python run-staged-analysis.py --test-only")
    
    print("\n🔒 Security Reminders:")
    print("- Never commit your actual API key to version control")
    print("- Use .env files for sensitive data when possible")
    print("- Review SECURITY.md for important security information")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
