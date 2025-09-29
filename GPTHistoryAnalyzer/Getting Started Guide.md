# GPT History Analyzer - Complete Setup Guide

A comprehensive system for analyzing your ChatGPT conversation history using OpenAI's batch processing API. This tool helps you identify patterns in your projects, learning approaches, and knowledge evolution over time.

## Prerequisites - What You Need Before Starting

### 1. Python Installation (Required)
Install Python 3.8 or newer and confirm the interpreter is on your PATH.

```bash
python --version    # or: python3 --version
```

If the command reports a version earlier than 3.8 (or is missing entirely), install the current release from [python.org](https://www.python.org/downloads/). Windows users should select **Add python.exe to PATH** during setup; macOS users can rely on the official installer or Homebrew if preferred.

> **Tip:** When multiple interpreters exist, use the fully qualified path (for example, `C:\Python311\python.exe --version`) to confirm which build your environment will use.

### 2. OpenAI API Key (Required)

Generate an API key from the [OpenAI dashboard](https://platform.openai.com/account/api-keys) and store it securely. The key appears only once—capture it immediately, add credits to your account, and plan for pay-as-you-go charges (batch processing typically costs $0.50–$2.00 per 1,000 conversations).

### 3. Your ChatGPT Export File (Required)

Request a JSON export of your conversation history from the ChatGPT **Settings → Data controls → Export data** workflow. OpenAI emails a ZIP archive; extract it and keep `conversations.json` available for configuration. Consult [OpenAI's export guide](https://help.openai.com/en/articles/7261859-export-your-chatgpt-data) if you need platform-specific navigation tips.

## Complete Installation Guide

### Step 1: Download the Project

```bash
git clone https://github.com/your-username/GPTHistoryAnalyzer.git
```

If you prefer a manual download, grab the ZIP from GitHub’s **Code ▸ Download ZIP** option and extract it to a convenient workspace.

### Step 2: Open a Terminal in the Project Folder

Use your shell of choice and change into the project directory:

```bash
cd /path/to/GPTHistoryAnalyzer
```

Refer to the official documentation for [Windows Terminal](https://learn.microsoft.com/windows/terminal/) or [macOS Terminal](https://support.apple.com/guide/terminal/welcome/mac) if you need a refresher on shell navigation commands.

### Step 3: Create a Virtual Environment

```bash
python -m venv env    # or: python3 -m venv env
```

> **Tip:** Ensure you run this from the repository root so the `env` directory lives alongside the project files.

### Step 4: Activate the Virtual Environment

```powershell
./env/Scripts/Activate.ps1   # Windows PowerShell
```

```bash
source env/bin/activate      # macOS/Linux shells
```

> **Edge case (Windows):** If PowerShell blocks script execution, adjust the execution policy as described in the [official documentation](https://learn.microsoft.com/powershell/module/microsoft.powershell.security/set-executionpolicy) and rerun the activation command.

### Step 5: Install Required Packages

```bash
pip install openai pandas numpy python-dotenv jsonlines tqdm tiktoken
```

Confirm `pip --version` reports the interpreter from your virtual environment before installing. The command completes in a few minutes depending on network speed.

## Configuration - Setting Up Your Settings

### Step 1: Find and Open the Configuration File

1. In the GPTHistoryAnalyzer folder, find the `config` folder
2. Inside, find `analysis_config.py`
3. Right-click it and select "Edit" or "Open with" → Notepad (Windows) or TextEdit (Mac)

### Step 2: Update the Configuration

#### 1. Add Your API Key:
Find this line:
```python
OPENAI_API_KEY = "sk-proj-your-actual-api-key-here"
```
Replace everything between the quotes with your actual API key:
```python
OPENAI_API_KEY = "sk-proj-ABC123...your-real-key-here"
```

**SECURITY WARNING**: Never share this file with anyone after adding your key!


#### 2. Add Your Chat Export File Path:
Find this line:
```python
CHAT_EXPORT_FILE = r"C:\path\to\your\chat_export.json"
```

Replace with the actual path to your conversations.json file.

**How to find the correct path:**

**Windows:**
1. Find your conversations.json file in File Explorer
2. Right-click it
3. Select "Properties"
4. Copy the "Location" path
5. Add the filename at the end
6. Your path should look like: `r"C:\Users\YourName\Downloads\conversations.json"`
   (Note the `r` before the quotes - keep that!)

**Mac:**
1. Find your conversations.json file in Finder
2. Right-click and select "Get Info"
3. Copy the path shown under "Where"
4. Your path should look like: `"/Users/YourName/Downloads/conversations.json"`

### Step 3: Specify Your Projects (Optional but Recommended)

Find this line:
```python
TARGET_PROJECTS = ["Project Name 1", "Project Name 2"]
```

Replace with topics you want to track. Examples:
```python
TARGET_PROJECTS = ["Python Learning", "Work Reports", "Creative Writing", "Research Papers"]
```

### Step 4: Save the File

1. Press `Ctrl+S` (Windows) or `Command+S` (Mac)
2. Close the editor

## Running Your First Analysis

### Step 1: Test Your Configuration

In your terminal/command prompt (with `(env)` showing), type:
```bash
python check_config.py
```

You should see:
```
✅ Configuration validation passed!
📊 Processing mode: batch
💰 Cost savings: ~50% compared to real-time processing
```

**If you see errors:**
- Double-check your API key
- Make sure your file path is correct

### Step 2: Estimate Costs (Recommended)

Check how much the analysis will cost:
```bash
python scripts/cost_estimator.py
```

This shows you the estimated cost before you commit to running the analysis.

### Step 3: Run the Analysis

For your most recent 6 months of conversations:
```bash
python run-staged-analysis.py --months 6
```

**What happens next:**
1. The system loads your conversations
2. Breaks them into chunks
3. Submits them to OpenAI for processing
4. **IMPORTANT**: Processing takes up to 24 hours
5. You'll see progress updates every 5 minutes

### Step 4: Check Progress

To see how your analysis is progressing:
```bash
python run-staged-analysis.py --status
```

### Step 5: Create Your Knowledge Base

Once all batches are complete:
```bash
python synthesize_knowledge.py
```

This creates:
- `knowledge_base/knowledge_report.md` - A readable report
- `knowledge_base/knowledge_base.json` - Raw data

## Understanding OpenAI Model Options

### Current Available Models (As of 2025)

#### GPT-4o-mini
- **Context Window**: 128,000 tokens (~200 pages)
- **Output Limit**: 16,000 tokens per request
- **Cost**: $0.15 per 1M input tokens, $0.60 per 1M output tokens
- **With Batch API**: 50% discount (24-hour processing)

#### GPT-4o
- **Context Window**: 128,000 tokens
- **Output Limit**: 4,096 tokens per request
- **Cost**: Higher than GPT-4o-mini

#### GPT-4.1 Series 
- **Context Window**: Up to 1,000,000 tokens (~1,500 pages)
- **Models**: GPT-4.1, GPT-4.1-mini, GPT-4.1-nano
- **Note**: May require API access approval

#### GPT-5 
- **Context Window**: 400,000 tokens (configured default in `config/analysis_config.py`)
- **Output Limit**: 128,000 tokens per response (the project clamps via `MAX_OUTPUT_TOKENS`)
- **Project Configuration**: Set `ANALYSIS_MODEL`/`SYNTHESIS_MODEL` to `"gpt-5-mini"` in `config/analysis_config.py` to target GPT-5 while respecting the guardrails baked into this repo

**Batch-specific note (GPT-5 + Batch API):**
> **Batch API with GPT-5:** Each queued request must fit within 272,000 input tokens (`MAX_INPUT_TOKENS`) plus 128,000 output tokens. The project caps batch submissions at 100 conversations per job (`BATCH_SIZE`/`MAX_CONVERSATIONS_PER_BATCH`) and 250 total requests (`MAX_REQUESTS_PER_JOB`) to stay under the API’s queue thresholds. Expect OpenAI to process GPT-5 batch jobs within the 24-hour window (`BATCH_COMPLETION_WINDOW`).

### Token Limits Explained

**What are tokens?**
- Tokens are pieces of words
- 1 token ≈ 4 characters in English
- 100 tokens ≈ 75 words
- Your conversations use tokens for both input (what you send) and output (what you receive)

**Why token limits matter:**
- Each model can only process a certain amount of text at once
- The system automatically splits your conversations to fit within limits
- Larger limits = can analyze more context at once = better insights

## Troubleshooting Guide

### Problem: "ModuleNotFoundError: No module named 'openai'"

**Solution:**
1. Make sure your virtual environment is activated (you see `(env)`)
2. Run: `pip install openai --upgrade`

### Problem: "Invalid API Key"

**Solution:**
1. Check your API key starts with `sk-`
2. Make sure you copied the entire key
3. Verify you have credits in your OpenAI account
4. Try generating a new key if needed

### Problem: "File not found" for conversations.json

**Solution:**
1. Check the file path is exactly correct
2. On Windows, make sure to keep the `r` before the quotes
3. Try moving the file to a simpler location like `C:\temp\conversations.json`


### Problem: Batch takes too long

**Remember:**
- Batch processing takes up to 24 hours
- This is normal and saves you 50% on costs
- Check progress with: `python run-staged-analysis.py --status`

### Problem: "Token limit exceeded"

**Solution:**
1. Reduce the number of months analyzed: `--months 3`
2. Or reduce BATCH_SIZE in config to 25

## Cost Optimization Tips

1. **Start Small**: Test with 1-3 months first
2. **Use Batch Processing**: Always use batch mode for 50% savings
3. **Monitor Progress**: Check status regularly to catch any issues early
4. **Estimate First**: Always run cost estimation before full analysis

## Security Best Practices

1. **Never share your API key**
2. **Don't commit config files to Git** after adding your key
3. **Store your API key in environment variables** for production use
4. **Regularly rotate your API keys** (regenerate every few months)
5. **Set usage limits** in your OpenAI account to prevent unexpected charges

## Getting Help

If you're stuck:
1. **Check the troubleshooting section** above
2. **Read error messages carefully** - they often tell you exactly what's wrong
3. **Try running `python check_config.py`** to validate your setup
4. **Start with a smaller analysis** (1 month) to test everything works

## What's Next?

After your analysis completes:
1. Open `knowledge_base/knowledge_report.md` in any text editor
2. Review your:
   - Project development patterns
   - Learning style insights
   - Knowledge evolution
   - Personalized recommendations
3. Use insights to improve your ChatGPT interactions
4. Run monthly to track your progress

---
