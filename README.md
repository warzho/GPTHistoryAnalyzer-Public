# GPT History Analyzer

A comprehensive system for analyzing ChatGPT conversation history using OpenAI's batch processing API. This tool processes your exported chat history to identify patterns in project development, learning approaches, and knowledge evolution.

## ⚠️ Security Notice

**Before using this tool:**
1. Replace the sample API key in `config/analysis_config.py`
2. Update file paths to point to your actual chat export
3. Review the [SECURITY.md](SECURITY.md) file for important security information
4. Never commit sensitive data to version control

## Prerequisites

- **Python 3.8+** (tested with Python 3.9+)
- **OpenAI API Key** with batch processing access
- **ChatGPT Export File** in JSON format

### Required Python Packages

The system requires the following packages (see `requirements.txt`):

```
openai>=1.12.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
pathlib
jsonlines>=4.0.0
tqdm>=4.65.0
tiktoken
```

## Installation & Setup

### 1. Clone and Setup Environment

**Windows PowerShell:**
```powershell
# Clone the repository
git clone <repository-url>
cd GPTHistoryAnalyzer

# Create virtual environment
python -m venv env
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Unix/Linux/macOS:**
```bash
# Clone the repository
git clone <repository-url>
cd GPTHistoryAnalyzer

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Analysis Settings

**⚠️ IMPORTANT**: Edit `config/analysis_config.py` with your specific settings:

```python
# Your OpenAI API Key - REPLACE WITH YOUR ACTUAL API KEY
OPENAI_API_KEY = "sk-proj-your-actual-api-key-here"

# Path to your JSON chat export file
CHAT_EXPORT_FILE = r"C:\path\to\your\chat_export.json"

# Specific projects you want to track
TARGET_PROJECTS = ["Project Name 1", "Project Name 2", "Learning Topic"]

# Processing mode: 'batch' for 50% cost savings, 'realtime' for immediate results
PROCESSING_MODE = "batch"

# Model configuration
ANALYSIS_MODEL = "gpt-4o-mini"  # Primary model for analysis
SYNTHESIS_MODEL = "gpt-4o-mini"  # Model for knowledge synthesis
```

### 3. Key Configuration Variables

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required | Must have batch processing access |
| `CHAT_EXPORT_FILE` | Path to JSON export | Required | Must be valid JSON format |
| `ANALYSIS_MODEL` | Model for analysis | `gpt-4o-mini` | Optimized for cost/performance |
| `MAX_INPUT_TOKENS` | Input token limit | `272,000` | For gpt-4o-mini |
| `MAX_OUTPUT_TOKENS` | Output token limit | `128,000` | For gpt-4o-mini |
| `BATCH_MAX_OUTPUT_TOKENS` | Batch output limit | `4,000` | Conservative for batch processing |
| `BATCH_SIZE` | Conversations per batch | `100` | Adjust based on token limits |
| `MAX_ENQUEUED_TOKENS` | Queue limit | `4,500,000` | 90% of OpenAI's 5M limit |

## Core Components Overview

### 1. `staged_batch_analyzer.py`
**Purpose**: Handles OpenAI's 5M token queue limit by intelligently staging batch submissions.

**Key Features**:
- Token estimation and chunking strategy
- Automatic batch sizing within limits
- Progress tracking and resumption capability
- Three-pronged analysis approach

### 2. `run-staged-analysis.py`
**Purpose**: Main entry point for running analysis with month filtering.

**Usage**:
```bash
# Analyze past 12 months (default)
python run-staged-analysis.py

# Analyze past 6 months
python run-staged-analysis.py --months 6

# Resume interrupted processing
python run-staged-analysis.py --resume

# Check current status
python run-staged-analysis.py --status

# Combine existing results only
python run-staged-analysis.py --combine-only
```

### 3. `synthesize_knowledge.py`
**Purpose**: Creates a comprehensive knowledge base from batch analysis results.

**Usage**:
```bash
# Synthesize from default location
python synthesize_knowledge.py

# Specify custom input file
python synthesize_knowledge.py --input staged_results/combined_analysis.json

# Custom output directory
python synthesize_knowledge.py --output my_knowledge_base
```

### 4. Diagnostic Scripts

- **`check_config.py`**: Validates configuration and file paths
- **`scripts/cost_estimator.py`**: Estimates analysis costs before running
- **`scripts/data_parser.py`**: Parses ChatGPT JSON exports
- **`tools/batch_diagnostics.py`**: Diagnoses batch processing issues

## Usage Workflow

### 1. Export ChatGPT Conversations

1. Go to ChatGPT settings → Data controls → Export data
2. Select "JSON" format
3. Download and save the file
4. Update `CHAT_EXPORT_FILE` path in `config/analysis_config.py`

### 2. Run Staged Analysis

**Windows PowerShell:**
```powershell
# Check configuration first
python check_config.py

# Run analysis for past 6 months
python run-staged-analysis.py --months 6

# Monitor progress
python run-staged-analysis.py --status
```

**Unix/Linux/macOS:**
```bash
# Check configuration first
python3 check_config.py

# Run analysis for past 6 months
python3 run-staged-analysis.py --months 6

# Monitor progress
python3 run-staged-analysis.py --status
```

### 3. Monitor Batch Progress

The system automatically tracks progress in `batch_progress.json`:

```json
{
  "batches": {
    "1": {
      "batch_id": "batch-abc123",
      "status": "completed",
      "submitted_at": "2024-01-15T10:30:00",
      "completed_at": "2024-01-15T14:45:00",
      "chunk_ids": ["chunk_0001", "chunk_0002"]
    }
  }
}
```

### 4. Synthesize Results

After all batches complete:

```bash
python synthesize_knowledge.py --input staged_results/combined_analysis.json
```

### 5. Review Outputs

Check the `knowledge_base/` directory for:
- `knowledge_base.json`: Complete structured data
- `knowledge_report.md`: Human-readable report
- Various analysis files with timestamps

## Architecture

### Staged Batch Processing Pipeline

1. **Data Loading**: Parse JSON export and validate format
2. **Date Filtering**: Filter conversations by specified months
3. **Chunking**: Split conversations into manageable chunks (50 conversations each)
4. **Token Estimation**: Calculate tokens for each chunk and analysis type
5. **Batch Creation**: Group chunks into batches under 4.5M token limit
6. **Sequential Submission**: Submit batches one at a time to avoid queue overflow
7. **Progress Tracking**: Monitor completion and handle failures
8. **Result Combination**: Merge all batch results into final analysis

### Three-Pronged Analysis Approach

Each conversation chunk is analyzed from three perspectives:

1. **Project Evolution**: Development patterns, milestones, project progression
2. **Inquiry Patterns**: Learning style, problem-solving approaches, interaction patterns
3. **Knowledge Evolution**: Expertise development, learning breakthroughs, domain growth

### Token Estimation and Chunking Strategy

- **Input Budget**: 80% of model's input limit (217,600 tokens for gpt-4o-mini)
- **System Overhead**: 3,000 tokens reserved for prompts and formatting
- **Chunk Size**: 50 conversations per chunk (adjustable)
- **Batch Limit**: 4.5M tokens per batch (90% of OpenAI's 5M limit)

## File Structure

```
GPTHistoryAnalyzer/
├── config/
│   └── analysis_config.py          # Main configuration file
├── scripts/
│   ├── data_parser.py              # JSON export parser
│   ├── cost_estimator.py           # Cost estimation
│   ├── batch_analyzer.py           # Batch planning logic
│   └── utils.py                    # Utility functions
├── tools/
│   └── batch_diagnostics.py       # Diagnostic tools
├── staged_results/                 # Batch processing results
│   ├── batch_001_requests.jsonl   # Request files
│   ├── batch_001_results.jsonl     # Result files
│   └── combined_analysis.json      # Merged results
├── knowledge_base/                 # Final outputs
│   ├── knowledge_base.json         # Structured data
│   └── knowledge_report.md         # Human-readable report
├── data/
│   ├── raw/                        # Original exports
│   ├── processed/                  # Parsed conversations
│   └── batch/                      # Batch files
├── batch_progress.json             # Progress tracking
├── staged_batch_analyzer.py       # Main batch processor
├── run-staged-analysis.py         # Entry point
├── synthesize_knowledge.py        # Knowledge synthesis
└── requirements.txt                # Python dependencies
```

## Troubleshooting

### Common Errors and Solutions

#### Token Limit Exceeded Errors
**Problem**: "Token limit exceeded" or batch submission failures
**Solutions**:
- Reduce `BATCH_SIZE` in config (try 25-50 conversations)
- Decrease `MAX_ENQUEUED_TOKENS` to 3M tokens
- Check for unusually long conversations in your export

#### Batch Job Failures
**Problem**: Batches fail with errors
**Solutions**:
```bash
# Check batch status
python run-staged-analysis.py --status

# Resume from last successful batch
python run-staged-analysis.py --resume

# Clean up failed batches and restart
rm batch_progress.json
python run-staged-analysis.py
```

#### Empty Analysis Results
**Problem**: Analysis completes but results are empty
**Solutions**:
- Verify JSON export format matches expected structure
- Check that conversations have sufficient content
- Ensure date filtering isn't excluding all conversations

#### Month Filtering Issues
**Problem**: Wrong conversations included/excluded
**Solutions**:
- Check `create_time` format in your JSON export
- Verify timezone handling in date parsing
- Use `--months` parameter to adjust scope

### How to Clean Up Failed Batch Attempts

**Windows PowerShell:**
```powershell
# Remove progress file to start fresh
Remove-Item batch_progress.json -ErrorAction SilentlyContinue

# Clean staged results
Remove-Item staged_results\batch_* -Recurse -Force

# Restart analysis
python run-staged-analysis.py
```

**Unix/Linux/macOS:**
```bash
# Remove progress file to start fresh
rm -f batch_progress.json

# Clean staged results
rm -rf staged_results/batch_*

# Restart analysis
python3 run-staged-analysis.py
```

### How to Resume Interrupted Processing

The system automatically tracks progress and can resume from where it left off:

```bash
# Check current status
python run-staged-analysis.py --status

# Resume processing
python run-staged-analysis.py --resume

# If resumption fails, clean and restart
rm batch_progress.json
python run-staged-analysis.py
```

## Cost Optimization

### Batch vs Real-time Processing

- **Batch Processing**: ~50% cost savings, 24-hour turnaround
- **Real-time Processing**: Immediate results, higher cost
- **Automatic Selection**: System chooses based on dataset size

### Cost Estimation

Before running analysis, estimate costs:
 
```bash
python scripts/cost_estimator.py
```

This provides detailed breakdown of:
- Input/output token usage
- Batch vs real-time pricing
- Expected savings
- Processing recommendations

## Advanced Configuration

### Custom Analysis Prompts

Modify prompts in `staged_batch_analyzer.py`:

```python
project_prompt = """Your custom project analysis prompt here..."""
inquiry_prompt = """Your custom inquiry pattern prompt here..."""
knowledge_prompt = """Your custom knowledge evolution prompt here..."""
```

### Model Selection

Supported models in `config/analysis_config.py`:
- `gpt-4o-mini`: Recommended for cost efficiency
- `gpt-4o`: Higher quality, higher cost
- `gpt-4-turbo`: Alternative option

### Token Limits by Model

| Model | Input Limit | Output Limit | Context Window |
|-------|-------------|--------------|----------------|
| gpt-4o-mini | 272,000 | 128,000 | 400,000 |
| gpt-4o | 128,000 | 4,096 | 128,000 |
| gpt-4-turbo | 128,000 | 4,096 | 128,000 |

## Support and Contributing

### Getting Help

1. Check the troubleshooting section above
2. Review configuration in `config/analysis_config.py`
3. Run diagnostic tools: `python check_config.py`
4. Check batch status: `python run-staged-analysis.py --status`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with your own data
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This tool processes your personal ChatGPT conversation data. Ensure you have appropriate permissions and consider privacy implications before sharing or storing analysis results.