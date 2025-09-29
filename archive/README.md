# Archive - Legacy Diagnostic Scripts

This folder contains legacy diagnostic scripts that have been superseded by the unified `tools/batch_diagnostics.py` CLI.

## Archived Scripts

All scripts in this folder are **compatibility wrappers** that delegate to the unified diagnostic system:

- `batch_failure_diagnostics.py` → `python tools/batch_diagnostics.py errors`
- `batch_status_fix+.py` → `python tools/batch_diagnostics.py status --from-progress`
- `check_config.py` → `python tools/batch_diagnostics.py config`
- `check_specific_batch.py` → `python tools/batch_diagnostics.py status --batch <id>`
- `debug_batch_errors.py` → `python tools/batch_diagnostics.py errors`
- `debug_request_content.py` → `python tools/batch_diagnostics.py requests`
- `diagnose_batch_results.py` → `python tools/batch_diagnostics.py status --from-progress`

## Recommended Usage

Instead of using these individual scripts, use the unified CLI:

```bash
# Check batch status and progress
python tools/batch_diagnostics.py status

# Analyze batch errors
python tools/batch_diagnostics.py errors batch_123

# Inspect configuration
python tools/batch_diagnostics.py config

# Preview staged requests
python tools/batch_diagnostics.py requests

# Get help for any command
python tools/batch_diagnostics.py --help
python tools/batch_diagnostics.py status --help
```

## Backward Compatibility

These archived scripts will continue to work as they delegate to the unified system. You can still run them from the archive folder:

```bash
# These still work from the archive folder:
python archive/check_config.py
python archive/debug_batch_errors.py batch_123
```

However, for new usage, please use the unified CLI directly.

## Migration Date

Scripts archived on: September 28, 2025