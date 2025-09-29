#!/usr/bin/env python3
"""Unified command line diagnostics for the staged batch pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config.analysis_config import AnalysisConfig  # noqa: E402  (import after path tweaks)

try:  # pragma: no cover - OpenAI library may not be installed in every environment
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

DEFAULT_RESULTS_DIR = REPO_ROOT / "staged_results"
PROGRESS_FILE = REPO_ROOT / "batch_progress.json"


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def _format_timestamp(value: Optional[int]) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromtimestamp(value).isoformat(sep=" ", timespec="seconds")
    except Exception:
        return str(value)


def _safe_json_loads(value: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _normalise_stream_text(stream: Any) -> str:
    if hasattr(stream, "text"):
        return stream.text  # type: ignore[return-value]
    data = stream.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _ensure_openai_client(required: bool = True) -> Optional["OpenAI"]:
    if OpenAI is None:
        if required:
            raise RuntimeError("The openai package is not installed")
        return None

    api_key = getattr(AnalysisConfig, "OPENAI_API_KEY", None)
    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        if required:
            raise RuntimeError("OPENAI_API_KEY is missing from AnalysisConfig")
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:
        if required:
            raise RuntimeError(f"Failed to initialise OpenAI client: {exc}") from exc
        print(f"⚠️  Unable to initialise OpenAI client: {exc}")
        return None


def _read_remote_jsonl(client: "OpenAI", file_id: str) -> List[Dict[str, Any]]:
    response = client.files.content(file_id)
    text = _normalise_stream_text(response)
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _safe_json_loads(line)
        if parsed:
            records.append(parsed)
    return records


@dataclass
class BatchSummary:
    batch_id: str
    status: str
    created_at: Optional[int]
    completed_at: Optional[int]
    failed_at: Optional[int]
    request_counts: Dict[str, int]
    output_file_id: Optional[str]
    error_file_id: Optional[str]
    errors_present: bool


def _summarise_batch(batch: Any) -> BatchSummary:
    request_counts = {
        "total": 0,
        "completed": 0,
        "failed": 0,
    }
    counts = getattr(batch, "request_counts", None)
    for key in request_counts:
        if counts is not None and hasattr(counts, key):
            request_counts[key] = getattr(counts, key) or 0

    return BatchSummary(
        batch_id=getattr(batch, "id", getattr(batch, "batch_id", "")),
        status=getattr(batch, "status", "unknown"),
        created_at=getattr(batch, "created_at", None),
        completed_at=getattr(batch, "completed_at", None),
        failed_at=getattr(batch, "failed_at", None),
        request_counts=request_counts,
        output_file_id=getattr(batch, "output_file_id", None),
        error_file_id=getattr(batch, "error_file_id", None),
        errors_present=bool(getattr(batch, "errors", None)),
    )


def _display_batch_summary(summary: BatchSummary, include_timestamps: bool = True) -> None:
    print(f"Status: {summary.status}")
    counts = summary.request_counts
    if counts["total"]:
        print(
            "Requests: total={total} completed={completed} failed={failed}".format(
                **counts
            )
        )
    if include_timestamps:
        print(f"Created: {_format_timestamp(summary.created_at)}")
        if summary.completed_at:
            print(f"Completed: {_format_timestamp(summary.completed_at)}")
        if summary.failed_at:
            print(f"Failed: {_format_timestamp(summary.failed_at)}")
    if summary.output_file_id:
        print(f"Output file: {summary.output_file_id}")
    if summary.error_file_id:
        print(f"Error file: {summary.error_file_id}")
    if summary.errors_present and not summary.error_file_id:
        print("Batch has inline errors in response payload")


def _inspect_progress_file() -> None:
    if not PROGRESS_FILE.exists():
        print("❌ No batch_progress.json file found")
        return

    with PROGRESS_FILE.open("r", encoding="utf-8") as fh:
        progress = json.load(fh)

    started = progress.get("started_at", "Unknown")
    print(f"Started: {started}")

    batches = progress.get("batches", {}) or {}
    print(f"Tracked batches: {len(batches)}")
    for key, info in batches.items():
        batch_id = info.get("batch_id", "Unknown")
        status = info.get("status", "Unknown")
        print(f"  • Batch {key}: {status} ({batch_id})")


def _inspect_results_directory(limit: int = 3) -> None:
    if not DEFAULT_RESULTS_DIR.exists():
        print("❌ No staged_results directory found")
        return

    request_files = sorted(DEFAULT_RESULTS_DIR.glob("batch_*_requests.jsonl"))
    result_files = sorted(DEFAULT_RESULTS_DIR.glob("batch_*_results.jsonl"))

    print(f"Request files: {len(request_files)}")
    print(f"Result files: {len(result_files)}")

    total_results = 0
    for result_file in result_files[:limit]:
        with result_file.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        total_results += len(lines)
        preview = _safe_json_loads(lines[0]) if lines else None
        print(f"  {result_file.name}: {len(lines)} records")
        if preview and "response" in preview:
            body = preview["response"].get("body", {})
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                print(f"    First response length: {len(content)} characters")
    if len(result_files) > limit:
        print(f"  ... {len(result_files) - limit} additional result files not shown")
    print(f"Total results found (first {limit} files counted): {total_results}")


def _inspect_request_file(path: Path, preview_messages: int, preview_chars: int) -> None:
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    print(f"\n{'=' * 60}")
    print(f"File: {path.name}")
    print("=" * 60)
    print(f"Requests in file: {len(lines)}")

    if not lines:
        return

    first = _safe_json_loads(lines[0])
    if not first:
        print("First line is not valid JSON")
        return

    print("First request overview:")
    print(f"  Custom ID: {first.get('custom_id', 'missing')}")
    print(f"  Method: {first.get('method', 'missing')}")
    print(f"  URL: {first.get('url', 'missing')}")

    body = first.get("body", {})
    if isinstance(body, dict):
        print("  Body fields:")
        for field in ("model", "max_tokens", "temperature"):
            if field in body:
                print(f"    {field}: {body[field]}")
        messages = body.get("messages", [])
        print(f"    Messages: {len(messages)}")
        for index, message in enumerate(messages[:preview_messages], start=1):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            preview = content[:preview_chars]
            suffix = "…" if len(content) > preview_chars else ""
            print(f"      Message {index} ({role}): {len(content)} chars")
            print(f"        {preview}{suffix}")

    if len(lines) > 1:
        sample_indices = [1, min(2, len(lines) - 1), len(lines) - 1]
        seen = set()
        print("\nAdditional request samples:")
        for idx in sample_indices:
            if idx >= len(lines) or idx in seen:
                continue
            seen.add(idx)
            parsed = _safe_json_loads(lines[idx])
            if not parsed:
                continue
            print(f"  Request {idx + 1}: {parsed.get('custom_id', 'unknown')}")


def _inspect_config(show_prompts: bool) -> None:
    _print_header("AnalysisConfig attributes")

    attributes = [attr for attr in dir(AnalysisConfig) if not attr.startswith("_")]
    categories = {
        "Prompts": [],
        "Token settings": [],
        "Files and paths": [],
        "Keys": [],
        "Other": [],
    }

    for attr in attributes:
        value = getattr(AnalysisConfig, attr)
        if callable(value):
            continue
        target = "Other"
        lower = attr.lower()
        if "prompt" in lower:
            target = "Prompts"
        elif "token" in lower:
            target = "Token settings"
        elif any(part in lower for part in ("file", "path", "dir")):
            target = "Files and paths"
        elif any(part in lower for part in ("key", "api")):
            target = "Keys"

        display = value
        if isinstance(value, str) and not show_prompts and target == "Prompts":
            preview = value.strip().splitlines()[0] if value else ""
            if len(preview) > 60:
                preview = preview[:60] + "…"
            display = preview
        elif target == "Keys" and isinstance(value, str) and len(value) > 12:
            display = value[:10] + "…"
        categories[target].append((attr, display, type(value).__name__))

    for label, items in categories.items():
        if not items:
            continue
        _print_header(label)
        for name, display, typename in sorted(items):
            print(f"{name} ({typename}): {display}")

    required = {
        "OPENAI_API_KEY": hasattr(AnalysisConfig, "OPENAI_API_KEY")
        and bool(getattr(AnalysisConfig, "OPENAI_API_KEY", "")),
        "CHAT_EXPORT_FILE": hasattr(AnalysisConfig, "CHAT_EXPORT_FILE"),
        "Prompts configured": any("prompt" in attr.lower() for attr in attributes),
        "Token budget helpers": hasattr(AnalysisConfig, "MAX_INPUT_TOKENS"),
    }

    _print_header("Required attribute check")
    for label, present in required.items():
        symbol = "✅" if present else "❌"
        print(f"{symbol} {label}")


def run_status(args: argparse.Namespace) -> int:
    _print_header("Local progress overview")
    _inspect_progress_file()

    _print_header("Staged results directory")
    _inspect_results_directory(limit=args.preview_files)

    batch_ids: List[str] = []
    if not args.skip_remote:
        if args.batch:
            batch_ids.extend(args.batch)
        if PROGRESS_FILE.exists() and args.from_progress:
            with PROGRESS_FILE.open("r", encoding="utf-8") as fh:
                progress = json.load(fh)
            for info in (progress.get("batches") or {}).values():
                batch_id = info.get("batch_id")
                if batch_id:
                    batch_ids.append(batch_id)

        if batch_ids:
            try:
                client = _ensure_openai_client(required=True)
            except RuntimeError as exc:
                print(f"❌ {exc}")
                return 1

            seen = set()
            _print_header("Remote batch status")
            for batch_id in batch_ids:
                if batch_id in seen:
                    continue
                seen.add(batch_id)
                print(f"Batch {batch_id}")
                print("-" * (6 + len(batch_id)))
                try:
                    batch = client.batches.retrieve(batch_id)
                except Exception as exc:
                    print(f"  ⚠️  Could not retrieve batch: {exc}")
                    continue
                summary = _summarise_batch(batch)
                _display_batch_summary(summary)
                print()
        else:
            print("No batch IDs supplied for remote lookups")
    else:
        print("Skipping remote batch lookups (--skip-remote)")

    return 0


def run_errors(args: argparse.Namespace) -> int:
    try:
        client = _ensure_openai_client(required=True)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    limit = args.limit
    for batch_id in args.batch_ids:
        print(f"\nAnalyzing batch {batch_id}")
        print("-" * (18 + len(batch_id)))
        try:
            batch = client.batches.retrieve(batch_id)
        except Exception as exc:
            print(f"  ⚠️  Could not retrieve batch: {exc}")
            continue

        summary = _summarise_batch(batch)
        _display_batch_summary(summary)

        if summary.error_file_id:
            try:
                errors = _read_remote_jsonl(client, summary.error_file_id)
            except Exception as exc:
                print(f"  ⚠️  Could not download error file: {exc}")
                errors = []

            print(f"  Error records: {len(errors)}")
            if errors:
                error_types: Dict[str, int] = {}
                for record in errors:
                    detail = record.get("error") or {}
                    code = detail.get("code") or detail.get("type") or "unknown"
                    error_types[code] = error_types.get(code, 0) + 1
                print("  Error type distribution:")
                for code, count in sorted(error_types.items(), key=lambda item: item[1], reverse=True):
                    print(f"    • {code}: {count}")

                print(f"\n  Sample errors (first {min(limit, len(errors))}):")
                for record in errors[:limit]:
                    detail = record.get("error") or {}
                    custom_id = record.get("custom_id", "unknown")
                    message = detail.get("message", "")
                    print(f"    - {custom_id}: {detail.get('code', detail.get('type', 'unknown'))}")
                    preview = message[:200]
                    suffix = "…" if len(message) > 200 else ""
                    if preview:
                        print(f"        {preview}{suffix}")

                if args.save_errors:
                    output_path = Path(f"batch_errors_{batch_id}.json")
                    output_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
                    print(f"\n  💾 Saved full error payload to {output_path}")
        else:
            print("  No error file reported for this batch")

        if args.download_inputs and getattr(batch, "input_file_id", None):
            try:
                content = client.files.content(batch.input_file_id)
                text = _normalise_stream_text(content)
                path = Path(f"batch_input_{batch_id}.jsonl")
                path.write_text(text, encoding="utf-8")
                print(f"  💾 Saved input JSONL to {path}")
            except Exception as exc:
                print(f"  ⚠️  Could not download input file: {exc}")

    return 0


def run_requests(args: argparse.Namespace) -> int:
    results_dir = args.directory
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1

    request_files = sorted(results_dir.glob("batch_*_requests.jsonl"))
    if not request_files:
        print("No request files found")
        return 0

    for path in request_files[: args.limit]:
        _inspect_request_file(path, args.preview_messages, args.preview_chars)

    remaining = len(request_files) - args.limit
    if remaining > 0:
        print(f"\n… {remaining} additional request files not shown")
    return 0


def run_config(args: argparse.Namespace) -> int:
    _inspect_config(show_prompts=args.show_prompts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostics helper for staged batch analysis runs.",
        prog="batch-diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser(
        "status",
        help="Inspect local progress files and optionally poll batch status from OpenAI.",
    )
    status.add_argument(
        "--batch",
        dest="batch",
        action="append",
        help="Explicit batch ID(s) to fetch from the API. Can be supplied multiple times.",
    )
    status.add_argument(
        "--from-progress",
        action="store_true",
        help="Fetch all batch IDs recorded in batch_progress.json.",
    )
    status.add_argument(
        "--skip-remote",
        action="store_true",
        help="Skip all OpenAI API calls and only show local files.",
    )
    status.add_argument(
        "--preview-files",
        type=int,
        default=3,
        help="How many result files to inspect for local previews (default: 3).",
    )
    status.set_defaults(func=run_status)

    errors = subparsers.add_parser(
        "errors",
        help="Download and summarise batch error payloads from OpenAI.",
    )
    errors.add_argument("batch_ids", nargs="+", help="Batch IDs to inspect")
    errors.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many sample errors to print per batch (default: 5).",
    )
    errors.add_argument(
        "--save-errors",
        action="store_true",
        help="Write the full error payload to batch_errors_<id>.json.",
    )
    errors.add_argument(
        "--download-inputs",
        action="store_true",
        help="Also download the original input JSONL for each batch.",
    )
    errors.set_defaults(func=run_errors)

    requests = subparsers.add_parser(
        "requests",
        help="Preview the staged JSONL requests that were uploaded to OpenAI.",
    )
    requests.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing staged request files (default: staged_results).",
    )
    requests.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many request files to inspect (default: 3).",
    )
    requests.add_argument(
        "--preview-messages",
        type=int,
        default=2,
        help="How many messages to preview per request (default: 2).",
    )
    requests.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Maximum characters to show per message (default: 120).",
    )
    requests.set_defaults(func=run_requests)

    config = subparsers.add_parser(
        "config", help="Inspect AnalysisConfig values and required settings."
    )
    config.add_argument(
        "--show-prompts",
        action="store_true",
        help="Display full prompt text instead of previews when listing config attributes.",
    )
    config.set_defaults(func=run_config)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
