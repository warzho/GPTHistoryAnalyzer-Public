"""Batch analysis orchestration using the OpenAI Batch API."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from openai import OpenAI

from config.analysis_config import AnalysisConfig
from scripts.utils import TokenCounter


SYSTEM_PROMPTS: Dict[str, str] = {
    "project": (
        "You are an expert project analyst. Identify project trajectories, milestone shifts, "
        "risks, and recommended next actions across the provided conversations."
    ),
    "inquiry": (
        "You are an expert learning analyst. Surface inquiry themes, question strategies, "
        "guidance requests, and behavioural patterns."
    ),
    "knowledge": (
        "You are an expert knowledge mapper. Track recurring domains, concept progression, "
        "connections between topics, and emerging expertise."
    ),
}

USER_FOCUS_TEXT: Dict[str, str] = {
    "project": "Focus on scope changes, blockers, dependencies, and forward-looking recommendations.",
    "inquiry": "Focus on question framing, learning objectives, research depth, and support needs.",
    "knowledge": "Focus on knowledge domains, conceptual links, depth changes, and open questions.",
}

ANALYSIS_TYPES: Tuple[Tuple[str, str], ...] = (
    ("project", "project_evolution"),
    ("inquiry", "inquiry_patterns"),
    ("knowledge", "knowledge_evolution"),
)

TEMPERATURE: float = 0.2


@dataclass
class PreparedConversation:
    """Conversation metadata prepared for batching."""

    index: int
    conversation_id: str
    title: str
    text: str
    token_count: int
    truncated: bool


@dataclass
class ConversationChunk:
    """Aggregated conversations grouped for a single batch request."""

    index: int
    conversations: List[PreparedConversation]
    conversation_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class RequestPlan:
    """Audit information for a single batch request line."""

    custom_id: str
    analysis_type: str
    chunk_index: int
    input_tokens: int
    output_tokens: int
    line_number: int


class BatchPlanner:
    """Prepare conversations and chunk them according to model limits."""

    def __init__(
        self,
        token_counter: TokenCounter,
        input_budget: Optional[int] = None,
        system_prompts: Optional[Dict[str, str]] = None,
        user_focus_text: Optional[Dict[str, str]] = None,
    ) -> None:
        self.token_counter = token_counter
        self.input_budget = input_budget or AnalysisConfig.get_input_token_budget()
        self.system_prompts = system_prompts or SYSTEM_PROMPTS
        self.user_focus_text = user_focus_text or USER_FOCUS_TEXT

    # ------------------------------------------------------------------
    # Conversation preparation
    # ------------------------------------------------------------------
    def prepare_conversations(
        self, conversations: List[Dict[str, Any]], subset_size: Optional[int] = None
    ) -> List[PreparedConversation]:
        if subset_size is not None and subset_size < 0:
            raise ValueError("subset_size must be positive when provided")

        limit = subset_size if subset_size is not None else len(conversations)
        prepared: List[PreparedConversation] = []

        for index, conversation in enumerate(conversations[:limit]):
            rendered_text, token_count, truncated = self._render_conversation(conversation, index)
            conversation_id = str(conversation.get("id", f"conversation-{index + 1}"))
            title = conversation.get("title") or f"Conversation {index + 1}"
            prepared.append(
                PreparedConversation(
                    index=index,
                    conversation_id=conversation_id,
                    title=title,
                    text=rendered_text,
                    token_count=token_count,
                    truncated=truncated,
                )
            )

        return prepared

    def _render_conversation(self, conversation: Dict[str, Any], index: int) -> Tuple[str, int, bool]:
        title = (conversation.get("title") or f"Conversation {index + 1}").strip()
        created = conversation.get("create_time") or conversation.get("created_at")

        header_lines = [f"Conversation {index + 1}: {title}"]
        if created:
            header_lines.append(f"Created: {created}")

        header_text = "\n".join(header_lines)
        header_tokens = self.token_counter.count_text(header_text)
        remaining_budget = max(self.input_budget - header_tokens, 0)

        message_entries: List[Tuple[str, int]] = []
        for message in conversation.get("messages", []):
            content = self._extract_message_content(message)
            if not content:
                continue
            role = self._normalise_role(message)
            entry_text = f"{role}: {content}"
            entry_tokens = self.token_counter.count_text(entry_text)
            message_entries.append((entry_text, entry_tokens))

        total_entry_tokens = sum(tokens for _, tokens in message_entries)
        truncated = total_entry_tokens > remaining_budget

        note_text = ""
        note_tokens = 0
        if truncated:
            note_text = "(Trimmed to recent turns to respect token budget.)"
            note_tokens = self.token_counter.count_text(note_text)
            remaining_for_entries = max(remaining_budget - note_tokens, 0)

            trimmed_entries: List[Tuple[str, int]] = []
            running = 0
            for entry_text, entry_tokens in reversed(message_entries):
                if entry_tokens == 0:
                    continue
                if running + entry_tokens > remaining_for_entries:
                    continue
                trimmed_entries.append((entry_text, entry_tokens))
                running += entry_tokens
                if running >= remaining_for_entries:
                    break

            trimmed_entries.reverse()
            message_entries = trimmed_entries
            total_entry_tokens = running

        lines = list(header_lines)
        if truncated and note_text:
            lines.append(note_text)
        if message_entries:
            lines.append("")
            lines.extend(entry_text for entry_text, _ in message_entries)
        else:
            lines.append("")
            lines.append("[No message content retained]")

        rendered_text = "\n".join(lines).strip()
        total_tokens = header_tokens + note_tokens + total_entry_tokens
        return rendered_text, total_tokens, truncated

    @staticmethod
    def _extract_message_content(message: Dict[str, Any]) -> str:
        content = message.get("content")

        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            parts: List[str] = []
            if isinstance(content.get("parts"), list):
                for part in content["parts"]:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        if isinstance(part.get("text"), str):
                            parts.append(part["text"])
                        elif isinstance(part.get("value"), str):
                            parts.append(part["value"])
            if isinstance(content.get("text"), str):
                parts.append(content["text"])
            text = "\n".join(parts)
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("value"), str):
                        parts.append(item["value"])
            text = "\n".join(parts)
        else:
            text = message.get("text", "")

        return (text or "").replace("\r\n", "\n").strip()

    @staticmethod
    def _normalise_role(message: Dict[str, Any]) -> str:
        role = message.get("role")
        if not role and isinstance(message.get("author"), dict):
            role = message["author"].get("role")

        role_str = str(role or "user")
        if role_str.lower() in {"user", "assistant", "system"}:
            return role_str.capitalize()
        return role_str.replace("_", " ").title()

    # ------------------------------------------------------------------
    # Chunk planning
    # ------------------------------------------------------------------
    def chunk_conversations(self, conversations: List[PreparedConversation]) -> List[ConversationChunk]:
        chunks: List[ConversationChunk] = []
        current: List[PreparedConversation] = []
        current_tokens = 0

        for conversation in conversations:
            conv_tokens = min(conversation.token_count, self.input_budget)

            if current and current_tokens + conv_tokens > self.input_budget:
                chunk_index = len(chunks)
                chunks.append(
                    ConversationChunk(
                        index=chunk_index,
                        conversations=current,
                        conversation_tokens=current_tokens,
                    )
                )
                current = []
                current_tokens = 0

            current.append(conversation)
            current_tokens += conv_tokens

        if current:
            chunk_index = len(chunks)
            chunks.append(
                ConversationChunk(
                    index=chunk_index,
                    conversations=current,
                    conversation_tokens=current_tokens,
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------
    def build_chunk_messages(self, chunk: ConversationChunk, analysis_key: str) -> List[Dict[str, str]]:
        if analysis_key not in self.system_prompts:
            raise KeyError(f"Unknown analysis key: {analysis_key}")

        system_prompt = self.system_prompts[analysis_key].strip()
        user_prompt = self._build_chunk_user_prompt(chunk, analysis_key)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_chunk_user_prompt(self, chunk: ConversationChunk, analysis_key: str) -> str:
        focus_text = self.user_focus_text[analysis_key].strip()
        lines = [
            f"You will review {len(chunk.conversations)} conversations from a chat archive.",
            focus_text,
            "",
        ]

        for conversation in chunk.conversations:
            lines.append(conversation.text)
            lines.append("")

        lines.append(
            "Synthesize patterns, reference conversation numbers, and provide actionable insights."
        )
        return "\n".join(lines).strip()


#Version with proper retrieve_and_process_results
class BatchAnalyzer:
    """Handles batch processing of chat history using OpenAI's Batch API."""

    def __init__(self) -> None:
        self.model = AnalysisConfig.ANALYSIS_MODEL
        self.token_counter = TokenCounter(self.model)
        self.input_budget = AnalysisConfig.get_input_token_budget()
        self.planner = BatchPlanner(self.token_counter, self.input_budget)

        AnalysisConfig.DATA_BATCH_DIR.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(api_key=AnalysisConfig.OPENAI_API_KEY)

        # NEW: where we persist known job ids for status checks
        self._tracked_jobs_file: Path = AnalysisConfig.DATA_BATCH_DIR / "tracked_jobs.json"

        self.last_request_plans: List[RequestPlan] = []
        self.last_request_files: List[Path] = []
        self.last_request_plan_map: Dict[Path, List[RequestPlan]] = {}

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------
    def create_comprehensive_analysis_batch(
        self,
        conversations: List[Dict[str, Any]],
        subset_size: Optional[int] = None,
        dry_run: bool = False,
        preview_lines: int = 3,
    ) -> List[str]:
        if not conversations:
            raise ValueError("No conversations supplied for batch analysis")

        prepared = self.planner.prepare_conversations(conversations, subset_size=subset_size)
        if not prepared:
            raise ValueError("No conversations contained usable message content")

        total_tokens = sum(conv.token_count for conv in prepared)
        print(
            f"Prepared {len(prepared)} conversations containing {total_tokens:,} transcript tokens."
        )
        print(
            f"Planning chunks with input budget {self.input_budget:,} tokens and model {self.model}."
        )

        chunks = self.planner.chunk_conversations(prepared)
        if not chunks:
            raise RuntimeError("Chunk planning produced no requests. Check conversation data.")

        self._log_chunk_audit(chunks)

        requests, plans = self._build_requests(chunks)
        request_files = self._write_request_files(requests, plans)
        self.last_request_plans = plans
        self.last_request_files = request_files

        self._preview_files(request_files, preview_lines)
        self._print_token_audit(plans)

        print(
            f"Generated {len(plans)} requests across {len(chunks)} chunks."
        )

        if not request_files:
            raise RuntimeError("No batch request files were created. Check request planning logic.")

        if len(request_files) == 1:
            print(f"JSONL ready at {request_files[0]}.")
        else:
            print(f"Created {len(request_files)} JSONL files to satisfy batch upload limits:")
            for index, path in enumerate(request_files, start=1):
                plan_count = len(self.last_request_plan_map.get(path, []))
                print(f"  [{index:02d}] {path.name} — {plan_count} requests")

        if dry_run:
            print("Dry-run mode enabled: no batch submission will be attempted.")

        return [str(path) for path in request_files]

    def submit_batch_job(
        self,
        request_file_paths: Union[str, Path, Sequence[Union[str, Path]]],
        dry_run: bool = False,
    ) -> Optional[List[str]]:
        if isinstance(request_file_paths, (str, Path)):
            path_list = [Path(request_file_paths)]
        else:
            path_list = [Path(item) for item in request_file_paths]

        if not path_list:
            raise ValueError("No request files supplied for submission")

        for path in path_list:
            if not path.exists():
                raise FileNotFoundError(f"Batch request file not found: {path}")

        if dry_run:
            print("Dry-run mode: skipping batch submission step.")
            return None

        job_ids: List[str] = []
        for path in path_list:
            with path.open("rb") as handle:
                upload = self.client.files.create(file=handle, purpose="batch")

            job = self.client.batches.create(
                input_file_id=upload.id,
                endpoint="/v1/chat/completions",
                completion_window=AnalysisConfig.BATCH_COMPLETION_WINDOW,
            )
            print(f"📬 Submitted batch job {job.id} for {path.name}")
            job_ids.append(job.id)

        # NEW: persist for later status checks
        self._track_jobs(job_ids)

        return job_ids

    # ------------------------------------------------------------------
    # NEW public methods used by check_batch_status.py
    # ------------------------------------------------------------------
    def check_batch_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Return status information for a single job (if job_id provided) or
        all tracked jobs (as {'tracked_jobs': [...]}) with fields that
        check_batch_status.py expects:
            - batch_id
            - status ('validating' | 'in_progress' | 'completed' | 'failed' | ...)
            - request_counts: {'total', 'completed', 'failed'} (when available)
            - completed_at (epoch seconds, when available)
        """
        import json

        def _fmt(job) -> Dict[str, Any]:
            info: Dict[str, Any] = {
                "batch_id": getattr(job, "id", None) or job.get("id"),
                "status": getattr(job, "status", None) or job.get("status", "unknown"),
            }
            # request_counts and completed_at vary by SDK/API version; be defensive.
            rc = getattr(job, "request_counts", None) or job.get("request_counts")
            if rc:
                info["request_counts"] = {
                    "total": rc.get("total", 0),
                    "completed": rc.get("completed", 0),
                    "failed": rc.get("failed", 0),
                }
            if getattr(job, "completed_at", None) or job.get("completed_at"):
                info["completed_at"] = getattr(job, "completed_at", None) or job.get("completed_at")
            # Include an error message if present
            if getattr(job, "error", None) or job.get("error"):
                info["error"] = getattr(job, "error", None) or job.get("error")
            return info

        if job_id:
            job = self.client.batches.retrieve(job_id)
            # Some SDKs return a pydantic-ish object; coerce via __dict__ if needed
            job_dict = getattr(job, "model_dump", None)
            job_data = job_dict() if callable(job_dict) else getattr(job, "__dict__", job)
            return _fmt(job_data)

        # No job id: iterate tracked ids
        ids = self._load_tracked_job_ids()
        tracked: List[Dict[str, Any]] = []
        for jid in ids:
            try:
                job = self.client.batches.retrieve(jid)
                job_dict = getattr(job, "model_dump", None)
                job_data = job_dict() if callable(job_dict) else getattr(job, "__dict__", job)
                tracked.append(_fmt(job_data))
            except Exception as e:  # show something useful but keep going
                tracked.append({"batch_id": jid, "status": "unknown", "error": str(e)})
        return {"tracked_jobs": tracked}

    def retrieve_and_process_results(self, job_id: str) -> Dict[str, Any]:
        """
        For a completed batch job:
          - download the output JSONL
          - parse each line, grouping by our custom_id prefixes:
                project_evolution / inquiry_patterns / knowledge_evolution
          - return a dict: {'project_evolution': [...], 'inquiry_patterns': [...],
                            'knowledge_evolution': [...], 'errors': [...]}
        If the job isn't complete, returns {'error': '...'}.
        """
        import json
        from io import BytesIO

        job = self.client.batches.retrieve(job_id)
        job_dict = getattr(job, "model_dump", None)
        job_data = job_dict() if callable(job_dict) else getattr(job, "__dict__", job)

        status = job_data.get("status")
        if status != "completed":
            return {"error": f"Batch job not completed (status: {status})."}

        output_file_id = job_data.get("output_file_id") or job_data.get("result_file_id")
        if not output_file_id:
            return {"error": "No output file id found on completed batch job."}

        # Get raw JSONL content from the file
        try:
            file_content_obj = self.client.files.content(output_file_id)
            # SDKs differ: some expose .text, others are a stream/bytes
            if hasattr(file_content_obj, "text"):
                raw = file_content_obj.text
            else:
                # Try to read() from a stream-like object
                raw = getattr(file_content_obj, "read", lambda: file_content_obj)()
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="replace")
        except Exception as e:
            return {"error": f"Unable to download batch output file: {e}"}

        results = {
            "project_evolution": [],
            "inquiry_patterns": [],
            "knowledge_evolution": [],
            "errors": [],
        }

        # Helper to assign to a bucket based on our custom_id pattern
        def bucket_for(custom_id: str) -> Optional[str]:
            if not isinstance(custom_id, str):
                return None
            if custom_id.startswith("project_evolution-"):
                return "project_evolution"
            if custom_id.startswith("inquiry_patterns-"):
                return "inquiry_patterns"
            if custom_id.startswith("knowledge_evolution-"):
                return "knowledge_evolution"
            return None

        # Parse each JSONL line defensively
        for i, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                results["errors"].append({"line": i, "error": f"JSON decode error: {exc}"})
                continue

            custom_id = obj.get("custom_id")
            bucket = bucket_for(custom_id)

            # The response body for chat completions is usually under obj['response']['body']
            body = None
            resp = obj.get("response") or {}
            if isinstance(resp, dict):
                body = resp.get("body") or resp  # fall back if structure differs

            # Pull out the assistant message content if present
            message_content = None
            usage = None
            if isinstance(body, dict):
                if "choices" in body and body["choices"]:
                    choice0 = body["choices"][0]
                    msg = choice0.get("message") or {}
                    message_content = msg.get("content")
                usage = body.get("usage")

            record = {
                "custom_id": custom_id,
                "chunk_index": self._chunk_index_from_custom_id(custom_id),
                "content": message_content,
                "usage": usage,
                "status_code": resp.get("status_code"),
                "request_id": resp.get("request_id"),
                "line_index": i,
            }

            if bucket:
                results[bucket].append(record)
            else:
                results["errors"].append(
                    {"line": i, "error": "Unrecognized or missing custom_id", "raw": obj}
                )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_chunk_audit(self, chunks: List[ConversationChunk]) -> None:
        print(
            f"\n📊 Chunk planning audit (input budget {self.input_budget:,} tokens per request)"
        )
        cumulative_input = 0
        for chunk in chunks:
            messages = self.planner.build_chunk_messages(chunk, ANALYSIS_TYPES[0][0])
            chunk_input = self.token_counter.count_chat_tokens(messages)
            chunk_output = AnalysisConfig.plan_output_tokens(chunk_input)
            chunk.input_tokens = chunk_input
            chunk.output_tokens = chunk_output
            cumulative_input += chunk_input
            print(
                f"  Chunk {chunk.index + 1:02d}: conv={len(chunk.conversations):>3} | "
                f"transcript_tokens={chunk.conversation_tokens:>7,} | input_tokens={chunk_input:>7,} | "
                f"max_tokens={chunk_output:>6,} | cumulative_input={cumulative_input:>7,}"
            )
        print("")

    # ... (keep all your existing helpers: _build_requests, _validate_request_dict,
    # _write_request_files, _preview_files, _print_token_audit, etc.)

    # NEW: tiny utilities for tracking and parsing
    def _track_jobs(self, job_ids: Sequence[str]) -> None:
        existing = set(self._load_tracked_job_ids())
        existing.update(job_ids)
        data = {"job_ids": sorted(existing)}
        self._tracked_jobs_file.parent.mkdir(parents=True, exist_ok=True)
        self._tracked_jobs_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_tracked_job_ids(self) -> List[str]:
        try:
            import json
            raw = self._tracked_jobs_file.read_text(encoding="utf-8")
            return list((json.loads(raw) or {}).get("job_ids", []))
        except FileNotFoundError:
            return []
        except Exception:
            return []

    @staticmethod
    def _chunk_index_from_custom_id(custom_id: Optional[str]) -> Optional[int]:
        # Expect formats like "project_evolution-chunk-0001"
        if not custom_id or "-chunk-" not in custom_id:
            return None
        try:
            suffix = custom_id.split("-chunk-")[-1]
            return int(suffix) - 1  # make it 0-based like internal indices
        except Exception:
            return None


# class BatchAnalyzer:
#     """Handles batch processing of chat history using OpenAI's Batch API."""
#
#     def __init__(self) -> None:
#         self.model = AnalysisConfig.ANALYSIS_MODEL
#         self.token_counter = TokenCounter(self.model)
#         self.input_budget = AnalysisConfig.get_input_token_budget()
#         self.planner = BatchPlanner(self.token_counter, self.input_budget)
#
#         AnalysisConfig.DATA_BATCH_DIR.mkdir(parents=True, exist_ok=True)
#         self.client = OpenAI(api_key=AnalysisConfig.OPENAI_API_KEY)
#
#         self.last_request_plans: List[RequestPlan] = []
#         self.last_request_files: List[Path] = []
#         self.last_request_plan_map: Dict[Path, List[RequestPlan]] = {}
#
#     # ------------------------------------------------------------------
#     # Public entrypoints
#     # ------------------------------------------------------------------
#     def create_comprehensive_analysis_batch(
#         self,
#         conversations: List[Dict[str, Any]],
#         subset_size: Optional[int] = None,
#         dry_run: bool = False,
#         preview_lines: int = 3,
#     ) -> List[str]:
#         if not conversations:
#             raise ValueError("No conversations supplied for batch analysis")
#
#         prepared = self.planner.prepare_conversations(conversations, subset_size=subset_size)
#         if not prepared:
#             raise ValueError("No conversations contained usable message content")
#
#         total_tokens = sum(conv.token_count for conv in prepared)
#         print(
#             f"Prepared {len(prepared)} conversations containing {total_tokens:,} transcript tokens."
#         )
#         print(
#             f"Planning chunks with input budget {self.input_budget:,} tokens and model {self.model}."
#         )
#
#         chunks = self.planner.chunk_conversations(prepared)
#         if not chunks:
#             raise RuntimeError("Chunk planning produced no requests. Check conversation data.")
#
#         self._log_chunk_audit(chunks)
#
#         requests, plans = self._build_requests(chunks)
#         request_files = self._write_request_files(requests, plans)
#         self.last_request_plans = plans
#         self.last_request_files = request_files
#
#         self._preview_files(request_files, preview_lines)
#         self._print_token_audit(plans)
#
#         print(
#             f"Generated {len(plans)} requests across {len(chunks)} chunks."
#         )
#
#         if not request_files:
#             raise RuntimeError("No batch request files were created. Check request planning logic.")
#
#         if len(request_files) == 1:
#             print(f"JSONL ready at {request_files[0]}.")
#         else:
#             print(f"Created {len(request_files)} JSONL files to satisfy batch upload limits:")
#             for index, path in enumerate(request_files, start=1):
#                 plan_count = len(self.last_request_plan_map.get(path, []))
#                 print(f"  [{index:02d}] {path.name} — {plan_count} requests")
#
#         if dry_run:
#             print("Dry-run mode enabled: no batch submission will be attempted.")
#
#         return [str(path) for path in request_files]
#
#     def submit_batch_job(
#         self,
#         request_file_paths: Union[str, Path, Sequence[Union[str, Path]]],
#         dry_run: bool = False,
#     ) -> Optional[List[str]]:
#         if isinstance(request_file_paths, (str, Path)):
#             path_list = [Path(request_file_paths)]
#         else:
#             path_list = [Path(item) for item in request_file_paths]
#
#         if not path_list:
#             raise ValueError("No request files supplied for submission")
#
#         for path in path_list:
#             if not path.exists():
#                 raise FileNotFoundError(f"Batch request file not found: {path}")
#
#         if dry_run:
#             print("Dry-run mode: skipping batch submission step.")
#             return None
#
#         job_ids: List[str] = []
#         for path in path_list:
#             with path.open("rb") as handle:
#                 upload = self.client.files.create(file=handle, purpose="batch")
#
#             job = self.client.batches.create(
#                 input_file_id=upload.id,
#                 endpoint="/v1/chat/completions",
#                 completion_window=AnalysisConfig.BATCH_COMPLETION_WINDOW,
#             )
#             print(f"📬 Submitted batch job {job.id} for {path.name}")
#             job_ids.append(job.id)
#
#         return job_ids
#
#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------
#     def _log_chunk_audit(self, chunks: List[ConversationChunk]) -> None:
#         print(
#             f"\n📊 Chunk planning audit (input budget {self.input_budget:,} tokens per request)"
#         )
#         cumulative_input = 0
#         for chunk in chunks:
#             messages = self.planner.build_chunk_messages(chunk, ANALYSIS_TYPES[0][0])
#             chunk_input = self.token_counter.count_chat_tokens(messages)
#             chunk_output = AnalysisConfig.plan_output_tokens(chunk_input)
#             chunk.input_tokens = chunk_input
#             chunk.output_tokens = chunk_output
#             cumulative_input += chunk_input
#             print(
#                 f"  Chunk {chunk.index + 1:02d}: conv={len(chunk.conversations):>3} | "
#                 f"transcript_tokens={chunk.conversation_tokens:>7,} | input_tokens={chunk_input:>7,} | "
#                 f"max_tokens={chunk_output:>6,} | cumulative_input={cumulative_input:>7,}"
#             )
#         print("")
#
#     def _build_requests(self, chunks: List[ConversationChunk]) -> Tuple[List[Dict[str, Any]], List[RequestPlan]]:
#         requests: List[Dict[str, Any]] = []
#         plans: List[RequestPlan] = []
#         line_number = 1
#
#         for chunk in chunks:
#             for analysis_key, custom_prefix in ANALYSIS_TYPES:
#                 messages = self.planner.build_chunk_messages(chunk, analysis_key)
#                 input_tokens = self.token_counter.count_chat_tokens(messages)
#                 output_tokens = AnalysisConfig.plan_output_tokens(input_tokens)
#
#                 custom_id = f"{custom_prefix}-chunk-{chunk.index + 1:04d}"
#                 request_dict = {
#                     "custom_id": custom_id,
#                     "method": "POST",
#                     "url": "/v1/chat/completions",
#                     "body": {
#                         "model": self.model,
#                         "messages": messages,
#                         "temperature": float(TEMPERATURE),
#                         "max_tokens": int(output_tokens),
#                     },
#                 }
#
#                 self._validate_request_dict(request_dict)
#                 requests.append(request_dict)
#                 plans.append(
#                     RequestPlan(
#                         custom_id=custom_id,
#                         analysis_type=analysis_key,
#                         chunk_index=chunk.index,
#                         input_tokens=input_tokens,
#                         output_tokens=output_tokens,
#                         line_number=line_number,
#                     )
#                 )
#                 line_number += 1
#
#         return requests, plans
#
#     def _validate_request_dict(self, request: Dict[str, Any]) -> None:
#         required_keys = {"custom_id", "method", "url", "body"}
#         if set(request.keys()) != required_keys:
#             raise ValueError("Request must contain only custom_id, method, url, and body")
#
#         if request["method"] != "POST":
#             raise ValueError("Request method must be POST")
#         if request["url"] != "/v1/chat/completions":
#             raise ValueError("Request URL must be /v1/chat/completions")
#
#         body = request["body"]
#         if set(body.keys()) != {"model", "messages", "temperature", "max_tokens"}:
#             raise ValueError("Request body must contain model, messages, temperature, max_tokens")
#
#         if body["model"] != self.model:
#             raise ValueError(f"Request body model must be {self.model}")
#         if not isinstance(body["messages"], list) or not body["messages"]:
#             raise ValueError("Request body messages must be a non-empty list")
#
#         for message in body["messages"]:
#             if set(message.keys()) != {"role", "content"}:
#                 raise ValueError("Each message must contain exactly role and content")
#             if message["role"] not in {"system", "user", "assistant"}:
#                 raise ValueError("Message role must be system, user, or assistant")
#             if not isinstance(message["content"], str):
#                 raise ValueError("Message content must be a string")
#
#         if not isinstance(body["temperature"], (float, int)):
#             raise ValueError("Temperature must be a float")
#         if not isinstance(body["max_tokens"], int) or body["max_tokens"] <= 0:
#             raise ValueError("max_tokens must be a positive integer")
#
#     def _write_request_files(
#         self, requests: List[Dict[str, Any]], plans: List[RequestPlan]
#     ) -> List[Path]:
#         if len(requests) != len(plans):
#             raise ValueError("Request and plan counts must match before serialisation")
#
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         base_name = f"analysis_requests_{timestamp}"
#         max_bytes = AnalysisConfig.MAX_BATCH_FILE_BYTES
#         max_requests = AnalysisConfig.MAX_REQUESTS_PER_JOB
#
#         segments: List[Tuple[List[str], List[RequestPlan], int]] = []
#         current_lines: List[str] = []
#         current_plans: List[RequestPlan] = []
#         current_bytes = 0
#
#         for request, plan in zip(requests, plans):
#             serialized = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
#             self._validate_serialised_line(serialized)
#             encoded = serialized.encode("utf-8")
#             line_bytes = len(encoded) + 1  # account for newline terminator
#
#             if line_bytes > max_bytes:
#                 raise ValueError(
#                     "Single request line exceeds configured batch file size limit. "
#                     "Reduce input budget or conversation size and retry."
#                 )
#
#             if current_lines and (
#                 current_bytes + line_bytes > max_bytes or len(current_lines) >= max_requests
#             ):
#                 segments.append((current_lines, current_plans, current_bytes))
#                 current_lines = []
#                 current_plans = []
#                 current_bytes = 0
#
#             current_lines.append(serialized)
#             current_plans.append(plan)
#             current_bytes += line_bytes
#
#         if current_lines:
#             segments.append((current_lines, current_plans, current_bytes))
#
#         if not segments:
#             return []
#
#         multiple_files = len(segments) > 1
#         output_paths: List[Path] = []
#         self.last_request_plan_map = {}
#
#         for index, (lines, segment_plans, byte_count) in enumerate(segments, start=1):
#             suffix = f"_part{index:02d}" if multiple_files else ""
#             filename = f"{base_name}{suffix}.jsonl"
#             output_path = AnalysisConfig.DATA_BATCH_DIR / filename
#
#             with output_path.open("w", encoding="utf-8", newline="\n") as handle:
#                 for line in lines:
#                     handle.write(line + "\n")
#
#             size_mb = byte_count / (1024 * 1024)
#             print(
#                 f"📄 Wrote {len(lines)} request lines ({size_mb:.1f} MB) to {output_path}"
#             )
#
#             output_paths.append(output_path)
#             self.last_request_plan_map[output_path] = segment_plans
#
#         return output_paths
#
#     def _validate_serialised_line(self, line: str) -> None:
#         try:
#             data = json.loads(line)
#         except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
#             raise ValueError(f"Serialised line is not valid JSON: {exc}") from exc
#         self._validate_request_dict(data)
#
#     def _preview_files(self, paths: Sequence[Path], preview_lines: int) -> None:
#         if preview_lines <= 0 or not paths:
#             return
#
#         primary = paths[0]
#         print(
#             f"\n🔍 Previewing first {preview_lines} request lines from {primary.name}:"
#         )
#         with primary.open("r", encoding="utf-8") as handle:
#             for line_number in range(1, preview_lines + 1):
#                 line = handle.readline()
#                 if not line:
#                     break
#                 print(f"  {line_number:02d}: {line.rstrip()}")
#         if len(paths) > 1:
#             print(f"… {len(paths) - 1} additional file(s) not shown")
#         print("")
#
#     def _print_token_audit(self, plans: List[RequestPlan]) -> None:
#         if not plans:
#             return
#
#         print("📑 Token audit per request line")
#         header = f"{'Line':>4} {'Custom ID':<28} {'Type':<10} {'Chunk':>5} {'Input':>10} {'Max':>10}"
#         print(header)
#         print("-" * len(header))
#
#         total_input = 0
#         total_output = 0
#         for plan in plans:
#             total_input += plan.input_tokens
#             total_output += plan.output_tokens
#             print(
#                 f"{plan.line_number:>4} {plan.custom_id:<28} {plan.analysis_type:<10} "
#                 f"{plan.chunk_index + 1:>5} {plan.input_tokens:>10,} {plan.output_tokens:>10,}"
#             )
#
#         print("-" * len(header))
#         print(f"{'':>4} {'TOTAL':<28} {'':<10} {'':>5} {total_input:>10,} {total_output:>10,}")
#         print("")