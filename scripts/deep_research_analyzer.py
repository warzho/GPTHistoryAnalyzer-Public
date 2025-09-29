"""Lightweight deep-dive analysis helpers powered by gpt-5-mini."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config.analysis_config import AnalysisConfig
from scripts.batch_analyzer import TokenCounter


class DeepResearchAnalyzer:
    """Performs targeted analyses outside of the batch workflow."""

    def __init__(self) -> None:
        if not AnalysisConfig.validate_config():
            raise ValueError("Configuration validation failed. Please check your config settings.")

        self.client = OpenAI(api_key=AnalysisConfig.OPENAI_API_KEY)
        self.model = AnalysisConfig.DEEP_RESEARCH_MODEL
        self.token_counter = TokenCounter()

        print(f"🔬 Initialized Deep Research Analyzer using model: {self.model}")

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def analyze_project_evolution(
        self,
        conversations: List[Dict[str, Any]],
        target_projects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Summarise project trajectories, milestones, and next steps."""

        messages = self._build_messages(
            conversations,
            prompt_key="project",
            extra_instruction=self._format_target_projects(target_projects),
        )
        response = self._execute_completion(messages)
        return self._build_result("project_evolution", response)

    def analyze_inquiry_patterns(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Highlight inquiry patterns and learning strategies."""

        messages = self._build_messages(conversations, prompt_key="inquiry")
        response = self._execute_completion(messages)
        return self._build_result("inquiry_patterns", response)

    def analyze_knowledge_evolution(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Map conceptual development across the conversations."""

        messages = self._build_messages(conversations, prompt_key="knowledge")
        response = self._execute_completion(messages)
        return self._build_result("knowledge_evolution", response)

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------
    SYSTEM_PROMPTS: Dict[str, str] = {
        "project": (
            "You are an expert project analyst tracking evolution, risks, and next steps across multiple initiatives."
        ),
        "inquiry": (
            "You are an expert learning analyst identifying question styles, learning goals, and support patterns."
        ),
        "knowledge": (
            "You are an expert knowledge mapper surfacing recurring domains, concept depth, and knowledge gaps."
        ),
    }

    USER_DIRECTIVES: Dict[str, str] = {
        "project": (
            "Focus on project scope changes, milestone progress, blockers, dependencies, and recommended next actions."
        ),
        "inquiry": (
            "Focus on question framing, investigation depth, guidance requests, and emerging learning strategies."
        ),
        "knowledge": (
            "Focus on key domains, concept progression, relationships between topics, and unresolved questions."
        ),
    }

    def _build_messages(
        self,
        conversations: List[Dict[str, Any]],
        prompt_key: str,
        extra_instruction: str = "",
    ) -> List[Dict[str, str]]:
        context_text = self._render_conversation_context(conversations, prompt_key)
        directive = self.USER_DIRECTIVES[prompt_key]
        if extra_instruction:
            directive = f"{directive}\n{extra_instruction.strip()}"

        header = (
            f"Analyze the following conversations. {directive} Provide concise bullet summaries with references "
            "to relevant conversations when useful."
        )

        user_content = f"{header}\n\n{context_text}".strip()
        return [
            {"role": "system", "content": self.SYSTEM_PROMPTS[prompt_key]},
            {"role": "user", "content": user_content},
        ]

    def _render_conversation_context(self, conversations: List[Dict[str, Any]], prompt_key: str) -> str:
        if not conversations:
            return "No conversations supplied."

        max_conversations = max(1, min(AnalysisConfig.MAX_CONVERSATIONS_PER_BATCH, len(conversations)))
        # Prioritise the most recent conversations which usually contain the most relevant context
        subset = conversations[-max_conversations:]

        segments: List[str] = []
        for idx, conv in enumerate(subset, start=1):
            title = (conv.get("title") or f"Conversation {idx}").strip()
            timestamp = conv.get("create_time")
            header = title if not timestamp else f"{title} — {timestamp}"
            segments.append(f"### Conversation {idx}: {header}")
            segments.extend(self._format_messages(conv.get("messages", []), prompt_key))

        text = "\n".join(segment for segment in segments if segment).strip()
        return self._enforce_token_budget(text)

    def _format_messages(self, messages: List[Dict[str, Any]], prompt_key: str) -> List[str]:
        lines: List[str] = []
        for message in messages:
            role = message.get("role", "user").strip().lower()
            content = self._clean_content(message.get("content", ""))
            if not content:
                continue
            if prompt_key == "inquiry" and role != "user":
                # Emphasise the questioning behaviour by skipping assistant responses
                continue
            lines.append(f"{role}: {content}")
        return lines

    def _clean_content(self, content: str) -> str:
        if not content:
            return ""
        cleaned = " ".join(content.replace("\r\n", "\n").replace("\r", "\n").split())
        return cleaned.strip()

    def _enforce_token_budget(self, text: str) -> str:
        budget = AnalysisConfig.get_input_token_budget()
        tokens = self.token_counter.count_text(text)
        if tokens <= budget:
            return text

        lines = text.split("\n")
        # Drop oldest lines until within budget
        while tokens > budget and len(lines) > 1:
            lines.pop(0)
            tokens = self.token_counter.count_text("\n".join(lines))

        if tokens <= budget:
            return "\n".join(lines)

        # As a last resort, truncate remaining text evenly
        return self._truncate_lines(lines, budget)

    def _truncate_lines(self, lines: List[str], token_budget: int) -> str:
        note = "[Context truncated to satisfy token budget; showing most recent segments only.]"
        note_tokens = self.token_counter.count_text(note)
        allowance = max(token_budget - note_tokens, 0)
        if allowance <= 0:
            return note

        tail: List[str] = []
        accumulated = 0
        for line in reversed(lines):
            tokens = self.token_counter.count_text(line)
            if accumulated + tokens > allowance:
                break
            tail.append(line)
            accumulated += tokens

        tail.reverse()
        compact = [note]
        compact.extend(tail)
        return "\n".join(compact)

    def _format_target_projects(self, target_projects: Optional[List[str]]) -> str:
        if not target_projects:
            return ""
        joined = ", ".join(sorted(set(p.strip() for p in target_projects if p)))
        if not joined:
            return ""
        return f"Highlight progress for: {joined}."

    # ------------------------------------------------------------------
    # Completion execution
    # ------------------------------------------------------------------
    def _execute_completion(self, messages: List[Dict[str, str]]):
        input_tokens = self.token_counter.count_chat_tokens(messages)
        if input_tokens > AnalysisConfig.MAX_INPUT_TOKENS:
            raise ValueError(
                f"Planned input tokens exceed model limit ({input_tokens} > {AnalysisConfig.MAX_INPUT_TOKENS})."
            )

        max_tokens = AnalysisConfig.plan_output_tokens(input_tokens)
        if input_tokens + max_tokens > AnalysisConfig.CONTEXT_WINDOW:
            max_tokens = max(AnalysisConfig.CONTEXT_WINDOW - input_tokens, 0)
        if max_tokens <= 0:
            raise ValueError("Unable to allocate output tokens within context window.")

        print(
            f"⚙️ Requesting completion (input tokens ≈ {input_tokens}, reserved output ≈ {max_tokens})"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response

    def _build_result(self, analysis_type: str, response) -> Dict[str, Any]:
        try:
            content = response.choices[0].message.content
        except Exception:  # pragma: no cover - defensive for unexpected response shapes
            content = ""
        return {
            "analysis_type": analysis_type,
            "timestamp": time.time(),
            "model_used": self.model,
            "analysis_content": content,
            "success": bool(content),
        }

