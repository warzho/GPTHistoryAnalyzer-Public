"""Cost estimation helpers aligned with the batch planning logic."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from config.analysis_config import AnalysisConfig
from scripts.batch_analyzer import (
    ANALYSIS_TYPES,
    BatchPlanner,
    ConversationChunk,
    SYSTEM_PROMPTS,
    USER_FOCUS_TEXT,
)
from scripts.utils import TokenCounter


class ComprehensiveCostEstimator:
    """Estimate token usage and pricing for the batch analysis pipeline."""

    TOKENS_PER_MILLION = 1_000_000

    PRICING: Dict[str, Dict[str, Dict[str, float]]] = {
        "gpt-5": {
            "realtime": {
                "input_per_million_tokens": 0.63,
                "cached_input_per_million_tokens": 0.06,
                "output_per_million_tokens": 5.00,
            },
            "batch": {
                "input_per_million_tokens": 0.625,
                "cached_input_per_million_tokens": 0.063,
                "output_per_million_tokens": 5.00,
            },
        },
        "gpt-5-mini": {
            "realtime": {
                "input_per_million_tokens": 0.13,
                "cached_input_per_million_tokens": 0.01,
                "output_per_million_tokens": 1.00,
            },
            "batch": {
                # No published batch differential for gpt-5 mini; mirror realtime pricing.
                "input_per_million_tokens": 0.13,
                "cached_input_per_million_tokens": 0.01,
                "output_per_million_tokens": 1.00,
            },
        },
    }

    def __init__(self) -> None:
        self.model = AnalysisConfig.ANALYSIS_MODEL
        self.token_counter = TokenCounter(self.model)
        self.planner = BatchPlanner(
            token_counter=self.token_counter,
            input_budget=AnalysisConfig.get_input_token_budget(),
            system_prompts=SYSTEM_PROMPTS,
            user_focus_text=USER_FOCUS_TEXT,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate_comprehensive_analysis_cost(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not conversations:
            raise ValueError("No conversations provided for cost estimation")

        prepared = self.planner.prepare_conversations(conversations)
        if not prepared:
            raise ValueError("Conversations contained no analysable content")

        chunks = self.planner.chunk_conversations(prepared)
        if not chunks:
            raise RuntimeError("Chunk planning produced zero chunks; cannot estimate costs")

        chunk_metrics = self._collect_chunk_metrics(chunks)
        analysis_estimates, totals = self._compute_analysis_usage(chunks)
        cost_breakdown = self._calculate_costs(totals)
        recommendations = self._build_recommendations(len(prepared), cost_breakdown)

        conversation_tokens = sum(conv.token_count for conv in prepared)
        average_tokens = conversation_tokens // len(prepared)

        return {
            "conversation_summary": {
                "total_conversations": len(prepared),
                "total_conversation_tokens": conversation_tokens,
                "average_tokens_per_conversation": average_tokens,
            },
            "chunk_metrics": chunk_metrics,
            "analysis_estimates": analysis_estimates,
            "token_totals": totals,
            "cost_breakdown": cost_breakdown,
            "recommendations": recommendations,
            "pricing_model": self.model,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_chunk_metrics(self, chunks: List[ConversationChunk]) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        for chunk in chunks:
            messages = self.planner.build_chunk_messages(chunk, ANALYSIS_TYPES[0][0])
            input_tokens = self.token_counter.count_chat_tokens(messages)
            output_tokens = AnalysisConfig.plan_output_tokens(input_tokens)
            metrics.append(
                {
                    "chunk_index": chunk.index + 1,
                    "conversation_count": len(chunk.conversations),
                    "conversation_tokens": chunk.conversation_tokens,
                    "input_tokens": input_tokens,
                    "max_tokens": output_tokens,
                }
            )
        return metrics

    def _compute_analysis_usage(
        self, chunks: List[ConversationChunk]
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        usage: Dict[str, Dict[str, int]] = {}
        total_input = 0
        total_output = 0
        total_requests = 0

        for analysis_key, custom_prefix in ANALYSIS_TYPES:
            request_tokens: List[Tuple[int, int]] = []
            for chunk in chunks:
                messages = self.planner.build_chunk_messages(chunk, analysis_key)
                input_tokens = self.token_counter.count_chat_tokens(messages)
                output_tokens = AnalysisConfig.plan_output_tokens(input_tokens)
                request_tokens.append((input_tokens, output_tokens))

            summary = self._summarise_request_tokens(request_tokens)
            usage[custom_prefix] = summary

            total_input += summary["total_input_tokens"]
            total_output += summary["total_output_tokens"]
            total_requests += summary["requests"]

        totals = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_requests": total_requests,
        }
        return usage, totals

    @staticmethod
    def _summarise_request_tokens(tokens: List[Tuple[int, int]]) -> Dict[str, int]:
        if not tokens:
            return {
                "requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "max_input_tokens": 0,
                "max_output_tokens": 0,
                "avg_input_tokens": 0,
                "avg_output_tokens": 0,
            }

        inputs = [item[0] for item in tokens]
        outputs = [item[1] for item in tokens]
        request_count = len(tokens)

        return {
            "requests": request_count,
            "total_input_tokens": sum(inputs),
            "total_output_tokens": sum(outputs),
            "max_input_tokens": max(inputs),
            "max_output_tokens": max(outputs),
            "avg_input_tokens": int(round(sum(inputs) / request_count)),
            "avg_output_tokens": int(round(sum(outputs) / request_count)),
        }

    def _calculate_costs(self, totals: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        pricing = self.PRICING.get(self.model)
        if pricing is None:
            raise KeyError(f"Pricing for model {self.model} is not configured")

        realtime_rates = pricing["realtime"]
        batch_rates = pricing.get("batch", realtime_rates)

        total_input_tokens = totals["total_input_tokens"]
        cached_input_tokens = totals.get("cached_input_tokens", 0)
        regular_input_tokens = max(total_input_tokens - cached_input_tokens, 0)
        total_output_tokens = totals["total_output_tokens"]

        def _cost(tokens: int, rate_per_million: float) -> float:
            return (tokens / self.TOKENS_PER_MILLION) * rate_per_million

        realtime_regular_input_cost = _cost(
            regular_input_tokens, realtime_rates["input_per_million_tokens"]
        )
        realtime_cached_input_cost = _cost(
            cached_input_tokens, realtime_rates["cached_input_per_million_tokens"]
        )
        realtime_output_cost = _cost(
            total_output_tokens, realtime_rates["output_per_million_tokens"]
        )
        realtime_total_cost = (
            realtime_regular_input_cost
            + realtime_cached_input_cost
            + realtime_output_cost
        )

        batch_regular_input_cost = _cost(
            regular_input_tokens, batch_rates["input_per_million_tokens"]
        )
        batch_cached_input_cost = _cost(
            cached_input_tokens, batch_rates["cached_input_per_million_tokens"]
        )
        batch_output_cost = _cost(
            total_output_tokens, batch_rates["output_per_million_tokens"]
        )
        batch_total_cost = (
            batch_regular_input_cost + batch_cached_input_cost + batch_output_cost
        )

        savings = realtime_total_cost - batch_total_cost
        savings_pct = (savings / realtime_total_cost * 100) if realtime_total_cost else 0.0

        return {
            "regular_pricing": {
                "input_cost": realtime_regular_input_cost,
                "cached_input_cost": realtime_cached_input_cost,
                "output_cost": realtime_output_cost,
                "total_cost": realtime_total_cost,
            },
            "batch_pricing": {
                "input_cost": batch_regular_input_cost,
                "cached_input_cost": batch_cached_input_cost,
                "output_cost": batch_output_cost,
                "total_cost": batch_total_cost,
            },
            "savings_analysis": {
                "dollar_savings": savings,
                "percentage_savings": savings_pct,
            },
        }

    def _build_recommendations(
        self, conversation_count: int, cost_breakdown: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        threshold = AnalysisConfig.REALTIME_THRESHOLD
        recommended_mode = "batch" if conversation_count >= threshold else "realtime"

        if recommended_mode == "batch":
            reasoning = (
                "Dataset size benefits from batch processing. Expect ~24h turnaround with "
                "roughly 50% cost savings compared to real-time calls."
            )
        else:
            reasoning = (
                "Small dataset detected. Real-time processing will provide faster results "
                "while staying within safe token limits."
            )

        batch_cost = cost_breakdown["batch_pricing"]["total_cost"]
        realtime_cost = cost_breakdown["regular_pricing"]["total_cost"]

        return {
            "recommended_mode": recommended_mode,
            "reasoning": reasoning,
            "batch_cost": f"${batch_cost:,.4f}",
            "realtime_cost": f"${realtime_cost:,.4f}",
        }


def display_comprehensive_cost_analysis(cost_analysis: Dict[str, Any]) -> None:
    """Pretty-print the structured cost analysis for human review."""

    summary = cost_analysis["conversation_summary"]
    chunk_metrics = cost_analysis["chunk_metrics"]
    analysis_estimates = cost_analysis["analysis_estimates"]
    totals = cost_analysis["token_totals"]
    costs = cost_analysis["cost_breakdown"]
    recommendations = cost_analysis["recommendations"]

    print("\n" + "=" * 70)
    print("📊 DATASET SUMMARY")
    print("=" * 70)
    print(
        f"Conversations: {summary['total_conversations']:,} | "
        f"Tokens: {summary['total_conversation_tokens']:,} | "
        f"Avg tokens/conversation: {summary['average_tokens_per_conversation']:,}"
    )

    chunk_count = len(chunk_metrics)
    if chunk_count:
        min_input = min(metric["input_tokens"] for metric in chunk_metrics)
        max_input = max(metric["input_tokens"] for metric in chunk_metrics)
        avg_input = sum(metric["input_tokens"] for metric in chunk_metrics) // chunk_count
        print(
            f"Chunks planned: {chunk_count} | "
            f"Input tokens per chunk (min/avg/max): {min_input:,} / {avg_input:,} / {max_input:,}"
        )
    else:
        print("Chunks planned: 0")

    print("\nANALYSIS REQUEST ESTIMATES")
    for name, estimate in analysis_estimates.items():
        print(
            f"  {name}: {estimate['requests']} requests | "
            f"input={estimate['total_input_tokens']:,} | "
            f"output={estimate['total_output_tokens']:,}"
        )

    print("\nTOTAL TOKEN BUDGET")
    print(
        f"  Requests: {totals['total_requests']} | "
        f"input tokens: {totals['total_input_tokens']:,} | "
        f"output tokens: {totals['total_output_tokens']:,}"
    )

    print("\nCOST BREAKDOWN")
    print(
        f"  Regular API: ${costs['regular_pricing']['total_cost']:,.4f} "
        f"(input ${costs['regular_pricing']['input_cost']:,.4f}, "
        f"cached ${costs['regular_pricing']['cached_input_cost']:,.4f}, "
        f"output ${costs['regular_pricing']['output_cost']:,.4f})"
    )
    print(
        f"  Batch API:   ${costs['batch_pricing']['total_cost']:,.4f} "
        f"(input ${costs['batch_pricing']['input_cost']:,.4f}, "
        f"cached ${costs['batch_pricing']['cached_input_cost']:,.4f}, "
        f"output ${costs['batch_pricing']['output_cost']:,.4f})"
    )
    print(
        f"  Savings:     ${costs['savings_analysis']['dollar_savings']:,.4f} "
        f"({costs['savings_analysis']['percentage_savings']:.1f}% reduction)"
    )

    print("\nRECOMMENDATION")
    print(f"  Mode: {recommendations['recommended_mode'].upper()}")
    print(f"  Why:  {recommendations['reasoning']}")
    print(
        f"  Batch cost: {recommendations['batch_cost']} | "
        f"Real-time cost: {recommendations['realtime_cost']}"
    )
    print("=" * 70 + "\n")
