"""
Multi-agent evaluation runner.

Orchestrates coordinator_agent (retrieval + answer synthesis) and
evaluator_agent (LLM-as-judge scoring) for each ground-truth Q&A pair.
"""

import time
from typing import Any

from app.agents.coordinator_agent import coordinator_agent
from app.agents.evaluator_agent import evaluator_agent
from app.core.prompts import SBI_SYSTEM_PROMPT


def _chunk_texts(chunks: list[dict]) -> list[str]:
    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]


async def evaluate_question(
    question: str,
    expected_answer: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Run one Q&A pair through the multi-agent eval pipeline.

    1. coordinator_agent — routes to retrieval agents, synthesizes final answer
    2. evaluator_agent — scores answer against expected_answer and retrieved chunks
    """
    t0 = time.time()

    coord = await coordinator_agent.run(
        question,
        {"top_k": top_k, "synthesis_system": SBI_SYSTEM_PROMPT},
    )
    chunks = coord.get("chunks", [])
    generated_answer = coord.get("answer", "")

    scores = await evaluator_agent.run(
        question,
        {
            "answer": generated_answer,
            "chunks": chunks,
            "expected_answer": expected_answer,
        },
    )

    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "retrieved_context": "\n---\n".join(_chunk_texts(chunks)),
        "accuracy": scores.get("accuracy", 0.0),
        "faithfulness": scores.get("faithfulness", 0.0),
        "answer_relevancy": scores.get("answer_relevancy", 0.0),
        "context_precision": scores.get("context_precision", 0.0),
        "context_recall": scores.get("context_recall", 0.0),
        "accuracy_rationale": scores.get("accuracy_rationale", ""),
        "faithfulness_rationale": scores.get("faithfulness_rationale", ""),
        "answer_relevancy_rationale": scores.get("answer_relevancy_rationale", ""),
        "context_precision_rationale": scores.get("context_precision_rationale", ""),
        "context_recall_rationale": scores.get("context_recall_rationale", ""),
        "latency_ms": latency_ms,
        "intent": coord.get("intent"),
        "agents_invoked": [r.get("agent") for r in coord.get("agent_results", [])],
        "retrieval_coverage": scores.get("retrieval_coverage"),
        "retrieval_ok": scores.get("retrieval_ok"),
        "overall_score": scores.get("overall_score"),
    }


def failed_question_row(question: str, expected_answer: str, error: str) -> dict[str, Any]:
    """Build a zeroed per-question row when the eval pipeline raises."""
    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": "",
        "retrieved_context": "",
        "accuracy": 0.0,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "accuracy_rationale": "",
        "faithfulness_rationale": "",
        "answer_relevancy_rationale": "",
        "context_precision_rationale": "",
        "context_recall_rationale": "",
        "latency_ms": 0.0,
        "intent": None,
        "agents_invoked": [],
        "retrieval_coverage": None,
        "retrieval_ok": None,
        "overall_score": None,
        "error": error,
    }
