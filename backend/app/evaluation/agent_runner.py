"""
Simple nearest-neighbor RAG evaluation.

Uses direct retrieval + LLM generation and scoring (no multi-agent orchestration).
"""

import time
from typing import Any

from app.services.rag_service import rag_service
from app.evaluation.evaluator import evaluate_single
from app.embeddings.openai_client import openai_client
from app.core.logging import get_logger

logger = get_logger("evaluation.agent_runner")

RAG_SYSTEM = """Answer the question carefully"""


def _chunk_texts(chunks: list[dict]) -> list[str]:
    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]


async def evaluate_question(
    question: str,
    expected_answer: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Run one Q&A pair through simple nearest-neighbor RAG.
    
    1. Retrieve top_k chunks (hybrid search)
    2. Generate answer from context
    3. Score with LLM-as-judge
    """
    t0 = time.time()

    # Retrieve chunks
    chunks = await rag_service.retrieve(question, strategy="hybrid", top_k=top_k)
    
    # Generate answer
    chunk_texts = _chunk_texts(chunks)
    context_text = "\n---\n".join(chunk_texts)
    
    if context_text.strip():
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    else:
        prompt = f"Question: {question}"
    
    generated_answer = await openai_client.generate(prompt, system=RAG_SYSTEM)
    
    # Score
    scores = await evaluate_single(question, expected_answer, generated_answer, chunk_texts)
    
    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "retrieved_context": context_text,
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
        "error": error,
    }
