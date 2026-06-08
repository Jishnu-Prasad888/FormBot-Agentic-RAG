"""
Simple nearest-neighbor RAG evaluation.

Uses direct retrieval + LLM generation and scoring (no multi-agent orchestration).
"""

import time
from typing import Any

from app.services.rag_service import rag_service
from app.evaluation.evaluator import evaluate_single
from app.embeddings.openai_client import openai_client
from app.rag.cross_encoder import cross_encoder
from app.core.logging import get_logger

logger = get_logger("evaluation.agent_runner")

RAG_SYSTEM = """Answer the question carefully"""

QUERY_EXPANSION_PROMPT = """Given the user's question, generate {num_expansions} alternative phrasings or related queries that would help retrieve relevant information from a knowledge base.

Original question: {question}

Generate {num_expansions} expanded queries as a numbered list (1., 2., 3., etc.). Each query should:
- Rephrase the original question differently
- Ask for related aspects that would help answer the original question
- Use different terminology or synonyms

Output only the numbered list, nothing else."""


def _chunk_texts(chunks: list[dict]) -> list[str]:
    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]


async def _expand_query(question: str, num_expansions: int = 2) -> list[str]:
    """Generate expanded queries using LLM."""
    prompt = QUERY_EXPANSION_PROMPT.format(question=question, num_expansions=num_expansions)
    response = await openai_client.generate(prompt, system="You are a query expansion assistant.")
    
    expanded = [question]  # Always include original
    for line in response.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            query = line.split(".", 1)[-1].strip() if "." in line else line.lstrip("- ")
            if query:
                expanded.append(query)
    
    return expanded[:num_expansions + 1]


async def evaluate_question(
    question: str,
    expected_answer: str,
    top_k: int = 5,
    use_query_expansion: bool = False,
    num_expansions: int = 2,
) -> dict[str, Any]:
    """
    Run one Q&A pair through RAG with optional query expansion.
    
    1. Expand query into multiple variants (if enabled)
    2. Retrieve chunks for each query variant
    3. Combine and deduplicate all chunks
    4. Generate answer from combined context
    5. Score with LLM-as-judge
    """
    t0 = time.time()
    
    # Query expansion before retrieval
    if use_query_expansion:
        queries = await _expand_query(question, num_expansions)
    else:
        queries = [question]
    
    # Retrieve chunks for all queries
    all_chunks = []
    chunk_ids_seen = set()
    
    for query in queries:
        chunks = await rag_service.retrieve(query, strategy="hybrid", top_k=top_k * 2)
        chunks = cross_encoder.rerank(query, chunks, top_k=top_k)
        
        for c in chunks:
            cid = c.get("chunk_id")
            if cid and cid not in chunk_ids_seen:
                chunk_ids_seen.add(cid)
                all_chunks.append(c)
    
    # Generate answer
    chunk_texts = _chunk_texts(all_chunks)
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
        "expanded_queries": queries if use_query_expansion else [],
        "num_chunks": len(all_chunks),
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
        "expanded_queries": [],
        "num_chunks": 0,
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
