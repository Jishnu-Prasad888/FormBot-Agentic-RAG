"""
LLM-as-a-Judge evaluator for RAG pipelines with nearest neighbor retrieval.
"""

import json
import re
import time
from typing import Any

from app.embeddings.openai_client import openai_client as ollama_client
from app.rag.vector_rag import vector_rag


_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')


async def _expand_query(question: str) -> str:
    """Use LLM to expand/rephrase the query for better retrieval."""
    prompt = f"""Rephrase the following question to improve retrieval from a document database. 
Make it more specific and include relevant keywords. Return only the rephrased question.

Original question: {question}"""
    return await ollama_client.chat([{"role": "user", "content": prompt}])


async def _check_sufficiency(question: str, context_chunks: list[str]) -> tuple[bool, str]:
    """Check if retrieved context is sufficient to answer the question."""
    context = "\n\n---\n\n".join(context_chunks[:5])
    prompt = f"""Does the CONTEXT below contain sufficient information to answer the QUESTION?
Respond with a JSON object: {{"sufficient": true/false, "reason": "brief explanation"}}

QUESTION: {question}

CONTEXT: {context}"""
    
    try:
        response = await ollama_client.chat([{"role": "user", "content": prompt}])
        response = response.strip().strip("```json").strip("```").strip()
        data = json.loads(response)
        return data.get("sufficient", False), data.get("reason", "")
    except:
        return len(context_chunks) > 0, "Parse error"


async def _llm_score(prompt: str) -> tuple[float, str]:
    system = (
        "You are an expert RAG evaluation judge. "
        "Respond ONLY with a JSON object containing exactly two keys: "
        '"score" (a float between 0.0 and 1.0) and '
        '"rationale" (a one-sentence explanation). '
        "Do not include any other text."
    )
    raw = ""
    try:
        raw = await ollama_client.chat(
            [{"role": "user", "content": prompt}],
            system=system,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        rationale = data.get("rationale", "")
        return round(max(0.0, min(1.0, score)), 4), rationale
    except Exception as exc:
        m = _SCORE_RE.search(raw)
        if m:
            return round(float(m.group(1)), 4), "Score extracted via regex fallback."
        return 0.0, f"Parse error: {exc}"


async def compute_accuracy(generated: str, expected: str) -> tuple[float, str]:
    prompt = f"""Rate how well the GENERATED ANSWER matches the EXPECTED ANSWER semantically.

EXPECTED ANSWER:
{expected}

GENERATED ANSWER:
{generated}

Score 1.0 if they convey the same meaning, 0.0 if completely unrelated."""
    return await _llm_score(prompt)


async def compute_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, str]:
    context = "\n\n---\n\n".join(context_chunks[:5])
    prompt = f"""Rate how faithfully the ANSWER is grounded in the CONTEXT below.
A faithful answer only makes claims supported by the context (score 1.0).
An unfaithful answer introduces hallucinated facts not in the context (score 0.0).

CONTEXT:
{context}

ANSWER:
{answer}"""
    return await _llm_score(prompt)


async def compute_answer_relevancy(question: str, answer: str) -> tuple[float, str]:
    prompt = f"""Rate how directly and completely the ANSWER addresses the QUESTION.
Score 1.0 if the answer is fully on-topic and complete, 0.0 if it ignores the question.

QUESTION:
{question}

ANSWER:
{answer}"""
    return await _llm_score(prompt)


async def compute_context_precision(question: str, context_chunks: list[str]) -> tuple[float, str]:
    if not context_chunks:
        return 0.0, "No context chunks provided."
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks[:8])
    )
    prompt = f"""You are evaluating retrieval quality.
Below are chunks retrieved for a QUESTION. Rate the proportion of chunks that contain
information genuinely useful for answering the question (0.0 = none relevant, 1.0 = all relevant).

QUESTION:
{question}

RETRIEVED CHUNKS:
{chunks_text}"""
    return await _llm_score(prompt)


async def compute_context_recall(expected_answer: str, context_chunks: list[str]) -> tuple[float, str]:
    if not context_chunks:
        return 0.0, "No context chunks provided."
    context = "\n\n---\n\n".join(context_chunks[:8])
    prompt = f"""Rate whether the RETRIEVED CONTEXT contains the information needed to produce the EXPECTED ANSWER.
Score 1.0 if all key facts for the expected answer are present in the context, 0.0 if completely missing.

EXPECTED ANSWER:
{expected_answer}

RETRIEVED CONTEXT:
{context}"""
    return await _llm_score(prompt)


async def evaluate_single(
    question: str,
    expected_answer: str,
    generated_answer: str,
    context_chunks: list[str] = None,
    collection_name: str = "text_documents",
    top_k: int = 5,
    max_tries: int = 2,
) -> dict[str, Any]:
    """
    Retrieve with iterative query expansion if context insufficient (max 2 tries).
    """
    t0 = time.time()
    
    current_query = question
    all_attempts = []
    retrieved_chunks = []
    
    for attempt in range(max_tries):
        # Retrieve with current query
        results = await vector_rag.retrieve(current_query, collection_name, top_k)
        retrieved_chunks = [r.get("chunk_text", "") for r in results]
        
        all_attempts.append({
            "attempt": attempt + 1,
            "query": current_query,
            "num_results": len(results)
        })
        
        # Check if context is sufficient
        if retrieved_chunks:
            is_sufficient, reason = await _check_sufficiency(question, retrieved_chunks)
            all_attempts[-1]["sufficient"] = is_sufficient
            all_attempts[-1]["reason"] = reason
            
            if is_sufficient or attempt == max_tries - 1:
                break
        
        # Expand query for next attempt
        if attempt < max_tries - 1:
            current_query = await _expand_query(current_query)
    
    # Run LLM-based evaluation metrics
    accuracy, acc_rationale = await compute_accuracy(generated_answer, expected_answer)
    faithfulness, fai_rationale = await compute_faithfulness(generated_answer, retrieved_chunks)
    answer_relevancy, rel_rationale = await compute_answer_relevancy(question, generated_answer)
    context_precision, pre_rationale = await compute_context_precision(question, retrieved_chunks)
    context_recall, rec_rationale = await compute_context_recall(expected_answer, retrieved_chunks)

    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "retrieved_context": "\n---\n".join(retrieved_chunks),
        "retrieval_attempts": all_attempts,
        "accuracy": accuracy,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "accuracy_rationale": acc_rationale,
        "faithfulness_rationale": fai_rationale,
        "answer_relevancy_rationale": rel_rationale,
        "context_precision_rationale": pre_rationale,
        "context_recall_rationale": rec_rationale,
        "latency_ms": latency_ms,
    }