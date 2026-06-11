"""
LLM-as-a-Judge evaluator for RAG pipelines.

Each metric is computed by prompting the LLM to score (0.0–1.0) with a brief rationale,
rather than relying on cosine-similarity heuristics.
"""

import json
import re
import time
from typing import Any

from app.embeddings.openai_client import openai_client as ollama_client



# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')


async def _llm_score(prompt: str) -> tuple[float, str]:
    """
    Call the LLM with `prompt` and extract a JSON payload like:
        {"score": 0.85, "rationale": "..."}
    Returns (score, rationale).  Falls back to (0.0, error_msg) on failure.
    """
    system = (
        "You are an expert RAG evaluation judge. "
        "Respond ONLY with a JSON object containing exactly two keys: "
        '"score" (a float between 0.0 and 1.0) and '
        '"rationale" (a one-sentence explanation). '
        "Do not include any other text."
    )
    raw = ""  # initialise so it's always bound
    try:
        raw = await ollama_client.chat(
            [{"role": "user", "content": prompt}],
            system=system,
        )
        # Strip markdown fences if present
        raw = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        rationale = data.get("rationale", "")
        return round(max(0.0, min(1.0, score)), 4), rationale
    except Exception as exc:
        # Fallback: try regex on whatever we got back
        m = _SCORE_RE.search(raw)
        if m:
            return round(float(m.group(1)), 4), "Score extracted via regex fallback."
        return 0.0, f"Parse error: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Metric functions
# ──────────────────────────────────────────────────────────────────────────────

async def compute_accuracy(generated: str, expected: str) -> tuple[float, str]:
    """Semantic accuracy: how well does the generated answer match the expected answer?"""
    prompt = f"""Rate how well the GENERATED ANSWER matches the EXPECTED ANSWER semantically.

EXPECTED ANSWER:
{expected}

GENERATED ANSWER:
{generated}

Score 1.0 if they convey the same meaning, 0.0 if completely unrelated."""
    return await _llm_score(prompt)


async def compute_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, str]:
    """Faithfulness: is the answer grounded in the retrieved context?"""
    context = "\n\n---\n\n".join(context_chunks[:5])  # cap to avoid token overflow
    prompt = f"""Rate how faithfully the ANSWER is grounded in the CONTEXT below.
A faithful answer only makes claims supported by the context (score 1.0).
An unfaithful answer introduces hallucinated facts not in the context (score 0.0).

CONTEXT:
{context}

ANSWER:
{answer}"""
    return await _llm_score(prompt)


async def compute_answer_relevancy(question: str, answer: str) -> tuple[float, str]:
    """Answer relevancy: does the answer directly address the question?"""
    prompt = f"""Rate how directly and completely the ANSWER addresses the QUESTION.
Score 1.0 if the answer is fully on-topic and complete, 0.0 if it ignores the question.

QUESTION:
{question}

ANSWER:
{answer}"""
    return await _llm_score(prompt)


async def compute_context_precision(question: str, context_chunks: list[str]) -> tuple[float, str]:
    """Context precision: what fraction of retrieved chunks are actually relevant?"""
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
    """Context recall: does the retrieved context contain what's needed to answer correctly?"""
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


# ──────────────────────────────────────────────────────────────────────────────
# Master evaluation function
# ──────────────────────────────────────────────────────────────────────────────

async def evaluate_single(
    question: str,
    expected_answer: str,
    generated_answer: str,
    context_chunks: list[str],
) -> dict[str, Any]:
    """
    Run all five metrics for a single Q&A pair via the LLM judge.
    Returns a dict with scores, rationales, and the raw inputs for export.
    """
    t0 = time.time()

    accuracy,          acc_rationale  = await compute_accuracy(generated_answer, expected_answer)
    faithfulness,      fai_rationale  = await compute_faithfulness(generated_answer, context_chunks)
    answer_relevancy,  rel_rationale  = await compute_answer_relevancy(question, generated_answer)
    context_precision, pre_rationale  = await compute_context_precision(question, context_chunks)
    context_recall,    rec_rationale  = await compute_context_recall(expected_answer, context_chunks)

    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        # ── inputs (for export) ───────────────────────────────────────────────
        "question":          question,
        "expected_answer":   expected_answer,
        "generated_answer":  generated_answer,
        "retrieved_context": "\n---\n".join(context_chunks),
        # ── scores ───────────────────────────────────────────────────────────
        "accuracy":          accuracy,
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_precision": context_precision,
        "context_recall":    context_recall,
        # ── rationales ───────────────────────────────────────────────────────
        "accuracy_rationale":          acc_rationale,
        "faithfulness_rationale":      fai_rationale,
        "answer_relevancy_rationale":  rel_rationale,
        "context_precision_rationale": pre_rationale,
        "context_recall_rationale":    rec_rationale,
        # ── meta ─────────────────────────────────────────────────────────────
        "latency_ms": latency_ms,
    }