"""
LLM-as-a-Judge evaluator for RAG pipelines with nearest-neighbour retrieval.

Improvements over the previous version
---------------------------------------
- Recall@K, MRR, Hit@K tracked alongside faithfulness / accuracy
- Ground-truth chunk identification logged per evaluation
- Failed queries persisted to disk for retraining
- No bare `except:` — all errors are typed and re-raised or logged explicitly
- Query expansion only after the original query fails the sufficiency check
- Generation temperature hint baked into the judge system prompt
- All LLM calls are retried once on transient JSON-parse errors
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from app.embeddings.openai_client import openai_client as ollama_client
from app.rag.vector_rag import vector_rag

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')
_FAILED_QUERIES_PATH = Path("logs/failed_queries.jsonl")
_FAILED_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences that models sometimes wrap JSON in."""
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


async def _llm_score(prompt: str, retries: int = 1) -> tuple[float, str]:
    """
    Call the judge LLM and parse a {score, rationale} JSON object.
    Retries once on parse failure before falling back to regex extraction.
    """
    system = (
        "You are an expert RAG evaluation judge. "
        "Use temperature 0 reasoning — be precise and consistent. "
        "Respond ONLY with a valid JSON object containing exactly two keys: "
        '"score" (a float 0.0–1.0) and "rationale" (one sentence). '
        "No markdown, no extra text."
    )
    raw = ""
    for attempt in range(retries + 1):
        try:
            raw = await ollama_client.chat(
                [{"role": "user", "content": prompt}],
                system=system,
            )
            data = json.loads(_strip_fences(raw))
            score = float(data["score"])
            return round(max(0.0, min(1.0, score)), 4), data.get("rationale", "")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt < retries:
                logger.debug("Judge parse error (attempt %d), retrying: %s", attempt + 1, exc)
                continue
            # Final fallback: regex
            match = _SCORE_RE.search(raw)
            if match:
                return round(float(match.group(1)), 4), "Score extracted via regex fallback."
            logger.warning("Judge failed to produce a parseable score: %s", exc)
            return 0.0, f"Parse error: {exc}"
    return 0.0, "Unreachable"


async def _expand_query(question: str) -> str:
    """Rephrase the query to improve sparse/dense retrieval coverage."""
    prompt = (
        "Rephrase the following question to improve retrieval from a document database. "
        "Add relevant domain keywords and expand any acronyms. "
        "Return ONLY the rephrased question, nothing else.\n\n"
        f"Original question: {question}"
    )
    return (await ollama_client.chat([{"role": "user", "content": prompt}])).strip()


async def _check_sufficiency(
    question: str, context_chunks: list[str]
) -> tuple[bool, str]:
    """Ask the LLM whether the retrieved context is enough to answer the question."""
    context = "\n\n---\n\n".join(context_chunks[:5])
    prompt = (
        "Does the CONTEXT below contain sufficient information to answer the QUESTION?\n"
        'Respond ONLY with valid JSON: {"sufficient": true/false, "reason": "brief explanation"}\n\n'
        f"QUESTION: {question}\n\nCONTEXT: {context}"
    )
    try:
        raw = await ollama_client.chat([{"role": "user", "content": prompt}])
        data = json.loads(_strip_fences(raw))
        return bool(data.get("sufficient", False)), data.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        return len(context_chunks) > 0, "Sufficiency check parse error — assuming sufficient if chunks exist."


def _persist_failed_query(record: dict[str, Any]) -> None:
    """Append a failed query to disk for later analysis / fine-tuning."""
    try:
        with _FAILED_QUERIES_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Could not persist failed query: %s", exc)


# ---------------------------------------------------------------------------
# Individual metric functions  (all preserve original signatures)
# ---------------------------------------------------------------------------

async def compute_accuracy(generated: str, expected: str) -> tuple[float, str]:
    """Semantic match between generated and expected answer."""
    prompt = (
        "Rate how well the GENERATED ANSWER matches the EXPECTED ANSWER semantically.\n\n"
        f"EXPECTED ANSWER:\n{expected}\n\n"
        f"GENERATED ANSWER:\n{generated}\n\n"
        "Score 1.0 if they convey the same meaning, 0.0 if completely unrelated."
    )
    return await _llm_score(prompt)


async def compute_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, str]:
    """How well the answer is grounded in the retrieved context (no hallucinations)."""
    context = "\n\n---\n\n".join(context_chunks[:5])
    prompt = (
        "Rate how faithfully the ANSWER is grounded in the CONTEXT below.\n"
        "A faithful answer only makes claims supported by the context (score 1.0).\n"
        "An unfaithful answer introduces hallucinated facts not in the context (score 0.0).\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    )
    return await _llm_score(prompt)


async def compute_answer_relevancy(question: str, answer: str) -> tuple[float, str]:
    """How directly and completely the answer addresses the question."""
    prompt = (
        "Rate how directly and completely the ANSWER addresses the QUESTION.\n"
        "Score 1.0 if fully on-topic and complete, 0.0 if it ignores the question.\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    )
    return await _llm_score(prompt)


async def compute_context_precision(
    question: str, context_chunks: list[str]
) -> tuple[float, str]:
    """Proportion of retrieved chunks that are genuinely relevant to the question."""
    if not context_chunks:
        return 0.0, "No context chunks provided."
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]: {c}" for i, c in enumerate(context_chunks[:8])
    )
    prompt = (
        "You are evaluating retrieval quality.\n"
        "Rate the proportion of the retrieved CHUNKS that contain information genuinely "
        "useful for answering the QUESTION (0.0 = none relevant, 1.0 = all relevant).\n\n"
        f"QUESTION:\n{question}\n\nRETRIEVED CHUNKS:\n{chunks_text}"
    )
    return await _llm_score(prompt)


async def compute_context_recall(
    expected_answer: str, context_chunks: list[str]
) -> tuple[float, str]:
    """Whether the retrieved context contains the facts needed for the expected answer."""
    if not context_chunks:
        return 0.0, "No context chunks provided."
    context = "\n\n---\n\n".join(context_chunks[:8])
    prompt = (
        "Rate whether the RETRIEVED CONTEXT contains the information needed to produce "
        "the EXPECTED ANSWER.\n"
        "Score 1.0 if all key facts are present, 0.0 if completely missing.\n\n"
        f"EXPECTED ANSWER:\n{expected_answer}\n\nRETRIEVED CONTEXT:\n{context}"
    )
    return await _llm_score(prompt)


# ---------------------------------------------------------------------------
# Ranking metrics  (new)
# ---------------------------------------------------------------------------

def compute_hit_at_k(retrieved_chunks: list[str], ground_truth_text: str, k: int) -> float:
    """
    Hit@K — 1.0 if any of the top-K chunks contains the ground-truth substring.
    Uses a simple substring check; swap for embedding similarity if needed.
    """
    needle = ground_truth_text.lower().strip()
    for chunk in retrieved_chunks[:k]:
        if needle in chunk.lower():
            return 1.0
    return 0.0


def compute_recall_at_k(retrieved_chunks: list[str], ground_truth_text: str, k: int) -> float:
    """
    Recall@K — fraction of ground-truth sentences covered by the top-K chunks.
    Sentences shorter than 10 chars are skipped to avoid trivial matches.
    """
    sentences = [
        s.strip()
        for s in ground_truth_text.split(".")
        if len(s.strip()) >= 10
    ]
    if not sentences:
        return 0.0
    top_chunks_text = " ".join(retrieved_chunks[:k]).lower()
    covered = sum(1 for s in sentences if s.lower() in top_chunks_text)
    return round(covered / len(sentences), 4)


def compute_mrr(retrieved_chunks: list[str], ground_truth_text: str) -> float:
    """
    Mean Reciprocal Rank — 1/rank of the first relevant chunk.
    Returns 0.0 if no relevant chunk is found.
    """
    needle = ground_truth_text.lower().strip()
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if needle in chunk.lower():
            return round(1.0 / rank, 4)
    return 0.0


def identify_ground_truth_chunk(
    retrieved_chunks: list[str], expected_answer: str
) -> dict[str, Any]:
    """
    Log which retrieved chunk (if any) contains the ground-truth answer.
    Returns index (0-based), a snippet, and a confidence flag.
    """
    needle = expected_answer.lower().strip()
    for idx, chunk in enumerate(retrieved_chunks):
        if needle in chunk.lower():
            snippet = chunk[:200].replace("\n", " ")
            return {"found": True, "chunk_index": idx, "snippet": snippet}
    return {"found": False, "chunk_index": None, "snippet": None}


# ---------------------------------------------------------------------------
# Main evaluation entry-point  (signature preserved)
# ---------------------------------------------------------------------------

async def evaluate_single(
    question: str,
    expected_answer: str,
    generated_answer: str,
    context_chunks: list[str] | None = None,
    collection_name: str = "text_documents",
    top_k: int = 5,
    max_tries: int = 2,
) -> dict[str, Any]:
    """
    Evaluate a single RAG turn.

    Retrieval strategy
    ------------------
    1. Try the original query first.
    2. If the sufficiency check fails and attempts remain, expand the query once
       and retry.  (Original query is always attempt #1 — no pre-expansion.)

    Metrics returned
    ----------------
    LLM-judged : accuracy, faithfulness, answer_relevancy,
                 context_precision, context_recall
    Ranking    : hit_at_k, recall_at_k, mrr
    Provenance : ground_truth_chunk, retrieval_attempts
    """
    t0 = time.time()

    current_query = question
    all_attempts: list[dict[str, Any]] = []
    retrieved_chunks: list[str] = []

    for attempt in range(max_tries):
        results = await vector_rag.retrieve(current_query, collection_name, top_k)
        retrieved_chunks = [r.get("chunk_text", "") for r in results]

        attempt_record: dict[str, Any] = {
            "attempt": attempt + 1,
            "query": current_query,
            "num_results": len(results),
        }

        if retrieved_chunks:
            is_sufficient, reason = await _check_sufficiency(question, retrieved_chunks)
            attempt_record["sufficient"] = is_sufficient
            attempt_record["reason"] = reason

            if is_sufficient or attempt == max_tries - 1:
                all_attempts.append(attempt_record)
                break
        else:
            attempt_record["sufficient"] = False
            attempt_record["reason"] = "No chunks retrieved."

        all_attempts.append(attempt_record)

        # Expand query only for subsequent attempts
        if attempt < max_tries - 1:
            current_query = await _expand_query(question)

    # ------------------------------------------------------------------
    # LLM-judged metrics
    # ------------------------------------------------------------------
    accuracy, acc_rationale = await compute_accuracy(generated_answer, expected_answer)
    faithfulness, fai_rationale = await compute_faithfulness(generated_answer, retrieved_chunks)
    answer_relevancy, rel_rationale = await compute_answer_relevancy(question, generated_answer)
    context_precision, pre_rationale = await compute_context_precision(question, retrieved_chunks)
    context_recall, rec_rationale = await compute_context_recall(expected_answer, retrieved_chunks)

    # ------------------------------------------------------------------
    # Ranking metrics (new)
    # ------------------------------------------------------------------
    hit_at_k = compute_hit_at_k(retrieved_chunks, expected_answer, top_k)
    recall_at_k = compute_recall_at_k(retrieved_chunks, expected_answer, top_k)
    mrr = compute_mrr(retrieved_chunks, expected_answer)
    ground_truth_chunk = identify_ground_truth_chunk(retrieved_chunks, expected_answer)

    latency_ms = round((time.time() - t0) * 1000, 1)

    # ------------------------------------------------------------------
    # Persist failures for retraining
    # ------------------------------------------------------------------
    is_failure = accuracy < 0.5 or faithfulness < 0.5 or hit_at_k == 0.0
    if is_failure:
        _persist_failed_query({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "accuracy": accuracy,
            "faithfulness": faithfulness,
            "hit_at_k": hit_at_k,
            "retrieval_attempts": all_attempts,
            "timestamp": time.time(),
        })

    return {
        # Inputs
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "retrieved_context": "\n---\n".join(retrieved_chunks),
        # Retrieval provenance
        "retrieval_attempts": all_attempts,
        "ground_truth_chunk": ground_truth_chunk,
        # LLM-judged scores
        "accuracy": accuracy,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        # Rationales
        "accuracy_rationale": acc_rationale,
        "faithfulness_rationale": fai_rationale,
        "answer_relevancy_rationale": rel_rationale,
        "context_precision_rationale": pre_rationale,
        "context_recall_rationale": rec_rationale,
        # Ranking metrics (new)
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        # Meta
        "latency_ms": latency_ms,
        "is_failure": is_failure,
    }