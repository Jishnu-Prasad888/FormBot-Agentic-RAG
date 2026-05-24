import time
from typing import Any
from app.embeddings.ollama_client import ollama_client
from app.core.logging import get_logger

logger = get_logger("evaluator")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def compute_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Approximate: embed answer and each context chunk, take max similarity."""
    if not context_chunks:
        return 0.0
    answer_emb = await ollama_client.embeddings(answer)
    scores = []
    for chunk in context_chunks:
        chunk_emb = await ollama_client.embeddings(chunk)
        scores.append(_cosine_similarity(answer_emb, chunk_emb))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


async def compute_answer_relevancy(question: str, answer: str) -> float:
    q_emb = await ollama_client.embeddings(question)
    a_emb = await ollama_client.embeddings(answer)
    return round(_cosine_similarity(q_emb, a_emb), 4)


async def compute_context_precision(question: str, context_chunks: list[str]) -> float:
    if not context_chunks:
        return 0.0
    q_emb = await ollama_client.embeddings(question)
    relevant = 0
    for chunk in context_chunks:
        c_emb = await ollama_client.embeddings(chunk)
        sim = _cosine_similarity(q_emb, c_emb)
        if sim > 0.6:
            relevant += 1
    return round(relevant / len(context_chunks), 4)


async def compute_context_recall(expected_answer: str, context_chunks: list[str]) -> float:
    if not context_chunks:
        return 0.0
    ea_emb = await ollama_client.embeddings(expected_answer)
    sims = []
    for chunk in context_chunks:
        c_emb = await ollama_client.embeddings(chunk)
        sims.append(_cosine_similarity(ea_emb, c_emb))
    return round(max(sims) if sims else 0.0, 4)


async def compute_accuracy(generated: str, expected: str) -> float:
    g_emb = await ollama_client.embeddings(generated)
    e_emb = await ollama_client.embeddings(expected)
    return round(_cosine_similarity(g_emb, e_emb), 4)
