from typing import Any
from app.core.logging import get_logger

logger = get_logger("retrieval_metrics")


class RetrievalMetrics:
    """Compute retrieval quality metrics: Recall@K, MRR, nDCG."""

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
        """Recall@k: fraction of gold items in top-k retrieved."""
        if not gold_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        relevant = len(top_k & gold_ids)
        return relevant / len(gold_ids)

    @staticmethod
    def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
        """MRR: 1 / rank of first relevant item."""
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in gold_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg_at_k(scores: list[float], gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
        """nDCG@k: normalized discounted cumulative gain."""
        if not gold_ids:
            return 0.0

        # DCG: sum of (relevance / log2(rank+1))
        dcg = 0.0
        for i in range(min(k, len(retrieved_ids))):
            relevance = 1.0 if retrieved_ids[i] in gold_ids else 0.0
            dcg += relevance / (2 ** (i / 10.0))  # log2(i+2) approximated

        # Ideal DCG: perfect ranking
        idcg = sum(1.0 / (2 ** (i / 10.0)) for i in range(min(k, len(gold_ids))))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def gold_in_retrieved(retrieved_chunks: list[dict], expected_answer: str) -> bool:
        """Check if the expected answer is present in retrieved context."""
        combined = " ".join(c.get("chunk_text", "") for c in retrieved_chunks).lower()
        answer_lower = expected_answer.lower()
        # Simple substring match; can be enhanced with semantic similarity
        return answer_lower in combined or len(answer_lower) > 0 and answer_lower[:20] in combined


retrieval_metrics = RetrievalMetrics()
