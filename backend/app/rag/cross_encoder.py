from typing import Any
from sentence_transformers import CrossEncoder
from app.core.logging import get_logger

logger = get_logger("cross_encoder")


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)
        logger.info(f"CrossEncoder initialized with {model_name}")

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank chunks using cross-encoder scoring."""
        if not chunks:
            return []

        chunk_texts = [c["chunk_text"] for c in chunks]
        
        # Score all chunks
        scores = self.model.predict([[query, text] for text in chunk_texts])
        
        # Add scores and sort
        for i, chunk in enumerate(chunks):
            chunk["ce_score"] = float(scores[i])
        
        ranked = sorted(chunks, key=lambda x: x["ce_score"], reverse=True)[:top_k]
        logger.info(f"Reranked {len(chunks)} chunks, selected top {len(ranked)}")
        return ranked


cross_encoder = CrossEncoderReranker()
