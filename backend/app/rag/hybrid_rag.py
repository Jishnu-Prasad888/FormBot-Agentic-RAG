from typing import Any, Optional
from app.rag.vector_rag import vector_rag
from app.rag.bm25 import bm25_retriever
from app.rag.metadata_filter import filter_results
from app.rag.parent_context import parent_context_expander
from app.rag.cross_encoder import cross_encoder
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("hybrid_rag")


class HybridRAG:
    def _normalize_scores(self, results: list[dict]) -> list[dict]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        max_score = max(r.get("score", 0) for r in results) or 1.0
        min_score = min(r.get("score", 0) for r in results) or 0.0
        range_val = max_score - min_score or 1.0
        for r in results:
            r["normalized_score"] = (r.get("score", 0) - min_score) / range_val
        return results

    def _merge_by_weighted_score(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Merge dense and BM25 results using weighted scoring."""
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        bm25_weight = settings.BM25_WEIGHT
        dense_weight = settings.DENSE_WEIGHT
        
        # Build combined map by chunk_id
        merged = {}
        for r in vector_results:
            chunk_id = r.get("chunk_id")
            merged[chunk_id] = {
                **r,
                "dense_score": r.get("normalized_score", 0),
                "bm25_score": 0,
                "combined_score": dense_weight * r.get("normalized_score", 0),
            }
        
        for r in bm25_results:
            chunk_id = r.get("chunk_id")
            if chunk_id in merged:
                merged[chunk_id]["bm25_score"] = r.get("normalized_score", 0)
                merged[chunk_id]["combined_score"] = (
                    dense_weight * merged[chunk_id]["dense_score"] +
                    bm25_weight * r.get("normalized_score", 0)
                )
            else:
                merged[chunk_id] = {
                    **r,
                    "dense_score": 0,
                    "bm25_score": r.get("normalized_score", 0),
                    "combined_score": bm25_weight * r.get("normalized_score", 0),
                }
        
        # Sort by combined score and return top_k
        ranked = sorted(merged.values(), key=lambda x: x.get("combined_score", 0), reverse=True)
        return ranked[:top_k]

    async def retrieve(
        self,
        query: str,
        collection_name: str = "text_documents",
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        logger.info(f"HybridRAG retrieve: query='{query[:60]}' top_k={top_k}")

        # Dense retrieval: top 50
        vector_results = []
        try:
            vector_results = await vector_rag.retrieve(
                query, collection_name, settings.DENSE_TOP_K, filters
            )
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")

        # BM25 retrieval: top 50
        bm25_results = []
        try:
            bm25_raw = bm25_retriever.search(collection_name, query, settings.BM25_TOP_K)
            bm25_results = filter_results(bm25_raw, filters or {})
        except Exception as e:
            logger.warning(f"BM25 retrieval failed: {e}")

        if not vector_results and not bm25_results:
            return []

        # Merge with weighted scoring
        merged = self._merge_by_weighted_score(vector_results, bm25_results, top_k * 2)
        
        # Rerank
        reranked = cross_encoder.rerank(query, merged, top_k=settings.RERANK_TOP_K)
        
        # Expand with parent context
        expanded = await parent_context_expander.expand(reranked, collection_name)
        
        return expanded


hybrid_rag = HybridRAG()
