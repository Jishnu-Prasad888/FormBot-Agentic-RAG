from typing import Any, Optional
from app.rag.vector_rag import vector_rag
from app.rag.bm25 import bm25_retriever
from app.rag.rrf import reciprocal_rank_fusion
from app.rag.metadata_filter import filter_results
from app.core.logging import get_logger

logger = get_logger("hybrid_rag")


class HybridRAG:
    async def retrieve(
        self,
        query: str,
        collection_name: str = "text_documents",
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        logger.info(f"HybridRAG retrieve: query='{query[:60]}'")

        # Vector retrieval
        vector_results = []
        try:
            vector_results = await vector_rag.retrieve(query, collection_name, top_k * 2, filters)
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")

        # BM25 retrieval
        bm25_results = []
        try:
            bm25_raw = bm25_retriever.search(collection_name, query, top_k * 2)
            bm25_results = filter_results(bm25_raw, filters or {})
        except Exception as e:
            logger.warning(f"BM25 retrieval failed: {e}")

        if not vector_results and not bm25_results:
            return []

        fused = reciprocal_rank_fusion([vector_results, bm25_results], top_k=top_k)
        return fused


hybrid_rag = HybridRAG()
