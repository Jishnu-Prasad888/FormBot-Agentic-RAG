from typing import Any, Optional
from app.chromadb.client import chroma_client
from app.embeddings.ollama_client import ollama_client
from app.rag.metadata_filter import build_chroma_filter
from app.core.logging import get_logger

logger = get_logger("vector_rag")


class VectorRAG:
    async def retrieve(
        self,
        query: str,
        collection_name: str = "text_documents",
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        logger.info(f"VectorRAG retrieve: query='{query[:60]}' collection='{collection_name}'")
        query_embedding = await ollama_client.embeddings(query)
        where = build_chroma_filter(filters) if filters else None
        results = chroma_client.search(collection_name, query_embedding, top_k, where)
        for r in results:
            r["document_id"] = r.get("metadata", {}).get("document_id", "")
            r["filename"] = r.get("metadata", {}).get("filename", "")
        return results

    async def retrieve_multi_collection(
        self,
        query: str,
        collection_names: list[str],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        query_embedding = await ollama_client.embeddings(query)
        where = build_chroma_filter(filters) if filters else None
        all_results = []
        for collection in collection_names:
            try:
                results = chroma_client.search(collection, query_embedding, top_k, where)
                for r in results:
                    r["collection"] = collection
                    r["document_id"] = r.get("metadata", {}).get("document_id", "")
                    r["filename"] = r.get("metadata", {}).get("filename", "")
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Collection '{collection}' search failed: {e}")
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]


vector_rag = VectorRAG()
