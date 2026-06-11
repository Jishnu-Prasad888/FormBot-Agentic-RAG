from typing import Any, Optional
from app.rag.vector_rag import vector_rag
from app.rag.hybrid_rag import hybrid_rag
from app.rag.table_rag import table_rag
from app.rag.pdf_rag import pdf_rag
from app.rag.markdown_rag import markdown_rag


DEFAULT_COLLECTIONS = [
    "text_documents",
    "pdf_documents",
    "table_documents",
    "markdown_documents",
    "audio_transcripts",
    "web_documents",
]


class MultiCollectionRetriever:
    """Retrieve from all collections and merge results."""

    async def retrieve_all_collections(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
        filters: Optional[dict] = None,
        collections: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve from multiple collections and deduplicate results by document_id + chunk_id.
        Rank by score, return top_k.
        """
        target_collections = collections or DEFAULT_COLLECTIONS
        all_results = []

        for collection in target_collections:
            try:
                if strategy == "hybrid":
                    results = await hybrid_rag.retrieve(query, collection, top_k * 3, filters)
                elif strategy == "vector":
                    results = await vector_rag.retrieve(query, collection, top_k * 3, filters)
                elif strategy == "table":
                    results = await table_rag.query(query, top_k=top_k * 3)
                elif strategy == "pdf":
                    results = await pdf_rag.query(query, top_k=top_k * 3)
                elif strategy == "markdown":
                    results = await markdown_rag.query(query, top_k=top_k * 3)
                else:
                    results = []

                for r in results:
                    r["source_collection"] = collection

                all_results.extend(results)
            except Exception as e:
                continue

        if not all_results:
            return []

        # Deduplicate by chunk_id
        seen = set()
        deduped = []
        for r in all_results:
            chunk_id = r.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                deduped.append(r)
            elif not chunk_id:
                deduped.append(r)

        # Sort by score
        ranked = sorted(deduped, key=lambda x: x.get("score", 0), reverse=True)
        return ranked[:top_k]


multi_collection_retriever = MultiCollectionRetriever()
