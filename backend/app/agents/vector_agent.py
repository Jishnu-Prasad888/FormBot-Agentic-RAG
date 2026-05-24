import time
from typing import Any, Optional
from app.agents.base import BaseAgent
from app.rag.hybrid_rag import hybrid_rag
from app.rag.vector_rag import vector_rag
from app.embeddings.ollama_client import ollama_client
from app.core.config import settings


class VectorRetrievalAgent(BaseAgent):
    name = "vector_agent"

    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
        ctx = context or {}
        return {
            "query": query,
            "expanded_query": query,
            "strategy": ctx.get("strategy", "hybrid"),
            "top_k": ctx.get("top_k", settings.TOP_K),
            "filters": ctx.get("filters"),
            "collection": ctx.get("collection", "text_documents"),
        }

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        query = plan["query"]
        strategy = plan["strategy"]
        top_k = plan["top_k"]
        filters = plan.get("filters")
        collection = plan["collection"]

        if strategy == "vector":
            chunks = await vector_rag.retrieve(query, collection, top_k, filters)
        else:
            chunks = await hybrid_rag.retrieve(query, collection, top_k, filters)

        context_str = "\n\n".join(
            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
        )
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer based only on context:"
        answer = await ollama_client.generate(prompt)
        latency = (time.time() - start) * 1000

        return {
            "agent": self.name,
            "query": query,
            "answer": answer,
            "chunks": chunks,
            "latency_ms": round(latency, 2),
        }

    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
        chunks = result.get("chunks", [])
        avg_score = sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1)
        result["confidence"] = round(avg_score, 4)
        result["sources"] = [
            {"filename": c.get("filename", ""), "chunk_id": c.get("chunk_id", ""), "score": c.get("score", 0)}
            for c in chunks
        ]
        return result


vector_agent = VectorRetrievalAgent()
