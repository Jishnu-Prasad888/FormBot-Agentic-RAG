import time
from typing import Any, Optional
from app.agents.base import BaseAgent
from app.services.web_service import web_service
from app.embeddings.openai_client import openai_client as ollama_client


class WebEnrichmentAgent(BaseAgent):
    name = "web_agent"

    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
        ctx = context or {}
        return {
            "query": query,
            "url": ctx.get("url"),
            "top_k": ctx.get("top_k", 5),
        }

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        query = plan["query"]
        url = plan.get("url")
        top_k = plan["top_k"]

        chunks = await web_service.query(query, url=url, top_k=top_k)

        if not chunks and url:
            # Try to ingest on-demand
            try:
                await web_service.ingest(url)
                chunks = await web_service.query(query, url=url, top_k=top_k)
            except Exception as e:

        context_str = "\n\n".join(r["chunk_text"] for r in chunks) if chunks else "No web context available."
        system = "You are a research assistant. Summarize and answer based on web content. Always note the source URL."
        prompt = f"Web content:\n{context_str}\n\nQuestion: {query}"
        answer = await ollama_client.generate(prompt, system=system)
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
        result["confidence"] = round(sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1), 4)
        result["sources"] = [
            {"url": c.get("metadata", {}).get("source_url", ""), "chunk_id": c.get("chunk_id", "")}
            for c in chunks
        ]
        return result


web_agent = WebEnrichmentAgent()