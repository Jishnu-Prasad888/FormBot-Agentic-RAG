import time
from typing import Any, Optional
from app.agents.base import BaseAgent
from app.rag.table_rag import table_rag
from app.embeddings.openai_client import openai_client as ollama_client


class SQLiteAgent(BaseAgent):
    name = "sqlite_agent"

    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
        ctx = context or {}
        return {
            "query": query,
            "document_id": ctx.get("document_id"),
            "top_k": ctx.get("top_k", 5),
        }

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        query = plan["query"]
        doc_id = plan.get("document_id")
        top_k = plan["top_k"]

        chunks = await table_rag.query(query, document_id=doc_id, top_k=top_k)
        context_str = "\n\n".join(r["chunk_text"] for r in chunks)

        system = """You are a data analyst. Answer structured data questions using the provided table context.
Be precise with numbers, names, and values. Format answers clearly."""
        prompt = f"Table data:\n{context_str}\n\nQuestion: {query}"
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
            {"filename": c.get("metadata", {}).get("filename", ""), "chunk_id": c.get("chunk_id", "")}
            for c in chunks
        ]
        return result


sqlite_agent = SQLiteAgent()