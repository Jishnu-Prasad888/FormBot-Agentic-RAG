import time
from typing import Any, Optional
from app.agents.base import BaseAgent
from app.rag.pdf_rag import pdf_rag
from app.rag.markdown_rag import markdown_rag
from app.rag.table_rag import table_rag
from app.rag.vector_rag import vector_rag
from app.embeddings.ollama_client import ollama_client


ROUTING_RULES = {
    "pdf": ["pdf", "document", "page", "report", "form"],
    "markdown": ["markdown", "readme", "guide", "documentation", "wiki"],
    "csv": ["table", "csv", "data", "rows", "columns", "spreadsheet", "excel"],
    "text": [],  # default
}


def _detect_doc_type(query: str) -> str:
    q_lower = query.lower()
    for doc_type, keywords in ROUTING_RULES.items():
        if any(kw in q_lower for kw in keywords):
            return doc_type
    return "text"


class DocumentRouterAgent(BaseAgent):
    name = "router_agent"

    async def plan(self, query: str, context: Optional[dict] = None) -> dict[str, Any]:
        ctx = context or {}
        doc_type = ctx.get("doc_type") or _detect_doc_type(query)
        return {
            "query": query,
            "doc_type": doc_type,
            "document_id": ctx.get("document_id"),
            "top_k": ctx.get("top_k", 5),
        }

    async def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        query = plan["query"]
        doc_type = plan["doc_type"]
        doc_id = plan.get("document_id")
        top_k = plan["top_k"]

        if doc_type == "pdf":
            chunks = await pdf_rag.query(query, document_id=doc_id, top_k=top_k)
            strategy = "hierarchical_rag"
        elif doc_type == "markdown":
            chunks = await markdown_rag.query(query, document_id=doc_id, top_k=top_k)
            strategy = "structure_aware_rag"
        elif doc_type == "csv":
            chunks = await table_rag.query(query, document_id=doc_id, top_k=top_k)
            strategy = "table_rag"
        else:
            chunks = await vector_rag.retrieve(query, "text_documents", top_k)
            strategy = "vector_rag"

        context_str = "\n\n".join(r["chunk_text"] for r in chunks)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
        answer = await ollama_client.generate(prompt)
        latency = (time.time() - start) * 1000

        return {
            "agent": self.name,
            "query": query,
            "answer": answer,
            "chunks": chunks,
            "strategy": strategy,
            "doc_type": doc_type,
            "latency_ms": round(latency, 2),
        }

    async def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
        chunks = result.get("chunks", [])
        result["confidence"] = round(sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1), 4)
        result["sources"] = [
            {"filename": c.get("metadata", {}).get("filename", c.get("filename", "")), "chunk_id": c.get("chunk_id", "")}
            for c in chunks
        ]
        return result


router_agent = DocumentRouterAgent()
