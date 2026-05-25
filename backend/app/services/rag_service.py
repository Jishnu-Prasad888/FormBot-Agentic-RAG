import time
import uuid
from typing import Any, AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.rag.vector_rag import vector_rag
from app.rag.hybrid_rag import hybrid_rag
from app.rag.bm25 import bm25_retriever
from app.rag.table_rag import table_rag
from app.rag.pdf_rag import pdf_rag
from app.rag.markdown_rag import markdown_rag
from app.rag.metadata_filter import filter_results
from app.rag.evaluator import (
    compute_accuracy, compute_faithfulness, compute_answer_relevancy,
    compute_context_precision, compute_context_recall,
)
from app.embeddings.openai_client import openai_client as ollama_client
from app.repositories.log_repository import log_repo
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("rag_service")

RAG_SYSTEM = """You are an expert assistant. Answer the question using only the provided context.
Be factual, concise, and cite sources. If the answer is not in the context, say 'Not found in available documents'."""


class RAGService:
    async def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
        filters: Optional[dict] = None,
        collection_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        col = collection_name or "text_documents"
        if strategy == "vector":
            return await vector_rag.retrieve(query, col, top_k, filters)
        elif strategy == "bm25":
            results = bm25_retriever.search(col, query, top_k)
            return filter_results(results, filters or {})
        elif strategy == "hybrid":
            return await hybrid_rag.retrieve(query, col, top_k, filters)
        elif strategy == "table":
            return await table_rag.query(query, top_k=top_k)
        elif strategy == "pdf":
            return await pdf_rag.query(query, top_k=top_k)
        elif strategy == "markdown":
            return await markdown_rag.query(query, top_k=top_k)
        else:
            return await hybrid_rag.retrieve(query, col, top_k, filters)

    async def query(
        self,
        db: AsyncSession,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> dict[str, Any]:
        start = time.time()
        chunks = await self.retrieve(query, strategy, top_k, filters)
        context = "\n\n".join(
            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
        latency = (time.time() - start) * 1000

        sources = [
            {"filename": r.get("filename", ""), "chunk_id": r.get("chunk_id", ""), "score": r.get("score", 0)}
            for r in chunks
        ]

        await log_repo.create_retrieval_log(db, {
            "id": str(uuid.uuid4()),
            "query": query,
            "retrieval_strategy": strategy,
            "retrieved_chunks": [r.get("chunk_id", "") for r in chunks],
            "generated_answer": answer,
            "latency_ms": latency,
            "agent_used": "rag_service",
        })

        confidence = round(sum(r.get("score", 0) for r in chunks) / max(len(chunks), 1), 4)
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "strategy": strategy,
            "latency_ms": round(latency, 2),
            "confidence": confidence,
        }

    async def query_stream(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        chunks = await self.retrieve(query, strategy, top_k, filters)
        context = "\n\n".join(
            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        async for token in ollama_client.generate_stream(prompt, system=RAG_SYSTEM):
            yield token

    async def evaluate(
        self,
        db: AsyncSession,
        questions: list[dict],
        dataset_name: str = "default",
    ) -> dict[str, Any]:
        results = {
            "accuracy": [], "faithfulness": [], "context_precision": [],
            "context_recall": [], "answer_relevancy": [], "latency_ms": [], "failed": [],
        }

        for q in questions:
            question = q["question"]
            expected = q["expected_answer"]
            try:
                start = time.time()
                chunks = await self.retrieve(question, strategy="hybrid", top_k=5)
                context_texts = [r["chunk_text"] for r in chunks]
                context = "\n\n".join(context_texts)
                prompt = f"Context:\n{context}\n\nQuestion: {question}"
                answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
                latency = (time.time() - start) * 1000

                acc = await compute_accuracy(answer, expected)
                faith = await compute_faithfulness(answer, context_texts)
                cp = await compute_context_precision(question, context_texts)
                cr = await compute_context_recall(expected, context_texts)
                ar = await compute_answer_relevancy(question, answer)

                results["accuracy"].append(acc)
                results["faithfulness"].append(faith)
                results["context_precision"].append(cp)
                results["context_recall"].append(cr)
                results["answer_relevancy"].append(ar)
                results["latency_ms"].append(latency)
            except Exception as e:
                logger.error(f"Eval failed for '{question}': {e}")
                results["failed"].append({"question": question, "error": str(e)})

        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

        final = {
            "accuracy": avg(results["accuracy"]),
            "faithfulness": avg(results["faithfulness"]),
            "context_precision": avg(results["context_precision"]),
            "context_recall": avg(results["context_recall"]),
            "answer_relevancy": avg(results["answer_relevancy"]),
            "latency_avg_ms": avg(results["latency_ms"]),
            "failed_questions": results["failed"],
        }

        await log_repo.create_evaluation_run(db, {
            "id": str(uuid.uuid4()),
            "dataset_name": dataset_name,
            "accuracy": final["accuracy"],
            "faithfulness": final["faithfulness"],
            "context_precision": final["context_precision"],
            "context_recall": final["context_recall"],
        })

        return final


rag_service = RAGService()