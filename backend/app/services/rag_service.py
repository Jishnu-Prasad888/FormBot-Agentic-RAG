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
from app.rag.multi_collection_retrieval import multi_collection_retriever
from app.rag.metadata_filter import filter_results
from app.rag.evaluator import (
    compute_accuracy, compute_faithfulness, compute_answer_relevancy,
    compute_context_precision, compute_context_recall,
)
from app.embeddings.openai_client import openai_client as ollama_client
from app.repositories.log_repository import log_repo
from app.core.config import settings
from app.services.elasticsearch_service import es_service
from app.core.evaluation_logger import EvaluationLogger
from app.services.graph_service import graph_service


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
        all_collections: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Retrieve from single or multiple collections.
        If all_collections=True, search across all available collections.
        """
        candidate_ids = None
        if settings.USE_KG_RETRIEVAL:
            graph_result = await graph_service.get_candidates(query, filters)
            candidate_ids = graph_result.candidate_document_ids or None

        if all_collections:
            return await multi_collection_retriever.retrieve_all_collections(
                query, strategy, top_k, filters
            )

        col = collection_name or "text_documents"
        if strategy == "vector":
            return await vector_rag.retrieve(query, col, top_k, filters, False, candidate_ids)
        elif strategy == "bm25":
            results = bm25_retriever.search(col, query, top_k)
            return filter_results(results, filters or {}, candidate_ids)
        elif strategy == "hybrid":
            return await hybrid_rag.retrieve(query, col, top_k, filters, candidate_ids)
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
        graph_context = ""
        if settings.USE_KG_RETRIEVAL:
            try:
                graph_result = await graph_service.get_candidates(query, filters)
                if graph_result.forms:
                    form_lines = "\n".join([f"- {f.get('name')}" for f in graph_result.forms if f.get("name")])
                    graph_context = f"Forms:\n{form_lines}\n"
            except Exception:
                graph_context = ""

        chunks = await self.retrieve(query, strategy, top_k, filters)
        context = "\n\n".join(
            f"[{r.get('filename','?')}] {r['chunk_text']}" for r in chunks
        )
        if graph_context:
            context = graph_context + "\n" + context
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

        try:
            await log_repo.create_query_log(db, {
                "id": str(uuid.uuid4()),
                "query": query,
                "response": answer,
                "latency": latency,
            })
        except Exception:
            pass

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
        logger = EvaluationLogger(f"{dataset_name}_{int(time.time())}")
        logger.log("EVALUATION_START", f"Dataset: {dataset_name}, Total questions: {len(questions)}")
        
        results = {
            "accuracy": [], "faithfulness": [], "context_precision": [],
            "context_recall": [], "answer_relevancy": [], "latency_ms": [], "failed": [],
        }

        for q_idx, q in enumerate(questions, 1):
            question = q["question"]
            expected = q["expected_answer"]
            
            logger.log_question(question, q_idx, len(questions))
            
            try:
                start = time.time()
                
                # Retrieve
                chunks = await self.retrieve(question, strategy="hybrid", top_k=settings.TOP_K)
                context_texts = [r["chunk_text"] for r in chunks]
                logger.log_retrieval("hybrid", settings.TOP_K, chunks)
                
                # Enhance with Elasticsearch iterative queries
                original_count = len(context_texts)
                enhanced_texts = await es_service.enhance_with_iterative_query(context_texts, question, max_tries=5, logger=logger)
                logger.log_es_enhancement(len(enhanced_texts), original_count)
                
                context = "\n\n".join(enhanced_texts)
                prompt = f"Context:\n{context}\n\nQuestion: {question}"
                
                llm_start = time.time()
                answer = await ollama_client.generate(prompt, system=RAG_SYSTEM)
                llm_latency = (time.time() - llm_start) * 1000
                logger.log_llm_call("Generate Answer", prompt, answer, llm_latency)
                
                latency = (time.time() - start) * 1000

                acc_score, acc_rat = await compute_accuracy(answer, expected)
                logger.log_metrics("Accuracy", acc_score, acc_rat)
                
                faith_score, faith_rat = await compute_faithfulness(answer, enhanced_texts)
                logger.log_metrics("Faithfulness", faith_score, faith_rat)
                
                cp_score, cp_rat = await compute_context_precision(question, enhanced_texts)
                logger.log_metrics("Context Precision", cp_score, cp_rat)
                
                cr_score, cr_rat = await compute_context_recall(expected, enhanced_texts)
                logger.log_metrics("Context Recall", cr_score, cr_rat)
                
                ar_score, ar_rat = await compute_answer_relevancy(question, answer)
                logger.log_metrics("Answer Relevancy", ar_score, ar_rat)

                results["accuracy"].append(acc_score)
                results["faithfulness"].append(faith_score)
                results["context_precision"].append(cp_score)
                results["context_recall"].append(cr_score)
                results["answer_relevancy"].append(ar_score)
                results["latency_ms"].append(latency)
            except Exception as e:
                logger.log_error(str(e), question)
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
        
        logger.log_summary(final)

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
