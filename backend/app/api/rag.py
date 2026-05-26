"""
POST /api/rag/evaluate

Accepts a list of {question, expected_answer} pairs, runs each through the
RAG pipeline, scores with the LLM-as-judge evaluator, and returns per-question
detail alongside aggregate metrics.
"""

import time
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.services.rag_service import rag_service
from app.embeddings.openai_client import openai_client as ollama_client
from app.evaluation.evaluator import evaluate_single   # ← new LLM-judge module
from app.core.logging import get_logger
from pydantic import BaseModel

from app.schemas.rag import RAGQueryRequest, RAGQueryResponse, RAGRetrieveRequest
from app.schemas.search import SearchResult

logger = get_logger("api.rag.evaluate")
router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest, db: AsyncSession = Depends(get_db)):
    result = await rag_service.query(db, req.query, req.strategy, req.top_k, req.filters)
    return result


@router.post("/query/stream")
async def rag_query_stream(req: RAGQueryRequest):
    async def generator():
        async for token in rag_service.query_stream(req.query, req.strategy, req.top_k, req.filters):
            yield token

    return StreamingResponse(generator(), media_type="text/plain")


@router.post("/retrieve", response_model=list[SearchResult])
async def rag_retrieve(req: RAGRetrieveRequest):
    chunks = await rag_service.retrieve(req.query, req.strategy, req.top_k, req.filters)
    results = []
    for r in chunks:
        meta = r.get("metadata", {})
        results.append(SearchResult(
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("document_id") or meta.get("document_id", ""),
            filename=r.get("filename") or meta.get("filename", ""),
            chunk_text=r.get("chunk_text", ""),
            score=r.get("score", 0.0),
            metadata=meta,
        ))
    return results


class EvalQuestion(BaseModel):
    question: str
    expected_answer: str


class EvaluateRequest(BaseModel):
    questions: list[EvalQuestion]
    dataset_name: str = "eval_run"


@router.post("/evaluate")
async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    per_question: list[dict] = []
    failed: list[dict] = []
    latencies: list[float] = []

    for qa in req.questions:
        try:
            # 1. Retrieve context
            retrieval_result = await rag_service.retrieve(
                qa.question, strategy="hybrid", top_k=5
            )
            context_chunks = [r["chunk_text"] for r in retrieval_result]

            # 2. Generate answer
            context_str = "\n\n".join(
                f"[Source: {r.get('filename', 'unknown')}]\n{r['chunk_text']}"
                for r in retrieval_result
            )
            messages = [{
                "role": "user",
                "content": (
                    f"Context:\n{context_str}\n\nQuestion: {qa.question}"
                    if context_chunks else qa.question
                ),
            }]
            generated_answer = await ollama_client.chat(
                messages,
                system=(
                    "You are a helpful assistant. Answer the question using the provided "
                    "context. Be concise and accurate."
                ),
            )

            # 3. LLM-as-judge scoring
            row = await evaluate_single(
                question=qa.question,
                expected_answer=qa.expected_answer,
                generated_answer=generated_answer,
                context_chunks=context_chunks,
            )
            per_question.append(row)
            latencies.append(row["latency_ms"])

        except Exception as exc:
            logger.error(f"Eval error for '{qa.question[:60]}': {exc}")
            failed.append({"question": qa.question, "error": str(exc)})
            per_question.append({
                "question": qa.question,
                "expected_answer": qa.expected_answer,
                "generated_answer": "",
                "retrieved_context": "",
                "accuracy": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "accuracy_rationale": "",
                "faithfulness_rationale": "",
                "answer_relevancy_rationale": "",
                "context_precision_rationale": "",
                "context_recall_rationale": "",
                "latency_ms": 0.0,
                "error": str(exc),
            })

    # Aggregate over successful rows only
    succeeded = [r for r in per_question if "error" not in r or not r.get("error")]

    def _avg(k: str) -> float:
        if not succeeded:
            return 0.0
        return round(sum(r.get(k, 0.0) for r in succeeded) / len(succeeded), 4)

    return {
        # Aggregate metrics (for backward compat with existing frontend)
        "accuracy":          _avg("accuracy"),
        "faithfulness":      _avg("faithfulness"),
        "context_precision": _avg("context_precision"),
        "context_recall":    _avg("context_recall"),
        "answer_relevancy":  _avg("answer_relevancy"),
        "latency_avg_ms":    round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "failed_questions":  failed,
        # NEW: per-question detail for the UI
        "per_question":      per_question,
        "dataset_name":      req.dataset_name,
    }