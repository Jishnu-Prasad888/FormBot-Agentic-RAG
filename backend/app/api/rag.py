"""
POST /api/rag/evaluate

Accepts a list of {question, expected_answer} pairs, runs each through the
multi-agent RAG pipeline (coordinator → evaluator), and returns per-question
detail alongside aggregate metrics.
"""

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.dependencies import get_db
from app.services.rag_service import rag_service
from app.evaluation.agent_runner import evaluate_question, failed_question_row
from app.core.logging import get_logger

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
    top_k: int = 5
    use_query_expansion: bool = False
    num_expansions: int = 2


@router.post("/evaluate")
async def evaluate_rag(req: EvaluateRequest, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    per_question: list[dict] = []
    failed: list[dict] = []
    latencies: list[float] = []

    for qa in req.questions:
        try:
            row = await evaluate_question(
                qa.question, 
                qa.expected_answer, 
                req.top_k,
                req.use_query_expansion,
                req.num_expansions
            )
            per_question.append(row)
            latencies.append(row["latency_ms"])
        except Exception as exc:
            logger.error(f"Eval error for '{qa.question[:60]}': {exc}")
            failed.append({"question": qa.question, "error": str(exc)})
            per_question.append(failed_question_row(qa.question, qa.expected_answer, str(exc)))

    succeeded = [r for r in per_question if not r.get("error")]

    def _avg(k: str) -> float:
        if not succeeded:
            return 0.0
        return round(sum(r.get(k, 0.0) for r in succeeded) / len(succeeded), 4)

    return {
        # LLM-as-judge metrics
        "accuracy_llm":      _avg("accuracy_llm"),
        "accuracy":          _avg("accuracy_llm"),  # Backward compat
        "faithfulness":      _avg("faithfulness"),
        "context_precision": _avg("context_precision"),
        "context_recall":    _avg("context_recall"),
        "answer_relevancy":  _avg("answer_relevancy"),
        # Accuracy methods
        "exact_match":       _avg("exact_match"),
        "semantic_similarity": _avg("semantic_similarity"),
        "f1":                _avg("f1"),
        "accuracy_combined": _avg("accuracy_combined"),
        # Retrieval metrics
        "recall_10":         _avg("recall_10"),
        "recall_20":         _avg("recall_20"),
        "recall_50":         _avg("recall_50"),
        "mrr":               _avg("mrr"),
        "ndcg_10":           _avg("ndcg_10"),
        # Meta
        "latency_avg_ms":    round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "failed_questions":  failed,
        "per_question":      per_question,
        "dataset_name":      req.dataset_name,
    }
