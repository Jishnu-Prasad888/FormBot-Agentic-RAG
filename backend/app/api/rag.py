from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.services.rag_service import rag_service
from app.schemas.rag import (
    RAGQueryRequest, RAGQueryResponse, RAGRetrieveRequest,
    EvaluationRequest, EvaluationResponse
)
from app.schemas.search import SearchResult
from app.core.logging import get_logger

router = APIRouter(prefix="/api/rag", tags=["RAG"])
logger = get_logger("api.rag")


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


@router.post("/evaluate", response_model=EvaluationResponse)
async def rag_evaluate(req: EvaluationRequest, db: AsyncSession = Depends(get_db)):
    questions = [q.model_dump() for q in req.questions]
    result = await rag_service.evaluate(db, questions, req.dataset_name or "default")
    return result
