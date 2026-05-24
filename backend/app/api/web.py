from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.services.web_service import web_service
from app.schemas.web import WebIngestRequest, WebQueryRequest, WebIngestResponse
from app.schemas.search import SearchResult
from app.core.logging import get_logger

router = APIRouter(prefix="/api/web", tags=["Web Ingestion"])
logger = get_logger("api.web")


@router.post("/ingest", response_model=WebIngestResponse)
async def ingest_url(req: WebIngestRequest):
    result = await web_service.ingest(
        url=req.url,
        collection_name=req.collection_name or "web_documents",
        metadata=req.metadata,
    )
    return result


@router.post("/query", response_model=list[SearchResult])
async def query_web(req: WebQueryRequest):
    chunks = await web_service.query(req.query, url=req.url, top_k=req.top_k)
    return [
        SearchResult(
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("metadata", {}).get("document_id", ""),
            filename=r.get("metadata", {}).get("source_url", ""),
            chunk_text=r.get("chunk_text", ""),
            score=r.get("score", 0.0),
            metadata=r.get("metadata", {}),
        )
        for r in chunks
    ]
