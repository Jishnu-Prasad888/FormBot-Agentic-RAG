from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import json

from app.core.dependencies import get_db
from app.rag.table_rag import table_rag
from app.services.document_service import document_service
from app.schemas.search import SearchResult
from app.core.logging import get_logger

router = APIRouter(prefix="/api/tablerag", tags=["TableRAG"])
logger = get_logger("api.tablerag")


@router.post("/index")
async def index_table(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    content = await file.read()
    extra_meta = json.loads(metadata) if metadata else {}
    result = await document_service.upload_and_index(db, file.filename, content, extra_meta)
    doc = result["document"]
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "chunk_count": result["chunk_count"],
        "message": "Table indexed successfully",
    }


@router.post("/query", response_model=list[SearchResult])
async def query_table(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
):
    chunks = await table_rag.query(query, document_id=document_id, top_k=top_k)
    return [
        SearchResult(
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("metadata", {}).get("document_id", ""),
            filename=r.get("metadata", {}).get("filename", ""),
            chunk_text=r.get("chunk_text", ""),
            score=r.get("score", 0.0),
            metadata=r.get("metadata", {}),
        )
        for r in chunks
    ]


@router.get("/schema/{document_id}")
async def get_table_schema(document_id: str):
    schema = await table_rag.get_schema(document_id)
    if not schema:
        return {"document_id": document_id, "schema": None, "message": "Schema not found"}
    return {"document_id": document_id, "schema": schema}
