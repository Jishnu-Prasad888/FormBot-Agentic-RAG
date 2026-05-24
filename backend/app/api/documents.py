from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import json

from app.core.dependencies import get_db
from app.services.document_service import document_service
from app.repositories.document_repository import document_repo
from app.schemas.document import DocumentResponse, DocumentListResponse, ChunkResponse, ReindexResponse
from app.core.exceptions import DocumentNotFoundError
from app.core.logging import get_logger

router = APIRouter(prefix="/api/documents", tags=["Documents"])
logger = get_logger("api.documents")


@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    content = await file.read()
    extra_meta = json.loads(metadata) if metadata else {}
    result = await document_service.upload_and_index(db, file.filename, content, extra_meta)
    doc = result["document"]
    return {
        "id": doc.id,
        "filename": doc.filename,
        "document_type": doc.document_type,
        "retrieval_strategy": doc.retrieval_strategy,
        "chunk_count": result["chunk_count"],
        "message": "Document uploaded and indexed successfully",
    }


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    docs = await document_repo.list_all(db, skip, limit)
    total = await document_repo.count(db)
    return {"documents": docs, "total": total}


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = await document_repo.get_by_id(db, doc_id)
    if not doc:
        raise DocumentNotFoundError(doc_id)
    return doc


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    await document_service.delete(db, doc_id)
    return {"message": f"Document {doc_id} deleted"}


@router.post("/{doc_id}/reindex", response_model=ReindexResponse)
async def reindex_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    result = await document_service.reindex(db, doc_id)
    return result


@router.get("/{doc_id}/chunks", response_model=list[ChunkResponse])
async def get_document_chunks(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = await document_repo.get_by_id(db, doc_id)
    if not doc:
        raise DocumentNotFoundError(doc_id)
    chunks = await document_repo.get_chunks(db, doc_id)
    return chunks


@router.get("/{doc_id}/metadata")
async def get_document_metadata(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = await document_repo.get_by_id(db, doc_id)
    if not doc:
        raise DocumentNotFoundError(doc_id)
    return {
        "id": doc.id,
        "filename": doc.filename,
        "document_type": doc.document_type,
        "retrieval_strategy": doc.retrieval_strategy,
        "language": doc.language,
        "collection_name": doc.collection_name,
        "embedding_model": doc.embedding_model,
        "chunk_count": doc.chunk_count,
        "metadata_json": doc.metadata_json,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }
