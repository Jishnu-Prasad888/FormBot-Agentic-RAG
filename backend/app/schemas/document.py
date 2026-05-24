from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime


class DocumentResponse(BaseModel):
    id: str
    filename: str
    filepath: str
    document_type: str
    retrieval_strategy: Optional[str] = None
    language: Optional[str] = "en"
    chunk_count: int = 0
    embedding_model: Optional[str] = None
    collection_name: Optional[str] = None
    metadata_json: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    chunk_text: str
    chunk_metadata: Optional[dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class ReindexResponse(BaseModel):
    document_id: str
    message: str
    chunk_count: int
