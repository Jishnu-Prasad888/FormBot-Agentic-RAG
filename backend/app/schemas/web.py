from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Any


class WebIngestRequest(BaseModel):
    url: str = Field(..., description="URL to ingest")
    collection_name: Optional[str] = "web_documents"
    metadata: Optional[dict[str, Any]] = None


class WebQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    url: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)


class WebIngestResponse(BaseModel):
    url: str
    document_id: str
    chunk_count: int
    message: str
