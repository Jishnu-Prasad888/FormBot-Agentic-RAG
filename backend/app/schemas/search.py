from pydantic import BaseModel, Field
from typing import Any, Optional


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None
    collection_name: Optional[str] = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_text: str
    score: float
    metadata: Optional[dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    confidence: float = 0.0
    sources: list[str] = []
    latency_ms: float = 0.0
    strategy: str = ""
