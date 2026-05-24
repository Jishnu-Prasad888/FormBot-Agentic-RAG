from pydantic import BaseModel, Field
from typing import Optional


class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: Optional[str] = None


class EmbeddingBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1)
    model: Optional[str] = None


class EmbeddingResponse(BaseModel):
    text: str
    embedding: list[float]
    model: str
    dimensions: int


class EmbeddingBatchResponse(BaseModel):
    embeddings: list[EmbeddingResponse]
    model: str


class EmbeddingModelInfo(BaseModel):
    name: str
    dimensions: Optional[int] = None
