from pydantic import BaseModel, Field
from typing import Any, Optional


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: Optional[dict[str, Any]] = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None


class AgentResponse(BaseModel):
    agent: str
    query: str
    answer: str
    sources: list[Any] = []
    reasoning: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Optional[dict[str, Any]] = None


class CoordinatorRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)
