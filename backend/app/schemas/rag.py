from pydantic import BaseModel, Field
from typing import Any, Optional


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    strategy: str = Field(default="hybrid", description="vector|bm25|hybrid|table|pdf|markdown")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None
    conversation_id: Optional[str] = None


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Any] = []
    strategy: str = ""
    latency_ms: float = 0.0
    confidence: float = 0.0


class RAGRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    strategy: str = Field(default="hybrid")
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None


class EvalQuestion(BaseModel):
    question: str
    expected_answer: str


class EvaluationRequest(BaseModel):
    questions: list[EvalQuestion]
    dataset_name: Optional[str] = "default"


class EvaluationResponse(BaseModel):
    accuracy: float = 0.0
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    latency_avg_ms: float = 0.0
    failed_questions: list[dict[str, Any]] = []
