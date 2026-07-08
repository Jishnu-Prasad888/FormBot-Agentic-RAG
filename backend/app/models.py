from pydantic import BaseModel
from typing import Optional


class Document(BaseModel):
    id: str
    filename: str
    filepath: str
    document_type: str
    retrieval_strategy: str = "vector"
    language: str = "en"
    chunk_count: int = 0
    embedding_model: str = ""
    collection_name: str = "default"
    metadata_json: dict = {}
    created_at: str = ""
    updated_at: str = ""


class Chunk(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    chunk_text: str
    chunk_metadata: dict = {}
    created_at: str = ""


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_text: str
    score: float
    metadata: dict = {}


class Source(BaseModel):
    filename: str
    chunk_id: str
    score: float


class Message(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources: list[Source] = []
    created_at: str = ""


class Conversation(BaseModel):
    id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    messages: list[Message] = []


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[dict] = None
    collection_name: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    confidence: float = 0.0
    sources: list[str] = []
    latency_ms: float = 0.0
    strategy: str = "vector"


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    top_k: int = 5


class LiveAskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = 5
    target_language: str = "en"
    voice: str = "alloy"
    speak: bool = False
    use_form_context: bool = True
    manual_context: Optional[str] = None


class RAGQueryRequest(BaseModel):
    query: str
    strategy: Optional[str] = "vector"
    top_k: int = 5
    filters: Optional[dict] = None


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    strategy: str
    latency_ms: float
    confidence: float


class EvalQuestion(BaseModel):
    question: str
    expected_answer: str


class EvaluationRequest(BaseModel):
    questions: list[EvalQuestion]
    dataset_name: Optional[str] = None


class PerQuestionResult(BaseModel):
    question: str
    expected_answer: str
    generated_answer: str = ""
    retrieved_context: str = ""
    accuracy_llm: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    exact_match: float = 0.0
    semantic_similarity: float = 0.0
    f1: float = 0.0
    accuracy_combined: float = 0.0
    recall_10: float = 0.0
    recall_20: float = 0.0
    recall_50: float = 0.0
    mrr: float = 0.0
    ndcg_10: float = 0.0
    gold_answer_found: bool = False
    accuracy_rationale: str = ""
    faithfulness_rationale: str = ""
    answer_relevancy_rationale: str = ""
    context_precision_rationale: str = ""
    context_recall_rationale: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None


class EvaluationResponse(BaseModel):
    accuracy: float = 0.0
    accuracy_llm: float = 0.0
    accuracy_combined: float = 0.0
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    exact_match: float = 0.0
    semantic_similarity: float = 0.0
    f1: float = 0.0
    recall_10: float = 0.0
    recall_20: float = 0.0
    recall_50: float = 0.0
    mrr: float = 0.0
    ndcg_10: float = 0.0
    latency_avg_ms: float = 0.0
    dataset_name: str = ""
    failed_questions: list[dict] = []
    per_question: list[PerQuestionResult] = []
