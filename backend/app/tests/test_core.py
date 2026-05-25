"""
Core unit tests — no external services required (no Ollama, no ChromaDB).
Run with: pytest app/tests/test_core.py -v
"""
import pytest
from app.rag.rrf import reciprocal_rank_fusion
from app.rag.metadata_filter import build_chroma_filter, filter_results
from app.rag.bm25 import BM25Retriever


# ─── RRF ──────────────────────────────────────────────────────────────────────

def test_rrf_single_list():
    results = [
        {"chunk_id": "a", "chunk_text": "hello", "score": 0.9},
        {"chunk_id": "b", "chunk_text": "world", "score": 0.8},
    ]
    fused = reciprocal_rank_fusion([results], top_k=2)
    assert len(fused) == 2
    assert fused[0]["chunk_id"] == "a"


def test_rrf_two_lists_overlap():
    list1 = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.8}]
    list2 = [{"chunk_id": "b", "score": 0.95}, {"chunk_id": "c", "score": 0.7}]
    fused = reciprocal_rank_fusion([list1, list2], top_k=3)
    ids = [r["chunk_id"] for r in fused]
    # "b" appears in both lists — should rank high
    assert "b" in ids
    assert len(fused) <= 3


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([[], []], top_k=5)
    assert fused == []


def test_rrf_top_k_limit():
    results = [{"chunk_id": str(i), "score": float(i)} for i in range(20)]
    fused = reciprocal_rank_fusion([results], top_k=5)
    assert len(fused) == 5


# ─── Metadata filter ──────────────────────────────────────────────────────────

def test_build_chroma_filter_empty():
    assert build_chroma_filter({}) is None
    assert build_chroma_filter(None) is None


def test_build_chroma_filter_single():
    f = build_chroma_filter({"filename": "test.pdf"})
    assert f == {"filename": {"$eq": "test.pdf"}}


def test_build_chroma_filter_multi():
    f = build_chroma_filter({"filename": "test.pdf", "language": "en"})
    assert "$and" in f
    assert len(f["$and"]) == 2


def test_build_chroma_filter_unsupported_key():
    f = build_chroma_filter({"unknown_key": "value"})
    assert f is None


def test_filter_results_empty_filters():
    items = [{"chunk_id": "1", "metadata": {"filename": "a.pdf"}}]
    assert filter_results(items, {}) == items


def test_filter_results_matching():
    items = [
        {"chunk_id": "1", "metadata": {"filename": "a.pdf", "language": "en"}},
        {"chunk_id": "2", "metadata": {"filename": "b.pdf", "language": "hi"}},
    ]
    filtered = filter_results(items, {"language": "en"})
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "1"


def test_filter_results_no_match():
    items = [{"chunk_id": "1", "metadata": {"filename": "a.pdf"}}]
    assert filter_results(items, {"filename": "b.pdf"}) == []


# ─── BM25 ─────────────────────────────────────────────────────────────────────

def test_bm25_index_and_search():
    retriever = BM25Retriever()
    chunks = [
        {"chunk_id": "1", "chunk_text": "government scheme eligibility farmers india",
         "metadata": {}, "document_id": "doc1", "filename": "a.txt"},
        {"chunk_id": "2", "chunk_text": "PM Kisan financial support rural households",
         "metadata": {}, "document_id": "doc1", "filename": "a.txt"},
        {"chunk_id": "3", "chunk_text": "solar panel installation renewable energy subsidy",
         "metadata": {}, "document_id": "doc2", "filename": "b.txt"},
    ]
    retriever.index("test_col", chunks)
    results = retriever.search("test_col", "PM Kisan farmers", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] in {"1", "2"}


def test_bm25_missing_collection():
    retriever = BM25Retriever()
    results = retriever.search("nonexistent", "query", top_k=5)
    assert results == []


def test_bm25_zero_score_excluded():
    retriever = BM25Retriever()
    retriever.index("col", [
        {"chunk_id": "x", "chunk_text": "apples oranges", "metadata": {}, "document_id": "d", "filename": "f"},
    ])
    results = retriever.search("col", "zzzzzzzzzzz", top_k=5)
    assert results == []


def test_bm25_remove_collection():
    retriever = BM25Retriever()
    retriever.index("col", [
        {"chunk_id": "1", "chunk_text": "test text", "metadata": {}, "document_id": "d", "filename": "f"},
    ])
    retriever.remove_collection("col")
    assert retriever.search("col", "test", top_k=5) == []


# ─── Config ───────────────────────────────────────────────────────────────────

def test_settings_defaults():
    from app.core.config import settings
    assert settings.OPENAI_LLM_MODEL == "gpt-4o-mini"
    assert settings.OPENAI_EMBED_MODEL == "text-embedding-3-small"
    assert settings.TOP_K == 5
    assert settings.CHUNK_SIZE == 512


# ─── Schemas ──────────────────────────────────────────────────────────────────

def test_search_request_validation():
    from app.schemas.search import SearchRequest
    req = SearchRequest(query="test query", top_k=10)
    assert req.query == "test query"
    assert req.top_k == 10


def test_rag_query_request_defaults():
    from app.schemas.rag import RAGQueryRequest
    req = RAGQueryRequest(query="what is PM Kisan?")
    assert req.strategy == "hybrid"
    assert req.top_k == 5


def test_eval_question_schema():
    from app.schemas.rag import EvalQuestion, EvaluationRequest
    req = EvaluationRequest(
        questions=[EvalQuestion(question="What?", expected_answer="This.")],
        dataset_name="smoke_test",
    )
    assert len(req.questions) == 1
    assert req.dataset_name == "smoke_test"


def test_agent_request_schema():
    from app.schemas.agent import AgentRequest
    req = AgentRequest(query="find schemes for farmers", top_k=3)
    assert req.top_k == 3
    assert req.filters is None


# ─── Markdown chunking ────────────────────────────────────────────────────────

def test_markdown_section_parsing():
    from app.rag.markdown_rag import _parse_markdown_sections
    md = """# Introduction
Some intro text.

## Section One
Content of section one.

### Subsection
Deep content here.

## Section Two
More content.
"""
    sections = _parse_markdown_sections(md)
    headings = [s["heading"] for s in sections]
    assert "Introduction" in headings
    assert "Section One" in headings
    assert "Section Two" in headings


def test_markdown_empty_content():
    from app.rag.markdown_rag import _parse_markdown_sections
    sections = _parse_markdown_sections("")
    assert sections == []


# ─── PDF heading detection ────────────────────────────────────────────────────

def test_pdf_heading_detection():
    from app.rag.pdf_rag import _detect_heading
    assert _detect_heading("INTRODUCTION") is not None
    assert _detect_heading("1. Overview") is not None
    assert _detect_heading("This is a long paragraph that should not be a heading " * 3) is None


# ─── Text chunking ────────────────────────────────────────────────────────────

def test_text_chunking():
    from app.services.document_service import _chunk_text
    text = " ".join([f"word{i}" for i in range(600)])
    chunks = _chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    # overlap: last words of chunk N should appear in chunk N+1
    words0 = set(chunks[0].split())
    words1 = set(chunks[1].split())
    assert len(words0 & words1) > 0


def test_text_chunking_short():
    from app.services.document_service import _chunk_text
    chunks = _chunk_text("short text", chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


# ─── File type detection ──────────────────────────────────────────────────────

def test_detect_supported_types():
    from app.services.document_service import _detect_type
    assert _detect_type("document.pdf") == "pdf"
    assert _detect_type("README.md") == "markdown"
    assert _detect_type("data.csv") == "csv"
    assert _detect_type("notes.txt") == "text"
    assert _detect_type("config.json") == "json"


def test_detect_unsupported_type():
    from app.services.document_service import _detect_type
    from app.core.exceptions import UnsupportedFileTypeError
    with pytest.raises(UnsupportedFileTypeError):
        _detect_type("image.png")


# ─── Exception classes ────────────────────────────────────────────────────────

def test_exception_hierarchy():
    from app.core.exceptions import (
        RAGPlatformException, DocumentNotFoundError,
        ConversationNotFoundError, OllamaConnectionError,
        ChromaDBError, UnsupportedFileTypeError,
    )
    exc = DocumentNotFoundError("abc-123")
    assert exc.status_code == 404
    assert "abc-123" in exc.message

    exc2 = ConversationNotFoundError("conv-99")
    assert exc2.status_code == 404

    exc3 = OllamaConnectionError("timeout")
    assert exc3.status_code == 503

    exc4 = ChromaDBError("collection missing")
    assert exc4.status_code == 503

    exc5 = UnsupportedFileTypeError("mp4")
    assert exc5.status_code == 422
    assert "mp4" in exc5.message


# ─── Router intent classification ────────────────────────────────────────────

def test_coordinator_intent_classification():
    from app.agents.coordinator_agent import _classify_intent
    assert _classify_intent("show me the CSV table data") == "table"
    assert _classify_intent("search the website for PM Kisan") == "web"
    assert _classify_intent("list all government schemes for Karnataka") == "structured"
    assert _classify_intent("what is the capital of France?") == "general"


def test_router_doc_type_detection():
    from app.agents.router_agent import _detect_doc_type
    assert _detect_doc_type("find in PDF report") == "pdf"
    assert _detect_doc_type("search the README markdown guide") == "markdown"
    assert _detect_doc_type("query the CSV table rows") == "csv"
    assert _detect_doc_type("general question") == "text"