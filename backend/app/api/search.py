import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.rag.vector_rag import vector_rag
from app.rag.hybrid_rag import hybrid_rag
from app.rag.bm25 import bm25_retriever
from app.rag.table_rag import table_rag
from app.rag.metadata_filter import filter_results, build_chroma_filter
from app.rag.synonym_expansion import get_synonym_expander
from app.embeddings.openai_client import openai_client as ollama_client
from app.chromadb.client import chroma_client
from app.schemas.search import SearchRequest, SearchResponse, SearchResult

router = APIRouter(prefix="/api/search", tags=["Search"])


def _search_with_synonyms(search_fn, query: str, *args, **kwargs) -> list[dict]:
    """Search with query and its synonym expansions, deduplicate and merge results"""
    expander = get_synonym_expander()
    expanded_queries = expander.expand_query(query)
    
    all_results = {}
    for q in expanded_queries:
        try:
            results = search_fn(q, *args, **kwargs)
            for r in results:
                doc_id = r.get("chunk_id", r.get("document_id", ""))
                if doc_id not in all_results:
                    all_results[doc_id] = r
        except Exception:
            pass
    
    return sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)


def _build_response(query: str, results: list[dict], strategy: str, start_time: float) -> SearchResponse:
    latency = (time.time() - start_time) * 1000
    search_results = []
    sources = []
    for r in results:
        meta = r.get("metadata", {})
        sr = SearchResult(
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("document_id") or meta.get("document_id", ""),
            filename=r.get("filename") or meta.get("filename", ""),
            chunk_text=r.get("chunk_text", ""),
            score=r.get("score", 0.0),
            metadata=meta,
        )
        search_results.append(sr)
        fn = sr.filename
        if fn and fn not in sources:
            sources.append(fn)
    confidence = round(sum(r.score for r in search_results) / max(len(search_results), 1), 4)
    return SearchResponse(
        query=query,
        results=search_results,
        confidence=confidence,
        sources=sources,
        latency_ms=round(latency, 2),
        strategy=strategy,
    )


@router.post("/vector", response_model=SearchResponse)
async def vector_search(req: SearchRequest):
    start = time.time()
    collection = req.collection_name or "text_documents"
    
    async def search(q):
        return await vector_rag.retrieve(q, collection, req.top_k, req.filters)
    
    expander = get_synonym_expander()
    expanded_queries = expander.expand_query(req.query)
    
    all_results = {}
    for q in expanded_queries:
        results = await search(q)
        for r in results:
            doc_id = r.get("chunk_id", r.get("document_id", ""))
            if doc_id not in all_results:
                all_results[doc_id] = r
    
    results = sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)[:req.top_k]
    return _build_response(req.query, results, "vector", start)


@router.post("/bm25", response_model=SearchResponse)
async def bm25_search(req: SearchRequest):
    start = time.time()
    collection = req.collection_name or "text_documents"
    
    expander = get_synonym_expander()
    expanded_queries = expander.expand_query(req.query)
    
    all_results = {}
    for q in expanded_queries:
        results = bm25_retriever.search(collection, q, req.top_k)
        if req.filters:
            results = filter_results(results, req.filters)
        for r in results:
            doc_id = r.get("chunk_id", r.get("document_id", ""))
            if doc_id not in all_results:
                all_results[doc_id] = r
    
    results = sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)[:req.top_k]
    return _build_response(req.query, results, "bm25", start)


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(req: SearchRequest):
    start = time.time()
    collection = req.collection_name or "text_documents"
    
    async def search(q):
        return await hybrid_rag.retrieve(q, collection, req.top_k, req.filters)
    
    expander = get_synonym_expander()
    expanded_queries = expander.expand_query(req.query)
    
    all_results = {}
    for q in expanded_queries:
        results = await search(q)
        for r in results:
            doc_id = r.get("chunk_id", r.get("document_id", ""))
            if doc_id not in all_results:
                all_results[doc_id] = r
    
    results = sorted(all_results.values(), key=lambda x: x.get("score", 0), reverse=True)[:req.top_k]
    return _build_response(req.query, results, "hybrid", start)


@router.post("/metadata", response_model=SearchResponse)
async def metadata_search(req: SearchRequest):
    start = time.time()
    collection = req.collection_name or "text_documents"
    query_emb = await ollama_client.embeddings(req.query)
    where = build_chroma_filter(req.filters or {})
    results = chroma_client.search(collection, query_emb, req.top_k, where)
    return _build_response(req.query, results, "metadata", start)


@router.post("/table", response_model=SearchResponse)
async def table_search(req: SearchRequest):
    start = time.time()
    results = await table_rag.query(req.query, top_k=req.top_k)
    return _build_response(req.query, results, "table", start)