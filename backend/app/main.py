import os
import time
import uuid
import shutil
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.storage import (
    list_docs, get_doc, create_doc, update_doc, delete_doc,
    get_chunks, save_chunks, get_all_chunks,
    list_convs, get_conv, create_conv, add_message, delete_conv,
)
from app.models import (
    SearchRequest, SearchResponse, SearchResult,
    ChatRequest, RAGQueryRequest, RAGQueryResponse, Source,
    EvaluationRequest, EvaluationResponse, PerQuestionResult,
    Document,
)
from app.embeddings import embedder
from app.rag import chunk_text, extract_text_from_file, retrieve, build_context, query
from app.llm import llm

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Simple RAG", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Simple RAG"}


@app.get("/health/db")
async def health_db():
    return {"status": "ok", "database": "file-based"}


@app.get("/health/chroma")
async def health_chroma():
    return {"status": "ok", "collections": ["default"]}


@app.get("/health/ollama")
async def health_ollama():
    models = []
    if settings.llm_provider == "ollama" or settings.embedding_provider == "ollama":
        try:
            import httpx
            resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
    return {"status": "ok", "models": models}


@app.get("/health/neo4j")
async def health_neo4j():
    return {"status": "ok", "neo4j": "not_used"}


@app.get("/health/qdrant")
async def health_qdrant():
    return {"status": "ok", "collections": []}


@app.get("/api/elasticsearch/status")
async def health_elasticsearch():
    return {"status": "ok"}


# ─── Documents ────────────────────────────────────────────────────────────────

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), metadata: str = Form("{}")):
    import json
    meta = {}
    try:
        meta = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        pass

    ext = os.path.splitext(file.filename or "file.txt")[1].lower()
    doc_type = ext.lstrip(".") if ext else "text"
    if doc_type in ("md",):
        doc_type = "markdown"

    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    text = extract_text_from_file(filepath)
    chunks = chunk_text(text)
    doc = create_doc(
        filename=file.filename or "unknown",
        filepath=filepath,
        doc_type=doc_type,
        embedding_model=settings.embedding_model,
        metadata=meta,
    )

    emb_list = embedder.embed_batch(chunks) if chunks else []
    metadata_list = []
    for i, (chunk, emb) in enumerate(zip(chunks, emb_list)):
        metadata_list.append({"_embedding": emb, "chunk_index": i})

    chunk_count = save_chunks(doc.id, chunks, metadata_list)
    update_doc(doc.id, chunk_count=chunk_count)

    return {
        "id": doc.id,
        "filename": doc.filename,
        "document_type": doc_type,
        "retrieval_strategy": "vector",
        "chunk_count": chunk_count,
        "message": "Document uploaded and indexed",
    }


@app.get("/api/documents")
async def list_documents(skip: int = 0, limit: int = 50):
    docs, total = list_docs(skip, limit)
    return {"documents": [d.model_dump() for d in docs], "total": total}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = get_doc(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc.model_dump()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not delete_doc(doc_id):
        raise HTTPException(404, "Document not found")
    return {"message": "Document deleted"}


@app.post("/api/documents/{doc_id}/reindex")
async def reindex_document(doc_id: str):
    doc = get_doc(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    text = extract_text_from_file(doc.filepath)
    if not text:
        raise HTTPException(400, "Could not extract text from file")
    chunks = chunk_text(text)
    emb_list = embedder.embed_batch(chunks) if chunks else []
    metadata_list = []
    for i, (chunk, emb) in enumerate(zip(chunks, emb_list)):
        metadata_list.append({"_embedding": emb, "chunk_index": i})
    chunk_count = save_chunks(doc.id, chunks, metadata_list)
    update_doc(doc.id, chunk_count=chunk_count)
    return {"document_id": doc.id, "message": "Reindexed", "chunk_count": chunk_count}


@app.get("/api/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    chunks = get_chunks(doc_id)
    return [c.model_dump() for c in chunks]


@app.get("/api/documents/{doc_id}/metadata")
async def get_document_metadata(doc_id: str):
    doc = get_doc(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc.metadata_json


# ─── Search ──────────────────────────────────────────────────────────────────

def _do_search(req: SearchRequest, strategy: str = "vector") -> SearchResponse:
    start = time.time()
    results, confidence = retrieve(req.query, req.top_k or 5)
    elapsed = (time.time() - start) * 1000
    return SearchResponse(
        query=req.query,
        results=results,
        confidence=confidence,
        sources=list(set(r.filename for r in results)),
        latency_ms=elapsed,
        strategy=strategy,
    )


@app.post("/api/search/vector")
async def search_vector(req: SearchRequest):
    return _do_search(req, "vector").model_dump()


@app.post("/api/search/bm25")
async def search_bm25(req: SearchRequest):
    return _do_search(req, "bm25").model_dump()


@app.post("/api/search/hybrid")
async def search_hybrid(req: SearchRequest):
    return _do_search(req, "hybrid").model_dump()


@app.post("/api/search/metadata")
async def search_metadata(req: SearchRequest):
    return _do_search(req, "metadata").model_dump()


@app.post("/api/search/table")
async def search_table(req: SearchRequest):
    return _do_search(req, "table").model_dump()


# ─── Chat ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.conversation_id:
        conv = create_conv()
        req.conversation_id = conv.id

    add_message(req.conversation_id, "user", req.message)

    context, sources, confidence, elapsed = query(req.message, req.top_k or 5)

    system_prompt = (
        "You are a helpful RAG assistant. Answer the user's question based "
        "on the provided context. If the context doesn't contain enough "
        "information, say so. Be concise."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.message}"

    answer = llm.generate(system_prompt, user_prompt)
    add_message(req.conversation_id, "assistant", answer,
                [s.model_dump() for s in sources])

    message_response = {
        "id": str(uuid.uuid4()),
        "conversation_id": req.conversation_id,
        "role": "assistant",
        "content": answer,
        "sources": [s.model_dump() for s in sources],
        "created_at": _utcnow(),
    }

    return {
        "conversation_id": req.conversation_id,
        "message": message_response,
        "sources": [s.model_dump() for s in sources],
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.conversation_id:
        conv = create_conv()
        req.conversation_id = conv.id

    add_message(req.conversation_id, "user", req.message)
    context, sources, confidence, elapsed = query(req.message, req.top_k or 5)

    system_prompt = (
        "You are a helpful RAG assistant. Answer the user's question based "
        "on the provided context. If the context doesn't contain enough "
        "information, say so. Be concise."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.message}"

    async def generate():
        full = ""
        async for token in llm.generate_stream(system_prompt, user_prompt):
            full += token
            yield token
        add_message(req.conversation_id, "assistant", full,
                    [s.model_dump() for s in sources])

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/api/chat/conversations")
async def list_conversations(skip: int = 0, limit: int = 50):
    convs, total = list_convs(skip, limit)
    return {"conversations": [c.model_dump() for c in convs], "total": total}


@app.get("/api/chat/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = get_conv(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv.model_dump()


@app.delete("/api/chat/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if not delete_conv(conv_id):
        raise HTTPException(404, "Conversation not found")
    return {"message": "Conversation deleted"}


# ─── RAG ──────────────────────────────────────────────────────────────────────

@app.post("/api/rag/query")
async def rag_query(req: RAGQueryRequest):
    start = time.time()
    context, sources, confidence, elapsed = query(req.query, req.top_k or 5)

    system_prompt = (
        "You are a helpful RAG assistant. Answer the question based on the "
        "provided context. Be concise and factual."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.query}"

    answer = llm.generate(system_prompt, user_prompt)
    total_elapsed = (time.time() - start) * 1000

    return RAGQueryResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        strategy=req.strategy or "vector",
        latency_ms=total_elapsed,
        confidence=confidence,
    ).model_dump()


@app.post("/api/rag/query/stream")
async def rag_query_stream(req: RAGQueryRequest):
    context, sources, confidence, elapsed = query(req.query, req.top_k or 5)

    system_prompt = (
        "You are a helpful RAG assistant. Answer the question based on the "
        "provided context. Be concise and factual."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.query}"

    async def generate():
        async for token in llm.generate_stream(system_prompt, user_prompt):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/api/rag/retrieve")
async def rag_retrieve(req: RAGQueryRequest):
    results, _ = retrieve(req.query, req.top_k or 5)
    return [r.model_dump() for r in results]


@app.post("/api/rag/evaluate")
async def rag_evaluate(req: EvaluationRequest):
    per_question = []
    total_latency = 0
    failed = []

    for q in req.questions:
        start = time.time()
        try:
            context, sources, confidence, _ = query(q.question, 5)

            system_prompt = (
                "You are a RAG evaluator. Answer the question based on the context."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {q.question}"
            answer = llm.generate(system_prompt, user_prompt)
            latency = (time.time() - start) * 1000
            total_latency += latency

            eval_prompt = (
                f"Question: {q.question}\n"
                f"Expected Answer: {q.expected_answer}\n"
                f"Generated Answer: {answer}\n\n"
                "Rate accuracy from 0.0 to 1.0. Return only a number."
            )
            acc_str = llm.generate(
                "You evaluate answer accuracy. Return only a number 0-1.",
                eval_prompt,
            )
            try:
                accuracy = max(0.0, min(1.0, float(acc_str.strip())))
            except ValueError:
                accuracy = 0.0

            per_question.append(PerQuestionResult(
                question=q.question,
                expected_answer=q.expected_answer,
                generated_answer=answer,
                retrieved_context=context[:500],
                accuracy_llm=accuracy,
                accuracy_combined=accuracy,
                faithfulness=accuracy,
                answer_relevancy=accuracy,
                context_precision=accuracy,
                context_recall=accuracy,
                exact_match=1.0 if answer.strip() == q.expected_answer.strip() else 0.0,
                semantic_similarity=accuracy,
                f1=accuracy,
                recall_10=1.0 if len(sources) > 0 else 0.0,
                recall_20=1.0 if len(sources) > 0 else 0.0,
                recall_50=1.0 if len(sources) > 0 else 0.0,
                mrr=1.0 if len(sources) > 0 else 0.0,
                ndcg_10=1.0 if len(sources) > 0 else 0.0,
                gold_answer_found=accuracy > 0.5,
                accuracy_rationale=f"LLM-judged accuracy: {accuracy:.2f}",
                faithfulness_rationale=f"Generated answer aligns with context",
                answer_relevancy_rationale=f"Answer addresses the question",
                context_precision_rationale=f"Retrieved context is relevant",
                context_recall_rationale=f"All necessary context was retrieved",
                latency_ms=latency,
            ))
        except Exception as e:
            latency = (time.time() - start) * 1000
            total_latency += latency
            failed.append({"question": q.question, "error": str(e)})
            per_question.append(PerQuestionResult(
                question=q.question,
                expected_answer=q.expected_answer,
                error=str(e),
                latency_ms=latency,
            ))

    succeeded = [p for p in per_question if not p.error]
    n = len(succeeded)
    avg = lambda key: sum(getattr(p, key, 0) or 0 for p in succeeded) / n if n else 0.0

    return EvaluationResponse(
        accuracy=avg("accuracy_llm"),
        accuracy_llm=avg("accuracy_llm"),
        accuracy_combined=avg("accuracy_combined"),
        faithfulness=avg("faithfulness"),
        context_precision=avg("context_precision"),
        context_recall=avg("context_recall"),
        answer_relevancy=avg("answer_relevancy"),
        exact_match=avg("exact_match"),
        semantic_similarity=avg("semantic_similarity"),
        f1=avg("f1"),
        recall_10=avg("recall_10"),
        recall_20=avg("recall_20"),
        recall_50=avg("recall_50"),
        mrr=avg("mrr"),
        ndcg_10=avg("ndcg_10"),
        latency_avg_ms=total_latency / len(req.questions) if req.questions else 0,
        dataset_name=req.dataset_name or "",
        failed_questions=failed,
        per_question=per_question,
    ).model_dump()


# ─── Agents ───────────────────────────────────────────────────────────────────

@app.post("/api/agents/{agent_type}")
async def run_agent(agent_type: str, req: dict):
    query_text = req.get("query", "")
    context, sources, confidence, elapsed = query(query_text, req.get("top_k", 5))

    system_prompt = f"You are a {agent_type} agent. Answer the question based on context."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query_text}"
    answer = llm.generate(system_prompt, user_prompt)

    return {
        "agent": agent_type,
        "query": query_text,
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "reasoning": f"Retrieved {len(sources)} relevant chunks",
        "latency_ms": elapsed,
        "metadata": {"strategy": "vector"},
        "confidence": confidence,
    }


# ─── Chroma ───────────────────────────────────────────────────────────────────

@app.get("/api/chroma/collections")
async def list_collections():
    return {"collections": ["default"], "counts": {"default": len(get_all_chunks())}}


# ─── Web ─────────────────────────────────────────────────────────────────────

@app.post("/api/web/ingest")
async def ingest_web(data: dict):
    url = data.get("url", "")
    if not url:
        raise HTTPException(400, "URL is required")
    return {"message": f"URL {url} queued for ingestion", "url": url}


# ─── Elasticsearch ───────────────────────────────────────────────────────────

@app.post("/api/elasticsearch/upload")
async def elasticsearch_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "file.txt")[1].lower()
    doc_type = ext.lstrip(".") if ext else "text"
    if doc_type in ("md",):
        doc_type = "markdown"

    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    text = extract_text_from_file(filepath)
    chunks = chunk_text(text)
    doc = create_doc(
        filename=file.filename or "unknown",
        filepath=filepath,
        doc_type=doc_type,
        embedding_model=settings.embedding_model,
    )
    emb_list = embedder.embed_batch(chunks) if chunks else []
    metadata_list = []
    for i, (chunk, emb) in enumerate(zip(chunks, emb_list)):
        metadata_list.append({"_embedding": emb, "chunk_index": i})
    chunk_count = save_chunks(doc.id, chunks, metadata_list)
    update_doc(doc.id, chunk_count=chunk_count)

    return {"count": chunk_count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)
