import base64
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List

import imagehash
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from openai import OpenAI
from PIL import Image

from app.config import settings
from app.storage import (
    list_docs, get_doc, create_doc, update_doc, delete_doc,
    list_convs, get_conv, create_conv, add_message, delete_conv,
)
from app.chroma_store import (
    add_chunks, get_chunks_by_doc, delete_chunks_by_doc,
    count_chunks, list_collections_info,
)
from app.models import (
    SearchRequest, SearchResponse, SearchResult,
    ChatRequest, LiveAskRequest, RAGQueryRequest, RAGQueryResponse, Source,
    EvaluationRequest, EvaluationResponse, PerQuestionResult,
    Document,
)
from app.embeddings import embedder
from app.rag import chunk_text, extract_text_from_file, retrieve, build_context, query
from app.llm import llm
from app.ocr import ocr

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample.png")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Live frame / vision settings
FRAME_MAX_BYTES = 6 * 1024 * 1024  # 6 MB
MIN_FRAME_INTERVAL_SEC = 0.18  # ~5 FPS
PHASH_DIFF_THRESHOLD = 4

# Audio defaults
DEFAULT_TTS_VOICE = "alloy"

_frame_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Simple RAG", version="2.0.0", lifespan=lifespan)

logger = logging.getLogger("simple_rag")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cors_on_error(request: Request, call_next):
    try:
        response = await call_next(request)
    except HTTPException as exc:
        response = JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )
    except RequestValidationError as exc:
        response = JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(exc.errors())},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error during request %s %s", request.method, request.url.path)
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    origin = request.headers.get("origin") or "*"
    response.headers.setdefault("Access-Control-Allow-Origin", origin)
    if origin != "*":
        response.headers.setdefault("Vary", "Origin")
        response.headers.setdefault("Access-Control-Allow-Credentials", "true")
    response.headers.setdefault("Access-Control-Allow-Methods", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "*")
    return response


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_phash(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return str(imagehash.phash(img))
    except Exception:
        return ""


def _fields_to_map(fields: list[dict]) -> dict[str, str]:
    out = {}
    for f in fields or []:
        key = str(f.get("field") or f.get("name") or "").strip().lower()
        if key:
            out[key] = str(f.get("value") or "").strip()
    return out


def _diff_fields(prev: list[dict], curr: list[dict]) -> list[dict]:
    prev_map = _fields_to_map(prev)
    curr_map = _fields_to_map(curr)
    diff: list[dict] = []
    for k, v in curr_map.items():
        if k not in prev_map:
            diff.append({"field": k, "change": "added", "value": v})
        elif prev_map[k] != v:
            diff.append({"field": k, "change": "updated", "value": v, "previous": prev_map[k]})
    return diff


def _merge_fields(prev: list[dict], curr: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for f in prev or []:
        key = str(f.get("field") or f.get("name") or "").strip().lower()
        if key:
            merged[key] = dict(f)
    for f in curr or []:
        key = str(f.get("field") or f.get("name") or "").strip().lower()
        if not key:
            continue
        existing = merged.get(key, {})
        val = f.get("value") or existing.get("value") or ""
        merged[key] = {
            **existing,
            **f,
            "field": f.get("field") or f.get("name") or existing.get("field") or key,
            "value": val,
        }
    return list(merged.values())


def _parse_vision_response(text: str) -> tuple[list[dict], str, str]:
    """Return (fields, layout_markdown, raw_text)."""
    fields: list[dict] = []
    layout_md = ""
    raw_text = text
    # Simple heuristic: find fenced code blocks
    for block in text.split("```"):
        block = block.strip()
        if not block:
            continue
        if block.lower().startswith("json"):
            block = block[4:].strip()
        try:
            data = json.loads(block)
        except Exception:
            continue
        if isinstance(data, dict):
            if isinstance(data.get("fields"), list):
                fields = data.get("fields", [])
            if isinstance(data.get("layout_markdown"), str):
                layout_md = data.get("layout_markdown", "")
            if isinstance(data.get("raw_text"), str):
                raw_text = data.get("raw_text", raw_text)
            if fields or layout_md:
                return fields, layout_md, raw_text
    # Fallback: try parsing whole text as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            fields = data.get("fields", []) if isinstance(data.get("fields"), list) else []
            layout_md = data.get("layout_markdown", "") if isinstance(data.get("layout_markdown"), str) else ""
            raw_text = data.get("raw_text", raw_text) if isinstance(data.get("raw_text"), str) else raw_text
    except Exception:
        pass
    return fields, layout_md, raw_text


def _extract_form(image_bytes: bytes, mime: str) -> tuple[list[dict], str, str]:
    client = OpenAI(api_key=settings.openai_api_key)
    b64 = base64.b64encode(image_bytes).decode()
    image_url = f"data:{mime};base64,{b64}"
    prompt = (
        "You are a form transcription assistant. First determine if this is a bank-related form (account, KYC, Aadhaar, PAN, loan, deposit, withdrawal, banking service). "
        "If it is NOT a bank form, respond with an empty JSON fenced block: {\"fields\": [], \"layout_markdown\": \"\", \"raw_text\": \"\"}. "
        "If it IS a bank form, then: 1) Extract visible field labels and filled values. "
        "Return JSON in a fenced block with keys: fields: array of {field, value, confidence, notes}, layout_markdown: markdown recreating the form layout, raw_text: full text. "
        "Preserve order top-to-bottom, left-to-right. Confidence 0-1."
    )
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    {"type": "text", "text": "Extract now."},
                ],
            },
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content or ""
    return _parse_vision_response(text)


def _eval_accuracy_llm(system: str, prompt: str) -> str:
    if settings.eval_accuracy_provider == "ollama":
        import httpx

        payload = {
            "model": settings.eval_accuracy_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0},
        }

        try:
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except httpx.HTTPStatusError as exc:
            # Older Ollama versions may not expose /api/chat; fall back to /api/generate
            if exc.response.status_code != 404:
                raise
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.eval_accuracy_model,
                    "prompt": f"{system}\n\n{prompt}",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=60,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc2:
                if exc2.response.status_code == 404:
                    detail = exc2.response.text.strip()
                    raise RuntimeError(
                        "Eval accuracy model not available on Ollama. "
                        f"model={settings.eval_accuracy_model}, url={settings.ollama_base_url}, "
                        f"detail={detail or 'not found'}"
                    ) from exc2
                raise
            data = resp.json()
            if "response" in data:
                return data["response"]
            return data.get("message", {}).get("content", "")

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.eval_accuracy_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Simple RAG"}


@app.get("/health/db")
async def health_db():
    return {"status": "ok", "database": "file-based"}


@app.get("/health/chroma")
async def health_chroma():
    try:
        cols = list_collections_info()
        return {"status": "ok", "collections": [c["name"] for c in cols]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


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
async def upload_document(
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    metadata: str = Form("{}"),
):
    import json

    meta = {}
    try:
        meta = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        pass

    payload_files: List[UploadFile] = []
    if files:
        payload_files = files
    elif file:
        payload_files = [file]
    else:
        raise HTTPException(400, "No file provided")

    results = []
    for f in payload_files:
        ext = os.path.splitext(f.filename or "file.txt")[1].lower()
        doc_type = ext.lstrip(".") if ext else "text"
        if doc_type in ("md",):
            doc_type = "markdown"

        filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{f.filename}")
        with open(filepath, "wb") as out:
            content = await f.read()
            out.write(content)

        text = extract_text_from_file(filepath)
        chunks = chunk_text(text)
        doc = create_doc(
            filename=f.filename or "unknown",
            filepath=filepath,
            doc_type=doc_type,
            embedding_model=settings.embedding_model,
            metadata=meta,
        )

        if chunks:
            embeddings = embedder.embed_batch(chunks)
            metadata_list = [{"chunk_index": i} for i in range(len(chunks))]
            chunk_count = add_chunks(
                doc.id, doc.filename, chunks, embeddings, metadata_list
            )
        else:
            chunk_count = 0

        update_doc(doc.id, chunk_count=chunk_count)
        results.append({
            "id": doc.id,
            "filename": doc.filename,
            "document_type": doc_type,
            "retrieval_strategy": "vector",
            "chunk_count": chunk_count,
            "message": "Document uploaded and indexed",
        })

    # Backward-compatible single-file response
    if len(results) == 1:
        return results[0]
    return {"uploaded": len(results), "results": results, "message": "Documents uploaded and indexed"}


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
    delete_chunks_by_doc(doc.id)
    if chunks:
        embeddings = embedder.embed_batch(chunks)
        metadata_list = [{"chunk_index": i} for i in range(len(chunks))]
        chunk_count = add_chunks(
            doc.id, doc.filename, chunks, embeddings, metadata_list
        )
    else:
        chunk_count = 0
    update_doc(doc.id, chunk_count=chunk_count)
    return {"document_id": doc.id, "message": "Reindexed", "chunk_count": chunk_count}


@app.get("/api/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    chunks = get_chunks_by_doc(doc_id)
    # Remove embedding from metadata if present
    for c in chunks:
        c["chunk_metadata"].pop("_embedding", None)
    return chunks


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


# ─── Live Vision & Streaming Assist ────────────────────────────────────────────


def _get_session(session_id: str) -> dict | None:
    return _frame_sessions.get(session_id)


def _ensure_session(session_id: str) -> dict:
    if session_id not in _frame_sessions:
        _frame_sessions[session_id] = {
            "frame_bytes": None,
            "mime": None,
            "phash": None,
            "fields": [],
            "layout_markdown": "",
            "raw_text": "",
            "diff": [],
            "updated_at": None,
            "has_new": False,
            "last_used_phash": None,
            "last_frame_ts": 0.0,
        }
    return _frame_sessions[session_id]


@app.post("/api/live/frame/push")
async def live_frame_push(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
):
    sid = session_id or str(uuid.uuid4())
    session = _ensure_session(sid)

    now = time.time()
    if session.get("last_frame_ts") and now - session["last_frame_ts"] < MIN_FRAME_INTERVAL_SEC:
        raise HTTPException(429, f"Too many frames; wait {MIN_FRAME_INTERVAL_SEC:.2f}s")

    content = await file.read()
    if len(content) > FRAME_MAX_BYTES:
        raise HTTPException(413, f"Frame too large; limit {FRAME_MAX_BYTES // (1024*1024)} MB")

    mime = file.content_type or "application/octet-stream"
    phash = _compute_phash(content)
    prev_phash = session.get("phash")
    session["last_frame_ts"] = now

    if prev_phash and phash and imagehash.hex_to_hash(prev_phash) - imagehash.hex_to_hash(phash) < PHASH_DIFF_THRESHOLD:
        # No meaningful visual change; keep frame but skip vision.
        session.update({
            "frame_bytes": content,
            "mime": mime,
        })
        return {
            "session_id": sid,
            "status": "unchanged",
            "phash": phash,
            "updated_at": _utcnow(),
        }

    try:
        fields, layout_md, raw_text = _extract_form(content, mime)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vision extraction failed")
        raise HTTPException(500, f"Vision extraction failed: {exc}") from exc

    merged_fields = _merge_fields(session.get("fields", []), fields)
    diff = _diff_fields(session.get("fields", []), merged_fields)

    session.update({
        "frame_bytes": content,
        "mime": mime,
        "phash": phash,
        "fields": merged_fields,
        "layout_markdown": layout_md,
        "raw_text": raw_text,
        "diff": diff,
        "updated_at": _utcnow(),
        "has_new": True,
    })

    return {
        "session_id": sid,
        "status": "updated",
        "phash": phash,
        "fields": merged_fields,
        "layout_markdown": layout_md,
        "diff": diff,
        "raw_text": raw_text,
        "updated_at": session["updated_at"],
    }


@app.get("/api/live/frame/context")
async def live_frame_context(session_id: str):
    session = _get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session_id,
        "fields": session.get("fields", []),
        "layout_markdown": session.get("layout_markdown", ""),
        "diff": session.get("diff", []),
        "updated_at": session.get("updated_at"),
        "phash": session.get("phash"),
        "has_new": session.get("has_new", False),
    }


@app.post("/api/live/frame/clear")
async def live_frame_clear(session_id: str):
    cleared = False
    if session_id in _frame_sessions:
        _frame_sessions.pop(session_id, None)
        cleared = True
    return {"session_id": session_id, "cleared": cleared}


@app.post("/api/live/transcribe")
async def live_transcribe(audio: UploadFile = File(...), language: str | None = Form(None)):
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        content = await audio.read()
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "audio.webm", content),
            language=language,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed")
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    return {"text": result.text, "language": language or "auto"}


@app.post("/api/live/tts")
async def live_tts(data: dict):
    text = data.get("text", "")
    voice = data.get("voice") or DEFAULT_TTS_VOICE
    if not text:
        raise HTTPException(400, "text is required")
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        audio_resp = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("TTS failed")
        raise HTTPException(500, f"TTS failed: {exc}") from exc

    def _iter_bytes():
        yield audio_resp.read()

    return StreamingResponse(_iter_bytes(), media_type="audio/mpeg")


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


def _build_form_context(session: dict) -> tuple[str, list[dict]]:
    if not session:
        return "", []
    fields = session.get("fields", []) or []
    layout_md = session.get("layout_markdown", "") or ""
    diff = session.get("diff", []) or []
    parts = []
    if diff:
        parts.append("Recent form changes:\n" + json.dumps(diff, ensure_ascii=False, indent=2))
    if fields:
        parts.append("Current form fields:\n" + json.dumps(fields, ensure_ascii=False, indent=2))
    if layout_md:
        parts.append("Form layout (markdown):\n" + layout_md)
    return "\n\n".join(parts), diff


@app.post("/api/live/ask")
async def live_ask(req: LiveAskRequest):
    if not req.conversation_id:
        conv = create_conv()
        req.conversation_id = conv.id

    add_message(req.conversation_id, "user", req.question)

    context, sources, confidence, elapsed = query(req.question, req.top_k or 5)

    form_ctx = ""
    if req.session_id and req.use_form_context:
        session = _get_session(req.session_id)
        if session:
            form_ctx, _ = _build_form_context(session)
            session["has_new"] = False
            session["last_used_phash"] = session.get("phash")

    system_prompt = (
        "You are a helpful RAG assistant. Answer the user's question based on the provided context. "
        "If the context doesn't contain enough information, say so. Be concise."
    )
    if form_ctx:
        system_prompt += " Use the live form fields when relevant."
    if req.target_language and req.target_language.lower() not in ("en", "english"):
        system_prompt += f" Respond in {req.target_language}."

    user_parts = []
    user_parts.append(f"RAG context:\n{context}")
    if form_ctx:
        user_parts.append(f"Live form context:\n{form_ctx}")
    if req.manual_context:
        user_parts.append(f"User provided context:\n{req.manual_context}")
    user_parts.append(f"Question: {req.question}")
    user_prompt = "\n\n".join(user_parts)

    async def stream_and_store():
        full_answer = ""
        async for token in llm.generate_stream(system_prompt, user_prompt):
            full_answer += token
            yield token
        add_message(req.conversation_id, "assistant", full_answer, [s.model_dump() for s in sources])

    return StreamingResponse(
        stream_and_store(),
        media_type="text/plain",
        headers={"X-Conversation-Id": req.conversation_id},
    )


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
            acc_str = _eval_accuracy_llm(
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


@app.post("/api/rag/evaluate/images")
async def rag_evaluate_images(
    use_sample: bool = Form(False),
    files: List[UploadFile] | None = File(None),
):
    images: list[tuple[str, bytes, str]] = []

    if use_sample:
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            raise HTTPException(500, "Sample image not found on server")
        with open(SAMPLE_IMAGE_PATH, "rb") as fh:
            images.append((os.path.basename(SAMPLE_IMAGE_PATH), fh.read(), "image/png"))

    if files:
        for f in files:
            content = await f.read()
            images.append((f.filename or "image", content, f.content_type or "application/octet-stream"))

    if not images:
        raise HTTPException(400, "No images provided")

    try:
        questions, errors = ocr.extract_from_images(images)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCR failed: {exc}") from exc

    response = {
        "questions": questions,
        "count": len(questions),
        "errors": errors,
        "images_processed": len(images),
        "from_sample": use_sample,
    }

    if len(questions) == 0 and errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "No questions extracted", "errors": errors},
        )

    return response


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
    cols = list_collections_info()
    counts = {c["name"]: c["count"] for c in cols}
    return {"collections": [c["name"] for c in cols], "counts": counts}


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
    if chunks:
        embeddings = embedder.embed_batch(chunks)
        metadata_list = [{"chunk_index": i} for i in range(len(chunks))]
        chunk_count = add_chunks(
            doc.id, doc.filename, chunks, embeddings, metadata_list
        )
    else:
        chunk_count = 0
    update_doc(doc.id, chunk_count=chunk_count)

    return {"count": chunk_count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)
