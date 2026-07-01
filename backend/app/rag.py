import os
import time
import uuid
from typing import Optional

from app.embeddings import embedder
from app.storage import get_all_chunks, get_doc
from app.models import SearchResult, Source


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_space = text.rfind(" ", start + chunk_size - overlap, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else end
    return [c for c in chunks if c]


def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    elif ext == ".docx":
        try:
            from docx import Document as DocxDoc
            doc = DocxDoc(filepath)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    else:
        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""


def retrieve(query: str, top_k: int = 5) -> tuple[list[SearchResult], float]:
    chunks = get_all_chunks()
    if not chunks:
        return [], 0.0

    query_emb = embedder.embed(query)
    chunk_embeddings = [c.chunk_metadata.get("_embedding") for c in chunks]

    if not any(chunk_embeddings):
        return [], 0.0

    scored = []
    for c, emb in zip(chunks, chunk_embeddings):
        if emb:
            score = embedder.cosine_similarity(query_emb, emb)
        else:
            score = 0.0
        doc = get_doc(c.document_id)
        scored.append(SearchResult(
            chunk_id=c.id,
            document_id=c.document_id,
            filename=doc.filename if doc else "unknown",
            chunk_text=c.chunk_text,
            score=score,
            metadata=c.chunk_metadata,
        ))

    scored.sort(key=lambda x: x.score, reverse=True)
    results = scored[:top_k]
    confidence = results[0].score if results else 0.0
    return results, confidence


def build_context(results: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[Source {i + 1}] {r.filename}:\n{r.chunk_text}"
        for i, r in enumerate(results)
    )


def query(query: str, top_k: int = 5) -> tuple[str, list[Source], float, float]:
    start = time.time()
    results, confidence = retrieve(query, top_k)
    context = build_context(results)
    sources = [
        Source(filename=r.filename, chunk_id=r.chunk_id, score=r.score)
        for r in results
    ]
    elapsed = (time.time() - start) * 1000
    return context, sources, confidence, elapsed
