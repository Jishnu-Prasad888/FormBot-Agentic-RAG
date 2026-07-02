import os
import time

from app.embeddings import embedder
from app.chroma_store import search
from app.models import SearchResult, Source


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
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
    query_emb = embedder.embed(query)
    results = search(query_emb, top_k)
    if not results:
        return [], 0.0
    scored = [
        SearchResult(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            filename=r["filename"],
            chunk_text=r["chunk_text"],
            score=r["score"],
            metadata=r["metadata"],
        )
        for r in results
    ]
    confidence = scored[0].score if scored else 0.0
    return scored, confidence


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
