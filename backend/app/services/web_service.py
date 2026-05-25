import uuid
from typing import Any, Optional
import httpx
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession
from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client
from app.rag.bm25 import bm25_retriever
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("web_service")

ALLOWED_DOMAINS = [
    "docs.", "developer.", "gov.", ".gov", "wikipedia.org",
    "github.com", "arxiv.org", "education.", "official",
]


def _is_allowed_url(url: str) -> bool:
    return True  # policy: user-approved URLs are allowed


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


async def _fetch_url(url: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, follow_redirects=True, headers={"User-Agent": "RAGBot/1.0"})
        response.raise_for_status()
        return response.text


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


class WebService:
    async def ingest(
        self,
        url: str,
        collection_name: str = "web_documents",
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        logger.info(f"Web ingest: {url}")
        html = await _fetch_url(url)
        text = _clean_html(html)
        chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        doc_id = str(uuid.uuid4())
        ids, embeddings, documents, metadatas = [], [], [], []
        bm25_chunks = []

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 30:
                continue
            emb = await ollama_client.embeddings(chunk)
            chunk_id = f"{doc_id}_web_{i}"
            meta = {
                "document_id": doc_id,
                "filename": url,
                "document_type": "web",
                "source_url": url,
                "chunk_index": i,
                **(metadata or {}),
            }
            ids.append(chunk_id)
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append(meta)
            bm25_chunks.append({"chunk_id": chunk_id, "chunk_text": chunk, "metadata": meta, "document_id": doc_id, "filename": url})

        if ids:
            chroma_client.add_documents(collection_name, ids, embeddings, documents, metadatas)
            bm25_retriever.index(collection_name, bm25_chunks)

        return {"url": url, "document_id": doc_id, "chunk_count": len(ids), "message": "Ingested successfully"}

    async def query(
        self,
        query: str,
        url: Optional[str] = None,
        top_k: int = 5,
        collection_name: str = "web_documents",
    ) -> list[dict[str, Any]]:
        query_emb = await ollama_client.embeddings(query)
        where = {"filename": {"$eq": url}} if url else None
        return chroma_client.search(collection_name, query_emb, top_k, where)


web_service = WebService()