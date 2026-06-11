import io
import uuid
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.document_repository import document_repo
from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client
from app.rag.table_rag import table_rag
from app.rag.pdf_rag import pdf_rag
from app.rag.markdown_rag import markdown_rag
from app.rag.bm25 import bm25_retriever
from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError


SUPPORTED_TYPES = {
    "pdf": "pdf",
    "md": "markdown",
    "txt": "text",
    "csv": "csv",
    "json": "json",
}

TYPE_TO_COLLECTION = {
    "pdf": "pdf_documents",
    "markdown": "markdown_documents",
    "text": "text_documents",
    "csv": "table_documents",
    "json": "text_documents",
}

TYPE_TO_STRATEGY = {
    "pdf": "hierarchical_rag",
    "markdown": "structure_aware_rag",
    "text": "vector_rag",
    "csv": "table_rag",
    "json": "vector_rag",
}


def _detect_type(filename: str) -> str:
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in SUPPORTED_TYPES:
        raise UnsupportedFileTypeError(ext)
    return SUPPORTED_TYPES[ext]


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


async def _index_text_chunks(
    document_id: str,
    filename: str,
    doc_type: str,
    text: str,
    collection: str,
    extra_metadata: Optional[dict] = None,
) -> list[dict]:
    chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    ids, embeddings, documents, metadatas = [], [], [], []
    chunk_records = []

    for i, chunk_text in enumerate(chunks):
        emb = await ollama_client.embeddings(chunk_text)
        chunk_id = f"{document_id}_chunk_{i}"
        ids.append(chunk_id)
        embeddings.append(emb)
        documents.append(chunk_text)
        meta = {
            "document_id": document_id,
            "filename": filename,
            "document_type": doc_type,
            "chunk_index": i,
            **(extra_metadata or {}),
        }
        metadatas.append(meta)
        chunk_records.append({
            "id": chunk_id,
            "document_id": document_id,
            "chunk_index": i,
            "chunk_text": chunk_text,
            "chunk_metadata": meta,
        })

    if ids:
        chroma_client.add_documents(collection, ids, embeddings, documents, metadatas)
        bm25_retriever.index(collection, [
            {"chunk_id": ids[j], "chunk_text": documents[j], "metadata": metadatas[j],
             "document_id": document_id, "filename": filename}
            for j in range(len(ids))
        ])
    return chunk_records


class DocumentService:
    async def upload_and_index(
        self,
        db: AsyncSession,
        filename: str,
        content: bytes,
        extra_metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        doc_type = _detect_type(filename)
        doc_id = str(uuid.uuid4())
        collection = TYPE_TO_COLLECTION[doc_type]
        strategy = TYPE_TO_STRATEGY[doc_type]

        # Save file
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        filepath = upload_dir / f"{doc_id}_{filename}"
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(content)

        # Index based on type
        chunk_count = 0
        chunk_records = []

        if doc_type == "pdf":
            result = await pdf_rag.index(doc_id, filename, content, extra_metadata)
            chunk_count = result["chunk_count"]
        elif doc_type == "markdown":
            text = content.decode("utf-8", errors="replace")
            result = await markdown_rag.index(doc_id, filename, text, extra_metadata)
            chunk_count = result["chunk_count"]
        elif doc_type == "csv":
            result = await table_rag.index_csv(doc_id, filename, content, extra_metadata)
            chunk_count = result["chunk_count"]
        else:
            # text / json
            text = content.decode("utf-8", errors="replace")
            chunk_records = await _index_text_chunks(doc_id, filename, doc_type, text, collection, extra_metadata)
            chunk_count = len(chunk_records)

        doc_data = {
            "id": doc_id,
            "filename": filename,
            "filepath": str(filepath),
            "document_type": doc_type,
            "retrieval_strategy": strategy,
            "language": (extra_metadata or {}).get("language", "en"),
            "chunk_count": chunk_count,
            "embedding_model": settings.OPENAI_EMBED_MODEL,
            "collection_name": collection,
            "metadata_json": extra_metadata or {},
        }

        doc = await document_repo.create(db, doc_data)

        # Persist chunks to DB for non-specialized types
        if chunk_records:
            await document_repo.bulk_create_chunks(db, chunk_records)

        return {"document": doc, "chunk_count": chunk_count}

    async def reindex(self, db: AsyncSession, doc_id: str) -> dict[str, Any]:
        doc = await document_repo.get_by_id(db, doc_id)
        if not doc:
            raise DocumentNotFoundError(doc_id)

        filepath = Path(doc.filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        async with aiofiles.open(filepath, "rb") as f:
            content = await f.read()

        # Delete existing vector data
        try:
            chroma_client.delete_by_document_id(doc.collection_name or "text_documents", doc_id)
        except Exception:
            pass

        await document_repo.delete_chunks(db, doc_id)

        doc_type = doc.document_type
        collection = TYPE_TO_COLLECTION.get(doc_type, "text_documents")
        chunk_count = 0

        if doc_type == "pdf":
            result = await pdf_rag.index(doc_id, doc.filename, content, doc.metadata_json)
            chunk_count = result["chunk_count"]
        elif doc_type == "markdown":
            text = content.decode("utf-8", errors="replace")
            result = await markdown_rag.index(doc_id, doc.filename, text, doc.metadata_json)
            chunk_count = result["chunk_count"]
        elif doc_type == "csv":
            result = await table_rag.index_csv(doc_id, doc.filename, content, doc.metadata_json)
            chunk_count = result["chunk_count"]
        else:
            text = content.decode("utf-8", errors="replace")
            chunk_records = await _index_text_chunks(doc_id, doc.filename, doc_type, text, collection, doc.metadata_json)
            chunk_count = len(chunk_records)
            if chunk_records:
                await document_repo.bulk_create_chunks(db, chunk_records)

        await document_repo.update(db, doc_id, {"chunk_count": chunk_count})
        return {"document_id": doc_id, "message": "Reindexed successfully", "chunk_count": chunk_count}

    async def delete(self, db: AsyncSession, doc_id: str) -> bool:
        doc = await document_repo.get_by_id(db, doc_id)
        if not doc:
            raise DocumentNotFoundError(doc_id)
        try:
            chroma_client.delete_by_document_id(doc.collection_name or "text_documents", doc_id)
        except Exception:
            pass
        filepath = Path(doc.filepath)
        if filepath.exists():
            filepath.unlink()
        await document_repo.delete(db, doc_id)
        return True


document_service = DocumentService()