import io
import uuid
import json
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import hashlib
import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.document_repository import document_repo
from app.repositories.kag_repository import kag_repo
from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client
from app.rag.table_rag import table_rag
from app.rag.pdf_rag import pdf_rag
from app.rag.markdown_rag import markdown_rag
from app.rag.bm25 import bm25_retriever
from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
from app.graph import graph_ingestor
from app.graph import graph_ingestor as gi


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


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 250) -> list[str]:
    """Character-based chunking with overlap."""
    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += step
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
        chunk_id = str(uuid.uuid4())
        ids.append(chunk_id)
        embeddings.append(emb)
        documents.append(chunk_text)
        meta = {
            "document_id": document_id,
            "filename": filename,
            "document_type": doc_type,
            "chunk_index": i,
            "chunk_id": chunk_id,
            "chunk_type": "paragraph",
            "chunk_position": i,
            **(extra_metadata or {}),
        }
        metadatas.append(meta)
        chunk_records.append({
            "id": chunk_id,
            "document_id": document_id,
            "chunk_index": i,
            "chunk_text": chunk_text,
            "chunk_metadata": meta,
            "metadata_json": meta,
            "qdrant_point_id": chunk_id,
            "vector_id": chunk_id,
            "chunk_type": meta.get("chunk_type"),
            "content_summary": meta.get("content_summary"),
            "extracted_entities": meta.get("extracted_entities"),
            "section": meta.get("section"),
            "field_name": meta.get("field_name"),
            "requirement_tags": meta.get("requirement_tags"),
            "regulatory_reference": meta.get("regulatory_reference"),
            "confidence_score": meta.get("confidence_score"),
            "chunk_position": meta.get("chunk_position"),
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
    async def _ingest_kag_structures(
        self,
        db: AsyncSession,
        doc_data: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Persist structured form/regulation data to Postgres + Neo4j."""
        form_name = metadata.get("form_name")
        form_version = metadata.get("version") or metadata.get("form_version") or "v1"
        kag_type = metadata.get("kag_type")

        # Only proceed if this looks like a form/regulation payload
        if kag_type not in {"form", "regulation", "guideline"} and not form_name:
            return

        # Postgres: FormVersion + Fields + Requirements + Regulations
        fv_id = str(uuid.uuid4())
        await kag_repo.upsert_form_version(db, metadata.get("form_id") or str(uuid.uuid4()), form_version, {
            "id": fv_id,
            "status": metadata.get("status"),
            "effective_date": metadata.get("effective_date"),
            "supersedes_id": metadata.get("supersedes_id"),
        })

        fields = metadata.get("fields") or []
        field_id_map: dict[str, str] = {}
        field_rows = []
        for f in fields:
            fid = f.get("id") or str(uuid.uuid4())
            field_id_map[f.get("name")] = fid
            field_rows.append({
                "id": fid,
                "form_version_id": fv_id,
                "name": f.get("name"),
                "field_type": f.get("type") or f.get("field_type"),
                "validation_rules": f.get("validation_rules"),
                "required": bool(f.get("required", False)),
                "description": f.get("description"),
            })
        await kag_repo.bulk_upsert_fields(db, field_rows)

        # Field dependencies
        deps = []
        for f in fields:
            source = field_id_map.get(f.get("name"))
            for dep in f.get("depends_on", []) or []:
                target = field_id_map.get(dep.get("field"))
                if source and target:
                    deps.append({
                        "id": str(uuid.uuid4()),
                        "source_field_id": source,
                        "target_field_id": target,
                        "condition": dep.get("condition"),
                    })
        await kag_repo.bulk_upsert_field_dependencies(db, deps)

        # Regulations and requirements
        regulation_rows = []
        for r in metadata.get("regulations", []) or []:
            rid = r.get("id") or str(uuid.uuid4())
            regulation_rows.append({
                "id": rid,
                "title": r.get("title") or r.get("citation") or "regulation",
                "authority": r.get("authority"),
                "effective_date": r.get("effective_date"),
                "citation": r.get("citation"),
                "description": r.get("description"),
            })
        for row in regulation_rows:
            await kag_repo.upsert_regulation(db, row)

        requirements = []
        req_links = []
        for req in metadata.get("requirements", []) or []:
            req_id = req.get("id") or str(uuid.uuid4())
            requirements.append({
                "id": req_id,
                "description": req.get("description") or "",
                "applicability": req.get("applicability"),
                "regulation_id": req.get("regulation_id"),
                "regulation_ref": req.get("regulation_ref"),
            })
            req_links.append({
                "id": str(uuid.uuid4()),
                "form_version_id": fv_id,
                "requirement_id": req_id,
                "applies_if": req.get("applies_if"),
            })
        await kag_repo.bulk_upsert_requirements(db, requirements)
        await kag_repo.bulk_upsert_form_requirements(db, req_links)

        # Form → Regulation links
        form_reg_links = []
        for r in regulation_rows:
            form_reg_links.append({
                "id": str(uuid.uuid4()),
                "form_version_id": fv_id,
                "regulation_id": r.get("id"),
                "relation_type": "REFERENCES",
            })
        await kag_repo.bulk_upsert_form_regulations(db, form_reg_links)

        # Graph: forms, versions, fields, dependencies, regulations, links
        try:
            if form_name:
                await gi.upsert_form(form_name, metadata.get("category"))
                await gi.upsert_form_version(form_name, form_version, metadata.get("status"), metadata.get("supersedes"))
            for f in fields:
                await gi.upsert_field(form_name or "", form_version, f.get("name"), f.get("type"), f.get("required"))
                for dep in f.get("depends_on", []) or []:
                    await gi.link_field_dependency(f.get("name"), dep.get("field"), dep.get("condition"))
            for r in regulation_rows:
                await gi.upsert_regulation(r.get("title"), r.get("citation"), r.get("authority"))
                if form_name:
                    await gi.link_form_regulation(form_name, form_version, r.get("title"), "REFERENCES")
            for req in requirements:
                await gi.upsert_requirement(req.get("description"), req.get("regulation_ref"))
                if form_name:
                    await gi.link_form_requirement(form_name, form_version, req.get("description"))
        except Exception:
            pass
    async def upload_and_index(
        self,
        db: AsyncSession,
        filename: str,
        content: bytes,
        extra_metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        doc_type = _detect_type(filename)
        doc_id = str(uuid.uuid4())
        kag_type = (extra_metadata or {}).get("kag_type")  # form | regulation | guideline
        collection = TYPE_TO_COLLECTION[doc_type]
        if kag_type in {"form", "regulation", "guideline"}:
            collection = {
                "form": "bank_forms_collection",
                "regulation": "regulations_collection",
                "guideline": "guidelines_collection",
            }[kag_type]
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
        content_hash = hashlib.sha256(content).hexdigest()
        document_version = (extra_metadata or {}).get("version", "v1")
        base_meta = {
            "document_version": document_version,
            "content_hash": content_hash,
            "kag_type": kag_type,
        }

        if doc_type == "pdf":
            result = await pdf_rag.index(doc_id, filename, content, {**(extra_metadata or {}), **base_meta})
            chunk_count = result["chunk_count"]
            chunk_records = result.get("chunks", [])
        elif doc_type == "markdown":
            text = content.decode("utf-8", errors="replace")
            result = await markdown_rag.index(doc_id, filename, text, {**(extra_metadata or {}), **base_meta})
            chunk_count = result["chunk_count"]
            chunk_records = result.get("chunks", [])
        elif doc_type == "csv":
            result = await table_rag.index_csv(doc_id, filename, content, {**(extra_metadata or {}), **base_meta})
            chunk_count = result["chunk_count"]
            chunk_records = result.get("chunks", [])
        else:
            # text / json
            text = content.decode("utf-8", errors="replace")
            chunk_records = await _index_text_chunks(doc_id, filename, doc_type, text, collection, {**(extra_metadata or {}), **base_meta})
            chunk_count = len(chunk_records)

        doc_data = {
            "id": doc_id,
            "filename": filename,
            "filepath": str(filepath),
            "document_type": doc_type,
            "title": (extra_metadata or {}).get("title"),
            "category": (extra_metadata or {}).get("category"),
            "source": (extra_metadata or {}).get("source"),
            "retrieval_strategy": strategy,
            "language": (extra_metadata or {}).get("language", "en"),
            "chunk_count": chunk_count,
            "embedding_model": settings.OPENAI_EMBED_MODEL,
            "collection_name": collection,
            "form_name": (extra_metadata or {}).get("form_name"),
            "metadata_json": {**(extra_metadata or {}), **base_meta},
            "content_hash": content_hash,
            "document_version": document_version,
            "processing_status": "indexed",
            "processing_log": None,
        }

        doc = await document_repo.create(db, doc_data)

        # Seed graph (metadata-only pass)
        try:
            await graph_ingestor.upsert_document_node(doc_data)
            await graph_ingestor.link_document_to_category(doc_id, doc_data.get("category"))
            await graph_ingestor.connect_form_to_document(doc_data.get("form_name"), doc_id)
            if doc_data.get("form_name"):
                await graph_ingestor.upsert_form(doc_data.get("form_name"), doc_data.get("category"))
                await graph_ingestor.upsert_form_version(doc_data.get("form_name"), document_version)
        except Exception:
            pass

        # Persist structured KAG metadata (forms/fields/requirements/regulations)
        try:
            await self._ingest_kag_structures(db, doc_data, doc_data.get("metadata_json") or {})
        except Exception:
            pass

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
        kag_type = (doc.metadata_json or {}).get("kag_type")
        if kag_type in {"form", "regulation", "guideline"}:
            collection = {
                "form": "bank_forms_collection",
                "regulation": "regulations_collection",
                "guideline": "guidelines_collection",
            }[kag_type]
        chunk_count = 0
        base_meta = {
            "document_version": (doc.metadata_json or {}).get("document_version", "v1"),
            "content_hash": doc.metadata_json.get("content_hash") if doc.metadata_json else None,
            "kag_type": kag_type,
        }

        if doc_type == "pdf":
            result = await pdf_rag.index(doc_id, doc.filename, content, {**(doc.metadata_json or {}), **base_meta})
            chunk_count = result["chunk_count"]
        elif doc_type == "markdown":
            text = content.decode("utf-8", errors="replace")
            result = await markdown_rag.index(doc_id, doc.filename, text, {**(doc.metadata_json or {}), **base_meta})
            chunk_count = result["chunk_count"]
            chunk_records = result.get("chunks", [])
        elif doc_type == "csv":
            result = await table_rag.index_csv(doc_id, doc.filename, content, {**(doc.metadata_json or {}), **base_meta})
            chunk_count = result["chunk_count"]
            chunk_records = result.get("chunks", [])
        else:
            text = content.decode("utf-8", errors="replace")
            chunk_records = await _index_text_chunks(doc_id, doc.filename, doc_type, text, collection, {**(doc.metadata_json or {}), **base_meta})
            chunk_count = len(chunk_records)

        if chunk_records:
            await document_repo.bulk_create_chunks(db, chunk_records)

        await document_repo.update(db, doc_id, {"chunk_count": chunk_count})

        try:
            await graph_ingestor.upsert_document_node({
                "id": doc.id,
                "filename": doc.filename,
                "title": doc.metadata_json.get("title") if doc.metadata_json else None,
                "category": doc.metadata_json.get("category") if doc.metadata_json else None,
                "source": doc.metadata_json.get("source") if doc.metadata_json else None,
                "form_name": doc.metadata_json.get("form_name") if doc.metadata_json else None,
            })
            await graph_ingestor.link_document_to_category(doc.id, doc.metadata_json.get("category") if doc.metadata_json else None)
            await graph_ingestor.connect_form_to_document(doc.metadata_json.get("form_name") if doc.metadata_json else None, doc.id)
        except Exception:
            pass
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
