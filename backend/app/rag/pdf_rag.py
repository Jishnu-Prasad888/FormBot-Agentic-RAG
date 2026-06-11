import io
import re
import uuid
from typing import Any, Optional
import pdfplumber
from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client
from app.core.config import settings

PDF_COLLECTION = "pdf_documents"


def _detect_heading(text: str) -> Optional[str]:
    stripped = text.strip()
    # Numbered headings: "1.", "1.2", "1.2.3" followed by title text
    if re.match(r"^\d+(\.\d+)*\.?\s+\S.{0,80}$", stripped):
        return stripped
    # ALL-CAPS short lines
    if len(stripped) < 100 and stripped.isupper() and len(stripped) > 2:
        return stripped
    # Title-case short lines (no trailing punctuation except colon)
    if re.match(r"^[A-Z][A-Za-z\s\-]{2,60}:?$", stripped) and len(stripped) < 80:
        return stripped
    return None


class PDFHierarchicalRAG:
    async def index(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        extra_metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        ids, embeddings, documents, metadatas = [], [], [], []
        current_section = "Introduction"
        chunk_index = 0

        with pdfplumber.open(io.BytesIO(content)) as pdf:
            full_text_parts = []
            page_texts = []
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                page_texts.append((page_num + 1, page_text))
                full_text_parts.append(page_text)

        # Chunk by page with section tracking
        for page_num, page_text in page_texts:
            if not page_text.strip():
                continue
            lines = page_text.split("\n")
            current_para = []
            for line in lines:
                heading = _detect_heading(line)
                if heading:
                    # flush current para
                    if current_para:
                        chunk_text = " ".join(current_para).strip()
                        if len(chunk_text) > 50:
                            chunk_id = f"{document_id}_p{page_num}_c{chunk_index}"
                            emb = await ollama_client.embeddings(chunk_text)
                            ids.append(chunk_id)
                            embeddings.append(emb)
                            documents.append(chunk_text)
                            metadatas.append({
                                "document_id": document_id,
                                "filename": filename,
                                "document_type": "pdf",
                                "section": current_section,
                                "page_number": page_num,
                                "chunk_index": chunk_index,
                                **(extra_metadata or {}),
                            })
                            chunk_index += 1
                        current_para = []
                    current_section = heading
                else:
                    current_para.append(line)

            # flush remaining
            if current_para:
                chunk_text = " ".join(current_para).strip()
                if len(chunk_text) > 50:
                    chunk_id = f"{document_id}_p{page_num}_c{chunk_index}"
                    emb = await ollama_client.embeddings(chunk_text)
                    ids.append(chunk_id)
                    embeddings.append(emb)
                    documents.append(chunk_text)
                    metadatas.append({
                        "document_id": document_id,
                        "filename": filename,
                        "document_type": "pdf",
                        "section": current_section,
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        **(extra_metadata or {}),
                    })
                    chunk_index += 1

        if ids:
            chroma_client.add_documents(PDF_COLLECTION, ids, embeddings, documents, metadatas)
        return {"document_id": document_id, "chunk_count": len(ids)}

    async def query(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        section: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query_emb = await ollama_client.embeddings(query)
        where_conditions = []
        if document_id:
            where_conditions.append({"document_id": {"$eq": document_id}})
        if section:
            where_conditions.append({"section": {"$eq": section}})
        where = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}
        return chroma_client.search(PDF_COLLECTION, query_emb, top_k, where)


pdf_rag = PDFHierarchicalRAG()