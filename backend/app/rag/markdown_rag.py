import re
from typing import Any, Optional
from app.chromadb.client import chroma_client
from app.embeddings.ollama_client import ollama_client
from app.core.logging import get_logger

logger = get_logger("markdown_rag")
MD_COLLECTION = "markdown_documents"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _parse_markdown_sections(text: str) -> list[dict]:
    """Split markdown by headings, preserving code blocks and links."""
    sections = []
    pos = 0
    current_heading = "Introduction"
    current_level = 1

    for match in HEADING_RE.finditer(text):
        chunk = text[pos:match.start()].strip()
        if chunk:
            sections.append({
                "heading": current_heading,
                "level": current_level,
                "content": chunk,
            })
        current_heading = match.group(2).strip()
        current_level = len(match.group(1))
        pos = match.end()

    remainder = text[pos:].strip()
    if remainder:
        sections.append({
            "heading": current_heading,
            "level": current_level,
            "content": remainder,
        })
    return sections


class MarkdownRAG:
    async def index(
        self,
        document_id: str,
        filename: str,
        content: str,
        extra_metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        sections = _parse_markdown_sections(content)
        ids, embeddings, documents, metadatas = [], [], [], []
        for i, section in enumerate(sections):
            text = f"# {section['heading']}\n\n{section['content']}"
            if len(text.strip()) < 20:
                continue
            emb = await ollama_client.embeddings(text)
            chunk_id = f"{document_id}_sec{i}"
            ids.append(chunk_id)
            embeddings.append(emb)
            documents.append(text)
            metadatas.append({
                "document_id": document_id,
                "filename": filename,
                "document_type": "markdown",
                "section": section["heading"],
                "heading_level": section["level"],
                "chunk_index": i,
                **(extra_metadata or {}),
            })

        if ids:
            chroma_client.add_documents(MD_COLLECTION, ids, embeddings, documents, metadatas)
        logger.info(f"Markdown indexed '{filename}': {len(ids)} sections")
        return {"document_id": document_id, "chunk_count": len(ids)}

    async def query(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_emb = await ollama_client.embeddings(query)
        where = None
        if document_id:
            where = {"document_id": {"$eq": document_id}}
        return chroma_client.search(MD_COLLECTION, query_emb, top_k, where)


markdown_rag = MarkdownRAG()
