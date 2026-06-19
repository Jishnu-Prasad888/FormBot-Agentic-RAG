import logging
from typing import Any, Dict, Optional

from app.core.config import settings
from app.graph.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


async def upsert_document_node(doc: Dict[str, Any]) -> None:
    """Seed a Document node using available metadata (no GPT required)."""
    if not neo4j_client.enabled:
        return
    cypher = (
        "MERGE (d:Document {document_id: $id}) "
        "SET d.filename=$filename, d.title=$title, d.category=$category, d.source=$source, d.form_name=$form_name"
    )
    params = {
        "id": doc.get("id"),
        "filename": doc.get("filename"),
        "title": doc.get("title"),
        "category": doc.get("category"),
        "source": doc.get("source"),
        "form_name": doc.get("form_name"),
    }
    try:
        await neo4j_client.run_write(cypher, params)
    except Exception as exc:
        logger.warning("Failed to upsert document node: %s", exc)


async def link_document_to_category(doc_id: str, category: Optional[str]) -> None:
    if not neo4j_client.enabled or not category:
        return
    cypher = (
        "MERGE (c:Concept {name: $category}) "
        "WITH c MATCH (d:Document {document_id: $doc_id}) "
        "MERGE (d)-[:RELATED_TO]->(c)"
    )
    try:
        await neo4j_client.run_write(cypher, {"doc_id": doc_id, "category": category})
    except Exception as exc:
        logger.warning("Failed to link document to category: %s", exc)


async def upsert_form(name: str, category: Optional[str], description: Optional[str] = None):
    if not neo4j_client.enabled:
        return
    cypher = (
        "MERGE (f:Form {name: $name}) "
        "SET f.category=$category, f.description=$description"
    )
    try:
        await neo4j_client.run_write(cypher, {"name": name, "category": category, "description": description})
    except Exception as exc:
        logger.warning("Failed to upsert form: %s", exc)


async def connect_form_to_document(form_name: Optional[str], doc_id: str) -> None:
    if not neo4j_client.enabled or not form_name:
        return
    cypher = (
        "MATCH (d:Document {document_id: $doc_id}) "
        "MERGE (f:Form {name: $form_name}) "
        "MERGE (f)-[:USES]->(d)"
    )
    try:
        await neo4j_client.run_write(cypher, {"doc_id": doc_id, "form_name": form_name})
    except Exception as exc:
        logger.warning("Failed to connect form to document: %s", exc)
