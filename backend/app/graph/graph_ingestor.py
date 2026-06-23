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


async def upsert_form_version(form_name: str, version: str, status: Optional[str] = None, supersedes: Optional[str] = None):
    if not neo4j_client.enabled:
        return
    cypher = (
        "MERGE (f:Form {name: $name}) "
        "MERGE (v:FormVersion {name: $name, version: $version}) "
        "MERGE (v)-[:VERSION_OF]->(f) "
        "SET v.status=$status"
    )
    params = {"name": form_name, "version": version, "status": status}
    try:
        await neo4j_client.run_write(cypher, params)
        if supersedes:
            sup_cypher = (
                "MATCH (v:FormVersion {name:$name, version:$version}) "
                "MATCH (s:FormVersion {name:$name, version:$supersedes}) "
                "MERGE (v)-[:SUPERSEDES]->(s)"
            )
            await neo4j_client.run_write(sup_cypher, {"name": form_name, "version": version, "supersedes": supersedes})
    except Exception as exc:
        logger.warning("Failed to upsert form version: %s", exc)


async def upsert_field(form_name: str, version: str, field_name: str, field_type: Optional[str] = None, required: Optional[bool] = None):
    if not neo4j_client.enabled:
        return
    cypher = (
        "MATCH (v:FormVersion {name:$name, version:$version}) "
        "MERGE (fld:Field {name:$field_name}) "
        "SET fld.type=$field_type, fld.required=$required "
        "MERGE (v)-[:REQUIRES]->(fld)"
    )
    try:
        await neo4j_client.run_write(cypher, {
            "name": form_name,
            "version": version,
            "field_name": field_name,
            "field_type": field_type,
            "required": required,
        })
    except Exception as exc:
        logger.warning("Failed to upsert field: %s", exc)


async def link_field_dependency(source_field: str, target_field: str, condition: Optional[str] = None):
    if not neo4j_client.enabled:
        return
    cypher = (
        "MATCH (s:Field {name:$source}), (t:Field {name:$target}) "
        "MERGE (s)-[r:DEPENDS_ON]->(t) "
        "SET r.condition=$condition"
    )
    try:
        await neo4j_client.run_write(cypher, {"source": source_field, "target": target_field, "condition": condition})
    except Exception as exc:
        logger.warning("Failed to link field dependency: %s", exc)


async def upsert_regulation(title: str, citation: Optional[str], authority: Optional[str]):
    if not neo4j_client.enabled:
        return
    cypher = (
        "MERGE (r:Regulation {title:$title}) "
        "SET r.citation=$citation, r.authority=$authority"
    )
    try:
        await neo4j_client.run_write(cypher, {"title": title, "citation": citation, "authority": authority})
    except Exception as exc:
        logger.warning("Failed to upsert regulation: %s", exc)


async def link_form_regulation(form_name: str, version: str, regulation_title: str, relation_type: Optional[str] = None):
    if not neo4j_client.enabled:
        return
    cypher = (
        "MATCH (v:FormVersion {name:$name, version:$version}), (r:Regulation {title:$reg_title}) "
        "MERGE (v)-[rel:REFERENCES]->(r) "
        "SET rel.type=$relation_type"
    )
    try:
        await neo4j_client.run_write(cypher, {
            "name": form_name,
            "version": version,
            "reg_title": regulation_title,
            "relation_type": relation_type,
        })
    except Exception as exc:
        logger.warning("Failed to link form to regulation: %s", exc)


async def upsert_requirement(description: str, regulation_ref: Optional[str] = None):
    if not neo4j_client.enabled:
        return
    cypher = (
        "MERGE (req:Requirement {description:$desc}) "
        "SET req.regulation_ref=$reg_ref"
    )
    try:
        await neo4j_client.run_write(cypher, {"desc": description, "reg_ref": regulation_ref})
    except Exception as exc:
        logger.warning("Failed to upsert requirement: %s", exc)


async def link_form_requirement(form_name: str, version: str, description: str):
    if not neo4j_client.enabled:
        return
    cypher = (
        "MATCH (v:FormVersion {name:$name, version:$version}), (req:Requirement {description:$desc}) "
        "MERGE (v)-[:REQUIRES]->(req)"
    )
    try:
        await neo4j_client.run_write(cypher, {"name": form_name, "version": version, "desc": description})
    except Exception as exc:
        logger.warning("Failed to link form to requirement: %s", exc)
