import logging
from typing import Any, Dict, List

from app.core.config import settings
from app.graph.entity_extraction import extract_entities
from app.graph.graph_retriever import retrieve_candidates
from app.graph.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


async def recommend_forms(query: str) -> Dict[str, Any]:
    """Return recommended forms plus requirements/eligibility/procedure when present."""
    if not neo4j_client.enabled:
        return {"forms": [], "documents": []}

    extraction = await extract_entities(query)
    entities = [e for e in extraction.get("entities", []) if isinstance(e, str)]

    # First, reuse candidate retrieval to collect forms tied to documents
    base = await retrieve_candidates(query)
    forms = base.forms.copy()

    try:
        if entities:
            cypher = (
                "MATCH (f:Form) "
                "WHERE any(e IN $entities WHERE toLower(f.name) CONTAINS toLower(e) OR toLower(coalesce(f.category,'')) CONTAINS toLower(e)) "
                "WITH DISTINCT f LIMIT 20 "
                "OPTIONAL MATCH (f)-[:HAS_ELIGIBILITY]->(el) "
                "OPTIONAL MATCH (f)-[:HAS_PROCEDURE]->(p) "
                "OPTIONAL MATCH (f)-[:REQUIRES]->(req:Field) "
                "RETURN f.name AS name, f.category AS category, f.description AS description, "
                "collect(DISTINCT req.name) AS requirements, collect(DISTINCT el.name) AS eligibility, collect(DISTINCT p.name) AS procedure"
            )
            rows = await neo4j_client.run_read(cypher, {"entities": [e.lower() for e in entities]})
            for row in rows:
                forms.append({
                    "name": row.get("name"),
                    "category": row.get("category"),
                    "description": row.get("description"),
                    "requirements": [r for r in row.get("requirements", []) if r],
                    "eligibility": [r for r in row.get("eligibility", []) if r],
                    "procedure": [r for r in row.get("procedure", []) if r],
                })
    except Exception as exc:
        logger.warning("Form recommendation graph query failed: %s", exc)

    # Deduplicate by form name
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for form in forms:
        name = form.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(form)

    return {"forms": deduped[: settings.TOP_K], "documents": list(base.candidate_document_ids)}
