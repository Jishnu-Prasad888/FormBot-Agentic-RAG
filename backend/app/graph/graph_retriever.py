import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.graph.entity_extraction import extract_entities
from app.graph.neo4j_client import neo4j_client

logger = logging.getLogger(__name__)


@dataclass
class GraphResult:
    candidate_document_ids: Set[str] = field(default_factory=set)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)


async def retrieve_candidates(query: str, filters: Optional[dict] = None) -> GraphResult:
    """
    Run entity extraction then use Neo4j to fetch candidate documents/forms.
    Falls back to empty result when graph is disabled or unavailable.
    """
    if not neo4j_client.enabled:
        return GraphResult()

    extraction = await extract_entities(query)
    entities = list({e.lower() for e in extraction.get("entities", []) if isinstance(e, str)})
    if not entities and filters:
        for key in ("category", "form_name", "document_type"):
            val = (filters or {}).get(key)
            if isinstance(val, str):
                entities.append(val.lower())

    if not entities:
        return GraphResult()

    try:
        cypher = (
            "MATCH (d:Document) "
            "WHERE any(e IN $entities WHERE toLower(coalesce(d.filename,'')) CONTAINS e "
            "  OR toLower(coalesce(d.category,'')) CONTAINS e "
            "  OR toLower(coalesce(d.form_name,'')) CONTAINS e "
            "  OR toLower(coalesce(d.title,'')) CONTAINS e) "
            "WITH DISTINCT d LIMIT $max_docs "
            "OPTIONAL MATCH (f:Form)-[:USES]->(d) "
            "RETURN d.document_id AS document_id, collect(DISTINCT f.name) AS forms, d.category AS category"
        )
        rows = await neo4j_client.run_read(
            cypher,
            {"entities": entities, "max_docs": settings.KG_MAX_DOCS},
        )
    except Exception as exc:
        logger.warning("Graph retrieval failed, falling back: %s", exc)
        return GraphResult()

    doc_ids: Set[str] = set()
    forms: List[Dict[str, Any]] = []
    concepts: List[str] = []
    for row in rows:
        doc_id = row.get("document_id")
        if doc_id:
            doc_ids.add(doc_id)
        for form_name in row.get("forms", []) or []:
            if form_name:
                forms.append({"name": form_name})
        cat = row.get("category")
        if cat:
            concepts.append(cat)

    return GraphResult(
        candidate_document_ids=doc_ids,
        forms=forms,
        concepts=list({c for c in concepts}),
        relationships=extraction.get("relationships", []),
    )
