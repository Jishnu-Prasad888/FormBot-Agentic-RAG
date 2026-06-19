from typing import Any, Optional


SUPPORTED_FILTERS = {
    "filename", "document_type", "section", "language",
    "date", "document_id", "retrieval_strategy", "state",
    "ministry", "department", "source", "category", "form_name",
}


def build_chroma_filter(filters: dict[str, Any], candidate_document_ids: Optional[set[str]] = None) -> Optional[dict]:
    """
    Build a vector-store friendly filter dict (Chroma-compatible) with support for
    document whitelisting via $in. Qdrant adapter also understands this shape.
    """
    if not filters and not candidate_document_ids:
        return None

    valid = {k: v for k, v in (filters or {}).items() if k in SUPPORTED_FILTERS and v is not None}

    clauses = []
    for k, v in valid.items():
        clauses.append({k: {"$eq": v}})

    if candidate_document_ids:
        clauses.append({"document_id": {"$in": list(candidate_document_ids)}})

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def filter_results(
    results: list[dict],
    filters: dict[str, Any],
    candidate_document_ids: Optional[set[str]] = None,
) -> list[dict]:
    """In-memory metadata filtering for BM25 or pre-fetched results."""
    if not filters and not candidate_document_ids:
        return results
    filtered = []
    for item in results:
        meta = item.get("metadata", {})
        match = all(
            meta.get(k) == v
            for k, v in (filters or {}).items()
            if k in SUPPORTED_FILTERS and v is not None
        )
        if candidate_document_ids is not None and meta.get("document_id") not in candidate_document_ids:
            match = False
        if match:
            filtered.append(item)
    return filtered
