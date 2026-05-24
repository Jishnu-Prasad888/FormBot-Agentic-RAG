from typing import Any, Optional


SUPPORTED_FILTERS = {
    "filename", "document_type", "section", "language",
    "date", "document_id", "retrieval_strategy", "state",
    "ministry", "department", "source",
}


def build_chroma_filter(filters: dict[str, Any]) -> Optional[dict]:
    """Build a ChromaDB $and/$eq compatible filter dict."""
    if not filters:
        return None
    valid = {k: v for k, v in filters.items() if k in SUPPORTED_FILTERS and v is not None}
    if not valid:
        return None
    if len(valid) == 1:
        key, val = next(iter(valid.items()))
        return {key: {"$eq": val}}
    return {"$and": [{k: {"$eq": v}} for k, v in valid.items()]}


def filter_results(results: list[dict], filters: dict[str, Any]) -> list[dict]:
    """In-memory metadata filtering for BM25 results."""
    if not filters:
        return results
    filtered = []
    for item in results:
        meta = item.get("metadata", {})
        match = all(
            meta.get(k) == v
            for k, v in filters.items()
            if k in SUPPORTED_FILTERS and v is not None
        )
        if match:
            filtered.append(item)
    return filtered
