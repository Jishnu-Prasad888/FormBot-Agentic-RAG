from typing import Any


def reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = 60, top_k: int = 5
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    Each result must have a 'chunk_id' field.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            chunk_id = item.get("chunk_id", "")
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = item

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
    results = []
    for chunk_id in sorted_ids:
        item = chunk_data[chunk_id].copy()
        item["score"] = round(scores[chunk_id], 6)
        results.append(item)
    return results
