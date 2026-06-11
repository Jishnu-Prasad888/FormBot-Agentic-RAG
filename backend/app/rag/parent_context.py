from typing import Any, Optional
from app.chromadb.client import chroma_client



class ParentContextExpander:
    """Expand retrieved chunks with parent (previous/next) chunks."""

    async def expand(
        self,
        chunks: list[dict[str, Any]],
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """
        For each chunk, retrieve adjacent chunks (previous and next).
        Preserve order: previous, current, next (if they exist).
        """
        if not chunks:
            return chunks

        expanded = []
        doc_chunk_index_map = {}

        # Build map: (document_id, chunk_index) -> chunk
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("document_id")
            chunk_idx = meta.get("chunk_index", -1)
            if doc_id and chunk_idx >= 0:
                doc_chunk_index_map[(doc_id, chunk_idx)] = chunk

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("document_id")
            chunk_idx = meta.get("chunk_index", -1)

            if not doc_id or chunk_idx < 0:
                expanded.append(chunk)
                continue

            expanded_chunk = {**chunk}
            parent_chunks = []

            # Previous chunk
            prev_key = (doc_id, chunk_idx - 1)
            if prev_key in doc_chunk_index_map:
                prev = doc_chunk_index_map[prev_key]
                parent_chunks.append(("prev", prev))

            # Current chunk already in expanded_chunk

            # Next chunk
            next_key = (doc_id, chunk_idx + 1)
            if next_key in doc_chunk_index_map:
                nxt = doc_chunk_index_map[next_key]
                parent_chunks.append(("next", nxt))

            # Augment chunk_text with parent context
            if parent_chunks:
                original_text = expanded_chunk.get("chunk_text", "")
                ctx_parts = []

                for rel_type, parent in parent_chunks:
                    parent_text = parent.get("chunk_text", "")
                    if rel_type == "prev":
                        ctx_parts.insert(0, f"[PREVIOUS]\n{parent_text}\n")
                    elif rel_type == "next":
                        ctx_parts.append(f"\n[NEXT]\n{parent_text}")

                expanded_chunk["chunk_text"] = "".join(ctx_parts) + original_text
                expanded_chunk["parent_chunks_count"] = len(parent_chunks)

            expanded.append(expanded_chunk)

        return expanded


parent_context_expander = ParentContextExpander()
