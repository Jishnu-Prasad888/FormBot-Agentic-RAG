from typing import Any
from rank_bm25 import BM25Okapi



class BM25Retriever:
    def __init__(self):
        self._index: dict[str, BM25Okapi] = {}
        self._corpus: dict[str, list[dict]] = {}

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def index(self, collection_name: str, chunks: list[dict]):
        """chunks: list of {chunk_id, chunk_text, metadata, document_id, filename}"""
        self._corpus[collection_name] = chunks
        tokenized = [self._tokenize(c["chunk_text"]) for c in chunks]
        self._index[collection_name] = BM25Okapi(tokenized)

    def search(self, collection_name: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if collection_name not in self._index:
            return []
        bm25 = self._index[collection_name]
        corpus = self._corpus[collection_name]
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = corpus[idx]
                results.append({
                    "chunk_id": chunk.get("chunk_id", str(idx)),
                    "chunk_text": chunk["chunk_text"],
                    "metadata": chunk.get("metadata", {}),
                    "document_id": chunk.get("document_id", ""),
                    "filename": chunk.get("filename", ""),
                    "score": float(scores[idx]),
                })
        return results

    def remove_collection(self, collection_name: str):
        self._index.pop(collection_name, None)
        self._corpus.pop(collection_name, None)


bm25_retriever = BM25Retriever()
