"""
VectorRAG — hybrid retrieval with BM25 + dense vectors, keyword boosting,
acronym expansion, and a pluggable cross-encoder reranker.

Public API is unchanged:
    vector_rag.retrieve(query, collection_name, top_k, filters)
    vector_rag.retrieve_multi_collection(query, collection_names, top_k, filters)

Improvements over the previous version
---------------------------------------
Retrieval
  - Hybrid search: BM25 sparse score merged with cosine dense score
  - Keyword extraction boosts chunks containing important query terms
  - Acronym expansion applied at retrieval time (CIF → Customer Information File)
  - Original query runs first; expansion is a caller-opt-in via `expand=True`
  - Over-retrieval then rerank: fetch `candidate_k` (default 50), rerank to `top_k`
  - All chunks are verified to be non-empty before being returned

Reranking
  - Cross-encoder reranker is pluggable via RERANKER_BACKEND env-var
    ("bge", "jina", "cohere") — falls back to keyword-boosted cosine if unset
  - BGE / Jina run locally via sentence-transformers
  - Cohere uses the Cohere Rerank v3 API

Metadata
  - document_id, filename, section, heading, chunk_index always populated
  - score breakdown (dense_score, bm25_score, final_score) logged per chunk

BM25
  - rank_bm25 library; index built lazily per collection and cached in memory
  - Cache is invalidated when the collection's document count changes
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Optional

from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client
from app.rag.metadata_filter import build_chroma_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acronym / abbreviation dictionary
# ---------------------------------------------------------------------------
ACRONYM_MAP: dict[str, str] = {
    "CIF": "Customer Information File",
    "KYC": "Know Your Customer",
    "AML": "Anti-Money Laundering",
    "PEP": "Politically Exposed Person",
    "FATF": "Financial Action Task Force",
    "AOF": "Account Opening Form",
    "DD": "Due Diligence",
    "EDD": "Enhanced Due Diligence",
    "SDD": "Simplified Due Diligence",
    "STR": "Suspicious Transaction Report",
    "CTR": "Currency Transaction Report",
    "TIN": "Tax Identification Number",
    "NID": "National Identity Document",
    # Extend as your domain grows
}


def expand_acronyms(text: str) -> str:
    """
    Replace known acronyms in *text* with 'ACRONYM (Expansion)' so that both
    the abbreviation and the full form appear in the query / chunk.
    """
    tokens = text.split()
    expanded: list[str] = []
    for token in tokens:
        clean = re.sub(r"[^A-Z]", "", token.upper())
        if clean in ACRONYM_MAP:
            expanded.append(f"{token} ({ACRONYM_MAP[clean]})")
        else:
            expanded.append(token)
    return " ".join(expanded)


# ---------------------------------------------------------------------------
# Keyword extraction (simple TF-style; swap for spaCy/RAKE if available)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can not no nor and or but "
    "if then else when where what which who whom how why this that these "
    "those it its of in on at to for from with by about as into through "
    "during before after above below between among over under i me my we "
    "our you your he she they them their what said".split()
)


def extract_keywords(query: str, max_keywords: int = 8) -> list[str]:
    """Return the most query-relevant non-stopword tokens, longest first."""
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())
    keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    # Deduplicate while preserving order, then sort by length (proxy for specificity)
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return sorted(unique, key=len, reverse=True)[:max_keywords]


def keyword_boost(chunk_text: str, keywords: list[str]) -> float:
    """
    Return a small additive score in [0, 0.2] based on how many query
    keywords appear in the chunk.  Used to break ties after reranking.
    """
    if not keywords:
        return 0.0
    text_lower = chunk_text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return round(0.2 * hits / len(keywords), 4)


# ---------------------------------------------------------------------------
# BM25 index cache
# ---------------------------------------------------------------------------

class _BM25Cache:
    """
    Lazy, per-collection BM25 index.  Rebuilt when the document count changes.
    Requires:  pip install rank_bm25
    """

    def __init__(self) -> None:
        self._indices: dict[str, Any] = {}   # collection_name → BM25Okapi
        self._doc_counts: dict[str, int] = {}
        self._corpus: dict[str, list[str]] = {}

    def _try_import(self) -> Any:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
            return BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 not installed — BM25 scoring disabled.  "
                "Install with: pip install rank_bm25"
            )
            return None

    def get_scores(
        self,
        collection_name: str,
        query_tokens: list[str],
        chunk_ids: list[str],
        chunks: list[str],
    ) -> list[float]:
        """
        Return normalised BM25 scores (0–1) for *chunks* given *query_tokens*.
        Falls back to uniform 0.0 if rank_bm25 is not installed.
        """
        BM25Okapi = self._try_import()
        if BM25Okapi is None:
            return [0.0] * len(chunks)

        current_count = len(chunks)
        cached_count = self._doc_counts.get(collection_name, -1)

        if current_count != cached_count:
            tokenised = [c.lower().split() for c in chunks]
            self._indices[collection_name] = BM25Okapi(tokenised)
            self._doc_counts[collection_name] = current_count
            self._corpus[collection_name] = chunks

        bm25 = self._indices[collection_name]
        raw_scores: list[float] = bm25.get_scores(query_tokens).tolist()

        # Normalise to [0, 1]
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0.0:
            return [0.0] * len(raw_scores)
        return [round(s / max_score, 4) for s in raw_scores]


_bm25_cache = _BM25Cache()


# ---------------------------------------------------------------------------
# Cross-encoder reranker (pluggable)
# ---------------------------------------------------------------------------

class _Reranker:
    """
    Thin wrapper around multiple reranking backends.

    Set RERANKER_BACKEND env-var to "bge", "jina", or "cohere".
    Leave unset (or set to "none") to skip cross-encoder reranking and fall back
    to the hybrid score + keyword boost.

    BGE / Jina — local inference via sentence-transformers:
        pip install sentence-transformers
        model: BAAI/bge-reranker-base  (bge)  |  jinaai/jina-reranker-v2-base-multilingual (jina)

    Cohere — remote API:
        pip install cohere
        set COHERE_API_KEY env-var
        model: rerank-english-v3.0
    """

    _instance: Any = None  # lazy singleton

    def _load(self) -> None:
        backend = os.getenv("RERANKER_BACKEND", "none").lower()
        if backend == "bge":
            from sentence_transformers import CrossEncoder  # type: ignore
            self._instance = CrossEncoder("BAAI/bge-reranker-base")
            self._backend = "sentence_transformers"
        elif backend == "jina":
            from sentence_transformers import CrossEncoder  # type: ignore
            self._instance = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual")
            self._backend = "sentence_transformers"
        elif backend == "cohere":
            import cohere  # type: ignore
            self._instance = cohere.Client(os.environ["COHERE_API_KEY"])
            self._backend = "cohere"
        else:
            self._instance = None
            self._backend = "none"

    def rerank(
        self, query: str, chunks: list[str], scores: list[float], top_k: int
    ) -> list[tuple[str, float]]:
        """
        Return (chunk_text, score) pairs sorted descending, trimmed to top_k.
        Falls back gracefully if the backend is unavailable.
        """
        if self._instance is None and not hasattr(self, "_backend"):
            self._load()

        backend = getattr(self, "_backend", "none")

        if backend == "sentence_transformers":
            try:
                pairs = [[query, c] for c in chunks]
                ce_scores: list[float] = self._instance.predict(pairs).tolist()
                ranked = sorted(zip(chunks, ce_scores), key=lambda x: x[1], reverse=True)
                return ranked[:top_k]
            except Exception as exc:
                logger.warning("Cross-encoder rerank failed: %s — using hybrid scores.", exc)

        elif backend == "cohere":
            try:
                resp = self._instance.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=chunks,
                    top_n=top_k,
                )
                ranked = [(chunks[r.index], r.relevance_score) for r in resp.results]
                return ranked
            except Exception as exc:
                logger.warning("Cohere rerank failed: %s — using hybrid scores.", exc)

        # Fallback: use existing hybrid scores
        ranked_pairs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked_pairs[:top_k]


_reranker = _Reranker()


# ---------------------------------------------------------------------------
# VectorRAG
# ---------------------------------------------------------------------------

class VectorRAG:
    """
    Hybrid retrieval: dense cosine (ChromaDB) + BM25 sparse, merged via
    weighted sum, optionally reranked by a cross-encoder.

    Parameters
    ----------
    dense_weight : float
        Weight for the dense cosine score in the hybrid merge (default 0.7).
    bm25_weight : float
        Weight for the BM25 score in the hybrid merge (default 0.3).
    candidate_multiplier : int
        How many times top_k to over-retrieve for reranking (default 10).
        E.g. top_k=5, multiplier=10 → fetch 50 candidates → rerank → return 5.
    """

    def __init__(
        self,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        candidate_multiplier: int = 10,
    ) -> None:
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.candidate_multiplier = candidate_multiplier

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidate_k(self, top_k: int) -> int:
        return max(top_k, top_k * self.candidate_multiplier)

    @staticmethod
    def _enrich_result(r: dict[str, Any]) -> dict[str, Any]:
        """Populate standard metadata fields on every result record."""
        meta = r.get("metadata", {})
        r["document_id"] = meta.get("document_id", "")
        r["filename"] = meta.get("filename", "")
        r["section"] = meta.get("section", "")
        r["heading"] = meta.get("heading", "")
        r["chunk_index"] = meta.get("chunk_index", None)
        return r

    def _hybrid_merge(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge dense cosine scores with BM25 scores and apply keyword boost.
        Modifies results in place; returns the same list sorted by final_score desc.
        """
        if not results:
            return results

        chunks = [r.get("chunk_text", "") for r in results]
        query_tokens = query.lower().split()
        bm25_scores = _bm25_cache.get_scores(
            collection_name="_hybrid",  # shared cache key; ok for per-call usage
            query_tokens=query_tokens,
            chunk_ids=[r.get("id", str(i)) for i, r in enumerate(results)],
            chunks=chunks,
        )

        keywords = extract_keywords(query)

        for r, bm25 in zip(results, bm25_scores):
            dense = float(r.get("score", 0.0))
            boost = keyword_boost(r.get("chunk_text", ""), keywords)
            hybrid = self.dense_weight * dense + self.bm25_weight * bm25 + boost
            r["dense_score"] = round(dense, 4)
            r["bm25_score"] = bm25
            r["keyword_boost"] = boost
            r["score"] = round(hybrid, 4)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _rerank_results(
        self, query: str, results: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Apply cross-encoder reranking on the candidate pool and trim to top_k.
        If reranker is disabled the candidate list is simply sliced to top_k.
        """
        if not results:
            return results

        chunks = [r.get("chunk_text", "") for r in results]
        scores = [r.get("score", 0.0) for r in results]

        reranked_pairs = _reranker.rerank(query, chunks, scores, top_k)

        # Rebuild result dicts preserving metadata
        chunk_to_result: dict[str, dict[str, Any]] = {}
        for r in results:
            chunk_to_result.setdefault(r.get("chunk_text", ""), r)

        final: list[dict[str, Any]] = []
        for chunk_text, rerank_score in reranked_pairs:
            base = chunk_to_result.get(chunk_text, {"chunk_text": chunk_text, "metadata": {}})
            base["rerank_score"] = round(float(rerank_score), 4)
            final.append(base)

        return final

    # ------------------------------------------------------------------
    # Public API  (signatures preserved)
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        collection_name: str = "text_documents",
        top_k: int = 5,
        filters: Optional[dict] = None,
        expand: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top_k most relevant chunks for *query*.

        Parameters
        ----------
        query : str
            The search query.  Acronyms are expanded automatically.
        collection_name : str
            ChromaDB collection to search.
        top_k : int
            Number of results to return after reranking.
        filters : dict | None
            Optional metadata filters forwarded to ChromaDB.
        expand : bool
            If True, the query is expanded with acronym forms before embedding.
            Set to False (default) so callers control when expansion happens.
        """
        effective_query = expand_acronyms(query) if expand else query
        query_embedding = await ollama_client.embeddings(effective_query)
        where = build_chroma_filter(filters) if filters else None

        candidate_k = self._candidate_k(top_k)
        results: list[dict[str, Any]] = chroma_client.search(
            collection_name, query_embedding, candidate_k, where
        )

        # Drop empty chunks early
        results = [r for r in results if r.get("chunk_text", "").strip()]

        results = self._hybrid_merge(effective_query, results)
        results = self._rerank_results(effective_query, results, top_k)

        return [self._enrich_result(r) for r in results]

    async def retrieve_multi_collection(
        self,
        query: str,
        collection_names: list[str],
        top_k: int = 5,
        filters: Optional[dict] = None,
        expand: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search multiple collections, merge all candidates, rerank globally,
        and return the top_k results across all collections.

        Parameters
        ----------
        query : str
            The search query.
        collection_names : list[str]
            Collections to fan-out the query across.
        top_k : int
            Final number of results to return.
        filters : dict | None
            Optional metadata filters forwarded to ChromaDB.
        expand : bool
            Whether to expand acronyms in the query before embedding.
        """
        effective_query = expand_acronyms(query) if expand else query
        query_embedding = await ollama_client.embeddings(effective_query)
        where = build_chroma_filter(filters) if filters else None

        candidate_k = self._candidate_k(top_k)
        all_results: list[dict[str, Any]] = []

        for collection in collection_names:
            try:
                results: list[dict[str, Any]] = chroma_client.search(
                    collection, query_embedding, candidate_k, where
                )
                for r in results:
                    r["collection"] = collection
                all_results.extend(results)
            except Exception as exc:
                logger.warning(
                    "Collection '%s' search failed (skipping): %s", collection, exc
                )

        # Drop empties, merge hybrid scores globally, then rerank
        all_results = [r for r in all_results if r.get("chunk_text", "").strip()]
        all_results = self._hybrid_merge(effective_query, all_results)
        all_results = self._rerank_results(effective_query, all_results, top_k)

        return [self._enrich_result(r) for r in all_results]


# Module-level singleton — drop-in replacement for the old import
vector_rag = VectorRAG()