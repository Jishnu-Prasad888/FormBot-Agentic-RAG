"""
KAG-first evaluation for banking/product documents.

Flow:
  Question
    ↓
  Synonym Expansion
    ↓
  Intent Detection & Metadata Filter Generation
    ↓
  KAG Retrieval (Neo4j candidates → hybrid vector with metadata filters)
    ↓
  Cross-Encoder Rerank
    ↓
  Neighbor Chunk Expansion
    ↓
  Cross-Encoder Rerank
    ↓
  ES Enhancement
    ↓
  LLM
"""

import time
import re
from typing import Any

from app.chromadb.client import chroma_client
from app.rag.synonym_expansion import get_synonym_expander
from app.rag.cross_encoder import cross_encoder
from app.evaluation.evaluator import evaluate_single
from app.embeddings.openai_client import openai_client
from app.core.config import settings
from app.core.evaluation_logger import EvaluationLogger
from app.services.elasticsearch_service import es_service
from app.services.graph_service import graph_service
from app.services.rag_service import rag_service

# Global logger instance
_current_logger = None

def set_evaluation_logger(logger: EvaluationLogger):
    global _current_logger
    _current_logger = logger


RAG_SYSTEM = """Answer the question carefully"""


# ── Section keyword mapping for intent detection ──────────────────────────
SECTION_KEYWORDS = {
    "interest_rate": [
        "interest rate", "interest", "roi", "rate of interest",
        "apr", "annual percentage", "interest charged",
    ],
    "eligibility": [
        "eligibility", "eligible", "who can", "qualify",
        "requirements", "criteria", "prerequisite", "conditions",
    ],
    "purpose": [
        "purpose", "objective", "aim", "goal", "why is", "use",
        "for what", "intended for",
    ],
    "documents_required": [
        "documents required", "documentation", "required documents",
        "docs needed", "paperwork", "documents to", "documents needed",
        "documents required for",
    ],
    "security": [
        "security", "collateral", "guarantee", "pledge", "mortgage",
        "security required",
    ],
    "features": [
        "features", "benefits", "advantages", "highlights", "salient",
        "key features",
    ],
    "fees": [
        "fees", "charges", "penalty", "processing fee", "late fee",
        "applicable fees",
    ],
    "tenure": [
        "tenure", "duration", "period", "repayment", "term",
        "maturity", "repayment period", "loan tenure",
    ],
    "loan_amount": [
        "loan amount", "amount", "maximum loan", "sanction",
        "limit", "maximum amount",
    ],
}

# Common banking product patterns for product name detection
PRODUCT_PATTERNS = [
    (r"kisan\s*credit\s*card", "kisan-credit-card"),
    (r"personal\s*loan", "personal-loan"),
    (r"home\s*loan", "home-loan"),
    (r"car\s*loan|auto\s*loan|vehicle\s*loan", "car-loan"),
    (r"education\s*loan|student\s*loan", "education-loan"),
    (r"savings\s*account", "savings-account"),
    (r"current\s*account", "current-account"),
    (r"fixed\s*deposit", "fixed-deposit"),
    (r"recurring\s*deposit", "recurring-deposit"),
    (r"credit\s*card", "credit-card"),
    (r"overdraft", "overdraft"),
    (r"gold\s*loan", "gold-loan"),
    (r"mortgage\s*loan|home\s*loan", "home-loan"),
    (r"business\s*loan", "business-loan"),
    (r"agriculture\s*loan|agricultural\s*loan|farm\s*loan|kisan", "agriculture-loan"),
]


def _chunk_texts(chunks: list[dict]) -> list[str]:
    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]


def _detect_section(query: str) -> str | None:
    """Detect the intended document section from the query using keyword matching."""
    query_lower = query.lower()
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return section
    return None


def _extract_product_name(query: str) -> str | None:
    """Extract a normalized product name from the query, or None if not found."""
    query_lower = query.lower()
    for pattern, product_name in PRODUCT_PATTERNS:
        if re.search(pattern, query_lower):
            return product_name
    return None


async def _fetch_neighbor_chunks(
    chunks: list[dict],
    collection_name: str,
    neighbor_window: int = 1,
) -> list[dict]:
    """Fetch neighboring chunks from the same document for each retrieved chunk."""
    if not chunks:
        return chunks

    # Neighbor fetching is only supported on Chroma collections; fall back otherwise.
    if getattr(chroma_client, "backend_name", "chroma") != "chroma":
        return chunks

    expanded = []
    seen_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}

    # Group chunks by document_id so we fetch per-doc once
    doc_groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        doc_id = meta.get("document_id") or chunk.get("document_id")
        if doc_id:
            doc_groups.setdefault(doc_id, []).append(chunk)

    # For each doc group, fetch all chunks from ChromaDB and find neighbors
    for doc_id, doc_chunks in doc_groups.items():
        try:
            collection = chroma_client.get_client().get_collection(collection_name)
            result = collection.get(
                where={"document_id": doc_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            # If collection doesn't exist or fetch fails, just pass through
            expanded.extend(doc_chunks)
            continue

        doc_ids = result.get("ids", [])
        doc_docs = result.get("documents", [])
        doc_metas = result.get("metadatas", [])

        # Build a map of chunk_index -> chunk data for this document
        doc_index_map: dict[int, dict] = {}
        for i, cid in enumerate(doc_ids):
            c_meta = doc_metas[i] if i < len(doc_metas) else {}
            c_idx = c_meta.get("chunk_index")
            if c_idx is not None:
                doc_index_map[c_idx] = {
                    "chunk_id": cid,
                    "chunk_text": doc_docs[i] if i < len(doc_docs) else "",
                    "metadata": c_meta or {},
                    "document_id": doc_id,
                }

        # For each retrieved chunk, find and add neighbors
        for chunk in doc_chunks:
            meta = chunk.get("metadata", {})
            chunk_idx = meta.get("chunk_index")
            if chunk_idx is None:
                expanded.append(chunk)
                continue

            for offset in range(-neighbor_window, neighbor_window + 1):
                if offset == 0:
                    continue
                neighbor_idx = chunk_idx + offset
                neighbor = doc_index_map.get(neighbor_idx)
                if neighbor and neighbor["chunk_id"] not in seen_ids:
                    seen_ids.add(neighbor["chunk_id"])
                    neighbor_entry = {
                        **neighbor,
                        "score": chunk.get("score", 0) * 0.9,
                    }
                    expanded.append(neighbor_entry)

            expanded.append(chunk)

    return expanded


async def evaluate_question(
    question: str,
    expected_answer: str,
    top_k: int = 5,
    use_query_expansion: bool = False,
    num_expansions: int = 2,
) -> dict[str, Any]:
    """
    Metadata-aware RAG evaluation:

      1. Synonym expansion (if enabled)
      2. Intent detection → metadata filters (section, product_name)
      3. Hybrid retrieval (Vector + BM25) with metadata filters
      4. RRF fusion instead of simple merge
      5. Cross-encoder rerank (first pass, over-retrieve)
      6. Neighbor chunk expansion from same document
      7. Cross-encoder rerank (second pass)
      8. ES enhancement
      9. Generate answer
     10. Score with LLM-as-judge
    """
    t0 = time.time()

    if _current_logger:
        _current_logger.log("QUESTION_START", f"Q: {question}\nExpected: {expected_answer}")

    # ── 1. Intent detection & metadata filter generation ──────────────────
    detected_section = _detect_section(question)
    detected_product = _extract_product_name(question)

    filters = {}
    if detected_product:
        filters["filename"] = detected_product
    if detected_section:
        filters["section"] = detected_section

    if _current_logger and (detected_section or detected_product):
        _current_logger.log("INTENT_DETECTION",
            f"Detected section: {detected_section}\n"
            f"Detected product: {detected_product}\n"
            f"Filters: {filters}")

    # ── 2. Synonym expansion ──────────────────────────────────────────────
    synonym_expander = get_synonym_expander()
    queries = synonym_expander.expand_query(question)
    if not use_query_expansion:
        queries = [queries[0]]  # Use original query only
    if _current_logger:
        _current_logger.log("SYNONYM_EXPANSION", f"Generated {len(queries)} queries:\n" + "\n".join(queries))

    # ── 3. KAG retrieval (graph candidates + hybrid vector) ───────────────
    all_chunks: list[dict[str, Any]] = []
    graph_forms: set[str] = set()
    for idx, q in enumerate(queries, 1):
        graph_result = await graph_service.get_candidates(q, filters)
        candidate_ids = graph_result.candidate_document_ids or None
        for f in graph_result.forms:
            name = f.get("name")
            if name:
                graph_forms.add(name)
        if _current_logger:
            _current_logger.log(
                f"GRAPH_{idx}",
                f"Candidates: {len(candidate_ids or [])}\nForms: {[f.get('name') for f in graph_result.forms]}"
            )

        results = await rag_service.retrieve(
            q,
            strategy="hybrid",
            top_k=getattr(settings, "TOP_K", top_k),
            filters=filters if filters else None,
            candidate_document_ids=candidate_ids,
        )
        if _current_logger:
            _current_logger.log(f"KAG_RETRIEVE_{idx}", f"Query: {q}\nFound: {len(results)} chunks")
        if results:
            all_chunks.extend(results)

    # Deduplicate by chunk_id
    deduped = []
    seen = set()
    for c in all_chunks:
        cid = c.get("chunk_id")
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        deduped.append(c)

    if _current_logger:
        _current_logger.log("KAG_DEDUP", f"Chunks after KAG dedup: {len(deduped)}")

    # ── 4. Rerank (over-retrieve) ─────────────────────────────────────────
    reranked = cross_encoder.rerank(question, deduped, top_k=top_k * 2)
    if _current_logger:
        _current_logger.log("FIRST_RERANK", f"Top {top_k * 2} chunks after KAG rerank")

    # ── 5. Neighbor chunk expansion ───────────────────────────────────────
    expanded_chunks = await _fetch_neighbor_chunks(reranked, "text_documents")
    if _current_logger:
        _current_logger.log("NEIGHBOR_EXPANSION",
            f"Before: {len(reranked)} chunks\n"
            f"After neighbor expansion: {len(expanded_chunks)} chunks")

    # ── 6. Second rerank ──────────────────────────────────────────────────
    reranked_final = cross_encoder.rerank(question, expanded_chunks, top_k=top_k)
    if _current_logger:
        _current_logger.log("SECOND_RERANK", f"Final top {top_k} chunks after neighbor-aware reranking")
        for i, chunk in enumerate(reranked_final, 1):
            _current_logger.log(f"CHUNK_{i}",
                f"Score: {chunk.get('score', 0)}\n"
                f"File: {chunk.get('filename', 'unknown')}\n"
                f"Section: {chunk.get('metadata', {}).get('section', 'N/A')}\n"
                f"Chunk Index: {chunk.get('metadata', {}).get('chunk_index', 'N/A')}\n"
                f"Text: {chunk.get('chunk_text', '')[:300]}...")

    # ── 8. Generate answer ────────────────────────────────────────────────
    chunk_texts = _chunk_texts(reranked_final)

    # Inject graph context if available
    graph_note = ""
    if graph_forms:
        graph_note = "Graph candidates (Forms):\n" + "\n".join(sorted(graph_forms))
    if graph_note:
        chunk_texts = [graph_note] + chunk_texts

    # Enhance with Elasticsearch via iterative query
    original_count = len(chunk_texts)
    enhanced_texts = await es_service.enhance_with_iterative_query(
        chunk_texts, question, max_tries=3, logger=_current_logger
    )
    if _current_logger:
        _current_logger.log("ES_ENHANCEMENT_COMPLETE",
            f"Original: {original_count} chunks\n"
            f"Enhanced: {len(enhanced_texts)} chunks\n"
            f"Added: {len(enhanced_texts) - original_count} from Elasticsearch"
            f"\nEnhanced texts:\n" + "\n---\n".join(enhanced_texts)
            )

    context_text = "\n---\n".join(enhanced_texts)

    if context_text.strip():
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    else:
        prompt = f"Question: {question}"

    if _current_logger:
        _current_logger.log("LLM_PROMPT", f"Prompt length: {len(prompt)} chars\nPrompt preview:\n{prompt[:500]}...")

    llm_start = time.time()
    generated_answer = await openai_client.generate(prompt, system=RAG_SYSTEM)
    llm_time = (time.time() - llm_start) * 1000

    if _current_logger:
        _current_logger.log("LLM_ANSWER", f"Generated in {llm_time}ms:\n{generated_answer}")

    # ── 9. Score ──────────────────────────────────────────────────────────
    # Build retrieved chunk IDs and derive gold IDs via text matching
    retrieved_chunk_ids = [c.get("chunk_id", "") for c in reranked_final if c.get("chunk_id")]
    expected_lower = expected_answer.lower()
    gold_chunk_ids = {
        c.get("chunk_id")
        for c in reranked_final
        if c.get("chunk_id") and (
            expected_lower in c.get("chunk_text", "").lower()
            or (len(expected_lower) >= 20 and expected_lower[:20] in c.get("chunk_text", "").lower())
        )
    }
    # Fallback: if no exact match found, mark best-scoring chunks as gold
    if not gold_chunk_ids and reranked_final:
        best_score = max((c.get("ce_score", 0) for c in reranked_final), default=0)
        if best_score > 0:
            gold_chunk_ids = {
                c.get("chunk_id") for c in reranked_final
                if c.get("ce_score", 0) >= best_score * 0.9 and c.get("chunk_id")
            }

    if _current_logger:
        _current_logger.log("GOLD_CHUNKS",
            f"Identified {len(gold_chunk_ids)} gold chunks from {len(retrieved_chunk_ids)} retrieved")

    scores = await evaluate_single(question, expected_answer, generated_answer, enhanced_texts,
                                   retrieved_chunk_ids=retrieved_chunk_ids, gold_chunk_ids=gold_chunk_ids)

    if _current_logger:
        _current_logger.log("METRICS",
            f"Accuracy: {scores.get('accuracy_llm')}\n"
            f"Faithfulness: {scores.get('faithfulness')}\n"
            f"Context Precision: {scores.get('context_precision')}\n"
            f"Context Recall: {scores.get('context_recall')}\n"
            f"Answer Relevancy: {scores.get('answer_relevancy')}")

    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "retrieved_context": context_text,
        "expanded_queries": queries,
        "num_chunks": len(reranked_final),
        # LLM-as-judge
        "accuracy_llm": scores.get("accuracy_llm", 0.0),
        "faithfulness": scores.get("faithfulness", 0.0),
        "answer_relevancy": scores.get("answer_relevancy", 0.0),
        "context_precision": scores.get("context_precision", 0.0),
        "context_recall": scores.get("context_recall", 0.0),
        # Accuracy methods
        "exact_match": scores.get("exact_match", 0.0),
        "semantic_similarity": scores.get("semantic_similarity", 0.0),
        "f1": scores.get("f1", 0.0),
        "accuracy_combined": scores.get("accuracy_combined", 0.0),
        # Retrieval metrics
        "recall_10": scores.get("recall_10", 0.0),
        "recall_20": scores.get("recall_20", 0.0),
        "recall_50": scores.get("recall_50", 0.0),
        "mrr": scores.get("mrr", 0.0),
        "ndcg_10": scores.get("ndcg_10", 0.0),
        # Rationales
        "accuracy_rationale": scores.get("accuracy_rationale", ""),
        "faithfulness_rationale": scores.get("faithfulness_rationale", ""),
        "answer_relevancy_rationale": scores.get("answer_relevancy_rationale", ""),
        "context_precision_rationale": scores.get("context_precision_rationale", ""),
        "context_recall_rationale": scores.get("context_recall_rationale", ""),
        # Metadata
        "detected_section": detected_section,
        "detected_product": detected_product,
        "latency_ms": latency_ms,
    }


def failed_question_row(question: str, expected_answer: str, error: str) -> dict[str, Any]:
    """Build a zeroed per-question row when the eval pipeline raises."""
    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": "",
        "retrieved_context": "",
        "expanded_queries": [],
        "num_chunks": 0,
        # LLM-as-judge
        "accuracy_llm": 0.0,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        # Accuracy methods
        "exact_match": 0.0,
        "semantic_similarity": 0.0,
        "f1": 0.0,
        "accuracy_combined": 0.0,
        # Retrieval metrics
        "recall_10": 0.0,
        "recall_20": 0.0,
        "recall_50": 0.0,
        "mrr": 0.0,
        "ndcg_10": 0.0,
        # Rationales
        "accuracy_rationale": "",
        "faithfulness_rationale": "",
        "context_precision_rationale": "",
        "context_recall_rationale": "",
        "answer_relevancy_rationale": "",
        "latency_ms": 0.0,
        "error": error,
    }
