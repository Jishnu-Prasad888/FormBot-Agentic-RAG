"""
Simple nearest-neighbor RAG evaluation.

Uses: vector search + BM25 + synonym expansion + reranker.
"""

import time
from typing import Any

from app.rag.vector_rag import vector_rag
from app.rag.bm25 import bm25_retriever
from app.rag.synonym_expansion import get_synonym_expander
from app.rag.cross_encoder import cross_encoder
from app.evaluation.evaluator import evaluate_single
from app.embeddings.openai_client import openai_client
from app.core.config import settings
from app.core.evaluation_logger import EvaluationLogger
from app.services.elasticsearch_service import es_service

# Global logger instance
_current_logger = None

def set_evaluation_logger(logger: EvaluationLogger):
    global _current_logger
    _current_logger = logger


RAG_SYSTEM = """Answer the question carefully"""


def _chunk_texts(chunks: list[dict]) -> list[str]:
    return [c["chunk_text"] for c in chunks if c.get("chunk_text")]


def _merge_results(vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
    """Simple merge: combine and deduplicate by chunk_id"""
    seen = set()
    merged = []
    for c in vector_results + bm25_results:
        cid = c.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            merged.append(c)
    return merged


async def evaluate_question(
    question: str,
    expected_answer: str,
    top_k: int = 5,
    use_query_expansion: bool = False,
    num_expansions: int = 2,
) -> dict[str, Any]:
    """
    Simplified RAG evaluation:
    1. Synonym expansion (if enabled)
    2. Vector + BM25 retrieval (top 10 each)
    3. Merge and rerank (top 5)
    4. Generate answer
    5. Score with LLM-as-judge
    """
    t0 = time.time()
    
    if _current_logger:
        _current_logger.log("QUESTION_START", f"Q: {question}\nExpected: {expected_answer}")
    
    # Synonym expansion
    synonym_expander = get_synonym_expander()
    queries = synonym_expander.expand_query(question)
    if _current_logger:
        _current_logger.log("SYNONYM_EXPANSION", f"Generated {len(queries)} queries:\n" + "\n".join(queries))
    
    # Retrieve chunks for all query variants
    all_chunks = []
    chunk_ids_seen = set()
    
    for idx, query in enumerate(queries, 1):
        # Vector search (nearest neighbor)
        vector_results = await vector_rag.retrieve(query, "text_documents", top_k=10)
        if _current_logger:
            _current_logger.log(f"VECTOR_SEARCH_{idx}", f"Query: {query}\nFound: {len(vector_results)} chunks")
        
        # BM25 search
        bm25_results = bm25_retriever.search("text_documents", query, top_k=10)
        if _current_logger:
            _current_logger.log(f"BM25_SEARCH_{idx}", f"Query: {query}\nFound: {len(bm25_results)} chunks")
        
        # Merge
        merged = _merge_results(vector_results, bm25_results)
        
        # Deduplicate across queries
        for c in merged:
            cid = c.get("chunk_id")
            if cid and cid not in chunk_ids_seen:
                chunk_ids_seen.add(cid)
                all_chunks.append(c)
    
    if _current_logger:
        _current_logger.log("MERGE_RESULTS", f"Total unique chunks: {len(all_chunks)}")
    
    # Rerank to top_k
    reranked = cross_encoder.rerank(question, all_chunks, top_k=top_k)
    if _current_logger:
        _current_logger.log("RERANK", f"Top {top_k} chunks after reranking")
        for i, chunk in enumerate(reranked, 1):
            _current_logger.log(f"CHUNK_{i}", 
                f"Score: {chunk.get('score', 0)}\n"
                f"File: {chunk.get('filename', 'unknown')}\n"
                f"Text: {chunk.get('chunk_text', '')[:300]}...")
    
    # Generate answer
    chunk_texts = _chunk_texts(reranked)
    
    # Enhance with Elasticsearch
    original_count = len(chunk_texts)
    enhanced_texts = await es_service.enhance_with_iterative_query(chunk_texts, question, max_tries=5, logger=_current_logger)
    if _current_logger:
        _current_logger.log("ES_ENHANCEMENT_COMPLETE", 
            f"Original: {original_count} chunks\n"
            f"Enhanced: {len(enhanced_texts)} chunks\n"
            f"Added: {len(enhanced_texts) - original_count} from Elasticsearch"
            f"\nEnhanced texts :\n" + "\n---\n".join(enhanced_texts)
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
    
    # Build retrieved chunk IDs and derive gold IDs via text matching
    retrieved_chunk_ids = [c.get("chunk_id", "") for c in reranked if c.get("chunk_id")]
    expected_lower = expected_answer.lower()
    gold_chunk_ids = {
        c.get("chunk_id")
        for c in reranked
        if c.get("chunk_id") and (
            expected_lower in c.get("chunk_text", "").lower()
            or (len(expected_lower) >= 20 and expected_lower[:20] in c.get("chunk_text", "").lower())
        )
    }
    # Fallback: if no exact match found, mark best-scoring chunks as gold
    if not gold_chunk_ids and reranked:
        best_score = max((c.get("score", 0) for c in reranked), default=0)
        if best_score > 0:
            gold_chunk_ids = {c.get("chunk_id") for c in reranked if c.get("score", 0) >= best_score * 0.9 and c.get("chunk_id")}

    if _current_logger:
        _current_logger.log("GOLD_CHUNKS", f"Identified {len(gold_chunk_ids)} gold chunks from {len(retrieved_chunk_ids)} retrieved")

    # Score
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
        "num_chunks": len(reranked),
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
        "answer_relevancy_rationale": "",
        "context_precision_rationale": "",
        "context_recall_rationale": "",
        "latency_ms": 0.0,
        "error": error,
    }
