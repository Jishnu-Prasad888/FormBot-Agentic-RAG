# Query Expansion Feature for Evaluation

## Overview

Added query expansion capability to the evaluation pipeline. The system generates multiple alternative phrasings of the user's question **before retrieval**, retrieves chunks for each expanded query, then combines all chunks before sending to the LLM for answer generation.

## Changes Made

### 1. **agent_runner.py** - Core Evaluation Logic
- Added `use_query_expansion` parameter (default: False) to `evaluate_question()`
- Added `num_expansions` parameter (default: 2) to control number of expanded queries
- Implemented `_expand_query()` function that uses LLM to generate alternative phrasings
- Retrieves chunks for **all query variants** (original + expanded)
- Deduplicates and combines chunks from all queries
- Returns `expanded_queries` and `num_chunks` in results

**Flow:**
```
1. Original question: "What is CIF?"
2. LLM generates expanded queries:
   - "What does CIF stand for in banking?"
   - "Explain Customer Information File"
3. Retrieve chunks for all 3 queries
4. Deduplicate chunks by chunk_id
5. Combine all unique chunks into single context
6. Send combined context + original question to LLM
7. Generate final answer
8. Score answer against combined context
```

### 2. **prompts.py** - Query Expansion Prompt
- Query expansion prompt embedded directly in `agent_runner.py`
- Instructs LLM to generate alternative phrasings and related queries
- Focuses on rephrasing, synonyms, and related aspects
- Simple numbered list output format

### 3. **rag.py** - API Endpoint Update
- Added `use_query_expansion` field to `EvaluateRequest` schema (default: False)
- Added `num_expansions` field (default: 2) - number of additional queries to generate
- Passes parameters to `evaluate_question()`
- Backward compatible: default behavior unchanged (use_query_expansion=False)

## Usage

### API Request Example

```json
{
  "questions": [
    {
      "question": "What is CIF?",
      "expected_answer": "Customer Information File"
    }
  ],
  "dataset_name": "eval_run",
  "top_k": 5,
  "use_query_expansion": true,
  "num_expansions": 2
}
```

### Response Format

```json
{
  "accuracy": 0.95,
  "faithfulness": 0.92,
  "context_precision": 0.88,
  "context_recall": 0.90,
  "answer_relevancy": 0.94,
  "latency_avg_ms": 2450.5,
  "per_question": [
    {
      "question": "What is CIF?",
      "expected_answer": "Customer Information File",
      "generated_answer": "CIF stands for...",
      "retrieved_context": "...",
      "expanded_queries": [
        "What is CIF?",
        "What does CIF stand for in banking?",
        "Explain Customer Information File"
      ],
      "num_chunks": 12,
      "accuracy": 0.95,
      ...
    }
  ]
}
```

## Benefits

1. **Broader Context Coverage**: Multiple query variants retrieve diverse relevant chunks
2. **Synonym Handling**: Expanded queries use different terminology to catch missed content
3. **Better Recall**: Multiple retrieval passes increase chance of finding relevant information
4. **Chunk Deduplication**: Avoids redundant context across queries
5. **Backward Compatible**: Works with existing code when use_query_expansion=False

## Example Scenario

**Original Question**: "What documents are needed for account opening?"

**LLM Generates Expanded Queries**:
1. "What documents are needed for account opening?" (original)
2. "What are the required documents for opening a bank account?"
3. "List of documentation needed for new account"

**Retrieval**:
- Query 1 retrieves: 5 chunks about account opening
- Query 2 retrieves: 5 chunks (3 overlapping, 2 new about specific documents)
- Query 3 retrieves: 5 chunks (2 overlapping, 3 new about KYC requirements)

**Result**: 13 unique chunks combined → LLM generates comprehensive answer

## Configuration

- **use_query_expansion=False**: Standard evaluation (single query)
- **use_query_expansion=True, num_expansions=1**: Original + 1 expanded query
- **use_query_expansion=True, num_expansions=2**: Original + 2 expanded queries (recommended)
- **use_query_expansion=True, num_expansions=3+**: More expansions (higher latency)

## Files Modified

1. `backend/app/evaluation/agent_runner.py`
2. `backend/app/core/prompts.py`
3. `backend/app/api/rag.py`

## Testing

Run evaluation with query expansion:

```bash
curl -X POST "http://localhost:8000/api/rag/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [{"question": "What is CIF?", "expected_answer": "Customer Information File"}],
    "use_query_expansion": true,
    "num_expansions": 2,
    "top_k": 5
  }'
```
