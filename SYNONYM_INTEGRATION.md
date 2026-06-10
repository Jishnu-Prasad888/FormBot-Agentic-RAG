# Synonym Dictionary Integration

## Overview
The RAG app now uses synonym dictionaries to expand user queries before retrieval, improving search accuracy by matching documents using canonical terms and their aliases.

## Files Used
- `backend/rag_synonym_dictionary.json` - JSON format with canonical terms and their aliases
- `backend/rag_synonym_dictionary_pairs.csv` - CSV format (canonical, alias pairs)

## How It Works

1. **Query Expansion**: When a user searches, the system expands the query using synonyms:
   - "service charges on advances" → ["service charges on advances", "advances related service charges"]
   - "bank code" → ["bank code", "bank coverdrafte"] (example)

2. **Search Integration**: Each search strategy (vector, BM25, hybrid) now:
   - Expands the query with synonyms
   - Searches with all variants
   - Deduplicates results by document ID
   - Returns top-k merged results

3. **Automatic Loading**: The synonym dictionaries are automatically loaded from the backend directory on first use.

## Modified Files

### `backend/app/rag/synonym_expansion.py` (NEW)
- `SynonymExpander` class: Loads and manages synonym mappings
- `get_synonym_expander()`: Lazy-loads the global expander instance

### `backend/app/api/search.py`
Updated search endpoints to use synonym expansion:
- `/api/search/vector` - Vector search with synonyms
- `/api/search/bm25` - BM25 search with synonyms  
- `/api/search/hybrid` - Hybrid search with synonyms

## Usage

The synonym expansion is automatic. Simply make API calls as usual:

```bash
# Vector search (now with synonym expansion)
curl -X POST "http://localhost:8000/api/search/vector" \
  -H "Content-Type: application/json" \
  -d '{"query": "service charges on advances", "top_k": 5}'

# BM25 search
curl -X POST "http://localhost:8000/api/search/bm25" \
  -H "Content-Type: application/json" \
  -d '{"query": "bank code reference", "top_k": 5}'

# Hybrid search
curl -X POST "http://localhost:8000/api/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "advance service charges", "top_k": 5}'
```

## Statistics

- **Canonical Terms**: 114
- **Total Synonym Mappings**: 614
- **Sources**: Both JSON and CSV formats loaded automatically

## Performance
- Query expansion is lightweight (sub-ms)
- Results are deduplicated to prevent duplicates
- No additional embeddings calls needed for expanded queries
