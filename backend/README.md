# Intelligent Multimodal Agentic RAG Platform
### Complete Technical Reference

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Key Concepts](#4-key-concepts)
5. [Folder Structure](#5-folder-structure)
6. [Database Layer](#6-database-layer)
7. [RAG Pipelines](#7-rag-pipelines)
8. [Agent Framework](#8-agent-framework)
9. [ChromaDB Layer](#9-chromadb-layer)
10. [Ollama Client](#10-ollama-client)
11. [Document Processing](#11-document-processing)
12. [API Reference](#12-api-reference)
13. [Configuration](#13-configuration)
14. [Logging](#14-logging)
15. [Evaluation Framework](#15-evaluation-framework)
16. [Testing](#16-testing)
17. [Setup & Running](#17-setup--running)
18. [Data Flow Walkthroughs](#18-data-flow-walkthroughs)

---

## 1. Overview

This is a production-grade **Retrieval-Augmented Generation (RAG)** backend that uses a multi-agent architecture to intelligently route and answer questions across multiple document types. Rather than a single generic retrieval pipeline, every document type gets its own optimised indexing and retrieval strategy, and a coordinator agent decides which combination of retrieval agents to invoke for each query.

**Core design goals:**

- **Multimodal** — PDFs, Markdown, CSV/tabular data, plain text, JSON, and web pages are all first-class citizens with different processing pipelines
- **Agentic** — six specialised agents, each with a plan/execute/evaluate lifecycle, orchestrated by a coordinator
- **Metadata-first** — every chunk stored in the vector database carries rich metadata so retrieval can be scoped by filename, language, section, document type, date, and more
- **Local-first** — the entire AI stack runs through Ollama with no external API calls required
- **Observable** — every request, retrieval result, agent decision, and evaluation run is logged to `logs/rag.log`

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │Documents │ │  Search  │ │   Chat   │ │  RAG / Agents │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬───────┘  │
└───────┼────────────┼────────────┼────────────────┼──────────┘
        │            │            │                │
        ▼            ▼            ▼                ▼
┌──────────────────────────────────────────────────────────────┐
│                     Services Layer                           │
│   DocumentService  │  ChatService  │  RAGService  │ WebSvc  │
└─────────┬──────────┴───────┬───────┴──────┬───────┴─────────┘
          │                  │              │
    ┌─────▼──────┐    ┌──────▼───────┐  ┌──▼──────────────────┐
    │  Document  │    │ Conversation │  │  Coordinator Agent  │
    │ Repository │    │  Repository  │  │  ┌───────────────┐  │
    └─────┬──────┘    └──────┬───────┘  │  │ Vector Agent  │  │
          │                  │          │  │ SQLite Agent  │  │
    ┌─────▼──────────────────▼────────┐ │  │ Router Agent  │  │
    │          SQLite Database        │ │  │   Web Agent   │  │
    │  documents │ chunks │ messages  │ │  │  Eval Agent   │  │
    │  conversations │ retrieval_logs │ │  └───────────────┘  │
    │  evaluation_runs                │ └──────────┬──────────┘
    └─────────────────────────────────┘            │
                                          ┌────────▼────────────┐
                                          │    RAG Pipelines    │
                                          │  VectorRAG          │
                                          │  HybridRAG          │
                                          │  TableRAG           │
                                          │  PDFHierarchical    │
                                          │  MarkdownRAG        │
                                          └────┬──────────┬─────┘
                                               │          │
                                    ┌──────────▼──┐  ┌────▼──────────┐
                                    │  ChromaDB   │  │ Ollama Client │
                                    │  (Vectors)  │  │  llama3.1:8b  │
                                    │             │  │ nomic-embed   │
                                    └─────────────┘  └───────────────┘
```

### Architectural Layers

The codebase follows strict layered architecture. Data only flows downward — upper layers call lower layers, never the reverse.

| Layer | Responsibility | Location |
|---|---|---|
| **Presentation** | HTTP routing, request parsing, response serialisation | `app/api/` |
| **Application** | Use-case orchestration, no business logic | `app/services/` |
| **Domain** | RAG pipelines, agent logic, retrieval algorithms | `app/rag/`, `app/agents/` |
| **Infrastructure** | Database, vector store, external AI clients | `app/database/`, `app/chromadb/`, `app/embeddings/` |

---

## 3. Tech Stack

| Component | Library | Version | Purpose |
|---|---|---|---|
| API framework | FastAPI | 0.115.5 | Async HTTP, routing, OpenAPI docs |
| Data validation | Pydantic v2 | 2.10.3 | Request/response models, settings |
| ORM | SQLAlchemy | 2.0.36 | Async database access |
| Migrations | Alembic | 1.14.0 | Schema versioning |
| Database | SQLite + aiosqlite | — | Persistent relational storage |
| Vector store | ChromaDB | 0.5.23 | Embedding storage and ANN search |
| LLM | Ollama `llama3.1:8b` | — | Text generation, chat, streaming |
| Embeddings | Ollama `nomic-embed-text-v2-moe` | — | Text → dense vector |
| Keyword search | rank-bm25 | 0.2.2 | Sparse BM25Okapi retrieval |
| PDF parsing | pdfplumber | 0.11.4 | Text and structure extraction |
| HTTP client | httpx | 0.28.1 | Async Ollama + web requests |
| Retry logic | tenacity | 9.0.0 | Exponential backoff on Ollama calls |
| File I/O | aiofiles | 24.1.0 | Async file writes |
| CSV/Excel | pandas + openpyxl | 2.2.3 / 3.1.5 | Tabular data processing |
| HTML parsing | beautifulsoup4 + lxml | 4.12.3 | Web content extraction |
| Markdown | markdown | 3.7 | Markdown utilities |

---

## 4. Key Concepts

### 4.1 Retrieval-Augmented Generation (RAG)

RAG is a technique that improves LLM answers by giving the model relevant document excerpts as context at query time, rather than relying purely on the model's training data. The flow is:

```
Query → Retrieve relevant chunks → Inject as context → Generate answer
```

This platform implements five distinct RAG strategies, each suited to a different document type.

### 4.2 Chunking

Large documents cannot fit in an LLM's context window, so they are split into smaller pieces called **chunks**. Each chunk is embedded independently and stored in the vector database. At query time only the most relevant chunks are retrieved.

**Chunking parameters:**
- `CHUNK_SIZE = 512` words — target chunk size
- `CHUNK_OVERLAP = 50` words — overlap between consecutive chunks to preserve cross-boundary context

Overlap is critical because information near chunk boundaries would otherwise be lost. If a sentence starts at word 510 and ends at word 525, an overlap of 50 ensures it appears in full in at least one chunk.

### 4.3 Embeddings

An **embedding** is a fixed-length vector of floating-point numbers (typically 768–1536 dimensions) that encodes the semantic meaning of a piece of text. Two pieces of text with similar meaning will have vectors that are close together in the embedding space, as measured by cosine similarity.

This platform uses `nomic-embed-text-v2-moe` via Ollama. This is a Mixture-of-Experts model, meaning different "expert" sub-networks activate for different input types, giving high quality embeddings with reasonable compute cost.

### 4.4 Vector Search (Dense Retrieval)

Given a query embedding, **Approximate Nearest Neighbour (ANN)** search finds the k stored chunks whose embeddings are closest to the query vector. ChromaDB uses the HNSW (Hierarchical Navigable Small World) graph algorithm for this, with cosine distance as the metric.

**Distance to score conversion:** ChromaDB returns distances (lower = more similar). The client converts: `score = 1.0 - distance`, so score 1.0 is a perfect match.

### 4.5 BM25 (Sparse / Keyword Retrieval)

BM25 (Best Match 25) is a probabilistic keyword ranking function. It scores documents based on:
- **Term frequency** — how often query terms appear in the chunk
- **Inverse document frequency** — how rare those terms are across all chunks (rare terms are more informative)
- **Document length normalisation** — penalises unusually long chunks that accumulate term matches by sheer size

BM25 excels at exact keyword matching where semantic similarity fails. For example, a query for "PM-KISAN" will score highly chunks containing that exact acronym, even if their semantic meaning is not especially close to the query embedding.

The BM25 index is held **in-memory** (not persisted), and is rebuilt from stored chunk text when the application restarts.

### 4.6 Hybrid Retrieval & Reciprocal Rank Fusion (RRF)

Hybrid retrieval runs both vector search and BM25, then merges the two ranked lists using **Reciprocal Rank Fusion**. The formula for each chunk's fused score is:

```
RRF_score(chunk) = Σ  1 / (k + rank_in_list_i)
                  lists i
```

Where `k = 60` is a smoothing constant. A chunk that ranks 1st in the vector list and 3rd in the BM25 list gets: `1/(60+1) + 1/(60+3) = 0.0164 + 0.0156 = 0.0320`.

RRF is robust because it depends only on rank position, not on raw scores — so it naturally handles the fact that cosine similarity and BM25 scores are on incomparable scales.

### 4.7 Metadata Filtering

Every chunk is stored with structured metadata alongside its embedding. At query time, ChromaDB's `where` parameter can narrow the search to only chunks matching metadata conditions before the ANN search runs. This is called **pre-filtering**.

Example filter: only search chunks from files in Kannada language uploaded after 2025:
```json
{ "$and": [{ "language": { "$eq": "kn" } }, { "document_type": { "$eq": "pdf" } }] }
```

The `metadata_filter.py` module converts user-facing filter dicts into the ChromaDB `$eq/$and` format.

### 4.8 TableRAG

Traditional vector search loses the structure of tabular data — a chunk containing row 5 of a CSV gives no indication that column A is "scheme_name" and column B is "state". TableRAG solves this by creating **two separate indexes**:

1. **Schema index** — one chunk per table containing the column names, data types, and sample rows. This lets queries like "what columns does this dataset have?" retrieve the right context.

2. **Row index** — batches of 5 rows per chunk, stored as CSV text. This lets queries like "show all schemes in Karnataka" retrieve the actual values.

Both indexes live in the `table_documents` ChromaDB collection, with a `chunk_type` metadata field (`"schema"` or `"rows"`) to distinguish them.

### 4.9 Hierarchical PDF RAG

A PDF document has natural structure: sections, subsections, paragraphs, tables. Flat chunking by word count destroys this structure. Hierarchical RAG preserves it by:

1. Detecting headings using regex patterns (all-caps lines, numbered sections like `1.2.3 Title`, title-case lines)
2. Tracking the current section as the parser moves through pages
3. Storing `section` metadata on every chunk

At query time, users can filter by `section` to narrow retrieval to a specific part of a document. The `parent-child` pattern is represented through section metadata rather than explicit parent IDs.

### 4.10 Agents and the Plan/Execute/Evaluate Pattern

Every agent inherits from `BaseAgent` and must implement three async methods:

- **`plan(query, context)`** — analyses the query and constructs an execution plan (a dict describing what to do, with what parameters). No external calls here.
- **`execute(plan)`** — carries out the plan: runs retrieval, calls the LLM, assembles results.
- **`evaluate(result)`** — scores or enriches the result: computes confidence, extracts sources, flags failures.

The public `run(query, context)` method calls all three in sequence. This pattern keeps concerns separated and makes each stage independently testable.

### 4.11 Coordinator Agent and Intent Classification

The `CoordinatorAgent` is the entry point for agentic queries. It classifies query intent using keyword matching:

| Intent | Trigger keywords | Agents invoked |
|---|---|---|
| `table` | table, csv, rows, columns, sum, count, average | SQLiteAgent |
| `web` | website, url, http, online, internet | WebAgent + VectorAgent |
| `structured` | scheme, state, ministry, eligibility | SQLiteAgent + VectorAgent |
| `general` | *(default)* | RouterAgent + VectorAgent |

After all agents run, the coordinator passes all their answers to the LLM for synthesis into a single coherent response.

### 4.12 Conversation Memory

The chat API maintains conversation history in SQLite. Each `POST /api/chat` call:
1. Retrieves the last 8 messages from the conversation
2. Appends retrieved context to the current user message
3. Sends the full history + context to Ollama's chat endpoint
4. Persists both the user message and assistant response

This gives the LLM access to recent conversation turns without the context window growing unboundedly.

---

## 5. Folder Structure

```
backend/
├── app/
│   ├── main.py                        # App factory, middleware, all routers registered
│   │
│   ├── core/
│   │   ├── config.py                  # pydantic-settings: all env vars with defaults
│   │   ├── logging.py                 # File + stream handlers, structured formatter
│   │   ├── exceptions.py              # Custom exception types + FastAPI handlers
│   │   └── dependencies.py            # get_db() async session injection
│   │
│   ├── database/
│   │   ├── base.py                    # DeclarativeBase with naming conventions
│   │   ├── models.py                  # 6 SQLAlchemy models (see §6)
│   │   ├── session.py                 # Async engine + AsyncSessionLocal factory
│   │   └── init_db.py                 # create_all on startup, drop_all utility
│   │
│   ├── schemas/
│   │   ├── document.py                # DocumentResponse, ChunkResponse, ReindexResponse
│   │   ├── search.py                  # SearchRequest, SearchResult, SearchResponse
│   │   ├── chat.py                    # ChatRequest, MessageResponse, ConversationResponse
│   │   ├── rag.py                     # RAGQueryRequest/Response, EvaluationRequest/Response
│   │   ├── agent.py                   # AgentRequest/Response, CoordinatorRequest
│   │   ├── embeddings.py              # EmbeddingRequest/Response, BatchRequest/Response
│   │   └── web.py                     # WebIngestRequest/Response, WebQueryRequest
│   │
│   ├── repositories/
│   │   ├── document_repository.py     # CRUD for Document + Chunk models
│   │   ├── conversation_repository.py # CRUD for Conversation + Message models
│   │   └── log_repository.py          # Insert for RetrievalLog + EvaluationRun
│   │
│   ├── services/
│   │   ├── document_service.py        # Upload, detect type, route to pipeline, persist
│   │   ├── chat_service.py            # Conversation management, context injection
│   │   ├── rag_service.py             # Strategy dispatch, query + evaluate
│   │   └── web_service.py             # Fetch URL, clean HTML, chunk, embed, index
│   │
│   ├── rag/
│   │   ├── vector_rag.py              # Dense retrieval via ChromaDB
│   │   ├── hybrid_rag.py              # Vector + BM25 + RRF fusion
│   │   ├── table_rag.py               # CSV schema + row indexing and query
│   │   ├── pdf_rag.py                 # Heading-aware hierarchical PDF chunking
│   │   ├── markdown_rag.py            # Header-aware Markdown section chunking
│   │   ├── bm25.py                    # In-memory BM25Okapi index per collection
│   │   ├── rrf.py                     # Reciprocal Rank Fusion implementation
│   │   ├── metadata_filter.py         # Filter builders for ChromaDB and in-memory
│   │   └── evaluator.py               # Embedding-based RAG quality metrics
│   │
│   ├── agents/
│   │   ├── base.py                    # BaseAgent ABC: plan / execute / evaluate
│   │   ├── coordinator_agent.py       # Orchestrates all agents, synthesises answer
│   │   ├── vector_agent.py            # Hybrid vector retrieval + answer generation
│   │   ├── sqlite_agent.py            # TableRAG structured data queries
│   │   ├── router_agent.py            # Routes by detected document type
│   │   ├── web_agent.py               # Web ingest on demand + retrieval
│   │   └── evaluator_agent.py         # Faithfulness / precision / recall scoring
│   │
│   ├── chromadb/
│   │   └── client.py                  # Persistent ChromaDB wrapper, 6 collections
│   │
│   ├── embeddings/
│   │   └── ollama_client.py           # Retry-safe async Ollama HTTP client
│   │
│   ├── api/
│   │   ├── health.py                  # /health, /health/db, /health/chroma, /health/ollama
│   │   ├── documents.py               # Upload, list, get, delete, reindex, chunks, metadata
│   │   ├── search.py                  # vector, bm25, hybrid, metadata, table search
│   │   ├── chat.py                    # chat, stream, conversations CRUD
│   │   ├── rag.py                     # query, stream, retrieve, evaluate
│   │   ├── tablerag.py                # index, query, schema endpoints
│   │   ├── pdf.py                     # index, query endpoints
│   │   ├── markdown.py                # index, query endpoints
│   │   ├── agents.py                  # coordinator, vector, sqlite, router, web, evaluator
│   │   ├── chroma.py                  # index, reindex, delete, search, collections
│   │   ├── embeddings.py              # generate, batch, models
│   │   └── web.py                     # ingest, query
│   │
│   └── tests/
│       └── test_core.py               # 30 unit tests, no external services required
│
├── alembic/
│   ├── env.py                         # Async Alembic config
│   ├── script.py.mako                 # Migration template
│   └── versions/
│       └── 0001_initial_schema.py     # Full initial schema migration
│
├── uploads/                           # Uploaded files stored here
├── chroma_db/                         # ChromaDB persistent storage
├── logs/
│   └── rag.log                        # All structured log output
├── alembic.ini
├── requirements.txt
└── .env.example
```

---

## 6. Database Layer

### 6.1 Models

#### Document
Tracks every uploaded file independently of the vector store. The registry enables management, reindexing, versioning, and deletion without touching ChromaDB.

| Field | Type | Notes |
|---|---|---|
| `id` | String(36) | UUID, primary key |
| `filename` | String(512) | Original filename |
| `filepath` | String(1024) | Absolute path on disk |
| `document_type` | String(64) | `pdf`, `markdown`, `text`, `csv`, `json` |
| `retrieval_strategy` | String(64) | `hierarchical_rag`, `table_rag`, `vector_rag`, etc. |
| `language` | String(16) | ISO 639-1 code, default `en` |
| `chunk_count` | Integer | Number of chunks generated |
| `embedding_model` | String(128) | Model used for embedding |
| `collection_name` | String(128) | Target ChromaDB collection |
| `metadata_json` | JSON | Arbitrary user-supplied metadata |
| `created_at` | DateTime | Auto-set on insert |
| `updated_at` | DateTime | Auto-set on update |

**Indexes:** `document_type`, `filename`, `created_at`

#### Chunk
Stores chunk text and metadata in the relational database (mirroring what is stored in ChromaDB). Enables SQL-level chunk queries without touching the vector store.

| Field | Type | Notes |
|---|---|---|
| `id` | String(36) | UUID, primary key |
| `document_id` | String(36) | FK → documents.id (CASCADE DELETE) |
| `chunk_index` | Integer | Sequential position within document |
| `chunk_text` | Text | Raw chunk content |
| `chunk_metadata` | JSON | Section, page number, heading, etc. |
| `created_at` | DateTime | |

**Indexes:** `document_id`, `chunk_index`

#### Conversation
Groups messages into named conversations for the chat API.

| Field | Type | Notes |
|---|---|---|
| `id` | String(36) | UUID, primary key |
| `title` | String(512) | Auto-set to first 60 chars of first message |
| `created_at` | DateTime | |
| `updated_at` | DateTime | |

#### Message
Individual chat turns within a conversation.

| Field | Type | Notes |
|---|---|---|
| `id` | String(36) | UUID, primary key |
| `conversation_id` | String(36) | FK → conversations.id (CASCADE DELETE) |
| `role` | String(32) | `user` or `assistant` |
| `content` | Text | Message text |
| `sources` | JSON | List of `{filename, chunk_id, score}` dicts |
| `created_at` | DateTime | |

**Indexes:** `conversation_id`, `role`

#### RetrievalLog
Audit trail for every retrieval operation. Powers latency analytics and debugging.

| Field | Type | Notes |
|---|---|---|
| `id` | String(36) | UUID, primary key |
| `query` | Text | Original query string |
| `retrieval_strategy` | String(64) | Strategy used |
| `retrieved_chunks` | JSON | List of chunk IDs retrieved |
| `generated_answer` | Text | LLM-generated answer |
| `latency_ms` | Float | End-to-end time in milliseconds |
| `agent_used` | String(64) | Agent name or `rag_service` |
| `created_at` | DateTime | |

#### EvaluationRun
Summary metrics from each `POST /api/rag/evaluate` call.

| Field | Type | Notes |
|---|---|---|
| `id` | String(36) | UUID, primary key |
| `dataset_name` | String(256) | User-supplied label |
| `accuracy` | Float | Average cosine sim: generated vs expected |
| `faithfulness` | Float | Average chunk–answer alignment |
| `context_precision` | Float | Fraction of chunks relevant to query |
| `context_recall` | Float | Best context match for expected answer |
| `created_at` | DateTime | |

### 6.2 Async Setup

```python
engine = create_async_engine(
    "sqlite+aiosqlite:///./rag_platform.db",
    echo=settings.DEBUG,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False},
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
```

All database operations are async — no thread blocking. The session is injected via FastAPI's dependency system (`Depends(get_db)`).

### 6.3 Migrations

Alembic is configured with async support via `async_engine_from_config`. The initial migration `0001_initial_schema.py` creates all tables and indexes. Schema is also auto-created at startup via `init_db()` using `Base.metadata.create_all`, so migrations are optional during development.

```bash
# Apply all migrations
alembic upgrade head

# Create new migration after model changes
alembic revision --autogenerate -m "add field xyz"

# Rollback one migration
alembic downgrade -1
```

---

## 7. RAG Pipelines

### 7.1 Vector RAG (`rag/vector_rag.py`)

Straight dense retrieval against a single ChromaDB collection.

```
Query text
    → ollama_client.embeddings(query)      # nomic-embed-text-v2-moe
    → chroma_client.search(collection,
          query_embedding, top_k, where)   # HNSW ANN search
    → [chunk_id, chunk_text, score, meta]  # sorted by score desc
```

Also supports `retrieve_multi_collection()` which queries multiple collections and merges results by score.

### 7.2 Hybrid RAG (`rag/hybrid_rag.py`)

Runs vector and BM25 in parallel, fuses with RRF:

```
Query
  ├─→ vector_rag.retrieve(top_k * 2)    # dense, over-fetch for fusion
  └─→ bm25_retriever.search(top_k * 2)  # sparse
       ↓
  reciprocal_rank_fusion([dense, sparse], top_k=top_k)
       ↓
  Final ranked list (top_k results)
```

Over-fetching (`top_k * 2`) before fusion gives RRF more candidates to work with, improving final result quality.

### 7.3 TableRAG (`rag/table_rag.py`)

Two-phase indexing for CSV files:

**Indexing:**
```
CSV file
  ├─→ Schema chunk: column names, dtypes, row_count, sample rows
  │       → embed → store in table_documents [chunk_type=schema]
  └─→ Row chunks: every 5 rows as CSV text
          → embed → store in table_documents [chunk_type=rows]
```

**Query:**
```
Query
  → embed query
  → search table_documents (optionally filter by document_id)
  → returns schema + row chunks, ranked by semantic similarity
```

The schema chunk ensures column-level questions ("what are the eligibility criteria columns?") retrieve the right context. Row chunks handle value-level questions ("show schemes where state = Karnataka").

Indexing is capped at 500 rows to prevent extremely large CSV files from overwhelming the embedding pipeline. Large files should be pre-filtered before upload.

### 7.4 Hierarchical PDF RAG (`rag/pdf_rag.py`)

```
PDF bytes
  → pdfplumber.open()
  → per page:
      → split into lines
      → detect heading lines (regex: numbered, ALL-CAPS, Title-Case)
      → when heading found: flush current paragraph as chunk
      → track current_section variable
  → for each chunk:
      → embed
      → store in pdf_documents with metadata:
          { section, page_number, chunk_index, document_id, filename }
```

**Heading detection patterns:**
- Numbered: `1.`, `1.2`, `1.2.3` followed by title text
- All-caps: `INTRODUCTION`, `ELIGIBILITY CRITERIA`
- Title-case short lines: `Overview`, `Application Process`

Minimum chunk length of 50 characters prevents empty or near-empty chunks from polluting the index.

### 7.5 Markdown RAG (`rag/markdown_rag.py`)

```
Markdown text
  → _parse_markdown_sections()
      → regex scan for # ## ### headings
      → split document at heading boundaries
      → each section: { heading, level, content }
  → for each section:
      → combine heading + content as chunk text
      → embed
      → store in markdown_documents with:
          { section=heading, heading_level, chunk_index }
```

Code blocks (` ``` `) and links (`[text](url)`) are preserved in the chunk text rather than stripped. This ensures code examples and URLs remain retrievable.

### 7.6 BM25 (`rag/bm25.py`)

Wraps `rank_bm25.BM25Okapi`. Each collection gets its own in-memory index:

```python
self._index: dict[str, BM25Okapi] = {}    # collection → BM25 model
self._corpus: dict[str, list[dict]] = {}   # collection → chunk records
```

**Tokenisation:** simple `text.lower().split()` — no stopword removal or stemming, keeping it fast and language-agnostic.

**Score filtering:** chunks with score `<= 0` are excluded from results (they have no term overlap with the query at all).

### 7.7 RRF (`rag/rrf.py`)

```python
def reciprocal_rank_fusion(result_lists, k=60, top_k=5):
    scores = {}
    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            chunk_id = item["chunk_id"]
            scores[chunk_id] += 1.0 / (k + rank + 1)
    return sorted by scores, top_k
```

`k=60` is the standard RRF constant from the original paper (Cormack et al., 2009). It prevents very high scores for top-ranked items from completely dominating items that appear in many lists at moderate ranks.

### 7.8 Metadata Filtering (`rag/metadata_filter.py`)

**Supported filter fields:**
`filename`, `document_type`, `section`, `language`, `date`, `document_id`, `retrieval_strategy`, `state`, `ministry`, `department`, `source`

**ChromaDB filter format (pre-filtering):**
```python
# Single field
{"filename": {"$eq": "pm_kisan.pdf"}}

# Multiple fields — ChromaDB $and
{"$and": [
    {"language": {"$eq": "en"}},
    {"document_type": {"$eq": "pdf"}}
]}
```

**In-memory filter (post-filtering for BM25):**
```python
def filter_results(results, filters):
    return [r for r in results if all(
        r["metadata"].get(k) == v for k, v in filters.items()
    )]
```

---

## 8. Agent Framework

### 8.1 BaseAgent (`agents/base.py`)

```python
class BaseAgent(ABC):
    name: str = "base_agent"

    async def plan(self, query, context) -> dict   # abstract
    async def execute(self, plan) -> dict           # abstract
    async def evaluate(self, result) -> dict        # abstract

    async def run(self, query, context=None) -> dict:
        plan   = await self.plan(query, context)
        result = await self.execute(plan)
        return  await self.evaluate(result)
```

All agents return a standardised result dict:
```json
{
  "agent": "agent_name",
  "query": "original query",
  "answer": "generated answer",
  "sources": [{"filename": "...", "chunk_id": "...", "score": 0.9}],
  "confidence": 0.85,
  "latency_ms": 342.1
}
```

### 8.2 CoordinatorAgent (`agents/coordinator_agent.py`)

```
Query
  → _classify_intent(query)       # keyword matching → table/web/structured/general
  → _detect_doc_type(query)       # pdf/markdown/csv/text
  → select agents list
  → run each agent concurrently (sequential implementation, easily parallelised)
  → collect all agent answers + chunks
  → synthesis_prompt = "Multiple agents found: [agent answers]... Final answer for: {query}"
  → ollama_client.generate(synthesis_prompt)
  → return synthesised answer + all sources
```

**Intent → Agent mapping:**
```
table      → [SQLiteAgent]
web        → [WebAgent, VectorAgent]
structured → [SQLiteAgent, VectorAgent]
general    → [RouterAgent, VectorAgent]
```

### 8.3 VectorRetrievalAgent (`agents/vector_agent.py`)

Wraps `HybridRAG` (or pure `VectorRAG` if strategy is set to `"vector"` in context). Adds confidence scoring as the mean retrieval score across all returned chunks.

### 8.4 SQLiteAgent (`agents/sqlite_agent.py`)

Uses `TableRAG.query()` to search the `table_documents` ChromaDB collection. Uses a data-analyst system prompt instructing the LLM to be precise with numbers and values.

### 8.5 DocumentRouterAgent (`agents/router_agent.py`)

Detects document type from the query text using `ROUTING_RULES`:
```python
{
    "pdf":      ["pdf", "document", "page", "report", "form"],
    "markdown": ["markdown", "readme", "guide", "documentation", "wiki"],
    "csv":      ["table", "csv", "data", "rows", "columns", "spreadsheet"],
    "text":     []   # default
}
```
Then routes to the appropriate RAG pipeline: `pdf_rag`, `markdown_rag`, `table_rag`, or `vector_rag`.

### 8.6 WebEnrichmentAgent (`agents/web_agent.py`)

Queries the `web_documents` ChromaDB collection. If a URL is provided and no chunks are found, performs an on-demand ingest of that URL before querying. Uses a research-assistant system prompt that instructs the LLM to note source URLs.

### 8.7 RetrievalEvaluationAgent (`agents/evaluator_agent.py`)

Takes a query, a generated answer, and retrieved chunks (passed via context), and computes:

| Metric | Description |
|---|---|
| `faithfulness` | Mean cosine similarity between answer embedding and each chunk embedding |
| `answer_relevancy` | Cosine similarity between query and answer embeddings |
| `context_precision` | Fraction of chunks with similarity > 0.6 to the query |
| `context_recall` | Max chunk similarity to expected answer (requires expected_answer in context) |
| `overall_score` | Weighted average: 30% faithfulness + 30% relevancy + 20% precision + 20% recall |

---

## 9. ChromaDB Layer

### 9.1 Collections

Six collections are created at startup:

| Collection | Content |
|---|---|
| `pdf_documents` | Hierarchical PDF chunks |
| `markdown_documents` | Markdown section chunks |
| `table_documents` | CSV schema + row chunks |
| `text_documents` | Plain text and JSON chunks |
| `audio_transcripts` | Audio transcription chunks (reserved) |
| `web_documents` | Web-ingested HTML chunks |

### 9.2 ChromaDB Client (`chromadb/client.py`)

```python
# Key methods
create_collection(name, metadata)          # get_or_create with cosine metric
delete_collection(name)
add_documents(collection, ids, embeddings, documents, metadatas)
search(collection, query_embedding, top_k, where)
metadata_filter(collection, query_embedding, filters, top_k)
reindex(collection, ids, embeddings, documents, metadatas)  # delete + re-add
list_collections()
get_collection_count(name)
delete_by_document_id(collection, document_id)  # delete all chunks for a doc
health_check()                                   # calls heartbeat()
```

**Persistence:** `PersistentClient` with path `./chroma_db`. Data survives application restarts automatically — no separate ChromaDB server needed.

**Distance metric:** `{"hnsw:space": "cosine"}` — cosine distance, appropriate for normalised text embeddings.

---

## 10. Ollama Client

### 10.1 Client Features (`embeddings/ollama_client.py`)

- **Connection pooling:** `httpx.AsyncClient` with `max_connections=20`, `max_keepalive_connections=10`
- **Retry logic:** `tenacity` with exponential backoff, 3 attempts, retries on `ConnectError` and `TimeoutException`
- **Timeout:** 120 seconds (configurable) — long timeout needed for first-token LLM responses
- **Streaming:** `client.stream()` + `aiter_lines()` for token-by-token generation
- **Lazy client:** `_get_client()` creates the httpx client on first use and reuses it

### 10.2 Methods

```python
generate(prompt, model, system)             # Non-streaming text generation
chat(messages, model, system)               # Non-streaming chat
generate_stream(prompt, model, system)      # → AsyncGenerator[str]
chat_stream(messages, model, system)        # → AsyncGenerator[str]
embeddings(text, model)                     # Single text → list[float]
batch_embeddings(texts, model)              # asyncio.gather() all embeddings
health_check()                              # GET /api/tags → bool
list_models()                               # GET /api/tags → list[dict]
```

### 10.3 Streaming Implementation

```python
async for token in ollama_client.chat_stream(messages):
    yield token   # SSE / StreamingResponse
```

The `/api/chat/stream` and `/api/rag/query/stream` endpoints use FastAPI's `StreamingResponse` with `media_type="text/plain"` to push tokens to the client as they are generated.

---

## 11. Document Processing

### 11.1 Upload Flow

```
POST /api/documents/upload
  → DocumentService.upload_and_index()
      → _detect_type(filename)           # by extension
      → generate UUID document_id
      → save file to uploads/{uuid}_{filename}
      → branch by document type:
          pdf      → pdf_rag.index()
          markdown → markdown_rag.index()
          csv      → table_rag.index_csv()
          txt/json → _index_text_chunks() → chroma + bm25
      → document_repo.create()           # save to SQLite
      → bulk_create_chunks()             # save chunks to SQLite (text/json only)
      → return { document, chunk_count }
```

### 11.2 Supported File Types

| Extension | Detected Type | Strategy | Collection |
|---|---|---|---|
| `.pdf` | `pdf` | `hierarchical_rag` | `pdf_documents` |
| `.md` | `markdown` | `structure_aware_rag` | `markdown_documents` |
| `.csv` | `csv` | `table_rag` | `table_documents` |
| `.txt` | `text` | `vector_rag` | `text_documents` |
| `.json` | `json` | `vector_rag` | `text_documents` |

### 11.3 Reindexing

`POST /api/documents/{id}/reindex`:
1. Reads the original file from `uploads/`
2. Deletes all existing vectors from ChromaDB (`delete_by_document_id`)
3. Deletes all chunk records from SQLite
4. Runs the full indexing pipeline again
5. Updates `chunk_count` in the Document record

### 11.4 Deletion

`DELETE /api/documents/{id}`:
1. Deletes vectors from ChromaDB
2. Deletes the file from `uploads/`
3. Deletes Document + all Chunks from SQLite (CASCADE)

---

## 12. API Reference

### Health Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service liveness check |
| `GET` | `/health/db` | SQLite `SELECT 1` connectivity check |
| `GET` | `/health/chroma` | ChromaDB heartbeat + collection list |
| `GET` | `/health/ollama` | Ollama `/api/tags` connectivity + loaded models |

### Documents

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload file (multipart/form-data), auto-detect type, index |
| `GET` | `/api/documents` | List all documents. Params: `skip`, `limit` |
| `GET` | `/api/documents/{id}` | Get document by UUID |
| `DELETE` | `/api/documents/{id}` | Delete document, file, and all vectors |
| `POST` | `/api/documents/{id}/reindex` | Re-chunk, re-embed, update vector store |
| `GET` | `/api/documents/{id}/chunks` | List all stored chunks |
| `GET` | `/api/documents/{id}/metadata` | Get full metadata including collection and strategy |

**Upload request:**
```bash
curl -X POST /api/documents/upload \
  -F "file=@document.pdf" \
  -F 'metadata={"language":"en","state":"Karnataka"}'
```

### Search

All search endpoints accept `SearchRequest`:
```json
{
  "query": "PM Kisan eligibility",
  "top_k": 5,
  "filters": { "language": "en", "document_type": "pdf" },
  "collection_name": "pdf_documents"
}
```

All return `SearchResponse`:
```json
{
  "query": "PM Kisan eligibility",
  "results": [
    { "chunk_id": "...", "document_id": "...", "filename": "...",
      "chunk_text": "...", "score": 0.92, "metadata": {} }
  ],
  "confidence": 0.87,
  "sources": ["pm_kisan.pdf"],
  "latency_ms": 124.3,
  "strategy": "hybrid"
}
```

| Method | Path | Strategy |
|---|---|---|
| `POST` | `/api/search/vector` | Dense embeddings only |
| `POST` | `/api/search/bm25` | Keyword BM25 only |
| `POST` | `/api/search/hybrid` | Vector + BM25 + RRF |
| `POST` | `/api/search/metadata` | Embed + ChromaDB metadata filter |
| `POST` | `/api/search/table` | TableRAG collection only |

### Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chat` | Chat message; creates or continues conversation |
| `POST` | `/api/chat/stream` | Streaming chat; returns `text/plain` token stream |
| `GET` | `/api/chat/conversations` | List all conversations |
| `GET` | `/api/chat/conversations/{id}` | Get conversation with full message history |
| `DELETE` | `/api/chat/conversations/{id}` | Delete conversation and all messages |

**Chat request:**
```json
{
  "message": "What are the eligibility criteria for PM Kisan?",
  "conversation_id": "optional-existing-uuid",
  "top_k": 5
}
```

### RAG

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/rag/query` | Retrieve context + generate answer |
| `POST` | `/api/rag/query/stream` | Same but streams tokens |
| `POST` | `/api/rag/retrieve` | Retrieve chunks only, no generation |
| `POST` | `/api/rag/evaluate` | Run evaluation over question set |

**RAG query request:**
```json
{
  "query": "list schemes for small farmers",
  "strategy": "hybrid",
  "top_k": 5,
  "filters": { "state": "Karnataka" }
}
```

**Evaluation request:**
```json
{
  "questions": [
    { "question": "What is PM Kisan?", "expected_answer": "A farmer income support scheme..." }
  ],
  "dataset_name": "farmer_schemes_v1"
}
```

### Specialised RAG

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/tablerag/index` | Index a CSV file |
| `POST` | `/api/tablerag/query` | Query CSV data. Params: `query`, `document_id`, `top_k` |
| `GET` | `/api/tablerag/schema/{id}` | Get schema chunk for a document |
| `POST` | `/api/pdf/index` | Index PDF with hierarchical chunking |
| `POST` | `/api/pdf/query` | Query PDF. Params: `query`, `document_id`, `section`, `top_k` |
| `POST` | `/api/markdown/index` | Index Markdown with header-aware chunking |
| `POST` | `/api/markdown/query` | Query Markdown. Params: `query`, `document_id`, `top_k` |

### Agents

All agent endpoints accept `AgentRequest`:
```json
{
  "query": "show PM Kisan schemes in Karnataka",
  "context": { "document_id": "uuid", "strategy": "hybrid" },
  "top_k": 5,
  "filters": { "state": "Karnataka" }
}
```

| Method | Path | Agent |
|---|---|---|
| `POST` | `/api/agents/coordinator` | Orchestrates all agents, synthesises final answer |
| `POST` | `/api/agents/vector` | Hybrid vector retrieval |
| `POST` | `/api/agents/sqlite` | Structured/tabular queries |
| `POST` | `/api/agents/router` | Auto-routes by document type |
| `POST` | `/api/agents/web` | Web content retrieval |
| `POST` | `/api/agents/evaluator` | Quality scoring (pass chunks in context) |

### ChromaDB

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/chroma/index` | Add raw documents (auto-embeds) |
| `POST` | `/api/chroma/reindex` | Replace collection contents |
| `DELETE` | `/api/chroma/delete` | Delete collection or document by ID |
| `POST` | `/api/chroma/search` | Direct collection search |
| `GET` | `/api/chroma/collections` | List all collections with document counts |

### Embeddings

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/embeddings/generate` | Embed a single text string |
| `POST` | `/api/embeddings/batch` | Embed a list of texts in parallel |
| `GET` | `/api/embeddings/models` | List models available in Ollama |

### Web Ingestion

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/web/ingest` | Fetch URL, extract text, chunk, embed, index |
| `POST` | `/api/web/query` | Query web-ingested content |

**Web ingest request:**
```json
{
  "url": "https://pmkisan.gov.in/Documents/eligibility.pdf",
  "collection_name": "web_documents",
  "metadata": { "source": "official", "state": "national" }
}
```

---

## 13. Configuration

All configuration is through environment variables (`.env` file loaded by pydantic-settings).

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `MultimodalRAGPlatform` | Application name in OpenAPI docs |
| `APP_VERSION` | `1.0.0` | Version string |
| `DEBUG` | `true` | Enables SQLAlchemy echo |
| `DATABASE_URL` | `sqlite+aiosqlite:///./rag_platform.db` | SQLAlchemy async DB URL |
| `CHROMA_HOST` | `localhost` | ChromaDB host (unused in persistent mode) |
| `CHROMA_PORT` | `8001` | ChromaDB port (unused in persistent mode) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local path for ChromaDB storage |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_LLM_MODEL` | `llama3.1:8b` | Model for text generation and chat |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text-v2-moe` | Model for embeddings |
| `OLLAMA_TIMEOUT` | `120` | HTTP timeout in seconds |
| `OLLAMA_MAX_RETRIES` | `3` | Tenacity retry attempts |
| `UPLOAD_DIR` | `./uploads` | Directory for uploaded files |
| `LOG_FILE` | `./logs/rag.log` | Log file path |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `TOP_K` | `5` | Default number of chunks to retrieve |
| `CHUNK_SIZE` | `512` | Target chunk size in words |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks in words |
| `MAX_CONTEXT_CHUNKS` | `10` | Maximum chunks injected into LLM context |

---

## 14. Logging

The logging middleware records every HTTP request and response:

```
2026-05-21 14:32:01 | INFO     | rag_platform.main | → POST /api/rag/query
2026-05-21 14:32:03 | INFO     | rag_platform.main | ← POST /api/rag/query 200 [1847.3ms]
```

Internal logger names follow `rag_platform.<module>`:

```
rag_platform.main
rag_platform.api.documents
rag_platform.api.search
rag_platform.services.document_service
rag_platform.rag.vector_rag
rag_platform.rag.hybrid_rag
rag_platform.agents.coordinator_agent
rag_platform.chromadb_client
rag_platform.ollama_client
...
```

Every retrieval operation logs:
- Query text (first 60 chars)
- Strategy and collection
- Number of results returned

Every agent logs:
- Agent name and query
- Which sub-agents were invoked
- Failures with full exception info

All logs write to both `logs/rag.log` (persistent) and stdout (for container/systemd capture).

---

## 15. Evaluation Framework

`POST /api/rag/evaluate` accepts a list of `{question, expected_answer}` pairs and returns averaged metrics across all questions.

### Metrics

All metrics are computed using **embedding similarity** rather than string matching, making them robust to paraphrase and language variation.

#### Accuracy
```
cosine_similarity(embed(generated_answer), embed(expected_answer))
```
Measures how semantically close the generated answer is to the ground truth.

#### Faithfulness
```
mean( cosine_similarity(embed(answer), embed(chunk_i)) for chunk_i in context )
```
Measures whether the answer is grounded in the retrieved context. A high faithfulness score with a low accuracy score means the model is faithfully summarising the retrieved content, but the retrieved content is not relevant to the expected answer.

#### Answer Relevancy
```
cosine_similarity(embed(question), embed(answer))
```
Measures whether the answer addresses the question directly.

#### Context Precision
```
count(chunks where cosine_sim(query, chunk) > 0.6) / total_chunks
```
Measures what fraction of retrieved chunks were actually relevant to the query. Low precision means the retrieval is noisy.

#### Context Recall
```
max( cosine_similarity(embed(expected_answer), embed(chunk_i)) for chunk_i in context )
```
Measures whether the retrieved context contains the information needed to answer correctly. Low recall means the right chunk was not retrieved.

### Output format

```json
{
  "accuracy": 0.78,
  "faithfulness": 0.84,
  "context_precision": 0.60,
  "context_recall": 0.72,
  "answer_relevancy": 0.81,
  "latency_avg_ms": 1423.5,
  "failed_questions": [
    { "question": "...", "error": "OllamaConnectionError: ..." }
  ]
}
```

Results are persisted to the `evaluation_runs` table for historical comparison.

---

## 16. Testing

```bash
pytest app/tests/test_core.py -v
```

**30 tests, 0 external services required.** All tests run against in-process objects only — no Ollama, no ChromaDB, no SQLite connections.

### Test Coverage

| Area | Tests |
|---|---|
| Reciprocal Rank Fusion | single list, two-list overlap, empty lists, top-k limit |
| Metadata filtering | empty filter, single field, multi-field, unsupported key, in-memory matching |
| BM25 | index + search, missing collection, zero-score exclusion, remove collection |
| Config | model names, default values |
| Pydantic schemas | SearchRequest, RAGQueryRequest, EvaluationRequest, AgentRequest |
| Markdown chunking | multi-level headings, empty content |
| PDF heading detection | numbered headings, ALL-CAPS, long paragraphs not misdetected |
| Text chunking | overlap preserved, short text single chunk |
| File type detection | all 5 supported types, unsupported raises exception |
| Exception hierarchy | status codes, messages for all custom exceptions |
| Intent classification | table/web/structured/general intents |
| Document type routing | pdf/markdown/csv/text routing |

---

## 17. Setup & Running

### Prerequisites

- Python 3.12+
- Ollama installed and running

```bash
# Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text-v2-moe

# Verify
ollama list
```

### Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env if Ollama is not on localhost:11434
```

### Start

```bash
uvicorn app.main:app --reload
```

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

### Production start (no reload)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Note: SQLite + in-memory BM25 indexes are single-process. For multi-worker deployments, migrate to PostgreSQL + a shared Redis BM25 cache, or use a dedicated ChromaDB server instance.

### Database migrations

```bash
alembic upgrade head          # apply all migrations
alembic downgrade base        # roll back everything
alembic history               # view migration history
```

---

## 18. Data Flow Walkthroughs

### 18.1 Upload a PDF

```
User: POST /api/documents/upload  (file=report.pdf, metadata={"state":"Karnataka"})

DocumentService.upload_and_index()
  _detect_type("report.pdf") → "pdf"
  doc_id = uuid4()
  save file → uploads/{doc_id}_report.pdf

  pdf_rag.index(doc_id, "report.pdf", content, {"state":"Karnataka"})
    pdfplumber.open(content)
    for each page:
      split into lines
      detect headings → update current_section
      accumulate paragraph chunks
    for each chunk:
      ollama_client.embeddings(chunk_text)  → [0.12, -0.34, ...]
      chunk_id = f"{doc_id}_p{page}_c{i}"
      chroma_client.add_documents("pdf_documents",
          ids=[chunk_id], embeddings=[emb], documents=[text],
          metadatas=[{document_id, filename, section, page_number, state, ...}])
    return { chunk_count: 47 }

  document_repo.create(db, { id, filename, document_type:"pdf",
      retrieval_strategy:"hierarchical_rag", collection_name:"pdf_documents",
      chunk_count:47, metadata_json:{"state":"Karnataka"} })

Response: { id, filename, document_type, chunk_count:47, message:"Uploaded and indexed" }
```

### 18.2 Hybrid RAG Query

```
User: POST /api/rag/query  { query:"eligibility for farmers Karnataka", strategy:"hybrid" }

RAGService.query()
  start = time.now()

  HybridRAG.retrieve("eligibility for farmers Karnataka", "text_documents", top_k=5)
    
    VectorRAG.retrieve()
      ollama_client.embeddings("eligibility for farmers Karnataka") → query_emb
      chroma_client.search("text_documents", query_emb, 10, where=None)
      → [chunk_a(0.91), chunk_b(0.88), chunk_c(0.82), ...]    # top-10 dense

    bm25_retriever.search("text_documents", "eligibility for farmers Karnataka", 10)
      tokenise → ["eligibility", "for", "farmers", "karnataka"]
      BM25Okapi.get_scores(tokens)
      → [chunk_b(4.2), chunk_d(3.8), chunk_a(3.1), ...]       # top-10 sparse

    reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_k=5)
      chunk_a: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0156 = 0.0320
      chunk_b: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  ← wins
      → [chunk_b, chunk_a, chunk_d, chunk_c, chunk_e]

  context = "[report.pdf] Eligibility: Small and marginal farmers...\n\n[scheme.txt] ..."
  prompt = f"Context:\n{context}\n\nQuestion: eligibility for farmers Karnataka"
  answer = ollama_client.generate(prompt, system=RAG_SYSTEM)

  latency = (time.now() - start) * 1000   # e.g. 2341ms

  log_repo.create_retrieval_log(db, { query, strategy:"hybrid",
      retrieved_chunks:[chunk_b, chunk_a, ...], generated_answer:answer, latency_ms:2341 })

Response: { query, answer, sources, strategy:"hybrid", latency_ms:2341, confidence:0.87 }
```

### 18.3 Coordinator Agent Query

```
User: POST /api/agents/coordinator  { query:"how many CSV schemes exist for Karnataka?" }

CoordinatorAgent.run()
  plan()
    _classify_intent("how many CSV schemes exist for Karnataka?") → "table"
    _detect_doc_type() → "text"   (no doc-type keywords)
    agents = ["sqlite"]

  execute(plan)
    SQLiteAgent.run("how many CSV schemes exist for Karnataka?")
      plan() → { query, top_k:5, document_id:None }
      execute()
        table_rag.query("how many CSV schemes exist for Karnataka?", top_k=5)
          embed query → chroma search "table_documents" → schema + row chunks
        context = "scheme_name,state,...\nPM Kisan,Karnataka,..."
        ollama_client.generate(f"Table data:\n{context}\n\nQuestion: {query}")
        → "Based on the table, there are 12 schemes for Karnataka..."
      evaluate() → confidence=0.78, sources=[...]
    
    all_chunks = sqlite_result["chunks"]
    combined_context = "[SQLITE_AGENT]:\nBased on the table, there are 12 schemes..."
    
    synthesis_prompt = "Multiple agents retrieved: [SQLITE_AGENT]: ... Final answer for: ..."
    final_answer = ollama_client.generate(synthesis_prompt)

  evaluate()
    confidence = mean(chunk scores)
    sources = unique filenames from all chunks

Response: { agent:"coordinator_agent", answer:"There are 12 government schemes...",
            sources:[...], confidence:0.78, latency_ms:3102 }
```

### 18.4 Streaming Chat

```
User: POST /api/chat/stream  { message:"explain PM Kisan benefits" }

ChatService.chat_stream()
  create Conversation(id=new_uuid, title="explain PM Kisan benefits")

  HybridRAG.retrieve("explain PM Kisan benefits", top_k=5)
  → 5 chunks from relevant documents

  history = [] (new conversation)
  messages = [{ role:"user", content:"Context:\n[doc.pdf] Benefits include...\n\nQuestion: explain PM Kisan benefits" }]

  add_message(db, { role:"user", content:"explain PM Kisan benefits" })

  async for token in ollama_client.chat_stream(messages, system=SYSTEM_PROMPT):
      yield token      → StreamingResponse sends each token to client immediately

  # After stream completes:
  add_message(db, { role:"assistant", content:"PM Kisan provides ₹6000 per year...", sources:[...] })

Client receives:  "PM " "Kisan " "provides " "₹6000 " "per " "year " ...
```

