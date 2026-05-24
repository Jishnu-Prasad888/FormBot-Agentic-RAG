# Intelligent Multimodal Agentic RAG Platform

Production-grade FastAPI backend for multimodal retrieval-augmented generation with a full agentic orchestration layer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Database | SQLite + SQLAlchemy 2.x + Alembic |
| Vector Store | ChromaDB (persistent) |
| LLM | Ollama — `llama3.1:8b` |
| Embeddings | Ollama — `nomic-embed-text-v2-moe` |
| Keyword Search | BM25 (rank-bm25) |
| PDF Parsing | pdfplumber |

---

## Prerequisites

1. **Python 3.12+**
2. **Ollama** running at `http://localhost:11434` with models pulled:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text-v2-moe
```

---

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Run migrations (optional — auto-created on startup)

```bash
alembic upgrade head
```

### Start the server

```bash
uvicorn app.main:app --reload
```

Interactive docs → http://localhost:8000/docs

---

## Folder Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app, all routers registered
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── logging.py           # Structured file + stream logging
│   │   ├── exceptions.py        # Global exception handlers
│   │   └── dependencies.py      # DB session injection
│   ├── database/
│   │   ├── base.py              # DeclarativeBase
│   │   ├── models.py            # All SQLAlchemy models
│   │   ├── session.py           # Async engine + session factory
│   │   └── init_db.py           # create_all on startup
│   ├── schemas/                 # Pydantic v2 request/response models
│   ├── repositories/            # Async DB access layer
│   ├── services/                # Business logic
│   ├── rag/
│   │   ├── vector_rag.py        # Dense retrieval via ChromaDB
│   │   ├── hybrid_rag.py        # Vector + BM25 + RRF
│   │   ├── table_rag.py         # CSV schema + row indexing
│   │   ├── pdf_rag.py           # Heading-aware hierarchical chunking
│   │   ├── markdown_rag.py      # Header-aware section chunking
│   │   ├── bm25.py              # In-memory BM25 index
│   │   ├── rrf.py               # Reciprocal Rank Fusion
│   │   ├── metadata_filter.py   # ChromaDB + in-memory filtering
│   │   └── evaluator.py         # Embedding-based RAG metrics
│   ├── agents/
│   │   ├── base.py              # BaseAgent: plan / execute / evaluate
│   │   ├── coordinator_agent.py # Orchestrates all other agents
│   │   ├── vector_agent.py      # Hybrid vector retrieval
│   │   ├── sqlite_agent.py      # TableRAG structured queries
│   │   ├── router_agent.py      # Routes by document type
│   │   ├── web_agent.py         # Web ingestion + retrieval
│   │   └── evaluator_agent.py   # Retrieval quality scoring
│   ├── chromadb/
│   │   └── client.py            # Persistent ChromaDB wrapper
│   ├── embeddings/
│   │   └── ollama_client.py     # Retry-safe Ollama HTTP client
│   ├── api/                     # All FastAPI routers
│   └── tests/
│       └── test_core.py         # 30 unit tests (no external deps)
├── alembic/                     # Migrations
├── uploads/                     # Uploaded files
├── chroma_db/                   # Persisted vector store
├── logs/rag.log                 # Structured log output
├── requirements.txt
├── alembic.ini
└── .env.example
```

---

## API Reference

### Documents

| Method | Path | Description |
|---|---|---|
| POST | `/api/documents/upload` | Upload + auto-index (PDF/MD/TXT/CSV/JSON) |
| GET | `/api/documents` | List all documents |
| GET | `/api/documents/{id}` | Get document by ID |
| DELETE | `/api/documents/{id}` | Delete document + vectors |
| POST | `/api/documents/{id}/reindex` | Re-chunk and re-embed |
| GET | `/api/documents/{id}/chunks` | List stored chunks |
| GET | `/api/documents/{id}/metadata` | Get full metadata |

### Search

| Method | Path | Description |
|---|---|---|
| POST | `/api/search/vector` | Dense embedding search |
| POST | `/api/search/bm25` | Keyword (BM25) search |
| POST | `/api/search/hybrid` | Vector + BM25 + RRF |
| POST | `/api/search/metadata` | Filter by metadata fields |
| POST | `/api/search/table` | Search table (CSV) data |

### Chat

| Method | Path | Description |
|---|---|---|
| POST | `/api/chat` | Chat with conversation memory |
| POST | `/api/chat/stream` | Streaming chat response |
| GET | `/api/chat/conversations` | List conversations |
| GET | `/api/chat/conversations/{id}` | Get conversation + messages |
| DELETE | `/api/chat/conversations/{id}` | Delete conversation |

### RAG

| Method | Path | Description |
|---|---|---|
| POST | `/api/rag/query` | RAG query (retrieve + generate) |
| POST | `/api/rag/query/stream` | Streaming RAG query |
| POST | `/api/rag/retrieve` | Retrieve chunks only |
| POST | `/api/rag/evaluate` | Evaluate RAG quality metrics |

### Specialized RAG

| Method | Path | Description |
|---|---|---|
| POST | `/api/tablerag/index` | Index CSV file |
| POST | `/api/tablerag/query` | Query table data |
| GET | `/api/tablerag/schema/{id}` | Get table schema |
| POST | `/api/pdf/index` | Index PDF (hierarchical) |
| POST | `/api/pdf/query` | Query PDF sections |
| POST | `/api/markdown/index` | Index Markdown |
| POST | `/api/markdown/query` | Query Markdown sections |

### Agents

| Method | Path | Description |
|---|---|---|
| POST | `/api/agents/coordinator` | Multi-agent orchestration |
| POST | `/api/agents/vector` | Vector retrieval agent |
| POST | `/api/agents/sqlite` | Structured/table agent |
| POST | `/api/agents/router` | Document type routing agent |
| POST | `/api/agents/web` | Web enrichment agent |
| POST | `/api/agents/evaluator` | Retrieval evaluation agent |

### ChromaDB

| Method | Path | Description |
|---|---|---|
| POST | `/api/chroma/index` | Add documents to collection |
| POST | `/api/chroma/reindex` | Replace collection contents |
| DELETE | `/api/chroma/delete` | Delete collection or document |
| POST | `/api/chroma/search` | Query a collection directly |
| GET | `/api/chroma/collections` | List all collections + counts |

### Embeddings

| Method | Path | Description |
|---|---|---|
| POST | `/api/embeddings/generate` | Embed a single text |
| POST | `/api/embeddings/batch` | Batch embed texts |
| GET | `/api/embeddings/models` | List available Ollama models |

### Web Ingestion

| Method | Path | Description |
|---|---|---|
| POST | `/api/web/ingest` | Fetch URL, chunk, embed, index |
| POST | `/api/web/query` | Query web-ingested content |

### Health

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status |
| GET | `/health/db` | SQLite connectivity |
| GET | `/health/chroma` | ChromaDB + collections |
| GET | `/health/ollama` | Ollama + loaded models |

---

## Retrieval Strategies

| Document Type | Strategy | Collection |
|---|---|---|
| PDF | Hierarchical heading-aware RAG | `pdf_documents` |
| Markdown | Header-aware section chunking | `markdown_documents` |
| CSV | TableRAG (schema + row index) | `table_documents` |
| TXT / JSON | Vector RAG | `text_documents` |
| Web (HTML) | Vector RAG | `web_documents` |

---

## Evaluation Metrics

POST `/api/rag/evaluate` computes per-question and averaged:

- **Accuracy** — cosine similarity between generated and expected answer
- **Faithfulness** — answer grounded in retrieved context
- **Answer Relevancy** — semantic alignment of answer to question
- **Context Precision** — fraction of retrieved chunks relevant to query
- **Context Recall** — best context alignment with expected answer
- **Latency** — average end-to-end ms per question

---

## Running Tests

```bash
pytest app/tests/test_core.py -v
# 30 tests, no Ollama or ChromaDB required
```

---

## Environment Variables

See `.env.example` for all available settings. Key variables:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text-v2-moe
DATABASE_URL=sqlite+aiosqlite:///./rag_platform.db
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
```
