# Banking KAG Migration Plan

This plan adapts the provided KAG specification to the current codebase while **keeping the frontend unchanged and preserving existing backend endpoints/response shapes**. The goal is to add graph-assisted retrieval (PostgreSQL + Neo4j + Qdrant) without breaking the current FastAPI routes under `app/api`.

## 0) Snapshot of Current State
- API surface to keep stable: `POST /api/rag/query`, `POST /api/rag/query/stream`, `POST /api/rag/retrieve`, `POST /api/rag/evaluate`, document upload/reindex/delete routes, chat routes, health routes.
- Retrieval stack: Chroma (vector) + BM25 + metadata filters + RRF + cross-encoder rerank + neighbor expansion, orchestrated in `HybridRAG`/`rag_service`.
- Storage: SQLite via SQLAlchemy models in `app/database/models.py` (documents, chunks, conversations, messages, retrieval_logs, evaluation_runs). Chroma stores vectors; BM25 index is in-memory; files land in `settings.UPLOAD_DIR`.
- Ingestion: `document_service` chunks text (512/50), embeds via OpenAI client, stores vectors in Chroma, metadata + chunks in SQLite.
- Config: `app/core/config.py` supplies host/ports for Chroma and OpenAI; embeddings use `text-embedding-3-large` by default.

## 1) Target Architecture (per spec, mapped to code)
```
Question → Intent Detection (reuse) → Entity Extraction (new) → KG Retrieval (Neo4j, depth 2)
→ Candidate Document Selection → Qdrant Semantic Search + BM25 (restricted set)
→ RRF → Cross Encoder → Neighbor Expansion → Cross Encoder → LLM
```
- **Vector store**: Qdrant collection `bank_documents` (payload contains chunk_id, document_id, filename, section, category, form_name).
- **Metadata store**: PostgreSQL (documents, chunks, forms, query_logs, evaluation data). No embeddings stored in Postgres.
- **Graph store**: Neo4j with node types Form, Document, Concept, Field, Requirement, Eligibility, Procedure and relationships from the spec.

## 2) Infrastructure (keep app container/ports unchanged)
- Add a separate compose file (e.g., `infra/docker-compose.kag.yml`) running the three services:
  - Postgres 17 (`bank_kag`, `bank_user/bank_password`, port 5432)
  - Neo4j 5 with APOC (ports 7474/7687)
  - Qdrant latest (ports 6333/6334)
- App changes stay in existing FastAPI container/process so the frontend keeps working against the same base URL.

## 3) Data Model Updates (SQLAlchemy / Alembic)
- Switch `DATABASE_URL` to PostgreSQL; keep SQLite support as a fallback during migration behind an env flag.
- Extend `Document` with `title`, `document_type`, `category`, `source`, `created_at` (already), `qdrant_collection` (if multiple), `form_name` (optional).
- Extend `Chunk` with `qdrant_point_id`, `metadata_json` (alias of current `chunk_metadata`), and indexes on `document_id`, `chunk_index`.
- Add new tables matching the spec:
  - `forms(id, name, category, description)`
  - `query_logs(id, query, response, latency, created_at)`
- Keep `retrieval_logs`/`evaluation_runs` for backward-compatible metrics; add migration scripts via Alembic.

## 4) Qdrant Integration (Phase 1)
- Create `app/qdrant/client.py` mirroring `chromadb/client.py` API to avoid touching call sites. Adapter methods: `init_collections()`, `add_documents()`, `query()`, `delete_by_document_id()`. Under the hood, use Qdrant HTTP/GRPC client; store payload with the required fields; persist returned `point_id` to Postgres `chunks.qdrant_point_id`.
- Replace Chroma usage in services with the adapter but keep function names the same for endpoint stability. Keep Chroma as a temporary fallback behind a feature flag (`USE_QDRANT=true`).

## 5) Ingestion Pipeline (Phase 2–5)
- **Step 1: Load** existing PDF/MD/TXT/CSV handlers; no frontend change.
- **Step 2: Chunking** update to 1500 chars with 250 overlap (config-driven). Store chunks in Postgres only (not embeddings).
- **Step 3: Embedding Generation** use OpenAI `text-embedding-3-large` with fallback to `text-embedding-3-small` on error. Store vectors in Qdrant; persist `point_id` to `chunks`.
- **Step 4: Entity Extraction** add service using GPT with schema `{entities: [], relationships: []}`. Save raw extraction per document for audit/debug.
- **Step 5: Graph Construction** add `graph_ingestor` that:
  - Seeds graph from filenames, categories, forms, and metadata (cheap, deterministic pass).
  - Adds GPT-derived entities/relationships incrementally.
  - Connects documents/forms to Concept/Requirement/Eligibility/Procedure nodes per spec.
  - Limits fan-out per document to keep graph depth 2 traversable.

## 6) Query Pipeline Changes (Phase 6)
- Insert entity extraction step before retrieval. Reuse existing intent detection output to scope graph traversal.
- Add KG retrieval service (Neo4j) that returns candidate document IDs and form nodes. Rules: depth ≤2, max nodes 100, max docs 50, max chunks 200.
- Constrain vector (Qdrant) and BM25 searches to the candidate document ID set. Apply existing metadata filters first.
- Run existing RRF, cross-encoder rerank, neighbor expansion, and final rerank unchanged. Keep response models the same (`RAGQueryResponse`, `SearchResult`).
- When no graph candidates are found, fall back to current hybrid flow to avoid empty answers.

## 7) Form Recommendation Module (Phase 7)
- Add a new service (can be exposed via a new endpoint without changing existing ones) that:
  - Extracts entities/intents from the natural-language request.
  - Traverses the graph (Forms → Requirements/Eligibility/Procedure → Documents) to produce: recommended form, required documents, eligibility, procedure steps, supporting document names.
- Response shape can mirror existing `SearchResult` style to keep frontend reuse optional; but existing endpoints stay untouched.

## 8) Evaluation & Success Criteria
- Keep current evaluation endpoints; extend metrics logging to Postgres `query_logs` and `evaluation_runs` so dashboards stay valid.
- Target deltas: +20% context precision, +15% context recall, -30% hallucination rate, form recommendation accuracy >95%, latency overhead <500ms.
- Add shadow-mode flag to run graph-guided retrieval alongside the legacy path and log differences before making it default.

## 9) Rollout Order (matches spec phases)
1. Qdrant adapter live behind flag; dual-write Chroma→Qdrant until stable.
2. Move metadata/chunks to Postgres; migrate SQLite data via Alembic script + backfill job.
3. Introduce Neo4j service + health check.
4. Seed graph from filenames/categories/forms/sections/metadata (no GPT yet).
5. Add GPT-based entity extraction + gradual enrichment jobs.
6. Enable graph-guided constrained retrieval; keep fallback to legacy hybrid.
7. Add form recommendation endpoint/service.
8. Optionally retire Elasticsearch enhancement once graph recall meets/exceeds baseline.

## 10) Compatibility Notes
- Keep request/response schemas unchanged for existing routes; add feature flags to toggle KAG pieces without frontend impact.
- Preserve current chunk metadata keys so the UI can still render sources. When introducing new fields (e.g., `form_name`, `category`), append them to payloads rather than renaming existing ones.
- Maintain logging shape used by `EvaluationLogger` to avoid breaking `POST /api/rag/evaluate` consumers.

## 11) Immediate Next Steps
- Add infra compose file for Postgres/Neo4j/Qdrant and env vars to `.env.example`.
- Implement Qdrant adapter mirroring `chroma_client` API and gate via `USE_QDRANT`.
- Update chunking config to 1500/250 and persist chunk `qdrant_point_id` in Postgres.
- Sketch Neo4j schema file (Cypher) for node/edge types from the spec and add a seeding script that runs without GPT.
- Add entity extraction service + unit tests with canned prompts to validate schema output.
