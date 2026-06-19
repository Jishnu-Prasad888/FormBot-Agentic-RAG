from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "MultimodalRAGPlatform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ── Primary DB ────────────────────────────────────────────────────────────
    # Default points to Postgres; set to SQLite URL for local/dev fallback.
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://bank_user:bank_password@localhost:5432/bank_kag"
    )

    # ── Legacy Chroma (kept for fallback) ──────────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # ── Qdrant (primary vector store) ─────────────────────────────────────────
    USE_QDRANT: bool = True
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "bank_documents"
    QDRANT_VECTOR_SIZE: int = 3072  # text-embedding-3-large dimension
    QDRANT_DISTANCE: str = "Cosine"

    # ── Neo4j (knowledge graph) ───────────────────────────────────────────────
    USE_KG_RETRIEVAL: bool = False
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    KG_MAX_DEPTH: int = 2
    KG_MAX_NODES: int = 100
    KG_MAX_DOCS: int = 50
    KG_MAX_CHUNKS: int = 200

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
    OPENAI_EMBED_FALLBACK_MODEL: str = "text-embedding-3-small"
    OPENAI_TIMEOUT: int = 120
    OPENAI_MAX_RETRIES: int = 3

    UPLOAD_DIR: str = "./uploads"
    LOG_FILE: str = "./logs/rag.log"
    LOG_LEVEL: str = "INFO"

    TOP_K: int = 20
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 250
    MAX_CONTEXT_CHUNKS: int = 10
    
    # ── Retrieval Improvements ────────────────────────────────────────────────
    DENSE_TOP_K: int = 50
    BM25_TOP_K: int = 50
    RERANK_TOP_K: int = 20
    BM25_WEIGHT: float = 0.5
    DENSE_WEIGHT: float = 0.5

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
