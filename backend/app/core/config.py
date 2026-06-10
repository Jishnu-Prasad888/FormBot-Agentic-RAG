from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "MultimodalRAGPlatform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    DATABASE_URL: str = "sqlite+aiosqlite:///./rag_platform.db"

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
    OPENAI_TIMEOUT: int = 120
    OPENAI_MAX_RETRIES: int = 3

    UPLOAD_DIR: str = "./uploads"
    LOG_FILE: str = "./logs/rag.log"
    LOG_LEVEL: str = "INFO"

    TOP_K: int = 5
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 150
    MAX_CONTEXT_CHUNKS: int = 10
    
    # ── Retrieval Improvements ────────────────────────────────────────────────
    DENSE_TOP_K: int = 50
    BM25_TOP_K: int = 50
    RERANK_TOP_K: int = 10
    BM25_WEIGHT: float = 0.5
    DENSE_WEIGHT: float = 0.5

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
