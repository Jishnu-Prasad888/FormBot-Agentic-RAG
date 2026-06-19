from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.dependencies import get_db
from app.chromadb.client import chroma_client
from app.embeddings.openai_client import openai_client as ollama_client
from app.graph.neo4j_client import neo4j_client
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "MultimodalRAGPlatform"}


@router.get("/health/db")
async def health_db(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "database": settings.DATABASE_URL}
    except Exception as e:
        return {"status": "error", "database": settings.DATABASE_URL, "detail": str(e)}


@router.get("/health/chroma")
async def health_chroma():
    ok = chroma_client.health_check()
    status = "ok" if ok else "error"
    collections = []
    if ok:
        try:
            collections = chroma_client.list_collections()
        except Exception:
            pass
    return {
        "status": status,
        "vector_store": getattr(chroma_client, "backend_name", "chroma"),
        "collections": collections,
    }


@router.get("/health/openai")
async def health_openai():
    ok = await ollama_client.health_check()
    models = []
    if ok:
        try:
            raw = await ollama_client.list_models()
            models = [m.get("name") for m in raw]
        except Exception:
            pass
    return {
        "status": "ok" if ok else "error",
        "openai": "connected" if ok else "unreachable",
        "models": models,
    }


@router.get("/health/neo4j")
async def health_neo4j():
    if not neo4j_client.enabled:
        return {"status": "disabled", "neo4j": "disabled"}
    ok = await neo4j_client.health_check()
    return {
        "status": "ok" if ok else "error",
        "neo4j": "reachable" if ok else "unreachable",
    }
