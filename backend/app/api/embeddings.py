from fastapi import APIRouter
from app.embeddings.openai_client import openai_client as ollama_client
from app.schemas.embeddings import (
    EmbeddingRequest, EmbeddingBatchRequest,
    EmbeddingResponse, EmbeddingBatchResponse, EmbeddingModelInfo,
)
from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])
logger = get_logger("api.embeddings")


@router.post("/generate", response_model=EmbeddingResponse)
async def generate_embedding(req: EmbeddingRequest):
    model = req.model or settings.OPENAI_EMBED_MODEL
    embedding = await ollama_client.embeddings(req.text, model)
    return EmbeddingResponse(
        text=req.text,
        embedding=embedding,
        model=model,
        dimensions=len(embedding),
    )


@router.post("/batch", response_model=EmbeddingBatchResponse)
async def batch_embeddings(req: EmbeddingBatchRequest):
    model = req.model or settings.OPENAI_EMBED_MODEL
    embeddings = await ollama_client.batch_embeddings(req.texts, model)
    responses = [
        EmbeddingResponse(text=t, embedding=e, model=model, dimensions=len(e))
        for t, e in zip(req.texts, embeddings)
    ]
    return EmbeddingBatchResponse(embeddings=responses, model=model)


@router.get("/models", response_model=list[EmbeddingModelInfo])
async def list_embedding_models():
    models = await ollama_client.list_models()
    return [EmbeddingModelInfo(name=m.get("id", m.get("name", "")), dimensions=None) for m in models]