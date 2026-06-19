import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.exceptions import (
    RAGPlatformException,
    rag_platform_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from app.database.init_db import init_db
from app.chromadb.client import chroma_client

# API routers
from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.api.rag import router as rag_router
from app.api.chroma import router as chroma_router
from app.api.forms import router as forms_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        chroma_client.init_collections()
    except Exception:
        pass
    yield
    from app.embeddings.openai_client import openai_client
    await openai_client.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent Multimodal Agentic RAG Platform",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000
    return response


# Exception handlers
app.add_exception_handler(RAGPlatformException, rag_platform_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Register all routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(chroma_router)
app.include_router(forms_router)
