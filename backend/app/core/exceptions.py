from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.logging import get_logger

logger = get_logger("exceptions")


class RAGPlatformException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class DocumentNotFoundError(RAGPlatformException):
    def __init__(self, doc_id: str):
        super().__init__(f"Document {doc_id} not found", 404)


class ChunkNotFoundError(RAGPlatformException):
    def __init__(self, chunk_id: str):
        super().__init__(f"Chunk {chunk_id} not found", 404)


class ConversationNotFoundError(RAGPlatformException):
    def __init__(self, conv_id: str):
        super().__init__(f"Conversation {conv_id} not found", 404)


class OpenAIConnectionError(RAGPlatformException):
    def __init__(self, detail: str = ""):
        super().__init__(f"OpenAI connection failed: {detail}", 503)


# Backward-compat alias so any existing catch clauses still work
OllamaConnectionError = OpenAIConnectionError


class ChromaDBError(RAGPlatformException):
    def __init__(self, detail: str = ""):
        super().__init__(f"ChromaDB error: {detail}", 503)


class UnsupportedFileTypeError(RAGPlatformException):
    def __init__(self, file_type: str):
        super().__init__(f"Unsupported file type: {file_type}", 422)


async def rag_platform_exception_handler(request: Request, exc: RAGPlatformException):
    logger.error(f"RAGPlatformException: {exc.message} | path={request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "status_code": exc.status_code},
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} | path={request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()} | path={request.url.path}")
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "details": exc.errors()},
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc} | path={request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )
