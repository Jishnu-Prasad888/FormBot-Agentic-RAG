from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.services.chat_service import chat_service
from app.repositories.conversation_repository import conversation_repo
from app.schemas.chat import (
    ChatRequest, ChatResponse, ConversationListResponse, ConversationResponse
)
from app.core.exceptions import ConversationNotFoundError

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    result = await chat_service.chat(db, req.message, req.conversation_id, req.top_k)
    return result


@router.post("/stream")
async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    async def token_generator():
        async for token in chat_service.chat_stream(db, req.message, req.conversation_id, req.top_k):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_db)):
    convs = await conversation_repo.list_all(db, skip, limit)
    total = await conversation_repo.count(db)
    return {"conversations": convs, "total": total}


@router.get("/conversations/{conv_id}", response_model=ConversationResponse)
async def get_conversation(conv_id: str, db: AsyncSession = Depends(get_db)):
    conv = await conversation_repo.get_by_id(db, conv_id, with_messages=True)
    if not conv:
        raise ConversationNotFoundError(conv_id)
    return conv


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, db: AsyncSession = Depends(get_db)):
    conv = await conversation_repo.get_by_id(db, conv_id)
    if not conv:
        raise ConversationNotFoundError(conv_id)
    await conversation_repo.delete(db, conv_id)
    return {"message": f"Conversation {conv_id} deleted"}
