import uuid
from typing import Any, AsyncGenerator, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.conversation_repository import conversation_repo
from app.services.rag_service import rag_service
from app.embeddings.openai_client import openai_client as ollama_client
from app.core.logging import get_logger
from app.core.exceptions import ConversationNotFoundError
from app.core.prompts import SYSTEM_PROMPT

logger = get_logger("chat_service")


class ChatService:
    async def chat(
        self,
        db: AsyncSession,
        message: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        # Get or create conversation
        if conversation_id:
            conv = await conversation_repo.get_by_id(db, conversation_id)
            if not conv:
                raise ConversationNotFoundError(conversation_id)
        else:
            conv = await conversation_repo.create(db, {
                "id": str(uuid.uuid4()),
                "title": message[:60],
            })

        # Retrieve context
        retrieval_result = await rag_service.retrieve(message, strategy="hybrid", top_k=top_k)
        context_chunks = [r["chunk_text"] for r in retrieval_result]
        sources = [
            {"filename": r.get("filename", ""), "chunk_id": r.get("chunk_id", ""), "score": r.get("score", 0)}
            for r in retrieval_result
        ]

        # Build messages with history
        history = await conversation_repo.get_messages(db, conv.id, limit=10)
        messages = []
        for m in history[-8:]:
            messages.append({"role": m.role, "content": m.content})

        # Add context to current message
        context_str = "\n\n".join(f"[Source: {r.get('filename','unknown')}]\n{r['chunk_text']}" for r in retrieval_result)
        user_content = f"Context:\n{context_str}\n\nQuestion: {message}" if context_chunks else message
        messages.append({"role": "user", "content": user_content})

        # Generate answer
        answer = await ollama_client.chat(messages, system=SYSTEM_PROMPT)

        # Save user and assistant messages
        await conversation_repo.add_message(db, {
            "id": str(uuid.uuid4()),
            "conversation_id": conv.id,
            "role": "user",
            "content": message,
            "sources": [],
        })
        assistant_msg = await conversation_repo.add_message(db, {
            "id": str(uuid.uuid4()),
            "conversation_id": conv.id,
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        return {
            "conversation_id": conv.id,
            "message": assistant_msg,
            "sources": sources,
        }

    async def chat_stream(
        self,
        db: AsyncSession,
        message: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        if conversation_id:
            conv = await conversation_repo.get_by_id(db, conversation_id)
            if not conv:
                raise ConversationNotFoundError(conversation_id)
        else:
            conv = await conversation_repo.create(db, {
                "id": str(uuid.uuid4()),
                "title": message[:60],
            })

        retrieval_result = await rag_service.retrieve(message, strategy="hybrid", top_k=top_k)
        context_str = "\n\n".join(
            f"[Source: {r.get('filename','unknown')}]\n{r['chunk_text']}"
            for r in retrieval_result
        )

        history = await conversation_repo.get_messages(db, conv.id, limit=10)
        messages = [{"role": m.role, "content": m.content} for m in history[-8:]]
        user_content = f"Context:\n{context_str}\n\nQuestion: {message}" if retrieval_result else message
        messages.append({"role": "user", "content": user_content})

        await conversation_repo.add_message(db, {
            "id": str(uuid.uuid4()),
            "conversation_id": conv.id,
            "role": "user",
            "content": message,
            "sources": [],
        })

        full_answer = []
        async for token in ollama_client.chat_stream(messages, system=SYSTEM_PROMPT):
            full_answer.append(token)
            yield token

        answer = "".join(full_answer)
        sources = [
            {"filename": r.get("filename", ""), "chunk_id": r.get("chunk_id", ""), "score": r.get("score", 0)}
            for r in retrieval_result
        ]
        await conversation_repo.add_message(db, {
            "id": str(uuid.uuid4()),
            "conversation_id": conv.id,
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


chat_service = ChatService()