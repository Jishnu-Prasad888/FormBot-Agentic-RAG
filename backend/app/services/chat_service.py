import uuid
from typing import Any, AsyncGenerator, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.conversation_repository import conversation_repo
from app.services.rag_service import rag_service
from app.embeddings.openai_client import openai_client as ollama_client
from app.core.logging import get_logger
from app.core.exceptions import ConversationNotFoundError

logger = get_logger("chat_service")

SYSTEM_PROMPT = """
You are an intelligent assistant with access to a document knowledge base answer all questions based on the fact that they are all related to SBI bank and bancking sector.

Primary rule:
- Use the provided context as the highest-priority source of information.

Answering guidelines:
1. When the answer is explicitly present in the context, answer using the context.
2. When a question refers to a form field, column value, code, abbreviation, label, or specific term, return its direct definition or expansion exactly as described in the context.
3. If multiple descriptions exist in the context, prefer the shortest definition that directly answers the question.
4. Do not provide additional domain knowledge, examples, background information, assumptions, interpretations, or explanations unless the user explicitly asks for them.

Handling incomplete context:
5. If the context is incomplete or does not directly answer the question:
   - Use your general knowledge only if you are highly confident in the answer.
   - Ensure the answer does not contradict any information present in the context.
   - Clearly prioritize context over prior knowledge whenever both are available.
6. If neither the context nor your knowledge provides a reliable answer, state that you do not have enough information.
7. Never invent field definitions, codes, abbreviations, values, policies, procedures, or document-specific details that are not supported by the context.

Location assumption:
- When the state is not explicitly provided in the question, assume Karnataka, India.

Response style:
- Be concise and answer the user's question directly.
- For definition-style questions, return only the definition unless additional detail is requested.
- Do not mention the source of the information or use phrases such as:
  - "The context provided does not define..."
  - "Based on the context..."
  - "According to the context..."
  - "The document states..."

If there is a conflict between the context and your general knowledge, always follow the context.
"""
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