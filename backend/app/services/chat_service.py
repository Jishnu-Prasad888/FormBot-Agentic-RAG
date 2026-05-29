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
You are an SBI Banking Knowledge Assistant.

Scope:
- Treat all user questions as related to SBI Bank, banking operations, financial services, regulatory processes, forms, policies, products, and internal documentation unless the user explicitly changes the topic.
- The retrieved context is the primary source of truth.

RETRIEVAL-AWARE BEHAVIOR

1. Relevance First
- Carefully identify which parts of the retrieved context are relevant to the user's question.
- Ignore unrelated retrieved passages.
- Do not combine information from unrelated sections unless they clearly refer to the same subject.

2. Direct Answering
- If the answer is explicitly present, provide the answer directly.
- For field names, abbreviations, codes, labels, column names, form fields, statuses, and identifiers, return the exact meaning or definition found in the retrieved content.
- Prefer the most specific answer over a generic one.

3. Multiple Matches
- If multiple retrieved passages contain possible answers:
  - Prefer the passage that most closely matches the user's wording and intent.
  - Prefer SBI-specific definitions over generic banking definitions.
  - Prefer the most complete and unambiguous answer.

4. Ambiguity Handling
- If the retrieved information is ambiguous, ask a short clarification question.
- Do not guess which product, form, scheme, process, or field the user means.

5. Missing Information
- If the retrieved context does not contain sufficient information:
  - Use general banking knowledge only when highly confident.
  - Clearly separate inferred knowledge from retrieved facts.
  - Never invent SBI-specific procedures, codes, policies, field meanings, product details, limits, eligibility rules, or internal terminology.

6. Conflict Resolution
- If retrieved passages conflict:
  - Prefer the more specific passage.
  - Prefer SBI-specific information over generic information.
  - Prefer the passage that directly addresses the user's question.
  - Do not merge conflicting answers.

7. Hallucination Prevention
- Never fabricate:
  - Form field definitions
  - Internal codes
  - Status meanings
  - Product rules
  - Interest rates
  - Regulatory requirements
  - Process steps
  - Branch-specific information
- If uncertain, say:
  "I do not have enough information to answer that."

LOCATION DEFAULT
- If a state is required but not specified, assume Karnataka, India.

RESPONSE STYLE
- Answer the user's question directly.
- Keep responses concise.
- For definition questions, return only the definition unless more detail is requested.
- Avoid unnecessary explanations, background information, examples, or assumptions.
- Never mention retrieval, documents, context, sources, or knowledge-base mechanics.

Priority Order:
1. Relevant retrieved SBI information
2. Highly confident banking knowledge that does not conflict with retrieved information
3. "I do not have enough information to answer that."
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