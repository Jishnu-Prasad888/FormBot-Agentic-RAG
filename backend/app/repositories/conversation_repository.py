from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from app.database.models import Conversation, Message



class ConversationRepository:
    async def create(self, db: AsyncSession, data: dict) -> Conversation:
        conv = Conversation(**data)
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        return conv

    async def get_by_id(self, db: AsyncSession, conv_id: str, with_messages: bool = False) -> Optional[Conversation]:
        query = select(Conversation).where(Conversation.id == conv_id)
        if with_messages:
            query = query.options(selectinload(Conversation.messages))
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list_all(self, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[Conversation]:
        result = await db.execute(
            select(Conversation).offset(skip).limit(limit).order_by(Conversation.updated_at.desc())
        )
        return list(result.scalars().all())

    async def delete(self, db: AsyncSession, conv_id: str) -> bool:
        await db.execute(delete(Conversation).where(Conversation.id == conv_id))
        await db.commit()
        return True

    async def add_message(self, db: AsyncSession, data: dict) -> Message:
        msg = Message(**data)
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        return msg

    async def get_messages(self, db: AsyncSession, conv_id: str, limit: int = 20) -> list[Message]:
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count(self, db: AsyncSession) -> int:
        from sqlalchemy import func
        result = await db.execute(select(func.count()).select_from(Conversation))
        return result.scalar() or 0


conversation_repo = ConversationRepository()
