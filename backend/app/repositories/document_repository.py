from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from app.database.models import Document, Chunk
from app.core.logging import get_logger

logger = get_logger("document_repo")


class DocumentRepository:
    async def create(self, db: AsyncSession, data: dict) -> Document:
        doc = Document(**data)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        return doc

    async def get_by_id(self, db: AsyncSession, doc_id: str) -> Optional[Document]:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        return result.scalar_one_or_none()

    async def list_all(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> list[Document]:
        result = await db.execute(select(Document).offset(skip).limit(limit).order_by(Document.created_at.desc()))
        return list(result.scalars().all())

    async def delete(self, db: AsyncSession, doc_id: str) -> bool:
        await db.execute(delete(Document).where(Document.id == doc_id))
        await db.commit()
        return True

    async def update(self, db: AsyncSession, doc_id: str, data: dict) -> Optional[Document]:
        doc = await self.get_by_id(db, doc_id)
        if not doc:
            return None
        for k, v in data.items():
            setattr(doc, k, v)
        await db.commit()
        await db.refresh(doc)
        return doc

    async def get_chunks(self, db: AsyncSession, doc_id: str) -> list[Chunk]:
        result = await db.execute(
            select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    async def create_chunk(self, db: AsyncSession, data: dict) -> Chunk:
        chunk = Chunk(**data)
        db.add(chunk)
        await db.commit()
        await db.refresh(chunk)
        return chunk

    async def delete_chunks(self, db: AsyncSession, doc_id: str) -> int:
        result = await db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
        await db.commit()
        return result.rowcount

    async def bulk_create_chunks(self, db: AsyncSession, chunks: list[dict]) -> int:
        db.add_all([Chunk(**c) for c in chunks])
        await db.commit()
        return len(chunks)

    async def count(self, db: AsyncSession) -> int:
        from sqlalchemy import func
        result = await db.execute(select(func.count()).select_from(Document))
        return result.scalar() or 0


document_repo = DocumentRepository()
