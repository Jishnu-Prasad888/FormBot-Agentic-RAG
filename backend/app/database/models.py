from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Text, Integer, Float, ForeignKey, DateTime, Index, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    document_type: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    retrieval_strategy: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    language: Mapped[str] = mapped_column(String(16), nullable=True, default="en")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    collection_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    form_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_document_type", "document_type"),
        Index("ix_documents_filename", "filename"),
        Index("ix_documents_created_at", "created_at"),
        Index("ix_documents_category", "category"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True, default=dict)
    qdrant_point_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_chunk_index", "chunk_index"),
        Index("ix_chunks_qdrant_point_id", "qdrant_point_id"),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=True, default="New Conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages: Mapped[list["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_conversations_created_at", "created_at"),)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[dict] = mapped_column(JSON, nullable=True, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_role", "role"),
    )


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_strategy: Mapped[str] = mapped_column(String(64), nullable=True)
    retrieved_chunks: Mapped[dict] = mapped_column(JSON, nullable=True, default=list)
    generated_answer: Mapped[str] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)
    agent_used: Mapped[str] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_retrieval_logs_created_at", "created_at"),
        Index("ix_retrieval_logs_retrieval_strategy", "retrieval_strategy"),
    )


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_query_logs_created_at", "created_at"),)


class Form(Base):
    __tablename__ = "forms"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_forms_name", "name"),
        Index("ix_forms_category", "category"),
    )


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_name: Mapped[str] = mapped_column(String(256), nullable=True)
    accuracy: Mapped[float] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_evaluation_runs_created_at", "created_at"),)
