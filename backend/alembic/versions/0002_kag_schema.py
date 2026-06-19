"""kag schema

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-18 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Documents
    op.add_column("documents", sa.Column("title", sa.String(length=512), nullable=True))
    op.add_column("documents", sa.Column("category", sa.String(length=128), nullable=True))
    op.add_column("documents", sa.Column("source", sa.String(length=128), nullable=True))
    op.add_column("documents", sa.Column("form_name", sa.String(length=256), nullable=True))
    op.create_index("ix_documents_category", "documents", ["category"], unique=False)

    # Chunks
    op.add_column("chunks", sa.Column("metadata_json", sa.JSON(), nullable=True))
    op.add_column("chunks", sa.Column("qdrant_point_id", sa.String(length=64), nullable=True))
    op.create_index("ix_chunks_qdrant_point_id", "chunks", ["qdrant_point_id"], unique=False)

    # Query logs
    op.create_table(
        "query_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("latency", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_query_logs"),
    )
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"], unique=False)

    # Forms
    op.create_table(
        "forms",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("category", sa.String(length=128), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_forms"),
    )
    op.create_index("ix_forms_name", "forms", ["name"], unique=False)
    op.create_index("ix_forms_category", "forms", ["category"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_forms_category", table_name="forms")
    op.drop_index("ix_forms_name", table_name="forms")
    op.drop_table("forms")

    op.drop_index("ix_query_logs_created_at", table_name="query_logs")
    op.drop_table("query_logs")

    op.drop_index("ix_chunks_qdrant_point_id", table_name="chunks")
    op.drop_column("chunks", "qdrant_point_id")
    op.drop_column("chunks", "metadata_json")

    op.drop_index("ix_documents_category", table_name="documents")
    op.drop_column("documents", "form_name")
    op.drop_column("documents", "source")
    op.drop_column("documents", "category")
    op.drop_column("documents", "title")
