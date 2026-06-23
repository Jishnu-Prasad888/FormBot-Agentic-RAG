"""kag extension

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Documents
    op.add_column("documents", sa.Column("content_hash", sa.String(length=128), nullable=True))
    op.add_column("documents", sa.Column("document_version", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("processing_status", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("processing_log", sa.Text(), nullable=True))
    op.create_index("ix_documents_document_version", "documents", ["document_version"], unique=False)
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"], unique=False)

    # Chunks
    op.add_column("chunks", sa.Column("vector_id", sa.String(length=64), nullable=True))
    op.add_column("chunks", sa.Column("chunk_type", sa.String(length=64), nullable=True))
    op.add_column("chunks", sa.Column("content_summary", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("extracted_entities", sa.JSON(), nullable=True))
    op.add_column("chunks", sa.Column("section", sa.String(length=256), nullable=True))
    op.add_column("chunks", sa.Column("field_name", sa.String(length=256), nullable=True))
    op.add_column("chunks", sa.Column("requirement_tags", sa.JSON(), nullable=True))
    op.add_column("chunks", sa.Column("regulatory_reference", sa.JSON(), nullable=True))
    op.add_column("chunks", sa.Column("confidence_score", sa.Float(), nullable=True))
    op.add_column("chunks", sa.Column("chunk_position", sa.Integer(), nullable=True))
    op.create_index("ix_chunks_vector_id", "chunks", ["vector_id"], unique=False)
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"], unique=False)

    # Form versions
    op.create_table(
        "form_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("form_id", sa.String(length=36), sa.ForeignKey("forms.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=True),
        sa.Column("supersedes_id", sa.String(length=36), sa.ForeignKey("form_versions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("effective_date", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_form_versions"),
    )
    op.create_index("ix_form_versions_form_id", "form_versions", ["form_id"], unique=False)
    op.create_index("ix_form_versions_version", "form_versions", ["version"], unique=False)

    # Fields
    op.create_table(
        "fields",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("form_version_id", sa.String(length=36), sa.ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("field_type", sa.String(length=64), nullable=True),
        sa.Column("validation_rules", sa.JSON(), nullable=True),
        sa.Column("required", sa.Boolean(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_fields"),
    )
    op.create_index("ix_fields_form_version_id", "fields", ["form_version_id"], unique=False)
    op.create_index("ix_fields_name", "fields", ["name"], unique=False)

    # Field dependencies
    op.create_table(
        "field_dependencies",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_field_id", sa.String(length=36), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_field_id", sa.String(length=36), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("condition", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_field_dependencies"),
    )
    op.create_index("ix_field_dependencies_source", "field_dependencies", ["source_field_id"], unique=False)
    op.create_index("ix_field_dependencies_target", "field_dependencies", ["target_field_id"], unique=False)

    # Regulations
    op.create_table(
        "regulations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("authority", sa.String(length=256), nullable=True),
        sa.Column("effective_date", sa.DateTime(), nullable=True),
        sa.Column("citation", sa.String(length=256), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_regulations"),
    )
    op.create_index("ix_regulations_title", "regulations", ["title"], unique=False)
    op.create_index("ix_regulations_citation", "regulations", ["citation"], unique=False)

    # Requirements
    op.create_table(
        "requirements",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("applicability", sa.String(length=256), nullable=True),
        sa.Column("regulation_id", sa.String(length=36), sa.ForeignKey("regulations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("regulation_ref", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_requirements"),
    )
    op.create_index("ix_requirements_regulation_ref", "requirements", ["regulation_ref"], unique=False)

    # Form requirements
    op.create_table(
        "form_requirements",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("form_version_id", sa.String(length=36), sa.ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("requirement_id", sa.String(length=36), sa.ForeignKey("requirements.id", ondelete="CASCADE"), nullable=False),
        sa.Column("applies_if", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_form_requirements"),
    )
    op.create_index("ix_form_requirements_form_version_id", "form_requirements", ["form_version_id"], unique=False)
    op.create_index("ix_form_requirements_requirement_id", "form_requirements", ["requirement_id"], unique=False)

    # Form regulations
    op.create_table(
        "form_regulations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("form_version_id", sa.String(length=36), sa.ForeignKey("form_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("regulation_id", sa.String(length=36), sa.ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation_type", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_form_regulations"),
    )
    op.create_index("ix_form_regulations_form_version_id", "form_regulations", ["form_version_id"], unique=False)
    op.create_index("ix_form_regulations_regulation_id", "form_regulations", ["regulation_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_form_regulations_regulation_id", table_name="form_regulations")
    op.drop_index("ix_form_regulations_form_version_id", table_name="form_regulations")
    op.drop_table("form_regulations")

    op.drop_index("ix_form_requirements_requirement_id", table_name="form_requirements")
    op.drop_index("ix_form_requirements_form_version_id", table_name="form_requirements")
    op.drop_table("form_requirements")

    op.drop_index("ix_requirements_regulation_ref", table_name="requirements")
    op.drop_table("requirements")

    op.drop_index("ix_regulations_citation", table_name="regulations")
    op.drop_index("ix_regulations_title", table_name="regulations")
    op.drop_table("regulations")

    op.drop_index("ix_field_dependencies_target", table_name="field_dependencies")
    op.drop_index("ix_field_dependencies_source", table_name="field_dependencies")
    op.drop_table("field_dependencies")

    op.drop_index("ix_fields_name", table_name="fields")
    op.drop_index("ix_fields_form_version_id", table_name="fields")
    op.drop_table("fields")

    op.drop_index("ix_form_versions_version", table_name="form_versions")
    op.drop_index("ix_form_versions_form_id", table_name="form_versions")
    op.drop_table("form_versions")

    op.drop_index("ix_chunks_chunk_type", table_name="chunks")
    op.drop_index("ix_chunks_vector_id", table_name="chunks")
    op.drop_column("chunks", "chunk_position")
    op.drop_column("chunks", "confidence_score")
    op.drop_column("chunks", "regulatory_reference")
    op.drop_column("chunks", "requirement_tags")
    op.drop_column("chunks", "field_name")
    op.drop_column("chunks", "section")
    op.drop_column("chunks", "extracted_entities")
    op.drop_column("chunks", "content_summary")
    op.drop_column("chunks", "chunk_type")
    op.drop_column("chunks", "vector_id")

    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.drop_index("ix_documents_document_version", table_name="documents")
    op.drop_column("documents", "processing_log")
    op.drop_column("documents", "processing_status")
    op.drop_column("documents", "document_version")
    op.drop_column("documents", "content_hash")
