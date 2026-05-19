"""production schema

Revision ID: 0001_production_schema
Revises:
Create Date: 2026-05-19
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision = "0001_production_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "videos",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=True),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("duration", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("video_id", sa.String(length=36), sa.ForeignKey("videos.id"), nullable=True),
        sa.Column("clip_id", sa.String(length=36), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "video_chunks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("video_id", sa.String(length=36), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("embedding_backend", sa.String(length=64), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("video_id", "start_time", name="uq_video_chunk_start"),
    )
    op.create_index("ix_video_chunks_video_id", "video_chunks", ["video_id"])
    op.create_index(
        "ix_video_chunks_embedding_cosine",
        "video_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_table(
        "clips",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("video_id", sa.String(length=36), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("object_key", sa.Text(), nullable=True),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "dead_letter_entries",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("video_id", sa.String(length=36), sa.ForeignKey("videos.id"), nullable=True),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_dead_letter_chunk_id", "dead_letter_entries", ["chunk_id"])


def downgrade() -> None:
    op.drop_index("ix_dead_letter_chunk_id", table_name="dead_letter_entries")
    op.drop_table("dead_letter_entries")
    op.drop_table("clips")
    op.drop_index("ix_video_chunks_embedding_cosine", table_name="video_chunks")
    op.drop_index("ix_video_chunks_video_id", table_name="video_chunks")
    op.drop_table("video_chunks")
    op.drop_table("jobs")
    op.drop_table("videos")
