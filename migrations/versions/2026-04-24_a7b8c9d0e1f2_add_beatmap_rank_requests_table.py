"""Add beatmap_rank_requests table.

Revision ID: a7b8c9d0e1f2
Revises: f1e2d3c4b5a6
Create Date: 2026-04-24
"""

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'a7b8c9d0e1f2'
down_revision = 'f1e2d3c4b5a6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create beatmap_rank_requests table
    op.create_table(
        'beatmap_rank_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('beatmapset_id', sa.Integer(), nullable=False),
        sa.Column('requester_id', sa.BigInteger(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'APPROVED', 'REJECTED', 'CANCELLED', name='rankrequeststatus'), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('reviewed_by', sa.BigInteger(), nullable=True),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('rejection_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('reviewed_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['beatmapset_id'], ['beatmapsets.id'], name='fk_beatmap_rank_requests_beatmapset_id'),
        sa.ForeignKeyConstraint(['requester_id'], ['lazer_users.id'], name='fk_beatmap_rank_requests_requester_id'),
        sa.ForeignKeyConstraint(['reviewed_by'], ['lazer_users.id'], name='fk_beatmap_rank_requests_reviewed_by'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_beatmap_rank_requests_beatmapset_id'), 'beatmap_rank_requests', ['beatmapset_id'], unique=False)
    op.create_index(op.f('ix_beatmap_rank_requests_created_at'), 'beatmap_rank_requests', ['created_at'], unique=False)
    op.create_index(op.f('ix_beatmap_rank_requests_requester_id'), 'beatmap_rank_requests', ['requester_id'], unique=False)
    op.create_index(op.f('ix_beatmap_rank_requests_reviewed_by'), 'beatmap_rank_requests', ['reviewed_by'], unique=False)
    op.create_index(op.f('ix_beatmap_rank_requests_status'), 'beatmap_rank_requests', ['status'], unique=False)


def downgrade() -> None:
    # Drop beatmap_rank_requests table
    op.drop_index(op.f('ix_beatmap_rank_requests_status'), table_name='beatmap_rank_requests')
    op.drop_index(op.f('ix_beatmap_rank_requests_reviewed_by'), table_name='beatmap_rank_requests')
    op.drop_index(op.f('ix_beatmap_rank_requests_requester_id'), table_name='beatmap_rank_requests')
    op.drop_index(op.f('ix_beatmap_rank_requests_created_at'), table_name='beatmap_rank_requests')
    op.drop_index(op.f('ix_beatmap_rank_requests_beatmapset_id'), table_name='beatmap_rank_requests')
    op.drop_table('beatmap_rank_requests')
    op.execute('DROP TYPE IF EXISTS rankrequeststatus')
