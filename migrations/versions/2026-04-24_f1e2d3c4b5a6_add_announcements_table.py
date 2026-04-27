"""Add announcements table.

Revision ID: f1e2d3c4b5a6
Revises: c7d8e9f0a1b2
Create Date: 2026-04-24
"""

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'f1e2d3c4b5a6'
down_revision = 'c7d8e9f0a1b2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create announcements table
    op.create_table(
        'announcements',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('type', sa.Enum('INFO', 'WARNING', 'ERROR', 'SUCCESS', 'EVENT', 'MAINTENANCE', name='announcementtype'), nullable=False),
        sa.Column('target_roles', sa.JSON(), nullable=True),
        sa.Column('start_at', sa.DateTime(), nullable=False),
        sa.Column('end_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_pinned', sa.Boolean(), nullable=False),
        sa.Column('show_in_client', sa.Boolean(), nullable=False),
        sa.Column('show_on_website', sa.Boolean(), nullable=False),
        sa.Column('created_by', sa.BigInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['lazer_users.id'], name='fk_announcements_created_by'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_announcements_created_at'), 'announcements', ['created_at'], unique=False)
    op.create_index(op.f('ix_announcements_created_by'), 'announcements', ['created_by'], unique=False)
    op.create_index(op.f('ix_announcements_end_at'), 'announcements', ['end_at'], unique=False)
    op.create_index(op.f('ix_announcements_is_active'), 'announcements', ['is_active'], unique=False)
    op.create_index(op.f('ix_announcements_is_pinned'), 'announcements', ['is_pinned'], unique=False)
    op.create_index(op.f('ix_announcements_start_at'), 'announcements', ['start_at'], unique=False)
    op.create_index(op.f('ix_announcements_type'), 'announcements', ['type'], unique=False)


def downgrade() -> None:
    # Drop announcements table
    op.drop_index(op.f('ix_announcements_type'), table_name='announcements')
    op.drop_index(op.f('ix_announcements_start_at'), table_name='announcements')
    op.drop_index(op.f('ix_announcements_is_pinned'), table_name='announcements')
    op.drop_index(op.f('ix_announcements_is_active'), table_name='announcements')
    op.drop_index(op.f('ix_announcements_end_at'), table_name='announcements')
    op.drop_index(op.f('ix_announcements_created_by'), table_name='announcements')
    op.drop_index(op.f('ix_announcements_created_at'), table_name='announcements')
    op.drop_table('announcements')
    op.execute('DROP TYPE IF EXISTS announcementtype')
