"""Add system_settings and recalculation_tasks tables.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-04-24
"""

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'd4e5f6a7b8c9'
down_revision = 'c3d4e5f6a7b8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create system_settings table
    op.create_table(
        'system_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('value_type', sa.String(length=50), nullable=False),
        sa.Column('description', sa.String(length=512), nullable=True),
        sa.Column('updated_by', sa.BigInteger(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['updated_by'], ['lazer_users.id'], name='fk_system_settings_updated_by'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key', name='uq_system_settings_key')
    )
    op.create_index(op.f('ix_system_settings_key'), 'system_settings', ['key'], unique=True)
    op.create_index(op.f('ix_system_settings_updated_at'), 'system_settings', ['updated_at'], unique=False)

    # Create recalculation_tasks table
    op.create_table(
        'recalculation_tasks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('task_type', sa.Enum('USER', 'BEATMAP', 'OVERALL', 'LEADERBOARD', name='recalculationtype'), nullable=False),
        sa.Column('target_id', sa.BigInteger(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', name='recalculationstatus'), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('progress', sa.Float(), nullable=False),
        sa.Column('total_items', sa.Integer(), nullable=True),
        sa.Column('processed_items', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('created_by', sa.BigInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['lazer_users.id'], name='fk_recalculation_tasks_created_by'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_recalculation_tasks_created_at'), 'recalculation_tasks', ['created_at'], unique=False)
    op.create_index(op.f('ix_recalculation_tasks_created_by'), 'recalculation_tasks', ['created_by'], unique=False)
    op.create_index(op.f('ix_recalculation_tasks_status'), 'recalculation_tasks', ['status'], unique=False)
    op.create_index(op.f('ix_recalculation_tasks_target_id'), 'recalculation_tasks', ['target_id'], unique=False)
    op.create_index(op.f('ix_recalculation_tasks_task_type'), 'recalculation_tasks', ['task_type'], unique=False)


def downgrade() -> None:
    # Drop recalculation_tasks table
    op.drop_index(op.f('ix_recalculation_tasks_task_type'), table_name='recalculation_tasks')
    op.drop_index(op.f('ix_recalculation_tasks_target_id'), table_name='recalculation_tasks')
    op.drop_index(op.f('ix_recalculation_tasks_status'), table_name='recalculation_tasks')
    op.drop_index(op.f('ix_recalculation_tasks_created_by'), table_name='recalculation_tasks')
    op.drop_index(op.f('ix_recalculation_tasks_created_at'), table_name='recalculation_tasks')
    op.drop_table('recalculation_tasks')
    op.execute('DROP TYPE IF EXISTS recalculationtype')
    op.execute('DROP TYPE IF EXISTS recalculationstatus')

    # Drop system_settings table
    op.drop_index(op.f('ix_system_settings_updated_at'), table_name='system_settings')
    op.drop_index(op.f('ix_system_settings_key'), table_name='system_settings')
    op.drop_table('system_settings')
