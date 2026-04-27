"""Add client_logs table.

Revision ID: c7d8e9f0a1b2
Revises: a3b2c1d4e5f6
Create Date: 2026-04-24
"""

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'c7d8e9f0a1b2'
down_revision = 'a3b2c1d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create client_logs table
    op.create_table(
        'client_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=True),
        sa.Column('username', sa.String(length=255), nullable=True),
        sa.Column('user_avatar_url', sa.String(length=512), nullable=True),
        sa.Column('client_version', sa.String(length=255), nullable=False),
        sa.Column('client_hash', sa.String(length=64), nullable=True),
        sa.Column('os_version', sa.String(length=255), nullable=True),
        sa.Column('log_type', sa.Enum('CRASH', 'ERROR', 'WARNING', 'PERFORMANCE', 'INFO', name='clientlogtype'), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('stack_trace', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['lazer_users.id'], name='fk_client_logs_user_id'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_client_logs_client_hash'), 'client_logs', ['client_hash'], unique=False)
    op.create_index(op.f('ix_client_logs_client_version'), 'client_logs', ['client_version'], unique=False)
    op.create_index(op.f('ix_client_logs_created_at'), 'client_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_client_logs_log_type'), 'client_logs', ['log_type'], unique=False)
    op.create_index(op.f('ix_client_logs_user_id'), 'client_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_client_logs_username'), 'client_logs', ['username'], unique=False)


def downgrade() -> None:
    # Drop client_logs table
    op.drop_index(op.f('ix_client_logs_username'), table_name='client_logs')
    op.drop_index(op.f('ix_client_logs_user_id'), table_name='client_logs')
    op.drop_index(op.f('ix_client_logs_log_type'), table_name='client_logs')
    op.drop_index(op.f('ix_client_logs_created_at'), table_name='client_logs')
    op.drop_index(op.f('ix_client_logs_client_version'), table_name='client_logs')
    op.drop_index(op.f('ix_client_logs_client_hash'), table_name='client_logs')
    op.drop_table('client_logs')
    op.execute('DROP TYPE IF EXISTS clientlogtype')
