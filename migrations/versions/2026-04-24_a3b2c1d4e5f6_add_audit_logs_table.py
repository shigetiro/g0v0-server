"""Add audit_logs table.

Revision ID: a3b2c1d4e5f6
Revises:
Create Date: 2026-04-24
"""

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'a3b2c1d4e5f6'
down_revision = 'd2c4f7a8b1e0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('actor_id', sa.BigInteger(), nullable=False),
        sa.Column('actor_username', sa.String(length=255), nullable=False),
        sa.Column('action_type', sa.Enum('USER_BAN', 'USER_UNBAN', 'USER_ROLE_CHANGE', 'USER_WIPE', 'BEATMAP_DELETE', 'BEATMAP_RANK', 'BEATMAP_UNRANK', 'BEATMAP_LOVE', 'BEATMAP_UNLOVE', 'SCORE_DELETE', 'TEAM_DISBAND', 'TEAM_CREATE', 'TEAM_UPDATE', 'SETTINGS_CHANGE', 'ANNOUNCEMENT_CREATE', 'ANNOUNCEMENT_UPDATE', 'ANNOUNCEMENT_DELETE', 'BADGE_CREATE', 'BADGE_UPDATE', 'BADGE_DELETE', 'REPORT_RESOLVE', 'MAINTENANCE_MODE', 'RECALCULATION_TRIGGERED', name='auditactiontype'), nullable=False),
        sa.Column('target_type', sa.Enum('USER', 'BEATMAP', 'BEATMAPSET', 'SCORE', 'TEAM', 'ANNOUNCEMENT', 'BADGE', 'REPORT', 'SYSTEM', name='targettype'), nullable=False),
        sa.Column('target_id', sa.BigInteger(), nullable=True),
        sa.Column('target_name', sa.String(length=255), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['actor_id'], ['lazer_users.id'], name='fk_audit_logs_actor_id'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_action_type'), 'audit_logs', ['action_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_actor_id'), 'audit_logs', ['actor_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_actor_username'), 'audit_logs', ['actor_username'], unique=False)
    op.create_index(op.f('ix_audit_logs_created_at'), 'audit_logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_audit_logs_target_id'), 'audit_logs', ['target_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_target_type'), 'audit_logs', ['target_type'], unique=False)


def downgrade() -> None:
    # Drop audit_logs table
    op.drop_index(op.f('ix_audit_logs_target_type'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_target_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_created_at'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_actor_username'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_actor_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_action_type'), table_name='audit_logs')
    op.drop_table('audit_logs')
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS targettype')
    op.execute('DROP TYPE IF EXISTS auditactiontype')
