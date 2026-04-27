"""Add daily_challenge table

Revision ID: a1b2c3d4e5f7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-26
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f7'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop table if exists from previous failed migration
    try:
        op.execute('DROP TABLE IF EXISTS daily_challenge')
    except Exception:
        pass

    # Drop foreign key constraint if exists
    try:
        op.drop_constraint('fk_daily_challenge_room', 'daily_challenge', type_='foreignkey')
    except Exception:
        pass

    # Create daily_challenge table with proper types (room_id must match rooms.id = BigInteger)
    op.create_table(
        'daily_challenge',
        sa.Column('date', sa.Date(), nullable=False, index=True),
        sa.Column('beatmap_id', sa.BigInteger(), nullable=False),
        sa.Column('ruleset_id', sa.Integer(), nullable=False),
        sa.Column('required_mods', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('allowed_mods', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('room_id', sa.BigInteger(), nullable=True),
        sa.Column('max_attempts', sa.Integer(), nullable=True),
        sa.Column('time_limit', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('date'),
    )


def downgrade() -> None:
    op.drop_table('daily_challenge')
