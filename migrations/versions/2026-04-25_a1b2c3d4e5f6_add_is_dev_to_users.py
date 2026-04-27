"""Add is_dev column to lazer_users table.

Revision ID: a1b2c3d4e5f6
Revises: f1e2d3c4b5a6
Create Date: 2026-04-25
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'd4e5f6a7b8c9'  # Point to actual head
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_dev column to lazer_users table
    op.add_column(
        'lazer_users',
        sa.Column('is_dev', sa.Boolean(), nullable=False, server_default=sa.false())
    )
    # Create index for is_dev
    op.create_index(
        op.f('ix_lazer_users_is_dev'),
        'lazer_users',
        ['is_dev'],
        unique=False
    )


def downgrade() -> None:
    # Drop index first
    op.drop_index(
        op.f('ix_lazer_users_is_dev'),
        table_name='lazer_users'
    )
    # Drop the column
    op.drop_column('lazer_users', 'is_dev')
