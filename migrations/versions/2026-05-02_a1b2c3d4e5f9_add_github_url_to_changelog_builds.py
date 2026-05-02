"""Add github_url to changelog_builds

Revision ID: a1b2c3d4e5f9
Revises: a1b2c3d4e5f8
Create Date: 2026-05-02
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f9'
down_revision = 'a1b2c3d4e5f8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('changelog_builds', sa.Column('github_url', sa.String(length=500), nullable=True))


def downgrade() -> None:
    op.drop_column('changelog_builds', 'github_url')