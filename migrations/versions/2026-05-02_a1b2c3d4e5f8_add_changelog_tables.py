"""Add changelog tables

Revision ID: a1b2c3d4e5f8
Revises: a1b2c3d4e5f7
Create Date: 2026-05-02
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f8'
down_revision = 'a1b2c3d4e5f7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create changelog_streams table
    op.create_table(
        'changelog_streams',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=50), nullable=False),
        sa.Column('display_name', sa.String(length=100), nullable=False),
        sa.Column('is_featured', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('user_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_changelog_streams_id', 'changelog_streams', ['id'])
    op.create_index('ix_changelog_streams_name', 'changelog_streams', ['name'], unique=True)

    # Create changelog_builds table
    op.create_table(
        'changelog_builds',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stream_id', sa.Integer(), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('display_version', sa.String(length=100), nullable=False),
        sa.Column('users', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('github_url', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['stream_id'], ['changelog_streams.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_changelog_builds_id', 'changelog_builds', ['id'])
    op.create_index('ix_changelog_builds_stream_id', 'changelog_builds', ['stream_id'])
    op.create_index('ix_changelog_builds_created_at', 'changelog_builds', ['created_at'])

    # Create changelog_entries table
    op.create_table(
        'changelog_entries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('build_id', sa.Integer(), nullable=False),
        sa.Column('repository', sa.String(length=100), nullable=True, server_default='torii-osu'),
        sa.Column('github_pull_request_id', sa.Integer(), nullable=True),
        sa.Column('github_url', sa.String(length=500), nullable=True, server_default='https://github.com/shikkesora'),
        sa.Column('url', sa.String(length=500), nullable=True),
        sa.Column('type', sa.String(length=20), nullable=True, server_default='misc'),
        sa.Column('category', sa.String(length=20), nullable=True, server_default='other'),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('message_html', sa.Text(), nullable=True),
        sa.Column('major', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('github_user', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['build_id'], ['changelog_builds.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_changelog_entries_id', 'changelog_entries', ['id'])
    op.create_index('ix_changelog_entries_build_id', 'changelog_entries', ['build_id'])
    op.create_index('ix_changelog_entries_type', 'changelog_entries', ['type'])
    op.create_index('ix_changelog_entries_category', 'changelog_entries', ['category'])


def downgrade() -> None:
    op.drop_index('ix_changelog_entries_category', table_name='changelog_entries')
    op.drop_index('ix_changelog_entries_type', table_name='changelog_entries')
    op.drop_index('ix_changelog_entries_build_id', table_name='changelog_entries')
    op.drop_index('ix_changelog_entries_id', table_name='changelog_entries')
    op.drop_table('changelog_entries')

    op.drop_index('ix_changelog_builds_created_at', table_name='changelog_builds')
    op.drop_index('ix_changelog_builds_stream_id', table_name='changelog_builds')
    op.drop_index('ix_changelog_builds_id', table_name='changelog_builds')
    op.drop_table('changelog_builds')

    op.drop_index('ix_changelog_streams_name', table_name='changelog_streams')
    op.drop_index('ix_changelog_streams_id', table_name='changelog_streams')
    op.drop_table('changelog_streams')