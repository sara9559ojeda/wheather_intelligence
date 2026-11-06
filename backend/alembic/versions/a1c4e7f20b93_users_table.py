"""users table (login)

Revision ID: a1c4e7f20b93
Revises: d8dc2bbbe11d
Create Date: 2026-11-06 10:00:00.000000+00:00
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = 'a1c4e7f20b93'
down_revision: str | None = 'd8dc2bbbe11d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(length=254), nullable=False),
    sa.Column('full_name', sa.String(length=80), nullable=True),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('failed_attempts', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('email', name=op.f('uq_users_email'))
    )


def downgrade() -> None:
    op.drop_table('users')
