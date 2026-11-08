"""model_runs y cluster_models por ubicación (multi-ciudad)

Hasta ahora había una sola ubicación y estas tablas no guardaban a cuál
pertenecían. Se añade ``location_id``: las filas existentes se asignan a la
única ubicación que había (Madrid-Barajas).

Revision ID: b7e21c4d9a05
Revises: a1c4e7f20b93
Create Date: 2026-11-08 12:00:00.000000+00:00
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = 'b7e21c4d9a05'
down_revision: str | None = 'a1c4e7f20b93'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ('model_runs', 'cluster_models')


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column('location_id', sa.Integer(), nullable=True))
        # Todo lo existente pertenece a la ubicación original (la de menor id).
        op.execute(
            f"UPDATE {table} SET location_id = (SELECT MIN(id) FROM locations) "
            "WHERE location_id IS NULL"
        )
        op.alter_column(table, 'location_id', nullable=False)
        op.create_foreign_key(
            op.f(f'fk_{table}_location_id_locations'), table, 'locations',
            ['location_id'], ['id'], ondelete='CASCADE',
        )
        op.create_index(op.f(f'ix_{table}_location_id'), table, ['location_id'])


def downgrade() -> None:
    for table in _TABLES:
        op.drop_index(op.f(f'ix_{table}_location_id'), table_name=table)
        op.drop_constraint(op.f(f'fk_{table}_location_id_locations'), table, type_='foreignkey')
        op.drop_column(table, 'location_id')
