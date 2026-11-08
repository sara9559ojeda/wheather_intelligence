"""índice por observation_id en cluster_assignments

La FK ``cluster_assignments.observation_id -> weather_observations.id`` es
``ON DELETE CASCADE`` pero la única restricción única empieza por
``cluster_model_id``: borrar una observación obligaba a un barrido completo de
~1 M de filas por cada fila borrada (borrar una ciudad de prueba tardaba >10 min).
Con este índice el borrado en cascada pasa a ser una búsqueda por índice.

Cambio aditivo (solo crea un índice); no modifica datos.

Revision ID: d5c8e2f1a7b3
Revises: c3a9f1e5b8d2
Create Date: 2026-11-08 19:00:00.000000+00:00
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = 'd5c8e2f1a7b3'
down_revision: str | None = 'c3a9f1e5b8d2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        'ix_cluster_assignments_observation_id', 'cluster_assignments', ['observation_id']
    )


def downgrade() -> None:
    op.drop_index('ix_cluster_assignments_observation_id', table_name='cluster_assignments')
