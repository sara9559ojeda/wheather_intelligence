"""forecast_log (pronósticos emitidos) y perfil de regímenes

- ``forecast_log``: registro append-only de cada pronóstico emitido, con su
  verificación posterior. Sobrevive a los re-entrenamientos (``model_run_id``
  pasa a NULL si el run desaparece; el resto del registro se conserva).
- ``cluster_models.profile``: perfil estadístico de los regímenes (JSONB).

Cambio aditivo: no modifica ni borra datos existentes.

Revision ID: c3a9f1e5b8d2
Revises: b7e21c4d9a05
Create Date: 2026-11-08 18:00:00.000000+00:00
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c3a9f1e5b8d2'
down_revision: str | None = 'b7e21c4d9a05'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('cluster_models', sa.Column('profile', postgresql.JSONB(), nullable=True))

    op.create_table(
        'forecast_log',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('location_id', sa.Integer(), nullable=False),
        sa.Column('horizon_hours', sa.SmallInteger(), nullable=False),
        sa.Column('issued_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column('base_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('target_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('predicted_temperature', sa.Numeric(5, 2), nullable=False),
        sa.Column('lower_bound', sa.Numeric(5, 2), nullable=True),
        sa.Column('upper_bound', sa.Numeric(5, 2), nullable=True),
        sa.Column('interval_level', sa.Numeric(4, 3), nullable=True),
        sa.Column('interval_method', sa.String(40), nullable=True),
        sa.Column('model_type', sa.String(40), nullable=False),
        sa.Column('model_run_id', sa.Integer(), nullable=True),
        sa.Column('model_version', sa.String(80), nullable=True),
        sa.Column('baseline_persistence', sa.Numeric(5, 2), nullable=True),
        sa.Column('reference_forecast', sa.Numeric(5, 2), nullable=True),
        sa.Column('reference_source', sa.String(40), nullable=True),
        sa.Column('status', sa.String(10), server_default=sa.text("'pending'"), nullable=False),
        sa.Column('observed_temperature', sa.Numeric(5, 2), nullable=True),
        sa.Column('observed_source', sa.String(16), nullable=True),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error', sa.Numeric(6, 3), nullable=True),
        sa.ForeignKeyConstraint(['location_id'], ['locations.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['model_run_id'], ['model_runs.id'], ondelete='SET NULL'),
        sa.UniqueConstraint('location_id', 'horizon_hours', 'base_time'),
        sa.CheckConstraint("status in ('pending', 'verified', 'expired')",
                           name='forecast_status_allowed'),
    )
    op.create_index('ix_forecast_log_location_id_status_target_time', 'forecast_log',
                    ['location_id', 'status', 'target_time'])
    op.create_index('ix_forecast_log_location_id_horizon_hours_target_time', 'forecast_log',
                    ['location_id', 'horizon_hours', 'target_time'])


def downgrade() -> None:
    op.drop_index('ix_forecast_log_location_id_horizon_hours_target_time',
                  table_name='forecast_log')
    op.drop_index('ix_forecast_log_location_id_status_target_time', table_name='forecast_log')
    op.drop_table('forecast_log')
    op.drop_column('cluster_models', 'profile')
