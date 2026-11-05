"""Engine y fábrica de sesiones de SQLAlchemy."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from backend.app.core.config import settings


def _engine_kwargs(pool_mode: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "pool_pre_ping": True,   # detecta conexiones caídas antes de usarlas
        "future": True,
    }
    if pool_mode == "null":
        # Serverless (Vercel): las instancias se congelan y mueren sin aviso, así que
        # no se guardan conexiones abiertas; el pooling lo hace el pooler de Neon
        # (PgBouncer en modo transacción), con el que no conviene usar prepared
        # statements del lado del servidor.
        kwargs["poolclass"] = NullPool
        kwargs["connect_args"] = {"prepare_threshold": None}
    return kwargs


engine = create_engine(settings.database_url, **_engine_kwargs(settings.db_pool_mode))

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Sesión transaccional: commit al salir, rollback si hay excepción."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
