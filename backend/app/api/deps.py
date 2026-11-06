"""Dependencias comunes de FastAPI."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.security import (
    TOKEN_ACCESS,
    TOKEN_STREAM,
    AuthNotConfigured,
    InvalidToken,
    decode_token,
)
from backend.app.db.models import User
from backend.app.db.session import SessionLocal

_bearer = HTTPBearer(auto_error=False)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def auth_disabled() -> HTTPException:
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Autenticación no configurada: falta AUTH_SECRET_KEY en el servidor",
    )


def _unauthorized(detail: str = "Sesión no válida o caducada") -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Exige ``Authorization: Bearer <JWT>`` de un usuario activo."""
    if credentials is None:
        raise _unauthorized("Inicia sesión para continuar")
    try:
        user_id = decode_token(credentials.credentials, TOKEN_ACCESS)
    except AuthNotConfigured:
        raise auth_disabled() from None
    except InvalidToken:
        raise _unauthorized() from None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _unauthorized()
    return user


def require_stream_ticket(ticket: str = Query(default="")) -> int:
    """Ticket de 60 s (``?ticket=``) para el SSE, donde ``EventSource`` no admite cabeceras."""
    try:
        return decode_token(ticket, TOKEN_STREAM)
    except AuthNotConfigured:
        raise auth_disabled() from None
    except InvalidToken:
        raise _unauthorized("Ticket no válido o caducado") from None


def get_location_slug() -> str:
    """Slug de la ubicación configurada (el proyecto trabaja con una)."""
    return settings.wi_location_slug
