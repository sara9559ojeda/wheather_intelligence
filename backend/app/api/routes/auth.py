"""Login del dashboard: registro, inicio de sesión y datos del usuario.

Sesión sin estado: ``POST /auth/login`` devuelve un JWT que el frontend envía en
``Authorization: Bearer``. Para el WebSocket (``/ws``) y el SSE (``/stream``),
que no admiten cabeceras desde el navegador, ``POST /auth/stream-ticket`` canjea
la sesión por un ticket de 60 s que va en la URL (``?ticket=``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.deps import auth_disabled, get_current_user, get_db
from backend.app.core.config import settings
from backend.app.core.security import (
    STREAM_TICKET_SECONDS,
    create_access_token,
    create_stream_ticket,
)
from backend.app.db.models import User
from backend.app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    StreamTicket,
    TokenResponse,
    UserOut,
)
from backend.app.services import users

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger(__name__)


def _require_configured() -> None:
    if not settings.auth_secret_key:
        raise auth_disabled()


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id),
        expires_in=settings.auth_token_minutes * 60,
        user=UserOut.model_validate(user),
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    _require_configured()
    if not settings.allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "El registro de cuentas está desactivado")
    try:
        user = users.create_user(db, body.email, body.password, body.full_name)
    except users.EmailTaken:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Ya existe una cuenta con ese email"
        ) from None
    log.info("Cuenta creada: id=%s", user.id)
    return _token_response(user)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    _require_configured()
    try:
        user = users.authenticate(db, body.email, body.password)
    except users.AccountLocked as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Demasiados intentos fallidos. Prueba de nuevo en {exc.minutes} min",
            headers={"Retry-After": str(exc.minutes * 60)},
        ) from None
    except users.InvalidCredentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Email o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    return _token_response(user)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.post("/stream-ticket", response_model=StreamTicket)
def stream_ticket(user: User = Depends(get_current_user)) -> StreamTicket:
    return StreamTicket(ticket=create_stream_ticket(user.id), expires_in=STREAM_TICKET_SECONDS)
