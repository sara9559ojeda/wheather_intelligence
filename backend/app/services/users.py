"""Altas y autenticación de usuarios (lógica compartida por la API y la CLI)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.security import DUMMY_HASH, hash_password, verify_password
from backend.app.db.models import User

MAX_FAILED_ATTEMPTS = 5
LOCK_MINUTES = 15


class EmailTaken(Exception):
    """Ya existe una cuenta con ese email."""


class InvalidCredentials(Exception):
    """Email o contraseña incorrectos (no se distingue cuál, a propósito)."""


class AccountLocked(Exception):
    """Demasiados intentos fallidos: cuenta bloqueada temporalmente."""

    def __init__(self, minutes: int) -> None:
        super().__init__(f"cuenta bloqueada {minutes} min")
        self.minutes = minutes


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == normalize_email(email)))


def create_user(db: Session, email: str, password: str, full_name: str | None = None) -> User:
    user = User(
        email=normalize_email(email),
        full_name=(full_name or "").strip() or None,
        password_hash=hash_password(password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise EmailTaken(email) from None
    db.refresh(user)
    return user


def authenticate(db: Session, email: str, password: str) -> User:
    """Valida las credenciales y devuelve el usuario.

    Tras ``MAX_FAILED_ATTEMPTS`` fallos seguidos la cuenta se bloquea
    ``LOCK_MINUTES`` (también frente a la contraseña correcta). Un email que no
    existe tarda lo mismo que una contraseña errónea.
    """
    user = get_by_email(db, email)
    if user is None:
        verify_password(password, DUMMY_HASH)
        raise InvalidCredentials

    now = datetime.now(UTC)
    if user.locked_until is not None and user.locked_until > now:
        remaining = math.ceil((user.locked_until - now).total_seconds() / 60)
        raise AccountLocked(max(remaining, 1))

    if not user.is_active or not verify_password(password, user.password_hash):
        user.failed_attempts += 1
        if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = now + timedelta(minutes=LOCK_MINUTES)
            user.failed_attempts = 0
        db.commit()
        raise InvalidCredentials

    user.failed_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    db.commit()
    return user
