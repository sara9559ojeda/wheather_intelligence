"""Esquemas Pydantic de autenticación."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

# Validación deliberadamente simple (sin la dependencia email-validator): algo@dominio.tld
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def _check_email(value: str) -> str:
    value = value.strip().lower()
    if not _EMAIL_RE.match(value):
        raise ValueError("Introduce un email válido")
    return value


Email = Annotated[str, StringConstraints(max_length=254), AfterValidator(_check_email)]
# Máximo 128: acota el coste de scrypt frente a contraseñas descomunales.
Password = Annotated[str, Field(min_length=8, max_length=128)]


class RegisterRequest(BaseModel):
    email: Email
    password: Password
    full_name: Annotated[str, StringConstraints(strip_whitespace=True, max_length=80)] = ""


class LoginRequest(BaseModel):
    email: Email
    # sin mínimo: que una contraseña corta falle como credencial incorrecta, no como formato
    password: Annotated[str, Field(min_length=1, max_length=128)]


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str | None
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # segundos
    user: UserOut


class StreamTicket(BaseModel):
    ticket: str
    expires_in: int  # segundos
