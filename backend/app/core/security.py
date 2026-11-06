"""Primitivas de seguridad: hash de contraseñas (scrypt) y tokens JWT (HS256).

* Contraseñas: ``hashlib.scrypt`` de la biblioteca estándar (sin dependencias
  extra). Formato almacenado: ``scrypt$n$r$p$salt_b64$hash_b64``; los
  parámetros viajan en el propio hash, así que se pueden subir más adelante sin
  invalidar los existentes.
* Tokens: JWT firmados con ``AUTH_SECRET_KEY``. Hay dos tipos (claim ``typ``):
  ``access`` (sesión normal, cabecera ``Authorization: Bearer``) y ``stream``
  (ticket de 60 s para abrir el WebSocket/SSE, donde el navegador no puede
  enviar cabeceras y el token tiene que ir en la URL: así lo que queda en los
  logs de acceso caduca casi al instante y no vale para la API REST).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import UTC, datetime, timedelta

import jwt

from backend.app.core.config import settings

ALGORITHM = "HS256"
TOKEN_ACCESS = "access"
TOKEN_STREAM = "stream"
STREAM_TICKET_SECONDS = 60

# Coste de scrypt (N=2^14, r=8, p=1 ≈ 16 MiB y ~50 ms): recomendación OWASP mínima.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
_SCRYPT_MAXMEM = 64 * 1024 * 1024
_SALT_BYTES = 16
_KEY_BYTES = 32


class AuthNotConfigured(RuntimeError):
    """``AUTH_SECRET_KEY`` está vacío: sin secreto no se emiten ni validan tokens."""


class InvalidToken(Exception):
    """Token ausente, mal firmado, caducado o de otro tipo."""


# ---------------------------------------------------------------------------
# Contraseñas
# ---------------------------------------------------------------------------
def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=_KEY_BYTES,
        maxmem=_SCRYPT_MAXMEM,
    )


def hash_password(password: str) -> str:
    salt = os.urandom(_SALT_BYTES)
    digest = _scrypt(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    """Comparación en tiempo constante. Un hash mal formado simplemente no valida."""
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = _scrypt(password, salt, int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# Hash de relleno: al probar un email que no existe se verifica contra este, para
# que "usuario inexistente" y "contraseña incorrecta" tarden lo mismo.
DUMMY_HASH = hash_password("contraseña-de-relleno-no-usar")


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def _secret() -> str:
    if not settings.auth_secret_key:
        raise AuthNotConfigured("AUTH_SECRET_KEY no está configurado")
    return settings.auth_secret_key


def _encode(subject: str, typ: str, lifetime: timedelta) -> str:
    now = datetime.now(UTC)
    claims = {"sub": subject, "typ": typ, "iat": now, "exp": now + lifetime}
    return jwt.encode(claims, _secret(), algorithm=ALGORITHM)


def create_access_token(user_id: int) -> str:
    return _encode(str(user_id), TOKEN_ACCESS, timedelta(minutes=settings.auth_token_minutes))


def create_stream_ticket(user_id: int) -> str:
    return _encode(str(user_id), TOKEN_STREAM, timedelta(seconds=STREAM_TICKET_SECONDS))


def decode_token(token: str, expected_type: str) -> int:
    """Devuelve el id de usuario del token o lanza ``InvalidToken``."""
    try:
        claims = jwt.decode(
            token, _secret(), algorithms=[ALGORITHM], options={"require": ["exp", "sub", "typ"]}
        )
        if claims["typ"] != expected_type:
            raise InvalidToken("tipo de token incorrecto")
        return int(claims["sub"])
    except (jwt.PyJWTError, ValueError, TypeError) as exc:
        raise InvalidToken(str(exc)) from None
