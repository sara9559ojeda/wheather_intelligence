"""Pruebas del login (usuarios, contraseñas, JWT, protección de rutas).

* Sin marcador (rápidas, sin BD): hash de contraseñas, tokens, esquemas y que las
  rutas protegidas rechacen peticiones sin sesión / sin ticket.
* ``db``: registro, login, bloqueo por intentos fallidos y ``/me`` contra PostgreSQL
  (cada prueba crea su propio email único y lo borra al terminar).
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.app.core.config import settings
from backend.app.core.security import (
    ALGORITHM,
    TOKEN_ACCESS,
    TOKEN_STREAM,
    AuthNotConfigured,
    InvalidToken,
    create_access_token,
    create_stream_ticket,
    decode_token,
    hash_password,
    verify_password,
)
from backend.app.db.models import User
from backend.app.db.session import SessionLocal
from backend.app.main import app
from backend.app.services import users

SECRET = "secreto-de-pruebas-auth-0123456789abcdef"
client = TestClient(app)


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setattr(settings, "auth_secret_key", SECRET)
    monkeypatch.setattr(settings, "allow_registration", True)


# ---------------------------------------------------------------------------
# Contraseñas
# ---------------------------------------------------------------------------
def test_password_hash_roundtrip_and_salting():
    h1, h2 = hash_password("una-clave-larga"), hash_password("una-clave-larga")
    assert h1 != h2  # sal distinta
    assert h1.startswith("scrypt$") and "una-clave-larga" not in h1
    assert verify_password("una-clave-larga", h1)
    assert not verify_password("otra-clave", h1)


@pytest.mark.parametrize("stored", ["", "texto-cualquiera", "bcrypt$1$2$3$4$5", "scrypt$x$y$z$a$b"])
def test_verify_password_rejects_malformed_hash(stored):
    assert verify_password("lo-que-sea", stored) is False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def test_access_token_roundtrip():
    assert decode_token(create_access_token(42), TOKEN_ACCESS) == 42


def test_token_types_are_not_interchangeable():
    with pytest.raises(InvalidToken):
        decode_token(create_stream_ticket(1), TOKEN_ACCESS)
    with pytest.raises(InvalidToken):
        decode_token(create_access_token(1), TOKEN_STREAM)


def test_expired_token_is_rejected():
    now = datetime.now(UTC)
    old = jwt.encode(
        {"sub": "1", "typ": TOKEN_ACCESS, "iat": now - timedelta(hours=2),
         "exp": now - timedelta(hours=1)},
        SECRET, algorithm=ALGORITHM,
    )
    with pytest.raises(InvalidToken):
        decode_token(old, TOKEN_ACCESS)


def test_token_signed_with_another_key_is_rejected():
    forged = jwt.encode(
        {"sub": "1", "typ": TOKEN_ACCESS, "exp": datetime.now(UTC) + timedelta(hours=1)},
        "otra-clave-distinta-0123456789abcdef", algorithm=ALGORITHM,
    )
    with pytest.raises(InvalidToken):
        decode_token(forged, TOKEN_ACCESS)


def test_unsigned_alg_none_token_is_rejected():
    claims = {"sub": "1", "typ": TOKEN_ACCESS, "exp": int(time.time()) + 3600}
    none_token = jwt.encode(claims, key=None, algorithm="none")
    with pytest.raises(InvalidToken):
        decode_token(none_token, TOKEN_ACCESS)


def test_token_without_required_claims_is_rejected():
    no_exp = jwt.encode({"sub": "1", "typ": TOKEN_ACCESS}, SECRET, algorithm=ALGORITHM)
    with pytest.raises(InvalidToken):
        decode_token(no_exp, TOKEN_ACCESS)


def test_decode_without_secret_raises_not_configured(monkeypatch):
    token = create_access_token(1)
    monkeypatch.setattr(settings, "auth_secret_key", "")
    with pytest.raises(AuthNotConfigured):
        decode_token(token, TOKEN_ACCESS)


# ---------------------------------------------------------------------------
# Protección de rutas (no necesita BD: se rechaza antes de consultar nada)
# ---------------------------------------------------------------------------
PROTECTED = [
    "/api/dashboard", "/api/weather/current", "/api/weather/history", "/api/stats",
    "/api/anomalies", "/api/clusters", "/api/insights", "/api/predictions/latest",
    "/api/auth/me",
]


@pytest.mark.parametrize("path", PROTECTED)
def test_data_routes_require_a_session(path):
    r = client.get(path)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("header", [
    "Bearer no-es-un-jwt",
    "Basic dXNlcjpwYXNz",
])
def test_data_routes_reject_garbage_credentials(header):
    assert client.get("/api/dashboard", headers={"Authorization": header}).status_code == 401


def test_stream_ticket_is_not_a_valid_session():
    """El ticket de WebSocket va en la URL: no debe servir para la API REST."""
    ticket = create_stream_ticket(1)
    r = client.get("/api/dashboard", headers={"Authorization": f"Bearer {ticket}"})
    assert r.status_code == 401


def test_sse_stream_requires_ticket():
    assert client.get("/api/stream").status_code == 401
    assert client.get("/api/stream?ticket=basura").status_code == 401
    # una sesión normal tampoco vale como ticket
    assert client.get(f"/api/stream?ticket={create_access_token(1)}").status_code == 401


def test_websocket_rejects_missing_or_invalid_ticket():
    from starlette.websockets import WebSocketDisconnect

    for url in ("/api/ws", "/api/ws?ticket=basura", f"/api/ws?ticket={create_access_token(1)}"):
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(url):
                pass
        assert exc.value.code == 1008


def test_protected_routes_closed_when_secret_missing(monkeypatch):
    monkeypatch.setattr(settings, "auth_secret_key", "")
    r = client.get("/api/dashboard", headers={"Authorization": "Bearer cualquiera"})
    assert r.status_code == 503
    login = client.post("/api/auth/login", json={"email": "a@b.co", "password": "x"})
    assert login.status_code == 503


def test_health_stays_public():
    assert client.get("/api/health").status_code == 200


def test_cors_preflight_allows_post_with_authorization():
    r = client.options(
        "/api/auth/login",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert r.status_code == 200
    assert "POST" in r.headers["access-control-allow-methods"]


# ---------------------------------------------------------------------------
# Validación de entrada (rechaza antes de tocar la BD)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    {"email": "sin-arroba", "password": "una-clave-larga"},
    {"email": "a@b", "password": "una-clave-larga"},
    {"email": "ana@example.com", "password": "corta"},
    {"email": "ana@example.com", "password": "x" * 129},
    {"email": "ana@example.com", "password": "una-clave-larga", "full_name": "n" * 81},
    {"password": "una-clave-larga"},
])
def test_register_validates_input(payload):
    assert client.post("/api/auth/register", json=payload).status_code == 422


def test_register_disabled(monkeypatch):
    monkeypatch.setattr(settings, "allow_registration", False)
    r = client.post("/api/auth/register",
                    json={"email": "ana@example.com", "password": "una-clave-larga"})
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Registro / login contra PostgreSQL
# ---------------------------------------------------------------------------
@pytest.fixture
def email():
    address = f"test-{uuid.uuid4().hex[:12]}@example.com"
    yield address
    with SessionLocal() as db:
        db.execute(delete(User).where(User.email == address))
        db.commit()


PASSWORD = "una-clave-larga-1"


@pytest.mark.db
def test_register_login_me_flow(email):
    r = client.post("/api/auth/register",
                    json={"email": f"  {email.upper()} ", "password": PASSWORD,
                          "full_name": "  Ana Pérez "})
    assert r.status_code == 201
    body = r.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == settings.auth_token_minutes * 60
    assert body["user"]["email"] == email  # normalizado a minúsculas y sin espacios
    assert body["user"]["full_name"] == "Ana Pérez"
    assert "password" not in str(body).lower().replace("password_hash", "")

    # con ese token ya se pasa la protección
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == email

    login = client.post("/api/auth/login", json={"email": email.upper(), "password": PASSWORD})
    assert login.status_code == 200
    token = login.json()["access_token"]

    ticket = client.post("/api/auth/stream-ticket", headers={"Authorization": f"Bearer {token}"})
    assert ticket.status_code == 200 and ticket.json()["expires_in"] == 60
    assert decode_token(ticket.json()["ticket"], TOKEN_STREAM) == body["user"]["id"]


@pytest.mark.db
def test_register_duplicate_email_conflicts(email):
    payload = {"email": email, "password": PASSWORD}
    assert client.post("/api/auth/register", json=payload).status_code == 201
    r = client.post("/api/auth/register", json={**payload, "email": email.upper()})
    assert r.status_code == 409


@pytest.mark.db
def test_login_wrong_password_and_unknown_user_look_the_same(email):
    client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    bad = client.post("/api/auth/login", json={"email": email, "password": "incorrecta-1"})
    unknown = client.post("/api/auth/login",
                          json={"email": f"nadie-{email}", "password": "incorrecta-1"})
    assert bad.status_code == unknown.status_code == 401
    assert bad.json() == unknown.json()


@pytest.mark.db
def test_account_locks_after_repeated_failures_even_with_right_password(email):
    client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    for _ in range(users.MAX_FAILED_ATTEMPTS):
        r = client.post("/api/auth/login", json={"email": email, "password": "incorrecta-1"})
        assert r.status_code == 401
    r = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) > 0

    # al vencer el bloqueo vuelve a funcionar y el contador se reinicia
    with SessionLocal() as db:
        user = users.get_by_email(db, email)
        user.locked_until = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()
    assert client.post("/api/auth/login",
                       json={"email": email, "password": PASSWORD}).status_code == 200
    with SessionLocal() as db:
        user = users.get_by_email(db, email)
        assert user.failed_attempts == 0 and user.locked_until is None
        assert user.last_login_at is not None


@pytest.mark.db
def test_successful_login_resets_failed_attempts(email):
    client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    for _ in range(users.MAX_FAILED_ATTEMPTS - 1):
        client.post("/api/auth/login", json={"email": email, "password": "incorrecta-1"})
    assert client.post("/api/auth/login",
                       json={"email": email, "password": PASSWORD}).status_code == 200
    for _ in range(users.MAX_FAILED_ATTEMPTS - 1):  # no llega a bloquear: el contador se reinició
        client.post("/api/auth/login", json={"email": email, "password": "incorrecta-1"})
    assert client.post("/api/auth/login",
                       json={"email": email, "password": PASSWORD}).status_code == 200


@pytest.mark.db
def test_deactivated_user_loses_access(email):
    token = client.post("/api/auth/register",
                        json={"email": email, "password": PASSWORD}).json()["access_token"]
    with SessionLocal() as db:
        users.get_by_email(db, email).is_active = False
        db.commit()
    assert client.get("/api/auth/me",
                      headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert client.post("/api/auth/login",
                       json={"email": email, "password": PASSWORD}).status_code == 401


@pytest.mark.db
def test_token_of_deleted_user_is_rejected():
    ghost = create_access_token(2_000_000_000)
    assert client.get("/api/auth/me",
                      headers={"Authorization": f"Bearer {ghost}"}).status_code == 401
