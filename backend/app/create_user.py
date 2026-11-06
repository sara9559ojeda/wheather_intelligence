"""Crea una cuenta desde la línea de comandos (útil con ``ALLOW_REGISTRATION=false``).

    python -m backend.app.create_user ana@example.com --name "Ana Pérez"

Pide la contraseña por teclado (sin eco). En Docker:

    docker compose exec backend python -m backend.app.create_user ana@example.com
"""

from __future__ import annotations

import argparse
import getpass
import sys

from backend.app.db.session import SessionLocal
from backend.app.schemas.auth import LoginRequest
from backend.app.services import users


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crea un usuario del dashboard")
    parser.add_argument("email")
    parser.add_argument("--name", default="", help="nombre para mostrar (opcional)")
    args = parser.parse_args(argv)

    try:
        email = LoginRequest(email=args.email, password="x").email  # reutiliza la validación
    except ValueError:
        print("Email no válido", file=sys.stderr)
        return 2

    password = getpass.getpass("Contraseña (mín. 8 caracteres): ")
    if password != getpass.getpass("Repite la contraseña: "):
        print("Las contraseñas no coinciden", file=sys.stderr)
        return 2
    if not 8 <= len(password) <= 128:
        print("La contraseña debe tener entre 8 y 128 caracteres", file=sys.stderr)
        return 2

    db = SessionLocal()
    try:
        user = users.create_user(db, email, password, args.name)
    except users.EmailTaken:
        print(f"Ya existe una cuenta con {email}", file=sys.stderr)
        return 1
    finally:
        db.close()
    print(f"Cuenta creada: {user.email} (id {user.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
