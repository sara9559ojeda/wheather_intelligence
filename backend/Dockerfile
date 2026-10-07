# Imagen del backend (API FastAPI + scheduler de ingesta).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias primero (mejor cacheo)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Código de la aplicación + artefactos de ML + configuración
COPY backend/ ./backend/

EXPOSE 8000

# Migraciones y luego la API. PORT/HOST los inyecta la plataforma en la nube
# (Railway usa HOST=:: para su red privada IPv6); en local valen 0.0.0.0:8000.
CMD ["sh", "-c", "cd /app/backend && python -m alembic upgrade head && cd /app && exec uvicorn backend.app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
