"""Entrypoint de Vercel (Python runtime / preset FastAPI).

Vercel busca una instancia ``app`` en ``index.py`` de la raíz del proyecto; la
aplicación real vive en ``backend/app/main.py`` (se importa como paquete
``backend.app.main``, igual que con ``uvicorn backend.app.main:app``).
Configuración de la función (duración, ficheros excluidos) en ``vercel.json``.
"""

from backend.app.main import app as _app

app = _app
