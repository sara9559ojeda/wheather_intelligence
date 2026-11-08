"""Analista contextual — fase 11 (F5: «Pregúntale a Weather Intelligence»).

    GET  /api/analyst/questions         preguntas sugeridas
    POST /api/analyst/ask?location=     respuesta fundamentada en el contexto de UNA ciudad

Funciona sin clave de Claude (respuesta determinista). Protegido con sesión (JWT);
el token nunca se envía al modelo: solo se usa el id de usuario para los límites.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user, get_db, get_location_slug
from backend.app.db.models import User
from backend.app.schemas.analysis import AnalystAnswer, AnalystAsk, AnalystQuestion
from backend.app.services.analyst import ask, suggested_questions
from backend.app.services.analyst.service import RateLimited

router = APIRouter(tags=["analyst"])


@router.get("/analyst/questions", response_model=list[AnalystQuestion])
def questions() -> list[dict]:
    return suggested_questions()


@router.post("/analyst/ask", response_model=AnalystAnswer)
def analyst_ask(
    body: AnalystAsk,
    db: Session = Depends(get_db),
    slug: str = Depends(get_location_slug),
    user: User = Depends(get_current_user),
) -> dict:
    try:
        return ask(db, slug, user_id=user.id, question=body.question, intent=body.intent,
                   detail=body.detail)
    except RateLimited as exc:
        raise HTTPException(
            429, "Has hecho muchas preguntas esta hora. Inténtalo más tarde.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
