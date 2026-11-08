"""Respuestas deterministas del analista (plantillas sobre datos reales).

Son la red de seguridad de la plataforma: si no hay clave de Claude, falla el
servicio, se agota el límite de uso o la respuesta de la IA no pasa la
verificación de cifras, el usuario recibe esto. Cada frase sale de ``context``;
si un dato falta, se dice que falta (nunca se rellena).

Cada respuesta distingue tres tipos de afirmación:
  - ``datos``       hecho calculado por el sistema;
  - ``asociacion``  relación estadística (no causal);
  - ``hipotesis``   conocimiento meteorológico general, claramente señalado.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

QUESTIONS: list[tuple[str, str]] = [
    ("normal_temp", "¿Es normal la temperatura actual?"),
    ("unusual", "¿Qué tiene de inusual el tiempo de hoy?"),
    ("regime", "¿Qué régimen meteorológico estamos experimentando?"),
    ("next_6h", "¿Qué se espera durante las próximas seis horas?"),
    ("reliability", "¿Qué tan confiable es el pronóstico?"),
    ("best_model", "¿Qué modelo está funcionando mejor en esta ciudad?"),
    ("why_anomaly", "¿Por qué el sistema considera que esta observación es anómala?"),
]
INTENT_IDS = {q[0] for q in QUESTIONS} | {"overview"}

_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("why_anomaly", ("anomal", "por que el sistema")),
    ("unusual", ("inusual", "raro", "extrano", "excepcional", "atipic")),
    ("regime", ("regimen", "patron", "cluster", "tipo de tiempo")),
    ("reliability", ("confiab", "confianza", "fiable", "incertidumbre", "intervalo", "cobertura",
                     "seguro")),
    ("best_model", ("modelo", "precision", "acierta", "error", "mejor funciona", "baseline",
                    "persistencia")),
    ("next_6h", ("proximas", "seis horas", "6 horas", "que se espera", "pronostico", "llover",
                 "lluvia", "manana", "va a hacer")),
    ("normal_temp", ("normal", "habitual", "percentil", "tipico", "calor", "frio", "temperatura")),
]


def _plain(text: str) -> str:
    n = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def route_intent(question: str) -> str | None:
    """Intención de una pregunta libre (None si no encaja con nada que sepamos responder)."""
    q = _plain(question)
    for intent, words in _KEYWORDS:
        if any(w in q for w in words):
            return intent
    return None


@dataclass
class Answer:
    text: str
    sections: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def _n(v, d: int = 1) -> str:
    return f"{float(v):.{d}f}".replace(".", ",")


def _ev(label: str, value, source: str) -> dict:
    return {"label": label, "value": str(value), "source": source}


def _missing(what: str) -> Answer:
    return Answer(
        text=f"No puedo responder con datos del sistema: {what}.",
        sections=[{"kind": "datos", "text": f"No disponible: {what}."}],
        limitations=["Sin este dato el sistema no emite conclusiones."],
    )


def _var(ctx: dict, key: str) -> dict | None:
    norm = ctx.get("normalidad")
    return next((v for v in norm["variables"] if v["variable"] == key), None) if norm else None


# ------------------------------------------------------------------ intenciones
def normal_temp(ctx: dict, detail: bool) -> Answer:
    norm, t = ctx.get("normalidad"), _var(ctx, "temperature_2m")
    if not norm or not t or t["estado"] != "ok":
        return _missing("no hay suficiente histórico comparable de esta ciudad, época y hora")
    ref = norm["referencia"]
    main = (f"La temperatura actual ({_n(t['valor'])} °C) está en el percentil {_n(t['percentil'], 0)} "
            f"respecto a {ref['n_observaciones']} observaciones comparables ({ref['descripcion']}, "
            f"{ref['periodo_inicio'][:4]}–{ref['periodo_fin'][:4]}). Es «{t['clasificacion'].lower()}»: "
            f"aproximadamente el {_n(t['percentil'], 0)} % de esos registros tuvo temperaturas "
            "iguales o inferiores.")
    lo, hi = t["rango_habitual_p10_p90"]
    main += f" Lo habitual (percentiles 10–90) va de {_n(lo)} a {_n(hi)} °C."
    secs = [{"kind": "datos", "text": main}]
    if t["extremo_univariable"]:
        secs.append({"kind": "hipotesis", "text": (
            "Conocimiento general (no deducido de estos datos): un valor así de poco frecuente "
            "suele relacionarse con cambios de masa de aire o de nubosidad, pero el sistema no "
            "tiene evidencia de la causa. Ser estadísticamente raro no implica que sea peligroso.")})
    lim = [f"Representatividad de la referencia: {ref['representatividad']}"
           + (" (se amplió la ventana por falta de datos)." if ref["ventana_ampliada"] else ".")]
    if detail:
        lim.append("Percentil = % de observaciones comparables con valor menor o igual; solo "
                   "datos históricos anteriores al instante analizado, en hora local de la ciudad.")
    ev = [_ev("Temperatura actual", f"{_n(t['valor'])} °C", "observación actual"),
          _ev("Percentil", _n(t["percentil"]), "normalidad histórica"),
          _ev("Muestra", f"{ref['n_observaciones']} obs. ({ref['anios']} años)", "normalidad histórica")]
    return Answer(" ".join(s["text"] for s in secs), secs, ev, lim)


def unusual(ctx: dict, detail: bool) -> Answer:
    norm = ctx.get("normalidad")
    if not norm:
        return _missing("no hay suficiente histórico comparable")
    an, sim = norm["anomalia"], norm["situaciones_parecidas"]
    extremes = [v for v in norm["variables"] if v["extremo_univariable"]]
    secs: list[dict] = []
    if extremes:
        secs.append({"kind": "datos", "text": "Valores extremos respecto a lo comparable: " + "; ".join(
            f"{v['nombre'].lower()} {_n(v['valor'])} {v['unidad']} (percentil {_n(v['percentil'], 0)})"
            for v in extremes) + "."})
    else:
        secs.append({"kind": "datos", "text": (
            "Ninguna variable está en un extremo (fuera del 2,5–97,5 %) de lo comparable.")})
    if an["disponible"]:
        secs.append({"kind": "datos", "text": (
            "El detector multivariable (Isolation Forest) "
            + ("MARCA la combinación de variables como inusual." if an["multivariable"]
               else "NO marca la combinación de variables como anómala."))})
    if sim["disponible"] and sim["porcentaje"] is not None:
        secs.append({"kind": "datos", "text": (
            f"Condiciones parecidas (temperatura ±1 °C y humedad ±8 puntos) aparecieron en el "
            f"{_n(sim['porcentaje'])} % de las observaciones comparables, en {sim['anios_con_casos']} "
            f"de {sim['anios_totales']} años.")})
    if an.get("variables_mas_alejadas"):
        secs.append({"kind": "asociacion", "text": (
            "Las variables más alejadas de lo habitual son " + " y ".join(
                f"{c['nombre'].lower()} ({c['direccion']})" for c in an["variables_mas_alejadas"])
            + ". Es una asociación estadística, no una causa.")})
    secs.append({"kind": "datos", "text": "Inusual no significa peligroso ni equivale a una alerta oficial."})
    ev = [_ev(v["nombre"], f"p{_n(v['percentil'], 0)}", "normalidad histórica")
          for v in norm["variables"] if v["percentil"] is not None]
    return Answer(" ".join(s["text"] for s in secs), secs, ev,
                  ["El sistema no emite alertas oficiales: para avisos, consulta al servicio "
                   "meteorológico de tu país."])


def why_anomaly(ctx: dict, detail: bool) -> Answer:
    norm = ctx.get("normalidad")
    if not norm or not norm["anomalia"]["disponible"]:
        return _missing("el detector de anomalías no está disponible para esta ciudad")
    an = norm["anomalia"]
    if not an["multivariable"]:
        base = ("El sistema NO considera anómala esta observación: la combinación de variables es "
                "compatible con lo visto en el histórico comparable.")
        extremes = [v["nombre"].lower() for v in norm["variables"] if v["extremo_univariable"]]
        if extremes:
            base += (" Sí hay valores extremos aislados (" + ", ".join(extremes)
                     + "), pero eso no es lo mismo que una anomalía multivariable.")
        return Answer(base, [{"kind": "datos", "text": base}],
                      [_ev("Puntuación", _n(an["puntuacion"], 3), "Isolation Forest")],
                      ["Anomalía = rareza estadística respecto al histórico, no fenómeno extremo."])
    secs = [{"kind": "datos", "text": (
        "El Isolation Forest, entrenado con las desviaciones de temperatura, humedad, presión y "
        "viento respecto a lo esperable para cada mes y hora, aísla esta observación con facilidad: "
        f"su combinación de variables es rara en el histórico (puntuación {_n(an['puntuacion'], 3)}; "
        "más negativa = más inusual).")}]
    if an["variables_mas_alejadas"]:
        secs.append({"kind": "asociacion", "text": (
            "Las variables que más se alejan de lo habitual son " + " y ".join(
                f"{c['nombre'].lower()} (percentil {_n(c['percentil'], 0)}, {c['direccion']})"
                for c in an["variables_mas_alejadas"])
            + ". Es una asociación estadística: el modelo no explica la causa meteorológica.")})
    secs.append({"kind": "hipotesis", "text": (
        "Hipótesis general (no comprobada con estos datos): combinaciones raras suelen deberse al "
        "paso de un frente, a una masa de aire poco común o a datos fuera de rango.")})
    return Answer(" ".join(s["text"] for s in secs), secs,
                  [_ev("Puntuación", _n(an["puntuacion"], 3), "Isolation Forest")],
                  ["Inusual no significa peligroso."])


def regime(ctx: dict, detail: bool) -> Answer:
    reg = ctx.get("regimen")
    if not reg:
        return _missing("esta ciudad no tiene un modelo de regímenes o faltan observaciones recientes")
    a = reg["actual"]
    # la descripción ya incluye valores típicos, frecuencia, estación y permanencia
    secs = [{"kind": "datos", "text": (
        f"La observación actual se parece más al régimen «{a['etiqueta']}». {a['descripcion']} "
        f"Distancia al centroide: {a['distancia']}.")}]
    if reg["transiciones_historicas"]:
        top = reg["transiciones_historicas"][0]
        secs.append({"kind": "datos", "text": (
            f"Cuando este régimen termina, lo más frecuente en el histórico ha sido pasar a "
            f"«{top['hacia']}» ({_n(top['frecuencia'] * 100, 0)} % de {top['casos']} cambios). Es una "
            "frecuencia histórica, no una predicción.")})
    if a["segundo_regimen_cercano"] and (a["razon_distancia_segundo"] or 0) > 0.85:
        secs.append({"kind": "datos", "text": (
            f"Está casi a la misma distancia de «{a['segundo_regimen_cercano']}»: la frontera es difusa.")})
    ev = [_ev("Régimen", a["etiqueta"], "K-Means"),
          _ev("Frecuencia histórica", f"{_n(a['frecuencia_historica_pct'])} %", "perfil de regímenes")]
    return Answer(" ".join(s["text"] for s in secs), secs, ev,
                  [reg["nota"], "Las etiquetas se derivan de los perfiles estadísticos de la ciudad."])


def next_6h(ctx: dict, detail: bool) -> Answer:
    fc = ctx.get("pronostico")
    if not fc:
        return _missing("no hay modelo entrenado o datos recientes suficientes para pronosticar")
    rows = [f for f in fc if f["horizonte_h"] in (1, 3, 6)]
    parts = [f"a {f['horizonte_h']} h: {_n(f['temperatura_c'])} °C (intervalo {_n(f['minimo_c'])}–"
             f"{_n(f['maximo_c'])} °C)" for f in rows]
    cur = ctx.get("observacion_actual")
    text = "Para las próximas horas el modelo estima: " + "; ".join(parts) + "."
    if cur:
        text += f" Ahora la temperatura es {_n(cur['temperatura_c'])} °C."
    h6 = next((f for f in rows if f["horizonte_h"] == 6), rows[-1])
    text += (f" La confiabilidad del pronóstico a {h6['horizonte_h']} h es "
             f"«{h6['confiabilidad'].replace('_', ' ')}».")
    secs = [{"kind": "datos", "text": text}]
    ev = [_ev(f"Pronóstico a {f['horizonte_h']} h", f"{_n(f['temperatura_c'])} °C", "modelo + intervalo")
          for f in rows]
    return Answer(text, secs, ev, [
        "Solo se pronostica la temperatura; el sistema no pronostica lluvia ni viento.",
        "El intervalo es un margen de error histórico, no una garantía."])


def reliability(ctx: dict, detail: bool) -> Answer:
    fc = ctx.get("pronostico")
    if not fc:
        return _missing("no hay pronóstico vigente que evaluar")
    secs = []
    ev = []
    for f in fc:
        h = f["horizonte_h"]
        if f["cobertura_real_pct"] is not None and f["pronosticos_verificados"] >= 30:
            cov = (f"los intervalos incluyeron el valor observado en el {_n(f['cobertura_real_pct'], 0)} % "
                   f"de {f['pronosticos_verificados']} casos verificados (nominal "
                   f"{_n(f['nivel_nominal'] * 100, 0)} %)")
        elif f["cobertura_offline_pct"] is not None:
            cov = (f"aún hay solo {f['pronosticos_verificados'] or 'cero'} casos reales verificados; en el "
                   f"periodo de prueba offline cubrieron el {_n(f['cobertura_offline_pct'], 0)} %")
        else:
            cov = "todavía no hay casos suficientes para medir la cobertura"
        secs.append({"kind": "datos", "text": (
            f"A {h} h: {_n(f['temperatura_c'])} °C, intervalo {_n(f['minimo_c'])}–{_n(f['maximo_c'])} °C "
            f"(ancho {_n(f['ancho_c'])} °C). Confiabilidad «{f['confiabilidad'].replace('_', ' ')}»: {cov}.")})
        ev.append(_ev(f"Confiabilidad a {h} h", f["confiabilidad"], "cobertura verificada"))
    secs.append({"kind": "datos", "text": (
        "Importante: la cobertura es una frecuencia histórica de aciertos del intervalo, no la "
        "probabilidad de que un pronóstico concreto acierte.")})
    return Answer(" ".join(s["text"] for s in secs), secs, ev,
                  ["La etiqueta se basa en cobertura verificada, deterioro reciente y comparación "
                   "con la persistencia, no solo en el ancho del intervalo."])


def best_model(ctx: dict, detail: bool) -> Answer:
    ev_live, models = ctx.get("evaluacion_pronosticos"), ctx.get("modelos")
    if not ev_live and not models:
        return _missing("no hay modelos registrados ni pronósticos verificados")
    secs: list[dict] = []
    ev: list[dict] = []
    if ev_live:
        for h in ev_live["horizontes"]:
            line = (f"A {h['horizonte_h']} h, con {h['n']} pronósticos verificados, el error absoluto "
                    f"medio fue {_n(h['mae_c'])} °C")
            if h["mae_persistencia_c"] is not None:
                line += f" frente a {_n(h['mae_persistencia_c'])} °C de suponer que nada cambia"
            if h["mae_open_meteo_c"] is not None:
                line += (f"; en las mismas horas Open-Meteo erró {_n(h['mae_open_meteo_c'])} °C y el "
                         f"modelo {_n(h['mae_modelo_en_esas_horas_c'])} °C")
            secs.append({"kind": "datos", "text": line + "."})
            ev.append(_ev(f"MAE a {h['horizonte_h']} h", f"{_n(h['mae_c'])} °C", "pronósticos verificados"))
    if models:
        champs = [m for m in models if m["campeon"]]
        if champs:
            secs.append({"kind": "datos", "text": (
                "Modelos elegidos en el entrenamiento (campeones por horizonte): " + "; ".join(
                    f"{m['horizonte_h']} h → {m['modelo'].replace('_', ' ')}"
                    + (f" (RMSE de prueba {_n(m['rmse_prueba_c'])} °C)" if m["rmse_prueba_c"] else "")
                    for m in champs) + ". Son resultados sobre un periodo de prueba histórico, no "
                "pronósticos reales emitidos.")})
    if not ev_live:
        secs.append({"kind": "datos", "text": (
            "Aún no hay pronósticos reales verificados, así que no se puede afirmar cuál funciona "
            "mejor en la práctica.")})
    return Answer(" ".join(s["text"] for s in secs), secs, ev,
                  ["Error pasado ≠ precisión garantizada.",
                   "«Observado» es el dato de Open-Meteo, no una estación."])


def overview(ctx: dict, detail: bool) -> Answer:
    parts: list[Answer] = []
    for fn in (normal_temp, regime, next_6h):
        a = fn(ctx, detail)
        if not a.text.startswith("No puedo responder"):
            parts.append(a)
    if not parts:
        return _missing("la ciudad todavía no tiene datos suficientes")
    cur = ctx.get("observacion_actual")
    head = ""
    if cur:
        head = (f"Situación en {ctx['ciudad']['nombre']}: {_n(cur['temperatura_c'])} °C, humedad "
                f"{_n(cur['humedad_pct'], 0)} %, viento {_n(cur['viento_kmh'])} km/h"
                f"{' (dato con retraso)' if cur['retrasado'] else ''}. ")
    secs = [s for a in parts for s in a.sections]
    return Answer(head + " ".join(a.text for a in parts), secs,
                  [e for a in parts for e in a.evidence],
                  sorted({lim for a in parts for lim in a.limitations}))


_HANDLERS = {
    "normal_temp": normal_temp, "unusual": unusual, "regime": regime, "next_6h": next_6h,
    "reliability": reliability, "best_model": best_model, "why_anomaly": why_anomaly,
    "overview": overview,
}


def deterministic_answer(intent: str, ctx: dict, detail: bool = False) -> Answer:
    return _HANDLERS[intent](ctx, detail)
