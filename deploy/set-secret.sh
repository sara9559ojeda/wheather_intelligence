#!/usr/bin/env bash
# Guarda un secreto en el .env del servidor sin que aparezca en pantalla ni en el
# historial, y reinicia el backend para que lo lea.
#
#   bash deploy/set-secret.sh ANTHROPIC_API_KEY
set -euo pipefail

NAME="${1:?Uso: set-secret.sh NOMBRE_VARIABLE}"
APP_DIR=/opt/weather-intelligence
ENV_FILE="$APP_DIR/.env"

[[ "$NAME" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "Nombre de variable no válido: $NAME" >&2; exit 1; }

read -r -s -p "Pega el valor de $NAME y pulsa Enter (no se verá): " VALUE
echo
# Quita espacios y las marcas invisibles del "pegado entre corchetes" de algunas terminales
VALUE="${VALUE//$'\e[200~'/}"
VALUE="${VALUE//$'\e[201~'/}"
VALUE="${VALUE//[$'\t\r\n ']/}"
[ -n "$VALUE" ] || { echo "Valor vacío: no se cambia nada." >&2; exit 1; }
if [ "$NAME" = ANTHROPIC_API_KEY ] && ! [[ "$VALUE" =~ ^sk-ant-[A-Za-z0-9_-]{20,}$ ]]; then
  echo "Eso no parece una clave de Anthropic (debe empezar por sk-ant-). No se cambia nada." >&2
  exit 1
fi

umask 077
tmp="$(mktemp "$APP_DIR/.env.XXXXXX")"
grep -v "^${NAME}=" "$ENV_FILE" > "$tmp" || true
printf '%s=%s\n' "$NAME" "$VALUE" >> "$tmp"
mv "$tmp" "$ENV_FILE"
chmod 600 "$ENV_FILE"
unset VALUE

echo "$NAME guardada en .env. Reiniciando el backend..."
cd "$APP_DIR"
docker compose -f docker-compose.prod.yml up -d backend >/dev/null 2>&1

if [ "$NAME" = ANTHROPIC_API_KEY ]; then
  echo "Probando la clave con una llamada corta a Claude..."
  docker compose -f docker-compose.prod.yml exec -T backend python -c '
from backend.app.services.claude.client import ClaudeUnavailable, call_claude
try:
    call_claude("Responde solo: ok", "ok", max_tokens=5)
    print("OK: Claude responde con esta clave.")
except ClaudeUnavailable as e:
    print("ERROR:", e, "- revisa la clave en console.anthropic.com (401 = clave no válida o revocada).")
' 2>&1 | tail -1
fi
echo "Listo. Ya puedes cerrar esta ventana."
