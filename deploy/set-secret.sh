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
VALUE="${VALUE//[$'\t\r\n ']/}"
[ -n "$VALUE" ] || { echo "Valor vacío: no se cambia nada." >&2; exit 1; }

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
echo "Listo. Ya puedes cerrar esta ventana."
