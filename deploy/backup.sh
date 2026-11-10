#!/usr/bin/env bash
# Copia de la BD con pg_dump (formato custom) y rotación: se guardan las 7 últimas.
# Complementa los snapshots automáticos de Lightsail (disco entero).
set -euo pipefail

APP_DIR=/opt/weather-intelligence
DEST=/var/backups/weather-intelligence
KEEP=7

cd "$APP_DIR"
# shellcheck disable=SC1091
set -a; . ./.env; set +a

file="$DEST/wi-$(date -u +%Y%m%dT%H%M%SZ).dump"
docker compose -f docker-compose.prod.yml exec -T db \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc --no-owner --no-privileges > "$file.tmp"
mv "$file.tmp" "$file"
echo "$(date -u +%FT%TZ) copia: $file ($(du -h "$file" | cut -f1))"

ls -1t "$DEST"/wi-*.dump | tail -n +$((KEEP + 1)) | xargs -r rm --
