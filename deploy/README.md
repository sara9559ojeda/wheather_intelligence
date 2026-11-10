# Despliegue en AWS Lightsail

Producción de Weather Intelligence: **un solo servidor** con Docker Compose
(`docker-compose.prod.yml`). El frontend sigue en Vercel.

```
Navegador ──► Vercel (frontend)  https://weather-intelligence-front.vercel.app
          └─► Lightsail us-east-2  https://3-128-123-180.sslip.io
                 ├─ caddy    HTTPS automático (Let's Encrypt) + proxy
                 ├─ backend  API + scheduler (ingesta 30 min) + reentrenamiento
                 └─ db       Postgres 16, sin puerto público
```

| Recurso | Valor |
|---|---|
| Instancia | `wi-server`, Ubuntu 24.04, plan `small_3_0` (2 GB, 2 vCPU, 60 GB SSD), `us-east-2a` |
| IP estática | `wi-static-ip` → 3.128.123.180 |
| Firewall | 80 y 443 abiertos; 22 solo desde la IP de administración |
| Snapshots | automáticos diarios a las 08:00 UTC (Lightsail guarda los 7 últimos) |
| Copias de la BD | `pg_dump` diario a las 03:30 UTC en `/var/backups/weather-intelligence` (7 últimas) |
| Secretos | `/opt/weather-intelligence/.env` (permisos 600, solo en el servidor) |
| Clave SSH | `~/.ssh/weather-intelligence-lightsail` (equipo de administración) |

Costo: 12 USD/mes de la instancia + ~1 USD de snapshots.

## Operación

```bash
ssh -i ~/.ssh/weather-intelligence-lightsail ubuntu@3.128.123.180
cd /opt/weather-intelligence
```

| Tarea | Comando (en el servidor) |
|---|---|
| Desplegar cambios del repo | `git pull && docker compose -f docker-compose.prod.yml up -d --build` |
| Estado | `docker compose -f docker-compose.prod.yml ps` |
| Logs del backend | `docker compose -f docker-compose.prod.yml logs -f --tail 100 backend` |
| Copia manual de la BD | `bash deploy/backup.sh` |
| Crear un usuario | `docker compose -f docker-compose.prod.yml exec backend python -m backend.app.create_user <email>` |
| Activar Claude (o cambiar otro secreto) | `bash deploy/set-secret.sh ANTHROPIC_API_KEY` (pide el valor sin mostrarlo y reinicia el backend) |

Restaurar una copia (sustituye la BD actual):

```bash
set -a; . ./.env; set +a
docker compose -f docker-compose.prod.yml stop backend
docker compose -f docker-compose.prod.yml exec -T db pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  --clean --if-exists --no-owner --no-privileges < /var/backups/weather-intelligence/<archivo>.dump
docker compose -f docker-compose.prod.yml start backend
```

Si cambia tu IP pública, el SSH dejará de conectar. Vuelve a abrir el puerto 22
para la nueva IP:

```bash
aws lightsail open-instance-public-ports --profile weather-intelligence --region us-east-2 \
  --instance-name wi-server --port-info "fromPort=22,toPort=22,protocol=tcp,cidrs=<tu-ip>/32"
```

## Montar un servidor desde cero

1. Instancia Ubuntu 24.04 en Lightsail con IP estática y puertos 80/443 abiertos.
2. `sudo bash deploy/setup.sh` (Docker, swap, repo en `/opt/weather-intelligence`, cron de copias).
3. Crear `/opt/weather-intelligence/.env` con: `POSTGRES_USER`, `POSTGRES_PASSWORD`,
   `POSTGRES_DB`, `SITE_ADDRESS` (`<ip-con-guiones>.sslip.io` o un dominio propio),
   `AUTH_SECRET_KEY`, `ALLOW_REGISTRATION=false`, `BACKEND_CORS_ORIGINS` (dominio del frontend)
   y, opcionalmente, `ANTHROPIC_API_KEY`.
4. `docker compose -f docker-compose.prod.yml up -d db`, restaurar un `pg_dump` y después
   `docker compose -f docker-compose.prod.yml up -d --build`.
5. En Vercel: `VITE_API_URL=https://<SITE_ADDRESS>` y volver a desplegar el frontend.

## Plan de AWS y duración

El proyecto de AWS está en el **plan gratuito**: todo se paga con los créditos
(100 USD) y AWS no cobra nada a la tarjeta mientras no se pase al plan de pago.
El despliegue está previsto para **1 mes (del 10/11/2026 al 10/12/2026)**:

| Concepto | Consumo de créditos en 1 mes |
|---|---|
| Lightsail `small_3_0` (IP estática y 3 TB incluidos) | ~12 USD |
| Snapshots automáticos | < 1 USD |
| **Total** | **~13 USD de 100** (0 USD de bolsillo) |

No se activa ningún servicio fuera del plan gratuito y no se pasa el proyecto al
plan de pago. El plan gratuito vence el 10/05/2027, mucho después del fin previsto.

### Cierre al terminar el mes

Borra todo lo que consume créditos (es irreversible; antes, guarda una copia de la BD):

```bash
scp -i ~/.ssh/weather-intelligence-lightsail ubuntu@3.128.123.180:/var/backups/weather-intelligence/*.dump .
aws lightsail delete-instance  --profile weather-intelligence --region us-east-2 --instance-name wi-server --force-delete-add-ons
aws lightsail release-static-ip --profile weather-intelligence --region us-east-2 --static-ip-name wi-static-ip
aws lightsail delete-key-pair   --profile weather-intelligence --region us-east-2 --key-pair-name weather-intelligence
```

`--force-delete-add-ons` borra también los snapshots automáticos. Después, en
Vercel, el frontend quedará sin backend hasta que cambie `VITE_API_URL`.
