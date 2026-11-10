#!/usr/bin/env bash
# =====================================================================
# Prepara una instancia Ubuntu 24.04 (AWS Lightsail) para Weather Intelligence.
# Idempotente: se puede volver a ejecutar.
#
#   sudo bash deploy/setup.sh
#
# Instala Docker (paquetes de Ubuntu), crea 2 GB de swap, clona/actualiza el
# repo en /opt/weather-intelligence y programa la copia nocturna de la BD.
# =====================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/sara9559ojeda/wheather_intelligence.git}"
APP_DIR=/opt/weather-intelligence
APP_USER="${SUDO_USER:-ubuntu}"

echo "==> Paquetes"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q docker.io docker-compose-v2 git unattended-upgrades
systemctl enable --now docker
usermod -aG docker "$APP_USER"

echo "==> Swap de 2 GB (el reentrenamiento puede tener picos de memoria)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
sysctl -q vm.swappiness=10
grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf

echo "==> Código en $APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$APP_DIR"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "==> Copia nocturna de la BD (03:30 UTC, se guardan 7)"
install -d -o "$APP_USER" -g "$APP_USER" /var/backups/weather-intelligence
cat > /etc/cron.d/weather-intelligence-backup <<EOF
30 3 * * * $APP_USER bash $APP_DIR/deploy/backup.sh >> /var/log/weather-intelligence-backup.log 2>&1
EOF
touch /var/log/weather-intelligence-backup.log
chown "$APP_USER:$APP_USER" /var/log/weather-intelligence-backup.log

echo "==> Listo. Falta: crear $APP_DIR/.env y arrancar:"
echo "    cd $APP_DIR && docker compose -f docker-compose.prod.yml up -d --build"
