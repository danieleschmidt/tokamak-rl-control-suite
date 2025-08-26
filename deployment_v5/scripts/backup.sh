#!/bin/bash
# Backup script

echo "💾 Starting backup process..."

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup database
echo "Backing up database..."
kubectl exec -n tokamak-rl-production deployment/postgres -- pg_dump tokamak_rl > "$BACKUP_DIR/database.sql"

# Backup configuration
echo "Backing up configuration..."
cp -r /root/repo/deployment_v5 "$BACKUP_DIR/config"

# Backup application data
echo "Backing up application data..."
kubectl exec -n tokamak-rl-production deployment/tokamak-rl-api -- tar czf - /app/data > "$BACKUP_DIR/app_data.tar.gz"

echo "✅ Backup completed: $BACKUP_DIR"
