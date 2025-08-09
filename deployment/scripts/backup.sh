#!/bin/bash
set -e

# Tokamak RL Control Suite - Backup Script
echo "💾 Starting backup process"

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/tokamak-rl/$BACKUP_DATE"
S3_BUCKET="tokamak-rl-backups"

# Create backup directory
mkdir -p $BACKUP_DIR

echo "📂 Backup directory: $BACKUP_DIR"

# Backup database
echo "🗃️ Backing up database..."
kubectl exec -n tokamak-rl deployment/postgres -- pg_dump -U tokamak_rl_user tokamak_rl > $BACKUP_DIR/database.sql

# Backup persistent volumes
echo "💾 Backing up persistent volumes..."
kubectl get pvc -n tokamak-rl -o json > $BACKUP_DIR/pvcs.json

# Backup configurations
echo "⚙️ Backing up configurations..."
kubectl get configmaps -n tokamak-rl -o yaml > $BACKUP_DIR/configmaps.yaml
kubectl get secrets -n tokamak-rl -o yaml > $BACKUP_DIR/secrets.yaml

# Backup application data
echo "📊 Backing up application data..."
kubectl cp tokamak-rl/tokamak-rl-data-production:/app/data $BACKUP_DIR/app-data

# Create compressed archive
echo "🗜️ Creating compressed archive..."
cd /backups/tokamak-rl
tar -czf tokamak-rl-backup-$BACKUP_DATE.tar.gz $BACKUP_DATE/

# Upload to S3
echo "☁️ Uploading to S3..."
aws s3 cp tokamak-rl-backup-$BACKUP_DATE.tar.gz s3://$S3_BUCKET/

# Clean up old backups (keep last 30 days)
echo "🧹 Cleaning up old backups..."
find /backups/tokamak-rl -name "tokamak-rl-backup-*.tar.gz" -mtime +30 -delete

echo "✅ Backup completed successfully: tokamak-rl-backup-$BACKUP_DATE.tar.gz"
