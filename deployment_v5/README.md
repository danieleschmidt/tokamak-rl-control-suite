# Tokamak RL Control Suite - Production Deployment

## Overview

This directory contains production deployment configurations and scripts for the Tokamak RL Control Suite.

## Prerequisites

- Docker 20.10+
- Kubernetes 1.24+
- kubectl configured for your cluster
- Helm 3.0+ (optional)

## Quick Start

### Local Development with Docker Compose

```bash
# Start all services
cd deployment_v5/docker
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f tokamak-rl
```

### Production Deployment to Kubernetes

```bash
# Deploy to Kubernetes cluster
cd deployment_v5
./scripts/deploy.sh

# Check deployment status
kubectl get pods -n tokamak-rl-production

# Run health checks
./scripts/health-check.sh
```

## Configuration

### Environment Variables

- `TOKAMAK_ENV`: Environment (development/staging/production)
- `TOKAMAK_LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `POSTGRES_PASSWORD`: Database password
- `GRAFANA_PASSWORD`: Grafana admin password

### Scaling

The deployment includes Horizontal Pod Autoscaler (HPA) configuration:

- **Min replicas**: 3
- **Max replicas**: 10
- **CPU threshold**: 70%
- **Memory threshold**: 80%

### Monitoring

Monitoring stack includes:

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Alert Manager**: Alert notifications

Access Grafana at: http://localhost:3000 (admin/admin)

### Backup and Recovery

```bash
# Create backup
./scripts/backup.sh

# Restore from backup
# Follow disaster recovery procedures in DR_PLAN.md
```

## Security

### Security Hardening Applied

- ✅ Non-root container user
- ✅ Minimal base image (Python slim)
- ✅ Security context constraints
- ✅ Network policies
- ✅ Resource limits
- ✅ Health checks
- ✅ Secret management

### Security Checklist

- [ ] Update default passwords
- [ ] Configure TLS certificates
- [ ] Set up proper RBAC
- [ ] Configure network policies
- [ ] Enable audit logging
- [ ] Set up vulnerability scanning

## Troubleshooting

### Common Issues

1. **Pod stuck in Pending state**
   ```bash
   kubectl describe pod <pod-name> -n tokamak-rl-production
   ```

2. **Service not accessible**
   ```bash
   kubectl get svc -n tokamak-rl-production
   kubectl port-forward svc/tokamak-rl-service 8080:80
   ```

3. **High resource usage**
   ```bash
   kubectl top pods -n tokamak-rl-production
   ```

### Logs

```bash
# Application logs
kubectl logs -f deployment/tokamak-rl-api -n tokamak-rl-production

# All pods logs
kubectl logs -f -l app=tokamak-rl -n tokamak-rl-production
```

## Maintenance

### Regular Tasks

- [ ] Monitor resource usage
- [ ] Review logs for errors
- [ ] Update dependencies
- [ ] Run security scans
- [ ] Test backup procedures
- [ ] Review and rotate secrets

### Update Procedure

1. Build new image with version tag
2. Update deployment with new image
3. Monitor rollout status
4. Run smoke tests
5. Rollback if issues detected

## Support

For deployment issues:
1. Check logs and monitoring dashboards
2. Consult troubleshooting guide
3. Contact the development team

## Disaster Recovery

Refer to `DR_PLAN.md` for complete disaster recovery procedures.
