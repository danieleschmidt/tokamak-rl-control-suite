# Operational Runbooks

Comprehensive runbooks for operational procedures and incident response for the tokamak-rl-control-suite.

## Contents

### Emergency Procedures
- [`plasma-disruption-response.md`](plasma-disruption-response.md) - Emergency response for plasma disruptions
- [`safety-system-failure.md`](safety-system-failure.md) - Response to safety system failures
- [`emergency-shutdown.md`](emergency-shutdown.md) - Emergency shutdown procedures

### System Operations
- [`service-restart.md`](service-restart.md) - Service restart and recovery procedures
- [`performance-issues.md`](performance-issues.md) - Diagnosing and resolving performance issues
- [`resource-scaling.md`](resource-scaling.md) - Resource scaling procedures
- [`backup-recovery.md`](backup-recovery.md) - Backup and recovery procedures

### Training Operations
- [`training-issues.md`](training-issues.md) - RL training troubleshooting
- [`model-deployment.md`](model-deployment.md) - Model deployment procedures
- [`gpu-issues.md`](gpu-issues.md) - GPU-related issue resolution

### Monitoring & Alerting
- [`alert-investigation.md`](alert-investigation.md) - Alert investigation procedures
- [`monitoring-issues.md`](monitoring-issues.md) - Monitoring system troubleshooting
- [`dashboard-recovery.md`](dashboard-recovery.md) - Dashboard and visualization recovery

## Runbook Format

Each runbook follows a standardized format:

1. **Overview** - Brief description of the issue/procedure
2. **Symptoms** - How to identify the problem
3. **Impact Assessment** - Severity and business impact
4. **Immediate Actions** - First response steps
5. **Investigation Steps** - Detailed troubleshooting
6. **Resolution** - Step-by-step fix procedures
7. **Prevention** - How to prevent recurrence
8. **Escalation** - When and how to escalate

## Emergency Contacts

### Critical Issues (24/7)
- **Plasma Safety Officer**: +1-XXX-XXX-XXXX
- **On-Call Engineer**: Use PagerDuty rotation
- **System Administrator**: +1-XXX-XXX-XXXX

### Business Hours
- **RL Team Lead**: email@company.com
- **DevOps Team**: devops@company.com
- **Security Team**: security@company.com

## Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|---------------|-----------------|
| Critical | 15 minutes | On-Call → Team Lead → Management |
| High | 1 hour | Assigned Engineer → Team Lead |
| Medium | 4 hours | Assigned Engineer |
| Low | Next business day | Queue assignment |

## Quick Reference

### Common Commands
```bash
# Check service status
systemctl status tokamak-rl-control

# View logs
journalctl -u tokamak-rl-control -f

# Check resource usage
htop
nvidia-smi

# Restart service
sudo systemctl restart tokamak-rl-control

# Check alerts
curl http://localhost:9093/api/v1/alerts
```

### Common Queries
```promql
# System health
up{job="tokamak-rl-control"}

# Error rate
rate(application_errors_total[5m])

# Response time
histogram_quantile(0.95, http_request_duration_seconds_bucket)

# GPU utilization
system_gpu_usage_percent
```

## Training and Certification

All operations team members must complete:
1. **Plasma Safety Training** - Understanding tokamak safety principles
2. **RL Operations Training** - Training and inference procedures
3. **System Administration** - Infrastructure and deployment
4. **Incident Response** - Emergency procedures and communication

## Continuous Improvement

- **Monthly runbook reviews** with operations team
- **Post-incident updates** to relevant runbooks
- **Quarterly effectiveness assessment** of procedures
- **Annual comprehensive review** and reorganization

## Related Documentation

- [Architecture Documentation](../ARCHITECTURE.md)
- [Monitoring Guide](../monitoring/README.md)
- [Deployment Procedures](../deployment.md)
- [Security Procedures](../SECURITY.md)