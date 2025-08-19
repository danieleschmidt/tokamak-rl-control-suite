# Tokamak RL Control System - Deployment Guide

## Overview

The Tokamak RL Control System is a production-ready reinforcement learning framework for autonomous tokamak plasma control, featuring advanced safety systems, real-time monitoring, and global compliance.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Linux/Windows/macOS (cross-platform)
- 4GB+ RAM (8GB+ recommended)
- GPU optional (NVIDIA CUDA, Apple Metal, AMD ROCm supported)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd tokamak-rl-control

# Install dependencies
pip install -r requirements.txt

# Run basic validation
python -m tests.test_basic_functionality
```

## 🏗️ Architecture

### Core Components

1. **Reinforcement Learning Agents** (`src/tokamak_rl/agents.py`)
   - Enhanced SAC (Soft Actor-Critic) with entropy tuning
   - Dreamer model-based agent
   - Plasma-specific reward shaping

2. **Safety Systems** (`src/tokamak_rl/safety.py`)
   - Real-time safety shield with disruption prediction
   - LSTM-based disruption predictor
   - Adaptive constraints and emergency protocols

3. **Physics Engine** (`src/tokamak_rl/physics.py`)
   - Grad-Shafranov equilibrium solver
   - MHD stability analysis
   - Real-time plasma state estimation

4. **Validation Framework** (`src/tokamak_rl/validation.py`)
   - Comprehensive input validation
   - Type checking and constraint verification
   - Plasma physics parameter validation

5. **Security Layer** (`src/tokamak_rl/security.py`)
   - Input sanitization and injection prevention
   - Role-based access control (RBAC)
   - Secure configuration management

6. **Performance Optimization** (`src/tokamak_rl/optimization.py`)
   - Adaptive caching with multiple strategies
   - Parallel processing and resource pooling
   - Memory management and optimization

7. **Monitoring & Diagnostics** (`src/tokamak_rl/enhanced_monitoring.py`)
   - Real-time metrics collection
   - Health monitoring and alerting
   - Performance diagnostics

8. **Distributed Computing** (`src/tokamak_rl/distributed_computing.py`)
   - Load balancing and auto-scaling
   - Multi-node coordination
   - Fault tolerance and recovery

## 🌍 Global-First Features

### Internationalization (`src/tokamak_rl/i18n.py`)

- **Supported Languages**: English, French, German, Italian, Spanish, Japanese, Korean, Chinese (Simplified/Traditional), Russian
- **Localization**: Date/time, numbers, units, measurements
- **Accessibility**: Screen reader support, verbose formatting

```python
from tokamak_rl.i18n import set_global_locale, SupportedLanguage, SupportedRegion

# Set French locale
set_global_locale(SupportedLanguage.FRENCH, SupportedRegion.FR)

# Get localized messages
from tokamak_rl.i18n import _
message = _("system.startup")  # Returns: "Démarrage du Système..."
```

### Compliance Framework (`src/tokamak_rl/compliance.py`)

- **Standards Supported**:
  - ISO 45001 (Occupational Health and Safety)
  - IEC 61513 (Nuclear I&C Systems)
  - IEEE 1012 (Software V&V)
  - NIST 800-53 (Security Controls)
  - GDPR (Data Protection)

```python
from tokamak_rl.compliance import create_compliance_system, ComplianceStandard

standards = [ComplianceStandard.ISO_45001, ComplianceStandard.IEC_61513]
monitor = create_compliance_system(standards)

# Check compliance
result = monitor.check_compliance(system_state)
```

### Cross-Platform Support (`src/tokamak_rl/cross_platform.py`)

- **Platforms**: Linux, Windows, macOS, FreeBSD
- **Architectures**: x86_64, ARM64, ARM32
- **Deployment**: Container, Kubernetes, Edge, Cloud

```python
from tokamak_rl.cross_platform import setup_cross_platform_environment

# Auto-configure for current platform
config = setup_cross_platform_environment()
```

## 🐳 Container Deployment

### Docker

```bash
# Build image
docker build -t tokamak-rl:latest .

# Run container
docker run -d \
  --name tokamak-rl \
  -p 8080:8080 \
  -v /data:/app/data \
  tokamak-rl:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  tokamak-rl:
    build: .
    ports:
      - "8080:8080"
    environment:
      - TOKAMAK_RL_ENV=production
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    restart: unless-stopped
```

## ☸️ Kubernetes Deployment

### Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tokamak-rl
  labels:
    app: tokamak-rl
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tokamak-rl
  template:
    metadata:
      labels:
        app: tokamak-rl
    spec:
      containers:
      - name: tokamak-rl
        image: tokamak-rl:latest
        ports:
        - containerPort: 8080
        env:
        - name: TOKAMAK_RL_ENV
          value: "kubernetes"
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 1000m
            memory: 2Gi
```

### Service and Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tokamak-rl-service
spec:
  selector:
    app: tokamak-rl
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tokamak-rl-ingress
spec:
  rules:
  - host: tokamak-rl.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tokamak-rl-service
            port:
              number: 80
```

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
TOKAMAK_RL_ENV=production          # Environment: development, testing, staging, production
TOKAMAK_RL_LOG_LEVEL=INFO          # Logging level
TOKAMAK_RL_DEBUG=false             # Debug mode

# Directories
TOKAMAK_RL_DATA_DIR=/app/data      # Data directory
TOKAMAK_RL_CONFIG_DIR=/app/config  # Configuration directory
TOKAMAK_RL_LOG_DIR=/app/logs       # Log directory
TOKAMAK_RL_CACHE_DIR=/app/cache    # Cache directory

# Performance
TOKAMAK_RL_MAX_WORKERS=8           # Maximum worker threads
TOKAMAK_RL_CACHE_SIZE=1000         # Cache size
TOKAMAK_RL_GPU_ENABLED=true        # GPU acceleration

# Security
TOKAMAK_RL_SECURITY_MODE=strict    # Security mode: basic, standard, strict
TOKAMAK_RL_ENCRYPT_LOGS=true       # Encrypt audit logs
TOKAMAK_RL_RBAC_ENABLED=true       # Role-based access control

# Compliance
TOKAMAK_RL_COMPLIANCE_LEVEL=high   # Compliance level: basic, standard, high
TOKAMAK_RL_AUDIT_ENABLED=true      # Audit logging
TOKAMAK_RL_RETENTION_DAYS=2555     # Data retention (days)

# Internationalization
TOKAMAK_RL_LANGUAGE=en             # Default language
TOKAMAK_RL_REGION=US               # Default region
TOKAMAK_RL_TIMEZONE=UTC            # Timezone
```

### Configuration File

Create `config/tokamak_rl.json`:

```json
{
  "environment": "production",
  "logging": {
    "level": "INFO",
    "format": "json",
    "rotation": "daily"
  },
  "safety": {
    "enabled": true,
    "disruption_threshold": 0.1,
    "emergency_response_time": 5.0,
    "adaptive_constraints": true
  },
  "performance": {
    "optimization_level": "standard",
    "cache_strategy": "adaptive",
    "parallel_workers": 8,
    "gpu_enabled": true
  },
  "compliance": {
    "standards": ["ISO_45001", "IEC_61513", "IEEE_1012"],
    "audit_level": "comprehensive",
    "data_retention_days": 2555
  },
  "internationalization": {
    "default_language": "en",
    "default_region": "US",
    "supported_locales": ["en_US", "fr_FR", "de_DE", "ja_JP", "zh_CN"]
  }
}
```

## 📊 Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/status

# Metrics endpoint
curl http://localhost:8080/metrics
```

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tokamak-rl'
    static_configs:
      - targets: ['tokamak-rl:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Key metrics to monitor:
- Safety system availability
- Disruption prediction accuracy
- Control action response time
- Plasma parameter stability
- System resource utilization
- Compliance violation count

## 🔒 Security

### Network Security

```bash
# Firewall rules (iptables)
iptables -A INPUT -p tcp --dport 8080 -s trusted_network -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -j DROP

# TLS Configuration
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
```

### Access Control

```python
from tokamak_rl.security import AccessController, SecurityLevel

controller = AccessController()
controller.add_role("operator", SecurityLevel.STANDARD)
controller.add_role("engineer", SecurityLevel.HIGH)
controller.add_role("admin", SecurityLevel.MAXIMUM)
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Missing dependencies
   pip install -r requirements.txt
   
   # Python path issues
   export PYTHONPATH=/path/to/tokamak-rl/src:$PYTHONPATH
   ```

2. **Performance Issues**
   ```bash
   # Check system resources
   python -c "from tokamak_rl.cross_platform import get_system_info; print(get_system_info())"
   
   # Enable performance monitoring
   export TOKAMAK_RL_PROFILE=true
   ```

3. **Safety System Failures**
   ```bash
   # Verify safety system
   python -m tests.test_basic_functionality
   
   # Check emergency response
   python -c "from tokamak_rl.safety import create_safety_system; s = create_safety_system(None); print(s.get_safety_statistics())"
   ```

4. **Compliance Violations**
   ```bash
   # Run compliance check
   python -c "from tokamak_rl.compliance import create_compliance_system, ComplianceStandard; m = create_compliance_system([ComplianceStandard.ISO_45001]); print(m.get_compliance_report())"
   ```

### Log Analysis

```bash
# View system logs
tail -f $TOKAMAK_RL_LOG_DIR/system.log

# Search for errors
grep -i error $TOKAMAK_RL_LOG_DIR/*.log

# Audit log analysis
grep "COMPLIANCE_VIOLATION" $TOKAMAK_RL_LOG_DIR/audit.log
```

## 📈 Performance Tuning

### CPU Optimization

```bash
# Set CPU affinity (Linux)
taskset -c 0-7 python -m tokamak_rl

# NUMA aware execution
numactl --interleave=all python -m tokamak_rl
```

### Memory Optimization

```python
# Configure memory limits
from tokamak_rl.optimization import get_global_optimizer

optimizer = get_global_optimizer()
optimizer.optimization_level = OptimizationLevel.MAXIMUM
```

### GPU Acceleration

```bash
# NVIDIA CUDA
export CUDA_VISIBLE_DEVICES=0
export CUDA_MEMORY_FRACTION=0.8

# Apple Metal
export PYTORCH_ENABLE_MPS_FALLBACK=1

# AMD ROCm
export HIP_VISIBLE_DEVICES=0
```

## 🌐 Multi-Site Deployment

### Site Configuration

```yaml
# Site A (Primary)
tokamak_rl_site_a:
  role: primary
  region: us-east-1
  replicas: 5
  resources:
    cpu: 2000m
    memory: 4Gi

# Site B (Secondary)
tokamak_rl_site_b:
  role: secondary
  region: eu-west-1
  replicas: 3
  resources:
    cpu: 1000m
    memory: 2Gi
```

### Data Synchronization

```bash
# Cross-site replication
kubectl apply -f config/multi-site-replication.yaml

# Backup and restore
kubectl create backup tokamak-rl-backup --include-cluster-resources
```

## 🚀 Scaling Guidelines

### Horizontal Scaling

- **Replicas**: Start with 3, scale based on load
- **CPU**: 1-2 cores per replica minimum
- **Memory**: 2-4GB per replica
- **Storage**: Shared persistent volumes for data

### Vertical Scaling

- **Development**: 2 CPU cores, 4GB RAM
- **Testing**: 4 CPU cores, 8GB RAM
- **Production**: 8+ CPU cores, 16+ GB RAM
- **Edge**: 1-2 CPU cores, 2-4GB RAM

## 📋 Maintenance

### Regular Tasks

```bash
# Daily
- Check system health and alerts
- Verify safety system status
- Review compliance violations
- Monitor resource usage

# Weekly
- Analyze performance metrics
- Review audit logs
- Update dependencies
- Run integration tests

# Monthly
- Security assessment
- Compliance audit
- Capacity planning
- Documentation updates
```

### Backup Strategy

```bash
# Data backup
tar -czf tokamak-rl-backup-$(date +%Y%m%d).tar.gz data/ config/ logs/

# Configuration backup
kubectl get configmap tokamak-rl-config -o yaml > config-backup.yaml

# Database backup (if applicable)
pg_dump tokamak_rl > backup/tokamak_rl_$(date +%Y%m%d).sql
```

## 📞 Support

### Documentation
- Technical Documentation: `/docs/`
- API Reference: `/docs/api/`
- Troubleshooting Guide: `/docs/troubleshooting/`

### Community
- GitHub Issues: Report bugs and feature requests
- Wiki: Community documentation and examples
- Discussions: Q&A and general discussion

### Enterprise Support
- Priority support for production deployments
- Custom compliance requirements
- Performance optimization consulting
- Training and certification programs

---

**Version**: 6.0  
**Last Updated**: 2024  
**Deployment Status**: Production Ready ✅