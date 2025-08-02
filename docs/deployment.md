# 🚀 Deployment Guide

This guide covers deployment strategies and instructions for the tokamak-rl-control-suite across different environments.

## Quick Start

### Development Environment
```bash
# Start development environment
docker-compose up tokamak-dev

# With GPU support
docker-compose --profile gpu up tokamak-gpu

# Access Jupyter Lab
open http://localhost:8888
```

### Production Deployment
```bash
# Build production image
./scripts/build.sh -t production --tag v1.0.0

# Run production container
docker run -d --name tokamak-prod tokamak-rl-control-suite:production-v1.0.0
```

## Build System

### Build Targets

The project uses multi-stage Docker builds with the following targets:

| Target | Purpose | Size | Dependencies |
|--------|---------|------|--------------|
| **development** | Full development environment | ~3GB | All dev tools, Jupyter, debugging |
| **production** | Optimized runtime | ~1GB | Minimal runtime dependencies |
| **docs** | Documentation server | ~500MB | Sphinx, documentation tools |

### Build Script

Use the automated build script for consistent builds:

```bash
# Basic development build
./scripts/build.sh

# Production build with tagging
./scripts/build.sh -t production --tag v1.2.0

# Clean build (no cache)
./scripts/build.sh --clean -v

# Build all targets
./scripts/build.sh -t all

# Build and push to registry
./scripts/build.sh -t production --registry myregistry.io --push
```

### Build Options

```bash
Options:
  -t, --target TARGET       Build target (development|production|docs|all)
  -r, --registry REGISTRY   Docker registry to push to
  --tag TAG                 Image tag [default: latest]
  --push                    Push image to registry after build
  --no-tests                Skip running tests during build
  --clean                   Clean build (no cache)
  --platform PLATFORM      Target platform [default: linux/amd64]
  -v, --verbose             Verbose output
```

## Container Orchestration

### Docker Compose Services

The `docker-compose.yml` defines multiple services for different use cases:

#### Core Services

**Development Environment (`tokamak-dev`)**
- Full development environment with hot reloading
- Jupyter Lab on port 8888
- TensorBoard on port 6006
- Volume mounting for live code editing

**Production Environment (`tokamak-prod`)**
- Optimized production container
- Minimal resource footprint
- Health checks enabled

**Documentation Server (`tokamak-docs`)**
- Sphinx documentation server
- Available on port 8000
- Auto-built documentation

#### Testing and Quality Assurance

**Test Runner (`tokamak-test`)**
```bash
# Run full test suite
docker-compose run --rm tokamak-test

# View test reports
open test-reports/coverage/index.html
```

**Performance Benchmarks (`tokamak-bench`)**
```bash
# Run performance benchmarks
docker-compose run --rm tokamak-bench

# View benchmark results
open benchmark-results/benchmark.html
```

#### GPU and Monitoring (Optional Profiles)

**GPU Environment (`tokamak-gpu`)**
```bash
# Start GPU-enabled environment
docker-compose --profile gpu up tokamak-gpu

# Requires: nvidia-docker2, Docker >= 19.03
```

**Monitoring Dashboard (`monitoring`)**
```bash
# Start monitoring
docker-compose --profile monitoring up monitoring

# Access dashboard
open http://localhost:8080
```

### Service Commands

```bash
# Start specific services
docker-compose up tokamak-dev tensorboard

# Run one-off commands
docker-compose run --rm tokamak-dev pytest tests/unit/

# Scale services
docker-compose up --scale tokamak-dev=3

# View logs
docker-compose logs -f tokamak-dev

# Stop all services
docker-compose down

# Clean up volumes
docker-compose down -v
```

## Production Deployment

### Container Registries

#### Docker Hub
```bash
# Tag and push
docker tag tokamak-rl-control-suite:production-latest myusername/tokamak-rl:v1.0.0
docker push myusername/tokamak-rl:v1.0.0
```

#### AWS ECR
```bash
# Authenticate
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

# Tag and push
docker tag tokamak-rl-control-suite:production-latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/tokamak-rl:v1.0.0
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/tokamak-rl:v1.0.0
```

#### Azure Container Registry
```bash
# Login
az acr login --name myregistry

# Tag and push
docker tag tokamak-rl-control-suite:production-latest myregistry.azurecr.io/tokamak-rl:v1.0.0
docker push myregistry.azurecr.io/tokamak-rl:v1.0.0
```

#### Google Container Registry
```bash
# Configure authentication
gcloud auth configure-docker

# Tag and push
docker tag tokamak-rl-control-suite:production-latest gcr.io/myproject/tokamak-rl:v1.0.0
docker push gcr.io/myproject/tokamak-rl:v1.0.0
```

### Kubernetes Deployment

#### Basic Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tokamak-rl
  namespace: tokamak
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
        image: myregistry.io/tokamak-rl:v1.0.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi" 
            cpu: "1000m"
        env:
        - name: ENVIRONMENT
          value: "production"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### GPU Nodes
```yaml
# For GPU workloads
spec:
  template:
    spec:
      containers:
      - name: tokamak-rl-gpu
        image: myregistry.io/tokamak-rl:gpu-v1.0.0
        resources:
          limits:
            nvidia.com/gpu: 1
      nodeSelector:
        accelerator: nvidia-tesla-k80
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

#### Service and Ingress
```yaml
# k8s/service.yaml
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
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tokamak-rl-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - tokamak-rl.example.com
    secretName: tokamak-rl-tls
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

### Helm Chart

#### Chart Structure
```
charts/tokamak-rl/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   └── secret.yaml
└── charts/
```

#### Installation
```bash
# Add Helm repository
helm repo add tokamak-rl https://charts.tokamak-rl.io

# Install chart
helm install my-tokamak-rl tokamak-rl/tokamak-rl \
  --namespace tokamak \
  --create-namespace \
  --values values-production.yaml

# Upgrade
helm upgrade my-tokamak-rl tokamak-rl/tokamak-rl

# Uninstall
helm uninstall my-tokamak-rl --namespace tokamak
```

## Cloud Platform Deployment

### AWS

#### ECS Fargate
```json
{
  "family": "tokamak-rl",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "tokamak-rl",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/tokamak-rl:v1.0.0",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/tokamak-rl",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### EKS Cluster
```bash
# Create EKS cluster
eksctl create cluster --name tokamak-rl --region us-west-2 --nodegroup-name standard-workers --node-type m5.large --nodes 3

# Deploy application
kubectl apply -f k8s/
```

### Azure

#### Container Instances
```bash
# Create container group
az container create \
  --resource-group myResourceGroup \
  --name tokamak-rl \
  --image myregistry.azurecr.io/tokamak-rl:v1.0.0 \
  --cpu 2 \
  --memory 4 \
  --ports 8080 \
  --dns-name-label tokamak-rl-unique \
  --location eastus
```

#### AKS Cluster
```bash
# Create AKS cluster
az aks create \
  --resource-group myResourceGroup \
  --name tokamak-rl-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group myResourceGroup --name tokamak-rl-cluster
```

### Google Cloud

#### Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy tokamak-rl \
  --image gcr.io/myproject/tokamak-rl:v1.0.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### GKE Cluster
```bash
# Create GKE cluster
gcloud container clusters create tokamak-rl-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials tokamak-rl-cluster --zone us-central1-a
```

## Monitoring and Observability

### Health Checks

The production container includes health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tokamak_rl; print('Health check passed')" || exit 1
```

### Monitoring Stack

Deploy monitoring with the monitoring profile:

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Components:
# - Prometheus metrics collection
# - Grafana dashboards  
# - Custom tokamak-rl metrics
```

### Logging

#### Structured Logging
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_record)

# Configure logging
logging.basicConfig(
    format='%(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()
logger.handlers[0].setFormatter(JSONFormatter())
```

#### Log Aggregation
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  tokamak-dev:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: tokamak.dev
```

## Security

### Container Security

#### Security Scanning
```bash
# Scan with Trivy
trivy image tokamak-rl-control-suite:production-latest

# Scan with Clair
docker run -d --name clair-db arminc/clair-db:latest
docker run -p 6060:6060 --link clair-db:postgres -d --name clair arminc/clair-local-scan:latest
```

#### Runtime Security
```bash
# Run with security options
docker run \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  --cap-drop ALL \
  --cap-add SETGID \
  --cap-add SETUID \
  --security-opt no-new-privileges:true \
  --user 1000:1000 \
  tokamak-rl-control-suite:production-latest
```

### Network Security

#### Docker Networks
```bash
# Create isolated network
docker network create --driver bridge tokamak-network

# Run containers in network
docker run --network tokamak-network tokamak-rl-control-suite:production-latest
```

#### TLS/SSL
```yaml
# docker-compose with TLS
services:
  tokamak-prod:
    environment:
      - SSL_CERT_PATH=/etc/ssl/certs/tokamak.crt
      - SSL_KEY_PATH=/etc/ssl/private/tokamak.key
    volumes:
      - ./ssl/tokamak.crt:/etc/ssl/certs/tokamak.crt:ro
      - ./ssl/tokamak.key:/etc/ssl/private/tokamak.key:ro
```

## Performance Optimization

### Resource Limits

#### Docker
```yaml
services:
  tokamak-prod:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

#### Kubernetes
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Caching

#### Multi-stage Build Optimization
```dockerfile
# Use build cache
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim as runtime
COPY --from=builder /root/.local /root/.local
```

#### Registry Layer Caching
```bash
# Use registry cache
docker buildx build \
  --cache-from type=registry,ref=myregistry.io/tokamak-rl:buildcache \
  --cache-to type=registry,ref=myregistry.io/tokamak-rl:buildcache,mode=max \
  .
```

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Clean build environment
docker system prune -a
docker volume prune

# Check build logs
docker-compose build --no-cache 2>&1 | tee build.log
```

#### Runtime Issues
```bash
# Check container logs
docker-compose logs -f tokamak-dev

# Debug container
docker-compose run --rm tokamak-dev bash

# Check resource usage
docker stats
```

#### Performance Issues
```bash
# Profile container
docker run --rm -it \
  --pid container:tokamak-rl-prod \
  --cap-add SYS_PTRACE \
  alpine:latest \
  sh -c 'apk add --no-cache htop && htop'
```

### Debug Commands

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' tokamak-rl-prod

# View environment variables
docker exec tokamak-rl-prod env

# Check filesystem
docker exec tokamak-rl-prod df -h

# Network debugging
docker exec tokamak-rl-prod netstat -tlnp
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Build and Deploy
on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build images
      run: ./scripts/build.sh -t all --push --registry ${{ secrets.REGISTRY }}
```

### GitLab CI
```yaml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - ./scripts/build.sh -t production --tag $CI_COMMIT_TAG
```

### Jenkins Pipeline
```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh './scripts/build.sh -t production --tag ${BUILD_NUMBER}'
            }
        }
        stage('Test') {
            steps {
                sh 'docker-compose run --rm tokamak-test'
            }
        }
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh './scripts/build.sh --push --registry $REGISTRY'
            }
        }
    }
}
```

---

For deployment questions, check the [troubleshooting guide](troubleshooting/) or file an issue on GitHub.