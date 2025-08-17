#!/usr/bin/env python3
"""
PRODUCTION DEPLOYMENT SYSTEM v6.0
===================================

Advanced production deployment configuration with Docker, Kubernetes,
monitoring, and infrastructure-as-code for tokamak-rl system.
"""

import sys
import os
import json
import time
import logging
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
# Try to import yaml, fallback to manual YAML generation
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    # Simple YAML formatter
    class SimpleYAML:
        @staticmethod
        def dump(data, stream=None, **kwargs):
            import json
            result = json.dumps(data, indent=2)
            if stream:
                stream.write(result)
            else:
                return result
    yaml = SimpleYAML()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    environment: str = "production"
    namespace: str = "tokamak-rl"
    image_tag: str = "v6.0"
    replicas: int = 3
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    storage_size: str = "10Gi"
    domain: str = "tokamak-rl.example.com"
    enable_tls: bool = True
    enable_monitoring: bool = True
    enable_autoscaling: bool = True
    max_replicas: int = 10
    min_replicas: int = 2

class DockerfileGenerator:
    """Generate optimized Dockerfile for production"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_dockerfile(self) -> str:
        """Generate production-ready Dockerfile"""
        dockerfile_content = f"""# Multi-stage build for tokamak-rl v6.0
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    gcc \\
    g++ \\
    gfortran \\
    libopenblas-dev \\
    liblapack-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY pyproject.toml /tmp/
WORKDIR /tmp
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \\
    && pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r tokamak && useradd -r -g tokamak tokamak

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libopenblas0 \\
    liblapack3 \\
    libgomp1 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY *.py /app/
COPY README.md LICENSE /app/

# Set ownership and permissions
RUN chown -R tokamak:tokamak /app \\
    && chmod -R 755 /app

# Create data directories
RUN mkdir -p /app/data /app/logs /app/checkpoints \\
    && chown -R tokamak:tokamak /app/data /app/logs /app/checkpoints

# Switch to non-root user
USER tokamak

# Environment variables
ENV PYTHONPATH="/app/src" \\
    PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    TOKAMAK_ENV=production \\
    TOKAMAK_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "tokamak_rl.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
"""
        return dockerfile_content

class KubernetesManifestGenerator:
    """Generate Kubernetes manifests for production deployment"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace manifest"""
        return {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': self.config.namespace,
                'labels': {
                    'app': 'tokamak-rl',
                    'environment': self.config.environment,
                    'version': self.config.image_tag
                }
            }
        }
    
    def generate_deployment(self) -> Dict[str, Any]:
        """Generate deployment manifest"""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'tokamak-rl',
                'namespace': self.config.namespace,
                'labels': {
                    'app': 'tokamak-rl',
                    'version': self.config.image_tag
                }
            },
            'spec': {
                'replicas': self.config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': 'tokamak-rl'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'tokamak-rl',
                            'version': self.config.image_tag
                        },
                        'annotations': {
                            'prometheus.io/scrape': 'true',
                            'prometheus.io/port': '8000',
                            'prometheus.io/path': '/metrics'
                        }
                    },
                    'spec': {
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        },
                        'containers': [{
                            'name': 'tokamak-rl',
                            'image': f'tokamak-rl:{self.config.image_tag}',
                            'imagePullPolicy': 'Always',
                            'ports': [{
                                'containerPort': 8000,
                                'protocol': 'TCP'
                            }],
                            'resources': {
                                'requests': {
                                    'cpu': self.config.cpu_request,
                                    'memory': self.config.memory_request
                                },
                                'limits': {
                                    'cpu': self.config.cpu_limit,
                                    'memory': self.config.memory_limit
                                }
                            },
                            'env': [
                                {
                                    'name': 'TOKAMAK_ENV',
                                    'value': self.config.environment
                                },
                                {
                                    'name': 'TOKAMAK_NAMESPACE',
                                    'valueFrom': {
                                        'fieldRef': {
                                            'fieldPath': 'metadata.namespace'
                                        }
                                    }
                                },
                                {
                                    'name': 'TOKAMAK_POD_NAME',
                                    'valueFrom': {
                                        'fieldRef': {
                                            'fieldPath': 'metadata.name'
                                        }
                                    }
                                }
                            ],
                            'volumeMounts': [
                                {
                                    'name': 'data-storage',
                                    'mountPath': '/app/data'
                                },
                                {
                                    'name': 'config',
                                    'mountPath': '/app/config',
                                    'readOnly': True
                                }
                            ],
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30,
                                'timeoutSeconds': 10,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            }
                        }],
                        'volumes': [
                            {
                                'name': 'data-storage',
                                'persistentVolumeClaim': {
                                    'claimName': 'tokamak-rl-data'
                                }
                            },
                            {
                                'name': 'config',
                                'configMap': {
                                    'name': 'tokamak-rl-config'
                                }
                            }
                        ]
                    }
                }
            }
        }
    
    def generate_service(self) -> Dict[str, Any]:
        """Generate service manifest"""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'tokamak-rl-service',
                'namespace': self.config.namespace,
                'labels': {
                    'app': 'tokamak-rl'
                }
            },
            'spec': {
                'selector': {
                    'app': 'tokamak-rl'
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8000,
                    'protocol': 'TCP'
                }],
                'type': 'ClusterIP'
            }
        }
    
    def generate_ingress(self) -> Dict[str, Any]:
        """Generate ingress manifest"""
        annotations = {
            'kubernetes.io/ingress.class': 'nginx',
            'nginx.ingress.kubernetes.io/rewrite-target': '/',
            'nginx.ingress.kubernetes.io/ssl-redirect': 'true',
            'nginx.ingress.kubernetes.io/force-ssl-redirect': 'true'
        }
        
        if self.config.enable_tls:
            annotations.update({
                'cert-manager.io/cluster-issuer': 'letsencrypt-prod',
                'nginx.ingress.kubernetes.io/ssl-protocols': 'TLSv1.2 TLSv1.3'
            })
        
        ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': 'tokamak-rl-ingress',
                'namespace': self.config.namespace,
                'annotations': annotations
            },
            'spec': {
                'rules': [{
                    'host': self.config.domain,
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': 'tokamak-rl-service',
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        if self.config.enable_tls:
            ingress['spec']['tls'] = [{
                'hosts': [self.config.domain],
                'secretName': 'tokamak-rl-tls'
            }]
        
        return ingress
    
    def generate_hpa(self) -> Dict[str, Any]:
        """Generate horizontal pod autoscaler manifest"""
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'tokamak-rl-hpa',
                'namespace': self.config.namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'tokamak-rl'
                },
                'minReplicas': self.config.min_replicas,
                'maxReplicas': self.config.max_replicas,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ]
            }
        }
    
    def generate_pvc(self) -> Dict[str, Any]:
        """Generate persistent volume claim manifest"""
        return {
            'apiVersion': 'v1',
            'kind': 'PersistentVolumeClaim',
            'metadata': {
                'name': 'tokamak-rl-data',
                'namespace': self.config.namespace
            },
            'spec': {
                'accessModes': ['ReadWriteMany'],
                'resources': {
                    'requests': {
                        'storage': self.config.storage_size
                    }
                },
                'storageClassName': 'fast-ssd'
            }
        }
    
    def generate_configmap(self) -> Dict[str, Any]:
        """Generate config map manifest"""
        config_data = {
            'app.yaml': yaml.dump({
                'tokamak_rl': {
                    'environment': self.config.environment,
                    'log_level': 'INFO',
                    'metrics': {
                        'enabled': True,
                        'port': 8000,
                        'path': '/metrics'
                    },
                    'health_check': {
                        'enabled': True,
                        'timeout': 30
                    },
                    'plasma_control': {
                        'max_current': 20.0,
                        'safety_factor_limit': 1.5,
                        'disruption_threshold': 0.8
                    },
                    'performance': {
                        'batch_size': 32,
                        'cache_size': 10000,
                        'worker_threads': 4
                    }
                }
            })
        }
        
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'tokamak-rl-config',
                'namespace': self.config.namespace
            },
            'data': config_data
        }

class MonitoringSetup:
    """Setup monitoring and observability stack"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_prometheus_config(self) -> Dict[str, Any]:
        """Generate Prometheus configuration"""
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'prometheus-config',
                'namespace': self.config.namespace
            },
            'data': {
                'prometheus.yml': yaml.dump({
                    'global': {
                        'scrape_interval': '15s',
                        'evaluation_interval': '15s'
                    },
                    'scrape_configs': [
                        {
                            'job_name': 'tokamak-rl',
                            'kubernetes_sd_configs': [{
                                'role': 'pod',
                                'namespaces': {
                                    'names': [self.config.namespace]
                                }
                            }],
                            'relabel_configs': [
                                {
                                    'source_labels': ['__meta_kubernetes_pod_annotation_prometheus_io_scrape'],
                                    'action': 'keep',
                                    'regex': 'true'
                                },
                                {
                                    'source_labels': ['__meta_kubernetes_pod_annotation_prometheus_io_path'],
                                    'action': 'replace',
                                    'target_label': '__metrics_path__',
                                    'regex': '(.+)'
                                }
                            ]
                        }
                    ]
                })
            }
        }
    
    def generate_grafana_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration"""
        dashboard = {
            'dashboard': {
                'id': None,
                'title': 'Tokamak-RL Production Metrics',
                'tags': ['tokamak-rl', 'production'],
                'timezone': 'UTC',
                'panels': [
                    {
                        'id': 1,
                        'title': 'Request Latency',
                        'type': 'graph',
                        'targets': [{
                            'expr': 'histogram_quantile(0.95, tokamak_request_duration_seconds_bucket)',
                            'legendFormat': 'p95 latency'
                        }]
                    },
                    {
                        'id': 2,
                        'title': 'Throughput',
                        'type': 'graph',
                        'targets': [{
                            'expr': 'rate(tokamak_requests_total[5m])',
                            'legendFormat': 'Requests/sec'
                        }]
                    },
                    {
                        'id': 3,
                        'title': 'Error Rate',
                        'type': 'graph',
                        'targets': [{
                            'expr': 'rate(tokamak_errors_total[5m])',
                            'legendFormat': 'Errors/sec'
                        }]
                    },
                    {
                        'id': 4,
                        'title': 'Plasma Safety Metrics',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'tokamak_q_min',
                                'legendFormat': 'q_min'
                            },
                            {
                                'expr': 'tokamak_disruption_risk',
                                'legendFormat': 'Disruption Risk'
                            }
                        ]
                    }
                ],
                'time': {
                    'from': 'now-1h',
                    'to': 'now'
                },
                'refresh': '5s'
            }
        }
        
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'grafana-dashboard',
                'namespace': self.config.namespace,
                'labels': {
                    'grafana_dashboard': '1'
                }
            },
            'data': {
                'tokamak-rl.json': json.dumps(dashboard)
            }
        }

class DeploymentOrchestrator:
    """Orchestrate the complete production deployment"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.dockerfile_gen = DockerfileGenerator(config)
        self.k8s_gen = KubernetesManifestGenerator(config)
        self.monitoring = MonitoringSetup(config)
        
        self.deployment_dir = "/root/repo/deployment_v6"
        self.manifests_dir = f"{self.deployment_dir}/k8s"
        self.docker_dir = f"{self.deployment_dir}/docker"
        self.monitoring_dir = f"{self.deployment_dir}/monitoring"
    
    def prepare_deployment(self) -> Dict[str, Any]:
        """Prepare complete deployment configuration"""
        logger.info("🚀 PREPARING PRODUCTION DEPLOYMENT v6.0")
        
        start_time = time.time()
        results = {
            'status': 'success',
            'files_created': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Create directories
            self._create_directories()
            
            # Generate Docker configuration
            self._generate_docker_files(results)
            
            # Generate Kubernetes manifests
            self._generate_k8s_manifests(results)
            
            # Generate monitoring configuration
            self._generate_monitoring_config(results)
            
            # Generate deployment scripts
            self._generate_deployment_scripts(results)
            
            # Generate documentation
            self._generate_deployment_docs(results)
            
            execution_time = time.time() - start_time
            results['execution_time_seconds'] = execution_time
            
            logger.info(f"✅ Deployment preparation completed in {execution_time:.2f}s")
            logger.info(f"📁 Files created: {len(results['files_created'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Deployment preparation failed: {e}")
            results['status'] = 'failed'
            results['errors'].append(str(e))
            return results
    
    def _create_directories(self):
        """Create deployment directory structure"""
        dirs = [
            self.deployment_dir,
            self.manifests_dir,
            self.docker_dir,
            self.monitoring_dir,
            f"{self.deployment_dir}/scripts",
            f"{self.deployment_dir}/docs"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _generate_docker_files(self, results: Dict[str, Any]):
        """Generate Docker-related files"""
        # Dockerfile
        dockerfile_path = f"{self.docker_dir}/Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(self.dockerfile_gen.generate_dockerfile())
        results['files_created'].append(dockerfile_path)
        
        # Docker Compose for local development
        compose_content = {
            'version': '3.8',
            'services': {
                'tokamak-rl': {
                    'build': {
                        'context': '../../',
                        'dockerfile': 'deployment_v6/docker/Dockerfile'
                    },
                    'ports': ['8000:8000'],
                    'environment': {
                        'TOKAMAK_ENV': 'development',
                        'TOKAMAK_LOG_LEVEL': 'DEBUG'
                    },
                    'volumes': [
                        '../../data:/app/data',
                        '../../logs:/app/logs'
                    ]
                }
            }
        }
        
        compose_path = f"{self.docker_dir}/docker-compose.yml"
        with open(compose_path, 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False)
        results['files_created'].append(compose_path)
        
        # .dockerignore
        dockerignore_content = """
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
.gitignore
.pytest_cache/
.coverage
.venv/
venv/
.env
*.log
deployment_v6/
tests/
docs/
*.md
Dockerfile*
docker-compose*.yml
"""
        
        dockerignore_path = f"{self.docker_dir}/.dockerignore"
        with open(dockerignore_path, 'w') as f:
            f.write(dockerignore_content.strip())
        results['files_created'].append(dockerignore_path)
    
    def _generate_k8s_manifests(self, results: Dict[str, Any]):
        """Generate Kubernetes manifests"""
        manifests = {
            'namespace.yaml': self.k8s_gen.generate_namespace(),
            'configmap.yaml': self.k8s_gen.generate_configmap(),
            'pvc.yaml': self.k8s_gen.generate_pvc(),
            'deployment.yaml': self.k8s_gen.generate_deployment(),
            'service.yaml': self.k8s_gen.generate_service(),
            'ingress.yaml': self.k8s_gen.generate_ingress()
        }
        
        if self.config.enable_autoscaling:
            manifests['hpa.yaml'] = self.k8s_gen.generate_hpa()
        
        for filename, manifest in manifests.items():
            file_path = f"{self.manifests_dir}/{filename}"
            with open(file_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
            results['files_created'].append(file_path)
    
    def _generate_monitoring_config(self, results: Dict[str, Any]):
        """Generate monitoring configuration"""
        if not self.config.enable_monitoring:
            return
        
        monitoring_configs = {
            'prometheus-config.yaml': self.monitoring.generate_prometheus_config(),
            'grafana-dashboard.yaml': self.monitoring.generate_grafana_dashboard()
        }
        
        for filename, config in monitoring_configs.items():
            file_path = f"{self.monitoring_dir}/{filename}"
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            results['files_created'].append(file_path)
    
    def _generate_deployment_scripts(self, results: Dict[str, Any]):
        """Generate deployment automation scripts"""
        
        # Build script
        build_script = f"""#!/bin/bash
set -e

echo "🐳 Building Tokamak-RL Docker image..."
docker build -t tokamak-rl:{self.config.image_tag} -f deployment_v6/docker/Dockerfile .

echo "🏷️  Tagging image..."
docker tag tokamak-rl:{self.config.image_tag} tokamak-rl:latest

echo "✅ Build completed successfully!"
"""
        
        build_path = f"{self.deployment_dir}/scripts/build.sh"
        with open(build_path, 'w') as f:
            f.write(build_script)
        os.chmod(build_path, 0o755)
        results['files_created'].append(build_path)
        
        # Deploy script
        deploy_script = f"""#!/bin/bash
set -e

NAMESPACE="{self.config.namespace}"

echo "🚀 Deploying Tokamak-RL to Kubernetes..."

# Apply manifests in order
echo "📁 Creating namespace..."
kubectl apply -f deployment_v6/k8s/namespace.yaml

echo "⚙️  Applying configuration..."
kubectl apply -f deployment_v6/k8s/configmap.yaml
kubectl apply -f deployment_v6/k8s/pvc.yaml

echo "🚀 Deploying application..."
kubectl apply -f deployment_v6/k8s/deployment.yaml
kubectl apply -f deployment_v6/k8s/service.yaml
kubectl apply -f deployment_v6/k8s/ingress.yaml

if [ -f "deployment_v6/k8s/hpa.yaml" ]; then
    echo "📈 Setting up autoscaling..."
    kubectl apply -f deployment_v6/k8s/hpa.yaml
fi

echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/tokamak-rl -n $NAMESPACE --timeout=300s

echo "✅ Deployment completed successfully!"
echo "🌐 Application available at: https://{self.config.domain}"
"""
        
        deploy_path = f"{self.deployment_dir}/scripts/deploy.sh"
        with open(deploy_path, 'w') as f:
            f.write(deploy_script)
        os.chmod(deploy_path, 0o755)
        results['files_created'].append(deploy_path)
        
        # Health check script
        health_script = """#!/bin/bash

NAMESPACE="tokamak-rl"

echo "🏥 Checking Tokamak-RL health..."

# Check pods
echo "📊 Pod status:"
kubectl get pods -n $NAMESPACE

# Check services
echo "🌐 Service status:"
kubectl get services -n $NAMESPACE

# Check ingress
echo "🚪 Ingress status:"
kubectl get ingress -n $NAMESPACE

# Check HPA (if exists)
if kubectl get hpa -n $NAMESPACE tokamak-rl-hpa >/dev/null 2>&1; then
    echo "📈 Autoscaler status:"
    kubectl get hpa -n $NAMESPACE
fi

echo "✅ Health check completed!"
"""
        
        health_path = f"{self.deployment_dir}/scripts/health-check.sh"
        with open(health_path, 'w') as f:
            f.write(health_script)
        os.chmod(health_path, 0o755)
        results['files_created'].append(health_path)
    
    def _generate_deployment_docs(self, results: Dict[str, Any]):
        """Generate deployment documentation"""
        
        readme_content = f"""# Tokamak-RL Production Deployment v6.0

## Overview

This directory contains all configuration and scripts needed for production deployment of Tokamak-RL v6.0.

## Architecture

- **Environment**: {self.config.environment}
- **Namespace**: {self.config.namespace}
- **Replicas**: {self.config.replicas} (auto-scaling: {self.config.min_replicas}-{self.config.max_replicas})
- **Resources**: {self.config.cpu_request}-{self.config.cpu_limit} CPU, {self.config.memory_request}-{self.config.memory_limit} Memory
- **Storage**: {self.config.storage_size}
- **Domain**: {self.config.domain}
- **TLS**: {'Enabled' if self.config.enable_tls else 'Disabled'}

## Quick Start

### 1. Build Docker Image
```bash
./scripts/build.sh
```

### 2. Deploy to Kubernetes
```bash
./scripts/deploy.sh
```

### 3. Check Health
```bash
./scripts/health-check.sh
```

## Directory Structure

```
deployment_v6/
├── docker/                 # Docker configuration
│   ├── Dockerfile         # Production Dockerfile
│   ├── docker-compose.yml # Local development
│   └── .dockerignore      # Docker ignore rules
├── k8s/                   # Kubernetes manifests
│   ├── namespace.yaml     # Namespace
│   ├── configmap.yaml     # Configuration
│   ├── pvc.yaml           # Persistent storage
│   ├── deployment.yaml    # Application deployment
│   ├── service.yaml       # Service definition
│   ├── ingress.yaml       # Ingress configuration
│   └── hpa.yaml           # Horizontal pod autoscaler
├── monitoring/            # Monitoring configuration
│   ├── prometheus-config.yaml
│   └── grafana-dashboard.yaml
├── scripts/              # Deployment scripts
│   ├── build.sh          # Build image
│   ├── deploy.sh         # Deploy to K8s
│   └── health-check.sh   # Health monitoring
└── docs/                 # Documentation
    └── README.md         # This file
```

## Prerequisites

- Docker
- Kubernetes cluster
- kubectl configured
- Helm (for monitoring stack)

## Configuration

Configuration is managed through:
- Environment variables
- ConfigMap (`k8s/configmap.yaml`)
- Deployment manifest (`k8s/deployment.yaml`)

## Monitoring

Monitoring stack includes:
- Prometheus for metrics collection
- Grafana for visualization
- Built-in health checks and readiness probes

Access Grafana dashboard at: https://{self.config.domain}/grafana

## Security

- Non-root container execution
- Resource limits and requests
- Network policies (recommended)
- TLS termination at ingress
- Security contexts applied

## Scaling

Auto-scaling is configured based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Min replicas: {self.config.min_replicas}
- Max replicas: {self.config.max_replicas}

## Troubleshooting

### Common Issues

1. **Pod not starting**: Check resource limits and node capacity
2. **Service unavailable**: Verify ingress and DNS configuration
3. **Performance issues**: Monitor resource usage and scale accordingly

### Useful Commands

```bash
# Check pod logs
kubectl logs -f deployment/tokamak-rl -n {self.config.namespace}

# Scale deployment
kubectl scale deployment tokamak-rl --replicas=5 -n {self.config.namespace}

# Port forward for debugging
kubectl port-forward svc/tokamak-rl-service 8000:80 -n {self.config.namespace}
```

## Support

For support and troubleshooting, refer to:
- Application logs: `kubectl logs`
- Monitoring dashboards: Grafana
- Health checks: `./scripts/health-check.sh`

---

Generated by Tokamak-RL Deployment System v6.0
Timestamp: {datetime.now().isoformat()}
"""
        
        readme_path = f"{self.deployment_dir}/docs/README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        results['files_created'].append(readme_path)

def main():
    """Main deployment preparation function"""
    print("=" * 80)
    print("🚀 PRODUCTION DEPLOYMENT SYSTEM v6.0")
    print("=" * 80)
    print()
    print("Preparing production deployment configuration:")
    print("• 🐳 Docker Container Optimization")
    print("• ☸️  Kubernetes Manifests")
    print("• 📊 Monitoring & Observability")
    print("• 🔒 Security & Resource Management")
    print("• 📈 Auto-scaling Configuration")
    print("• 📚 Documentation & Scripts")
    print("=" * 80)
    print()
    
    try:
        # Production configuration
        config = DeploymentConfig(
            environment="production",
            namespace="tokamak-rl",
            image_tag="v6.0",
            replicas=3,
            domain="tokamak-rl.terragonlabs.io",
            enable_tls=True,
            enable_monitoring=True,
            enable_autoscaling=True
        )
        
        orchestrator = DeploymentOrchestrator(config)
        results = orchestrator.prepare_deployment()
        
        if results['status'] == 'success':
            print("=" * 80)
            print("✅ PRODUCTION DEPLOYMENT PREPARATION COMPLETED")
            print("=" * 80)
            print(f"📁 Files Created: {len(results['files_created'])}")
            print(f"⏱️  Execution Time: {results['execution_time_seconds']:.2f}s")
            print(f"📂 Deployment Directory: {orchestrator.deployment_dir}")
            print()
            print("Next Steps:")
            print("1. Review generated configuration files")
            print("2. Build Docker image: ./deployment_v6/scripts/build.sh")
            print("3. Deploy to Kubernetes: ./deployment_v6/scripts/deploy.sh")
            print("4. Monitor health: ./deployment_v6/scripts/health-check.sh")
            print("=" * 80)
            
            # Save deployment summary
            summary_path = f"{orchestrator.deployment_dir}/deployment_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return results
        else:
            print("=" * 80)
            print("❌ DEPLOYMENT PREPARATION FAILED")
            print("=" * 80)
            for error in results['errors']:
                print(f"❌ {error}")
            return None
            
    except Exception as e:
        logger.critical(f"Deployment preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results and results['status'] == 'success':
        print("\n✅ Production deployment v6.0 preparation successful!")
        sys.exit(0)
    else:
        print("\n❌ Deployment preparation failed")
        sys.exit(1)