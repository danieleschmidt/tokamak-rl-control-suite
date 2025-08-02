# Health Check Endpoints

Health check endpoints for monitoring application availability and readiness.

## Endpoint Specifications

### Liveness Probe - `/health/live`

Basic application liveness check.

```python
from flask import Flask, jsonify
import time

app = Flask(__name__)
start_time = time.time()

@app.route('/health/live')
def liveness():
    """Basic liveness check - application is running"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'uptime_seconds': time.time() - start_time,
        'service': 'tokamak-rl-control-suite'
    }), 200
```

### Readiness Probe - `/health/ready`

Comprehensive readiness check including dependencies.

```python
@app.route('/health/ready')
def readiness():
    """Readiness check - application is ready to serve traffic"""
    checks = {
        'database': check_database_connection(),
        'gpu_availability': check_gpu_resources(),
        'model_loaded': check_rl_models(),
        'physics_solver': check_physics_solver(),
        'safety_systems': check_safety_shields()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return jsonify({
        'status': 'ready' if all_healthy else 'not_ready',
        'timestamp': time.time(),
        'checks': checks,
        'service': 'tokamak-rl-control-suite'
    }), status_code

def check_database_connection():
    """Check database connectivity"""
    try:
        # Implement database ping
        return True
    except:
        return False

def check_gpu_resources():
    """Check GPU availability for RL training"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def check_rl_models():
    """Verify RL models are loaded and functional"""
    try:
        # Check model loading and basic inference
        return True
    except:
        return False

def check_physics_solver():
    """Verify Grad-Shafranov solver functionality"""
    try:
        # Basic solver validation
        return True
    except:
        return False

def check_safety_shields():
    """Verify safety systems are operational"""
    try:
        # Safety system validation
        return True
    except:
        return False
```

### Deep Health Check - `/health/deep`

Extended health check for detailed system status.

```python
@app.route('/health/deep')
def deep_health():
    """Deep health check with detailed metrics"""
    import psutil
    import GPUtil
    
    system_metrics = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'gpu_utilization': get_gpu_utilization(),
        'active_connections': len(psutil.net_connections()),
        'process_count': len(psutil.pids())
    }
    
    rl_metrics = {
        'trained_models': count_trained_models(),
        'training_sessions': get_active_training_sessions(),
        'simulation_environments': get_active_environments(),
        'safety_violations': get_safety_violation_count()
    }
    
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'system_metrics': system_metrics,
        'rl_metrics': rl_metrics,
        'version': get_application_version(),
        'service': 'tokamak-rl-control-suite'
    }), 200

def get_gpu_utilization():
    """Get GPU utilization metrics"""
    try:
        gpus = GPUtil.getGPUs()
        return [{'id': gpu.id, 'load': gpu.load, 'memory': gpu.memoryUtil} for gpu in gpus]
    except:
        return []
```

## Kubernetes Configuration

### Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### Readiness Probe

```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

## Monitoring Integration

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Health check metrics
health_check_requests = Counter('health_check_requests_total', 'Health check requests', ['endpoint', 'status'])
health_check_duration = Histogram('health_check_duration_seconds', 'Health check duration', ['endpoint'])
system_health_score = Gauge('system_health_score', 'Overall system health score (0-1)')

@app.route('/health/live')
def liveness():
    with health_check_duration.labels(endpoint='live').time():
        # ... health check logic
        health_check_requests.labels(endpoint='live', status='success').inc()
        return result
```

### Alert Rules

```yaml
groups:
  - name: tokamak-rl-health
    rules:
      - alert: ApplicationDown
        expr: up{job="tokamak-rl-control"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Tokamak RL Control application is down"
          
      - alert: HighFailureRate
        expr: rate(health_check_requests_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High health check failure rate"
```

## Load Balancer Integration

### NGINX Configuration

```nginx
upstream tokamak_rl {
    server app1:8080 max_fails=3 fail_timeout=30s;
    server app2:8080 max_fails=3 fail_timeout=30s;
}

server {
    location / {
        proxy_pass http://tokamak_rl;
        
        # Health check for upstream
        health_check uri=/health/ready interval=5s fails=3 passes=2;
    }
}
```

## Best Practices

1. **Timeout Configuration**: Set appropriate timeouts for each health check type
2. **Graceful Degradation**: Return partial health information even if some checks fail
3. **Security**: Limit health check endpoints exposure in production
4. **Caching**: Cache expensive health check results for short periods
5. **Logging**: Log health check failures for debugging
6. **Metrics**: Export health check metrics to monitoring systems