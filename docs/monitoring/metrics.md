# Prometheus Metrics Configuration

Comprehensive metrics configuration for monitoring tokamak-rl-control-suite performance and behavior.

## Metrics Overview

The system exposes metrics in three categories:
- **System Metrics**: Infrastructure and resource utilization
- **Application Metrics**: RL training and plasma control performance  
- **Business Metrics**: Domain-specific KPIs for fusion research

## Basic Setup

### Prometheus Client Configuration

```python
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, make_wsgi_app, start_http_server
)
import time
from typing import Dict, Any

# Create custom registry
REGISTRY = CollectorRegistry()

# System Metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=REGISTRY
)

system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes',
    ['type'],  # total, available, used
    registry=REGISTRY
)

system_gpu_usage = Gauge(
    'system_gpu_usage_percent',
    'GPU utilization percentage',
    ['gpu_id'],
    registry=REGISTRY
)

system_gpu_memory = Gauge(
    'system_gpu_memory_bytes',
    'GPU memory usage in bytes',
    ['gpu_id', 'type'],  # total, used, free
    registry=REGISTRY
)

# Application Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

application_errors = Counter(
    'application_errors_total',
    'Total application errors',
    ['error_type', 'component'],
    registry=REGISTRY
)
```

### RL Training Metrics

```python
# Training Metrics
training_episodes = Counter(
    'rl_training_episodes_total',
    'Total training episodes completed',
    ['model_name', 'environment'],
    registry=REGISTRY
)

training_episode_reward = Histogram(
    'rl_training_episode_reward',
    'Episode reward distribution',
    ['model_name', 'environment'],
    buckets=[-100, -50, -10, -1, 0, 1, 10, 50, 100, 500],
    registry=REGISTRY
)

training_episode_length = Histogram(
    'rl_training_episode_length_steps',
    'Episode length in steps',
    ['model_name', 'environment'],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000],
    registry=REGISTRY
)

model_loss = Gauge(
    'rl_model_loss',
    'Current model loss',
    ['model_name', 'loss_type'],  # actor, critic, total
    registry=REGISTRY
)

model_learning_rate = Gauge(
    'rl_model_learning_rate',
    'Current learning rate',
    ['model_name', 'component'],  # actor, critic
    registry=REGISTRY
)

training_fps = Gauge(
    'rl_training_fps',
    'Training frames per second',
    ['model_name'],
    registry=REGISTRY
)
```

### Plasma Control Metrics

```python
# Plasma Physics Metrics
plasma_current = Gauge(
    'plasma_current_amperes',
    'Plasma current in amperes',
    ['tokamak_config'],
    registry=REGISTRY
)

plasma_beta_normalized = Gauge(
    'plasma_beta_normalized',
    'Normalized beta (plasma pressure / magnetic pressure)',
    ['tokamak_config'],
    registry=REGISTRY
)

plasma_q_min = Gauge(
    'plasma_q_min',
    'Minimum safety factor',
    ['tokamak_config'],
    registry=REGISTRY
)

plasma_shape_error = Gauge(
    'plasma_shape_error_cm',
    'Plasma shape error in centimeters',
    ['tokamak_config', 'metric'],  # rms, max, mean
    registry=REGISTRY
)

plasma_confinement_time = Gauge(
    'plasma_confinement_time_seconds',
    'Energy confinement time',
    ['tokamak_config'],
    registry=REGISTRY
)

control_power = Gauge(
    'control_power_watts',
    'Control system power consumption',
    ['system'],  # PF_coils, heating, pumping
    registry=REGISTRY
)

# Safety Metrics
disruptions_total = Counter(
    'plasma_disruptions_total',
    'Total plasma disruptions',
    ['cause', 'severity'],
    registry=REGISTRY
)

safety_violations = Counter(
    'safety_violations_total',
    'Safety constraint violations',
    ['constraint_type', 'severity'],
    registry=REGISTRY
)

safety_margin = Gauge(
    'safety_margin_percent',
    'Safety margin as percentage of limit',
    ['constraint_type'],  # q_min, density, beta
    registry=REGISTRY
)
```

### Performance Metrics

```python
# Simulation Performance
simulation_step_duration = Histogram(
    'simulation_step_duration_seconds',
    'Physics simulation step duration',
    ['solver_type'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=REGISTRY
)

inference_duration = Histogram(
    'rl_inference_duration_seconds',
    'RL model inference duration',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    registry=REGISTRY
)

memory_pool_usage = Gauge(
    'memory_pool_usage_bytes',
    'Memory pool usage for simulations',
    ['pool_type'],  # gpu, cpu, shared
    registry=REGISTRY
)

active_environments = Gauge(
    'active_simulation_environments',
    'Number of active simulation environments',
    registry=REGISTRY
)
```

## Metrics Collection

### Automatic System Metrics

```python
import psutil
import GPUtil
import threading
import time

class SystemMetricsCollector:
    """Collect system-level metrics"""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.running = False
        self.thread = None
    
    def start(self):
        """Start metrics collection"""
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collect_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                self._collect_cpu_metrics()
                self._collect_memory_metrics()
                self._collect_gpu_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                application_errors.labels(
                    error_type='metrics_collection',
                    component='system'
                ).inc()
    
    def _collect_cpu_metrics(self):
        """Collect CPU metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_usage.set(cpu_percent)
    
    def _collect_memory_metrics(self):
        """Collect memory metrics"""
        memory = psutil.virtual_memory()
        system_memory_usage.labels(type='total').set(memory.total)
        system_memory_usage.labels(type='available').set(memory.available)
        system_memory_usage.labels(type='used').set(memory.used)
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics"""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                system_gpu_usage.labels(gpu_id=str(gpu.id)).set(gpu.load * 100)
                system_gpu_memory.labels(
                    gpu_id=str(gpu.id), type='total'
                ).set(gpu.memoryTotal * 1024 * 1024)
                system_gpu_memory.labels(
                    gpu_id=str(gpu.id), type='used'
                ).set(gpu.memoryUsed * 1024 * 1024)
        except Exception:
            pass  # GPU metrics optional

# Start system metrics collection
system_collector = SystemMetricsCollector()
system_collector.start()
```

### RL Training Metrics Integration

```python
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    """Callback to collect RL training metrics"""
    
    def __init__(self, model_name: str, environment: str):
        super().__init__()
        self.model_name = model_name
        self.environment = environment
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called at each training step"""
        # Update FPS
        if hasattr(self.locals.get('infos', [{}])[0], 'fps'):
            fps = self.locals['infos'][0]['fps']
            training_fps.labels(model_name=self.model_name).set(fps)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of rollout"""
        # Update episode metrics
        if 'episode' in self.locals.get('infos', [{}])[0]:
            episode_info = self.locals['infos'][0]['episode']
            
            training_episodes.labels(
                model_name=self.model_name,
                environment=self.environment
            ).inc()
            
            training_episode_reward.labels(
                model_name=self.model_name,
                environment=self.environment
            ).observe(episode_info['r'])
            
            training_episode_length.labels(
                model_name=self.model_name,
                environment=self.environment
            ).observe(episode_info['l'])
        
        # Update loss metrics if available
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            for key, value in self.model.logger.name_to_value.items():
                if 'loss' in key.lower():
                    loss_type = key.replace('train/', '').replace('_loss', '')
                    model_loss.labels(
                        model_name=self.model_name,
                        loss_type=loss_type
                    ).set(value)

# Usage in training
from stable_baselines3 import SAC

model = SAC("MlpPolicy", env)
metrics_callback = MetricsCallback(model_name="SAC", environment="ITER")
model.learn(total_timesteps=100000, callback=metrics_callback)
```

### Custom Plasma Metrics

```python
class PlasmaMetricsCollector:
    """Collect plasma-specific metrics"""
    
    def __init__(self, tokamak_config: str):
        self.tokamak_config = tokamak_config
    
    def update_plasma_state(self, state: Dict[str, Any]):
        """Update plasma state metrics"""
        
        # Core plasma parameters
        if 'plasma_current' in state:
            plasma_current.labels(
                tokamak_config=self.tokamak_config
            ).set(state['plasma_current'])
        
        if 'beta_normalized' in state:
            plasma_beta_normalized.labels(
                tokamak_config=self.tokamak_config
            ).set(state['beta_normalized'])
        
        if 'q_profile' in state:
            q_min_value = min(state['q_profile'])
            plasma_q_min.labels(
                tokamak_config=self.tokamak_config
            ).set(q_min_value)
        
        # Shape error metrics
        if 'shape_error' in state:
            plasma_shape_error.labels(
                tokamak_config=self.tokamak_config,
                metric='current'
            ).set(state['shape_error'])
        
        # Confinement time
        if 'confinement_time' in state:
            plasma_confinement_time.labels(
                tokamak_config=self.tokamak_config
            ).set(state['confinement_time'])
    
    def update_control_metrics(self, control_state: Dict[str, Any]):
        """Update control system metrics"""
        
        if 'pf_coil_power' in control_state:
            control_power.labels(system='PF_coils').set(
                control_state['pf_coil_power']
            )
        
        if 'heating_power' in control_state:
            control_power.labels(system='heating').set(
                control_state['heating_power']
            )
    
    def record_safety_event(self, event_type: str, severity: str):
        """Record safety events"""
        
        if event_type == 'disruption':
            disruptions_total.labels(
                cause='unknown',  # Could be enhanced with cause detection
                severity=severity
            ).inc()
        else:
            safety_violations.labels(
                constraint_type=event_type,
                severity=severity
            ).inc()
    
    def update_safety_margins(self, margins: Dict[str, float]):
        """Update safety margin metrics"""
        
        for constraint_type, margin_percent in margins.items():
            safety_margin.labels(
                constraint_type=constraint_type
            ).set(margin_percent)

# Usage in simulation
plasma_metrics = PlasmaMetricsCollector(tokamak_config="ITER")

# During simulation loop
def simulation_step(state, action):
    start_time = time.time()
    
    # Physics simulation
    new_state, reward, done, info = env.step(action)
    
    # Update metrics
    plasma_metrics.update_plasma_state(new_state)
    plasma_metrics.update_control_metrics(info.get('control', {}))
    
    # Record performance
    duration = time.time() - start_time
    simulation_step_duration.labels(
        solver_type='grad_shafranov'
    ).observe(duration)
    
    return new_state, reward, done, info
```

## Metrics Endpoint

### Flask Integration

```python
from flask import Flask
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

# Add prometheus wsgi middleware
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app(REGISTRY)
})

@app.before_request
def before_request():
    """Record request metrics"""
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Record response metrics"""
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        
        http_requests_total.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown',
            status=str(response.status_code)
        ).inc()
        
        http_request_duration.labels(
            method=request.method,
            endpoint=request.endpoint or 'unknown'
        ).observe(duration)
    
    return response
```

## Alert Rules

### Prometheus Alert Configuration

```yaml
# prometheus-alerts.yml
groups:
  - name: tokamak-rl-system
    rules:
      - alert: HighCPUUsage
        expr: system_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}% for 5 minutes"
      
      - alert: HighMemoryUsage
        expr: (system_memory_usage_bytes{type="used"} / system_memory_usage_bytes{type="total"}) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
      
      - alert: GPUMemoryExhaustion
        expr: (system_gpu_memory_bytes{type="used"} / system_gpu_memory_bytes{type="total"}) * 100 > 90
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory nearly exhausted"

  - name: tokamak-rl-plasma
    rules:
      - alert: LowSafetyFactor
        expr: plasma_q_min < 1.5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Low safety factor detected"
          description: "q_min = {{ $value }} is below safe threshold"
      
      - alert: HighShapeError
        expr: plasma_shape_error_cm{metric="current"} > 5.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High plasma shape error"
      
      - alert: DisruptionRate
        expr: rate(plasma_disruptions_total[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High disruption rate detected"
      
      - alert: TrainingStalled
        expr: rate(rl_training_episodes_total[10m]) == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RL training appears stalled"

  - name: tokamak-rl-performance
    rules:
      - alert: SlowInference
        expr: histogram_quantile(0.95, rl_inference_duration_seconds_bucket) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow RL model inference"
      
      - alert: SlowSimulation
        expr: histogram_quantile(0.95, simulation_step_duration_seconds_bucket) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow physics simulation"
```

## Grafana Dashboards

### Main Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Tokamak RL Control Suite",
    "panels": [
      {
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "system_cpu_usage_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "(system_memory_usage_bytes{type=\"used\"} / system_memory_usage_bytes{type=\"total\"}) * 100",
            "legendFormat": "Memory %"
          }
        ]
      },
      {
        "title": "Plasma Safety Metrics",
        "type": "timeseries",
        "targets": [
          {
            "expr": "plasma_q_min",
            "legendFormat": "q_min"
          },
          {
            "expr": "plasma_beta_normalized",
            "legendFormat": "β_N"
          }
        ]
      },
      {
        "title": "Training Progress",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(rl_training_episodes_total[5m])",
            "legendFormat": "Episodes/sec"
          },
          {
            "expr": "training_fps",
            "legendFormat": "Training FPS"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

1. **Metric Naming**: Use descriptive names with units (e.g., `_seconds`, `_bytes`, `_percent`)
2. **Labels**: Use consistent label names across metrics
3. **Cardinality**: Avoid high-cardinality labels (>1000 unique values)
4. **Collection Interval**: Balance granularity with storage requirements
5. **Retention**: Configure appropriate retention policies
6. **Alerting**: Set up alerts for critical system and domain metrics
7. **Documentation**: Document metric meanings and expected ranges
8. **Performance**: Use histograms for timing metrics, counters for events