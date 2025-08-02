# Observability Best Practices

Comprehensive guide for implementing effective observability in the tokamak-rl-control-suite.

## Observability Pillars

### 1. Metrics (RED/USE Methods)
- **Rate**: Request/operation rate
- **Errors**: Error rate and types
- **Duration**: Latency/response time
- **Utilization**: Resource usage
- **Saturation**: Resource capacity
- **Errors**: Error rates and types

### 2. Logging (Structured)
- **Structured format**: JSON with consistent schema
- **Correlation**: Request/session tracking
- **Context**: Rich metadata for debugging
- **Security**: No sensitive data exposure

### 3. Tracing (Distributed)
- **Request flow**: End-to-end transaction tracking
- **Performance**: Identify bottlenecks
- **Dependencies**: Service interaction mapping
- **Error correlation**: Link errors across services

## Implementation Strategy

### Metrics Strategy

```python
# metrics_strategy.py
from prometheus_client import Counter, Histogram, Gauge, Summary
from functools import wraps
import time

class MetricsStrategy:
    """Centralized metrics strategy implementation"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.setup_base_metrics()
    
    def setup_base_metrics(self):
        """Setup fundamental metrics for any service"""
        
        # RED Metrics
        self.request_total = Counter(
            'requests_total',
            'Total requests',
            ['method', 'endpoint', 'status'],
            registry=None
        )
        
        self.request_duration = Histogram(
            'request_duration_seconds',
            'Request duration',
            ['method', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.error_total = Counter(
            'errors_total',
            'Total errors',
            ['type', 'component']
        )
        
        # USE Metrics  
        self.cpu_utilization = Gauge(
            'cpu_utilization_percent',
            'CPU utilization percentage'
        )
        
        self.memory_utilization = Gauge(
            'memory_utilization_percent', 
            'Memory utilization percentage'
        )
        
        self.connection_pool_saturation = Gauge(
            'connection_pool_saturation_percent',
            'Connection pool saturation',
            ['pool_name']
        )
    
    def track_request(self, method: str, endpoint: str):
        """Decorator for tracking HTTP requests"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    self.error_total.labels(
                        type=type(e).__name__,
                        component=func.__module__
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    
                    self.request_total.labels(
                        method=method,
                        endpoint=endpoint,
                        status=status
                    ).inc()
                    
                    self.request_duration.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
            
            return wrapper
        return decorator
    
    def track_function_performance(self, component: str):
        """Decorator for tracking function performance"""
        def decorator(func):
            function_duration = Histogram(
                f'{component}_function_duration_seconds',
                f'{component} function duration',
                ['function_name']
            )
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                with function_duration.labels(function_name=func.__name__).time():
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator

# Usage example
metrics = MetricsStrategy("tokamak-rl-control")

@metrics.track_request("POST", "/train")
def train_model(config):
    # Training logic
    pass

@metrics.track_function_performance("rl_training")
def compute_gradients():
    # Gradient computation
    pass
```

### Logging Strategy

```python
# logging_strategy.py
import logging
import json
import time
import uuid
import contextvars
from typing import Dict, Any, Optional

# Context variables for request tracking
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id')
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('user_id')
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('session_id')

class StructuredLogger:
    """Advanced structured logging with context"""
    
    def __init__(self, name: str, service: str):
        self.logger = logging.getLogger(name)
        self.service = service
        self.setup_formatter()
    
    def setup_formatter(self):
        """Setup structured JSON formatter"""
        formatter = ContextualFormatter(service=self.service)
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, level: str, event_type: str, message: str, **kwargs):
        """Log structured event with context"""
        
        extra = {
            'event_type': event_type,
            'service': self.service,
            'timestamp': time.time(),
            **kwargs
        }
        
        # Add request context if available
        try:
            extra['request_id'] = request_id_var.get()
        except LookupError:
            pass
        
        try:
            extra['user_id'] = user_id_var.get()
        except LookupError:
            pass
        
        try:
            extra['session_id'] = session_id_var.get()
        except LookupError:
            pass
        
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        self.log_event(
            'info',
            'performance',
            f'Operation {operation} completed',
            operation=operation,
            duration_seconds=duration,
            **metrics
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.log_event(
            'error',
            'error',
            f'Error occurred: {str(error)}',
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events"""
        self.log_event(
            'warning',
            'security',
            f'Security event: {event_type}',
            security_event_type=event_type,
            **details
        )

class ContextualFormatter(logging.Formatter):
    """Formatter that includes contextual information"""
    
    def __init__(self, service: str):
        super().__init__()
        self.service = service
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': self.service,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields from record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process']:
                    log_entry[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str, separators=(',', ':'))

# Usage example
logger = StructuredLogger("tokamak.training", "tokamak-rl-control")

def train_model_with_logging():
    request_id_var.set(str(uuid.uuid4()))
    
    logger.log_event("info", "training_start", "Starting model training", 
                     model_name="SAC", environment="ITER")
    
    start_time = time.time()
    try:
        # Training logic here
        result = {"loss": 0.5, "episodes": 1000}
        
        logger.log_performance("model_training", time.time() - start_time,
                              final_loss=result["loss"],
                              episodes_completed=result["episodes"])
        
        return result
        
    except Exception as e:
        logger.log_error(e, {"model_name": "SAC", "environment": "ITER"})
        raise
```

### Tracing Strategy

```python
# tracing_strategy.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import time
from functools import wraps

class TracingStrategy:
    """Distributed tracing implementation"""
    
    def __init__(self, service_name: str, jaeger_endpoint: str):
        self.service_name = service_name
        self.setup_tracing(jaeger_endpoint)
    
    def setup_tracing(self, jaeger_endpoint: str):
        """Setup OpenTelemetry tracing"""
        
        # Configure tracer provider
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=14268,
            collector_endpoint=jaeger_endpoint,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(self.service_name)
        
        # Auto-instrument Flask and requests
        FlaskInstrumentor().instrument()
        RequestsInstrumentor().instrument()
    
    def trace_function(self, span_name: str = None, attributes: dict = None):
        """Decorator for tracing functions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = span_name or f"{func.__module__}.{func.__name__}"
                
                with self.tracer.start_as_current_span(name) as span:
                    # Add function attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("function.result", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("function.result", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        span.record_exception(e)
                        raise
            
            return wrapper
        return decorator
    
    def trace_rl_training(self, model_name: str, environment: str):
        """Specialized tracing for RL training"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span("rl_training") as span:
                    span.set_attribute("rl.model_name", model_name)
                    span.set_attribute("rl.environment", environment)
                    span.set_attribute("rl.operation", func.__name__)
                    
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        
                        # Add training metrics to span
                        if isinstance(result, dict):
                            for key, value in result.items():
                                if isinstance(value, (int, float)):
                                    span.set_attribute(f"rl.{key}", value)
                        
                        return result
                    finally:
                        span.set_attribute("rl.duration_seconds", 
                                         time.time() - start_time)
            
            return wrapper
        return decorator
    
    def trace_plasma_simulation(self, tokamak_config: str):
        """Specialized tracing for plasma simulation"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span("plasma_simulation") as span:
                    span.set_attribute("plasma.tokamak_config", tokamak_config)
                    span.set_attribute("plasma.operation", func.__name__)
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # Add plasma metrics if available
                        if hasattr(result, 'plasma_current'):
                            span.set_attribute("plasma.current", result.plasma_current)
                        if hasattr(result, 'q_min'):
                            span.set_attribute("plasma.q_min", result.q_min)
                        if hasattr(result, 'beta_normalized'):
                            span.set_attribute("plasma.beta_n", result.beta_normalized)
                        
                        return result
                    except Exception as e:
                        span.set_attribute("plasma.error", str(e))
                        raise
            
            return wrapper
        return decorator

# Usage example
tracing = TracingStrategy("tokamak-rl-control", "http://localhost:14268/api/traces")

@tracing.trace_rl_training("SAC", "ITER")
def train_sac_model():
    # Training implementation
    return {"loss": 0.5, "episodes": 1000}

@tracing.trace_plasma_simulation("ITER")
def simulate_plasma_step():
    # Simulation implementation
    pass
```

## SLI/SLO Definition

### Service Level Indicators (SLIs)

```python
# sli_slo_strategy.py
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class SLI:
    """Service Level Indicator definition"""
    name: str
    description: str
    query: str
    unit: str
    good_threshold: float
    measurement_window: str

@dataclass
class SLO:
    """Service Level Objective definition"""
    name: str
    sli: SLI
    target_percentage: float
    measurement_period: str
    alerting_burn_rate: float

class SLIStrategy:
    """SLI/SLO implementation for tokamak control system"""
    
    def __init__(self):
        self.slis = self.define_slis()
        self.slos = self.define_slos()
    
    def define_slis(self) -> Dict[str, SLI]:
        """Define SLIs for the system"""
        
        return {
            "availability": SLI(
                name="availability",
                description="System availability",
                query="avg_over_time(up[5m])",
                unit="ratio",
                good_threshold=0.99,
                measurement_window="5m"
            ),
            
            "inference_latency": SLI(
                name="inference_latency",
                description="RL model inference latency",
                query="histogram_quantile(0.95, rl_inference_duration_seconds_bucket)",
                unit="seconds",
                good_threshold=0.01,  # 10ms
                measurement_window="5m"
            ),
            
            "plasma_safety": SLI(
                name="plasma_safety",
                description="Plasma safety factor compliance",
                query="avg_over_time(plasma_q_min[1m])",
                unit="safety_factor",
                good_threshold=1.5,
                measurement_window="1m"
            ),
            
            "shape_accuracy": SLI(
                name="shape_accuracy",
                description="Plasma shape control accuracy",
                query="avg_over_time(plasma_shape_error_cm[5m])",
                unit="centimeters",
                good_threshold=5.0,
                measurement_window="5m"
            ),
            
            "training_success_rate": SLI(
                name="training_success_rate", 
                description="RL training episode success rate",
                query="rate(rl_training_episodes_total{status=\"success\"}[1h]) / rate(rl_training_episodes_total[1h])",
                unit="ratio",
                good_threshold=0.95,
                measurement_window="1h"
            )
        }
    
    def define_slos(self) -> Dict[str, SLO]:
        """Define SLOs based on SLIs"""
        
        return {
            "system_availability": SLO(
                name="system_availability",
                sli=self.slis["availability"],
                target_percentage=99.9,
                measurement_period="30d",
                alerting_burn_rate=10.0
            ),
            
            "inference_performance": SLO(
                name="inference_performance",
                sli=self.slis["inference_latency"],
                target_percentage=95.0,
                measurement_period="7d",
                alerting_burn_rate=5.0
            ),
            
            "plasma_safety_compliance": SLO(
                name="plasma_safety_compliance",
                sli=self.slis["plasma_safety"],
                target_percentage=99.99,
                measurement_period="24h",
                alerting_burn_rate=2.0
            ),
            
            "control_accuracy": SLO(
                name="control_accuracy",
                sli=self.slis["shape_accuracy"],
                target_percentage=90.0,
                measurement_period="7d",
                alerting_burn_rate=3.0
            )
        }
    
    def generate_slo_alerts(self) -> List[Dict]:
        """Generate Prometheus alert rules for SLOs"""
        
        alerts = []
        
        for slo_name, slo in self.slos.items():
            # Fast burn rate (immediate attention)
            fast_burn_alert = {
                "alert": f"SLO_{slo_name}_FastBurn",
                "expr": f"({slo.sli.query} > {slo.sli.good_threshold}) and (rate(slo_errors[1m]) > {slo.alerting_burn_rate})",
                "for": "2m",
                "labels": {
                    "severity": "critical",
                    "slo": slo_name,
                    "burn_rate": "fast"
                },
                "annotations": {
                    "summary": f"SLO {slo_name} is burning error budget rapidly",
                    "description": f"Fast burn rate detected for {slo_name}"
                }
            }
            
            # Slow burn rate (warning)
            slow_burn_alert = {
                "alert": f"SLO_{slo_name}_SlowBurn",
                "expr": f"({slo.sli.query} > {slo.sli.good_threshold}) and (rate(slo_errors[1h]) > {slo.alerting_burn_rate / 10})",
                "for": "15m",
                "labels": {
                    "severity": "warning",
                    "slo": slo_name,
                    "burn_rate": "slow"
                },
                "annotations": {
                    "summary": f"SLO {slo_name} is burning error budget slowly",
                    "description": f"Slow burn rate detected for {slo_name}"
                }
            }
            
            alerts.extend([fast_burn_alert, slow_burn_alert])
        
        return alerts

# Usage
sli_strategy = SLIStrategy()
slo_alerts = sli_strategy.generate_slo_alerts()
```

## Observability Dashboard Strategy

### Grafana Dashboard Templates

```python
# dashboard_strategy.py
import json
from typing import Dict, List

class DashboardStrategy:
    """Strategy for creating observability dashboards"""
    
    def create_system_overview_dashboard(self) -> Dict:
        """Create system overview dashboard"""
        
        return {
            "dashboard": {
                "title": "Tokamak RL Control - System Overview",
                "tags": ["tokamak", "system", "overview"],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s",
                "panels": [
                    self.create_slo_panel(),
                    self.create_golden_signals_panel(),
                    self.create_system_health_panel(),
                    self.create_error_rate_panel()
                ]
            }
        }
    
    def create_plasma_control_dashboard(self) -> Dict:
        """Create plasma control specific dashboard"""
        
        return {
            "dashboard": {
                "title": "Tokamak RL Control - Plasma Control",
                "tags": ["tokamak", "plasma", "control"],
                "panels": [
                    self.create_plasma_safety_panel(),
                    self.create_shape_control_panel(),
                    self.create_safety_margins_panel(),
                    self.create_control_power_panel()
                ]
            }
        }
    
    def create_rl_training_dashboard(self) -> Dict:
        """Create RL training dashboard"""
        
        return {
            "dashboard": {
                "title": "Tokamak RL Control - Training",
                "tags": ["tokamak", "rl", "training"],
                "panels": [
                    self.create_training_progress_panel(),
                    self.create_model_performance_panel(),
                    self.create_training_efficiency_panel(),
                    self.create_gpu_utilization_panel()
                ]
            }
        }
    
    def create_slo_panel(self) -> Dict:
        """Create SLO compliance panel"""
        return {
            "title": "SLO Compliance",
            "type": "stat",
            "targets": [
                {
                    "expr": "avg_over_time(up[30d]) * 100",
                    "legendFormat": "Availability %"
                },
                {
                    "expr": "(1 - rate(rl_inference_duration_seconds_bucket{le=\"0.01\"}[7d])) * 100",
                    "legendFormat": "Inference Latency SLO %"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 95},
                            {"color": "green", "value": 99}
                        ]
                    }
                }
            }
        }
    
    def create_golden_signals_panel(self) -> Dict:
        """Create golden signals panel (Rate, Errors, Duration)"""
        return {
            "title": "Golden Signals",
            "type": "timeseries",
            "targets": [
                {
                    "expr": "sum(rate(http_requests_total[5m]))",
                    "legendFormat": "Request Rate"
                },
                {
                    "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100",
                    "legendFormat": "Error Rate %"
                },
                {
                    "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
                    "legendFormat": "95th Percentile Latency"
                }
            ]
        }

# Usage
dashboard_strategy = DashboardStrategy()
system_dashboard = dashboard_strategy.create_system_overview_dashboard()
```

## Best Practices Summary

### 1. Metrics Best Practices
- **Use consistent naming conventions** (e.g., `_total` for counters, `_seconds` for durations)
- **Include units in metric names** where appropriate
- **Avoid high cardinality labels** (>1000 unique values)
- **Use histograms for timing metrics** instead of averages
- **Export metrics in Prometheus format** for standardization

### 2. Logging Best Practices
- **Use structured logging** with consistent schema
- **Include correlation IDs** for request tracing
- **Log at appropriate levels** (DEBUG, INFO, WARN, ERROR, FATAL)
- **Never log sensitive information** (passwords, tokens, PII)
- **Use sampling** for high-volume logs
- **Include context** (user ID, session ID, operation ID)

### 3. Tracing Best Practices
- **Trace critical user journeys** end-to-end
- **Include relevant attributes** in spans
- **Use consistent span naming** conventions
- **Minimize trace sampling overhead** in production
- **Correlate traces with logs** and metrics
- **Monitor trace sampling rates** and coverage

### 4. Alerting Best Practices
- **Alert on symptoms**, not causes
- **Make alerts actionable** with clear resolution steps
- **Use multi-window alerting** to reduce false positives
- **Implement proper escalation** procedures
- **Test alert configurations** regularly
- **Avoid alert fatigue** through proper grouping and suppression

### 5. Dashboard Best Practices
- **Follow the 5-second rule** (key information visible in 5 seconds)
- **Use consistent color schemes** and layouts
- **Include context** (time ranges, filters, annotations)
- **Create role-specific dashboards** (operator, developer, executive)
- **Implement drill-down capabilities** for investigation
- **Regular dashboard reviews** and updates

### 6. SLI/SLO Best Practices
- **Choose user-centric SLIs** that reflect actual experience
- **Set realistic SLO targets** based on business requirements
- **Monitor error budgets** and burn rates
- **Use SLOs for prioritization** and decision-making
- **Regular SLO reviews** and adjustments
- **Document SLI definitions** and measurement methods

### 7. Security and Privacy
- **Sanitize logs** to remove sensitive data
- **Use secure channels** for metrics export
- **Implement access controls** for observability tools
- **Regular security reviews** of observability data
- **Comply with data retention** policies
- **Monitor observability infrastructure** itself

### 8. Performance and Cost
- **Monitor observability overhead** (CPU, memory, network)
- **Use appropriate retention** policies for different data types
- **Implement efficient aggregation** for long-term storage
- **Regular capacity planning** for observability infrastructure
- **Cost monitoring** for cloud-based solutions
- **Optimize query performance** in dashboards and alerts