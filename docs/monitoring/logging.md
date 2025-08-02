# Structured Logging Configuration

Structured logging setup for the tokamak-rl-control-suite with JSON formatting and correlation tracking.

## Configuration Overview

The logging configuration provides:
- JSON-formatted structured logs
- Correlation ID tracking across requests
- Log level management
- Performance metrics integration
- Security-conscious log handling

## Python Logging Configuration

### Basic Setup

```python
import logging
import json
import time
import uuid
from typing import Dict, Any

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'service': 'tokamak-rl-control-suite'
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
            
        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup structured logging configuration"""
    
    # Create formatter
    formatter = StructuredFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Root logger configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format='%(message)s'  # Formatter handles the actual format
    )
```

### Correlation ID Middleware

```python
import contextvars
from flask import Flask, request, g

# Context variable for correlation ID
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id')

class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        try:
            record.correlation_id = correlation_id_var.get()
        except LookupError:
            record.correlation_id = None
        return True

def setup_correlation_middleware(app: Flask):
    """Setup correlation ID middleware for Flask"""
    
    @app.before_request
    def before_request():
        # Get or generate correlation ID
        correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
        correlation_id_var.set(correlation_id)
        g.correlation_id = correlation_id
    
    @app.after_request
    def after_request(response):
        # Add correlation ID to response headers
        if hasattr(g, 'correlation_id'):
            response.headers['X-Correlation-ID'] = g.correlation_id
        return response

# Add filter to all loggers
logging.getLogger().addFilter(CorrelationIdFilter())
```

### Application-Specific Loggers

```python
class TokamakLogger:
    """Specialized logger for tokamak RL operations"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"tokamak_rl.{name}")
    
    def log_training_event(self, event_type: str, model_name: str, **kwargs):
        """Log training-related events"""
        extra_fields = {
            'event_type': 'training',
            'model_name': model_name,
            'training_event': event_type,
            **kwargs
        }
        
        self.logger.info(
            f"Training event: {event_type} for model {model_name}",
            extra={'extra_fields': extra_fields}
        )
    
    def log_plasma_state(self, plasma_params: Dict[str, Any], safety_status: str):
        """Log plasma state information"""
        extra_fields = {
            'event_type': 'plasma_state',
            'safety_status': safety_status,
            'plasma_current': plasma_params.get('current'),
            'beta_normalized': plasma_params.get('beta_n'),
            'q_min': plasma_params.get('q_min'),
            'shape_error': plasma_params.get('shape_error')
        }
        
        self.logger.info(
            f"Plasma state: safety={safety_status}",
            extra={'extra_fields': extra_fields}
        )
    
    def log_safety_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log safety-related events"""
        extra_fields = {
            'event_type': 'safety',
            'safety_event': event_type,
            'severity': severity,
            **details
        }
        
        level = logging.WARNING if severity == 'high' else logging.INFO
        self.logger.log(
            level,
            f"Safety event: {event_type} (severity: {severity})",
            extra={'extra_fields': extra_fields}
        )
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics"""
        extra_fields = {
            'event_type': 'performance',
            'metrics': metrics
        }
        
        self.logger.info(
            "Performance metrics recorded",
            extra={'extra_fields': extra_fields}
        )

# Usage examples
training_logger = TokamakLogger('training')
safety_logger = TokamakLogger('safety')
performance_logger = TokamakLogger('performance')
```

## Environment-Specific Configuration

### Development Configuration

```yaml
# logging-dev.yaml
version: 1
formatters:
  structured:
    class: __main__.StructuredFormatter
    
handlers:
  console:
    class: logging.StreamHandler
    formatter: structured
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: structured
    filename: logs/tokamak-rl-dev.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  tokamak_rl:
    level: DEBUG
    handlers: [console, file]
    propagate: false
    
root:
  level: INFO
  handlers: [console]
```

### Production Configuration

```yaml
# logging-prod.yaml
version: 1
formatters:
  structured:
    class: __main__.StructuredFormatter
    
handlers:
  console:
    class: logging.StreamHandler
    formatter: structured
    stream: ext://sys.stdout
    
  syslog:
    class: logging.handlers.SysLogHandler
    formatter: structured
    address: ['localhost', 514]
    facility: local0

loggers:
  tokamak_rl:
    level: INFO
    handlers: [console, syslog]
    propagate: false
    
  tokamak_rl.safety:
    level: WARNING
    handlers: [console, syslog]
    propagate: false
    
root:
  level: WARNING
  handlers: [console]
```

## Log Aggregation & Analysis

### ELK Stack Integration

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/tokamak-rl/*.log
  json.keys_under_root: true
  json.add_error_key: true
  fields:
    service: tokamak-rl-control-suite
    environment: production
    
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "tokamak-rl-logs-%{+yyyy.MM.dd}"
  
processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
```

### Logstash Pipeline

```ruby
# logstash-tokamak.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [service] == "tokamak-rl-control-suite" {
    # Parse timestamp
    date {
      match => [ "timestamp", "UNIX" ]
    }
    
    # Extract training metrics
    if [event_type] == "training" {
      mutate {
        add_tag => [ "training" ]
      }
    }
    
    # Flag safety events
    if [event_type] == "safety" and [severity] == "high" {
      mutate {
        add_tag => [ "critical_safety" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "tokamak-rl-logs-%{+YYYY.MM.dd}"
  }
}
```

### Kibana Dashboards

```json
{
  "dashboard": {
    "title": "Tokamak RL Control - System Overview",
    "panels": [
      {
        "title": "Log Volume by Level",
        "type": "histogram",
        "query": "service:tokamak-rl-control-suite"
      },
      {
        "title": "Safety Events Timeline",
        "type": "line",
        "query": "event_type:safety AND severity:high"
      },
      {
        "title": "Training Progress",
        "type": "table",
        "query": "event_type:training"
      }
    ]
  }
}
```

## Security Considerations

### Sensitive Data Handling

```python
import re
from typing import Any

class SecurityFilter(logging.Filter):
    """Filter sensitive information from logs"""
    
    SENSITIVE_PATTERNS = [
        r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
        r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
        r'api_key["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
        r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize log message
        record.msg = self.sanitize_message(str(record.msg))
        
        # Sanitize extra fields if present
        if hasattr(record, 'extra_fields'):
            record.extra_fields = self.sanitize_dict(record.extra_fields)
        
        return True
    
    def sanitize_message(self, message: str) -> str:
        """Remove sensitive data from log message"""
        for pattern in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, r'\1=***REDACTED***', message, flags=re.IGNORECASE)
        return message
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from dictionary"""
        sensitive_keys = ['password', 'token', 'api_key', 'secret', 'auth']
        
        for key in data:
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                data[key] = '***REDACTED***'
        
        return data

# Add security filter to all handlers
security_filter = SecurityFilter()
for handler in logging.getLogger().handlers:
    handler.addFilter(security_filter)
```

## Performance Monitoring

### Log Performance Metrics

```python
import time
from functools import wraps

def log_performance(logger: logging.Logger):
    """Decorator to log function performance"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        'extra_fields': {
                            'event_type': 'performance',
                            'function': func.__name__,
                            'duration_seconds': duration,
                            'status': 'success'
                        }
                    }
                )
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Function {func.__name__} failed",
                    extra={
                        'extra_fields': {
                            'event_type': 'performance',
                            'function': func.__name__,
                            'duration_seconds': duration,
                            'status': 'error',
                            'error_type': type(e).__name__
                        }
                    }
                )
                raise
                
        return wrapper
    return decorator

# Usage
@log_performance(performance_logger.logger)
def train_rl_model(config):
    # Training logic here
    pass
```

## Best Practices

1. **Structured Format**: Always use structured logging with consistent field names
2. **Correlation IDs**: Track requests across service boundaries
3. **Log Levels**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
4. **Security**: Never log sensitive information (passwords, tokens, personal data)
5. **Performance**: Use async logging for high-throughput applications
6. **Retention**: Configure appropriate log retention policies
7. **Monitoring**: Set up alerts on log patterns indicating system issues
8. **Documentation**: Document log schema and field meanings