# Tokamak RL Control System - API Reference

## Overview

This document provides comprehensive API documentation for the Tokamak RL Control System, including all modules, classes, and functions.

## 🧠 Core Agent APIs

### `tokamak_rl.agents`

#### `SACAgent`

Enhanced Soft Actor-Critic agent optimized for tokamak plasma control.

```python
class SACAgent(BaseAgent):
    def __init__(self, observation_space, action_space, 
                 learning_rate=3e-4, buffer_size=1000000,
                 batch_size=256, tau=0.005, gamma=0.99,
                 alpha=0.2, hidden_dim=256, device=None,
                 auto_entropy_tuning=True, target_entropy_ratio=-1.0,
                 gradient_steps=1, update_frequency=1)
```

**Parameters:**
- `observation_space`: Gym observation space
- `action_space`: Gym action space  
- `learning_rate`: Learning rate for optimizers (default: 3e-4)
- `buffer_size`: Replay buffer capacity (default: 1M)
- `batch_size`: Training batch size (default: 256)
- `tau`: Soft update coefficient (default: 0.005)
- `gamma`: Discount factor (default: 0.99)
- `alpha`: Entropy regularization coefficient (default: 0.2)
- `hidden_dim`: Hidden layer dimensions (default: 256)
- `device`: Device for computation (default: auto-detect)
- `auto_entropy_tuning`: Enable automatic entropy tuning (default: True)
- `target_entropy_ratio`: Target entropy ratio (default: -1.0)
- `gradient_steps`: Number of gradient steps per update (default: 1)
- `update_frequency`: Update frequency (default: 1)

**Methods:**

```python
def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
    """Select action using current policy."""
    
def learn(self, experiences: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Enhanced training with multiple gradient steps and entropy tuning."""
    
def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                  next_state: np.ndarray, done: bool) -> None:
    """Add experience to replay buffer."""
    
def save(self, path: str) -> None:
    """Save agent models."""
    
def load(self, path: str) -> None:
    """Load agent models."""
```

#### `DreamerAgent`

Model-based RL agent with world model learning.

```python
class DreamerAgent(BaseAgent):
    def __init__(self, observation_space, action_space,
                 learning_rate=1e-4, hidden_dim=200,
                 latent_dim=30, device=None)
```

**Example Usage:**

```python
import gymnasium as gym
from tokamak_rl.agents import create_agent

# Create environment
env = gym.make('TokamakControl-v1')

# Create SAC agent
agent = create_agent('SAC', env.observation_space, env.action_space,
                     learning_rate=1e-4, auto_entropy_tuning=True)

# Training loop
for episode in range(1000):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.add_experience(obs, action, reward, next_obs, done)
        
        if len(agent.replay_buffer) > agent.batch_size:
            losses = agent.learn()
            
        obs = next_obs
```

## 🛡️ Safety System APIs

### `tokamak_rl.safety`

#### `SafetyShield`

Real-time safety shield with predictive risk assessment.

```python
class SafetyShield:
    def __init__(self, limits: Optional[SafetyLimits] = None,
                 disruption_predictor: Optional[DisruptionPredictor] = None,
                 adaptive_constraints: bool = True,
                 safety_margin_factor: float = 1.2)
```

**Methods:**

```python
def filter_action(self, proposed_action: np.ndarray, 
                 plasma_state: PlasmaState) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Filter proposed action through safety constraints.
    
    Returns:
        Tuple of (safe_action, safety_info)
    """
    
def get_safety_statistics(self) -> Dict[str, float]:
    """Get comprehensive safety performance statistics."""
    
def reset(self) -> None:
    """Reset safety shield state and tracking metrics."""
```

#### `DisruptionPredictor`

LSTM-based disruption prediction system.

```python
class DisruptionPredictor:
    def __init__(self, model_path: Optional[str] = None)
    
    def predict_disruption(self, plasma_state: PlasmaState) -> float:
        """Predict disruption probability for current plasma state."""
        
    def load_model(self, model_path: str) -> None:
        """Load pre-trained disruption prediction model."""
        
    def reset(self) -> None:
        """Reset history buffer."""
```

**Example Usage:**

```python
from tokamak_rl.safety import create_safety_system
from tokamak_rl.physics import TokamakConfig

# Create safety system
config = TokamakConfig()
safety_shield = create_safety_system(config)

# Use in control loop
proposed_action = agent.act(observation)
safe_action, safety_info = safety_shield.filter_action(proposed_action, plasma_state)

if safety_info['action_modified']:
    print(f"Safety violations: {safety_info['violations']}")
    print(f"Disruption risk: {safety_info['disruption_risk']:.3f}")
```

## ⚡ Performance Optimization APIs

### `tokamak_rl.optimization`

#### `AdaptiveCache`

High-performance cache with multiple strategies.

```python
class AdaptiveCache:
    def __init__(self, max_size: int = 1000, 
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 ttl_seconds: Optional[float] = None,
                 memory_limit_mb: Optional[int] = None)
    
    def get(self, key: str, default=None):
        """Get value from cache."""
        
    def put(self, key: str, value: Any, ttl_override: Optional[float] = None) -> None:
        """Put value in cache."""
        
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
```

#### `PerformanceOptimizer`

Central performance optimization coordinator.

```python
class PerformanceOptimizer:
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD)
    
    def optimize_function(self, cache_size: int = None,
                         use_parallel: bool = False,
                         cache_strategy: CacheStrategy = CacheStrategy.LRU):
        """Decorator to optimize function performance."""
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
```

**Decorators:**

```python
from tokamak_rl.optimization import cached, optimized

@cached(cache_size=1000)
def expensive_computation(x, y):
    return complex_calculation(x, y)

@optimized(use_parallel=True, cache_size=500)
def parallel_processing(data_list):
    return [process_item(item) for item in data_list]
```

## 🔍 Validation and Security APIs

### `tokamak_rl.validation`

#### `InputValidator`

Comprehensive input validation system.

```python
class InputValidator:
    def validate_numeric_input(self, value: Any, field_name: str,
                              min_val: Optional[float] = None,
                              max_val: Optional[float] = None) -> ValidationResult:
        """Validate numeric input with range checking."""
        
    def validate_array_input(self, value: Any, field_name: str,
                            expected_shape: Optional[Tuple[int, ...]] = None,
                            dtype: Optional[type] = None) -> ValidationResult:
        """Validate array input with shape and type checking."""
        
    def validate_config_input(self, config: Dict[str, Any],
                             schema: Dict[str, Any]) -> ValidationResult:
        """Validate configuration dictionary against schema."""
```

### `tokamak_rl.security`

#### `InputSanitizer`

Input sanitization against code injection.

```python
class InputSanitizer:
    def sanitize_string(self, input_str: str, field_name: str = "input",
                       max_length: int = 1000) -> str:
        """Sanitize string input against injection attacks."""
        
    def sanitize_numeric_input(self, value: str, field_name: str = "input") -> float:
        """Sanitize and convert numeric input."""
        
    def sanitize_config_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration dictionary."""
```

#### `AccessController`

Role-based access control system.

```python
class AccessController:
    def add_role(self, role_name: str, security_level: SecurityLevel) -> None:
        """Add a new role with specified security level."""
        
    def check_access(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has access to perform action on resource."""
        
    def get_user_permissions(self, user_id: str) -> Dict[str, List[str]]:
        """Get all permissions for a user."""
```

## 📊 Monitoring and Diagnostics APIs

### `tokamak_rl.enhanced_monitoring`

#### `SystemHealthMonitor`

Real-time system health monitoring.

```python
class SystemHealthMonitor:
    def run_health_check(self, check_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive health check."""
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        
    def register_health_check(self, name: str, check_func: Callable[[], bool],
                             critical: bool = False) -> None:
        """Register custom health check."""
```

#### `MetricCollector`

Metric collection and aggregation.

```python
class MetricCollector:
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        
    def get_metric_summary(self, name: str, time_window: int = 300) -> Dict[str, float]:
        """Get metric summary statistics."""
        
    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
```

### `tokamak_rl.diagnostics`

#### `SystemDiagnostics`

System-wide diagnostic coordination.

```python
class SystemDiagnostics:
    def run_full_diagnostics(self, env=None, agent=None, safety_shield=None) -> Dict[str, DiagnosticResult]:
        """Run comprehensive system diagnostics."""
        
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        
    def schedule_diagnostic(self, name: str, interval: int, diagnostic_func: Callable) -> None:
        """Schedule recurring diagnostic."""
```

## 🌍 Global-First APIs

### `tokamak_rl.i18n`

#### `LocalizationManager`

Central localization management.

```python
class LocalizationManager:
    def __init__(self, locale_config: Optional[LocaleConfig] = None)
    
    def set_locale(self, language: SupportedLanguage, region: SupportedRegion):
        """Change current locale."""
        
    def get_message(self, key: str, **kwargs) -> str:
        """Get localized message."""
        
    def format_system_status(self, status: Dict[str, Any]) -> Dict[str, str]:
        """Format system status for display."""
```

**Convenience Functions:**

```python
from tokamak_rl.i18n import _, format_for_locale, set_global_locale

# Set global locale
set_global_locale(SupportedLanguage.FRENCH, SupportedRegion.FR)

# Get localized message
message = _("system.startup")

# Format value for locale
formatted = format_for_locale(1234.56, "number")
```

### `tokamak_rl.compliance`

#### `ComplianceMonitor`

Real-time compliance monitoring.

```python
class ComplianceMonitor:
    def __init__(self, standards: List[ComplianceStandard])
    
    def check_compliance(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive compliance check."""
        
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
```

#### `AuditLogger`

Tamper-resistant audit logging.

```python
class AuditLogger:
    def log_event(self, user_id: str, action: str, resource: str,
                  level: AuditLevel = AuditLevel.INFO,
                  details: Optional[Dict[str, Any]] = None,
                  classification: DataClassification = DataClassification.INTERNAL,
                  compliance_tags: Optional[List[str]] = None) -> str:
        """Log an auditable event."""
        
    def get_entries(self, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   user_id: Optional[str] = None,
                   level: Optional[AuditLevel] = None) -> List[AuditLogEntry]:
        """Retrieve audit entries with filtering."""
        
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit log."""
```

### `tokamak_rl.cross_platform`

#### `SystemDetector`

System information detection.

```python
class SystemDetector:
    @staticmethod
    def get_system_info() -> SystemInfo:
        """Get comprehensive system information."""
        
    @staticmethod
    def detect_platform() -> PlatformType:
        """Detect the current platform."""
        
    @staticmethod
    def detect_architecture() -> ArchitectureType:
        """Detect CPU architecture."""
        
    @staticmethod
    def detect_gpu() -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Detect GPU availability and capabilities."""
```

#### `EnvironmentManager`

Environment configuration management.

```python
class EnvironmentManager:
    def detect_environment(self) -> DeploymentEnvironment:
        """Detect current deployment environment."""
        
    def setup_environment(self) -> Dict[str, Any]:
        """Setup environment with platform-specific optimizations."""
        
    def get_environment_config(self, env: Optional[DeploymentEnvironment] = None) -> Dict[str, Any]:
        """Get environment-specific configuration."""
```

## 🔧 Utility APIs

### Factory Functions

```python
# Agents
from tokamak_rl.agents import create_agent
agent = create_agent('SAC', obs_space, action_space, **kwargs)

# Safety
from tokamak_rl.safety import create_safety_system
safety = create_safety_system(config, custom_limits)

# Compliance
from tokamak_rl.compliance import create_compliance_system, create_audit_logger
monitor = create_compliance_system([ComplianceStandard.ISO_45001])
logger = create_audit_logger(encryption_key)

# Cross-platform
from tokamak_rl.cross_platform import setup_cross_platform_environment
config = setup_cross_platform_environment()
```

### Global Managers

```python
# Optimization
from tokamak_rl.optimization import get_global_optimizer
optimizer = get_global_optimizer()

# Monitoring
from tokamak_rl.enhanced_monitoring import get_global_monitor
monitor = get_global_monitor()

# Localization
from tokamak_rl.i18n import get_global_l10n_manager
l10n = get_global_l10n_manager()

# System Info
from tokamak_rl.cross_platform import get_system_info, get_environment_manager
system_info = get_system_info()
env_manager = get_environment_manager()
```

## 📝 Configuration Schemas

### Agent Configuration

```python
agent_config = {
    "type": "SAC",
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "alpha": 0.2,
    "hidden_dim": 256,
    "auto_entropy_tuning": True,
    "gradient_steps": 1,
    "update_frequency": 1
}
```

### Safety Configuration

```python
safety_config = {
    "limits": {
        "q_min_threshold": 1.5,
        "beta_limit": 0.04,
        "density_limit": 1.2e20,
        "disruption_probability_limit": 0.1,
        "pf_coil_current_limit": 10.0,
        "emergency_response_time": 5.0
    },
    "adaptive_constraints": True,
    "safety_margin_factor": 1.2,
    "disruption_predictor": {
        "model_path": "/models/disruption_predictor.pt",
        "buffer_size": 50,
        "input_size": 45,
        "hidden_size": 64
    }
}
```

### Performance Configuration

```python
performance_config = {
    "optimization_level": "standard",
    "cache": {
        "strategy": "adaptive",
        "max_size": 1000,
        "ttl_seconds": 3600,
        "memory_limit_mb": 512
    },
    "parallel": {
        "max_workers": 8,
        "use_processes": False,
        "chunk_size": None
    },
    "gpu": {
        "enabled": True,
        "memory_fraction": 0.8,
        "device_id": 0
    }
}
```

## 🚨 Error Handling

### Exception Classes

```python
# Validation errors
class ValidationError(Exception): pass
class InvalidInputError(ValidationError): pass
class SchemaValidationError(ValidationError): pass

# Safety errors
class SafetyViolationError(Exception): pass
class EmergencyStopError(SafetyViolationError): pass
class DisruptionRiskError(SafetyViolationError): pass

# Security errors
class SecurityError(Exception): pass
class AccessDeniedError(SecurityError): pass
class InjectionAttemptError(SecurityError): pass

# Compliance errors
class ComplianceError(Exception): pass
class AuditLogError(ComplianceError): pass
class StandardViolationError(ComplianceError): pass
```

### Error Handling Best Practices

```python
from tokamak_rl.validation import ValidationError
from tokamak_rl.safety import SafetyViolationError

try:
    # Validate input
    result = validator.validate_numeric_input(value, "temperature", 0, 1000)
    if not result.is_valid:
        raise ValidationError(f"Invalid temperature: {result.message}")
    
    # Apply safety filter
    safe_action, safety_info = safety_shield.filter_action(action, state)
    if safety_info['emergency_mode']:
        raise SafetyViolationError("Emergency mode activated")
        
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    # Handle validation error
    
except SafetyViolationError as e:
    logger.critical(f"Safety violation: {e}")
    # Trigger emergency procedures
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## 📚 Examples

### Complete Control Loop

```python
import numpy as np
from tokamak_rl.agents import create_agent
from tokamak_rl.safety import create_safety_system
from tokamak_rl.validation import InputValidator
from tokamak_rl.enhanced_monitoring import create_monitoring_system
from tokamak_rl.i18n import set_global_locale, SupportedLanguage, SupportedRegion

# Setup
set_global_locale(SupportedLanguage.ENGLISH, SupportedRegion.US)
agent = create_agent('SAC', obs_space, action_space)
safety_shield = create_safety_system(config)
validator = InputValidator()
monitor = create_monitoring_system()

# Control loop
def control_loop(plasma_state):
    try:
        # Validate plasma state
        validation_result = validator.validate_plasma_state(plasma_state)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.message)
        
        # Get observation
        observation = plasma_state.get_observation()
        
        # Agent action
        proposed_action = agent.act(observation)
        
        # Safety filtering
        safe_action, safety_info = safety_shield.filter_action(
            proposed_action, plasma_state
        )
        
        # Record metrics
        monitor.record_metric("control_cycle_time", cycle_time)
        monitor.record_metric("safety_violations", len(safety_info['violations']))
        monitor.record_metric("disruption_risk", safety_info['disruption_risk'])
        
        return safe_action
        
    except Exception as e:
        monitor.record_metric("control_errors", 1)
        logger.error(f"Control loop error: {e}")
        return emergency_action()
```

---

**API Version**: 6.0  
**Last Updated**: 2024  
**Stability**: Production Ready ✅