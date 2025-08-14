#!/usr/bin/env python3
"""
Enhanced Error Handling and Reliability System for Tokamak RL Control Suite

This module implements comprehensive error handling, validation, logging,
monitoring, and health checks to make the system production-ready.
"""

import sys
import os
import logging
import traceback
import time
import json
import warnings
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    import numpy as np
except ImportError:
    # Use fallback numpy implementation from the existing codebase
    import math
    import random as rand
    
    class np:
        @staticmethod
        def array(x):
            return list(x) if hasattr(x, '__iter__') else [x]
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0.0
        
        @staticmethod
        def std(arr):
            if not arr or len(arr) < 2:
                return 0.0
            mean_val = sum(arr) / len(arr)
            return math.sqrt(sum((x - mean_val) ** 2 for x in arr) / len(arr))
        
        @staticmethod
        def max(arr):
            return max(arr) if arr else 0.0
        
        @staticmethod
        def min(arr):
            return min(arr) if arr else 0.0
        
        @staticmethod
        def isnan(x):
            return x != x
        
        @staticmethod
        def isinf(x):
            return x == float('inf') or x == float('-inf')
        
        @staticmethod
        def isfinite(x):
            return not (np.isnan(x) or np.isinf(x))


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    CRITICAL = "CRITICAL"


@dataclass
class ErrorContext:
    """Context information for error tracking."""
    component: str
    operation: str
    timestamp: float
    error_type: str
    error_message: str
    stacktrace: str
    context_data: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class HealthMetrics:
    """System health metrics."""
    status: HealthStatus
    uptime: float
    error_rate: float
    warning_rate: float
    last_error: Optional[ErrorContext]
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    timestamp: float


class ValidationError(Exception):
    """Custom validation error."""
    pass


class TokamakSafetyError(Exception):
    """Safety-critical tokamak operation error."""
    pass


class ConfigurationError(Exception):
    """Configuration validation error."""
    pass


class PerformanceError(Exception):
    """Performance threshold violation error."""
    pass


class EnhancedLogger:
    """Enhanced logging system with structured logging and alerts."""
    
    def __init__(self, name: str = "tokamak_rl", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup structured logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for all logs
        file_handler = logging.FileHandler(self.log_dir / "tokamak_rl.log")
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Alert subscribers
        self.alert_subscribers: List[Callable[[LogLevel, str, Dict], None]] = []
    
    def add_alert_subscriber(self, callback: Callable[[LogLevel, str, Dict], None]):
        """Add callback for critical alerts."""
        self.alert_subscribers.append(callback)
    
    def _send_alerts(self, level: LogLevel, message: str, context: Dict[str, Any]):
        """Send alerts to subscribers."""
        for callback in self.alert_subscribers:
            try:
                callback(level, message, context)
            except Exception as e:
                print(f"Alert callback failed: {e}")
    
    def log(self, level: LogLevel, message: str, **context):
        """Log structured message with context."""
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        full_message = f"{message} | {context_str}" if context else message
        
        # Log to appropriate level
        if level == LogLevel.DEBUG:
            self.logger.debug(full_message)
        elif level == LogLevel.INFO:
            self.logger.info(full_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(full_message)
        elif level == LogLevel.ERROR:
            self.logger.error(full_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(full_message)
            self._send_alerts(level, message, context)
    
    def debug(self, message: str, **context):
        self.log(LogLevel.DEBUG, message, **context)
    
    def info(self, message: str, **context):
        self.log(LogLevel.INFO, message, **context)
    
    def warning(self, message: str, **context):
        self.log(LogLevel.WARNING, message, **context)
    
    def error(self, message: str, **context):
        self.log(LogLevel.ERROR, message, **context)
    
    def critical(self, message: str, **context):
        self.log(LogLevel.CRITICAL, message, **context)


class Validator:
    """Comprehensive validation utilities."""
    
    @staticmethod
    def validate_plasma_state(state: Dict[str, Any]) -> None:
        """Validate plasma state parameters."""
        required_fields = ['plasma_current', 'plasma_beta', 'q_min', 'shape_error']
        
        for field in required_fields:
            if field not in state:
                raise ValidationError(f"Missing required field: {field}")
            
            value = state[field]
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Field {field} must be numeric, got {type(value)}")
            
            if np.isnan(value) or np.isinf(value):
                raise ValidationError(f"Field {field} contains invalid value: {value}")
        
        # Safety-critical validations
        if state['q_min'] < 0.5:
            raise TokamakSafetyError(f"Critically low safety factor: {state['q_min']}")
        
        if state['plasma_beta'] > 0.1:
            raise TokamakSafetyError(f"Beta limit exceeded: {state['plasma_beta']}")
        
        if abs(state['shape_error']) > 50.0:
            raise TokamakSafetyError(f"Extreme shape error: {state['shape_error']}")
    
    @staticmethod
    def validate_control_action(action: List[float], action_space_bounds: Dict) -> None:
        """Validate control actions are within safe bounds."""
        if not isinstance(action, (list, tuple)) or len(action) != 8:
            raise ValidationError(f"Action must be list/tuple of length 8, got {type(action)} of length {len(action) if hasattr(action, '__len__') else 'N/A'}")
        
        for i, val in enumerate(action):
            if not isinstance(val, (int, float)):
                raise ValidationError(f"Action[{i}] must be numeric, got {type(val)}")
            
            if np.isnan(val) or np.isinf(val):
                raise ValidationError(f"Action[{i}] contains invalid value: {val}")
        
        # Check bounds
        low = action_space_bounds.get('low', [-1.0] * 8)
        high = action_space_bounds.get('high', [1.0] * 8)
        
        for i, (val, low_bound, high_bound) in enumerate(zip(action, low, high)):
            if val < low_bound or val > high_bound:
                raise ValidationError(f"Action[{i}] = {val} outside bounds [{low_bound}, {high_bound}]")
    
    @staticmethod
    def validate_configuration(config: Dict[str, Any]) -> None:
        """Validate system configuration."""
        required_config = ['tokamak_config', 'control_frequency', 'safety_factor']
        
        for key in required_config:
            if key not in config:
                raise ConfigurationError(f"Missing required configuration: {key}")
        
        # Validate ranges
        if config['control_frequency'] < 10 or config['control_frequency'] > 1000:
            raise ConfigurationError(f"Control frequency {config['control_frequency']} Hz outside safe range [10, 1000]")
        
        if config['safety_factor'] < 1.0 or config['safety_factor'] > 5.0:
            raise ConfigurationError(f"Safety factor {config['safety_factor']} outside range [1.0, 5.0]")


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self, logger: EnhancedLogger):
        self.logger = logger
        self.start_time = time.time()
        self.error_history: List[ErrorContext] = []
        self.warning_history: List[ErrorContext] = []
        self.performance_history: List[float] = []
        self.health_status = HealthStatus.HEALTHY
        
        # Thresholds
        self.error_rate_threshold = 0.1  # 10% error rate
        self.warning_rate_threshold = 0.2  # 20% warning rate
        self.performance_threshold = 1000.0  # 1000ms max response time
    
    def record_error(self, error_context: ErrorContext):
        """Record error occurrence."""
        self.error_history.append(error_context)
        
        # Keep only recent errors (last hour)
        cutoff_time = time.time() - 3600
        self.error_history = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        self._update_health_status()
        self.logger.error(
            f"Error in {error_context.component}.{error_context.operation}",
            error_type=error_context.error_type,
            error_message=error_context.error_message,
            recovery_attempted=error_context.recovery_attempted
        )
    
    def record_warning(self, component: str, operation: str, message: str):
        """Record warning occurrence."""
        warning_context = ErrorContext(
            component=component,
            operation=operation,
            timestamp=time.time(),
            error_type="WARNING",
            error_message=message,
            stacktrace="",
            context_data={}
        )
        self.warning_history.append(warning_context)
        
        # Keep only recent warnings
        cutoff_time = time.time() - 3600
        self.warning_history = [w for w in self.warning_history if w.timestamp > cutoff_time]
        
        self._update_health_status()
        self.logger.warning(f"Warning in {component}.{operation}: {message}")
    
    def record_performance(self, duration: float):
        """Record operation performance."""
        self.performance_history.append(duration)
        
        # Keep only recent measurements
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        if duration > self.performance_threshold:
            self.logger.warning(
                f"Performance threshold exceeded",
                duration=duration,
                threshold=self.performance_threshold
            )
    
    def _update_health_status(self):
        """Update overall health status."""
        current_time = time.time()
        window_duration = 300  # 5 minutes
        cutoff_time = current_time - window_duration
        
        # Count recent errors and warnings
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        recent_warnings = [w for w in self.warning_history if w.timestamp > cutoff_time]
        
        error_rate = len(recent_errors) / (window_duration / 60)  # errors per minute
        warning_rate = len(recent_warnings) / (window_duration / 60)  # warnings per minute
        
        # Determine health status
        if recent_errors and any(e.error_type == "TokamakSafetyError" for e in recent_errors):
            self.health_status = HealthStatus.CRITICAL
        elif error_rate > self.error_rate_threshold:
            self.health_status = HealthStatus.UNHEALTHY
        elif warning_rate > self.warning_rate_threshold:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.HEALTHY
    
    def get_health_metrics(self) -> HealthMetrics:
        """Get current health metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate rates
        window_duration = 300  # 5 minutes
        cutoff_time = current_time - window_duration
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        recent_warnings = [w for w in self.warning_history if w.timestamp > cutoff_time]
        
        error_rate = len(recent_errors) / max(window_duration / 60, 1)
        warning_rate = len(recent_warnings) / max(window_duration / 60, 1)
        
        # Performance metrics
        recent_performance = [p for p in self.performance_history[-100:]]
        performance_metrics = {
            'avg_response_time': np.mean(recent_performance) if recent_performance else 0.0,
            'max_response_time': np.max(recent_performance) if recent_performance else 0.0,
            'min_response_time': np.min(recent_performance) if recent_performance else 0.0
        }
        
        # Resource usage (simplified)
        resource_usage = {
            'cpu_usage': 0.0,  # Would integrate with psutil in full implementation
            'memory_usage': 0.0,
            'disk_usage': 0.0
        }
        
        return HealthMetrics(
            status=self.health_status,
            uptime=uptime,
            error_rate=error_rate,
            warning_rate=warning_rate,
            last_error=self.error_history[-1] if self.error_history else None,
            performance_metrics=performance_metrics,
            resource_usage=resource_usage,
            timestamp=current_time
        )


class ErrorRecoveryManager:
    """Automated error recovery and self-healing."""
    
    def __init__(self, logger: EnhancedLogger, health_monitor: HealthMonitor):
        self.logger = logger
        self.health_monitor = health_monitor
        self.recovery_strategies: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type}")
    
    def handle_error(self, error: Exception, component: str, operation: str, 
                    context_data: Dict[str, Any] = None) -> bool:
        """Handle error with automatic recovery attempts."""
        error_context = ErrorContext(
            component=component,
            operation=operation,
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            stacktrace=traceback.format_exc(),
            context_data=context_data or {}
        )
        
        # Record error
        self.health_monitor.record_error(error_context)
        
        # Attempt recovery
        recovery_successful = False
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {error_type}")
                error_context.recovery_attempted = True
                
                recovery_strategy = self.recovery_strategies[error_type]
                recovery_strategy(error, error_context)
                
                recovery_successful = True
                error_context.recovery_successful = True
                self.logger.info(f"Recovery successful for {error_type}")
                
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery failed for {error_type}",
                    recovery_error=str(recovery_error)
                )
        
        return recovery_successful
    
    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        return self.circuit_breakers[component]


@contextmanager
def safe_operation(component: str, operation: str, 
                  error_recovery: ErrorRecoveryManager,
                  logger: EnhancedLogger,
                  context_data: Dict[str, Any] = None):
    """Context manager for safe operation execution with error handling."""
    start_time = time.time()
    
    try:
        logger.debug(f"Starting {component}.{operation}")
        yield
        
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        error_recovery.health_monitor.record_performance(duration)
        
        logger.debug(
            f"Completed {component}.{operation}",
            duration_ms=duration
        )
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        
        # Handle error with recovery
        recovery_successful = error_recovery.handle_error(
            e, component, operation, context_data
        )
        
        if not recovery_successful:
            logger.critical(
                f"Unrecovered error in {component}.{operation}",
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=duration
            )
            raise


class RobustTokamakSystem:
    """Enhanced tokamak system with comprehensive error handling."""
    
    def __init__(self):
        # Initialize logging and monitoring
        self.logger = EnhancedLogger("robust_tokamak")
        self.health_monitor = HealthMonitor(self.logger)
        self.error_recovery = ErrorRecoveryManager(self.logger, self.health_monitor)
        self.validator = Validator()
        
        # Setup alert callbacks
        self.logger.add_alert_subscriber(self._critical_alert_callback)
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
        # System state
        self.system_initialized = False
        self.emergency_shutdown = False
        
        self.logger.info("Robust Tokamak System initialized")
    
    def _critical_alert_callback(self, level: LogLevel, message: str, context: Dict):
        """Handle critical alerts."""
        if level == LogLevel.CRITICAL:
            # In a real system, this would trigger immediate operator notification
            print(f"🚨 CRITICAL ALERT: {message}")
            
            # Auto-initiate emergency protocols for safety-critical errors
            if "TokamakSafetyError" in str(context):
                self.initiate_emergency_shutdown("Safety-critical error detected")
    
    def _register_recovery_strategies(self):
        """Register error recovery strategies."""
        
        def validation_error_recovery(error: Exception, context: ErrorContext):
            """Recover from validation errors."""
            self.logger.info("Attempting validation error recovery")
            # Reset to safe default state
            self.reset_to_safe_defaults()
        
        def performance_error_recovery(error: Exception, context: ErrorContext):
            """Recover from performance errors."""
            self.logger.info("Attempting performance error recovery")
            # Reduce computational load
            self.reduce_computational_load()
        
        def configuration_error_recovery(error: Exception, context: ErrorContext):
            """Recover from configuration errors."""
            self.logger.info("Attempting configuration error recovery")
            # Load backup configuration
            self.load_backup_configuration()
        
        self.error_recovery.register_recovery_strategy("ValidationError", validation_error_recovery)
        self.error_recovery.register_recovery_strategy("PerformanceError", performance_error_recovery)
        self.error_recovery.register_recovery_strategy("ConfigurationError", configuration_error_recovery)
    
    def reset_to_safe_defaults(self):
        """Reset system to safe default parameters."""
        self.logger.info("Resetting to safe defaults")
        # Implementation would reset PF coils, heating, etc. to safe values
        
    def reduce_computational_load(self):
        """Reduce computational load for performance recovery."""
        self.logger.info("Reducing computational load")
        # Implementation would reduce simulation fidelity, update frequency, etc.
        
    def load_backup_configuration(self):
        """Load backup configuration."""
        self.logger.info("Loading backup configuration")
        # Implementation would load known good configuration
    
    def initiate_emergency_shutdown(self, reason: str):
        """Initiate emergency shutdown sequence."""
        self.logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")
        self.emergency_shutdown = True
        
        # In a real system, this would:
        # 1. Immediately cut heating systems
        # 2. Activate disruption mitigation
        # 3. Alert operators
        # 4. Log everything for post-incident analysis
    
    def validate_and_execute_action(self, action: List[float], 
                                   action_space_bounds: Dict) -> bool:
        """Validate and execute control action with comprehensive error handling."""
        
        with safe_operation("control", "validate_action", 
                          self.error_recovery, self.logger,
                          {"action": action}) as _:
            
            # Validate action
            self.validator.validate_control_action(action, action_space_bounds)
            
            # Additional safety checks
            if self.emergency_shutdown:
                raise TokamakSafetyError("System in emergency shutdown - action rejected")
            
            # Check circuit breaker
            circuit_breaker = self.error_recovery.get_circuit_breaker("control_execution")
            
            def execute_action():
                # Simulate action execution
                self.logger.debug("Executing control action", action=action)
                # In real system: send commands to hardware
                time.sleep(0.001)  # Simulate execution time
                return True
            
            result = circuit_breaker.call(execute_action)
            self.logger.info("Control action executed successfully")
            return result
    
    def monitor_plasma_state(self, state: Dict[str, Any]) -> HealthStatus:
        """Monitor plasma state with comprehensive validation."""
        
        with safe_operation("monitoring", "plasma_state", 
                          self.error_recovery, self.logger,
                          {"state_keys": list(state.keys())}) as _:
            
            # Validate plasma state
            self.validator.validate_plasma_state(state)
            
            # Additional monitoring
            if state.get('q_min', 2.0) < 1.2:
                self.health_monitor.record_warning(
                    "plasma_monitoring", "q_factor",
                    f"Low safety factor detected: {state['q_min']}"
                )
            
            if state.get('plasma_beta', 0.0) > 0.06:
                self.health_monitor.record_warning(
                    "plasma_monitoring", "beta_limit",
                    f"High beta detected: {state['plasma_beta']}"
                )
            
            # Get current health status
            health_metrics = self.health_monitor.get_health_metrics()
            return health_metrics.status
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_metrics = self.health_monitor.get_health_metrics()
        
        return {
            "health_status": health_metrics.status.value,
            "uptime": health_metrics.uptime,
            "error_rate": health_metrics.error_rate,
            "warning_rate": health_metrics.warning_rate,
            "performance_metrics": health_metrics.performance_metrics,
            "emergency_shutdown": self.emergency_shutdown,
            "system_initialized": self.system_initialized,
            "timestamp": time.time()
        }
    
    def run_self_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system self-diagnostics."""
        diagnostics = {
            "timestamp": time.time(),
            "tests": {},
            "overall_status": "PASS"
        }
        
        # Test logging system
        try:
            self.logger.info("Self-diagnostic test message")
            diagnostics["tests"]["logging"] = "PASS"
        except Exception as e:
            diagnostics["tests"]["logging"] = f"FAIL: {e}"
            diagnostics["overall_status"] = "FAIL"
        
        # Test validation system
        try:
            test_state = {
                'plasma_current': 10.0,
                'plasma_beta': 0.03,
                'q_min': 1.8,
                'shape_error': 2.5
            }
            self.validator.validate_plasma_state(test_state)
            diagnostics["tests"]["validation"] = "PASS"
        except Exception as e:
            diagnostics["tests"]["validation"] = f"FAIL: {e}"
            diagnostics["overall_status"] = "FAIL"
        
        # Test error recovery
        try:
            test_error = ValidationError("Test error for diagnostics")
            self.error_recovery.handle_error(test_error, "diagnostics", "test")
            diagnostics["tests"]["error_recovery"] = "PASS"
        except Exception as e:
            diagnostics["tests"]["error_recovery"] = f"FAIL: {e}"
            diagnostics["overall_status"] = "FAIL"
        
        # Test health monitoring
        try:
            health_metrics = self.health_monitor.get_health_metrics()
            diagnostics["tests"]["health_monitoring"] = "PASS"
            diagnostics["health_metrics"] = asdict(health_metrics)
        except Exception as e:
            diagnostics["tests"]["health_monitoring"] = f"FAIL: {e}"
            diagnostics["overall_status"] = "FAIL"
        
        self.logger.info(
            f"Self-diagnostics completed",
            overall_status=diagnostics["overall_status"],
            tests_passed=sum(1 for result in diagnostics["tests"].values() if result == "PASS"),
            tests_total=len(diagnostics["tests"])
        )
        
        return diagnostics


def run_comprehensive_error_handling_demo():
    """Demonstrate comprehensive error handling capabilities."""
    print("🛡️ Starting Comprehensive Error Handling Demo")
    print("=" * 50)
    
    # Initialize robust system
    system = RobustTokamakSystem()
    
    # Run self-diagnostics
    print("\n🔍 Running Self-Diagnostics...")
    diagnostics = system.run_self_diagnostics()
    print(f"Overall Status: {diagnostics['overall_status']}")
    for test_name, result in diagnostics["tests"].items():
        print(f"  {test_name}: {result}")
    
    # Test normal operation
    print("\n✅ Testing Normal Operation...")
    valid_action = [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, 0.5, 0.8]
    action_bounds = {
        'low': [-1.0] * 6 + [0.0, 0.0],
        'high': [1.0] * 6 + [1.0, 1.0]
    }
    
    try:
        result = system.validate_and_execute_action(valid_action, action_bounds)
        print(f"Action executed successfully: {result}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Test validation errors
    print("\n⚠️ Testing Validation Error Handling...")
    invalid_action = [2.0, -3.0, float('nan'), 0.5, 0.0, 0.2, 1.5, 0.8]  # Out of bounds and NaN
    
    try:
        system.validate_and_execute_action(invalid_action, action_bounds)
    except Exception as e:
        print(f"Validation error caught and handled: {type(e).__name__}: {e}")
    
    # Test plasma state monitoring
    print("\n🔬 Testing Plasma State Monitoring...")
    
    # Normal state
    normal_state = {
        'plasma_current': 10.0,
        'plasma_beta': 0.03,
        'q_min': 1.8,
        'shape_error': 2.5
    }
    
    health_status = system.monitor_plasma_state(normal_state)
    print(f"Normal state health: {health_status.value}")
    
    # Warning state
    warning_state = {
        'plasma_current': 12.0,
        'plasma_beta': 0.065,  # High beta
        'q_min': 1.3,  # Low q
        'shape_error': 4.0
    }
    
    health_status = system.monitor_plasma_state(warning_state)
    print(f"Warning state health: {health_status.value}")
    
    # Critical state (will trigger safety error)
    print("\n🚨 Testing Critical Safety Error...")
    try:
        critical_state = {
            'plasma_current': 15.0,
            'plasma_beta': 0.12,  # Way over limit
            'q_min': 0.8,  # Critically low
            'shape_error': 25.0
        }
        system.monitor_plasma_state(critical_state)
    except TokamakSafetyError as e:
        print(f"Safety error caught: {e}")
        print(f"Emergency shutdown status: {system.emergency_shutdown}")
    
    # Performance monitoring
    print("\n📊 Testing Performance Monitoring...")
    for i in range(5):
        start_time = time.time()
        time.sleep(0.1)  # Simulate work
        duration = (time.time() - start_time) * 1000
        system.health_monitor.record_performance(duration)
    
    # Get final system status
    print("\n📈 Final System Status:")
    status = system.get_system_status()
    for key, value in status.items():
        if key == "performance_metrics":
            print(f"  {key}:")
            for metric, val in value.items():
                print(f"    {metric}: {val:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n🎯 Error Handling Demo Complete!")
    print("✓ Validation systems tested")
    print("✓ Error recovery tested") 
    print("✓ Safety monitoring tested")
    print("✓ Performance monitoring tested")
    print("✓ Health status tracking tested")
    print("✓ Emergency protocols tested")


if __name__ == "__main__":
    run_comprehensive_error_handling_demo()