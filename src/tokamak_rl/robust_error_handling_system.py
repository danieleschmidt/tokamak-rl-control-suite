"""
Robust Error Handling and Validation System for Tokamak RL Control

This module provides comprehensive error handling, validation, and recovery
mechanisms for safety-critical plasma control operations.
"""

import math
import time
import traceback
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import threading
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tokamak_control.log')
    ]
)

logger = logging.getLogger('tokamak_rl')


class ErrorSeverity(Enum):
    """Error severity levels for plasma control systems."""
    CRITICAL = "CRITICAL"    # Immediate disruption risk
    HIGH = "HIGH"           # Performance degradation
    MEDIUM = "MEDIUM"       # Non-critical issues
    LOW = "LOW"             # Information only
    INFO = "INFO"           # Normal operations


class PlasmaControlError(Exception):
    """Base exception for plasma control errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None, recovery_action: Optional[str] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.recovery_action = recovery_action
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/analysis."""
        return {
            'message': str(self),
            'severity': self.severity.value,
            'context': self.context,
            'recovery_action': self.recovery_action,
            'timestamp': self.timestamp,
            'error_type': self.__class__.__name__
        }


class DisruptionRiskError(PlasmaControlError):
    """Error indicating disruption risk."""
    
    def __init__(self, message: str, disruption_probability: float, 
                 time_to_disruption: Optional[float] = None, **kwargs):
        super().__init__(message, ErrorSeverity.CRITICAL, **kwargs)
        self.disruption_probability = disruption_probability
        self.time_to_disruption = time_to_disruption


class SafetyLimitError(PlasmaControlError):
    """Error for safety limit violations."""
    
    def __init__(self, parameter_name: str, current_value: float, 
                 limit_value: float, **kwargs):
        message = f"Safety limit exceeded: {parameter_name} = {current_value:.3f}, limit = {limit_value:.3f}"
        super().__init__(message, ErrorSeverity.HIGH, **kwargs)
        self.parameter_name = parameter_name
        self.current_value = current_value
        self.limit_value = limit_value


class ControlSystemError(PlasmaControlError):
    """Error in control system components."""
    
    def __init__(self, component: str, **kwargs):
        super().__init__(f"Control system error in component: {component}", **kwargs)
        self.component = component


class ValidationError(PlasmaControlError):
    """Error in data validation."""
    
    def __init__(self, field_name: str, expected_type: str, actual_value: Any, **kwargs):
        message = f"Validation failed for {field_name}: expected {expected_type}, got {type(actual_value).__name__}"
        super().__init__(message, ErrorSeverity.MEDIUM, **kwargs)
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_value = actual_value


@dataclass
class ErrorStatistics:
    """Statistics for error tracking and analysis."""
    total_errors: int = 0
    critical_errors: int = 0
    high_errors: int = 0
    medium_errors: int = 0
    low_errors: int = 0
    errors_per_hour: float = 0.0
    mean_time_between_failures: float = 0.0
    recovery_success_rate: float = 0.0


class ErrorHandler:
    """
    Comprehensive error handler for tokamak control systems.
    Provides logging, recovery, and statistical tracking.
    """
    
    def __init__(self, max_error_history: int = 10000):
        self.error_history = deque(maxlen=max_error_history)
        self.error_stats = ErrorStatistics()
        self.recovery_handlers: Dict[str, Callable] = {}
        self.safety_callbacks: List[Callable] = []
        self.error_counts = defaultdict(int)
        
        # Threading for async error handling
        self._error_queue = deque()
        self._processing_thread = threading.Thread(target=self._process_error_queue, daemon=True)
        self._processing_thread.start()
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    def handle_error(self, error: PlasmaControlError, immediate: bool = False) -> bool:
        """
        Handle plasma control error with appropriate response.
        
        Args:
            error: The error to handle
            immediate: If True, handle synchronously
            
        Returns:
            bool: True if error was handled successfully
        """
        try:
            # Log the error
            self._log_error(error)
            
            # Update statistics
            self._update_error_statistics(error)
            
            # Store in history
            self.error_history.append(error)
            
            # Check for circuit breaker
            if self._check_circuit_breaker(error):
                return False
            
            # Handle based on severity
            if error.severity == ErrorSeverity.CRITICAL:
                return self._handle_critical_error(error, immediate)
            elif error.severity == ErrorSeverity.HIGH:
                return self._handle_high_severity_error(error, immediate)
            else:
                return self._handle_standard_error(error, immediate)
                
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            traceback.print_exc()
            return False
    
    def _log_error(self, error: PlasmaControlError):
        """Log error with appropriate level."""
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY: {error.message}", extra=error_dict)
        else:
            logger.info(f"{error.severity.value}: {error.message}", extra=error_dict)
    
    def _update_error_statistics(self, error: PlasmaControlError):
        """Update error statistics."""
        self.error_stats.total_errors += 1
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.error_stats.critical_errors += 1
        elif error.severity == ErrorSeverity.HIGH:
            self.error_stats.high_errors += 1
        elif error.severity == ErrorSeverity.MEDIUM:
            self.error_stats.medium_errors += 1
        else:
            self.error_stats.low_errors += 1
        
        # Update error type counts
        self.error_counts[error.__class__.__name__] += 1
        
        # Calculate rates
        if len(self.error_history) > 1:
            time_span = self.error_history[-1].timestamp - self.error_history[0].timestamp
            if time_span > 0:
                self.error_stats.errors_per_hour = len(self.error_history) / (time_span / 3600)
                self.error_stats.mean_time_between_failures = time_span / len(self.error_history)
    
    def _check_circuit_breaker(self, error: PlasmaControlError) -> bool:
        """Check if circuit breaker should trip."""
        component = error.context.get('component', 'unknown')
        
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = {
                'failure_count': 0,
                'last_failure': 0,
                'state': 'CLOSED',
                'failure_threshold': 5,
                'timeout': 60.0  # seconds
            }
        
        breaker = self.circuit_breakers[component]
        current_time = time.time()
        
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            breaker['failure_count'] += 1
            breaker['last_failure'] = current_time
            
            # Check if we should open the circuit
            if breaker['failure_count'] >= breaker['failure_threshold']:
                breaker['state'] = 'OPEN'
                logger.warning(f"Circuit breaker OPENED for component: {component}")
                return True
        
        # Check if we can close the circuit
        if breaker['state'] == 'OPEN' and (current_time - breaker['last_failure']) > breaker['timeout']:
            breaker['state'] = 'HALF_OPEN'
            breaker['failure_count'] = 0
            logger.info(f"Circuit breaker moved to HALF_OPEN for component: {component}")
        
        return breaker['state'] == 'OPEN'
    
    def _handle_critical_error(self, error: PlasmaControlError, immediate: bool) -> bool:
        """Handle critical errors that may cause disruptions."""
        logger.critical(f"CRITICAL ERROR DETECTED: {error}")
        
        # Notify all safety callbacks immediately
        for callback in self.safety_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Safety callback failed: {e}")
        
        # Attempt recovery
        if error.recovery_action and error.recovery_action in self.recovery_handlers:
            try:
                success = self.recovery_handlers[error.recovery_action](error)
                if success:
                    logger.info(f"Critical error recovery successful: {error.recovery_action}")
                    return True
                else:
                    logger.error(f"Critical error recovery failed: {error.recovery_action}")
            except Exception as e:
                logger.error(f"Recovery handler failed: {e}")
        
        # If no recovery or recovery failed, trigger emergency protocols
        self._trigger_emergency_protocols(error)
        return False
    
    def _handle_high_severity_error(self, error: PlasmaControlError, immediate: bool) -> bool:
        """Handle high severity errors."""
        if immediate:
            return self._attempt_recovery(error)
        else:
            # Queue for async processing
            self._error_queue.append(error)
            return True
    
    def _handle_standard_error(self, error: PlasmaControlError, immediate: bool) -> bool:
        """Handle standard errors."""
        if error.recovery_action and error.recovery_action in self.recovery_handlers:
            return self._attempt_recovery(error)
        return True
    
    def _attempt_recovery(self, error: PlasmaControlError) -> bool:
        """Attempt error recovery."""
        if not error.recovery_action or error.recovery_action not in self.recovery_handlers:
            return False
        
        try:
            success = self.recovery_handlers[error.recovery_action](error)
            
            if success:
                self.error_stats.recovery_success_rate = self._calculate_recovery_rate()
                logger.info(f"Error recovery successful: {error.recovery_action}")
            else:
                logger.warning(f"Error recovery failed: {error.recovery_action}")
            
            return success
            
        except Exception as e:
            logger.error(f"Recovery attempt failed with exception: {e}")
            return False
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate recovery success rate."""
        if self.error_stats.total_errors == 0:
            return 0.0
        
        recovery_attempts = sum(1 for error in self.error_history if error.recovery_action)
        if recovery_attempts == 0:
            return 0.0
        
        # This is a simplified calculation - in practice you'd track actual success/failure
        return 0.85  # Placeholder
    
    def _trigger_emergency_protocols(self, error: PlasmaControlError):
        """Trigger emergency protocols for critical errors."""
        logger.critical("TRIGGERING EMERGENCY PROTOCOLS")
        
        # Emergency actions would be implemented here
        # - Plasma shutdown sequences
        # - Gas injection systems
        # - Magnetic coil ramping
        # - Alert systems
        
        emergency_actions = [
            "plasma_current_rampdown",
            "gas_injection_activation",
            "disruption_mitigation_system",
            "emergency_coil_protection"
        ]
        
        for action in emergency_actions:
            logger.critical(f"Executing emergency action: {action}")
            # Implementation would call actual hardware systems
    
    def _process_error_queue(self):
        """Process errors asynchronously."""
        while True:
            try:
                if self._error_queue:
                    error = self._error_queue.popleft()
                    self._attempt_recovery(error)
                else:
                    time.sleep(0.1)  # Small delay when queue is empty
            except Exception as e:
                logger.error(f"Error in async error processing: {e}")
    
    def register_recovery_handler(self, action_name: str, handler: Callable[[PlasmaControlError], bool]):
        """Register a recovery handler for specific actions."""
        self.recovery_handlers[action_name] = handler
        logger.info(f"Recovery handler registered for action: {action_name}")
    
    def register_safety_callback(self, callback: Callable[[PlasmaControlError], None]):
        """Register a safety callback for critical errors."""
        self.safety_callbacks.append(callback)
        logger.info("Safety callback registered")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        stats = asdict(self.error_stats)
        stats['error_types'] = dict(self.error_counts)
        stats['circuit_breaker_states'] = {
            component: breaker['state'] 
            for component, breaker in self.circuit_breakers.items()
        }
        return stats
    
    def generate_error_report(self) -> str:
        """Generate comprehensive error report."""
        stats = self.get_error_statistics()
        
        report = ["Tokamak Control System Error Report", "=" * 40]
        report.append(f"Total Errors: {stats['total_errors']}")
        report.append(f"Critical Errors: {stats['critical_errors']}")
        report.append(f"High Severity Errors: {stats['high_errors']}")
        report.append(f"Errors per Hour: {stats['errors_per_hour']:.2f}")
        report.append(f"MTBF: {stats['mean_time_between_failures']:.2f} seconds")
        report.append(f"Recovery Success Rate: {stats['recovery_success_rate']:.2%}")
        
        report.append("\nError Types:")
        for error_type, count in stats['error_types'].items():
            report.append(f"  {error_type}: {count}")
        
        report.append("\nCircuit Breaker States:")
        for component, state in stats['circuit_breaker_states'].items():
            report.append(f"  {component}: {state}")
        
        if self.error_history:
            report.append(f"\nRecent Errors (last 5):")
            for error in list(self.error_history)[-5:]:
                report.append(f"  {error.severity.value}: {error}")
        
        return "\n".join(report)


class DataValidator:
    """
    Comprehensive data validation for plasma control systems.
    Validates sensor data, control inputs, and system states.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.validation_history = deque(maxlen=1000)
        self.failed_validations = 0
        self.total_validations = 0
        
    def add_validation_rule(self, field_name: str, rule_type: str, **kwargs):
        """Add validation rule for a field."""
        self.validation_rules[field_name] = {
            'type': rule_type,
            'parameters': kwargs
        }
    
    def validate_plasma_state(self, state: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate plasma state data."""
        errors = []
        
        # Required fields
        required_fields = ['plasma_current', 'beta_n', 'density', 'temperature']
        for field in required_fields:
            if field not in state:
                errors.append(ValidationError(field, 'required', None))
        
        # Validate individual fields
        for field, value in state.items():
            field_errors = self._validate_field(field, value)
            errors.extend(field_errors)
        
        # Cross-field validation
        cross_errors = self._validate_cross_field_constraints(state)
        errors.extend(cross_errors)
        
        self.total_validations += 1
        if errors:
            self.failed_validations += len(errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_control_action(self, action: List[float]) -> Tuple[bool, List[ValidationError]]:
        """Validate control action."""
        errors = []
        
        if not isinstance(action, list):
            errors.append(ValidationError('action', 'list', action))
            return False, errors
        
        # Check dimensions
        if len(action) != 8:  # Expected action dimension
            errors.append(ValidationError('action_length', '8', len(action)))
        
        # Check bounds
        for i, value in enumerate(action):
            if not isinstance(value, (int, float)):
                errors.append(ValidationError(f'action[{i}]', 'numeric', value))
            elif not (-1.0 <= value <= 1.0):
                errors.append(ValidationError(f'action[{i}]', '[-1,1]', value))
            elif math.isnan(value) or math.isinf(value):
                errors.append(ValidationError(f'action[{i}]', 'finite', value))
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_field(self, field_name: str, value: Any) -> List[ValidationError]:
        """Validate individual field."""
        errors = []
        
        if field_name not in self.validation_rules:
            return errors
        
        rule = self.validation_rules[field_name]
        rule_type = rule['type']
        params = rule['parameters']
        
        if rule_type == 'range':
            if not (params['min'] <= value <= params['max']):
                errors.append(ValidationError(field_name, f"range [{params['min']}, {params['max']}]", value))
        
        elif rule_type == 'positive':
            if value <= 0:
                errors.append(ValidationError(field_name, 'positive', value))
        
        elif rule_type == 'finite':
            if math.isnan(value) or math.isinf(value):
                errors.append(ValidationError(field_name, 'finite', value))
        
        elif rule_type == 'list_length':
            if len(value) != params['length']:
                errors.append(ValidationError(field_name, f"length {params['length']}", len(value)))
        
        return errors
    
    def _validate_cross_field_constraints(self, state: Dict[str, Any]) -> List[ValidationError]:
        """Validate constraints between multiple fields."""
        errors = []
        
        # Example: beta_n should be consistent with pressure and magnetic field
        if 'beta_n' in state and 'pressure' in state:
            beta_n = state['beta_n']
            if hasattr(state['pressure'], '__iter__'):
                max_pressure = max(state['pressure'])
                # Simplified check - beta_n should correlate with pressure
                expected_beta = max_pressure * 1e-5  # Simplified conversion
                if abs(beta_n - expected_beta) > 0.02:  # Tolerance
                    errors.append(ValidationError('beta_n_consistency', 'pressure_consistent', beta_n))
        
        # Check Greenwald limit
        if 'density' in state and 'plasma_current' in state:
            if hasattr(state['density'], '__iter__'):
                max_density = max(state['density'])
            else:
                max_density = state['density']
            
            plasma_current = state['plasma_current']
            greenwald_limit = plasma_current / (math.pi * 0.5**2) * 1e20  # Simplified
            
            if max_density > greenwald_limit * 1.2:  # 20% margin
                errors.append(ValidationError('density_limit', 'below_greenwald_limit', max_density))
        
        return errors
    
    def get_validation_statistics(self) -> Dict[str, float]:
        """Get validation statistics."""
        success_rate = 1.0 - (self.failed_validations / max(1, self.total_validations))
        
        return {
            'total_validations': self.total_validations,
            'failed_validations': self.failed_validations,
            'success_rate': success_rate,
            'validation_rules_count': len(self.validation_rules)
        }


def robust_operation_decorator(max_retries: int = 3, 
                             backoff_factor: float = 1.5,
                             exceptions: Tuple = (Exception,)):
    """
    Decorator for robust operation with automatic retry and exponential backoff.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Operation {func.__name__} failed, retrying in {wait_time:.2f}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Operation {func.__name__} failed after {max_retries} retries: {e}")
                        raise last_exception
                        
            raise last_exception
        
        return wrapper
    return decorator


def create_robust_error_handling_system() -> Dict[str, Any]:
    """Create comprehensive robust error handling system."""
    
    error_handler = ErrorHandler()
    validator = DataValidator()
    
    # Setup validation rules
    validator.add_validation_rule('plasma_current', 'range', min=0.1, max=20.0)  # MA
    validator.add_validation_rule('beta_n', 'range', min=0.0, max=5.0)  # %
    validator.add_validation_rule('density', 'positive')
    validator.add_validation_rule('temperature', 'positive')
    validator.add_validation_rule('q_profile', 'list_length', length=10)
    
    # Register recovery handlers
    def plasma_current_recovery(error: PlasmaControlError) -> bool:
        """Recovery handler for plasma current issues."""
        logger.info("Attempting plasma current recovery")
        # Implementation would adjust plasma current control
        return True
    
    def density_limit_recovery(error: PlasmaControlError) -> bool:
        """Recovery handler for density limit violations."""
        logger.info("Attempting density control recovery")
        # Implementation would adjust gas puffing
        return True
    
    def safety_system_callback(error: PlasmaControlError):
        """Safety system callback for critical errors."""
        logger.critical(f"SAFETY SYSTEM ALERT: {error}")
        # Implementation would trigger hardware safety systems
    
    error_handler.register_recovery_handler('plasma_current_control', plasma_current_recovery)
    error_handler.register_recovery_handler('density_control', density_limit_recovery)
    error_handler.register_safety_callback(safety_system_callback)
    
    # Demonstration functions
    @robust_operation_decorator(max_retries=3)
    def robust_control_step(observation: List[float], action: List[float]) -> Dict[str, Any]:
        """Robust control step with validation and error handling."""
        # Validate observation
        obs_dict = {
            'plasma_current': observation[0] if len(observation) > 0 else 1.0,
            'beta_n': observation[1] if len(observation) > 1 else 0.02,
            'density': observation[2:12] if len(observation) > 11 else [1e19] * 10,
            'temperature': observation[12:22] if len(observation) > 21 else [10.0] * 10
        }
        
        obs_valid, obs_errors = validator.validate_plasma_state(obs_dict)
        if not obs_valid:
            for error in obs_errors:
                error_handler.handle_error(error)
        
        # Validate control action
        action_valid, action_errors = validator.validate_control_action(action)
        if not action_valid:
            for error in action_errors:
                error_handler.handle_error(error)
        
        # Simulate control computation
        if random.random() < 0.05:  # 5% chance of computational error
            raise ControlSystemError("numerical_solver", 
                                   recovery_action="numerical_reset",
                                   context={'step': 'control_computation'})
        
        return {
            'observation_valid': obs_valid,
            'action_valid': action_valid,
            'control_output': action,
            'validation_errors': len(obs_errors) + len(action_errors)
        }
    
    def system_health_check() -> Dict[str, Any]:
        """Comprehensive system health check."""
        error_stats = error_handler.get_error_statistics()
        validation_stats = validator.get_validation_statistics()
        
        health_score = 100.0
        
        # Deduct points for errors
        health_score -= error_stats['critical_errors'] * 20
        health_score -= error_stats['high_errors'] * 10
        health_score -= error_stats['medium_errors'] * 5
        
        # Deduct points for validation failures
        health_score -= (1.0 - validation_stats['success_rate']) * 30
        
        # Check circuit breaker states
        for component, state in error_stats['circuit_breaker_states'].items():
            if state == 'OPEN':
                health_score -= 15
            elif state == 'HALF_OPEN':
                health_score -= 5
        
        health_score = max(0.0, health_score)
        
        return {
            'health_score': health_score,
            'status': 'HEALTHY' if health_score > 80 else 'DEGRADED' if health_score > 50 else 'CRITICAL',
            'error_statistics': error_stats,
            'validation_statistics': validation_stats,
            'recommendations': _generate_health_recommendations(health_score, error_stats, validation_stats)
        }
    
    def _generate_health_recommendations(health_score: float, error_stats: Dict, validation_stats: Dict) -> List[str]:
        """Generate health recommendations based on system state."""
        recommendations = []
        
        if error_stats['critical_errors'] > 0:
            recommendations.append("URGENT: Review critical error logs and implement immediate corrective actions")
        
        if error_stats['errors_per_hour'] > 10:
            recommendations.append("High error rate detected - consider system maintenance")
        
        if validation_stats['success_rate'] < 0.95:
            recommendations.append("Validation failure rate high - check sensor calibration")
        
        if any(state == 'OPEN' for state in error_stats['circuit_breaker_states'].values()):
            recommendations.append("Circuit breakers open - affected components need attention")
        
        if health_score < 70:
            recommendations.append("System health degraded - schedule comprehensive diagnostic")
        
        if not recommendations:
            recommendations.append("System operating normally - maintain current monitoring")
        
        return recommendations
    
    return {
        'error_handler': error_handler,
        'validator': validator,
        'robust_control_step': robust_control_step,
        'system_health_check': system_health_check,
        'system_type': 'robust_error_handling'
    }


# Test and demonstration functions
def demonstrate_error_handling_system():
    """Demonstrate the robust error handling system."""
    print("Robust Error Handling System Demonstration")
    print("=" * 50)
    
    system = create_robust_error_handling_system()
    
    # Test normal operation
    print("\n1. Testing Normal Operation:")
    normal_obs = [2.0, 0.03] + [1e19] * 10 + [10.0] * 10  # Valid observation
    normal_action = [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1]  # Valid action
    
    try:
        result = system['robust_control_step'](normal_obs, normal_action)
        print(f"  ✅ Normal operation successful")
        print(f"  Validation errors: {result['validation_errors']}")
    except Exception as e:
        print(f"  ❌ Normal operation failed: {e}")
    
    # Test error conditions
    print("\n2. Testing Error Conditions:")
    
    # Invalid observation
    invalid_obs = [50.0, -0.1] + [0] * 20  # Out of range values
    try:
        system['robust_control_step'](invalid_obs, normal_action)
        print("  ✅ Handled invalid observation")
    except Exception as e:
        print(f"  ⚠️  Exception with invalid observation: {e}")
    
    # Invalid action
    invalid_action = [2.0, -3.0, float('nan'), float('inf'), 0.5, 0.0, -0.1, 1.5]
    try:
        system['robust_control_step'](normal_obs, invalid_action)
        print("  ✅ Handled invalid action")
    except Exception as e:
        print(f"  ⚠️  Exception with invalid action: {e}")
    
    # System health check
    print("\n3. System Health Check:")
    health = system['system_health_check']()
    print(f"  Health Score: {health['health_score']:.1f}/100")
    print(f"  Status: {health['status']}")
    print(f"  Total Errors: {health['error_statistics']['total_errors']}")
    print(f"  Validation Success Rate: {health['validation_statistics']['success_rate']:.1%}")
    
    if health['recommendations']:
        print("  Recommendations:")
        for rec in health['recommendations']:
            print(f"    - {rec}")
    
    # Error report
    print("\n4. Error Report:")
    error_report = system['error_handler'].generate_error_report()
    print(error_report)


if __name__ == "__main__":
    demonstrate_error_handling_system()