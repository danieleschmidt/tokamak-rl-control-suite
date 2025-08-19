"""
Comprehensive input validation and error handling for tokamak RL control.

This module provides robust validation for all inputs, parameters, and states
to ensure safe and reliable operation of the tokamak control system.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import warnings
import logging
from dataclasses import dataclass
from enum import Enum
import functools

# Setup module logger
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field_name: Optional[str] = None
    suggested_fix: Optional[str] = None
    value: Optional[Any] = None


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 suggested_fix: Optional[str] = None):
        super().__init__(message)
        self.field_name = field_name
        self.suggested_fix = suggested_fix


class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self, strict_mode: bool = True, log_warnings: bool = True):
        self.strict_mode = strict_mode
        self.log_warnings = log_warnings
        self.validation_history: List[ValidationResult] = []
        
    def validate_numeric_input(self, value: Any, field_name: str, 
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None,
                             allow_inf: bool = False,
                             allow_nan: bool = False) -> ValidationResult:
        """Validate numeric input with comprehensive checks."""
        
        # Check if value is numeric
        if not isinstance(value, (int, float, np.number)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} must be numeric, got {type(value).__name__}",
                    field_name=field_name,
                    suggested_fix="Provide a numeric value (int, float, or numpy number)",
                    value=value
                )
        
        # Check for NaN
        if np.isnan(value) and not allow_nan:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} cannot be NaN",
                field_name=field_name,
                suggested_fix="Ensure calculations don't produce NaN values",
                value=value
            )
        
        # Check for infinity
        if np.isinf(value) and not allow_inf:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} cannot be infinite",
                field_name=field_name,
                suggested_fix="Ensure calculations don't produce infinite values",
                value=value
            )
        
        # Check range constraints
        if min_val is not None and value < min_val:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} ({value}) is below minimum ({min_val})",
                field_name=field_name,
                suggested_fix=f"Use value >= {min_val}",
                value=value
            )
        
        if max_val is not None and value > max_val:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} ({value}) exceeds maximum ({max_val})",
                field_name=field_name,
                suggested_fix=f"Use value <= {max_val}",
                value=value
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"{field_name} validation passed",
            field_name=field_name,
            value=value
        )
    
    def validate_array_input(self, value: Any, field_name: str,
                           expected_shape: Optional[Tuple[int, ...]] = None,
                           expected_dtype: Optional[type] = None,
                           min_val: Optional[float] = None,
                           max_val: Optional[float] = None,
                           allow_empty: bool = False) -> ValidationResult:
        """Validate array/list input with shape and value checks."""
        
        # Convert to numpy array if possible
        try:
            arr = np.asarray(value)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} cannot be converted to array: {e}",
                field_name=field_name,
                suggested_fix="Provide array-like input (list, tuple, numpy array)",
                value=value
            )
        
        # Check if empty
        if arr.size == 0 and not allow_empty:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} cannot be empty",
                field_name=field_name,
                suggested_fix="Provide non-empty array",
                value=value
            )
        
        # Check shape
        if expected_shape is not None and arr.shape != expected_shape:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} shape {arr.shape} doesn't match expected {expected_shape}",
                field_name=field_name,
                suggested_fix=f"Reshape input to {expected_shape}",
                value=value
            )
        
        # Check data type
        if expected_dtype is not None and not np.issubdtype(arr.dtype, expected_dtype):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"{field_name} dtype {arr.dtype} doesn't match expected {expected_dtype}",
                field_name=field_name,
                suggested_fix=f"Convert to {expected_dtype}",
                value=value
            )
        
        # Check for NaN/Inf values
        if arr.dtype.kind in ['f', 'c']:  # Float or complex
            if np.any(np.isnan(arr)):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} contains NaN values",
                    field_name=field_name,
                    suggested_fix="Remove or replace NaN values",
                    value=value
                )
            
            if np.any(np.isinf(arr)):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} contains infinite values",
                    field_name=field_name,
                    suggested_fix="Remove or replace infinite values",
                    value=value
                )
        
        # Check value ranges
        if min_val is not None and np.any(arr < min_val):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} contains values below minimum ({min_val})",
                field_name=field_name,
                suggested_fix=f"Ensure all values >= {min_val}",
                value=value
            )
        
        if max_val is not None and np.any(arr > max_val):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} contains values above maximum ({max_val})",
                field_name=field_name,
                suggested_fix=f"Ensure all values <= {max_val}",
                value=value
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"{field_name} validation passed",
            field_name=field_name,
            value=value
        )
    
    def validate_config_dict(self, config: Dict[str, Any], 
                           required_fields: List[str],
                           field_validators: Dict[str, Callable] = None) -> List[ValidationResult]:
        """Validate configuration dictionary."""
        results = []
        
        # Check for required fields
        for field in required_fields:
            if field not in config:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field}' missing from config",
                    field_name=field,
                    suggested_fix=f"Add '{field}' to configuration"
                ))
        
        # Validate individual fields
        if field_validators:
            for field, validator in field_validators.items():
                if field in config:
                    try:
                        result = validator(config[field], field)
                        results.append(result)
                    except Exception as e:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"Validation failed for '{field}': {e}",
                            field_name=field,
                            suggested_fix="Check field value and type"
                        ))
        
        return results
    
    def validate_and_sanitize_action(self, action: Any) -> Tuple[np.ndarray, List[ValidationResult]]:
        """Validate and sanitize RL action input."""
        results = []
        
        # Validate as array
        result = self.validate_array_input(
            action, "action", 
            expected_shape=(8,),
            min_val=-10.0,
            max_val=10.0
        )
        results.append(result)
        
        if not result.is_valid:
            # Return safe default action
            safe_action = np.zeros(8)
            return safe_action, results
        
        # Convert and sanitize
        action_array = np.asarray(action, dtype=np.float32)
        
        # Clip to safe ranges
        action_array[:6] = np.clip(action_array[:6], -1.0, 1.0)  # PF coils
        action_array[6] = np.clip(action_array[6], 0.0, 1.0)     # Gas puff
        action_array[7] = np.clip(action_array[7], 0.0, 1.0)     # Heating
        
        # Check for suspicious rapid changes
        if hasattr(self, '_last_action'):
            action_change = np.abs(action_array - self._last_action)
            if np.any(action_change > 0.5):
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message="Large action change detected",
                    field_name="action",
                    suggested_fix="Consider gradual action changes for stability"
                ))
        
        self._last_action = action_array.copy()
        
        return action_array, results
    
    def process_validation_results(self, results: List[ValidationResult]) -> None:
        """Process validation results according to settings."""
        for result in results:
            self.validation_history.append(result)
            
            if self.log_warnings and result.severity != ValidationSeverity.INFO:
                log_method = getattr(logger, result.severity.value, logger.info)
                log_method(f"Validation: {result.message}")
            
            if self.strict_mode and result.severity == ValidationSeverity.ERROR:
                raise ValidationError(
                    result.message,
                    result.field_name,
                    result.suggested_fix
                )
    
    def get_validation_summary(self) -> Dict[str, int]:
        """Get summary of validation history."""
        summary = {severity.value: 0 for severity in ValidationSeverity}
        
        for result in self.validation_history:
            summary[result.severity.value] += 1
            
        return summary
    
    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()


def validate_inputs(validator_instance: Optional[InputValidator] = None):
    """Decorator for automatic input validation."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use provided validator or create default
            validator = validator_instance or InputValidator()
            
            # Get function signature for parameter names
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Validate positional arguments
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    
                    # Skip self/cls parameters
                    if param_name in ['self', 'cls']:
                        continue
                    
                    # Basic validation for numeric types
                    if isinstance(arg, (int, float, np.number)):
                        result = validator.validate_numeric_input(arg, param_name)
                        validator.process_validation_results([result])
                    
                    # Basic validation for array types
                    elif isinstance(arg, (list, tuple, np.ndarray)):
                        result = validator.validate_array_input(arg, param_name)
                        validator.process_validation_results([result])
            
            # Validate keyword arguments
            for param_name, arg in kwargs.items():
                if isinstance(arg, (int, float, np.number)):
                    result = validator.validate_numeric_input(arg, param_name)
                    validator.process_validation_results([result])
                elif isinstance(arg, (list, tuple, np.ndarray)):
                    result = validator.validate_array_input(arg, param_name)
                    validator.process_validation_results([result])
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class PhysicsValidator:
    """Specialized validator for physics-related inputs and states."""
    
    def __init__(self):
        self.base_validator = InputValidator()
    
    def validate_plasma_parameters(self, params: Dict[str, Any]) -> List[ValidationResult]:
        """Validate plasma physics parameters."""
        results = []
        
        # Current validation
        if 'plasma_current' in params:
            result = self.base_validator.validate_numeric_input(
                params['plasma_current'], 'plasma_current',
                min_val=0.1, max_val=20.0
            )
            results.append(result)
        
        # Beta validation
        if 'plasma_beta' in params:
            result = self.base_validator.validate_numeric_input(
                params['plasma_beta'], 'plasma_beta',
                min_val=0.0, max_val=0.1
            )
            results.append(result)
        
        # Q-profile validation
        if 'q_profile' in params:
            result = self.base_validator.validate_array_input(
                params['q_profile'], 'q_profile',
                expected_shape=(10,),
                min_val=0.5, max_val=10.0
            )
            results.append(result)
        
        # Density profile validation
        if 'density_profile' in params:
            result = self.base_validator.validate_array_input(
                params['density_profile'], 'density_profile',
                expected_shape=(10,),
                min_val=0.0, max_val=2e20
            )
            results.append(result)
        
        return results
    
    def validate_tokamak_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate tokamak configuration parameters."""
        results = []
        
        required_fields = ['major_radius', 'minor_radius', 'toroidal_field', 'plasma_current']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required tokamak parameter '{field}' missing",
                    field_name=field,
                    suggested_fix=f"Add '{field}' to tokamak configuration"
                ))
        
        # Validate specific parameters
        if 'major_radius' in config:
            result = self.base_validator.validate_numeric_input(
                config['major_radius'], 'major_radius',
                min_val=0.5, max_val=10.0
            )
            results.append(result)
        
        if 'minor_radius' in config:
            result = self.base_validator.validate_numeric_input(
                config['minor_radius'], 'minor_radius',
                min_val=0.1, max_val=5.0
            )
            results.append(result)
        
        # Aspect ratio check
        if 'major_radius' in config and 'minor_radius' in config:
            try:
                aspect_ratio = config['major_radius'] / config['minor_radius']
                if aspect_ratio < 1.5 or aspect_ratio > 10.0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Unusual aspect ratio {aspect_ratio:.2f}",
                        field_name="aspect_ratio",
                        suggested_fix="Check major_radius/minor_radius ratio"
                    ))
            except (ZeroDivisionError, TypeError):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Cannot compute aspect ratio",
                    field_name="aspect_ratio",
                    suggested_fix="Ensure major_radius and minor_radius are valid numbers"
                ))
        
        return results


# Global validator instances
default_validator = InputValidator()
physics_validator = PhysicsValidator()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with validation and fallback."""
    try:
        if abs(denominator) < 1e-10:
            logger.warning(f"Division by near-zero value: {denominator}")
            return default
        
        result = numerator / denominator
        
        if np.isnan(result) or np.isinf(result):
            logger.warning(f"Division produced invalid result: {numerator}/{denominator}")
            return default
            
        return result
    except Exception as e:
        logger.error(f"Division error: {e}")
        return default


def safe_sqrt(value: float, default: float = 0.0) -> float:
    """Safe square root with validation."""
    try:
        if value < 0:
            logger.warning(f"Square root of negative value: {value}")
            return default
        
        result = np.sqrt(value)
        
        if np.isnan(result) or np.isinf(result):
            logger.warning(f"Square root produced invalid result: sqrt({value})")
            return default
            
        return result
    except Exception as e:
        logger.error(f"Square root error: {e}")
        return default


def sanitize_array(arr: np.ndarray, min_val: float = None, max_val: float = None,
                  replace_nan: float = 0.0, replace_inf: float = None) -> np.ndarray:
    """Sanitize numpy array by replacing invalid values."""
    sanitized = arr.copy()
    
    # Replace NaN values
    if np.any(np.isnan(sanitized)):
        sanitized = np.nan_to_num(sanitized, nan=replace_nan)
        logger.warning("Replaced NaN values in array")
    
    # Replace infinite values
    if replace_inf is not None and np.any(np.isinf(sanitized)):
        sanitized = np.nan_to_num(sanitized, posinf=replace_inf, neginf=-replace_inf)
        logger.warning("Replaced infinite values in array")
    
    # Clip to range
    if min_val is not None or max_val is not None:
        sanitized = np.clip(sanitized, min_val, max_val)
    
    return sanitized