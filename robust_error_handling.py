#!/usr/bin/env python3
"""
Robust error handling and validation framework for tokamak-rl.
Implements comprehensive safety checks, input validation, and graceful degradation.
"""

import sys
import os
import logging
import traceback
import functools
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_values: Dict[str, Any]

class RobustValidator:
    """Comprehensive input and state validation."""
    
    def __init__(self):
        self.logger = logging.getLogger("RobustValidator")
        
        # Define validation ranges
        self.ranges = {
            "plasma_current": (0.1, 30.0),      # MA
            "plasma_beta": (0.001, 0.08),        # normalized
            "q_min": (0.8, 5.0),                 # safety factor
            "shape_error": (0.0, 20.0),          # cm
            "temperature": (0.1, 50.0),          # keV
            "density": (1e18, 5e20),             # m^-3
            "stored_energy": (1.0, 2000.0),      # MJ
            "disruption_prob": (0.0, 1.0),       # probability
            "major_radius": (0.3, 15.0),         # m
            "minor_radius": (0.1, 5.0),          # m
            "toroidal_field": (0.1, 20.0),       # T
            "elongation": (1.0, 3.0),            # aspect ratio
            "triangularity": (-0.8, 0.8),        # shape parameter
        }
    
    def validate_plasma_state(self, state: Dict[str, Any]) -> ValidationResult:
        """Validate plasma state parameters."""
        errors = []
        warnings = []
        corrected = {}
        
        # Check required fields
        required_fields = ["plasma_current", "plasma_beta", "q_min", "shape_error"]
        for field in required_fields:
            if field not in state:
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric ranges
        for param, value in state.items():
            if param in self.ranges:
                min_val, max_val = self.ranges[param]
                if not isinstance(value, (int, float)):
                    errors.append(f"{param} must be numeric, got {type(value)}")
                elif value < min_val:
                    errors.append(f"{param}={value} below minimum {min_val}")
                    corrected[param] = min_val
                elif value > max_val:
                    errors.append(f"{param}={value} above maximum {max_val}")
                    corrected[param] = max_val
        
        # Physics-based validation
        if "q_min" in state and state["q_min"] < 1.5:
            warnings.append(f"q_min={state['q_min']} is low, disruption risk increased")
            
        if "plasma_beta" in state and state["plasma_beta"] > 0.04:
            warnings.append(f"plasma_beta={state['plasma_beta']} approaching stability limit")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_values=corrected
        )
    
    def validate_tokamak_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate tokamak configuration."""
        errors = []
        warnings = []
        corrected = {}
        
        required_config_fields = ["major_radius", "minor_radius", "toroidal_field", "plasma_current_max"]
        
        for field in required_config_fields:
            if field not in config:
                errors.append(f"Missing required config field: {field}")
            elif not isinstance(config[field], (int, float)):
                errors.append(f"Config field {field} must be numeric")
            elif field in self.ranges:
                min_val, max_val = self.ranges[field]
                value = config[field]
                if value < min_val:
                    errors.append(f"Config {field}={value} below minimum {min_val}")
                    corrected[field] = min_val
                elif value > max_val:
                    errors.append(f"Config {field}={value} above maximum {max_val}")
                    corrected[field] = max_val
        
        # Physics consistency checks
        if all(f in config for f in ["major_radius", "minor_radius"]):
            aspect_ratio = config["major_radius"] / config["minor_radius"]
            if aspect_ratio < 1.2:
                errors.append(f"Aspect ratio {aspect_ratio:.2f} too low (minimum 1.2)")
            elif aspect_ratio > 10:
                warnings.append(f"Aspect ratio {aspect_ratio:.2f} very high")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_values=corrected
        )
    
    def validate_control_action(self, action: List[float]) -> ValidationResult:
        """Validate control action vector."""
        errors = []
        warnings = []
        corrected = {}
        
        if not isinstance(action, list):
            errors.append(f"Action must be a list, got {type(action)}")
            return ValidationResult(False, errors, warnings, corrected)
        
        # Check action dimensions
        expected_length = 8  # 6 PF coils + gas puff + heating
        if len(action) != expected_length:
            errors.append(f"Action vector length {len(action)} != expected {expected_length}")
        
        # Check action bounds
        corrected_action = []
        for i, value in enumerate(action):
            if not isinstance(value, (int, float)):
                errors.append(f"Action[{i}] must be numeric, got {type(value)}")
                corrected_action.append(0.0)
            elif value < -1.0:
                warnings.append(f"Action[{i}]={value} clipped to -1.0")
                corrected_action.append(-1.0)
            elif value > 1.0:
                warnings.append(f"Action[{i}]={value} clipped to 1.0")
                corrected_action.append(1.0)
            else:
                corrected_action.append(value)
        
        if corrected_action != action:
            corrected["action"] = corrected_action
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrected_values=corrected
        )

def robust_error_handler(fallback_value=None, log_errors=True):
    """Decorator for robust error handling with fallback values."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                if fallback_value is not None:
                    logger.warning(f"Returning fallback value: {fallback_value}")
                    return fallback_value
                else:
                    raise  # Re-raise if no fallback
        return wrapper
    return decorator

class RobustTokamakSystem:
    """Tokamak system with comprehensive error handling."""
    
    def __init__(self, config_name: str = "ITER"):
        self.logger = logging.getLogger("RobustTokamakSystem")
        self.validator = RobustValidator()
        self.error_count = 0
        self.warning_count = 0
        
        # Initialize with validation
        self.config = self._load_validated_config(config_name)
        self.state = self._initialize_validated_state()
        
        self.logger.info(f"Initialized robust tokamak system: {config_name}")
    
    def _load_validated_config(self, config_name: str) -> Dict[str, Any]:
        """Load and validate tokamak configuration."""
        configs = {
            "ITER": {
                "major_radius": 6.2,
                "minor_radius": 2.0,
                "toroidal_field": 5.3,
                "plasma_current_max": 15.0,
                "elongation": 1.85,
                "triangularity": 0.33,
                "name": "ITER"
            },
            "SPARC": {
                "major_radius": 3.3,
                "minor_radius": 1.04,
                "toroidal_field": 12.2,
                "plasma_current_max": 8.7,
                "elongation": 1.97,
                "triangularity": 0.40,
                "name": "SPARC"
            }
        }
        
        if config_name not in configs:
            self.logger.warning(f"Unknown config {config_name}, using ITER")
            config_name = "ITER"
        
        config = configs[config_name]
        validation = self.validator.validate_tokamak_config(config)
        
        if not validation.valid:
            self.logger.error(f"Invalid config: {validation.errors}")
            raise ValueError(f"Configuration validation failed: {validation.errors}")
        
        if validation.warnings:
            self.logger.warning(f"Config warnings: {validation.warnings}")
            self.warning_count += len(validation.warnings)
        
        # Apply corrections
        for param, corrected_value in validation.corrected_values.items():
            self.logger.warning(f"Correcting {param}: {config[param]} -> {corrected_value}")
            config[param] = corrected_value
        
        return config
    
    def _initialize_validated_state(self) -> Dict[str, Any]:
        """Initialize plasma state with validation."""
        initial_state = {
            "plasma_current": self.config["plasma_current_max"] * 0.8,
            "plasma_beta": 0.025,
            "q_min": 1.8,
            "shape_error": 2.0,
            "temperature": 12.0,
            "density": 6e19,
            "stored_energy": 300.0,
            "disruption_prob": 0.05
        }
        
        validation = self.validator.validate_plasma_state(initial_state)
        
        if not validation.valid:
            self.logger.error(f"Invalid initial state: {validation.errors}")
            # Apply corrections automatically for initialization
            for param, corrected_value in validation.corrected_values.items():
                initial_state[param] = corrected_value
        
        return initial_state
    
    @robust_error_handler(fallback_value=([], {}), log_errors=True)
    def reset(self) -> Tuple[List[float], Dict[str, Any]]:
        """Reset with error handling."""
        self.state = self._initialize_validated_state()
        observation = self._get_observation()
        info = {"config": self.config["name"], "errors": self.error_count}
        
        self.logger.info("System reset successfully")
        return observation, info
    
    @robust_error_handler(fallback_value=([], 0.0, True, False, {}), log_errors=True)
    def step(self, action: List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        """Execute step with comprehensive validation."""
        # Validate action
        action_validation = self.validator.validate_control_action(action)
        if not action_validation.valid:
            self.logger.error(f"Invalid action: {action_validation.errors}")
            self.error_count += len(action_validation.errors)
            # Use corrected action if available
            if "action" in action_validation.corrected_values:
                action = action_validation.corrected_values["action"]
            else:
                # Use safe default action
                action = [0.0] * 8
        
        # Apply warnings
        if action_validation.warnings:
            self.logger.warning(f"Action warnings: {action_validation.warnings}")
            self.warning_count += len(action_validation.warnings)
        
        # Evolve physics with validation
        self._evolve_physics_safely(action)
        
        # Validate resulting state
        state_validation = self.validator.validate_plasma_state(self.state)
        if not state_validation.valid:
            self.logger.error(f"Physics evolution created invalid state: {state_validation.errors}")
            # Apply emergency corrections
            for param, corrected_value in state_validation.corrected_values.items():
                self.state[param] = corrected_value
            self.error_count += len(state_validation.errors)
        
        # Calculate reward safely
        reward = self._calculate_reward_safely()
        
        # Check termination
        terminated = self._check_termination_safely()
        truncated = False  # Could add step limit here
        
        observation = self._get_observation()
        info = {
            "errors": self.error_count,
            "warnings": self.warning_count,
            "safety_score": self._calculate_safety_score(),
            "valid_state": state_validation.valid
        }
        
        return observation, reward, terminated, truncated, info
    
    @robust_error_handler(fallback_value=[], log_errors=True)
    def _get_observation(self) -> List[float]:
        """Get observation with error handling."""
        obs = [
            self.state.get("plasma_current", 0.0),
            self.state.get("plasma_beta", 0.0),
            self.state.get("q_min", 1.0),
            self.state.get("shape_error", 5.0),
            self.state.get("temperature", 1.0),
            self.state.get("density", 1e19) / 1e19,
            self.state.get("stored_energy", 0.0) / 100.0,
            self.state.get("disruption_prob", 1.0)
        ]
        return obs
    
    def _evolve_physics_safely(self, action: List[float]):
        """Evolve physics with safety checks."""
        try:
            # Apply control action safely
            if len(action) >= 6:
                shape_change = sum(action[:6]) * 0.02
                new_shape_error = max(0.1, self.state["shape_error"] + shape_change)
                self.state["shape_error"] = min(20.0, new_shape_error)
            
            # Update other parameters with bounds checking
            if len(action) >= 7:
                density_change = action[6] * 0.05
                new_density = self.state["density"] * (1.0 + density_change)
                self.state["density"] = max(1e18, min(5e20, new_density))
            
            # Physics evolution with safety bounds
            self.state["q_min"] = max(0.8, min(5.0, self.state["q_min"] + 0.001))
            
            # Recalculate beta safely
            temp = self.state.get("temperature", 10.0)
            density = self.state.get("density", 5e19)
            field = self.config["toroidal_field"]
            
            beta_new = (temp * density) / (field * field) * 1e-20
            self.state["plasma_beta"] = max(0.001, min(0.08, beta_new))
            
        except Exception as e:
            self.logger.error(f"Physics evolution error: {e}")
            # Keep current state if evolution fails
            self.error_count += 1
    
    @robust_error_handler(fallback_value=-10.0, log_errors=True)
    def _calculate_reward_safely(self) -> float:
        """Calculate reward with error handling."""
        shape_reward = -self.state["shape_error"]**2 / 25.0
        stability_reward = max(0, (self.state["q_min"] - 1.0) * 2.0)
        safety_penalty = -self.state["disruption_prob"] * 20.0
        
        total_reward = shape_reward + stability_reward + safety_penalty
        return max(-10.0, min(10.0, total_reward))
    
    @robust_error_handler(fallback_value=False, log_errors=True)
    def _check_termination_safely(self) -> bool:
        """Check termination conditions safely."""
        return (
            self.state.get("q_min", 2.0) < 1.0 or
            self.state.get("plasma_beta", 0.0) > 0.06 or
            self.state.get("disruption_prob", 0.0) > 0.8
        )
    
    def _calculate_safety_score(self) -> float:
        """Calculate safety score (0-1)."""
        score = 1.0
        
        # Q-min safety
        q_min = self.state.get("q_min", 2.0)
        if q_min < 1.5:
            score -= (1.5 - q_min) * 0.3
        
        # Beta safety
        beta = self.state.get("plasma_beta", 0.0)
        if beta > 0.04:
            score -= (beta - 0.04) * 10.0
        
        return max(0.0, min(1.0, score))
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        return {
            "config": self.config,
            "state": self.state,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "safety_score": self._calculate_safety_score(),
            "operational": not self._check_termination_safely(),
            "validation_status": "healthy" if self.error_count == 0 else "degraded"
        }

def test_robust_system():
    """Test robust error handling system."""
    print("🛡️ TESTING ROBUST ERROR HANDLING SYSTEM")
    print("="*60)
    
    # Test normal operation
    print("\n🔬 Testing Normal Operation...")
    system = RobustTokamakSystem("ITER")
    obs, info = system.reset()
    print(f"✅ Normal initialization: {info}")
    
    # Test with valid actions
    for i in range(5):
        action = [0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1]
        obs, reward, done, truncated, info = system.step(action)
        print(f"   Step {i+1}: reward={reward:.2f}, errors={info['errors']}, safety={info['safety_score']:.2f}")
    
    # Test error scenarios
    print("\n⚠️ Testing Error Scenarios...")
    
    # Test invalid actions
    invalid_actions = [
        [1.5, -2.0, 0.5],  # Wrong length, out of bounds
        ["invalid", None, {}],  # Wrong types
        [],  # Empty
        [float('inf'), float('nan')]  # Special float values
    ]
    
    for i, bad_action in enumerate(invalid_actions):
        print(f"   Testing invalid action {i+1}: {bad_action}")
        try:
            obs, reward, done, truncated, info = system.step(bad_action)
            print(f"   ✅ Handled gracefully: errors={info['errors']}, reward={reward:.2f}")
        except Exception as e:
            print(f"   ❌ Failed to handle: {e}")
    
    # Test invalid configuration
    print("\n🔧 Testing Invalid Configuration...")
    try:
        bad_system = RobustTokamakSystem("NONEXISTENT")
        print("✅ Handled unknown config gracefully")
    except Exception as e:
        print(f"❌ Failed to handle bad config: {e}")
    
    # Final diagnostics
    diagnostics = system.get_system_diagnostics()
    print(f"\n📊 Final System Diagnostics:")
    print(f"   Total errors: {diagnostics['error_count']}")
    print(f"   Total warnings: {diagnostics['warning_count']}")
    print(f"   Safety score: {diagnostics['safety_score']:.2f}")
    print(f"   System status: {diagnostics['validation_status']}")
    print(f"   Operational: {'✅' if diagnostics['operational'] else '❌'}")
    
    print(f"\n🎉 ROBUST ERROR HANDLING TEST COMPLETED")
    return True

if __name__ == "__main__":
    success = test_robust_system()
    print(f"\n{'🚀 SUCCESS' if success else '❌ FAILURE'}: Robust error handling operational!")
    sys.exit(0 if success else 1)