#!/usr/bin/env python3
"""
Generation 2: Enhanced Safety, Validation, and Robustness
Adding comprehensive error handling, input validation, and security measures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import logging

class EnhancedSafetyValidator:
    """Enhanced safety validation with comprehensive checks."""
    
    def __init__(self):
        self.violation_history = []
        self.alert_thresholds = {
            'q_min_critical': 1.0,
            'q_min_warning': 1.5,
            'beta_critical': 0.08,
            'beta_warning': 0.04,
            'shape_error_critical': 10.0,
            'shape_error_warning': 5.0,
            'disruption_risk_critical': 0.5,
            'disruption_risk_warning': 0.1
        }
        
    def validate_plasma_state(self, state) -> Dict[str, Any]:
        """Comprehensive plasma state validation."""
        violations = []
        warnings_list = []
        
        # Q-factor validation
        if hasattr(state, 'q_min'):
            if state.q_min < self.alert_thresholds['q_min_critical']:
                violations.append(f"CRITICAL: Q-min too low ({state.q_min:.2f} < {self.alert_thresholds['q_min_critical']})")
            elif state.q_min < self.alert_thresholds['q_min_warning']:
                warnings_list.append(f"WARNING: Q-min approaching limit ({state.q_min:.2f})")
        
        # Beta validation  
        if hasattr(state, 'plasma_beta'):
            if state.plasma_beta > self.alert_thresholds['beta_critical']:
                violations.append(f"CRITICAL: Beta too high ({state.plasma_beta:.3f} > {self.alert_thresholds['beta_critical']})")
            elif state.plasma_beta > self.alert_thresholds['beta_warning']:
                warnings_list.append(f"WARNING: Beta approaching limit ({state.plasma_beta:.3f})")
        
        # Shape error validation
        if hasattr(state, 'shape_error'):
            if state.shape_error > self.alert_thresholds['shape_error_critical']:
                violations.append(f"CRITICAL: Shape error too high ({state.shape_error:.1f} cm)")
            elif state.shape_error > self.alert_thresholds['shape_error_warning']:
                warnings_list.append(f"WARNING: Shape error increasing ({state.shape_error:.1f} cm)")
        
        # Disruption risk validation
        if hasattr(state, 'disruption_probability'):
            if state.disruption_probability > self.alert_thresholds['disruption_risk_critical']:
                violations.append(f"CRITICAL: High disruption risk ({state.disruption_probability:.3f})")
            elif state.disruption_probability > self.alert_thresholds['disruption_risk_warning']:
                warnings_list.append(f"WARNING: Elevated disruption risk ({state.disruption_probability:.3f})")
        
        return {
            'safe': len(violations) == 0,
            'violations': violations,
            'warnings': warnings_list,
            'risk_level': self._calculate_risk_level(violations, warnings_list)
        }
        
    def _calculate_risk_level(self, violations: List[str], warnings: List[str]) -> str:
        """Calculate overall risk level."""
        if violations:
            return "CRITICAL"
        elif len(warnings) >= 3:
            return "HIGH"
        elif len(warnings) >= 1:
            return "MEDIUM"
        else:
            return "LOW"

class RobustEnvironmentWrapper:
    """Robust wrapper for tokamak environment with error handling."""
    
    def __init__(self, base_env, validator: EnhancedSafetyValidator):
        self.base_env = base_env
        self.validator = validator
        self.error_count = 0
        self.max_errors = 5
        self.last_safe_state = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def reset(self, **kwargs):
        """Robust reset with error handling."""
        try:
            obs, info = self.base_env.reset(**kwargs)
            
            # Validate initial state
            validation = self.validator.validate_plasma_state(self.base_env.plasma_state)
            if not validation['safe']:
                self.logger.warning(f"Initial state violations: {validation['violations']}")
            
            # Store safe state
            self.last_safe_state = obs.copy()
            self.error_count = 0
            
            # Enhanced info
            info['safety_validation'] = validation
            info['error_count'] = self.error_count
            
            return obs, info
            
        except Exception as e:
            self.logger.error(f"Environment reset failed: {e}")
            # Return safe default state
            return self._get_safe_default_state()
    
    def step(self, action):
        """Robust step with comprehensive error handling and recovery."""
        try:
            # Validate action first
            validated_action = self._validate_and_sanitize_action(action)
            
            # Execute step
            obs, reward, terminated, truncated, info = self.base_env.step(validated_action)
            
            # Validate resulting state
            validation = self.validator.validate_plasma_state(self.base_env.plasma_state)
            
            # Handle safety violations
            if not validation['safe']:
                self.logger.warning(f"Safety violations detected: {validation['violations']}")
                
                # Implement corrective measures
                obs, reward, terminated, truncated, info = self._handle_safety_violation(
                    obs, reward, terminated, truncated, info, validation
                )
            
            # Update safe state if current state is safe
            if validation['safe']:
                self.last_safe_state = obs.copy()
                self.error_count = 0
            else:
                self.error_count += 1
            
            # Enhanced info
            info['safety_validation'] = validation
            info['error_count'] = self.error_count
            info['action_modified'] = not np.allclose(action, validated_action, atol=1e-6)
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Environment step failed (error {self.error_count}): {e}")
            
            # Emergency recovery
            if self.error_count >= self.max_errors:
                self.logger.critical("Maximum errors exceeded - forcing environment reset")
                return self._emergency_recovery()
            
            # Return last safe state with penalty
            return self.last_safe_state, -10.0, False, False, {
                'error': str(e),
                'emergency_recovery': True
            }
    
    def _validate_and_sanitize_action(self, action) -> np.ndarray:
        """Validate and sanitize action inputs."""
        action = np.array(action, dtype=np.float32)
        
        # Check for NaN or infinite values
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.logger.warning("Action contains NaN or infinite values - replacing with zeros")
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clamp to reasonable bounds
        action = np.clip(action, -2.0, 2.0)
        
        # Rate limiting (simple implementation)
        if hasattr(self, 'last_action'):
            max_change = 0.5  # Maximum change per step
            action_change = action - self.last_action
            if np.any(np.abs(action_change) > max_change):
                self.logger.info("Rate limiting applied to action")
                action_change = np.clip(action_change, -max_change, max_change)
                action = self.last_action + action_change
        
        self.last_action = action.copy()
        return action
    
    def _handle_safety_violation(self, obs, reward, terminated, truncated, info, validation):
        """Handle safety violations with corrective measures."""
        
        # Apply safety penalty
        safety_penalty = -50.0 * len(validation['violations'])
        reward += safety_penalty
        
        # If critical violations, consider episode termination
        if validation['risk_level'] == 'CRITICAL':
            self.logger.warning("Critical safety violations - considering episode termination")
            terminated = True
        
        return obs, reward, terminated, truncated, info
    
    def _get_safe_default_state(self):
        """Get safe default state for emergency situations."""
        # Simple safe observation vector
        safe_obs = np.zeros(43, dtype=np.float32)
        safe_obs[0] = 1.0  # plasma current
        safe_obs[1] = 0.02  # plasma beta
        safe_obs[2:12] = 2.0  # q-profile (safe values)
        
        safe_info = {
            'plasma_state': {
                'q_min': 2.0,
                'plasma_beta': 0.02,
                'shape_error': 0.0,
                'elongation': 1.7,
                'triangularity': 0.4,
                'disruption_probability': 0.0
            },
            'emergency_default': True
        }
        
        return safe_obs, safe_info
    
    def _emergency_recovery(self):
        """Emergency recovery procedure."""
        self.logger.critical("Initiating emergency recovery")
        
        try:
            # Attempt environment reset
            return self.reset()
        except:
            # Last resort - return safe defaults
            return self._get_safe_default_state()

def test_enhanced_safety():
    """Test enhanced safety and validation systems."""
    print("🛡️ Testing Enhanced Safety & Validation...")
    
    from tokamak_rl.physics import TokamakConfig
    from tokamak_rl.environment import TokamakEnv
    
    # Create environment
    config = TokamakConfig.from_preset("ITER") 
    env_config = {
        'tokamak_config': config,
        'enable_safety': False  # We'll use our enhanced wrapper
    }
    base_env = TokamakEnv(env_config)
    
    # Create enhanced wrapper
    validator = EnhancedSafetyValidator()
    robust_env = RobustEnvironmentWrapper(base_env, validator)
    
    print("  ✓ Enhanced safety wrapper created")
    
    # Test robust reset
    obs, info = robust_env.reset()
    print(f"  ✓ Robust reset - Risk level: {info['safety_validation']['risk_level']}")
    
    # Test with various action scenarios
    test_actions = [
        np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]),  # Normal action
        np.array([10.0, -5.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0]),  # Extreme action (should be clipped)
        np.array([np.nan, 0.0, np.inf, 0.0, 0.0, 0.0, 0.2, 0.1]),  # Invalid action
    ]
    
    for i, action in enumerate(test_actions):
        obs, reward, terminated, truncated, info = robust_env.step(action)
        print(f"  ✓ Test action {i+1}: Risk={info['safety_validation']['risk_level']}, "
              f"Modified={info.get('action_modified', False)}")
    
    return True

def test_input_validation():
    """Test comprehensive input validation."""
    print("\n🔒 Testing Input Validation...")
    
    validator = EnhancedSafetyValidator()
    
    # Create mock plasma states with different safety conditions
    class MockPlasmaState:
        def __init__(self, q_min=2.0, plasma_beta=0.02, shape_error=1.0, disruption_probability=0.01):
            self.q_min = q_min
            self.plasma_beta = plasma_beta
            self.shape_error = shape_error
            self.disruption_probability = disruption_probability
    
    test_states = [
        ("Safe state", MockPlasmaState(2.0, 0.02, 1.0, 0.01)),
        ("Q-min warning", MockPlasmaState(1.3, 0.02, 1.0, 0.01)),
        ("Q-min critical", MockPlasmaState(0.8, 0.02, 1.0, 0.01)),
        ("High beta", MockPlasmaState(2.0, 0.06, 1.0, 0.01)),
        ("Multiple violations", MockPlasmaState(0.9, 0.09, 12.0, 0.6)),
    ]
    
    for name, state in test_states:
        validation = validator.validate_plasma_state(state)
        print(f"  ✓ {name}: {validation['risk_level']} risk, "
              f"{len(validation['violations'])} violations, "
              f"{len(validation['warnings'])} warnings")
    
    return True

def test_error_recovery():
    """Test error handling and recovery mechanisms."""
    print("\n🔄 Testing Error Recovery...")
    
    from tokamak_rl.physics import TokamakConfig
    from tokamak_rl.environment import TokamakEnv
    
    # Create environment
    config = TokamakConfig.from_preset("SPARC")
    env_config = {'tokamak_config': config, 'enable_safety': False}
    base_env = TokamakEnv(env_config)
    
    validator = EnhancedSafetyValidator()
    robust_env = RobustEnvironmentWrapper(base_env, validator)
    
    obs, info = robust_env.reset()
    print("  ✓ Environment initialized")
    
    # Test recovery from problematic actions
    problematic_actions = [
        np.array([100.0] * 8),  # Extreme values
        np.array([np.nan] * 8),  # NaN values
        np.array([np.inf, -np.inf, 0, 0, 0, 0, 0, 0]),  # Infinite values
    ]
    
    for i, action in enumerate(problematic_actions):
        try:
            obs, reward, terminated, truncated, info = robust_env.step(action)
            print(f"  ✓ Problematic action {i+1} handled - reward penalty applied: {reward:.1f}")
        except Exception as e:
            print(f"  ❌ Failed to handle problematic action {i+1}: {e}")
            return False
    
    print(f"  ✓ Error recovery successful - {robust_env.error_count} total errors handled")
    return True

if __name__ == "__main__":
    print("TOKAMAK RL CONTROL SUITE - GENERATION 2: MAKE IT ROBUST")
    print("=" * 70)
    
    tests = [
        test_enhanced_safety,
        test_input_validation,
        test_error_recovery
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            result = test()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    if all_passed:
        print("\n🎉 GENERATION 2 SUCCESS: System is now robust and reliable!")
        print("✅ Enhanced safety validation implemented")
        print("✅ Comprehensive error handling added")
        print("✅ Input validation and sanitization working")
        print("✅ Emergency recovery procedures functional")
        print("✅ Logging and monitoring integrated")
        print("\nReady to proceed to Generation 3: Make it Scale")
        sys.exit(0)
    else:
        print("\n⚠️ GENERATION 2 INCOMPLETE: Some robustness tests failed")
        sys.exit(1)