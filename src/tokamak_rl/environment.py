"""
Tokamak environment factory and core components.

This module provides the main interface for creating tokamak RL environments.
"""

try:
    import gymnasium as gym
except ImportError:
    # Fallback gym-like interface
    class gym:
        class Env:
            def __init__(self):
                self.np_random = None
                
            def reset(self, seed=None, options=None):
                return None, {}
                
            def step(self, action):
                return None, 0, False, False, {}
                
        class spaces:
            class Box:
                def __init__(self, low, high, shape=None, dtype=float):
                    self.low = low
                    self.high = high 
                    self.shape = shape
                    self.dtype = dtype

try:
    import numpy as np
except ImportError:
    # Use same fallback as physics module
    import math
    
    class np:
        @staticmethod
        def array(x):
            return list(x) if hasattr(x, '__iter__') else [x]
        
        @staticmethod
        def concatenate(arrays):
            result = []
            for arr in arrays:
                if hasattr(arr, '__iter__'):
                    result.extend(arr)
                else:
                    result.append(arr)
            return result
        
        @staticmethod
        def sum(arr):
            return sum(arr)
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr)
        
        @staticmethod
        def max(arr):
            return max(arr)
        
        @staticmethod
        def clip(arr, min_val, max_val):
            if hasattr(arr, '__iter__'):
                return [max(min_val, min(max_val, x)) for x in arr]
            else:
                return max(min_val, min(max_val, arr))
        
        float32 = float
        ndarray = list  # Type alias for compatibility

from typing import Dict, Any, Optional, Tuple
from .physics import TokamakConfig, PlasmaState, GradShafranovSolver
from .safety import SafetyShield, create_safety_system


def make_tokamak_env(
    tokamak_config: str = "ITER",
    control_frequency: int = 100,
    safety_factor: float = 1.2,
    enable_safety: bool = True,
    **kwargs: Any
) -> gym.Env:
    """
    Create a tokamak plasma control environment.
    
    Args:
        tokamak_config: Tokamak configuration ("ITER", "SPARC", "NSTX", "DIII-D")
        control_frequency: Control loop frequency in Hz
        safety_factor: Safety margin for disruption prevention
        enable_safety: Whether to enable safety shield
        **kwargs: Additional environment parameters
        
    Returns:
        Configured tokamak environment
    """
    # Get tokamak configuration
    if isinstance(tokamak_config, str):
        config = TokamakConfig.from_preset(tokamak_config)
    else:
        config = tokamak_config
        
    config.control_frequency = control_frequency
    
    # Create environment
    env_config = {
        'tokamak_config': config,
        'enable_safety': enable_safety,
        'safety_factor': safety_factor,
        **kwargs
    }
    
    return TokamakEnv(env_config)


class TokamakEnv(gym.Env):
    """
    Tokamak plasma control environment.
    
    This environment simulates tokamak plasma equilibrium using the 
    Grad-Shafranov equation and provides RL interfaces for plasma shape control.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tokamak environment with given configuration."""
        super().__init__()
        self.env_config = config
        self.tokamak_config = config['tokamak_config']
        
        # Initialize physics solver
        self.physics_solver = GradShafranovSolver(self.tokamak_config)
        self.plasma_state = PlasmaState(self.tokamak_config)
        
        # Initialize safety system
        if config.get('enable_safety', True):
            self.safety_shield = create_safety_system(self.tokamak_config)
        else:
            self.safety_shield = None
            
        # Define observation and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 1000
        self.target_shape = self._get_target_shape()
        
        # Metrics tracking
        self.episode_metrics = {
            'shape_error': [],
            'disruption_risk': [],
            'safety_violations': [],
            'control_effort': []
        }
        
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        # Observation space: 45-dimensional vector
        obs_low = np.array([
            0.0,    # plasma_current (MA)
            0.0,    # plasma_beta
            *([0.5] * 10),   # q_profile
            *([0.5] * 6),    # shape_parameters 
            *([-10.0] * 6),  # pf_coil_currents (MA)
            *([0.0] * 10),   # density_profile
            *([0.0] * 5),    # temperature_profile
            -10.0   # shape_error
        ])
        
        obs_high = np.array([
            20.0,   # plasma_current (MA)
            0.1,    # plasma_beta
            *([10.0] * 10),  # q_profile
            *([3.0] * 6),    # shape_parameters
            *([10.0] * 6),   # pf_coil_currents (MA)
            *([2e20] * 10),  # density_profile
            *([50.0] * 5),   # temperature_profile
            10.0    # shape_error
        ])
        
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        
        # Action space: 8-dimensional continuous
        # [pf_coil_adjustments(6), gas_puff_rate(1), heating_power(1)]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0] * 6 + [0.0, 0.0]),
            high=np.array([1.0] * 6 + [1.0, 1.0]),
            dtype=np.float32
        )
        
    def _get_target_shape(self) -> Dict[str, float]:
        """Get target plasma shape parameters."""
        return {
            'elongation': self.tokamak_config.elongation,
            'triangularity': self.tokamak_config.triangularity,
            'q_min': 1.5,
            'beta_target': 0.025
        }
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial plasma state."""
        try:
            super().reset(seed=seed)
        except AttributeError:
            # Handle fallback gym implementation
            self.np_random = None
        
        # Reset plasma state
        self.plasma_state.reset()
        
        # Reset safety system
        if self.safety_shield:
            self.safety_shield.reset()
            
        # Reset episode tracking
        self.episode_step = 0
        self.episode_metrics = {
            'shape_error': [],
            'disruption_risk': [],
            'safety_violations': [],
            'control_effort': []
        }
        
        # Add small random perturbations for training diversity
        if self.np_random is not None:
            perturbation = self.np_random.normal(0, 0.01, size=6)
            self.plasma_state.pf_coil_currents += perturbation
            
        observation = self.plasma_state.get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute control action and return next state."""
        action = np.array(action, dtype=np.float32)
        
        # Apply safety shield if enabled
        original_action = action.copy()
        safety_info = {}
        
        if self.safety_shield:
            action, safety_info = self.safety_shield.filter_action(action, self.plasma_state)
            
        # Execute physics step
        pf_adjustments = action[:6]  # PF coil current adjustments
        gas_puff = action[6]         # Gas puff rate
        heating = action[7]          # Heating power
        
        # Apply PF coil adjustments (normalized to actual currents)
        max_current = 2.0  # MA
        current_currents = np.array(self.plasma_state.pf_coil_currents)
        pf_adjustments_clipped = np.array(pf_adjustments)
        
        # Ensure arrays have compatible shapes
        if len(current_currents) != len(pf_adjustments_clipped):
            # Resize to minimum size
            min_size = min(len(current_currents), len(pf_adjustments_clipped))
            current_currents = current_currents[:min_size]
            pf_adjustments_clipped = pf_adjustments_clipped[:min_size]
            
        pf_currents = current_currents + pf_adjustments_clipped * max_current * 0.1
        pf_currents = np.clip(pf_currents, -max_current, max_current)
        
        # Solve new equilibrium
        self.plasma_state = self.physics_solver.solve_equilibrium(self.plasma_state, pf_currents)
        
        # Apply auxiliary heating and gas puff effects (simplified)
        heating_effect = heating * 0.02  # Increase temperature
        self.plasma_state.temperature_profile *= (1 + heating_effect)
        
        gas_effect = gas_puff * 0.05  # Increase density
        self.plasma_state.density_profile *= (1 + gas_effect)
        
        # Compute reward
        reward = self._compute_reward(action, original_action, safety_info)
        
        # Check termination conditions
        self.episode_step += 1
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_episode_steps
        
        # Get next observation
        observation = self.plasma_state.get_observation()
        info = self._get_info(safety_info, action, original_action)
        
        # Track metrics
        self._update_metrics(safety_info, action)
        
        return observation, reward, terminated, truncated, info
        
    def _compute_reward(self, action: np.ndarray, original_action: np.ndarray, 
                       safety_info: Dict) -> float:
        """Compute reward for current state and action."""
        reward = 0.0
        
        # Shape accuracy reward (primary objective)
        shape_error = self.plasma_state.shape_error
        shape_reward = -shape_error**2 * 0.01  # Quadratic penalty
        reward += shape_reward
        
        # Safety factor reward (stability)
        q_min = self.plasma_state.q_min
        if q_min > 1.5:
            stability_reward = min(q_min - 1.5, 2.0)  # Bonus for high q
        else:
            stability_reward = -10.0 * (1.5 - q_min)**2  # Penalty for low q
        reward += stability_reward
        
        # Beta optimization (confinement)
        beta_target = self.target_shape['beta_target']
        beta_error = abs(self.plasma_state.plasma_beta - beta_target)
        beta_reward = -beta_error * 50.0
        reward += beta_reward
        
        # Control effort penalty
        control_effort = np.sum(action[:6]**2)  # PF coil effort
        control_penalty = -0.01 * control_effort
        reward += control_penalty
        
        # Safety penalties
        if safety_info.get('action_modified', False):
            reward -= 1.0  # Small penalty for safety interventions
            
        violations = safety_info.get('violations', [])
        violation_penalty = -len(violations) * 0.5
        reward += violation_penalty
        
        # Disruption risk penalty
        disruption_risk = safety_info.get('disruption_risk', 0.0)
        if disruption_risk > 0.1:
            reward -= 10.0 * disruption_risk
            
        # Emergency mode penalty
        if safety_info.get('emergency_mode', False):
            reward -= 20.0
            
        return float(reward)
        
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to unsafe conditions."""
        # Terminate on disruption conditions
        if self.plasma_state.q_min < 1.0:
            return True
            
        if self.plasma_state.plasma_beta > 0.08:  # Well above limits
            return True
            
        if self.plasma_state.disruption_probability > 0.5:
            return True
            
        if self.plasma_state.shape_error > 20.0:  # Very large shape error
            return True
            
        return False
        
    def _get_info(self, safety_info: Optional[Dict] = None, 
                  action: Optional[np.ndarray] = None,
                  original_action: Optional[np.ndarray] = None) -> Dict:
        """Get environment info dictionary."""
        info = {
            'plasma_state': {
                'q_min': self.plasma_state.q_min,
                'beta': self.plasma_state.plasma_beta,
                'shape_error': self.plasma_state.shape_error,
                'elongation': self.plasma_state.elongation,
                'triangularity': self.plasma_state.triangularity,
                'disruption_probability': self.plasma_state.disruption_probability
            },
            'episode_step': self.episode_step,
            'target_shape': self.target_shape
        }
        
        if safety_info:
            info['safety'] = safety_info
            
        if action is not None:
            info['action_taken'] = action
            
        if original_action is not None:
            info['action_requested'] = original_action
            
        return info
        
    def _update_metrics(self, safety_info: Dict, action: np.ndarray) -> None:
        """Update episode metrics."""
        self.episode_metrics['shape_error'].append(self.plasma_state.shape_error)
        self.episode_metrics['disruption_risk'].append(
            safety_info.get('disruption_risk', 0.0)
        )
        self.episode_metrics['safety_violations'].append(
            len(safety_info.get('violations', []))
        )
        self.episode_metrics['control_effort'].append(
            np.sum(action[:6]**2)
        )
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render current plasma state."""
        if mode == "human":
            print(f"Episode Step: {self.episode_step}")
            print(f"Shape Error: {self.plasma_state.shape_error:.2f} cm")
            print(f"Q-min: {self.plasma_state.q_min:.2f}")
            print(f"Beta: {self.plasma_state.plasma_beta:.3f}")
            print(f"Disruption Risk: {self.plasma_state.disruption_probability:.3f}")
            print("---")
        elif mode == "rgb_array":
            # In a full implementation, this would return a visual rendering
            # For now, return a placeholder image
            return np.zeros((400, 600, 3), dtype=np.uint8)
            
    def get_episode_metrics(self) -> Dict[str, float]:
        """Get summary metrics for completed episode."""
        if not self.episode_metrics['shape_error']:
            return {}
            
        return {
            'mean_shape_error': np.mean(self.episode_metrics['shape_error']),
            'final_shape_error': self.episode_metrics['shape_error'][-1],
            'max_disruption_risk': np.max(self.episode_metrics['disruption_risk']),
            'total_safety_violations': np.sum(self.episode_metrics['safety_violations']),
            'mean_control_effort': np.mean(self.episode_metrics['control_effort']),
            'episode_length': len(self.episode_metrics['shape_error'])
        }