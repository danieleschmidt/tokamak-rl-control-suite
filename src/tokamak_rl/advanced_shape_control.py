"""
Advanced Plasma Shape Control System
Implements enhanced trajectory planning and real-time adaptation for tokamak plasma control.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import gymnasium as gym
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlasmaTarget:
    """Target plasma configuration"""
    elongation: float = 1.8
    triangularity: float = 0.4
    major_radius: float = 6.2
    minor_radius: float = 2.0
    plasma_current: float = 15.0  # MA
    safety_factor_min: float = 1.5
    beta_normalized: float = 2.5

@dataclass
class ControlLimits:
    """Physical control system limits"""
    max_coil_current: float = 50000.0  # A
    max_current_rate: float = 1000.0   # A/s
    max_power: float = 100.0           # MW
    min_q_factor: float = 1.2
    max_density: float = 1.2e20        # m^-3

class ShapeTrajectoryPlanner:
    """Advanced trajectory planner for plasma shape evolution"""
    
    def __init__(self, target: PlasmaTarget, limits: ControlLimits):
        self.target = target
        self.limits = limits
        self.trajectory_length = 100
        self.optimization_horizon = 20
        
    def plan_trajectory(self, current_state: np.ndarray, 
                       time_horizon: float = 10.0) -> np.ndarray:
        """Plan optimal trajectory from current state to target"""
        
        # Define time points
        t = np.linspace(0, time_horizon, self.trajectory_length)
        
        # Current plasma parameters
        current_elongation = current_state[0] if len(current_state) > 0 else 1.0
        current_triangularity = current_state[1] if len(current_state) > 1 else 0.0
        
        # Smooth trajectory using splines
        elongation_traj = self._create_smooth_trajectory(
            current_elongation, self.target.elongation, t
        )
        triangularity_traj = self._create_smooth_trajectory(
            current_triangularity, self.target.triangularity, t
        )
        
        # Stack trajectories
        trajectory = np.column_stack([elongation_traj, triangularity_traj])
        
        # Add safety factor trajectory
        q_min_traj = np.full(len(t), self.target.safety_factor_min)
        trajectory = np.column_stack([trajectory, q_min_traj])
        
        return trajectory
    
    def _create_smooth_trajectory(self, start: float, end: float, 
                                 time_points: np.ndarray) -> np.ndarray:
        """Create smooth trajectory using cubic splines"""
        # Control points for smooth transition
        mid_time = time_points[-1] / 2
        control_times = [0, mid_time * 0.3, mid_time * 0.7, time_points[-1]]
        
        # Smooth acceleration/deceleration
        progress = np.array([0, 0.15, 0.85, 1.0])
        control_values = start + (end - start) * progress
        
        # Create spline
        spline = UnivariateSpline(control_times, control_values, s=0, k=3)
        return spline(time_points)

class AdaptiveShapeController(nn.Module):
    """Neural network based adaptive controller for plasma shape"""
    
    def __init__(self, input_dim: int = 45, hidden_dim: int = 256, 
                 output_dim: int = 8):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.shape_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3)  # elongation, triangularity, q_min
        )
        
        self.action_network = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through adaptive controller"""
        encoded = self.encoder(state)
        
        # Predict future plasma shape
        shape_pred = self.shape_predictor(encoded)
        
        # Generate control actions based on encoded state and predictions
        combined = torch.cat([encoded, shape_pred], dim=-1)
        actions = self.action_network(combined)
        
        return actions, shape_pred

class RealTimeShapeOptimizer:
    """Real-time optimization system for plasma shape control"""
    
    def __init__(self, controller: AdaptiveShapeController, 
                 planner: ShapeTrajectoryPlanner):
        self.controller = controller
        self.planner = planner
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=1e-4)
        self.loss_history = []
        
    def optimize_step(self, state: np.ndarray, target_shape: np.ndarray) -> np.ndarray:
        """Single optimization step for real-time control"""
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        target_tensor = torch.FloatTensor(target_shape)
        
        # Forward pass
        actions, shape_pred = self.controller(state_tensor)
        
        # Compute loss
        shape_loss = nn.MSELoss()(shape_pred.squeeze(), target_tensor)
        action_penalty = 0.01 * torch.sum(actions ** 2)  # Control effort penalty
        total_loss = shape_loss + action_penalty
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
        self.optimizer.step()
        
        # Store loss
        self.loss_history.append(total_loss.item())
        
        return actions.detach().numpy().squeeze()

class PlasmaShapeControlSuite:
    """Complete plasma shape control system integrating all components"""
    
    def __init__(self, target: Optional[PlasmaTarget] = None, 
                 limits: Optional[ControlLimits] = None):
        
        self.target = target or PlasmaTarget()
        self.limits = limits or ControlLimits()
        
        # Initialize components
        self.planner = ShapeTrajectoryPlanner(self.target, self.limits)
        self.controller = AdaptiveShapeController()
        self.optimizer = RealTimeShapeOptimizer(self.controller, self.planner)
        
        # State tracking
        self.current_state = None
        self.trajectory = None
        self.step_count = 0
        
        # Performance metrics
        self.metrics = {
            'shape_errors': [],
            'control_efforts': [],
            'safety_violations': [],
            'optimization_losses': []
        }
        
        logger.info("Advanced Plasma Shape Control Suite initialized")
    
    def reset(self, initial_state: np.ndarray) -> None:
        """Reset the control system with new initial state"""
        self.current_state = initial_state.copy()
        self.step_count = 0
        
        # Clear metrics
        for key in self.metrics:
            self.metrics[key].clear()
        
        # Plan new trajectory
        self.trajectory = self.planner.plan_trajectory(initial_state)
        
        logger.info("Shape control system reset with new trajectory")
    
    def control_step(self, observation: np.ndarray) -> Dict[str, Any]:
        """Execute single control step"""
        
        if self.trajectory is None:
            self.reset(observation)
        
        # Get current target from trajectory
        traj_index = min(self.step_count, len(self.trajectory) - 1)
        target_shape = self.trajectory[traj_index, :3]  # elongation, triangularity, q_min
        
        # Compute optimal control action
        action = self.optimizer.optimize_step(observation, target_shape)
        
        # Apply safety constraints
        action = self._apply_safety_constraints(action, observation)
        
        # Compute metrics
        shape_error = self._compute_shape_error(observation, target_shape)
        control_effort = np.sum(action ** 2)
        safety_violation = self._check_safety_violations(observation)
        
        # Update metrics
        self.metrics['shape_errors'].append(shape_error)
        self.metrics['control_efforts'].append(control_effort)
        self.metrics['safety_violations'].append(safety_violation)
        self.metrics['optimization_losses'].append(
            self.optimizer.loss_history[-1] if self.optimizer.loss_history else 0.0
        )
        
        # Update state
        self.current_state = observation.copy()
        self.step_count += 1
        
        return {
            'action': action,
            'target_shape': target_shape,
            'shape_error': shape_error,
            'control_effort': control_effort,
            'safety_violation': safety_violation,
            'trajectory_progress': min(traj_index / len(self.trajectory), 1.0)
        }
    
    def _apply_safety_constraints(self, action: np.ndarray, 
                                state: np.ndarray) -> np.ndarray:
        """Apply safety constraints to control actions"""
        
        # Clip actions to physical limits
        action = np.clip(action, -1.0, 1.0)
        
        # Check if action would violate safety factor constraint
        if len(state) > 10:  # Assuming q_profile starts at index 10
            q_min = np.min(state[10:20])  # Safety factor profile
            if q_min < self.limits.min_q_factor:
                # Reduce control action magnitude to preserve safety
                action *= 0.5
        
        return action
    
    def _compute_shape_error(self, state: np.ndarray, 
                           target: np.ndarray) -> float:
        """Compute current shape error from target"""
        
        if len(state) < 6:
            return 0.0
        
        # Extract current shape parameters (assuming they start at index 0)
        current_elongation = state[0] if len(state) > 0 else 1.0
        current_triangularity = state[1] if len(state) > 1 else 0.0
        
        # Compute normalized errors
        elongation_error = abs(current_elongation - target[0]) / target[0]
        triangularity_error = abs(current_triangularity - target[1]) / abs(target[1] + 1e-6)
        
        return np.sqrt(elongation_error**2 + triangularity_error**2)
    
    def _check_safety_violations(self, state: np.ndarray) -> bool:
        """Check for safety constraint violations"""
        
        violations = False
        
        # Check safety factor
        if len(state) > 10:
            q_min = np.min(state[10:20])
            if q_min < self.limits.min_q_factor:
                violations = True
        
        # Check density limit (assuming density at index 35)
        if len(state) > 35:
            max_density = np.max(state[35:45])
            if max_density > self.limits.max_density:
                violations = True
        
        return violations
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary statistics"""
        
        if not self.metrics['shape_errors']:
            return {}
        
        return {
            'avg_shape_error': np.mean(self.metrics['shape_errors']),
            'max_shape_error': np.max(self.metrics['shape_errors']),
            'avg_control_effort': np.mean(self.metrics['control_efforts']),
            'safety_violation_rate': np.mean(self.metrics['safety_violations']),
            'final_optimization_loss': self.metrics['optimization_losses'][-1] if self.metrics['optimization_losses'] else 0.0,
            'trajectory_completion': min(self.step_count / 100, 1.0) if self.trajectory is not None else 0.0
        }

def create_advanced_shape_control_system(config: Optional[Dict] = None) -> PlasmaShapeControlSuite:
    """Factory function to create advanced shape control system"""
    
    if config is None:
        config = {}
    
    # Create target configuration
    target = PlasmaTarget(
        elongation=config.get('target_elongation', 1.8),
        triangularity=config.get('target_triangularity', 0.4),
        plasma_current=config.get('target_current', 15.0),
        safety_factor_min=config.get('min_q_factor', 1.5)
    )
    
    # Create control limits
    limits = ControlLimits(
        max_coil_current=config.get('max_coil_current', 50000.0),
        max_current_rate=config.get('max_current_rate', 1000.0),
        min_q_factor=config.get('min_q_factor', 1.2)
    )
    
    return PlasmaShapeControlSuite(target, limits)

# Integration with existing tokamak environment
class EnhancedTokamakWrapper(gym.Wrapper):
    """Wrapper to integrate advanced shape control with existing environment"""
    
    def __init__(self, env: gym.Env, config: Optional[Dict] = None):
        super().__init__(env)
        self.shape_controller = create_advanced_shape_control_system(config)
        self.use_advanced_control = True
        
    def step(self, action):
        """Step with advanced shape control integration"""
        
        if self.use_advanced_control and hasattr(self, '_last_obs'):
            # Get enhanced control action
            control_result = self.shape_controller.control_step(self._last_obs)
            enhanced_action = control_result['action']
            
            # Blend with original action (weighted combination)
            alpha = 0.7  # Weight for advanced controller
            action = alpha * enhanced_action + (1 - alpha) * action
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        
        # Add advanced control metrics to info
        if self.use_advanced_control:
            info['advanced_control'] = self.shape_controller.get_performance_summary()
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset with shape controller initialization"""
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        
        if self.use_advanced_control:
            self.shape_controller.reset(obs)
        
        return obs, info

if __name__ == "__main__":
    # Demonstration of advanced shape control system
    
    # Create system with custom configuration
    config = {
        'target_elongation': 1.9,
        'target_triangularity': 0.45,
        'target_current': 16.0,
        'min_q_factor': 1.3
    }
    
    control_system = create_advanced_shape_control_system(config)
    
    # Simulate plasma control scenario
    np.random.seed(42)
    initial_state = np.random.randn(45)  # 45-dimensional observation
    control_system.reset(initial_state)
    
    print("Advanced Plasma Shape Control Demonstration")
    print("=" * 50)
    
    # Run control simulation
    for step in range(50):
        # Simulate evolving plasma state
        noise = 0.1 * np.random.randn(45)
        current_obs = initial_state + noise * (step + 1) * 0.1
        
        # Execute control step
        result = control_system.control_step(current_obs)
        
        if step % 10 == 0:
            print(f"Step {step:2d}: Shape Error = {result['shape_error']:.4f}, "
                  f"Control Effort = {result['control_effort']:.4f}, "
                  f"Progress = {result['trajectory_progress']:.2%}")
    
    # Final performance summary
    summary = control_system.get_performance_summary()
    print("\nFinal Performance Summary:")
    for metric, value in summary.items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nAdvanced Plasma Shape Control System successfully demonstrated!")