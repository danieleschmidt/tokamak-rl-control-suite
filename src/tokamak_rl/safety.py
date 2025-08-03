"""
Safety systems for tokamak plasma control.

This module implements safety shields, disruption prediction, and constraint
management to ensure safe operation of RL-controlled tokamak plasmas.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import warnings
from dataclasses import dataclass
import torch
import torch.nn as nn
from .physics import PlasmaState, TokamakConfig


@dataclass
class SafetyLimits:
    """Safety limits for tokamak operation."""
    
    q_min_threshold: float = 1.5  # Minimum safety factor
    beta_limit: float = 0.04  # Maximum plasma beta (Troyon limit)
    density_limit: float = 1.2e20  # Greenwald density limit (m^-3)
    shape_error_limit: float = 5.0  # Maximum shape error (cm)
    disruption_probability_limit: float = 0.1  # Maximum disruption probability
    
    # Control limits
    pf_coil_current_limit: float = 10.0  # Maximum PF coil current (MA)
    pf_coil_rate_limit: float = 5.0  # Maximum current ramp rate (MA/s)
    heating_power_limit: float = 100.0  # Maximum heating power (MW)
    gas_puff_rate_limit: float = 1.0  # Maximum gas puff rate (Pa·m³/s)


class DisruptionPredictor:
    """LSTM-based disruption prediction system."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model()
        self.history_buffer = []
        self.buffer_size = 50  # Store last 50 time steps
        
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with random weights - in production, load pre-trained model
            self._initialize_weights()
            
    def _create_model(self) -> nn.Module:
        """Create LSTM disruption prediction model."""
        
        class DisruptionLSTM(nn.Module):
            def __init__(self, input_size: int = 45, hidden_size: int = 64, 
                        num_layers: int = 2, dropout: float = 0.2):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=dropout
                )
                self.fc1 = nn.Linear(hidden_size, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 1)
                self.dropout = nn.Dropout(dropout)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                # x shape: (batch_size, sequence_length, input_size)
                lstm_out, _ = self.lstm(x)
                # Take the last output
                out = lstm_out[:, -1, :]
                out = self.relu(self.fc1(out))
                out = self.dropout(out)
                out = self.relu(self.fc2(out))
                out = self.dropout(out)
                out = self.sigmoid(self.fc3(out))
                return out.squeeze()
                
        return DisruptionLSTM().to(self.device)
        
    def _initialize_weights(self) -> None:
        """Initialize model with reasonable weights."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.zeros_(param.data)
                        
    def predict_disruption(self, plasma_state: PlasmaState) -> float:
        """Predict disruption probability for current plasma state."""
        # Add current observation to history
        obs = plasma_state.get_observation()
        self.history_buffer.append(obs)
        
        # Maintain buffer size
        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)
            
        # Need minimum history for prediction
        if len(self.history_buffer) < 10:
            return 0.0
            
        # Prepare input sequence
        sequence = np.array(self.history_buffer[-20:])  # Last 20 time steps
        if sequence.shape[0] < 20:
            # Pad with first observation if insufficient history
            padding = np.tile(sequence[0], (20 - sequence.shape[0], 1))
            sequence = np.vstack([padding, sequence])
            
        # Convert to tensor and predict
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            disruption_prob = self.model(input_tensor).item()
            
        return np.clip(disruption_prob, 0.0, 1.0)
        
    def load_model(self, model_path: str) -> None:
        """Load pre-trained disruption prediction model."""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except FileNotFoundError:
            warnings.warn(f"Model file {model_path} not found. Using random initialization.")
            self._initialize_weights()
            
    def reset(self) -> None:
        """Reset history buffer."""
        self.history_buffer.clear()


class SafetyShield:
    """Real-time safety shield for filtering RL actions."""
    
    def __init__(self, limits: Optional[SafetyLimits] = None, 
                 disruption_predictor: Optional[DisruptionPredictor] = None):
        self.limits = limits or SafetyLimits()
        self.predictor = disruption_predictor or DisruptionPredictor()
        self.last_action = None
        self.emergency_mode = False
        
    def filter_action(self, proposed_action: np.ndarray, 
                     plasma_state: PlasmaState) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filter proposed action through safety constraints.
        
        Args:
            proposed_action: Raw action from RL agent [pf_coils(6), gas_puff(1), heating(1)]
            plasma_state: Current plasma state
            
        Returns:
            Tuple of (safe_action, safety_info)
        """
        safety_info = {
            'action_modified': False,
            'violations': [],
            'disruption_risk': 0.0,
            'emergency_mode': self.emergency_mode
        }
        
        # Copy action for modification
        safe_action = proposed_action.copy()
        
        # Check current safety metrics
        safety_metrics = plasma_state.compute_safety_metrics()
        disruption_risk = self.predictor.predict_disruption(plasma_state)
        safety_info['disruption_risk'] = disruption_risk
        
        # Emergency mode activation
        if (safety_metrics['q_min'] < 1.2 or 
            disruption_risk > self.limits.disruption_probability_limit * 2):
            self.emergency_mode = True
            safety_info['emergency_mode'] = True
            
        # Apply safety constraints
        safe_action, violations = self._apply_constraints(safe_action, plasma_state, safety_metrics)
        safety_info['violations'] = violations
        
        if len(violations) > 0:
            safety_info['action_modified'] = True
            
        # Emergency shutdown if critical violations
        if self.emergency_mode:
            safe_action = self._emergency_action(plasma_state)
            safety_info['action_modified'] = True
            
        self.last_action = safe_action.copy()
        return safe_action, safety_info
        
    def _apply_constraints(self, action: np.ndarray, plasma_state: PlasmaState,
                          safety_metrics: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        """Apply individual safety constraints to action."""
        violations = []
        
        # PF coil current limits (first 6 elements)
        pf_currents = action[:6]
        
        # Absolute current limits
        for i, current in enumerate(pf_currents):
            if abs(current) > self.limits.pf_coil_current_limit:
                action[i] = np.sign(current) * self.limits.pf_coil_current_limit
                violations.append(f"PF coil {i} current limit")
                
        # Rate limits (if we have previous action)
        if self.last_action is not None:
            last_pf = self.last_action[:6]
            dt = 1.0 / 100.0  # Assuming 100 Hz control
            for i, (current, last_current) in enumerate(zip(pf_currents, last_pf)):
                rate = abs(current - last_current) / dt
                if rate > self.limits.pf_coil_rate_limit:
                    max_change = self.limits.pf_coil_rate_limit * dt
                    action[i] = last_current + np.sign(current - last_current) * max_change
                    violations.append(f"PF coil {i} rate limit")
                    
        # Gas puff rate limit (element 6)
        if action[6] < 0:
            action[6] = 0
            violations.append("Gas puff negative")
        elif action[6] > self.limits.gas_puff_rate_limit:
            action[6] = self.limits.gas_puff_rate_limit
            violations.append("Gas puff rate limit")
            
        # Heating power limit (element 7)
        if action[7] < 0:
            action[7] = 0
            violations.append("Heating power negative")
        elif action[7] > self.limits.heating_power_limit:
            action[7] = self.limits.heating_power_limit
            violations.append("Heating power limit")
            
        # Physics-based constraints
        if safety_metrics['q_min'] < self.limits.q_min_threshold:
            # Reduce heating and gas puff to increase q
            action[6] *= 0.5  # Reduce gas puff
            action[7] *= 0.7  # Reduce heating
            violations.append("Low q_min safety")
            
        if safety_metrics['beta_limit_fraction'] > 0.9:
            # Reduce heating power if approaching beta limit
            action[7] *= 0.5
            violations.append("High beta limit")
            
        if safety_metrics['density_limit_fraction'] > 0.9:
            # Stop gas puff if approaching density limit
            action[6] = 0
            violations.append("High density limit")
            
        return action, violations
        
    def _emergency_action(self, plasma_state: PlasmaState) -> np.ndarray:
        """Generate emergency control action for safe shutdown."""
        # Emergency action prioritizes safety over performance
        emergency_action = np.zeros(8)
        
        # Conservative PF coil currents to maintain basic equilibrium
        emergency_action[:6] = 0.1 * np.ones(6)  # Small positive currents
        
        # Stop gas puff
        emergency_action[6] = 0.0
        
        # Minimal heating
        emergency_action[7] = 0.1
        
        return emergency_action
        
    def reset(self) -> None:
        """Reset safety shield state."""
        self.last_action = None
        self.emergency_mode = False
        self.predictor.reset()


class ConstraintManager:
    """Manages physics and operational constraints."""
    
    def __init__(self, config: TokamakConfig):
        self.config = config
        self.constraints = self._setup_constraints()
        
    def _setup_constraints(self) -> List[Callable]:
        """Setup constraint functions."""
        constraints = []
        
        # Kruskal-Shafranov constraint: q > 1 everywhere
        def q_constraint(state: PlasmaState) -> bool:
            return np.all(state.q_profile > 1.0)
        constraints.append(q_constraint)
        
        # Troyon beta limit
        def beta_constraint(state: PlasmaState) -> bool:
            beta_limit = 0.028 * state.config.plasma_current / (state.config.minor_radius * state.config.toroidal_field)
            return state.plasma_beta < beta_limit
        constraints.append(beta_constraint)
        
        # Greenwald density limit
        def density_constraint(state: PlasmaState) -> bool:
            greenwald_limit = 1e20 * state.config.plasma_current / (np.pi * state.config.minor_radius**2)
            return np.max(state.density_profile) < greenwald_limit
        constraints.append(density_constraint)
        
        return constraints
        
    def check_constraints(self, state: PlasmaState) -> Dict[str, bool]:
        """Check all constraints for given plasma state."""
        results = {}
        constraint_names = ['q_constraint', 'beta_constraint', 'density_constraint']
        
        for name, constraint in zip(constraint_names, self.constraints):
            try:
                results[name] = constraint(state)
            except Exception as e:
                warnings.warn(f"Constraint {name} failed: {e}")
                results[name] = False
                
        return results


def create_safety_system(config: TokamakConfig, 
                        custom_limits: Optional[SafetyLimits] = None) -> SafetyShield:
    """Factory function to create complete safety system."""
    limits = custom_limits or SafetyLimits()
    predictor = DisruptionPredictor()
    shield = SafetyShield(limits, predictor)
    return shield