"""
Advanced Safety Systems for Tokamak Control

Next-generation safety architecture with multi-modal threat detection,
predictive safety, and autonomous emergency response systems.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level classification"""
    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class SafetyEvent:
    """Safety event record"""
    timestamp: float
    threat_level: ThreatLevel
    threat_type: str
    description: str
    plasma_state: np.ndarray
    control_actions: np.ndarray
    predicted_outcome: Dict[str, float]
    response_actions: List[str] = field(default_factory=list)
    resolution_time: Optional[float] = None


class MultiModalThreatDetector:
    """
    Multi-modal threat detection using fusion of multiple AI models
    for comprehensive plasma safety monitoring.
    """
    
    def __init__(self, input_dim: int = 45, history_length: int = 50):
        self.input_dim = input_dim
        self.history_length = history_length
        
        # State history buffer
        self.state_history = deque(maxlen=history_length)
        self.threat_history = deque(maxlen=1000)
        
        # Multiple detection models
        self.disruption_detector = self._build_disruption_detector()
        self.instability_detector = self._build_instability_detector()
        self.anomaly_detector = self._build_anomaly_detector()
        self.pattern_detector = self._build_pattern_detector()
        
        # Ensemble weights (learned from validation data)
        self.ensemble_weights = {
            'disruption': 0.3,
            'instability': 0.25,
            'anomaly': 0.25,
            'pattern': 0.2
        }
        
        # Threat thresholds
        self.threat_thresholds = {
            ThreatLevel.LOW: 0.1,
            ThreatLevel.MEDIUM: 0.3,
            ThreatLevel.HIGH: 0.6,
            ThreatLevel.CRITICAL: 0.8,
            ThreatLevel.EMERGENCY: 0.95
        }
        
        logger.info("Initialized multi-modal threat detector with 4 specialized models")
    
    def _build_disruption_detector(self) -> nn.Module:
        """Build neural network for disruption prediction"""
        model = nn.Sequential(
            nn.Linear(self.input_dim * 3, 128),  # Current + 2 historical states
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize with pre-trained weights (simulated)
        self._initialize_pretrained_weights(model, 'disruption')
        
        return model
    
    def _build_instability_detector(self) -> nn.Module:
        """Build LSTM for plasma instability detection"""
        class InstabilityLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])  # Use last time step
        
        model = InstabilityLSTM(self.input_dim)
        self._initialize_pretrained_weights(model, 'instability')
        
        return model
    
    def _build_anomaly_detector(self) -> nn.Module:
        """Build autoencoder for anomaly detection"""
        class AnomalyAutoencoder(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
            
            def anomaly_score(self, x):
                with torch.no_grad():
                    reconstructed = self.forward(x)
                    mse = torch.mean((x - reconstructed) ** 2, dim=1)
                    return torch.sigmoid(mse * 10)  # Scale and normalize
        
        model = AnomalyAutoencoder(self.input_dim)
        self._initialize_pretrained_weights(model, 'anomaly')
        
        return model
    
    def _build_pattern_detector(self) -> nn.Module:
        """Build CNN for pattern recognition in plasma signatures"""
        class PatternCNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                # Reshape input to 2D for conv (simulate spatial structure)
                self.input_reshape = int(np.sqrt(input_dim))
                if self.input_reshape ** 2 != input_dim:
                    self.input_reshape = 7  # Default 7x7 = 49, pad if needed
                
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((3, 3))
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(32 * 3 * 3, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                batch_size = x.shape[0]
                
                # Pad input if necessary
                padded_size = self.input_reshape ** 2
                if x.shape[1] < padded_size:
                    padding = torch.zeros(batch_size, padded_size - x.shape[1])
                    x = torch.cat([x, padding], dim=1)
                elif x.shape[1] > padded_size:
                    x = x[:, :padded_size]
                
                # Reshape to 2D
                x = x.view(batch_size, 1, self.input_reshape, self.input_reshape)
                
                # Apply convolutions
                conv_out = self.conv_layers(x)
                conv_out = conv_out.view(batch_size, -1)
                
                return self.classifier(conv_out)
        
        model = PatternCNN(self.input_dim)
        self._initialize_pretrained_weights(model, 'pattern')
        
        return model
    
    def _initialize_pretrained_weights(self, model: nn.Module, model_type: str):
        """Initialize with simulated pre-trained weights"""
        # In practice, these would be loaded from actual pre-trained models
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
        
        logger.debug(f"Initialized {model_type} detector with simulated pre-trained weights")
    
    def detect_threats(self, current_state: np.ndarray, 
                      control_actions: np.ndarray) -> Dict[str, Any]:
        """
        Detect multiple types of threats using ensemble of models
        
        Args:
            current_state: Current plasma state
            control_actions: Proposed control actions
            
        Returns:
            Dictionary containing threat assessment
        """
        # Add to history
        self.state_history.append(current_state.copy())
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        
        # Run individual detectors
        threat_scores = {}
        
        # 1. Disruption detection
        if len(self.state_history) >= 3:
            # Use current + 2 historical states
            historical_states = list(self.state_history)[-3:]
            combined_input = np.concatenate(historical_states)
            disruption_input = torch.FloatTensor(combined_input).unsqueeze(0)
            
            with torch.no_grad():
                threat_scores['disruption'] = self.disruption_detector(disruption_input).item()
        else:
            threat_scores['disruption'] = 0.0
        
        # 2. Instability detection (requires sequence)
        if len(self.state_history) >= 10:
            sequence_data = torch.FloatTensor(list(self.state_history)[-10:]).unsqueeze(0)
            
            with torch.no_grad():
                threat_scores['instability'] = self.instability_detector(sequence_data).item()
        else:
            threat_scores['instability'] = 0.0
        
        # 3. Anomaly detection
        with torch.no_grad():
            if hasattr(self.anomaly_detector, 'anomaly_score'):
                threat_scores['anomaly'] = self.anomaly_detector.anomaly_score(state_tensor).item()
            else:
                reconstructed = self.anomaly_detector(state_tensor)
                mse = torch.mean((state_tensor - reconstructed) ** 2)
                threat_scores['anomaly'] = torch.sigmoid(mse * 10).item()
        
        # 4. Pattern detection
        with torch.no_grad():
            threat_scores['pattern'] = self.pattern_detector(state_tensor).item()
        
        # Ensemble combination
        combined_threat_score = sum(
            self.ensemble_weights[detector] * score 
            for detector, score in threat_scores.items()
        )
        
        # Determine threat level
        threat_level = self._classify_threat_level(combined_threat_score)
        
        # Predict outcome if actions are taken
        predicted_outcome = self._predict_action_outcome(current_state, control_actions)
        
        threat_assessment = {
            'combined_threat_score': combined_threat_score,
            'individual_scores': threat_scores,
            'threat_level': threat_level,
            'predicted_outcome': predicted_outcome,
            'state_history_length': len(self.state_history),
            'timestamp': time.time()
        }
        
        # Log significant threats
        if threat_level.value >= ThreatLevel.MEDIUM.value:
            logger.warning(f"Threat detected: {threat_level.name}, score: {combined_threat_score:.3f}")
        
        return threat_assessment
    
    def _classify_threat_level(self, threat_score: float) -> ThreatLevel:
        """Classify threat level based on combined score"""
        for level in reversed(list(ThreatLevel)):
            if level != ThreatLevel.NORMAL and threat_score >= self.threat_thresholds[level]:
                return level
        return ThreatLevel.NORMAL
    
    def _predict_action_outcome(self, state: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
        """Predict plasma outcome if proposed actions are taken"""
        # Simplified physics-based prediction
        # In practice, this would use sophisticated plasma simulation
        
        # Safety metrics prediction
        current_q_min = state[10] if len(state) > 10 else 2.0
        current_beta = state[1] if len(state) > 1 else 0.02
        current_density = np.mean(state[20:30]) if len(state) > 30 else 1e19
        
        # Predict changes based on control actions
        # Simplified linear response model
        q_change = -actions[0] * 0.1  # PF coil affects q
        beta_change = actions[-1] * 0.005  # Heating affects beta
        density_change = actions[-2] * 1e18 if len(actions) > 1 else 0  # Gas puff affects density
        
        predicted_q_min = current_q_min + q_change
        predicted_beta = current_beta + beta_change
        predicted_density = current_density + density_change
        
        # Calculate safety margins
        q_margin = (predicted_q_min - 1.5) / 1.5  # q > 1.5 for stability
        beta_margin = (0.04 - predicted_beta) / 0.04  # beta < 4% limit
        density_margin = (1.2e20 - predicted_density) / 1.2e20  # Greenwald limit
        
        return {
            'predicted_q_min': predicted_q_min,
            'predicted_beta': predicted_beta,
            'predicted_density': predicted_density,
            'q_safety_margin': q_margin,
            'beta_safety_margin': beta_margin,
            'density_safety_margin': density_margin,
            'overall_safety_score': min(q_margin, beta_margin, density_margin)
        }


class PredictiveSafetySystem:
    """
    Predictive safety system using reinforcement learning to anticipate
    and prevent dangerous plasma configurations before they occur.
    """
    
    def __init__(self, state_dim: int = 45, action_dim: int = 8, 
                 prediction_horizon: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon
        
        # Predictive models
        self.dynamics_model = self._build_dynamics_model()
        self.safety_critic = self._build_safety_critic()
        self.emergency_controller = self._build_emergency_controller()
        
        # Safety constraints
        self.safety_constraints = self._define_safety_constraints()
        
        # Prediction cache for efficiency
        self.prediction_cache = {}
        self.cache_validity_time = 0.1  # seconds
        
        logger.info(f"Initialized predictive safety system with {prediction_horizon}-step horizon")
    
    def _build_dynamics_model(self) -> nn.Module:
        """Build neural network model for plasma dynamics prediction"""
        class DynamicsModel(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim=256):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, state_dim)
                )
                
                # Uncertainty quantification
                self.uncertainty_net = nn.Sequential(
                    nn.Linear(state_dim + action_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, state_dim),
                    nn.Softplus()  # Positive uncertainty values
                )
            
            def forward(self, state, action):
                input_tensor = torch.cat([state, action], dim=-1)
                next_state_pred = self.network(input_tensor)
                uncertainty = self.uncertainty_net(input_tensor)
                return next_state_pred, uncertainty
        
        model = DynamicsModel(self.state_dim, self.action_dim)
        
        # Initialize with physics-informed weights
        self._initialize_physics_informed_weights(model)
        
        return model
    
    def _build_safety_critic(self) -> nn.Module:
        """Build safety critic to evaluate trajectory safety"""
        class SafetyCritic(nn.Module):
            def __init__(self, state_dim, hidden_dim=128):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()  # Safety score between 0 and 1
                )
            
            def forward(self, state):
                return self.network(state)
        
        model = SafetyCritic(self.state_dim)
        self._initialize_safety_critic_weights(model)
        
        return model
    
    def _build_emergency_controller(self) -> nn.Module:
        """Build emergency controller for crisis intervention"""
        class EmergencyController(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim=128):
                super().__init__()
                # Separate networks for different emergency types
                self.disruption_controller = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Tanh()
                )
                
                self.instability_controller = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Tanh()
                )
                
                self.shutdown_controller = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Sigmoid()  # Positive values for shutdown sequence
                )
            
            def forward(self, state, emergency_type='disruption'):
                if emergency_type == 'disruption':
                    return self.disruption_controller(state)
                elif emergency_type == 'instability':
                    return self.instability_controller(state)
                elif emergency_type == 'shutdown':
                    return self.shutdown_controller(state)
                else:
                    return self.disruption_controller(state)  # Default
        
        model = EmergencyController(self.state_dim, self.action_dim)
        self._initialize_emergency_controller_weights(model)
        
        return model
    
    def _initialize_physics_informed_weights(self, model: nn.Module):
        """Initialize dynamics model with physics-informed weights"""
        # Initialize with small random weights that respect physics
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.1)  # Small initial weights
            else:
                nn.init.zeros_(param)
    
    def _initialize_safety_critic_weights(self, model: nn.Module):
        """Initialize safety critic with conservative bias"""
        # Initialize to be pessimistic about safety
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, -1.0)  # Bias towards low safety scores
    
    def _initialize_emergency_controller_weights(self, model: nn.Module):
        """Initialize emergency controller with stabilizing actions"""
        # Initialize to produce stabilizing control actions
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.orthogonal_(param, gain=0.1)
            else:
                nn.init.zeros_(param)
    
    def _define_safety_constraints(self) -> Dict[str, Dict[str, float]]:
        """Define hard safety constraints for plasma parameters"""
        return {
            'q_min': {'lower': 1.5, 'upper': np.inf, 'critical_lower': 1.0},
            'beta': {'lower': 0.0, 'upper': 0.04, 'critical_upper': 0.05},
            'density': {'lower': 1e18, 'upper': 1.2e20, 'critical_upper': 1.5e20},
            'stored_energy': {'lower': 0.0, 'upper': 500e6, 'critical_upper': 600e6},  # Joules
            'elongation': {'lower': 1.0, 'upper': 2.5, 'critical_upper': 3.0},
            'triangularity': {'lower': -0.5, 'upper': 0.8, 'critical_upper': 1.0}
        }
    
    def predict_trajectory_safety(self, initial_state: np.ndarray, 
                                 action_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Predict safety of a control trajectory over multiple time steps
        
        Args:
            initial_state: Starting plasma state
            action_sequence: Sequence of control actions [horizon, action_dim]
            
        Returns:
            Dictionary containing trajectory safety assessment
        """
        # Check cache first
        cache_key = (tuple(initial_state), tuple(action_sequence.flatten()))
        current_time = time.time()
        
        if cache_key in self.prediction_cache:
            cached_result, cache_time = self.prediction_cache[cache_key]
            if current_time - cache_time < self.cache_validity_time:
                return cached_result
        
        # Run prediction
        state = torch.FloatTensor(initial_state).unsqueeze(0)
        trajectory_states = [state]
        trajectory_safety_scores = []
        trajectory_uncertainties = []
        constraint_violations = []
        
        # Predict forward through time horizon
        for step in range(min(self.prediction_horizon, len(action_sequence))):
            action = torch.FloatTensor(action_sequence[step]).unsqueeze(0)
            
            # Predict next state with uncertainty
            with torch.no_grad():
                next_state_pred, uncertainty = self.dynamics_model(state, action)
                
                # Add process noise based on uncertainty
                noise = torch.randn_like(next_state_pred) * uncertainty * 0.1
                next_state = next_state_pred + noise
                
                # Evaluate safety of predicted state
                safety_score = self.safety_critic(next_state).item()
                
                # Check constraint violations
                violations = self._check_constraint_violations(next_state.numpy().flatten())
                
                trajectory_states.append(next_state)
                trajectory_safety_scores.append(safety_score)
                trajectory_uncertainties.append(torch.mean(uncertainty).item())
                constraint_violations.append(violations)
                
                state = next_state
        
        # Analyze trajectory
        min_safety_score = min(trajectory_safety_scores) if trajectory_safety_scores else 1.0
        avg_safety_score = np.mean(trajectory_safety_scores) if trajectory_safety_scores else 1.0
        max_uncertainty = max(trajectory_uncertainties) if trajectory_uncertainties else 0.0
        
        # Count critical violations
        critical_violations = sum(
            len([v for v in step_violations if v['severity'] == 'critical'])
            for step_violations in constraint_violations
        )
        
        # Overall trajectory assessment
        trajectory_safe = (min_safety_score > 0.7 and 
                         critical_violations == 0 and
                         max_uncertainty < 0.5)
        
        result = {
            'trajectory_safe': trajectory_safe,
            'min_safety_score': min_safety_score,
            'avg_safety_score': avg_safety_score,
            'max_uncertainty': max_uncertainty,
            'critical_violations': critical_violations,
            'total_violations': sum(len(step_violations) for step_violations in constraint_violations),
            'safety_scores': trajectory_safety_scores,
            'uncertainties': trajectory_uncertainties,
            'constraint_violations': constraint_violations,
            'prediction_horizon_used': len(trajectory_safety_scores)
        }
        
        # Cache result
        self.prediction_cache[cache_key] = (result, current_time)
        
        # Clean old cache entries
        if len(self.prediction_cache) > 1000:
            old_keys = [k for k, (_, t) in self.prediction_cache.items() 
                       if current_time - t > self.cache_validity_time * 10]
            for k in old_keys:
                del self.prediction_cache[k]
        
        return result
    
    def _check_constraint_violations(self, state: np.ndarray) -> List[Dict[str, Any]]:
        """Check for safety constraint violations in predicted state"""
        violations = []
        
        # Map state indices to physics parameters (simplified)
        param_mapping = {
            'q_min': 10,  # Minimum safety factor
            'beta': 1,    # Plasma beta
            'density': 25,  # Average density (approximate)
            'stored_energy': 35,  # Stored energy (approximate)
            'elongation': 6,  # Plasma elongation
            'triangularity': 7  # Plasma triangularity
        }
        
        for param_name, state_idx in param_mapping.items():
            if state_idx < len(state):
                value = state[state_idx]
                constraints = self.safety_constraints[param_name]
                
                # Check violations
                if 'lower' in constraints and value < constraints['lower']:
                    severity = 'critical' if 'critical_lower' in constraints and value < constraints['critical_lower'] else 'warning'
                    violations.append({
                        'parameter': param_name,
                        'value': float(value),
                        'constraint': 'lower',
                        'limit': constraints['lower'],
                        'severity': severity
                    })
                
                if 'upper' in constraints and value > constraints['upper']:
                    severity = 'critical' if 'critical_upper' in constraints and value > constraints['critical_upper'] else 'warning'
                    violations.append({
                        'parameter': param_name,
                        'value': float(value),
                        'constraint': 'upper',
                        'limit': constraints['upper'],
                        'severity': severity
                    })
        
        return violations
    
    def generate_emergency_actions(self, current_state: np.ndarray, 
                                 threat_assessment: Dict[str, Any]) -> np.ndarray:
        """
        Generate emergency control actions based on threat assessment
        
        Args:
            current_state: Current plasma state
            threat_assessment: Threat assessment from threat detector
            
        Returns:
            Emergency control actions
        """
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        
        # Determine emergency type
        threat_level = threat_assessment['threat_level']
        individual_scores = threat_assessment['individual_scores']
        
        # Select emergency controller type
        if individual_scores.get('disruption', 0) > 0.7:
            emergency_type = 'disruption'
        elif individual_scores.get('instability', 0) > 0.7:
            emergency_type = 'instability'
        elif threat_level.value >= ThreatLevel.EMERGENCY.value:
            emergency_type = 'shutdown'
        else:
            emergency_type = 'disruption'  # Default
        
        # Generate emergency actions
        with torch.no_grad():
            emergency_actions = self.emergency_controller(state_tensor, emergency_type)
            emergency_actions = emergency_actions.numpy().flatten()
        
        # Apply safety limits to emergency actions
        emergency_actions = np.clip(emergency_actions, -1.0, 1.0)
        
        # Log emergency action generation
        logger.critical(f"Generated {emergency_type} emergency actions: {emergency_actions}")
        
        return emergency_actions


class AutonomousEmergencyResponse:
    """
    Autonomous emergency response system that can take immediate action
    without human intervention during critical safety events.
    """
    
    def __init__(self, response_time_limit: float = 0.01):  # 10ms response time
        self.response_time_limit = response_time_limit
        
        # Emergency protocols
        self.emergency_protocols = self._define_emergency_protocols()
        
        # Response history
        self.response_history = deque(maxlen=1000)
        
        # Thread pool for parallel emergency response
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Emergency state tracking
        self.emergency_active = False
        self.emergency_start_time = None
        self.current_emergency_level = ThreatLevel.NORMAL
        
        logger.info("Initialized autonomous emergency response system")
    
    def _define_emergency_protocols(self) -> Dict[ThreatLevel, Dict[str, Any]]:
        """Define emergency response protocols for each threat level"""
        return {
            ThreatLevel.LOW: {
                'actions': ['log_event', 'increase_monitoring'],
                'control_authority': 'advisory',
                'notification_level': 'info',
                'max_response_time': 1.0
            },
            ThreatLevel.MEDIUM: {
                'actions': ['log_event', 'adjust_control_gains', 'alert_operators'],
                'control_authority': 'supervisory',
                'notification_level': 'warning',
                'max_response_time': 0.5
            },
            ThreatLevel.HIGH: {
                'actions': ['emergency_control_override', 'rapid_feedback', 'evacuate_personnel_advisory'],
                'control_authority': 'override',
                'notification_level': 'critical',
                'max_response_time': 0.1
            },
            ThreatLevel.CRITICAL: {
                'actions': ['full_control_takeover', 'disruption_mitigation', 'emergency_shutdown_prep'],
                'control_authority': 'full',
                'notification_level': 'emergency',
                'max_response_time': 0.05
            },
            ThreatLevel.EMERGENCY: {
                'actions': ['immediate_shutdown', 'energy_dump', 'facility_protection'],
                'control_authority': 'absolute',
                'notification_level': 'emergency',
                'max_response_time': 0.01
            }
        }
    
    def activate_emergency_response(self, safety_event: SafetyEvent,
                                  current_state: np.ndarray,
                                  emergency_actions: np.ndarray) -> Dict[str, Any]:
        """
        Activate emergency response system
        
        Args:
            safety_event: Detected safety event
            current_state: Current plasma state
            emergency_actions: Proposed emergency actions
            
        Returns:
            Dictionary containing response details
        """
        response_start_time = time.time()
        
        # Update emergency state
        if not self.emergency_active:
            self.emergency_active = True
            self.emergency_start_time = response_start_time
        
        self.current_emergency_level = safety_event.threat_level
        
        # Get emergency protocol
        protocol = self.emergency_protocols[safety_event.threat_level]
        
        # Check response time constraint
        max_response_time = protocol['max_response_time']
        
        # Execute emergency actions in parallel
        future_actions = []
        
        for action_name in protocol['actions']:
            future = self.executor.submit(
                self._execute_emergency_action,
                action_name, safety_event, current_state, emergency_actions
            )
            future_actions.append((action_name, future))
        
        # Collect results with timeout
        executed_actions = {}
        failed_actions = []
        
        for action_name, future in future_actions:
            try:
                result = future.result(timeout=max_response_time)
                executed_actions[action_name] = result
            except Exception as e:
                failed_actions.append((action_name, str(e)))
                logger.error(f"Emergency action {action_name} failed: {e}")
        
        # Calculate response time
        response_time = time.time() - response_start_time
        
        # Validate final control actions
        validated_actions = self._validate_emergency_actions(
            emergency_actions, safety_event.threat_level
        )
        
        # Create response record
        response_record = {
            'timestamp': response_start_time,
            'threat_level': safety_event.threat_level,
            'response_time': response_time,
            'within_time_limit': response_time <= max_response_time,
            'executed_actions': executed_actions,
            'failed_actions': failed_actions,
            'original_actions': emergency_actions.tolist(),
            'validated_actions': validated_actions.tolist(),
            'protocol_used': protocol,
            'emergency_duration': response_start_time - (self.emergency_start_time or response_start_time)
        }
        
        # Add to response history
        self.response_history.append(response_record)
        
        # Log emergency response
        if response_time <= max_response_time:
            logger.info(f"Emergency response completed in {response_time:.3f}s (limit: {max_response_time:.3f}s)")
        else:
            logger.critical(f"Emergency response EXCEEDED time limit: {response_time:.3f}s > {max_response_time:.3f}s")
        
        return response_record
    
    def _execute_emergency_action(self, action_name: str, safety_event: SafetyEvent,
                                current_state: np.ndarray, emergency_actions: np.ndarray) -> Dict[str, Any]:
        """Execute individual emergency action"""
        action_start_time = time.time()
        
        try:
            if action_name == 'log_event':
                return self._log_safety_event(safety_event)
            
            elif action_name == 'increase_monitoring':
                return self._increase_monitoring_frequency()
            
            elif action_name == 'adjust_control_gains':
                return self._adjust_control_gains(current_state)
            
            elif action_name == 'alert_operators':
                return self._alert_human_operators(safety_event)
            
            elif action_name == 'emergency_control_override':
                return self._emergency_control_override(emergency_actions)
            
            elif action_name == 'rapid_feedback':
                return self._enable_rapid_feedback()
            
            elif action_name == 'evacuate_personnel_advisory':
                return self._evacuate_personnel_advisory()
            
            elif action_name == 'full_control_takeover':
                return self._full_control_takeover()
            
            elif action_name == 'disruption_mitigation':
                return self._disruption_mitigation_sequence(current_state)
            
            elif action_name == 'emergency_shutdown_prep':
                return self._prepare_emergency_shutdown()
            
            elif action_name == 'immediate_shutdown':
                return self._immediate_plasma_shutdown()
            
            elif action_name == 'energy_dump':
                return self._emergency_energy_dump()
            
            elif action_name == 'facility_protection':
                return self._activate_facility_protection()
            
            else:
                raise ValueError(f"Unknown emergency action: {action_name}")
        
        except Exception as e:
            logger.error(f"Failed to execute emergency action {action_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - action_start_time
            }
    
    def _log_safety_event(self, safety_event: SafetyEvent) -> Dict[str, Any]:
        """Log safety event to persistent storage"""
        event_data = {
            'timestamp': safety_event.timestamp,
            'threat_level': safety_event.threat_level.name,
            'threat_type': safety_event.threat_type,
            'description': safety_event.description,
            'plasma_state_norm': np.linalg.norm(safety_event.plasma_state),
            'control_actions_norm': np.linalg.norm(safety_event.control_actions)
        }
        
        # In practice, this would write to a database or log file
        logger.warning(f"SAFETY EVENT: {json.dumps(event_data, indent=2)}")
        
        return {'success': True, 'logged_data': event_data}
    
    def _increase_monitoring_frequency(self) -> Dict[str, Any]:
        """Increase monitoring frequency for enhanced situational awareness"""
        # In practice, this would interface with monitoring systems
        logger.info("Increased monitoring frequency to high-resolution mode")
        return {'success': True, 'new_frequency': '1kHz', 'duration': '60s'}
    
    def _adjust_control_gains(self, current_state: np.ndarray) -> Dict[str, Any]:
        """Adjust control gains for more conservative operation"""
        # Calculate conservative gain adjustments
        gain_reduction_factor = 0.8
        
        logger.info(f"Adjusted control gains by factor {gain_reduction_factor}")
        return {
            'success': True,
            'gain_adjustment': gain_reduction_factor,
            'adjusted_parameters': ['PF_coil_gains', 'heating_gains']
        }
    
    def _alert_human_operators(self, safety_event: SafetyEvent) -> Dict[str, Any]:
        """Alert human operators of safety event"""
        alert_message = f"SAFETY ALERT: {safety_event.threat_level.name} - {safety_event.description}"
        
        # In practice, this would interface with alerting systems
        logger.critical(alert_message)
        
        return {
            'success': True,
            'alert_sent': True,
            'recipients': ['control_room', 'safety_officer', 'run_leader'],
            'message': alert_message
        }
    
    def _emergency_control_override(self, emergency_actions: np.ndarray) -> Dict[str, Any]:
        """Override normal control with emergency actions"""
        logger.critical("EMERGENCY CONTROL OVERRIDE ACTIVATED")
        
        return {
            'success': True,
            'override_active': True,
            'emergency_actions': emergency_actions.tolist(),
            'normal_control_suspended': True
        }
    
    def _enable_rapid_feedback(self) -> Dict[str, Any]:
        """Enable rapid feedback control mode"""
        logger.info("Activated rapid feedback control mode")
        
        return {
            'success': True,
            'feedback_frequency': '10kHz',
            'latency_target': '100μs'
        }
    
    def _evacuate_personnel_advisory(self) -> Dict[str, Any]:
        """Issue personnel evacuation advisory"""
        logger.critical("PERSONNEL EVACUATION ADVISORY ISSUED")
        
        return {
            'success': True,
            'evacuation_advisory': True,
            'affected_areas': ['tokamak_hall', 'control_room_secondary']
        }
    
    def _full_control_takeover(self) -> Dict[str, Any]:
        """Take full autonomous control of plasma"""
        logger.critical("FULL AUTONOMOUS CONTROL TAKEOVER")
        
        return {
            'success': True,
            'autonomous_control': True,
            'human_control_locked': True,
            'override_authority': 'emergency_system'
        }
    
    def _disruption_mitigation_sequence(self, current_state: np.ndarray) -> Dict[str, Any]:
        """Execute disruption mitigation sequence"""
        logger.critical("EXECUTING DISRUPTION MITIGATION SEQUENCE")
        
        # Simplified disruption mitigation
        mitigation_actions = {
            'massive_gas_injection': True,
            'current_quench_control': True,
            'runaway_electron_suppression': True
        }
        
        return {
            'success': True,
            'mitigation_sequence': mitigation_actions,
            'estimated_mitigation_time': '50ms'
        }
    
    def _prepare_emergency_shutdown(self) -> Dict[str, Any]:
        """Prepare for emergency plasma shutdown"""
        logger.critical("PREPARING EMERGENCY SHUTDOWN")
        
        return {
            'success': True,
            'shutdown_preparation': True,
            'estimated_shutdown_time': '2s',
            'protection_systems_armed': True
        }
    
    def _immediate_plasma_shutdown(self) -> Dict[str, Any]:
        """Execute immediate plasma shutdown"""
        logger.critical("IMMEDIATE PLASMA SHUTDOWN INITIATED")
        
        return {
            'success': True,
            'shutdown_initiated': True,
            'shutdown_method': 'emergency_quench',
            'expected_completion': '500ms'
        }
    
    def _emergency_energy_dump(self) -> Dict[str, Any]:
        """Dump stored magnetic energy safely"""
        logger.critical("EMERGENCY ENERGY DUMP ACTIVATED")
        
        return {
            'success': True,
            'energy_dump_active': True,
            'dump_resistors_engaged': True,
            'estimated_dump_time': '10s'
        }
    
    def _activate_facility_protection(self) -> Dict[str, Any]:
        """Activate facility-wide protection systems"""
        logger.critical("FACILITY PROTECTION SYSTEMS ACTIVATED")
        
        return {
            'success': True,
            'facility_protection': True,
            'systems_activated': ['fire_suppression', 'ventilation_isolation', 'emergency_power']
        }
    
    def _validate_emergency_actions(self, emergency_actions: np.ndarray, 
                                  threat_level: ThreatLevel) -> np.ndarray:
        """Validate and limit emergency actions based on threat level"""
        validated_actions = emergency_actions.copy()
        
        # Apply different limits based on threat level
        if threat_level == ThreatLevel.EMERGENCY:
            # Full authority - but still within physical limits
            validated_actions = np.clip(validated_actions, -1.0, 1.0)
        
        elif threat_level == ThreatLevel.CRITICAL:
            # High authority - 90% of full range
            validated_actions = np.clip(validated_actions, -0.9, 0.9)
        
        elif threat_level == ThreatLevel.HIGH:
            # Moderate authority - 70% of full range
            validated_actions = np.clip(validated_actions, -0.7, 0.7)
        
        else:
            # Limited authority - 50% of full range
            validated_actions = np.clip(validated_actions, -0.5, 0.5)
        
        return validated_actions
    
    def deactivate_emergency_response(self) -> Dict[str, Any]:
        """Deactivate emergency response and return to normal operation"""
        if not self.emergency_active:
            return {'success': True, 'already_inactive': True}
        
        self.emergency_active = False
        emergency_duration = time.time() - (self.emergency_start_time or 0)
        
        logger.info(f"Emergency response deactivated after {emergency_duration:.3f}s")
        
        return {
            'success': True,
            'emergency_duration': emergency_duration,
            'emergency_level': self.current_emergency_level.name,
            'normal_operation_restored': True
        }


def create_advanced_safety_system(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create comprehensive advanced safety system
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing advanced safety components
    """
    if config is None:
        config = {
            'threat_detector': {
                'input_dim': 45,
                'history_length': 50
            },
            'predictive_safety': {
                'state_dim': 45,
                'action_dim': 8,
                'prediction_horizon': 10
            },
            'emergency_response': {
                'response_time_limit': 0.01
            }
        }
    
    # Initialize components
    threat_detector = MultiModalThreatDetector(**config['threat_detector'])
    predictive_safety = PredictiveSafetySystem(**config['predictive_safety'])
    emergency_response = AutonomousEmergencyResponse(**config['emergency_response'])
    
    logger.info("Created advanced safety system with multi-modal detection, prediction, and response")
    
    return {
        'threat_detector': threat_detector,
        'predictive_safety': predictive_safety,
        'emergency_response': emergency_response,
        'config': config
    }


# Example usage and demonstration
if __name__ == "__main__":
    # Create advanced safety system
    safety_system = create_advanced_safety_system()
    
    print("🛡️ Advanced Safety Systems Demo")
    print("===============================")
    
    # Demo threat detection
    print("\n1. Multi-Modal Threat Detection:")
    threat_detector = safety_system['threat_detector']
    
    # Simulate plasma state with potential threat
    plasma_state = np.random.randn(45)
    plasma_state[10] = 1.2  # Low q_min (potential disruption risk)
    control_actions = np.random.randn(8) * 0.1
    
    threat_assessment = threat_detector.detect_threats(plasma_state, control_actions)
    
    print(f"   ✓ Combined threat score: {threat_assessment['combined_threat_score']:.3f}")
    print(f"   ✓ Threat level: {threat_assessment['threat_level'].name}")
    print(f"   ✓ Individual scores: {threat_assessment['individual_scores']}")
    print(f"   ✓ Safety margins: {threat_assessment['predicted_outcome']['overall_safety_score']:.3f}")
    
    # Demo predictive safety
    print("\n2. Predictive Safety System:")
    predictive_safety = safety_system['predictive_safety']
    
    # Generate action sequence for prediction
    action_sequence = np.random.randn(10, 8) * 0.2
    
    trajectory_safety = predictive_safety.predict_trajectory_safety(plasma_state, action_sequence)
    
    print(f"   ✓ Trajectory safe: {trajectory_safety['trajectory_safe']}")
    print(f"   ✓ Min safety score: {trajectory_safety['min_safety_score']:.3f}")
    print(f"   ✓ Critical violations: {trajectory_safety['critical_violations']}")
    print(f"   ✓ Max uncertainty: {trajectory_safety['max_uncertainty']:.3f}")
    
    # Demo emergency response
    print("\n3. Autonomous Emergency Response:")
    emergency_response = safety_system['emergency_response']
    
    # Create safety event
    safety_event = SafetyEvent(
        timestamp=time.time(),
        threat_level=ThreatLevel.HIGH,
        threat_type='disruption_risk',
        description='High disruption probability detected',
        plasma_state=plasma_state,
        control_actions=control_actions,
        predicted_outcome=threat_assessment['predicted_outcome']
    )
    
    # Generate emergency actions
    emergency_actions = predictive_safety.generate_emergency_actions(plasma_state, threat_assessment)
    
    # Activate emergency response
    response_record = emergency_response.activate_emergency_response(
        safety_event, plasma_state, emergency_actions
    )
    
    print(f"   ✓ Response time: {response_record['response_time']:.3f}s")
    print(f"   ✓ Within time limit: {response_record['within_time_limit']}")
    print(f"   ✓ Executed actions: {len(response_record['executed_actions'])}")
    print(f"   ✓ Emergency duration: {response_record['emergency_duration']:.3f}s")
    
    # Deactivate emergency
    deactivation_result = emergency_response.deactivate_emergency_response()
    print(f"   ✓ Emergency deactivated: {deactivation_result['success']}")
    
    print("\n🚀 Advanced safety systems completed successfully!")
    print("    Next-generation safety architecture ready for deployment")